#!/usr/bin/env python3
"""Plan or execute bounded qualification for WRC adaptive checkpoint rebinding."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import (  # noqa: E402
    atomic_write_json,
    execute_campaign,
    prepare_adaptive_campaign_continuation_v1,
)
from engine.campaigns.preflight import preflight_campaign  # noqa: E402
from engine.campaigns.providers import (  # noqa: E402
    default_execution_provider_registry,
)
from engine.campaigns.registry import default_profile_registry  # noqa: E402
from engine.campaigns.wrc_adaptive_rebinding import (  # noqa: E402
    materialize_wrc_adaptive_restore_plan_v1,
    validate_wrc_checkpoint_adaptive_rebinding_receipt_v1,
    validate_wrc_checkpoint_adaptive_rebinding_v1,
)


CAPABILITY_ID = "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1"
QUALIFICATION_ID = "WRC-CHECKPOINT-ADAPTIVE-REBINDING-V1-B48-PER-CELL-DELTA"
SOURCE_GENERATION = 3
RESTORE_GENERATION = 6
TICKS_PER_GENERATION = 160
REBOUND_CELL_ROTATION = 14
REBOUND_PARAMETERS = (
    "env_cfg.jungle_recovery_activity_base_growth",
    "env_cfg.jungle_recovery_activity_decay_per_step",
    "env_cfg.jungle_recovery_activity_fertility_weight",
    "env_cfg.jungle_recovery_activity_waste_penalty_weight",
    "env_cfg.move_cost",
    "env_cfg.patch_movement_cost_energy_scale",
    "env_cfg.patch_food_gather_amount",
    "env_cfg.patch_food_regrow_per_step",
    "env_cfg.jungle_predator_base_risk",
    "env_cfg.jungle_predator_health_damage",
    "env_cfg.jungle_predator_risk_cap",
    "env_cfg.reproduce_cooldown_steps",
    "env_cfg.reproduce_min_energy_buffer",
    "env_cfg.reproduce_parent_energy_share",
    "env_cfg.reproduce_try_prob",
)
HOST_TESTS = (
    "tests/test_eiel_wrc_checkpoint_adaptive_rebinding_v1.py",
    "tests/test_eiel_campaign_adaptive_continuation_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_source_allocator_v1.py",
    "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py",
)


class AdaptiveRebindingQualificationError(RuntimeError):
    """The bounded qualification target or its evidence is invalid."""


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptiveRebindingQualificationError(
            f"cannot read {name}: {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise AdaptiveRebindingQualificationError(f"{name} must be an object")
    return dict(value)


def _verified_plan(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    plan = dict(value)
    declared = plan.pop("plan_sha256", None)
    if not isinstance(declared, str) or canonical_sha256(plan) != declared:
        raise AdaptiveRebindingQualificationError(f"{name} plan hash differs")
    return {**plan, "plan_sha256": declared}


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise AdaptiveRebindingQualificationError("cannot resolve source commit")
    return commit


def _set_generation(plan: dict[str, Any], generation: int) -> None:
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise AdaptiveRebindingQualificationError("base execution is absent")
    updated = dict(execution)
    updated["generations"] = generation
    updated["checkpoints"] = [generation]
    schedule = updated.get("tick_schedule")
    if not isinstance(schedule, Mapping):
        raise AdaptiveRebindingQualificationError("base tick schedule is absent")
    schedule = dict(schedule)
    projection = schedule.get("legacy_projection")
    if (
        not isinstance(projection, Mapping)
        or projection.get("ticks_per_generation") != TICKS_PER_GENERATION
    ):
        raise AdaptiveRebindingQualificationError("base T160 projection differs")
    schedule["total_ticks"] = generation * TICKS_PER_GENERATION
    schedule["legacy_projection"] = {
        **dict(projection),
        "generations": generation,
    }
    updated["tick_schedule"] = schedule
    plan["execution"] = updated


def materialize_qualification_plans(
    base_plan: Mapping[str, Any], *, evidence_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build an exact G3 source and G6 rebound plan inside the qualified route."""

    base = _verified_plan(base_plan, name="base")
    cells = base.get("cells")
    batches = base.get("physical_batches")
    execution = base.get("execution")
    if (
        not isinstance(cells, list)
        or len(cells) != 48
        or not isinstance(batches, list)
        or len(batches) != 1
        or not isinstance(batches[0], Mapping)
        or len(batches[0].get("cell_ids", [])) != 48
        or not isinstance(execution, Mapping)
        or execution.get("ticks_per_generation") != TICKS_PER_GENERATION
        or execution.get("physical_window_generations") != 1
        or base.get("resolved_profile") != "eiel-production-runtime-v1"
    ):
        raise AdaptiveRebindingQualificationError(
            "base plan is not the qualified resident B48/T160 route"
        )
    root = evidence_root.expanduser().resolve()
    source = deepcopy(base)
    source.pop("plan_sha256", None)
    source["campaign_id"] = "wrc-adaptive-rebinding-v1-b48-canary"
    output = source.get("output")
    if not isinstance(output, Mapping):
        raise AdaptiveRebindingQualificationError("base output contract is absent")
    source["output"] = {
        **dict(output),
        "runtime_root": str(root / "declared-runtime-root"),
        # The qualified reporting contract is observational. The resident
        # owner does not emit provider-native scientific_full segments, so
        # this bounded sentinel uses its supported compact/notable profile.
        "chronicle_profile": "notable",
        "chronicle_archive_enabled": False,
        "evidence_closeout_enabled": False,
    }
    if any(
        not isinstance(cell, Mapping)
        or not isinstance(cell.get("parameters"), Mapping)
        or any(name not in cell["parameters"] for name in REBOUND_PARAMETERS)
        for cell in source["cells"]
    ):
        raise AdaptiveRebindingQualificationError(
            "base per-cell experiment-variable panel is incomplete"
        )
    _set_generation(source, SOURCE_GENERATION)
    source["plan_sha256"] = canonical_sha256(source)

    # Apply one deterministic cyclic cell mapping selected to be a derangement
    # of this frozen panel.  Every rebound bundle is an already-declared bundle;
    # the mapping changes every cell and exercises all fifteen native vectors.
    parameter_arrays: dict[str, list[Any]] = {}
    for name in REBOUND_PARAMETERS:
        values = [cell["parameters"][name] for cell in source["cells"]]
        parameter_arrays[name] = (
            values[REBOUND_CELL_ROTATION:] + values[:REBOUND_CELL_ROTATION]
        )
    restore = materialize_wrc_adaptive_restore_plan_v1(
        source,
        restore_generation=RESTORE_GENERATION,
        parameter_arrays=parameter_arrays,
    )
    changed_cells = sum(
        any(
            before["parameters"][name] != after["parameters"][name]
            for name in REBOUND_PARAMETERS
        )
        for before, after in zip(source["cells"], restore["cells"], strict=True)
    )
    if changed_cells != 48:
        raise AdaptiveRebindingQualificationError(
            "qualification rebound must change all 48 canary cells"
        )
    providers = default_execution_provider_registry()
    provider = providers.get(str(source.get("resolved_profile", "")))
    runtime_profile = next(
        (
            candidate
            for candidate in default_profile_registry().profiles()
            if candidate.profile_id == source.get("resolved_profile")
        ),
        None,
    )
    if provider is None or runtime_profile is None:
        raise AdaptiveRebindingQualificationError(
            "production provider/profile contract is unavailable"
        )
    batch_id = str(source["physical_batches"][0]["batch_id"])
    source_request = provider.translate_batch(source, batch_id)
    restore_request = provider.translate_batch(restore, batch_id)
    static_contract = validate_wrc_checkpoint_adaptive_rebinding_v1(
        source_plan=source,
        restore_plan=restore,
        source_request=source_request,
        restore_request=restore_request,
        checkpoint_generation=SOURCE_GENERATION,
        profile=runtime_profile,
        generation_min=provider.resident_segment_generation_min,
        generation_max=provider.resident_segment_generation_max,
    )
    if (
        static_contract.get("changed_parameter_names")
        != sorted(REBOUND_PARAMETERS)
        or len(static_contract.get("changed_cells", [])) != 48
    ):
        raise AdaptiveRebindingQualificationError(
            "static adaptive contract does not cover all B48 cells"
        )
    qualification_receipt = source.get("artifacts", {}).get(
        "qualification_receipt"
    )
    if not isinstance(qualification_receipt, Mapping) or not isinstance(
        qualification_receipt.get("sha256"), str
    ):
        raise AdaptiveRebindingQualificationError(
            "existing route qualification receipt is absent"
        )
    qualification_impact_map = {
        "procedure": "EIEL-QPDQ-1",
        "existing_qualified_groups": [
            {
                "group_id": "B48-T160-G3-G128-PRODUCTION-ROUTE",
                "classification": "PRESERVED_UNCHANGED",
                "evidence_sha256": qualification_receipt["sha256"],
                "reason": (
                    "route, mechanics, artifacts, schema, qualification controls, "
                    "and T160 generation-boundary envelope are unchanged"
                ),
            },
            {
                "group_id": "WRC-PARAMETER-VARIABILITY-ENVELOPE-V1",
                "classification": "PRESERVED_COMPATIBLE_INTERFACE",
                "evidence_sha256": static_contract[
                    "variability_envelope_sha256"
                ],
                "reason": (
                    "adaptive rebinding accepts only already-covered per-cell "
                    "EXPERIMENT_VARIABLE values and validates complete context"
                ),
            },
            {
                "group_id": "LEGACY-GENERATION-BOUNDARY-RESTORE",
                "classification": "PRESERVED_COMPATIBLE_INTERFACE",
                "reason": (
                    "unchanged adaptive entry delegates to and must be identical "
                    "to legacy restore preparation"
                ),
            },
            {
                "group_id": "PRODUCTION-REPORTING-OBSERVATIONAL-CONTRACT",
                "classification": "PRESERVED_COMPATIBLE_INTERFACE",
                "reason": (
                    "the bounded canary uses the supported compact/notable "
                    "profile under the existing observational reporting "
                    "contract; reporting cannot mutate resident state"
                ),
            },
            {
                "group_id": CAPABILITY_ID,
                "classification": "DELTA_QUALIFICATION_REQUIRED",
                "reason": (
                    "new post-checkpoint parameter-array replacement and runtime "
                    "readback boundary"
                ),
            },
        ],
        "exact_changed_layer": CAPABILITY_ID,
        "invariants_unchanged": [
            "world state",
            "population",
            "lineages",
            "RNG state and order",
            "state schema",
            "mechanics binding",
            "production artifact",
            "route identity",
            "qualification-affecting controls",
        ],
        "possible_interaction_surfaces": [
            "generation-boundary checkpoint projection and reload",
            "per-cell WRC parameter-array translation",
            "first continued-segment runtime readback",
        ],
        "representative_sentinels_required": [
            "one physical-B48 T160 G3-to-G6 rebound/continue canary",
            "unchanged legacy-restore identity proof from the same checkpoint",
            "host fail-closed envelope, context, and mapping negatives",
        ],
        "qualification_inherited": [
            "B48/T160/G3-G128 route and artifact qualification",
            "WRC variability envelope value coverage",
        ],
        "targeted_delta_qualification_required": [
            CAPABILITY_ID,
            "checkpoint reload plus adaptive array translation interaction",
            "runtime readback of all 48 rebound cells",
        ],
        "genuinely_invalidated_qualification": [],
        "cartesian_matrix_planned": False,
        "broad_requalification_required": False,
        "broad_requalification_reason": (
            "no supporting assumption of an existing qualified group changed"
        ),
    }
    profile = {
        "schema_version": "wrc_checkpoint_adaptive_rebinding_readiness_profile_v1",
        "qualification_id": QUALIFICATION_ID,
        "capability_id": CAPABILITY_ID,
        "source_commit": _source_commit(),
        "base_plan_sha256": base["plan_sha256"],
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "static_contract_receipt_sha256": static_contract["receipt_sha256"],
        "qualification_impact_map": qualification_impact_map,
        "route_scope": {
            "profile_id": source["resolved_profile"],
            "physical_batch_count": 1,
            "physical_batch_size": 48,
            "ticks_per_generation": TICKS_PER_GENERATION,
            "source_generation": SOURCE_GENERATION,
            "restore_generation": RESTORE_GENERATION,
            "qualified_route_reused": True,
            "broad_route_requalification": False,
            "tick_native_horizon": False,
            "mid_generation_checkpoint_restore": False,
        },
        "host_proofs": list(HOST_TESTS),
        "live_canary": {
            "backend": "cuda_resident",
            "source_shape": "physical B48 G3/T160",
            "restore_shape": "same physical B48 G3-to-G6/T160",
            "mechanics_context": (
                "one existing representative shared-owner configuration"
            ),
            "rebound_parameters": list(REBOUND_PARAMETERS),
            "changed_cells": changed_cells,
        },
        "acceptance": [
            "unchanged API output is byte-identical to legacy restore preparation",
            "all 48 checkpoint projections reload with exact parent digests",
            "cell, seed, RNG-lineage, route, schema, mechanics, and artifact identity remain bound",
            "only the fifteen declared covered experiment-variable arrays change",
            "first continued segment runtime readback matches all rebound arrays",
            "qualification receipt action remains unchanged",
        ],
        "status": "QUALPREP_PLAN_ONLY",
    }
    return (
        source,
        restore,
        static_contract,
        {**profile, "receipt_sha256": canonical_sha256(profile)},
    )


def materialize_plan_packet(
    base_plan_path: Path,
    *,
    output_root: Path,
    allocation_diagnostic_receipt_path: Path | None = None,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    source, restore, static_contract, profile = materialize_qualification_plans(
        _load_json(base_plan_path, name="base plan"), evidence_root=root
    )
    allocation_diagnostic_receipt = (
        None
        if allocation_diagnostic_receipt_path is None
        else _load_json(
            allocation_diagnostic_receipt_path.expanduser().resolve(),
            name="allocation diagnostic receipt",
        )
    )
    allocation_diagnostic_receipt_sha256 = (
        None
        if allocation_diagnostic_receipt is None
        else canonical_sha256(allocation_diagnostic_receipt)
    )
    if allocation_diagnostic_receipt_sha256 is not None:
        unsigned_profile = dict(profile)
        unsigned_profile.pop("receipt_sha256", None)
        unsigned_profile["allocation_diagnostic_receipt_sha256"] = (
            allocation_diagnostic_receipt_sha256
        )
        profile = {
            **unsigned_profile,
            "receipt_sha256": canonical_sha256(unsigned_profile),
        }
    source_path = root / "source_plan.json"
    restore_path = root / "restore_plan.json"
    profile_path = root / "readiness_profile.json"
    static_contract_path = root / "static_contract_receipt.json"
    packet_files: list[tuple[Path, Mapping[str, Any]]] = [
        (source_path, source),
        (restore_path, restore),
        (static_contract_path, static_contract),
        (profile_path, profile),
    ]
    allocation_diagnostic_receipt_path_in_packet = (
        root / "allocation_diagnostic_receipt.json"
    )
    if allocation_diagnostic_receipt is not None:
        packet_files.append(
            (
                allocation_diagnostic_receipt_path_in_packet,
                allocation_diagnostic_receipt,
            )
        )
    for path, value in packet_files:
        if path.exists() and _load_json(path, name=path.name) != value:
            raise AdaptiveRebindingQualificationError(
                f"qualification packet collision: {path}"
            )
        atomic_write_json(path, value)
    receipt = {
        "schema_version": "wrc_checkpoint_adaptive_rebinding_plan_receipt_v1",
        "qualification_id": QUALIFICATION_ID,
        "capability_id": CAPABILITY_ID,
        "source_plan_path": str(source_path),
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_path": str(restore_path),
        "restore_plan_sha256": restore["plan_sha256"],
        "readiness_profile_path": str(profile_path),
        "readiness_profile_sha256": profile["receipt_sha256"],
        "static_contract_receipt_path": str(static_contract_path),
        "static_contract_receipt_sha256": static_contract["receipt_sha256"],
        "qualification_executed": False,
        "gpu_used": False,
        "status": "QUALPREP_PLAN_GENERATED",
    }
    if allocation_diagnostic_receipt_sha256 is not None:
        receipt["allocation_diagnostic_receipt_path"] = str(
            allocation_diagnostic_receipt_path_in_packet
        )
        receipt["allocation_diagnostic_receipt_sha256"] = (
            allocation_diagnostic_receipt_sha256
        )
    receipt = {**receipt, "receipt_sha256": canonical_sha256(receipt)}
    atomic_write_json(root / "plan_receipt.json", receipt)
    return receipt


def _write_exact(path: Path, value: Mapping[str, Any], *, name: str) -> None:
    if path.exists() and _load_json(path, name=name) != value:
        raise AdaptiveRebindingQualificationError(f"{name} differs: {path}")
    atomic_write_json(path, value)


def _preflight_with_measurement(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    provider_registry: Any,
    allocation_diagnostic_resource_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Promote one exact plan through its authorized zero-dispatch measurement."""

    initial = preflight_campaign(
        plan,
        provider_registry=provider_registry,
        allocation_diagnostic_resource_receipt=(
            allocation_diagnostic_resource_receipt
        ),
    )
    if initial.get("ready_for_execution") is True:
        receipt = initial
    else:
        prepared = initial.get("prepared_owner_resource_audit")
        authorization = (
            prepared.get("measurement_authorization")
            if isinstance(prepared, Mapping)
            else None
        )
        batches = plan.get("physical_batches")
        if (
            initial.get("ready_for_prepared_owner_measurement") is not True
            or not isinstance(authorization, Mapping)
            or not isinstance(batches, list)
            or len(batches) != 1
            or not isinstance(batches[0], Mapping)
        ):
            raise AdaptiveRebindingQualificationError(
                "exact canary plan is not eligible for resource measurement"
            )
        provider = provider_registry.get(str(plan.get("resolved_profile", "")))
        measure = getattr(provider, "measure_prepared_owner_resources_v1", None)
        if provider is None or not callable(measure):
            raise AdaptiveRebindingQualificationError(
                "canary provider lacks prepared-owner measurement"
            )
        batch_id = str(batches[0].get("batch_id"))
        request = provider.translate_batch(plan, batch_id)
        resource_receipt = measure(
            plan,
            request,
            audit_root=run_root / "prepared_owner_resource_audit",
            audit_id=f"{batch_id}-{str(plan['plan_sha256'])[:12]}",
            measurement_authorization=authorization,
        )
        if not isinstance(resource_receipt, Mapping):
            raise AdaptiveRebindingQualificationError(
                "prepared-owner measurement returned no receipt"
            )
        atomic_write_json(
            run_root / "prepared_owner_resource_receipt.json",
            dict(resource_receipt),
        )
        receipt = preflight_campaign(
            plan,
            provider_registry=provider_registry,
            prepared_owner_resource_receipt=resource_receipt,
        )
    if (
        receipt.get("ready_for_execution") is not True
        or receipt.get("reasons") not in (None, [])
        or receipt.get("execution_authority_reasons") not in (None, [])
    ):
        raise AdaptiveRebindingQualificationError(
            "measured canary plan is not execution-ready"
        )
    atomic_write_json(run_root / "preflight_receipt.json", receipt)
    return receipt


def _run_host_proofs(root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *HOST_TESTS]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    receipt = {
        "schema_version": "wrc_checkpoint_adaptive_rebinding_host_proof_v1",
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
    receipt = {**receipt, "receipt_sha256": canonical_sha256(receipt)}
    atomic_write_json(root / "host_proof_receipt.json", receipt)
    if result.returncode != 0:
        raise AdaptiveRebindingQualificationError("host proof suite failed")
    return receipt


def _batch_manifest(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or len(batches) != 1:
        raise AdaptiveRebindingQualificationError("canary batch mapping differs")
    batch_id = str(batches[0].get("batch_id"))
    return _load_json(
        root / "batches" / batch_id / "manifest.json",
        name="canary batch manifest",
    )


def _validated_readiness(
    root: Path,
    *,
    source: Mapping[str, Any],
    restore: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    readiness = _load_json(
        root / "readiness_receipt.json", name="QualPrep readiness receipt"
    )
    unsigned = dict(readiness)
    declared = unsigned.pop("receipt_sha256", None)
    if (
        readiness.get("schema_version")
        != "wrc_checkpoint_adaptive_rebinding_readiness_v1"
        or readiness.get("qualification_id") != QUALIFICATION_ID
        or readiness.get("capability_id") != CAPABILITY_ID
        or readiness.get("source_commit") != source_commit
        or readiness.get("source_plan_sha256") != source.get("plan_sha256")
        or readiness.get("restore_plan_sha256") != restore.get("plan_sha256")
        or readiness.get("status") != "READY_FOR_QUALIFICATION"
        or readiness.get("qualification_executed") is not False
        or readiness.get("source_repairs_after_complete_pass") != 0
        or not isinstance(declared, str)
        or canonical_sha256(unsigned) != declared
    ):
        raise AdaptiveRebindingQualificationError(
            "repair-free QualPrep readiness receipt is absent or stale"
        )
    return {**unsigned, "receipt_sha256": declared}


def _execute_bounded_canary_packet(
    packet_root: Path,
    *,
    phase: str,
    provider_registry: Any | None = None,
) -> dict[str, Any]:
    """Execute one bounded live B48 canary for QualPrep or Qualification."""

    root = packet_root.expanduser().resolve()
    if phase not in {"qualprep", "qualification"}:
        raise AdaptiveRebindingQualificationError("unknown canary phase")
    source = _verified_plan(
        _load_json(root / "source_plan.json", name="source plan"), name="source"
    )
    restore = _verified_plan(
        _load_json(root / "restore_plan.json", name="restore plan"), name="restore"
    )
    static_contract = validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
        _load_json(
            root / "static_contract_receipt.json",
            name="static adaptive contract receipt",
        )
    )
    profile = _load_json(root / "readiness_profile.json", name="readiness profile")
    allocation_diagnostic_receipt_path = (
        root / "allocation_diagnostic_receipt.json"
    )
    allocation_diagnostic_receipt = (
        _load_json(
            allocation_diagnostic_receipt_path,
            name="allocation diagnostic receipt",
        )
        if allocation_diagnostic_receipt_path.is_file()
        else None
    )
    allocation_diagnostic_receipt_sha256 = (
        canonical_sha256(allocation_diagnostic_receipt)
        if allocation_diagnostic_receipt is not None
        else None
    )
    declared_profile = profile.pop("receipt_sha256", None)
    if (
        profile.get("qualification_id") != QUALIFICATION_ID
        or profile.get("source_commit") != _source_commit()
        or profile.get("source_plan_sha256") != source["plan_sha256"]
        or profile.get("restore_plan_sha256") != restore["plan_sha256"]
        or profile.get("static_contract_receipt_sha256")
        != static_contract["receipt_sha256"]
        or static_contract.get("source_plan_sha256") != source["plan_sha256"]
        or static_contract.get("restore_plan_sha256") != restore["plan_sha256"]
        or profile.get("allocation_diagnostic_receipt_sha256")
        != allocation_diagnostic_receipt_sha256
        or not isinstance(declared_profile, str)
        or canonical_sha256(profile) != declared_profile
    ):
        raise AdaptiveRebindingQualificationError(
            "qualification packet is stale or malformed"
        )
    readiness = (
        _validated_readiness(
            root,
            source=source,
            restore=restore,
            source_commit=str(profile["source_commit"]),
        )
        if phase == "qualification"
        else None
    )
    providers = provider_registry or default_execution_provider_registry()
    phase_root = root / phase
    host = _run_host_proofs(phase_root)
    source_root = phase_root / "source_run"
    restore_root = phase_root / "restore_run"
    _write_exact(
        source_root / "execution_plan.json", source, name="source execution plan"
    )
    _preflight_with_measurement(
        source,
        run_root=source_root,
        provider_registry=providers,
        allocation_diagnostic_resource_receipt=(
            allocation_diagnostic_receipt
        ),
    )
    source_execution = execute_campaign(
        source, run_root=source_root, provider_registry=providers
    )
    if source_execution.get("status") != "completed":
        raise AdaptiveRebindingQualificationError("source B48 canary did not complete")
    source_manifest = _batch_manifest(source_root, source)
    checkpoint = source_manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise AdaptiveRebindingQualificationError("source checkpoint is absent")
    provider = providers.get(str(source.get("resolved_profile", "")))
    if provider is None:
        raise AdaptiveRebindingQualificationError("canary provider is unavailable")
    batch_id = str(source["physical_batches"][0]["batch_id"])
    source_request = provider.translate_batch(source, batch_id)
    legacy = provider.prepare_fresh_owner_restore_checkpoint(
        source, source_request, source_request, checkpoint
    )
    unchanged = provider.prepare_wrc_checkpoint_adaptive_rebinding_v1(
        source, source, source_request, source_request, checkpoint
    )
    if legacy != unchanged:
        raise AdaptiveRebindingQualificationError(
            "unchanged adaptive restore differs from legacy restore"
        )
    unchanged_proof = {
        "schema_version": "wrc_adaptive_unchanged_legacy_identity_v1",
        "source_checkpoint_sha256": canonical_sha256(checkpoint),
        "legacy_restore_sha256": canonical_sha256(legacy),
        "adaptive_unchanged_restore_sha256": canonical_sha256(unchanged),
        "byte_identical": True,
    }
    unchanged_proof = {
        **unchanged_proof,
        "receipt_sha256": canonical_sha256(unchanged_proof),
    }
    atomic_write_json(root / "unchanged_legacy_proof.json", unchanged_proof)

    _write_exact(
        restore_root / "execution_plan.json", restore, name="restore execution plan"
    )
    _preflight_with_measurement(
        restore,
        run_root=restore_root,
        provider_registry=providers,
        allocation_diagnostic_resource_receipt=(
            allocation_diagnostic_receipt
        ),
    )
    transition = prepare_adaptive_campaign_continuation_v1(
        source,
        restore,
        source_run_root=source_root,
        restore_run_root=restore_root,
        provider_registry=providers,
    )
    restore_execution = execute_campaign(
        restore, run_root=restore_root, provider_registry=providers
    )
    if restore_execution.get("status") != "completed":
        raise AdaptiveRebindingQualificationError(
            "rebound B48 continuation did not complete"
        )
    restore_manifest = _batch_manifest(restore_root, restore)
    result = restore_manifest.get("result")
    restore_checkpoint = restore_manifest.get("checkpoint")
    if not isinstance(result, Mapping) or not isinstance(
        restore_checkpoint, Mapping
    ):
        raise AdaptiveRebindingQualificationError(
            "rebound continuation result is incomplete"
        )
    fresh = result.get("fresh_owner_restore_receipt")
    runtime_readback = result.get("wrc_checkpoint_adaptive_runtime_readback")
    adaptive = (
        fresh.get("wrc_checkpoint_adaptive_rebinding")
        if isinstance(fresh, Mapping)
        else None
    )
    if not isinstance(fresh, Mapping) or not isinstance(
        runtime_readback, Mapping
    ) or not isinstance(adaptive, Mapping):
        raise AdaptiveRebindingQualificationError(
            "adaptive continuity/readback evidence is absent"
        )
    validated_adaptive = validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
        adaptive
    )
    source_digests = checkpoint.get("terminal_cell_digests")
    reload_receipts = fresh.get("reload_receipts")
    runtime_unsigned = dict(runtime_readback)
    runtime_declared = runtime_unsigned.pop("receipt_sha256", None)
    expected_cell_ids = list(source["physical_batches"][0]["cell_ids"])
    reload_exact = (
        isinstance(reload_receipts, list)
        and len(reload_receipts) == 48
        and not any(
            not isinstance(row, Mapping)
            or row.get("cell_id") != expected_cell_ids[index]
            or row.get("source_projection_sha256")
            != row.get("loaded_projection_sha256")
            for index, row in enumerate(reload_receipts)
        )
    )
    checkpoint_identity = checkpoint.get("checkpoint_identity_contract")
    parent_digests = (
        checkpoint_identity.get("terminal_projection_digest_sha256s")
        if isinstance(checkpoint_identity, Mapping)
        else source_digests
    )
    continuity_checks = {
        "source_generation": checkpoint.get("generation") == SOURCE_GENERATION,
        "restore_generation": (
            restore_checkpoint.get("generation") == RESTORE_GENERATION
        ),
        "source_digest_cells": (
            isinstance(source_digests, Mapping)
            and set(source_digests) == set(expected_cell_ids)
        ),
        "reloaded_source_digests": (
            fresh.get("source_checkpoint_cell_digests") == source_digests
        ),
        "parent_projection_digests": (
            fresh.get("parent_checkpoint_projection_digest_sha256s")
            == parent_digests
        ),
        "reload_receipts": reload_exact,
        "adaptive_generations": (
            validated_adaptive.get("source_generation") == SOURCE_GENERATION
            and validated_adaptive.get("restore_generation")
            == RESTORE_GENERATION
        ),
        "adaptive_cells": (
            validated_adaptive.get("cell_ids") == expected_cell_ids
            and isinstance(validated_adaptive.get("changed_cells"), list)
            and len(validated_adaptive["changed_cells"]) == 48
        ),
        "adaptive_parameter": (
            validated_adaptive.get("changed_parameter_names")
            == sorted(REBOUND_PARAMETERS)
        ),
        "runtime_status": (
            runtime_readback.get("status") == "READBACK_COMPLETE"
            and runtime_readback.get("schema_version")
            == "wrc_checkpoint_adaptive_runtime_readback_v1"
        ),
        "runtime_binding": (
            runtime_readback.get("adaptive_rebinding_receipt_sha256")
            == validated_adaptive["receipt_sha256"]
            and isinstance(runtime_readback.get("verified_cells"), list)
            and len(runtime_readback["verified_cells"]) == 48
        ),
        "runtime_receipt": (
            isinstance(runtime_declared, str)
            and canonical_sha256(runtime_unsigned) == runtime_declared
        ),
        "qualification_receipt_action": (
            transition.get("qualification_receipt_action") == "unchanged"
        ),
    }
    failed_checks = sorted(
        name for name, passed in continuity_checks.items() if not passed
    )
    if failed_checks:
        raise AdaptiveRebindingQualificationError(
            "adaptive checkpoint continuity proof differs: "
            + ", ".join(failed_checks)
        )
    evidence = {
        "qualification_id": QUALIFICATION_ID,
        "capability_id": CAPABILITY_ID,
        "source_commit": profile["source_commit"],
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "host_proof_receipt_sha256": host["receipt_sha256"],
        "unchanged_legacy_proof_sha256": unchanged_proof["receipt_sha256"],
        "adaptive_transition_receipt_sha256": transition["receipt_sha256"],
        "adaptive_rebinding_receipt_sha256": validated_adaptive["receipt_sha256"],
        "runtime_readback_receipt_sha256": runtime_readback.get("receipt_sha256"),
        "source_checkpoint_sha256": canonical_sha256(checkpoint),
        "restore_checkpoint_sha256": canonical_sha256(restore_checkpoint),
        "physical_batch_size": 48,
        "source_generation": SOURCE_GENERATION,
        "restore_generation": RESTORE_GENERATION,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "qualification_receipt_action": "unchanged",
        "broad_route_requalification_performed": False,
        "qualification_impact_map_sha256": canonical_sha256(
            profile["qualification_impact_map"]
        ),
        "tick_native_or_mid_generation_scope": False,
    }
    if phase == "qualprep":
        receipt = {
            "schema_version": "wrc_checkpoint_adaptive_rebinding_readiness_v1",
            **evidence,
            "q2_host_contract_matrix": "PASS",
            "q3_q4_composed_live_rehearsal": "PASS",
            "q5_repair_free_convergence": "PASS",
            "source_repairs_after_complete_pass": 0,
            "qualification_executed": False,
            "status": "READY_FOR_QUALIFICATION",
        }
        output_path = root / "readiness_receipt.json"
    else:
        receipt = {
            "schema_version": "wrc_checkpoint_adaptive_rebinding_qualification_v1",
            **evidence,
            "readiness_receipt_sha256": readiness["receipt_sha256"],
            "qualification_context": "fresh_frozen_q6",
            "decision": "QUALIFIED",
        }
        output_path = root / "qualification_receipt.json"
    receipt = {**receipt, "receipt_sha256": canonical_sha256(receipt)}
    atomic_write_json(output_path, receipt)
    return receipt


def execute_qualprep_packet(
    packet_root: Path, *, provider_registry: Any | None = None
) -> dict[str, Any]:
    """Run the composed Q2-Q5 rehearsal without making a qualification claim."""

    return _execute_bounded_canary_packet(
        packet_root,
        phase="qualprep",
        provider_registry=provider_registry,
    )


def execute_qualification_packet(
    packet_root: Path, *, provider_registry: Any | None = None
) -> dict[str, Any]:
    """Run fresh frozen Q6 after an exact repair-free readiness receipt."""

    return _execute_bounded_canary_packet(
        packet_root,
        phase="qualification",
        provider_registry=provider_registry,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--base-plan", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--allocation-diagnostic-receipt", type=Path)
    qualprep = subparsers.add_parser("qualprep")
    qualprep.add_argument("packet_root", type=Path)
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("packet_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = materialize_plan_packet(
                args.base_plan,
                output_root=args.output_root,
                allocation_diagnostic_receipt_path=(
                    args.allocation_diagnostic_receipt
                ),
            )
        elif args.command == "qualprep":
            result = execute_qualprep_packet(args.packet_root)
        else:
            result = execute_qualification_packet(args.packet_root)
    except Exception as exc:  # noqa: BLE001 - fail-closed machine receipt.
        result = {
            "schema_version": "wrc_checkpoint_adaptive_rebinding_failure_v1",
            "qualification_id": QUALIFICATION_ID,
            "capability_id": CAPABILITY_ID,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    else:
        result = {**result, "ok": True}
    print(json.dumps(result, indent=2, sort_keys=True) + "\n", end="")
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
