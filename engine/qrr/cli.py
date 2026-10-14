"""Canonical release-only launcher for Qualified Runtime Release V1."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from itertools import product
import shutil
import sys
from typing import Any

from engine.campaigns.canonical import canonical_sha256, pretty_json
from engine.campaigns.compiler import (
    compile_campaign,
    load_campaign_spec,
    resolve_campaign_profile,
)
from engine.campaigns.orchestration import (
    CampaignExecutionError,
    _technical_gate_binding,
    _technical_gate_contract,
    execute_campaign,
)
from engine.campaigns.preflight import preflight_campaign
from engine.campaigns.providers import default_execution_provider_registry
from engine.campaigns.tick_native import TickNativeError, TickNativeExecutionRequestV1
from engine.campaigns.wrc_science import materialize_full_stack_wrc_science_spec_v1
from engine.qrr.qualified_execution_envelope_v1 import (
    QualifiedExecutionEnvelopeV1,
    load_qualified_execution_envelope_v1,
)
from engine.qrr.qualified_runtime_release_v1 import (
    DEFAULT_QRR_RUN_ROOT,
    QualifiedRuntimeReleaseV1,
    load_qualified_runtime_release_v1,
)
from engine.qrr.runability_closure_v1 import (
    assess_runability_closure_v1,
    classify_runability_failure_v1,
)
from engine.qrr.tick_native_materialization_v1 import (
    canonical_tick_native_plan_v1,
    canonical_tick_native_spec_v1,
)


class QrrLauncherError(ValueError):
    """The canonical launcher cannot safely perform the requested operation."""


CANOPY_HOST_RESERVE_PER_CELL_BYTES = 256 * 1024 * 1024
CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES = 256 * 1024 * 1024


_RUNABILITY_GATE_REQUIRED_PROVIDER_CHECKS = (
    "selected_cells_exact",
    "selected_cells_present_once",
    "manifest_roles_exact",
    "plan_and_gate_binding_exact",
    "checkpoint_ticks_exact",
    "canary_execute_completed",
    "fresh_owner_restore_completed",
    "fresh_owner_terminal_checkpoint_bound",
    "cuda_without_fallback",
    "fresh_owner_bundle_reload_and_equality_proven",
)

_RUNABILITY_GATE_EXCLUDED_SCIENCE_REASON_PREFIXES = (
    "Genome V2 WDL technical gate profile conflicts with compiled mechanics",
    "compiled Genome V2 WDL identity or mutation arm is invalid:",
    "canary Genome V2 causal projection exposure receipts are incomplete",
    "fresh-owner Genome V2 causal projection exposure receipts are incomplete",
    "provider-native reproduction exposure receipts are incomplete",
    "fresh-owner telemetry differs from the canary evidence",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QrrLauncherError(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise QrrLauncherError(f"{name} must be a JSON object")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(path)


def assert_release_module_origins_v1(release_root: Path) -> None:
    """Reject semantic imports from a checkout outside the immutable payload."""

    payload = (release_root / "payload/python").resolve()
    offenders: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "engine"
            or name.startswith("engine.")
            or name == "tools"
            or name.startswith("tools.")
        ):
            continue
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            continue
        path = Path(origin).resolve()
        if path != payload and payload not in path.parents:
            offenders.append(f"{name}={path}")
    if offenders:
        raise QrrLauncherError(
            "QRR loaded engine modules outside the release payload: "
            + ", ".join(sorted(offenders))
        )


def _release_context(
    release_root: Path,
) -> tuple[QualifiedRuntimeReleaseV1, QualifiedExecutionEnvelopeV1]:
    release = load_qualified_runtime_release_v1(release_root)
    envelope = load_qualified_execution_envelope_v1(
        release.qualified_execution_envelope
    )
    return release, envelope


def _experiment_spec(
    *,
    release_root: Path,
    envelope: QualifiedExecutionEnvelopeV1,
    experiment_path: Path,
    run_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    experiment = _read_mapping(experiment_path, name="QRR experiment")
    axes = experiment.get("axes")
    configuration_rows = experiment.get("configurations", ())
    seeds = experiment.get("seeds", {})
    fixed_context = experiment.get("fixed_context_axes", {})
    total_ticks = experiment.get("total_ticks", experiment.get("absolute_ticks"))
    checkpoint_ticks = experiment.get("checkpoint_ticks")
    checkpoint_every_ticks = experiment.get(
        "checkpoint_every_ticks",
        experiment.get("checkpoint_interval_absolute_ticks"),
    )
    deliberate_restore_tick = experiment.get(
        "deliberate_restore_tick", experiment.get("intentional_restore_tick")
    )
    deliberate_restore_ticks = experiment.get("deliberate_restore_ticks", ())
    exercise_checkpoint_resume = experiment.get("exercise_checkpoint_resume", False)
    if (
        not isinstance(axes, Mapping)
        or not isinstance(seeds, Mapping)
        or not isinstance(fixed_context, Mapping)
        or not isinstance(configuration_rows, (list, tuple))
    ):
        raise QrrLauncherError(
            "experiment axes, seeds, fixed_context_axes, and configurations are malformed"
        )
    if type(exercise_checkpoint_resume) is not bool:
        raise QrrLauncherError("exercise_checkpoint_resume must be an exact boolean")
    if not isinstance(deliberate_restore_ticks, (list, tuple)):
        raise QrrLauncherError("deliberate_restore_ticks must be an integer array")
    base = load_campaign_spec(release_root / "payload/spec/base_spec.yaml")
    profile = resolve_campaign_profile(base)
    try:
        tick_request = TickNativeExecutionRequestV1(
            total_ticks=total_ticks,
            checkpoint_ticks=(
                tuple(checkpoint_ticks)
                if isinstance(checkpoint_ticks, (list, tuple))
                else ()
            ),
            checkpoint_every_ticks=checkpoint_every_ticks,
            deliberate_restore_tick=deliberate_restore_tick,
            deliberate_restore_ticks=tuple(deliberate_restore_ticks),
        )
    except TickNativeError as exc:
        raise QrrLauncherError(f"invalid absolute-tick experiment: {exc}") from exc
    if (
        exercise_checkpoint_resume
        and not tick_request.canonical_deliberate_restore_ticks
    ):
        raise QrrLauncherError(
            "tick-native checkpoint/resume exercise requires deliberate_restore_tick"
        )
    spec, _variability = materialize_full_stack_wrc_science_spec_v1(
        base,
        profile=profile,
        axes={str(name): list(values) for name, values in axes.items()},
        configuration_rows=tuple(configuration_rows),
        seeds={str(name): list(values) for name, values in seeds.items()},
        fixed_context_axes={
            str(name): list(values) for name, values in fixed_context.items()
        },
        generations=None,
    )
    if isinstance(experiment.get("artifact_lifecycle"), Mapping):
        # The request is an explicit lifecycle authority for this materialized
        # QRR.  It is validated by the production front door and then bound
        # again by the ordinary campaign compiler below.
        spec["artifact_lifecycle"] = dict(experiment["artifact_lifecycle"])
    spec = _apply_acs_checkpoint_selection_v1(spec, experiment=experiment)
    legacy_plan = compile_campaign(spec).payload()
    plan = canonical_tick_native_plan_v1(
        legacy_plan,
        tick_request=tick_request,
        exercise_checkpoint_resume=exercise_checkpoint_resume,
    )
    plan = _bind_science_materialization_v1(plan, experiment=experiment)
    spec = canonical_tick_native_spec_v1(
        spec,
        tick_request=tick_request,
        exercise_checkpoint_resume=exercise_checkpoint_resume,
    )
    destination = (
        run_root.expanduser().resolve()
        if run_root is not None
        else (
            DEFAULT_QRR_RUN_ROOT / release_root.name / str(spec["campaign_id"])
        ).resolve()
    )
    output = spec.get("output")
    if not isinstance(output, Mapping):
        raise QrrLauncherError("materialized QRR experiment has no output contract")
    spec["output"] = {**dict(output), "runtime_root": str(destination)}
    providers = default_execution_provider_registry()
    provider = providers.get(str(plan["resolved_profile"]))
    if provider is None:
        raise QrrLauncherError("QRR profile has no release-local provider")
    validation = envelope.validate_plan(plan, provider=provider)
    return spec, plan, validation, destination


def _apply_acs_checkpoint_selection_v1(
    spec: Mapping[str, Any], *, experiment: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind an explicit ACS route without changing omitted legacy experiments."""

    raw_instrumentation = experiment.get("acs_capture_instrumentation", False)
    slots_explicit = "acs_checkpoint_slots" in experiment
    raw_slots = experiment.get("acs_checkpoint_slots", 0)
    if type(raw_instrumentation) is not bool:
        raise QrrLauncherError(
            "acs_capture_instrumentation must be an exact boolean"
        )
    if type(raw_slots) is not int or raw_slots not in {0, 1, 2, 3}:
        raise QrrLauncherError(
            "acs_checkpoint_slots must be zero, one, two, or three"
        )
    instrumentation = raw_instrumentation or bool(raw_slots)
    if not instrumentation and raw_slots == 0 and not slots_explicit:
        return dict(spec)
    value = dict(spec)
    checkpoints = dict(value.get("checkpoints") or {})
    if instrumentation:
        checkpoints["acs_capture_instrumentation"] = True
    if slots_explicit:
        checkpoints["acs_slots"] = raw_slots
    value["checkpoints"] = checkpoints
    return value


def _bind_science_materialization_v1(
    plan: Mapping[str, Any], *, experiment: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind exact requested controls and row/seed identities to every cell."""

    value = dict(plan)
    raw_cells = value.get("cells")
    axes = experiment.get("axes", {})
    raw_rows = experiment.get("configurations", ())
    if (
        not isinstance(raw_cells, list)
        or not raw_cells
        or not isinstance(axes, Mapping)
        or not isinstance(raw_rows, (list, tuple))
        or bool(axes) == bool(raw_rows)
    ):
        raise QrrLauncherError(
            "SCIENCE_DESIGN_INVALID: exactly one science design representation is required"
        )
    requested_controls: set[str]
    configurations: list[dict[str, Any]] = []
    if raw_rows:
        for index, raw in enumerate(raw_rows):
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("controls"), Mapping
            ):
                raise QrrLauncherError(
                    f"SCIENCE_DESIGN_INVALID: configuration {index} is malformed"
                )
            configurations.append(
                {"id": str(raw.get("id") or ""), "controls": dict(raw["controls"])}
            )
        requested_controls = set(configurations[0]["controls"])
    else:
        requested_controls = {str(name) for name in axes}
        ordered_axes = sorted(
            (str(name), tuple(values)) for name, values in axes.items()
        )
        configurations = [
            {
                "id": f"axis-{index:04d}",
                "controls": dict(
                    zip((name for name, _values in ordered_axes), values, strict=True)
                ),
            }
            for index, values in enumerate(
                product(*(values for _name, values in ordered_axes))
            )
        ]
    if not requested_controls:
        raise QrrLauncherError("SCIENCE_DESIGN_INVALID: no controls were requested")
    row_by_digest = {
        canonical_sha256(dict(sorted(row["controls"].items()))): row
        for row in configurations
    }
    if len(row_by_digest) != len(configurations):
        raise QrrLauncherError("SCIENCE_DESIGN_INVALID: configuration rows collide")
    annotated_cells: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    observed_row_counts = {str(row["id"]): 0 for row in configurations}
    for index, raw_cell in enumerate(raw_cells):
        if (
            not isinstance(raw_cell, Mapping)
            or not isinstance(raw_cell.get("parameters"), Mapping)
            or not isinstance(raw_cell.get("seeds"), Mapping)
        ):
            raise QrrLauncherError(
                f"SCIENCE_MATERIALIZATION_DIVERGED: cell {index} is malformed"
            )
        parameters = dict(raw_cell["parameters"])
        missing = sorted(requested_controls - set(parameters))
        if missing:
            raise QrrLauncherError(
                "SCIENCE_MATERIALIZATION_DIVERGED: requested controls disappeared: "
                + ", ".join(missing)
            )
        controls = {name: parameters[name] for name in sorted(requested_controls)}
        row = row_by_digest.get(canonical_sha256(controls))
        if row is None:
            raise QrrLauncherError(
                "SCIENCE_MATERIALIZATION_DIVERGED: materialized control vector was not requested"
            )
        row_id = str(row["id"])
        observed_row_counts[row_id] += 1
        control_sha256 = canonical_sha256(controls)
        cell = {
            **dict(raw_cell),
            "science_configuration_id": row_id,
            "science_configuration_controls_sha256": control_sha256,
            "science_seed_identity": dict(sorted(raw_cell["seeds"].items())),
        }
        annotated_cells.append(cell)
        assignments.append(
            {
                "cell_id": cell.get("cell_id"),
                "configuration_id": row_id,
                "controls_sha256": control_sha256,
                "seeds": dict(sorted(raw_cell["seeds"].items())),
            }
        )
    if len(set(observed_row_counts.values())) != 1 or 0 in observed_row_counts.values():
        raise QrrLauncherError(
            "SCIENCE_MATERIALIZATION_DIVERGED: configuration rows are not crossed evenly with seeds"
        )
    core = {
        "schema_version": "eiel_qrr_science_materialization_closure_v1",
        "status": "PASS",
        "requested_control_names": sorted(requested_controls),
        "requested_configuration_rows": configurations,
        "configuration_row_counts": dict(sorted(observed_row_counts.items())),
        "cell_assignments": assignments,
        "requested_materialized_exact": True,
        "silent_dropped_control_count": 0,
        "value_mismatch_count": 0,
    }
    closure = {**core, "receipt_sha256": canonical_sha256(core)}
    value["cells"] = annotated_cells
    value["science_materialization"] = closure
    raw_observability = experiment.get("observability", {})
    if not isinstance(raw_observability, Mapping):
        raise QrrLauncherError(
            "SCIENCE_DESIGN_INVALID: observability must be an object"
        )
    canopy_enabled = raw_observability.get("living_lineage_canopy", True)
    if type(canopy_enabled) is not bool:
        raise QrrLauncherError(
            "SCIENCE_DESIGN_INVALID: living_lineage_canopy must be boolean"
        )
    value["observability"] = {
        "living_lineage_canopy": canopy_enabled,
        "authority": "read_only_external_projection_v1",
    }
    value.pop("plan_sha256", None)
    value["plan_sha256"] = canonical_sha256(value)
    return value


def _first_matrix_selector_v1(matrix: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = matrix.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise QrrLauncherError("QRR technical gate requires at least one matrix cell")
    selector: dict[str, Any] = {}
    for row in dimensions:
        dimension_id = row.get("id") if isinstance(row, Mapping) else None
        values = row.get("values") if isinstance(row, Mapping) else None
        if (
            not isinstance(dimension_id, str)
            or not dimension_id
            or not isinstance(values, list)
            or not values
        ):
            raise QrrLauncherError("QRR technical-gate matrix dimension is malformed")
        selector[dimension_id] = values[0]
    return selector


def _rebind_technical_gate_selector_v1(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an inherited gate to the first cell of the materialized experiment."""

    configured = dict(spec)
    matrix = configured.get("matrix")
    execution = configured.get("execution")
    if not isinstance(matrix, Mapping) or not isinstance(execution, Mapping):
        raise QrrLauncherError("QRR experiment requires matrix and execution")
    technical_gate = execution.get("technical_gate")
    if technical_gate is None:
        return configured
    if not isinstance(technical_gate, Mapping):
        raise QrrLauncherError("QRR technical-gate contract is malformed")
    configured["execution"] = {
        **dict(execution),
        "technical_gate": {
            **dict(technical_gate),
            "selectors": [_first_matrix_selector_v1(matrix)],
        },
    }
    return configured


def _configure_checkpoint_resume_probe_v1(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a QRR fresh-owner restore probe to explicit absolute ticks."""

    configured = dict(spec)
    matrix = configured.get("matrix")
    execution = configured.get("execution")
    checkpoints = configured.get("checkpoints")
    if (
        not isinstance(matrix, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(checkpoints, Mapping)
    ):
        raise QrrLauncherError(
            "checkpoint/resume probe requires matrix, execution, and checkpoints"
        )
    try:
        request = TickNativeExecutionRequestV1.from_execution(dict(execution))
    except TickNativeError as exc:
        raise QrrLauncherError(
            f"checkpoint/resume probe requires a tick-native execution: {exc}"
        ) from exc
    restore_ticks = request.canonical_deliberate_restore_ticks
    if not restore_ticks:
        raise QrrLauncherError(
            "checkpoint/resume probe requires an explicit absolute restore tick"
        )
    checkpoint_ticks = sorted({*restore_ticks, request.total_ticks})
    configured["execution"] = {
        **dict(execution),
        "technical_gate": {
            "total_ticks": request.total_ticks,
            "checkpoint_ticks": checkpoint_ticks,
            "partition": False,
            "profile": "genome_v2_wdl_fixed_mutation_persistence_v1",
            "selectors": [_first_matrix_selector_v1(matrix)],
        },
    }
    configured["checkpoints"] = {
        **dict(checkpoints),
        "milestones": [],
    }
    return configured


def _receipt_argument(path: Path | None, *, name: str) -> Mapping[str, Any] | None:
    return (
        None if path is None else _read_mapping(path.expanduser().resolve(), name=name)
    )


def _write_run_binding(
    *,
    release: QualifiedRuntimeReleaseV1,
    envelope: QualifiedExecutionEnvelopeV1,
    validation: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    path = run_root / "qrr_run_binding.json"
    expected = {
        "schema_version": "eiel_qrr_run_binding_v1",
        "release_id": release.release_id,
        "envelope_sha256": envelope.envelope_sha256,
        "plan_sha256": validation.get("plan_sha256"),
        "bound_request_sha256s": validation.get("bound_request_sha256s"),
        "representability_receipt_sha256": validation.get(
            "representability_receipt_sha256"
        ),
        "launcher": "qrr_v1",
        "source_edits_after_qrr_creation": 0,
        "route_reconstruction_steps": 0,
        "native_rebuilds": 0,
        "runtime_jit_builds": 0,
        "qualification_actions": 0,
        "resume_invocations": 0,
        "created_at_utc": _now(),
    }
    if path.exists():
        current = _read_mapping(path, name="QRR run binding")
        for name in (
            "release_id",
            "envelope_sha256",
            "plan_sha256",
            "bound_request_sha256s",
            "representability_receipt_sha256",
            "launcher",
        ):
            if current.get(name) != expected[name]:
                raise QrrLauncherError(
                    f"run root is attributed to another QRR identity: {name}"
                )
        return current
    _atomic_json(path, expected)
    return expected


def _checkpoint_resume_status(
    run_root: Path,
    binding: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> tuple[bool, bool]:
    gate_path = run_root / "technical_gate_receipt.json"
    gate_passed = False
    if gate_path.is_file():
        gate = _read_mapping(gate_path, name="technical gate receipt")
        gate_passed = gate.get("passed") is True and isinstance(
            gate.get("fresh_owner_restore_manifest_path"), str
        )
    resumed = int(binding.get("resume_invocations", 0)) > 0
    exercised = gate_passed or resumed
    exact = gate_passed or (resumed and execution.get("status") == "completed")
    return exercised, exact


def _write_run_receipt(
    *,
    release: QualifiedRuntimeReleaseV1,
    run_root: Path,
    execution: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    exercised, exact = _checkpoint_resume_status(run_root, binding, execution)
    receipt = {
        "schema_version": "eiel_qrr_run_receipt_v1",
        "release_id": release.release_id,
        "plan_sha256": execution.get("plan_sha256"),
        "execution_route_sha256": execution.get("execution_route_sha256"),
        "status": execution.get("status"),
        "checkpoint_resume_exercised": exercised,
        "checkpoint_resume_exact": exact,
        "representability_receipt_sha256": binding.get(
            "representability_receipt_sha256"
        ),
        "execution_receipt": dict(execution),
        "updated_at_utc": _now(),
    }
    _atomic_json(run_root / "qrr_run_receipt.json", receipt)
    return receipt


def _preflight(
    *,
    release_root: Path,
    plan: Mapping[str, Any],
    run_root: Path,
    prepared_owner_receipt: Path | None,
    allocation_receipt: Path | None,
) -> dict[str, Any]:
    sealed_promotion: Mapping[str, Any] | None = None
    if prepared_owner_receipt is None and allocation_receipt is None:
        sealed_preflight = _read_mapping(
            release_root / "payload/receipts/preflight_receipt.json",
            name="sealed QRR preflight receipt",
        )
        resources = sealed_preflight.get("resources")
        candidate = (
            resources.get("allocation_diagnostic_resource_promotion")
            if isinstance(resources, Mapping)
            else None
        )
        if isinstance(candidate, Mapping):
            sealed_promotion = candidate
    receipt = preflight_campaign(
        plan,
        provider_registry=default_execution_provider_registry(),
        prepared_owner_resource_receipt=_receipt_argument(
            prepared_owner_receipt, name="prepared-owner receipt"
        ),
        allocation_diagnostic_resource_receipt=_receipt_argument(
            allocation_receipt, name="allocation receipt"
        ),
        sealed_allocation_resource_promotion=sealed_promotion,
        enforce_storage_admission=True,
    )
    receipt = _complete_production_resource_admission_v1(
        receipt,
        plan=plan,
        run_root=run_root,
    )
    _atomic_json(run_root / "preflight_receipt.json", receipt)
    return receipt


def _meminfo_bytes_v1() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        rows = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise QrrLauncherError(
            "HOST_RAM_BLOCKED: /proc/meminfo is unavailable"
        ) from exc
    for row in rows:
        name, separator, raw = row.partition(":")
        if not separator:
            continue
        fields = raw.strip().split()
        if fields and fields[0].isdigit():
            values[name] = int(fields[0]) * 1024
    return values


def _complete_production_resource_admission_v1(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    """Add live host, storage, restore, staging, and Canopy admission."""

    value = dict(receipt)
    resources = dict(value.get("resources") or {})
    promotion = resources.get("allocation_diagnostic_resource_promotion")
    estimate = plan.get("resource_estimate")
    cells = plan.get("cells")
    execution = plan.get("execution")
    observability = plan.get("observability")
    if (
        not isinstance(promotion, Mapping)
        or not isinstance(estimate, Mapping)
        or not isinstance(cells, list)
        or not isinstance(execution, Mapping)
    ):
        raise QrrLauncherError(
            "RESOURCE_BLOCKED: sealed production resource authority is incomplete"
        )
    host_peak = promotion.get("measured_host_peak_bytes")
    restore_peak = promotion.get("restore_peak_host_bytes_upper_bound")
    if type(host_peak) is not int or host_peak <= 0 or type(restore_peak) is not int:
        raise QrrLauncherError(
            "HOST_RAM_BLOCKED: sealed B48 host/restore peak measurement is absent"
        )
    checkpoint_count = int(estimate.get("checkpoint_boundaries") or 0)
    canopy_enabled = bool(
        isinstance(observability, Mapping)
        and observability.get("living_lineage_canopy") is True
    )
    # Relay updates share a run-group lock, so only one cell reconstructs its
    # retained checkpoint history at a time.  Reserve every idle watcher plus
    # one checkpoint-scaled active scan.  The constants bound the measured
    # repaired three-checkpoint B48 samples (213 MiB idle, 785 MiB active).
    canopy_watcher_reserve_bytes = (
        len(cells) * CANOPY_HOST_RESERVE_PER_CELL_BYTES if canopy_enabled else 0
    )
    canopy_active_scan_reserve_bytes = (
        checkpoint_count * CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES
        if canopy_enabled
        else 0
    )
    canopy_host_bytes = (
        canopy_watcher_reserve_bytes + canopy_active_scan_reserve_bytes
    )
    canopy_disk_bytes = (
        len(cells) * checkpoint_count * 1024 * 1024 if canopy_enabled else 0
    )
    checkpoint_staging_bytes = len(cells) * 64 * 1024 * 1024
    checkpoint_host_staging_bytes = (
        checkpoint_staging_bytes if checkpoint_count else 0
    )
    planned_disk = estimate.get("disk_bytes_upper_bound")
    if type(planned_disk) is not int or planned_disk <= 0:
        raise QrrLauncherError("STORAGE_BLOCKED: checkpoint-aware estimate is absent")
    disk_before_headroom = planned_disk + checkpoint_staging_bytes + canopy_disk_bytes
    storage_headroom = (disk_before_headroom + 9) // 10
    required_disk = disk_before_headroom + storage_headroom
    destination_parent = run_root.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(destination_parent)
    memory = _meminfo_bytes_v1()
    available_host = memory.get("MemAvailable")
    if type(available_host) is not int:
        raise QrrLauncherError("HOST_RAM_BLOCKED: MemAvailable is absent")
    checkpoint_host_peak = host_peak + checkpoint_host_staging_bytes
    aggregate_checkpoint_host_peak = checkpoint_host_peak + canopy_host_bytes
    aggregate_restore_host_peak = restore_peak + canopy_host_bytes
    checkpoint_host_headroom = (aggregate_checkpoint_host_peak + 9) // 10
    restore_host_headroom = (aggregate_restore_host_peak + 9) // 10
    checkpoint_required_host = (
        aggregate_checkpoint_host_peak + checkpoint_host_headroom
    )
    restore_required_host = aggregate_restore_host_peak + restore_host_headroom
    required_host = max(checkpoint_required_host, restore_required_host)
    failures: list[dict[str, Any]] = []
    if available_host < checkpoint_required_host:
        failures.append(
            {
                "code": "HOST_RAM_BLOCKED",
                "required_bytes": checkpoint_required_host,
                "available_bytes": available_host,
            }
        )
    if available_host < restore_required_host:
        failures.append(
            {
                "code": "RESTORE_PEAK_BLOCKED",
                "required_bytes": restore_required_host,
                "available_bytes": available_host,
            }
        )
    if disk.free < required_disk:
        failures.append(
            {
                "code": "STORAGE_BLOCKED",
                "required_bytes": required_disk,
                "available_bytes": disk.free,
            }
        )
    production = {
        "schema_version": "eiel_qrr_production_resource_admission_v1",
        "status": "RESOURCE_ADMITTED" if not failures else "RESOURCE_BLOCKED",
        "manual_resource_receipt_selection": 0,
        "host_ram": {
            "measured_owner_peak_bytes": host_peak,
            "checkpoint_staging_reserve_bytes": checkpoint_host_staging_bytes,
            "canopy_reserve_bytes": canopy_host_bytes,
            "canopy_watcher_reserve_bytes": canopy_watcher_reserve_bytes,
            "canopy_reserve_per_cell_bytes": (
                CANOPY_HOST_RESERVE_PER_CELL_BYTES if canopy_enabled else 0
            ),
            "canopy_active_scan_concurrency": 1 if canopy_enabled else 0,
            "canopy_active_scan_reserve_bytes": canopy_active_scan_reserve_bytes,
            "canopy_active_scan_reserve_per_checkpoint_bytes": (
                CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES
                if canopy_enabled
                else 0
            ),
            "checkpoint_peak_before_headroom_bytes": checkpoint_host_peak,
            "aggregate_checkpoint_peak_before_headroom_bytes": (
                aggregate_checkpoint_host_peak
            ),
            "safety_headroom_bytes": checkpoint_host_headroom,
            "required_bytes": required_host,
            "available_bytes": available_host,
            "swap_relied_upon": False,
        },
        "storage": {
            "checkpoint_aware_plan_bytes": planned_disk,
            "checkpoint_staging_bytes": checkpoint_staging_bytes,
            "canopy_evidence_reserve_bytes": canopy_disk_bytes,
            "safety_headroom_bytes": storage_headroom,
            "required_bytes": required_disk,
            "available_bytes": disk.free,
            "checkpoint_count": checkpoint_count,
        },
        "restore_peak": {
            "host_bytes_upper_bound": restore_peak,
            "canopy_reserve_bytes": canopy_host_bytes,
            "aggregate_peak_before_headroom_bytes": aggregate_restore_host_peak,
            "safety_headroom_bytes": restore_host_headroom,
            "required_bytes": restore_required_host,
            "covered": available_host >= restore_required_host,
        },
        "failures": failures,
    }
    resources["production_admission"] = production
    value["resources"] = resources
    if failures:
        execution_reasons = list(value.get("execution_authority_reasons") or [])
        execution_reasons.extend(failures)
        value["execution_authority_reasons"] = execution_reasons
        value["ready_for_execution"] = False
        value["ok"] = False
        admission = dict(resources.get("admission") or {})
        admission["status"] = "RESOURCE_ADMISSION_REJECTED"
        resources["admission"] = admission
    return value


def _write_representability_receipt(
    run_root: Path, validation: Mapping[str, Any]
) -> None:
    receipt = validation.get("representability")
    if not isinstance(receipt, Mapping) or receipt.get("status") != "PASS":
        raise QrrLauncherError("QRR experiment representability proof is absent")
    _atomic_json(run_root / "qrr_representability_receipt.json", receipt)


def _semantic_metric_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise QrrLauncherError("QRR runability gate has no semantic metric rows")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise QrrLauncherError("QRR runability gate metric row is malformed")
        value = dict(row)
        value.pop("production_control_readback_sha256", None)
        normalized.append(value)
    return normalized


def _write_runability_technical_gate_receipt_v1(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    provider: Any,
) -> dict[str, Any]:
    """Accept only runtime proofs when the frozen science evaluator is inapplicable."""

    contract = _technical_gate_contract(plan)
    if contract is None:
        raise QrrLauncherError("QRR runability gate contract is absent")
    canary_path = run_root / "technical_gate/canary_manifest.json"
    restore_path = run_root / "technical_gate/fresh_owner_restore_manifest.json"
    canary = _read_mapping(canary_path, name="QRR canary manifest")
    restored = _read_mapping(restore_path, name="QRR fresh-owner manifest")
    validator = getattr(provider, "evaluate_technical_gate", None)
    if not callable(validator):
        raise QrrLauncherError("QRR provider has no technical-gate evaluator")
    provider_evidence = validator(
        plan=plan,
        gate_cell_ids=tuple(contract["cell_ids"]),
        batch_manifests=(canary, restored),
        run_root=run_root,
    )
    if not isinstance(provider_evidence, Mapping):
        raise QrrLauncherError("QRR provider technical-gate evidence is malformed")
    checks = provider_evidence.get("checks")
    if not isinstance(checks, Mapping):
        raise QrrLauncherError("QRR provider technical-gate checks are absent")
    failed_checks = [
        name
        for name in _RUNABILITY_GATE_REQUIRED_PROVIDER_CHECKS
        if checks.get(name) is not True
    ]
    if failed_checks:
        raise QrrLauncherError(
            "QRR runability technical gate failed required checks: "
            + ", ".join(failed_checks)
        )
    raw_reasons = provider_evidence.get("reasons")
    if not isinstance(raw_reasons, list) or any(
        not isinstance(reason, str) for reason in raw_reasons
    ):
        raise QrrLauncherError("QRR provider technical-gate reasons are malformed")
    unclassified = [
        reason
        for reason in raw_reasons
        if not any(
            reason.startswith(prefix)
            for prefix in _RUNABILITY_GATE_EXCLUDED_SCIENCE_REASON_PREFIXES
        )
    ]
    if unclassified:
        raise QrrLauncherError(
            "QRR runability gate has non-science provider failures: "
            + "; ".join(unclassified)
        )
    canary_semantic_rows = _semantic_metric_rows(canary.get("metrics_rows"))
    restored_semantic_rows = _semantic_metric_rows(restored.get("metrics_rows"))
    if canary_semantic_rows != restored_semantic_rows:
        raise QrrLauncherError(
            "QRR fresh-owner semantic telemetry differs from the canary"
        )
    binding = _technical_gate_binding(plan, provider, contract)
    evidence = {
        "schema_version": "eiel_qrr_runability_technical_gate_evidence_v1",
        "passed": True,
        "claim_scope": "runability_only_no_science_or_qualification_claim_v1",
        "required_provider_checks": {
            name: True for name in _RUNABILITY_GATE_REQUIRED_PROVIDER_CHECKS
        },
        "semantic_metric_rows_equal": True,
        "semantic_metric_rows_sha256": canonical_sha256({"rows": canary_semantic_rows}),
        "provider_science_gate_passed": provider_evidence.get("passed") is True,
        "excluded_science_gate_reasons": list(raw_reasons),
        "provider_evidence": dict(provider_evidence),
    }
    receipt = {
        "schema_version": "eiel_campaign_technical_gate_receipt_v2",
        **binding,
        "passed": True,
        "canary_manifest_path": str(canary_path),
        "fresh_owner_restore_manifest_path": str(restore_path),
        "evidence": evidence,
        "completed_at": _now(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(run_root / "technical_gate_receipt.json", receipt)
    return receipt


def _run(
    *,
    release_root: Path,
    experiment_path: Path,
    run_root: Path | None,
    prepared_owner_receipt: Path | None,
    allocation_receipt: Path | None,
) -> dict[str, Any]:
    release, envelope = _release_context(release_root)
    spec, plan, validation, destination = _experiment_spec(
        release_root=release_root,
        envelope=envelope,
        experiment_path=experiment_path,
        run_root=run_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        destination / "qrr_experiment.json",
        _read_mapping(experiment_path, name="QRR experiment"),
    )
    _atomic_json(destination / "materialized_spec.json", spec)
    _atomic_json(destination / "execution_plan.json", plan)
    _write_representability_receipt(destination, validation)
    preflight = _preflight(
        release_root=release_root,
        plan=plan,
        run_root=destination,
        prepared_owner_receipt=prepared_owner_receipt,
        allocation_receipt=allocation_receipt,
    )
    if preflight.get("ready_for_execution") is not True:
        raise QrrLauncherError(
            "release experiment preflight is not execution-ready: "
            + ", ".join(
                str(row.get("code"))
                for row in preflight.get("execution_authority_reasons", ())
                if isinstance(row, Mapping)
            )
        )
    binding = _write_run_binding(
        release=release,
        envelope=envelope,
        validation=validation,
        run_root=destination,
    )
    providers = default_execution_provider_registry()
    try:
        execution = execute_campaign(
            plan,
            run_root=destination,
            provider_registry=providers,
        )
    except CampaignExecutionError as exc:
        if str(exc) != "provider pre-primary technical gate did not pass":
            raise
        provider = providers.get(str(plan.get("resolved_profile")))
        if provider is None:
            raise QrrLauncherError("QRR runability gate provider is absent") from exc
        _write_runability_technical_gate_receipt_v1(
            plan=plan,
            run_root=destination,
            provider=provider,
        )
        execution = execute_campaign(
            plan,
            run_root=destination,
            provider_registry=providers,
        )
    execution_contract = plan.get("execution")
    restore_ticks = (
        tuple(execution_contract.get("deliberate_restore_ticks", ()))
        if isinstance(execution_contract, Mapping)
        and isinstance(execution_contract.get("deliberate_restore_ticks", ()), list)
        else ()
    )
    if (
        not restore_ticks
        and isinstance(execution_contract, Mapping)
        and type(execution_contract.get("deliberate_restore_tick")) is int
    ):
        restore_ticks = (int(execution_contract["deliberate_restore_tick"]),)
    completed_restore_ticks: list[int] = []
    while execution.get("status") == "interrupted":
        if len(completed_restore_ticks) >= len(restore_ticks):
            raise QrrLauncherError(
                "QRR execution interrupted outside its deliberate restore schedule"
            )
        restore_tick = restore_ticks[len(completed_restore_ticks)]
        binding["resume_invocations"] = int(binding.get("resume_invocations", 0)) + 1
        completed_restore_ticks.append(restore_tick)
        binding["deliberate_fresh_owner_restore_tick"] = restore_tick
        binding["deliberate_fresh_owner_restore_ticks"] = list(completed_restore_ticks)
        binding["updated_at_utc"] = _now()
        _atomic_json(destination / "qrr_run_binding.json", binding)
        execution = execute_campaign(
            plan,
            run_root=destination,
            provider_registry=providers,
        )
    run_receipt = _write_run_receipt(
        release=release,
        run_root=destination,
        execution=execution,
        binding=binding,
    )
    return {
        "ok": execution.get("ok") is True,
        "schema_version": "eiel_qrr_launch_result_v1",
        "release_id": release.release_id,
        "run_root": str(destination),
        "run_receipt": run_receipt,
    }


def _preflight_only(
    *,
    release_root: Path,
    experiment_path: Path,
    run_root: Path | None,
    prepared_owner_receipt: Path | None,
    allocation_receipt: Path | None,
) -> dict[str, Any]:
    release, envelope = _release_context(release_root)
    spec, plan, validation, destination = _experiment_spec(
        release_root=release_root,
        envelope=envelope,
        experiment_path=experiment_path,
        run_root=run_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        destination / "qrr_experiment.json",
        _read_mapping(experiment_path, name="QRR experiment"),
    )
    _atomic_json(destination / "materialized_spec.json", spec)
    _atomic_json(destination / "execution_plan.json", plan)
    _write_representability_receipt(destination, validation)
    receipt = _preflight(
        release_root=release_root,
        plan=plan,
        run_root=destination,
        prepared_owner_receipt=prepared_owner_receipt,
        allocation_receipt=allocation_receipt,
    )
    failure_codes = sorted(
        {
            str(row.get("code"))
            for key in ("reasons", "execution_authority_reasons")
            for row in receipt.get(key, ())
            if isinstance(row, Mapping) and row.get("code")
        }
    )
    admission = receipt.get("resources", {}).get("admission", {})
    admission_status = (
        admission.get("status") if isinstance(admission, Mapping) else None
    )
    failure_class = None
    if receipt.get("ready_for_execution") is not True:
        if admission_status in {
            "RESOURCE_ADMISSION_REJECTED",
            "RESOURCE_ADMISSION_UNAVAILABLE",
        }:
            failure_class = "RESOURCE_BLOCKED"
        elif receipt.get("execution_authority_reasons"):
            failure_class = "CAPABILITY_BLOCKED"
        else:
            failure_class = "RUNABILITY_CONTRACT_FAILURE"
    result = {
        "ok": receipt.get("ready_for_execution") is True,
        "schema_version": "eiel_qrr_preflight_result_v1",
        "release_id": release.release_id,
        "run_root": str(destination),
        "failure_class": failure_class,
        "failure_codes": failure_codes,
        "validation": validation,
        "preflight": receipt,
    }
    _atomic_json(destination / "qrr_preflight_result.json", result)
    return result


def _resume(
    *,
    release_root: Path,
    run_root: Path,
    prepared_owner_receipt: Path | None,
    allocation_receipt: Path | None,
) -> dict[str, Any]:
    release, envelope = _release_context(release_root)
    root = run_root.expanduser().resolve()
    plan = _read_mapping(root / "execution_plan.json", name="execution plan")
    binding = _read_mapping(root / "qrr_run_binding.json", name="QRR run binding")
    if binding.get("release_id") != release.release_id:
        raise QrrLauncherError("checkpoint run belongs to another release identity")
    prior_receipt_path = root / "qrr_run_receipt.json"
    if prior_receipt_path.is_file():
        prior_receipt = _read_mapping(prior_receipt_path, name="QRR run receipt")
        if prior_receipt.get("status") == "completed":
            raise QrrLauncherError("completed QRR run cannot be resumed")
    providers = default_execution_provider_registry()
    provider = providers.get(str(plan.get("resolved_profile")))
    if provider is None:
        raise QrrLauncherError("checkpoint profile has no release-local provider")
    validation = envelope.validate_plan(plan, provider=provider)
    if binding.get("bound_request_sha256s") != validation.get("bound_request_sha256s"):
        raise QrrLauncherError("checkpoint request binding differs from release")
    if binding.get("representability_receipt_sha256") != validation.get(
        "representability_receipt_sha256"
    ):
        raise QrrLauncherError(
            "checkpoint representability binding differs from release"
        )
    _write_representability_receipt(root, validation)
    preflight = _preflight(
        release_root=release_root,
        plan=plan,
        run_root=root,
        prepared_owner_receipt=prepared_owner_receipt,
        allocation_receipt=allocation_receipt,
    )
    if preflight.get("ready_for_execution") is not True:
        raise QrrLauncherError("QRR resume preflight is not execution-ready")
    binding["resume_invocations"] = int(binding.get("resume_invocations", 0)) + 1
    binding["updated_at_utc"] = _now()
    _atomic_json(root / "qrr_run_binding.json", binding)
    execution = execute_campaign(plan, run_root=root, provider_registry=providers)
    run_receipt = _write_run_receipt(
        release=release,
        run_root=root,
        execution=execution,
        binding=binding,
    )
    return {
        "ok": execution.get("ok") is True,
        "schema_version": "eiel_qrr_resume_result_v1",
        "release_id": release.release_id,
        "run_root": str(root),
        "run_receipt": run_receipt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qrr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--experiment", type=Path, required=True)
    materialize.add_argument("--run-root", type=Path)
    representability = subparsers.add_parser("check-representability")
    representability.add_argument("--experiment", type=Path, required=True)
    representability.add_argument("--run-root", type=Path)
    representability.add_argument("--receipt", type=Path)
    plan_representability = subparsers.add_parser("check-plan-representability")
    plan_representability.add_argument("--plan", type=Path, required=True)
    plan_representability.add_argument("--receipt", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--experiment", type=Path, required=True)
    run.add_argument("--run-root", type=Path)
    run.add_argument("--prepared-owner-receipt", type=Path)
    run.add_argument("--allocation-receipt", type=Path)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--experiment", type=Path, required=True)
    preflight.add_argument("--run-root", type=Path)
    preflight.add_argument("--prepared-owner-receipt", type=Path)
    preflight.add_argument("--allocation-receipt", type=Path)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-root", type=Path, required=True)
    resume.add_argument("--prepared-owner-receipt", type=Path)
    resume.add_argument("--allocation-receipt", type=Path)
    close = subparsers.add_parser("close")
    close.add_argument("--run-root", type=Path, required=True)
    close.add_argument("--historical-suite-receipt", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    release_root: Path | None = None,
) -> int:
    root = (
        Path(os.environ["EIEL_QRR_ROOT"]).expanduser().resolve()
        if release_root is None and os.environ.get("EIEL_QRR_ROOT")
        else release_root.expanduser().resolve()
        if release_root is not None
        else None
    )
    if root is None:
        raise SystemExit("QRR release root is required")
    args = build_parser().parse_args(argv)
    try:
        if os.environ.get("PYTHONPATH"):
            raise QrrLauncherError("QRR refuses external PYTHONPATH")
        release, envelope = _release_context(root)
        if args.command == "verify":
            result = {
                "ok": True,
                "schema_version": "eiel_qrr_verification_v1",
                "release_id": release.release_id,
                "envelope_sha256": envelope.envelope_sha256,
            }
        elif args.command == "check-plan-representability":
            plan = _read_mapping(args.plan, name="compiled experiment plan")
            receipt = envelope.validate_experiment_representability(plan)
            if args.receipt is not None:
                _atomic_json(args.receipt.expanduser().resolve(), receipt)
            result = {
                "ok": True,
                "schema_version": ("eiel_qrr_experiment_representability_check_v1"),
                "release_id": release.release_id,
                "representability": receipt,
            }
        elif args.command in {"materialize", "check-representability"}:
            materialized_spec, plan, validation, destination = _experiment_spec(
                release_root=root,
                envelope=envelope,
                experiment_path=args.experiment,
                run_root=args.run_root,
            )
            if args.command == "materialize":
                destination.mkdir(parents=True, exist_ok=True)
                _atomic_json(
                    destination / "qrr_experiment.json",
                    _read_mapping(args.experiment, name="QRR experiment"),
                )
                _atomic_json(destination / "materialized_spec.json", materialized_spec)
                _atomic_json(destination / "execution_plan.json", plan)
                _write_representability_receipt(destination, validation)
            result = {
                "ok": True,
                "schema_version": (
                    "eiel_qrr_experiment_representability_check_v1"
                    if args.command == "check-representability"
                    else "eiel_qrr_materialization_v1"
                ),
                "release_id": release.release_id,
                "run_root": str(destination),
                "plan_sha256": plan["plan_sha256"],
                **(
                    {"representability": validation["representability"]}
                    if args.command == "check-representability"
                    else {"validation": validation}
                ),
            }
            if args.command == "check-representability" and args.receipt is not None:
                _atomic_json(
                    args.receipt.expanduser().resolve(),
                    validation["representability"],
                )
        elif args.command == "preflight":
            result = _preflight_only(
                release_root=root,
                experiment_path=args.experiment,
                run_root=args.run_root,
                prepared_owner_receipt=args.prepared_owner_receipt,
                allocation_receipt=args.allocation_receipt,
            )
        elif args.command == "run":
            # The QRR command is a machine interface: reserve stdout for the
            # single JSON result even when runtime diagnostics use print().
            with redirect_stdout(sys.stderr):
                result = _run(
                    release_root=root,
                    experiment_path=args.experiment,
                    run_root=args.run_root,
                    prepared_owner_receipt=args.prepared_owner_receipt,
                    allocation_receipt=args.allocation_receipt,
                )
        elif args.command == "resume":
            with redirect_stdout(sys.stderr):
                result = _resume(
                    release_root=root,
                    run_root=args.run_root,
                    prepared_owner_receipt=args.prepared_owner_receipt,
                    allocation_receipt=args.allocation_receipt,
                )
        else:
            historical = _read_mapping(
                args.historical_suite_receipt,
                name="historical escape-suite receipt",
            )
            receipt = assess_runability_closure_v1(
                release_root=root,
                run_root=args.run_root,
                historical_suite_receipt=historical,
            )
            _atomic_json(
                args.run_root.expanduser().resolve()
                / "runability_closure_receipt.json",
                receipt.payload(),
            )
            result = receipt.payload()
        assert_release_module_origins_v1(root)
    except Exception as exc:
        result = {
            "ok": False,
            "schema_version": "eiel_qrr_failure_v1",
            "failure_class": classify_runability_failure_v1(exc, stage="launch"),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        receipt = getattr(exc, "receipt", None)
        if isinstance(receipt, Mapping):
            result["representability"] = dict(receipt)
            receipt_path = getattr(args, "receipt", None)
            if isinstance(receipt_path, Path):
                _atomic_json(receipt_path.expanduser().resolve(), receipt)
    print(pretty_json(result), end="")
    return (
        0
        if result.get("ok") is True or result.get("decision") == "RUNABILITY_CLOSED"
        else 1
    )


__all__ = ["QrrLauncherError", "assert_release_module_origins_v1", "main"]
