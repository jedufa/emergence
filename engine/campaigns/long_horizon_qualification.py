"""Evidence-bound qualification dimensions for long tick-native routes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_sha256

BOUNDED_ROUTE_QUALIFIED = "BOUNDED_ROUTE_QUALIFIED"
LONG_HORIZON_RUNABILITY_QUALIFIED = "LONG_HORIZON_RUNABILITY_QUALIFIED"
SCIENCE_ROUTE_QUALIFIED = "SCIENCE_ROUTE_QUALIFIED"
CURRENT_B136_LONG_HORIZON_MINIMUM_TICKS = 5_000
LONG_HORIZON_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_long_horizon_qualification_evidence_v1"
)
QUALIFICATION_EVIDENCE_BINDING_SCHEMA_VERSION: Final[str] = (
    "eiel_qualification_evidence_binding_v1"
)
SCIENCE_ROUTE_ATTESTATION_SCHEMA_VERSION: Final[str] = (
    "eiel_canonical_full_production_route_attestation_v1"
)
SCIENCE_ROUTE_EVIDENCE_REF_ROLES_V1: Final[tuple[str, ...]] = (
    "gate26_evidence",
    "gate26_uninterrupted_plan",
    "gate26_restored_plan",
    "gate26_uninterrupted_passport",
    "gate26_restored_passport",
    "gate26_uninterrupted_cell_metrics",
    "gate26_restored_cell_metrics",
    "promoted_route_receipt",
    "gate24_resource_decision",
    "gate25_impact_map",
    "gate27_impact_map",
)
SCIENCE_ROUTE_REQUIRED_OBLIGATIONS_V1: Final[frozenset[str]] = frozenset(
    {
        "acs_capture_barrier_measurement",
        "alternate_batch_partition",
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
        "checkpoint_export",
        "cold_start",
        "disk_checkpoint_reload",
        "durable_frontier_publication",
        "exact_absolute_tick_checkpoint_export",
        "first_committed_tick_after_partial_restore",
        "first_post_boundary_tick",
        "first_post_restore_refill",
        "first_post_restore_tick",
        "first_tick",
        "fresh_owner_partial_tick_restore",
        "fresh_owner_restore",
        "multicell_execution",
        "non_lattice_divisible_checkpoint_tick",
        "non_lattice_divisible_terminal_tick",
        "odd_physical_tail",
        "ordinary_birth",
        "reporting_parity",
        "second_post_boundary_tick",
        "terminal_settlement",
    }
)
SCIENCE_ROUTE_GATE24_REQUIRED_CRITERIA_V1: Final[tuple[str, ...]] = (
    "exact physical B136 W32/P256/C512 and 5000 absolute ticks",
    "exact event capacity and bytes",
    "all four expected in-place owner transitions with no overlap",
    "two-slot detached ACS high-water exactly two with zero queued CUDA references",
    "capture/release and successor work within ten seconds",
    "minimum VRAM, host RSS, pinned memory and temporary-file bounds",
    "no CPU staging/refill or future RNG/event tapes",
    "GPU duty, SM utilization, dispatch overhead and idle-gap thresholds",
    "terminal GPU release before separately measured ACS tail",
    "mechanical lifecycle, passport, plan and storage receipt bindings",
    "absolute-tick public runtime and checkpoint authority with compatibility metadata removed",
)
SCIENCE_ROUTE_GATE24_REQUIRED_MEASUREMENTS_V1: Final[frozenset[str]] = frozenset(
    {
        "event_rows_per_cell",
        "event_row_bytes",
        "total_event_bytes",
        "minimum_free_vram_bytes",
        "minimum_free_vram_threshold_bytes",
        "capture_barrier_tick2500_seconds",
        "capture_barrier_tick5000_seconds",
        "capture_release_threshold_seconds",
        "successor_gpu_dispatch_delay_tick2500_seconds",
        "successor_dispatch_threshold_seconds",
        "active_gpu_duty_percent",
        "active_gpu_duty_threshold_percent",
        "mean_sm_percent",
        "mean_sm_threshold_percent",
        "dispatch_overhead_fraction",
        "dispatch_overhead_threshold_fraction",
        "longest_unexplained_idle_interval_seconds",
        "idle_interval_threshold_seconds",
        "queue_occupancy_high_water",
        "queued_cuda_references",
        "queued_cuda_callbacks",
        "temporary_capture_file_high_water",
        "temporary_capture_file_terminal_count",
        "host_pinned_bytes",
        "rss_high_water_bytes",
        "native_host_refill_calls",
        "intermediate_host_to_device_copies",
        "intermediate_device_to_host_copies",
        "t_sim_seconds",
        "t_detach_seconds",
        "t_persist_seconds",
        "t_acs_tail_seconds",
        "t_restore_seconds",
        "terminal_gpu_device_reset_return_code",
        "post_release_target_gpu_resident_samples",
        "post_release_target_max_framebuffer_bytes",
        "wall_reconciliation_error_fraction",
    }
)
SCIENCE_ROUTE_PARAMETER_CONTRACT_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "activation_contract",
        "control_role",
        "name",
        "per_cell",
        "qualification_affecting",
        "type",
        "value_contract",
        "variability_route_scopes",
        "variability_semantics",
    }
)
CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE: Final[str] = "canonical_full_production_route_v1"
LONG_HORIZON_RUNABILITY_ROUTE_SCOPE: Final[str] = "long_horizon_runability_route_v1"

LONG_HORIZON_INVALIDATING_SURFACES_V1 = frozenset(
    {
        "owner_lifetime",
        "dispatch_window_planning",
        "tick_schedule",
        "event_buffering",
        "event_drain",
        "event_history_retention",
        "source_allocation",
        "source_ownership",
        "resident_resource_estimator",
        "checkpoint_settlement_ownership",
        "restore_frontier",
        "runtime_control_cursor",
        "host_gpu_staging_architecture",
        "memory_allocator_strategy",
    }
)

REQUIRED_LONG_HORIZON_MEASUREMENTS_V1 = (
    "ticks_requested",
    "ticks_committed",
    "owner_boundaries_crossed",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "minimum_free_gpu_bytes",
    "gpu_memory_slope_bytes_per_window",
    "gpu_allocated_memory_slope_bytes_per_window",
    "host_peak_rss_bytes",
    "host_memory_slope_bytes_per_window",
    "resident_event_bytes",
    "resident_event_bytes_slope_per_window",
    "resident_source_bytes",
    "resident_source_bytes_slope_per_window",
    "event_ring_capacity",
    "event_ring_peak_occupancy",
    "event_ring_occupancy_slope_per_window",
    "events_emitted",
    "events_drained",
    "terminal_event_ring_occupancy",
    "external_event_history_count",
    "external_event_history_digest",
    "checkpoint_frontier",
    "checkpoint_restore",
    "terminal_digest",
    "event_digest",
    "windows",
    "errors",
    "ooms",
)

_WINDOW_FIELDS_V1 = (
    "start_tick",
    "end_tick",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "minimum_free_gpu_bytes",
    "host_peak_rss_bytes",
    "host_terminal_rss_bytes",
    "resident_event_bytes",
    "resident_source_bytes",
    "event_ring_peak_occupancy",
    "events_emitted",
    "events_drained",
)
_DIGEST_FIELDS_V1 = (
    "checkpoint_digest",
    "restored_checkpoint_digest",
    "fresh_owner_uninterrupted_terminal_digest",
    "fresh_owner_restored_terminal_digest",
    "fresh_owner_uninterrupted_event_digest",
    "fresh_owner_restored_event_digest",
)


class LongHorizonQualificationError(ValueError):
    """Long-horizon evidence is absent, inconsistent, or insufficient."""


def validate_canonical_full_production_route_attestation_v1(
    value: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_evidence_sha256: str,
    expected_evidence_binding_sha256: str,
    expected_source_route_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the independent Gate27 authority attached to a route receipt.

    Long-horizon evidence deliberately remains scoped to
    ``LONG_HORIZON_RUNABILITY_ROUTE_SCOPE``.  This attestation is the separate
    authority that can extend the claim to the canonical production route; a
    scope string in the evidence alone can never do so.
    """

    attestation = _mapping(value, name="canonical_full_production_route_attestation")
    if attestation.get("schema_version") != SCIENCE_ROUTE_ATTESTATION_SCHEMA_VERSION:
        raise LongHorizonQualificationError(
            "canonical full production route attestation schema is unsupported"
        )
    if attestation.get("route_scope") != CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE:
        raise LongHorizonQualificationError(
            "canonical full production route attestation scope is invalid"
        )
    if attestation.get("claim_dimension") != SCIENCE_ROUTE_QUALIFIED:
        raise LongHorizonQualificationError(
            "canonical full production route attestation claim is invalid"
        )
    _sha256(
        attestation.get("evidence_sha256"),
        name="canonical_full_production_route_attestation.evidence_sha256",
    )
    _sha256(
        attestation.get("evidence_binding_sha256"),
        name="canonical_full_production_route_attestation.evidence_binding_sha256",
    )
    _sha256(
        attestation.get("source_route_receipt_sha256"),
        name="canonical_full_production_route_attestation.source_route_receipt_sha256",
    )
    if attestation.get("evidence_sha256") != expected_evidence_sha256:
        raise LongHorizonQualificationError(
            "canonical route attestation differs from long-horizon evidence"
        )
    if attestation.get("evidence_binding_sha256") != expected_evidence_binding_sha256:
        raise LongHorizonQualificationError(
            "canonical route attestation differs from long-horizon evidence binding"
        )
    if (
        expected_source_route_receipt_sha256 is not None
        and attestation.get("source_route_receipt_sha256")
        != expected_source_route_receipt_sha256
    ):
        raise LongHorizonQualificationError(
            "canonical route attestation differs from the source route receipt"
        )
    for name, expected in expected_identity.items():
        if name == "geometry":
            actual = attestation.get(name)
            if actual != expected:
                raise LongHorizonQualificationError(
                    f"canonical route attestation differs at {name}"
                )
        elif name in {
            "execution_route_sha256",
            "source_sha256",
            "plan_sha256",
            "run_receipt_sha256",
        }:
            if attestation.get(name) != expected:
                raise LongHorizonQualificationError(
                    f"canonical route attestation differs at {name}"
                )
        elif name in {"provider_id", "profile_id", "run_id"}:
            if attestation.get(name) != expected:
                raise LongHorizonQualificationError(
                    f"canonical route attestation differs at {name}"
                )
        elif name == "artifact_sha256s" and attestation.get(name) != expected:
            raise LongHorizonQualificationError(
                "canonical route attestation differs at artifact_sha256s"
            )
    for name in (
        "canonical_route_pass",
        "production_resource_safety_pass",
        "terminal_finalization_pass",
        "science_output_pass",
    ):
        if attestation.get(name) is not True:
            raise LongHorizonQualificationError(
                f"canonical route attestation lacks {name}"
            )
    raw_refs = attestation.get("evidence_refs")
    if (
        isinstance(raw_refs, (str, bytes))
        or not isinstance(raw_refs, Sequence)
        or not raw_refs
    ):
        raise LongHorizonQualificationError(
            "canonical route attestation requires structured evidence_refs"
        )
    ref_result = _read_science_route_evidence_refs_v1(
        raw_refs,
        expected_identity=expected_identity,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_evidence_binding_sha256=expected_evidence_binding_sha256,
        expected_source_route_receipt_sha256=(
            attestation.get("source_route_receipt_sha256")
        ),
    )
    if (
        attestation.get("canonical_route_pass")
        != ref_result["canonical_route_pass"]
        or attestation.get("production_resource_safety_pass")
        != ref_result["production_resource_safety_pass"]
        or attestation.get("terminal_finalization_pass")
        != ref_result["terminal_finalization_pass"]
        or attestation.get("science_output_pass")
        != ref_result["science_output_pass"]
    ):
        raise LongHorizonQualificationError(
            "canonical route attestation pass facts differ from evidence refs"
        )
    if ref_result["evidence_refs"] != list(raw_refs):
        raise LongHorizonQualificationError(
            "canonical route attestation evidence refs are not canonical"
        )
    body = {
        name: item for name, item in attestation.items() if name != "attestation_sha256"
    }
    attestation_sha256 = _sha256(
        attestation.get("attestation_sha256"),
        name="canonical_full_production_route_attestation.attestation_sha256",
    )
    if attestation_sha256 != canonical_sha256(body):
        raise LongHorizonQualificationError(
            "canonical full production route attestation SHA-256 differs"
        )
    return {**body, "attestation_sha256": attestation_sha256}


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LongHorizonQualificationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongHorizonQualificationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LongHorizonQualificationError(f"{name} must be a finite number")
    return result


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LongHorizonQualificationError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongHorizonQualificationError(f"{name} must be nonempty text")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LongHorizonQualificationError(f"{name} must be an object")
    return dict(value)


def _validate_canonical_wrc_parameter_contracts_v1(
    value: Any, *, name: str
) -> list[dict[str, Any]]:
    """Require the exact canonical shape used by the frozen Gate26 WRC."""

    if not isinstance(value, list) or not value:
        raise LongHorizonQualificationError(
            f"{name} must contain non-empty canonical parameter_contracts"
        )
    canonical = sorted(
        value,
        key=lambda row: str(row.get("name")) if isinstance(row, Mapping) else "",
    )
    if value != canonical:
        raise LongHorizonQualificationError(
            f"{name} parameter_contracts are not canonically ordered"
        )
    names: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, name=f"{name}.parameter_contracts[{index}]")
        if set(row) != SCIENCE_ROUTE_PARAMETER_CONTRACT_FIELDS_V1:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}] has a non-canonical field set"
            )
        parameter_name = row.get("name")
        if not isinstance(parameter_name, str) or not parameter_name:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}].name is invalid"
            )
        if parameter_name in names:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts contains duplicate names"
            )
        names.add(parameter_name)
        if row.get("type") not in {"bool", "int", "float"}:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}].type is invalid"
            )
        if type(row.get("per_cell")) is not bool or type(
            row.get("qualification_affecting")
        ) is not bool:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}] boolean flags are invalid"
            )
        if not isinstance(row.get("control_role"), str) or not row["control_role"]:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}].control_role is invalid"
            )
        if not isinstance(row.get("variability_semantics"), str) or not row[
            "variability_semantics"
        ]:
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}].variability_semantics is invalid"
            )
        scopes = row.get("variability_route_scopes")
        if not isinstance(scopes, list) or not all(
            isinstance(scope, str) and bool(scope) for scope in scopes
        ):
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}].variability_route_scopes is invalid"
            )
        if not isinstance(row.get("activation_contract"), Mapping) or not isinstance(
            row.get("value_contract"), Mapping
        ):
            raise LongHorizonQualificationError(
                f"{name}.parameter_contracts[{index}] contracts are invalid"
            )
        result.append(row)
    return result


def _canonical_production_wrc_envelope_v1(
    parameter_contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize the current production-profile WRC envelope for comparison."""

    from .registry import EIEL_PRODUCTION_PROFILE_ID, default_profile_registry
    from .wrc_variability import (
        WRCVariabilityEnvelopeError,
        build_wrc_parameter_variability_envelope_v1,
    )

    profile = next(
        (
            candidate
            for candidate in default_profile_registry().profiles()
            if candidate.profile_id == EIEL_PRODUCTION_PROFILE_ID
        ),
        None,
    )
    if profile is None:
        raise LongHorizonQualificationError(
            "canonical production profile is unavailable for WRC validation"
        )
    try:
        envelope = build_wrc_parameter_variability_envelope_v1(
            profile,
            world_base="patch",
            world_overlays=(
                "eed_resource_impact",
                "jungle",
                "wdl_delayed_hazard",
            ),
            requested_parameter_names={str(row["name"]) for row in parameter_contracts},
        )
    except WRCVariabilityEnvelopeError as exc:
        raise LongHorizonQualificationError(
            f"canonical production WRC envelope cannot be built: {exc}"
        ) from exc
    if envelope is None:
        raise LongHorizonQualificationError(
            "canonical production WRC envelope is absent"
        )
    return envelope.payload()


def _validate_canonical_production_wrc_envelope_v1(
    value: Mapping[str, Any],
    *,
    parameter_contracts: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    expected = _canonical_production_wrc_envelope_v1(parameter_contracts)
    if dict(value) != expected:
        raise LongHorizonQualificationError(
            f"{name} does not bind the exact current production WRC envelope"
        )


def _read_science_route_evidence_refs_v1(
    evidence_refs: Sequence[Mapping[str, Any]],
    *,
    expected_identity: Mapping[str, Any],
    expected_evidence_sha256: str,
    expected_evidence_binding_sha256: str,
    expected_source_route_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and derive Gate27 authority from its exact compact input files."""

    if isinstance(evidence_refs, (str, bytes)) or not isinstance(
        evidence_refs, Sequence
    ):
        raise LongHorizonQualificationError("science-route evidence_refs must be an array")
    expected_roles = set(SCIENCE_ROUTE_EVIDENCE_REF_ROLES_V1)
    by_role: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(evidence_refs):
        ref = _mapping(raw, name=f"evidence_refs[{index}]")
        role = _text(ref.get("role"), name=f"evidence_refs[{index}].role")
        if role not in expected_roles or role in by_role:
            raise LongHorizonQualificationError(
                f"science-route evidence_refs has unsupported or duplicate role: {role}"
            )
        raw_path = _text(ref.get("path"), name=f"evidence_refs[{index}].path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise LongHorizonQualificationError(
                f"science-route evidence ref is not a regular non-symlink file: {raw_path}"
            )
        declared_sha256 = _sha256(
            ref.get("file_sha256"), name=f"evidence_refs[{index}].file_sha256"
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise LongHorizonQualificationError(
                f"science-route evidence ref cannot be read: {raw_path}"
            ) from exc
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != declared_sha256:
            raise LongHorizonQualificationError(
                f"science-route evidence ref hash differs for {role}"
            )
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LongHorizonQualificationError(
                f"science-route evidence ref is not JSON: {role}"
            ) from exc
        if not isinstance(document, Mapping):
            raise LongHorizonQualificationError(
                f"science-route evidence ref must contain an object: {role}"
            )
        normalized = {
            "role": role,
            "path": str(path),
            "file_sha256": declared_sha256,
        }
        source_sha256 = ref.get("source_route_receipt_sha256")
        if source_sha256 is not None:
            normalized["source_route_receipt_sha256"] = _sha256(
                source_sha256,
                name=f"evidence_refs[{index}].source_route_receipt_sha256",
            )
        by_role[role] = normalized
        documents[role] = dict(document)
    if set(by_role) != expected_roles:
        raise LongHorizonQualificationError(
            "science-route evidence_refs must cover the complete Gate27 input set"
        )

    evidence = documents["gate26_evidence"]
    validated_evidence = validate_long_horizon_evidence_v1(
        evidence,
        expected_identity=expected_identity,
    )
    if evidence.get("evidence_sha256") != expected_evidence_sha256:
        raise LongHorizonQualificationError(
            "Gate26 evidence ref differs from the supplied long-horizon evidence"
        )
    if validated_evidence["evidence_binding"]["binding_sha256"] != (
        expected_evidence_binding_sha256
    ):
        raise LongHorizonQualificationError(
            "Gate26 evidence ref binding differs from the supplied binding"
        )
    paired = evidence.get("paired_run_bindings")
    if not isinstance(paired, Mapping) or evidence.get("paired_run_bindings_sha256") != (
        canonical_sha256(dict(paired))
    ):
        raise LongHorizonQualificationError(
            "Gate26 evidence ref lacks its exact paired-run binding"
        )

    expected_capabilities = {
        "patch": ("world.patch.v1", "patch_world_mechanics_v1"),
        "eed": (
            "world.overlay.eed_resource_impact.v1",
            "eed_resource_impact_world_mechanics_v1",
        ),
        "jungle": ("world.overlay.jungle.v1", "jungle_world_mechanics_v1"),
        "wdl_delayed_hazard": (
            "world.overlay.wdl_delayed_hazard.v1",
            "wdl_delayed_hazard_world_mechanics_v1",
        ),
        "genome_v2_ordinary_birth": ("genome.genome_upgrade_v2.v1", None),
        "genome_v2_mutation": ("genome.genome_upgrade_v2.v1", None),
        "recombination_profile": ("genome.genome_upgrade_v2.v1", None),
    }
    expected_required_capabilities = set(expected_capabilities)

    def validate_plan(
        role: str, *, expected_sha256: str, restore: bool
    ) -> dict[str, Any]:
        plan = documents[role]
        unsigned = dict(plan)
        plan_sha256 = _sha256(plan.get("plan_sha256"), name=f"{role}.plan_sha256")
        unsigned.pop("plan_sha256", None)
        if (
            plan.get("schema_version") != "eiel_campaign_plan_v1"
            or plan.get("resolved_profile") != "eiel-production-runtime-v1"
            or plan_sha256 != canonical_sha256(unsigned)
            or plan_sha256 != expected_sha256
        ):
            raise LongHorizonQualificationError(
                f"{role} is not the exact self-hashed Gate26 plan"
            )
        execution = _mapping(plan.get("execution"), name=f"{role}.execution")
        cells = plan.get("cells")
        physical_batches = plan.get("physical_batches")
        if not isinstance(cells, list) or not isinstance(physical_batches, list):
            raise LongHorizonQualificationError(
                f"{role} does not contain canonical cells and physical batches"
            )
        if (
            execution.get("execution_route_identity") != "tick_native_execution_route_v1"
            or execution.get("time_authority") != "absolute_tick_v1"
            or execution.get("total_ticks") != 5000
            or execution.get("checkpoint_ticks") != [2500, 5000]
            or len(cells) != 136
            or len(physical_batches) != 1
            or execution.get("acs_checkpoint_slots") != 2
            or execution.get("acs_capture_instrumentation") is not True
            or execution.get("native_streamed_checkpoint") is not True
            or execution.get("checkpoint_storage_mode") != "disk"
            or execution.get("tick_schedule")
            != {
                "schema_version": "eiel-absolute-tick-schedule-v2",
                "science_sample_interval_ticks": None,
                "total_ticks": 5000,
            }
        ):
            raise LongHorizonQualificationError(
                f"{role} does not describe the exact physical-B136 Gate26 route"
            )
        control = _mapping(
            execution.get("tick_native_control"), name=f"{role}.tick_native_control"
        )
        if (
            control.get("schema_version") != "eiel-tick-native-control-v2"
            or control.get("time_authority") != "absolute_tick_v1"
            or control.get("total_ticks") != 5000
            or control.get("checkpoint_ticks") != [2500, 5000]
            or control.get("source_window_ticks") != 160
            or control.get("native_dispatch_window_ticks") != 1120
            or control.get("resident_owner_span_ticks") != 1120
            or control.get("event_window_capacity") != 41804
            or control.get("schedules")
            != {
                "archive_ticks": [5000],
                "checkpoint_ticks": [2500, 5000],
                "reducer_ticks": [5000],
                "report_ticks": [5000],
                "science_sample_ticks": [5000],
            }
        ):
            raise LongHorizonQualificationError(
                f"{role} does not bind the exact absolute-tick schedules"
            )
        deliberate_ticks = execution.get("deliberate_restore_ticks", [])
        if restore:
            if execution.get("deliberate_restore_tick") != 2500 or deliberate_ticks != [2500]:
                raise LongHorizonQualificationError(
                    "Gate26 restored plan does not bind the exact restore tick"
                )
        elif execution.get("deliberate_restore_tick") is not None or deliberate_ticks:
            raise LongHorizonQualificationError(
                "Gate26 uninterrupted plan unexpectedly contains restore authority"
            )
        qualification_key = _mapping(
            plan.get("qualification_key"), name=f"{role}.qualification_key"
        )
        plan_qualification_sha256 = _sha256(
            plan.get("qualification_key_sha256"),
            name=f"{role}.qualification_key_sha256",
        )
        if (
            plan_qualification_sha256 != canonical_sha256(qualification_key)
            or qualification_key.get("schema_version")
            != "eiel_runtime_qualification_key_v1"
            or qualification_key.get("profile_id") != "eiel-production-runtime-v1"
            or qualification_key.get("profile_version") != 2
            or qualification_key.get("backend") != "cuda_resident"
        ):
            raise LongHorizonQualificationError(
                f"{role} does not bind Production Baseline V2"
            )
        fixed = _mapping(
            qualification_key.get("fixed_invariants"),
            name=f"{role}.qualification_key.fixed_invariants",
        )
        required_fixed = {
            "checkpoint_schema": "authoritative_state_schema_bound",
            "device_owner": "cuda_resident_no_fallback",
            "parameter.cuda_architecture_b.cpu_request_staging_enable": False,
            "parameter.cuda_architecture_b.cpython_gauss_segment_capacity": 0,
            "parameter.cuda_architecture_b.policy_birth_segment_refill_enable": False,
            "parameter.cuda_architecture_b.turnover_mlp_shadow_elide_enable": False,
            "parameter.eed.ecology_frozen": False,
            "parameter.eed.phenotype_fixed": False,
            "parameter.eed.resource_impact_apply_enable": True,
            "parameter.genome_upgrade_v2.enable": True,
            "parameter.genome_upgrade_v2.mutation_rate_q10": 20,
            "parameter.genome_upgrade_v2.mutation_step_q10": 64,
            "parameter.genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            "parameter.patch.capacity": 512,
            "parameter.patch.grid_size": 32,
            "parameter.patch.initial_population": 256,
            "parameter.production_lifecycle.life_history_profile": "stage1s_overlap_tick_life_history_v1",
            "parameter.production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
            "parameter.production_lifecycle.ordinary_births_enabled": True,
            "parameter.production_lifecycle.turnover_cull_fraction": 0.0,
            "parameter.production_lifecycle.turnover_enabled": True,
            "parameter.production_lifecycle.turnover_parent_fraction": 1.0,
            "runtime.policy_family": "fixed_mlp",
            "runtime.selection_mode": "genotype_blind_neutral",
            "runtime.world_base": "patch",
            "runtime.world_overlays": '["eed_resource_impact","jungle","wdl_delayed_hazard"]',
            "runtime_identity_contract": "campaign_requirements_bound_runtime_identity_v1",
            "semantic_registry": "eiel_global_genome_registry_v1",
        }
        if any(fixed.get(name) != value for name, value in required_fixed.items()):
            raise LongHorizonQualificationError(
                f"{role} has a non-canonical fixed invariant"
            )
        envelope = _mapping(
            qualification_key.get("variability_envelope"),
            name=f"{role}.qualification_key.variability_envelope",
        )
        envelope_sha256 = _sha256(
            envelope.get("envelope_sha256"),
            name=f"{role}.variability_envelope.envelope_sha256",
        )
        parameter_contracts = _validate_canonical_wrc_parameter_contracts_v1(
            envelope.get("parameter_contracts"),
            name=f"{role}.variability_envelope",
        )
        _validate_canonical_production_wrc_envelope_v1(
            envelope,
            parameter_contracts=parameter_contracts,
            name=f"{role}.variability_envelope",
        )
        if (
            envelope.get("schema_version")
            != "eiel_wrc_parameter_variability_envelope_v1"
            or envelope.get("backend") != "cuda_resident"
            or envelope.get("profile_id") != "eiel-production-runtime-v1"
            or envelope.get("envelope_id")
            != "wrc-eiel-production-runtime-v1-eed-jungle-patch-wdl-v1"
            or envelope_sha256 != canonical_sha256(
                {k: v for k, v in envelope.items() if k != "envelope_sha256"}
            )
            or fixed.get("wrc.parameter_variability_envelope_sha256")
            != envelope_sha256
        ):
            raise LongHorizonQualificationError(
                f"{role} does not bind the exact WRC envelope"
            )
        run_contract = _mapping(plan.get("run_contract"), name=f"{role}.run_contract")
        if (
            run_contract.get("schema_version") != "eiel_resolved_run_contract_v1"
            or run_contract.get("requested")
            != {"mode": "strict", "profile_id": "eiel_full_evolution_v1"}
            or set(run_contract.get("required_capabilities", []))
            != expected_required_capabilities
        ):
            raise LongHorizonQualificationError(
                f"{role} does not bind strict EIEL_FULL_EVOLUTION_V1"
            )
        capabilities = run_contract.get("capabilities")
        if not isinstance(capabilities, Mapping) or set(capabilities) != expected_required_capabilities:
            raise LongHorizonQualificationError(
                f"{role} capability set is not the canonical production route"
            )
        for capability, (component_id, contract_version) in expected_capabilities.items():
            detail = _mapping(
                capabilities.get(capability), name=f"{role}.capabilities.{capability}"
            )
            evidence_detail = _mapping(
                detail.get("evidence"),
                name=f"{role}.capabilities.{capability}.evidence",
            )
            if (
                detail.get("configured") is not True
                or detail.get("initializable") is not True
                or detail.get("supported") is not True
                or evidence_detail.get("component_id") != component_id
                or (
                    contract_version is not None
                    and evidence_detail.get("contract_version") != contract_version
                )
            ):
                raise LongHorizonQualificationError(
                    f"{role} capability {capability} is not admitted"
                )
        artifacts = _mapping(
            qualification_key.get("artifact_hashes"),
            name=f"{role}.qualification_key.artifact_hashes",
        )
        if set(artifacts) != set(expected_identity.get("artifact_sha256s", {})):
            raise LongHorizonQualificationError(
                f"{role} artifact map does not match Gate26 evidence"
            )
        if dict(artifacts) != dict(expected_identity.get("artifact_sha256s", {})):
            raise LongHorizonQualificationError(
                f"{role} artifact identities differ from Gate26 evidence"
            )
        return {
            "plan": plan,
            "plan_sha256": plan_sha256,
            "qualification_key_sha256": plan_qualification_sha256,
            "qualification_key": qualification_key,
            "runtime_root": Path(
                _text(
                    _mapping(plan.get("output"), name=f"{role}.output").get(
                        "runtime_root"
                    ),
                    name=f"{role}.output.runtime_root",
                )
            ).expanduser(),
            "cells": cells,
        }

    uninterrupted_binding = paired.get("uninterrupted")
    restored_binding = paired.get("restored")
    if not isinstance(uninterrupted_binding, Mapping) or not isinstance(
        restored_binding, Mapping
    ):
        raise LongHorizonQualificationError("Gate26 paired-run bindings are incomplete")
    uninterrupted_plan = validate_plan(
        "gate26_uninterrupted_plan",
        expected_sha256=uninterrupted_binding.get("plan_sha256"),
        restore=False,
    )
    restored_plan = validate_plan(
        "gate26_restored_plan",
        expected_sha256=restored_binding.get("plan_sha256"),
        restore=True,
    )
    if (
        uninterrupted_plan["qualification_key_sha256"]
        != restored_plan["qualification_key_sha256"]
        or uninterrupted_plan["qualification_key"]
        != restored_plan["qualification_key"]
        or uninterrupted_plan["cells"]
        != restored_plan["cells"]
        or uninterrupted_binding.get("run_root")
        != str(uninterrupted_plan["runtime_root"])
        or restored_binding.get("run_root") != str(restored_plan["runtime_root"])
    ):
        raise LongHorizonQualificationError(
            "Gate26 twin plans do not share the exact production identity"
        )

    def validate_passport(
        role: str,
        *,
        expected_plan: Mapping[str, Any],
        expected_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        passport = documents[role]
        unsigned = dict(passport)
        passport_sha256 = _sha256(
            passport.get("passport_sha256"), name=f"{role}.passport_sha256"
        )
        unsigned.pop("passport_sha256", None)
        runtime_root = expected_plan["runtime_root"]
        passport_ref_path = Path(by_role[role]["path"])
        if (
            passport_sha256 != canonical_sha256(unsigned)
            or passport_sha256 != expected_binding.get("run_passport_sha256")
            or passport.get("terminal_status") != "completed"
            or passport.get("plan_sha256") != expected_plan["plan_sha256"]
            or passport.get("qualification_key_sha256")
            != expected_plan["qualification_key_sha256"]
            or passport.get("lifecycle_contract_sha256")
            != expected_plan["plan"].get("artifact_lifecycle", {}).get("contract_sha256")
            or passport.get("run_id") != runtime_root.name
            or passport_ref_path != runtime_root / "run_passport.json"
        ):
            raise LongHorizonQualificationError(f"{role} self-hash differs")
        provider = _mapping(passport.get("provider_identity"), name=f"{role}.provider_identity")
        if (
            provider.get("execution_route_sha256")
            != expected_identity.get("execution_route_sha256")
            or provider.get("provider_id") != expected_identity.get("provider_id")
            or provider.get("profile_id") != expected_identity.get("profile_id")
        ):
            raise LongHorizonQualificationError(
                f"{role} provider identity differs from Gate26 evidence"
            )
        source_identity = _mapping(
            passport.get("source_identity"), name=f"{role}.source_identity"
        )
        if source_identity.get("artifact_hashes") != expected_plan["qualification_key"].get(
            "artifact_hashes"
        ):
            raise LongHorizonQualificationError(
                f"{role} artifact identity differs from the frozen plan"
            )
        return passport

    validate_passport(
        "gate26_uninterrupted_passport",
        expected_plan=uninterrupted_plan,
        expected_binding=uninterrupted_binding,
    )
    validate_passport(
        "gate26_restored_passport",
        expected_plan=restored_plan,
        expected_binding=restored_binding,
    )

    def validate_cell_metrics(
        role: str, *, expected_plan: Mapping[str, Any]
    ) -> None:
        metrics = documents[role]
        rows = metrics.get("rows")
        expected_cell_ids = {
            cell.get("cell_id")
            for cell in expected_plan["cells"]
            if isinstance(cell, Mapping)
        }
        if (
            metrics.get("schema_version") != "eiel_campaign_cell_metrics_v1"
            or metrics.get("campaign_id") != expected_identity.get("run_id")
            or metrics.get("plan_sha256") != expected_plan["plan_sha256"]
            or not isinstance(rows, list)
            or len(rows) != 136
            or {row.get("cell_id") for row in rows if isinstance(row, Mapping)}
            != expected_cell_ids
            or any(
                not isinstance(row, Mapping)
                or not isinstance(row.get("metrics"), Mapping)
                or row["metrics"].get("runtime.terminal_absolute_tick") != 5000
                for row in rows
            )
            or Path(by_role[role]["path"])
            != expected_plan["runtime_root"] / "cell_metrics.json"
        ):
            raise LongHorizonQualificationError(
                f"{role} does not prove 136 completed Gate26 terminal metric rows"
            )

    validate_cell_metrics(
        "gate26_uninterrupted_cell_metrics", expected_plan=uninterrupted_plan
    )
    validate_cell_metrics("gate26_restored_cell_metrics", expected_plan=restored_plan)

    route = documents["promoted_route_receipt"]
    route_cases = route.get("case_ids")
    reporting = route.get("reporting_results")
    covered_obligations = route.get("covered_obligations")
    route_contract = _mapping(
        route.get("route_contract"), name="promoted_route_receipt.route_contract"
    )
    route_contract_obligations = route_contract.get("required_obligations")
    expected_route_cases = {
        "disk-continuation",
        "fresh-repeat-a",
        "fresh-repeat-b",
        "partition-alternate",
        "partition-default",
        "reporting-disabled",
        "reporting-enabled",
        "tick-native-partial-restore",
        "uninterrupted",
    }
    if (
        route.get("schema_version") != "eiel_execution_route_qualification_v1"
        or route.get("authority_state") != "route_qualified"
        or route.get("execution_route_sha256")
        != expected_identity.get("execution_route_sha256")
        or route.get("route_contract_sha256")
        != expected_identity.get("execution_route_sha256")
        or route.get("missing_obligations") != []
        or not isinstance(covered_obligations, Sequence)
        or isinstance(covered_obligations, (str, bytes))
        or not all(isinstance(item, str) for item in covered_obligations)
        or len(set(covered_obligations)) != len(covered_obligations)
        or not isinstance(route_contract_obligations, list)
        or route_contract_obligations != list(covered_obligations)
        or route_contract_obligations != sorted(route_contract_obligations)
        or set(route_contract_obligations) != SCIENCE_ROUTE_REQUIRED_OBLIGATIONS_V1
        or not isinstance(route_cases, Sequence)
        or isinstance(route_cases, (str, bytes))
        or not all(isinstance(item, str) for item in route_cases)
        or len(set(route_cases)) != len(route_cases)
        or set(route_cases) != expected_route_cases
        or not isinstance(reporting, Mapping)
        or reporting.get("exact") is not True
        or reporting.get("case_count") != len(route_cases)
        or not {"reporting-disabled", "reporting-enabled"}.issubset(route_cases)
    ):
        raise LongHorizonQualificationError(
            "promoted route receipt lacks exact reporting and route obligations"
        )

    gate24 = documents["gate24_resource_decision"]
    criteria = gate24.get("criteria")
    acceptance = _mapping(
        gate24.get("acceptance_measurements"), name="Gate24 acceptance_measurements"
    )
    required_measurements = (
        ("minimum_free_vram_bytes", "minimum_free_vram_threshold_bytes", ">="),
        ("active_gpu_duty_percent", "active_gpu_duty_threshold_percent", ">="),
        ("mean_sm_percent", "mean_sm_threshold_percent", ">="),
    )
    criteria_exact = (
        isinstance(criteria, list)
        and [
            row.get("criterion")
            for row in criteria
            if isinstance(row, Mapping)
        ]
        == list(SCIENCE_ROUTE_GATE24_REQUIRED_CRITERIA_V1)
    )
    measurements_present = SCIENCE_ROUTE_GATE24_REQUIRED_MEASUREMENTS_V1.issubset(
        acceptance
    )
    if (
        gate24.get("decision") != "PASS"
        or not criteria_exact
        or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS"
            for row in criteria
        )
        or not measurements_present
        or any(
            not isinstance(acceptance.get(measured), (int, float))
            or not isinstance(acceptance.get(threshold), (int, float))
            or acceptance[measured] < acceptance[threshold]
            for measured, threshold, _operator in required_measurements
        )
        or acceptance.get("event_rows_per_cell") != 41804
        or acceptance.get("event_row_bytes") != 141
        or acceptance.get("total_event_bytes") != 801633504
        or acceptance.get("capture_barrier_tick2500_seconds", 11.0)
        > acceptance.get("capture_release_threshold_seconds", 0.0)
        or acceptance.get("capture_barrier_tick5000_seconds", 11.0)
        > acceptance.get("capture_release_threshold_seconds", 0.0)
        or acceptance.get("successor_gpu_dispatch_delay_tick2500_seconds", 11.0)
        > acceptance.get("successor_dispatch_threshold_seconds", 0.0)
        or acceptance.get("dispatch_overhead_fraction", 1.0)
        > acceptance.get("dispatch_overhead_threshold_fraction", 0.0)
        or acceptance.get("longest_unexplained_idle_interval_seconds", 3.0)
        > acceptance.get("idle_interval_threshold_seconds", 2.0)
        or acceptance.get("temporary_capture_file_terminal_count") != 0
        or acceptance.get("host_pinned_bytes") != 0
        or acceptance.get("native_host_refill_calls") != 0
        or acceptance.get("intermediate_host_to_device_copies") != 0
        or acceptance.get("intermediate_device_to_host_copies") != 0
        or acceptance.get("queued_cuda_references") != 0
        or acceptance.get("queued_cuda_callbacks") != 0
        or acceptance.get("terminal_gpu_device_reset_return_code") != 0
        or acceptance.get("post_release_target_gpu_resident_samples") != 0
        or acceptance.get("post_release_target_max_framebuffer_bytes") != 0
    ):
        raise LongHorizonQualificationError(
            "Gate24 resource/performance decision is not an exact PASS"
        )
    gate24_execution = _mapping(
        gate24.get("execution_identity"), name="Gate24 execution_identity"
    )
    gate24_physical = _mapping(
        gate24.get("physical_authority"), name="Gate24 physical_authority"
    )
    if (
        gate24_execution.get("qualification_key_sha256")
        != uninterrupted_plan["qualification_key_sha256"]
        or gate24_execution.get("result_class") != "SUCCESS"
        or gate24_execution.get("baseline_profile")
        != "EIEL_PRODUCTION_BASELINE_V1"
        or gate24_execution.get("baseline_profile_version") != 2
        or gate24_execution.get("baseline_status") != "BASELINE_CONFORMANT"
        or gate24_execution.get("baseline_deviations") != []
        or gate24_physical
        != {
            "batch_count": 1,
            "cell_count": 136,
            "world_width": 32,
            "population": 256,
            "capacity": 512,
            "time_authority": "absolute_tick_v1",
            "total_ticks": 5000,
            "checkpoint_ticks": [2500, 5000],
            "expected_owner_transition_ticks": [1120, 2240, 3360, 4480],
            "observed_owner_transition_ticks": [1120, 2240, 3360, 4480],
            "physical_owner_identity_count": 1,
            "resident_owner_max_simultaneous": 1,
            "resident_owner_live_count_terminal": 0,
        }
    ):
        raise LongHorizonQualificationError(
            "Gate24 decision is not bound to the current Gate26 production identity"
        )

    gate25 = documents["gate25_impact_map"]
    gate25_target = _mapping(gate25.get("target"), name="Gate25 impact target")
    gate25_uncertainty = _mapping(
        gate25.get("uncertainty"), name="Gate25 impact uncertainty"
    )
    gate25_classifications = gate25.get("impact_classifications")
    gate25_preserved = [
        item
        for item in (gate25_classifications if isinstance(gate25_classifications, list) else [])
        if isinstance(item, Mapping)
        and item.get("classification") == "PRESERVED_COMPATIBLE_INTERFACE"
    ]
    if (
        gate25.get("schema_version") != "eiel_qualification_impact_map_v1"
        or gate25.get("gate") != 25
        or gate25.get("decision")
        != "TARGETED_BOUNDED_ROUTE_REQUALIFICATION_COMPLETE_LONG_HORIZON_DELTA_PENDING"
        or gate25_target.get("qualification_key_sha256")
        != uninterrupted_plan["qualification_key_sha256"]
        or gate25_target.get("execution_route_sha256")
        != expected_identity.get("execution_route_sha256")
        or gate25_uncertainty.get("impact_uncertain_count") != 0
        or gate25_uncertainty.get("unsafe_inheritance_count") != 0
        or not isinstance(gate25_classifications, list)
        or not any(
            "Gates 21-24 physical B136" in item.get("group", "")
            for item in gate25_preserved
        )
    ):
        raise LongHorizonQualificationError(
            "Gate25 impact map does not preserve the inherited physical interface"
        )

    impact = documents["gate27_impact_map"]
    targeted_delta = _mapping(impact.get("targeted_delta"), name="Gate27 targeted_delta")
    uncertainty = _mapping(impact.get("uncertainty"), name="Gate27 uncertainty")
    impact_target = _mapping(impact.get("target"), name="Gate27 impact target")
    if (
        impact.get("schema_version") != "eiel_qualification_impact_map_v1"
        or impact.get("gate") != 27
        or impact.get("decision")
        != "TARGETED_SCIENCE_ROUTE_AUTHORITY_DELTA_ONLY"
        or impact_target.get("execution_route_sha256")
        != expected_identity.get("execution_route_sha256")
        or impact_target.get("gate26_evidence_sha256")
        != expected_evidence_sha256
        or targeted_delta.get("native_rebuild") is not False
        or targeted_delta.get("gpu_execution") is not False
        or targeted_delta.get("broad_requalification") is not False
        or uncertainty.get("impact_uncertain_count") != 0
        or uncertainty.get("unsafe_inheritance_count") != 0
    ):
        raise LongHorizonQualificationError(
            "Gate27 impact map does not authorize the bounded inherited route"
        )

    source_ref = by_role["promoted_route_receipt"]
    source_route_receipt_sha256 = source_ref.get(
        "source_route_receipt_sha256", source_ref["file_sha256"]
    )
    if expected_source_route_receipt_sha256 is not None and source_route_receipt_sha256 != (
        expected_source_route_receipt_sha256
    ):
        raise LongHorizonQualificationError(
            "science-route source receipt hash differs"
        )
    normalized_refs = [by_role[role] for role in sorted(by_role)]
    return {
        "evidence_refs": normalized_refs,
        "documents": documents,
        "evidence_identity": validated_evidence["evidence_identity"],
        "evidence_binding_sha256": validated_evidence["evidence_binding"][
            "binding_sha256"
        ],
        "source_route_receipt_sha256": source_route_receipt_sha256,
        "canonical_route_pass": True,
        "production_resource_safety_pass": True,
        "terminal_finalization_pass": True,
        "science_output_pass": True,
    }


def materialize_canonical_full_production_route_attestation_v1(
    *,
    evidence_refs: Sequence[Mapping[str, Any]],
    expected_identity: Mapping[str, Any],
    expected_evidence_sha256: str,
    expected_evidence_binding_sha256: str,
    expected_source_route_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive a digest-bound Gate27 attestation from exact compact inputs."""

    refs = _read_science_route_evidence_refs_v1(
        evidence_refs,
        expected_identity=expected_identity,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_evidence_binding_sha256=expected_evidence_binding_sha256,
        expected_source_route_receipt_sha256=expected_source_route_receipt_sha256,
    )
    identity = refs["evidence_identity"]
    body = {
        "schema_version": SCIENCE_ROUTE_ATTESTATION_SCHEMA_VERSION,
        **{name: item for name, item in identity.items() if name != "route_scope"},
        "route_scope": CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
        "claim_dimension": SCIENCE_ROUTE_QUALIFIED,
        "evidence_sha256": expected_evidence_sha256,
        "evidence_binding_sha256": expected_evidence_binding_sha256,
        "source_route_receipt_sha256": refs["source_route_receipt_sha256"],
        "evidence_refs": refs["evidence_refs"],
        "canonical_route_pass": refs["canonical_route_pass"],
        "production_resource_safety_pass": refs["production_resource_safety_pass"],
        "terminal_finalization_pass": refs["terminal_finalization_pass"],
        "science_output_pass": refs["science_output_pass"],
    }
    return {**body, "attestation_sha256": canonical_sha256(body)}


def _identity_v1(
    evidence: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    identity = _mapping(evidence.get("evidence_identity"), name="evidence_identity")
    route_scope = identity.get("route_scope")
    if route_scope not in {
        CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
        LONG_HORIZON_RUNABILITY_ROUTE_SCOPE,
    }:
        raise LongHorizonQualificationError("evidence route_scope is unsupported")
    artifacts = _mapping(identity.get("artifact_sha256s"), name="artifact_sha256s")
    if not artifacts:
        raise LongHorizonQualificationError("artifact_sha256s must not be empty")
    normalized_artifacts = {
        _text(name, name="artifact identity"): _sha256(
            digest, name=f"artifact_sha256s.{name}"
        )
        for name, digest in sorted(artifacts.items())
    }
    geometry = _mapping(identity.get("geometry"), name="geometry")
    normalized_geometry = {
        name: _exact_int(geometry.get(name), name=f"geometry.{name}", minimum=1)
        for name in (
            "batch_cells",
            "world_width",
            "population_capacity",
            "controller_capacity",
        )
    }
    normalized = {
        "execution_route_sha256": _sha256(
            identity.get("execution_route_sha256"),
            name="evidence_identity.execution_route_sha256",
        ),
        "provider_id": _text(identity.get("provider_id"), name="provider_id"),
        "profile_id": _text(identity.get("profile_id"), name="profile_id"),
        "source_sha256": _sha256(
            identity.get("source_sha256"), name="evidence_identity.source_sha256"
        ),
        "artifact_sha256s": normalized_artifacts,
        "plan_sha256": _sha256(
            identity.get("plan_sha256"), name="evidence_identity.plan_sha256"
        ),
        "run_id": _text(identity.get("run_id"), name="run_id"),
        "run_receipt_sha256": _sha256(
            identity.get("run_receipt_sha256"),
            name="evidence_identity.run_receipt_sha256",
        ),
        "route_scope": route_scope,
        "geometry": normalized_geometry,
    }
    if expected_identity is not None:
        for name, expected in expected_identity.items():
            if name in normalized and normalized[name] != expected:
                raise LongHorizonQualificationError(
                    f"evidence identity differs at {name}"
                )
    return normalized


def _resource_limits_v1(evidence: Mapping[str, Any]) -> dict[str, int]:
    raw = _mapping(evidence.get("resource_limits"), name="resource_limits")
    return {
        name: _exact_int(raw.get(name), name=f"resource_limits.{name}", minimum=1)
        for name in (
            "gpu_budget_bytes",
            "host_budget_bytes",
            "minimum_free_gpu_bytes_required",
            "maximum_gpu_memory_slope_bytes_per_window",
            "maximum_host_memory_slope_bytes_per_window",
            "maximum_event_ring_occupancy_slope_per_window",
        )
    }


def _window_slope(values: Sequence[int]) -> float:
    return float(values[-1] - values[0]) / float(len(values) - 1)


def _validated_windows_v1(
    evidence: Mapping[str, Any], *, requested: int, limits: Mapping[str, int]
) -> tuple[list[dict[str, int]], dict[str, list[int] | dict[str, float]]]:
    raw = evidence.get("windows")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 4:
        raise LongHorizonQualificationError(
            "windows must contain at least four contiguous measurement windows"
        )
    windows: list[dict[str, int]] = []
    expected_start = 0
    for index, value in enumerate(raw):
        row = _mapping(value, name=f"windows[{index}]")
        normalized = {
            name: _exact_int(row.get(name), name=f"windows[{index}].{name}")
            for name in _WINDOW_FIELDS_V1
        }
        if normalized["start_tick"] != expected_start:
            raise LongHorizonQualificationError(
                "measurement windows are not contiguous"
            )
        if normalized["end_tick"] <= normalized["start_tick"]:
            raise LongHorizonQualificationError("measurement window is empty")
        if (
            normalized["peak_gpu_allocated_bytes"]
            > normalized["peak_gpu_reserved_bytes"]
        ):
            raise LongHorizonQualificationError(
                "GPU allocated bytes exceed reserved bytes in a window"
            )
        if normalized["peak_gpu_reserved_bytes"] > limits["gpu_budget_bytes"]:
            raise LongHorizonQualificationError("GPU budget is exceeded")
        if (
            normalized["minimum_free_gpu_bytes"]
            < limits["minimum_free_gpu_bytes_required"]
        ):
            raise LongHorizonQualificationError("GPU free-memory reserve is not met")
        if normalized["host_peak_rss_bytes"] > limits["host_budget_bytes"]:
            raise LongHorizonQualificationError("host-memory budget is exceeded")
        windows.append(normalized)
        expected_start = normalized["end_tick"]
    if expected_start != requested:
        raise LongHorizonQualificationError(
            "measurement windows do not cover the exact requested horizon"
        )
    vectors = {name: [row[name] for row in windows] for name in _WINDOW_FIELDS_V1[2:]}
    declared_pairs = (
        ("gpu_memory_slope_bytes_per_window", "peak_gpu_reserved_bytes"),
        (
            "gpu_allocated_memory_slope_bytes_per_window",
            "peak_gpu_allocated_bytes",
        ),
        ("host_memory_slope_bytes_per_window", "host_terminal_rss_bytes"),
        ("resident_event_bytes_slope_per_window", "resident_event_bytes"),
        ("resident_source_bytes_slope_per_window", "resident_source_bytes"),
        ("event_ring_occupancy_slope_per_window", "event_ring_peak_occupancy"),
    )
    slopes: dict[str, float] = {}
    for declared_name, vector_name in declared_pairs:
        declared = _finite_number(evidence.get(declared_name), name=declared_name)
        calculated = _window_slope(vectors[vector_name])
        if not math.isclose(declared, calculated, rel_tol=0.0, abs_tol=1e-9):
            raise LongHorizonQualificationError(
                f"{declared_name} differs from the measured windows"
            )
        slopes[declared_name] = calculated
    if slopes["resident_event_bytes_slope_per_window"] != 0.0:
        raise LongHorizonQualificationError("resident event bytes grow with horizon")
    if slopes["resident_source_bytes_slope_per_window"] != 0.0:
        raise LongHorizonQualificationError("resident source bytes grow with horizon")
    if (
        slopes["gpu_memory_slope_bytes_per_window"]
        > limits["maximum_gpu_memory_slope_bytes_per_window"]
    ):
        raise LongHorizonQualificationError(
            "GPU memory has an unbounded positive slope"
        )
    if (
        slopes["gpu_allocated_memory_slope_bytes_per_window"]
        > limits["maximum_gpu_memory_slope_bytes_per_window"]
    ):
        raise LongHorizonQualificationError(
            "GPU allocated memory has an unbounded positive slope"
        )
    if (
        slopes["host_memory_slope_bytes_per_window"]
        > limits["maximum_host_memory_slope_bytes_per_window"]
    ):
        raise LongHorizonQualificationError(
            "host memory has an unbounded positive slope"
        )
    event_capacity = _exact_int(
        evidence.get("event_ring_capacity"), name="event_ring_capacity", minimum=1
    )
    if (
        slopes["event_ring_occupancy_slope_per_window"]
        > limits["maximum_event_ring_occupancy_slope_per_window"]
    ):
        raise LongHorizonQualificationError(
            "event-ring occupancy has an unbounded positive slope"
        )
    return windows, {"vectors": vectors, "slopes": slopes}


def _validate_restore_v1(
    evidence: Mapping[str, Any],
    *,
    requested: int,
    terminal_digest: str,
    event_digest: str,
) -> dict[str, Any]:
    raw = _mapping(evidence.get("checkpoint_restore"), name="checkpoint_restore")
    checkpoint_tick = _exact_int(
        raw.get("checkpoint_tick"), name="checkpoint_restore.checkpoint_tick", minimum=1
    )
    fresh_owner_tick = _exact_int(
        raw.get("fresh_owner_restore_tick"),
        name="checkpoint_restore.fresh_owner_restore_tick",
        minimum=1,
    )
    if checkpoint_tick >= requested or fresh_owner_tick >= requested:
        raise LongHorizonQualificationError(
            "restore checkpoints must precede terminal tick"
        )
    digests = {
        name: _sha256(raw.get(name), name=f"checkpoint_restore.{name}")
        for name in _DIGEST_FIELDS_V1
    }
    if digests["checkpoint_digest"] != digests["restored_checkpoint_digest"]:
        raise LongHorizonQualificationError(
            "checkpoint export/load digest is not exact"
        )
    if (
        digests["fresh_owner_uninterrupted_terminal_digest"] != terminal_digest
        or digests["fresh_owner_restored_terminal_digest"] != terminal_digest
        or digests["fresh_owner_uninterrupted_event_digest"] != event_digest
        or digests["fresh_owner_restored_event_digest"] != event_digest
    ):
        raise LongHorizonQualificationError(
            "fresh-owner checkpoint/restore continuation digests are not exact"
        )
    return {
        "checkpoint_tick": checkpoint_tick,
        "fresh_owner_restore_tick": fresh_owner_tick,
        **digests,
        "checkpoint_export_load_exact": True,
        "fresh_owner_checkpoint_continuation_exact": True,
    }


def validate_long_horizon_evidence_v1(
    evidence: Mapping[str, Any],
    *,
    target_ticks: int = CURRENT_B136_LONG_HORIZON_MINIMUM_TICKS,
    minimum_owner_boundaries: int = 3,
    expected_identity: Mapping[str, Any] | None = None,
    canonical_full_production_route_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate complete immutable evidence; never infer facts from labels."""

    if not isinstance(evidence, Mapping):
        raise LongHorizonQualificationError("long-horizon evidence must be an object")
    if evidence.get("schema_version") != LONG_HORIZON_EVIDENCE_SCHEMA_VERSION:
        raise LongHorizonQualificationError(
            "long-horizon evidence schema is unsupported"
        )
    target = _exact_int(target_ticks, name="target_ticks", minimum=1)
    missing = sorted(
        name for name in REQUIRED_LONG_HORIZON_MEASUREMENTS_V1 if name not in evidence
    )
    if missing:
        raise LongHorizonQualificationError(
            f"long-horizon evidence is missing measurements: {missing}"
        )
    identity = _identity_v1(evidence, expected_identity=expected_identity)
    limits = _resource_limits_v1(evidence)
    requested = _exact_int(
        evidence["ticks_requested"], name="ticks_requested", minimum=1
    )
    committed = _exact_int(evidence["ticks_committed"], name="ticks_committed")
    boundaries = _exact_int(
        evidence["owner_boundaries_crossed"], name="owner_boundaries_crossed"
    )
    if requested < target or committed != requested:
        raise LongHorizonQualificationError(
            "long-horizon evidence does not reach the exact requested terminal tick"
        )
    if boundaries < minimum_owner_boundaries:
        raise LongHorizonQualificationError(
            "long-horizon evidence does not cross enough owner boundaries"
        )
    windows, measured = _validated_windows_v1(
        evidence, requested=requested, limits=limits
    )
    vectors = measured["vectors"]
    assert isinstance(vectors, dict)
    exact_aggregates = {
        "peak_gpu_allocated_bytes": max(vectors["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": max(vectors["peak_gpu_reserved_bytes"]),
        "minimum_free_gpu_bytes": min(vectors["minimum_free_gpu_bytes"]),
        "host_peak_rss_bytes": max(vectors["host_peak_rss_bytes"]),
        "host_terminal_rss_bytes": max(vectors["host_terminal_rss_bytes"]),
        "resident_event_bytes": max(vectors["resident_event_bytes"]),
        "resident_source_bytes": max(vectors["resident_source_bytes"]),
        "event_ring_peak_occupancy": max(vectors["event_ring_peak_occupancy"]),
        "events_emitted": sum(vectors["events_emitted"]),
        "events_drained": sum(vectors["events_drained"]),
    }
    for name, expected in exact_aggregates.items():
        if _exact_int(evidence.get(name), name=name) != expected:
            raise LongHorizonQualificationError(
                f"{name} differs from the per-window measurements"
            )
    capacity = _exact_int(
        evidence["event_ring_capacity"], name="event_ring_capacity", minimum=1
    )
    terminal_occupancy = _exact_int(
        evidence["terminal_event_ring_occupancy"],
        name="terminal_event_ring_occupancy",
    )
    external_count = _exact_int(
        evidence["external_event_history_count"],
        name="external_event_history_count",
    )
    if exact_aggregates["event_ring_peak_occupancy"] > capacity:
        raise LongHorizonQualificationError("event ring capacity was exceeded")
    if terminal_occupancy > capacity:
        raise LongHorizonQualificationError("terminal event occupancy exceeds capacity")
    if exact_aggregates["events_emitted"] != (
        exact_aggregates["events_drained"] + terminal_occupancy
    ):
        raise LongHorizonQualificationError("event accounting is not conservative")
    if external_count != exact_aggregates["events_drained"]:
        raise LongHorizonQualificationError(
            "external event history does not equal drained event history"
        )
    terminal_digest = _sha256(evidence["terminal_digest"], name="terminal_digest")
    event_digest = _sha256(evidence["event_digest"], name="event_digest")
    external_event_digest = _sha256(
        evidence["external_event_history_digest"],
        name="external_event_history_digest",
    )
    if external_event_digest != event_digest:
        raise LongHorizonQualificationError(
            "external event history digest differs from the exact event digest"
        )
    _validate_restore_v1(
        evidence,
        requested=requested,
        terminal_digest=terminal_digest,
        event_digest=event_digest,
    )
    frontier = _mapping(evidence["checkpoint_frontier"], name="checkpoint_frontier")
    if (
        frontier.get("next_absolute_tick", frontier.get("committed_next_tick"))
        != committed
    ):
        raise LongHorizonQualificationError(
            "checkpoint frontier does not equal the committed terminal tick"
        )
    if frontier.get("terminal_digest") != terminal_digest:
        raise LongHorizonQualificationError("checkpoint frontier digest differs")
    for name in ("errors", "ooms"):
        value = evidence[name]
        if not isinstance(value, (list, tuple)):
            raise LongHorizonQualificationError(f"{name} must be an array")
        if value:
            raise LongHorizonQualificationError(
                f"long-horizon evidence contains {name}"
            )
    body = {
        name: value for name, value in evidence.items() if name != "evidence_sha256"
    }
    evidence_sha256 = _sha256(evidence.get("evidence_sha256"), name="evidence_sha256")
    if evidence_sha256 != canonical_sha256(body):
        raise LongHorizonQualificationError("long-horizon evidence SHA-256 differs")
    dimensions = [LONG_HORIZON_RUNABILITY_QUALIFIED]
    binding_core = {
        "schema_version": QUALIFICATION_EVIDENCE_BINDING_SCHEMA_VERSION,
        "evidence_schema_version": LONG_HORIZON_EVIDENCE_SCHEMA_VERSION,
        "evidence_sha256": evidence_sha256,
        **identity,
        "target_ticks": requested,
        "resource_limits": limits,
        "qualification_dimensions": dimensions,
    }
    binding = {**binding_core, "binding_sha256": canonical_sha256(binding_core)}
    normalized_attestation: dict[str, Any] | None = None
    if canonical_full_production_route_attestation is not None:
        if identity["route_scope"] != LONG_HORIZON_RUNABILITY_ROUTE_SCOPE:
            raise LongHorizonQualificationError(
                "canonical route attestation requires unchanged long-horizon evidence scope"
            )
        if identity["geometry"] != {
            "batch_cells": 136,
            "world_width": 32,
            "population_capacity": 256,
            "controller_capacity": 512,
        }:
            raise LongHorizonQualificationError(
                "canonical full production route requires physical B136 geometry"
            )
        normalized_attestation = (
            validate_canonical_full_production_route_attestation_v1(
                canonical_full_production_route_attestation,
                expected_identity=identity,
                expected_evidence_sha256=evidence_sha256,
                expected_evidence_binding_sha256=binding["binding_sha256"],
            )
        )
        dimensions.append(SCIENCE_ROUTE_QUALIFIED)
        binding_core = {
            **binding_core,
            "qualification_dimensions": dimensions,
            "canonical_full_production_route_attestation_sha256": normalized_attestation[
                "attestation_sha256"
            ],
        }
        binding = {**binding_core, "binding_sha256": canonical_sha256(binding_core)}
    science_eligible = normalized_attestation is not None
    return {
        "target_ticks": target,
        "ticks_requested": requested,
        "ticks_committed": committed,
        "owner_boundaries_crossed": boundaries,
        "window_count": len(windows),
        "resident_memory_bounded": True,
        "event_history_externalized": True,
        "event_accounting_exact": True,
        "checkpoint_export_load_exact": True,
        "fresh_owner_checkpoint_continuation_exact": True,
        "exact_terminal": True,
        "evidence_identity": identity,
        "science_eligible": science_eligible,
        "qualification_dimensions": dimensions,
        "evidence_binding": binding,
        "canonical_full_production_route_attestation": normalized_attestation,
        "errors": [],
        "ooms": [],
    }


def validate_qualification_evidence_binding_v1(
    value: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable compact binding propagated into a QRR."""

    binding = _mapping(value, name="qualification_evidence_binding")
    core = {name: item for name, item in binding.items() if name != "binding_sha256"}
    if core.get("schema_version") != QUALIFICATION_EVIDENCE_BINDING_SCHEMA_VERSION:
        raise LongHorizonQualificationError(
            "qualification evidence binding schema is unsupported"
        )
    if binding.get("binding_sha256") != canonical_sha256(core):
        raise LongHorizonQualificationError(
            "qualification evidence binding SHA-256 differs"
        )
    for name in (
        "evidence_sha256",
        "execution_route_sha256",
        "source_sha256",
        "plan_sha256",
        "run_receipt_sha256",
    ):
        _sha256(binding.get(name), name=f"qualification_evidence_binding.{name}")
    for name in ("provider_id", "profile_id", "run_id"):
        _text(binding.get(name), name=f"qualification_evidence_binding.{name}")
    artifacts = _mapping(
        binding.get("artifact_sha256s"), name="binding artifact_sha256s"
    )
    if not artifacts:
        raise LongHorizonQualificationError("binding artifact identity is empty")
    for name, digest in artifacts.items():
        _text(name, name="binding artifact identity")
        _sha256(digest, name=f"binding artifact {name}")
    geometry = _mapping(binding.get("geometry"), name="binding geometry")
    for name in (
        "batch_cells",
        "world_width",
        "population_capacity",
        "controller_capacity",
    ):
        _exact_int(geometry.get(name), name=f"binding geometry.{name}", minimum=1)
    _exact_int(binding.get("target_ticks"), name="binding target_ticks", minimum=1)
    _resource_limits_v1({"resource_limits": binding.get("resource_limits")})
    dimensions = binding.get("qualification_dimensions")
    if not isinstance(dimensions, list) or dimensions not in (
        [LONG_HORIZON_RUNABILITY_QUALIFIED],
        [LONG_HORIZON_RUNABILITY_QUALIFIED, SCIENCE_ROUTE_QUALIFIED],
    ):
        raise LongHorizonQualificationError(
            "binding qualification dimensions are invalid"
        )
    attestation_sha256 = binding.get(
        "canonical_full_production_route_attestation_sha256"
    )
    if SCIENCE_ROUTE_QUALIFIED in dimensions:
        _sha256(
            attestation_sha256,
            name=(
                "qualification_evidence_binding."
                "canonical_full_production_route_attestation_sha256"
            ),
        )
    elif attestation_sha256 is not None:
        raise LongHorizonQualificationError(
            "long-horizon binding cannot carry a science-route attestation"
        )
    if expected_identity is not None:
        for name, expected in expected_identity.items():
            if binding.get(name) != expected:
                raise LongHorizonQualificationError(
                    f"qualification evidence binding differs at {name}"
                )
    return dict(binding)


def qualification_dimensions_v1(
    *,
    bounded_route_pass: bool,
    long_horizon_evidence: Mapping[str, Any] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    canonical_full_production_route_attestation: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return dimensions supported by validated evidence, never booleans."""

    if type(bounded_route_pass) is not bool:
        raise LongHorizonQualificationError(
            "bounded_route_pass must be an exact boolean"
        )
    if not bounded_route_pass:
        return ()
    dimensions = [BOUNDED_ROUTE_QUALIFIED]
    if long_horizon_evidence is None:
        return tuple(dimensions)
    validated = validate_long_horizon_evidence_v1(
        long_horizon_evidence,
        expected_identity=expected_identity,
        canonical_full_production_route_attestation=(
            canonical_full_production_route_attestation
        ),
    )
    dimensions.extend(validated["qualification_dimensions"])
    return tuple(dimensions)


def qualification_scope_for_plan_v1(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the claim boundary bound to one compiled plan."""

    execution = plan.get("execution") if isinstance(plan, Mapping) else None
    tick_native = (
        isinstance(execution, Mapping)
        and execution.get("execution_route_identity")
        == "tick_native_execution_route_v1"
    )
    total_ticks = (
        execution.get("total_ticks") if isinstance(execution, Mapping) else None
    )
    return {
        "dimensions": [BOUNDED_ROUTE_QUALIFIED],
        "long_horizon_required": bool(tick_native),
        "long_horizon_minimum_ticks": (
            CURRENT_B136_LONG_HORIZON_MINIMUM_TICKS if tick_native else None
        ),
        "science_claim_requires": [
            BOUNDED_ROUTE_QUALIFIED,
            LONG_HORIZON_RUNABILITY_QUALIFIED,
            "IMMUTABLE_EVIDENCE_IDENTITY",
            "BOUNDED_PER_WINDOW_RESOURCES",
            "EXACT_EVENT_ACCOUNTING",
            "CHECKPOINT_AND_FRESH_OWNER_RESTORE",
            "CANONICAL_FULL_ROUTE",
        ],
        "requested_total_ticks": total_ticks,
    }


def classify_qualification_impact_v1(
    changed_surfaces: Iterable[str],
) -> dict[str, Any]:
    surfaces = {str(item) for item in changed_surfaces}
    affected = sorted(surfaces & LONG_HORIZON_INVALIDATING_SURFACES_V1)
    return {
        "long_horizon_runability_invalidated": bool(affected),
        "affected_surfaces": affected,
        "preserve_unaffected_evidence": True,
        "required_action": (
            "targeted_long_horizon_delta"
            if affected
            else "preserve_existing_long_horizon_evidence"
        ),
    }


__all__ = [
    "BOUNDED_ROUTE_QUALIFIED",
    "CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE",
    "CURRENT_B136_LONG_HORIZON_MINIMUM_TICKS",
    "LONG_HORIZON_EVIDENCE_SCHEMA_VERSION",
    "LONG_HORIZON_INVALIDATING_SURFACES_V1",
    "LONG_HORIZON_RUNABILITY_QUALIFIED",
    "LONG_HORIZON_RUNABILITY_ROUTE_SCOPE",
    "LongHorizonQualificationError",
    "QUALIFICATION_EVIDENCE_BINDING_SCHEMA_VERSION",
    "REQUIRED_LONG_HORIZON_MEASUREMENTS_V1",
    "SCIENCE_ROUTE_QUALIFIED",
    "SCIENCE_ROUTE_ATTESTATION_SCHEMA_VERSION",
    "SCIENCE_ROUTE_EVIDENCE_REF_ROLES_V1",
    "SCIENCE_ROUTE_GATE24_REQUIRED_CRITERIA_V1",
    "SCIENCE_ROUTE_GATE24_REQUIRED_MEASUREMENTS_V1",
    "SCIENCE_ROUTE_PARAMETER_CONTRACT_FIELDS_V1",
    "SCIENCE_ROUTE_REQUIRED_OBLIGATIONS_V1",
    "classify_qualification_impact_v1",
    "qualification_dimensions_v1",
    "qualification_scope_for_plan_v1",
    "materialize_canonical_full_production_route_attestation_v1",
    "validate_long_horizon_evidence_v1",
    "validate_canonical_full_production_route_attestation_v1",
    "validate_qualification_evidence_binding_v1",
]
