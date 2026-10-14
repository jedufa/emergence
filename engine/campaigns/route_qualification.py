"""Generic execution and atomic publication for bounded route canaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from .canonical import canonical_sha256
from .contracts import (
    ExecutionRouteContractV1,
    ExecutionRouteQualificationReceiptV1,
)
from .orchestration import atomic_write_json, execute_campaign
from .qualification import (
    ACS_FOCUSED_ROUTE_OBLIGATIONS_V1,
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
    acs_checkpoint_staging_source_sha256_v1,
    execution_route_qualification_sidecar_path_v1,
    execution_route_science_qualification_projection_path_v1,
    read_execution_route_qualification_sidecar_v1,
    validate_execution_route_acs_focused_evidence_v1,
)
from .route_sentinels import RouteTransitionSentinelV1
from .long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    LONG_HORIZON_RUNABILITY_ROUTE_SCOPE,
    SCIENCE_ROUTE_QUALIFIED,
    LongHorizonQualificationError,
    materialize_canonical_full_production_route_attestation_v1,
    qualification_scope_for_plan_v1,
    validate_canonical_full_production_route_attestation_v1,
    validate_long_horizon_evidence_v1,
)
from .tick_native_control import (
    FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1,
    TickNativeControlV1,
)


class RouteQualificationError(RuntimeError):
    """A generated route canary did not establish complete exact evidence."""


PreflightRunnerV1 = Callable[[Mapping[str, Any]], Mapping[str, Any]]
OrchestrationRunnerV1 = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
ACSFocusedProbeRunnerV1 = Callable[[str, frozenset[str], int], Mapping[str, Any] | None]

_B136_5000TICK_QUALIFICATION_PAIR_PROFILE_V1 = (
    "eiel_b136_5000tick_qualification_pair_v1"
)


def _validate_b136_5000tick_qualification_pair_v1(
    evidence: Mapping[str, Any],
    *,
    qualification_dimensions: Sequence[str],
    canonical_revalidation: bool = True,
) -> None:
    """Reject generic/self-labelled evidence at the formal Q6 promotion seam."""

    identity = evidence.get("evidence_identity")
    restore = evidence.get("checkpoint_restore")
    bindings = evidence.get("paired_run_bindings")
    if (
        evidence.get("materialization_profile")
        != _B136_5000TICK_QUALIFICATION_PAIR_PROFILE_V1
        or not isinstance(identity, Mapping)
        or identity.get("route_scope") != LONG_HORIZON_RUNABILITY_ROUTE_SCOPE
        or identity.get("geometry")
        != {
            "batch_cells": 136,
            "world_width": 32,
            "population_capacity": 256,
            "controller_capacity": 512,
        }
        or evidence.get("ticks_requested") != 5000
        or evidence.get("ticks_committed") != 5000
        or not isinstance(restore, Mapping)
        or restore.get("checkpoint_tick") != 2500
        or restore.get("fresh_owner_restore_tick") != 2500
        or list(qualification_dimensions) != [LONG_HORIZON_RUNABILITY_QUALIFIED]
        or not isinstance(bindings, Mapping)
        or evidence.get("paired_run_bindings_sha256")
        != canonical_sha256(dict(bindings))
    ):
        raise RouteQualificationError(
            "long-horizon promotion requires the dedicated exact physical-B136 "
            "5000/2500 qualification-pair attestation"
        )
    required_binding_fields = {
        "run_root",
        "request_sha256",
        "result_sha256",
        "plan_sha256",
        "run_passport_sha256",
        "lifecycle_contract_sha256",
        "disposition_manifest_sha256",
        "storage_disposition_receipt_sha256",
        "campaign_spec_sha256",
        "execution_route_sha256",
        "qualification_receipt_sha256",
        "invocation_identity_sha256",
    }
    arms: list[Mapping[str, Any]] = []
    for arm_name in ("uninterrupted", "restored"):
        arm = bindings.get(arm_name)
        if not isinstance(arm, Mapping) or not required_binding_fields.issubset(arm):
            raise RouteQualificationError(
                f"qualification-pair attestation lacks canonical {arm_name} lineage"
            )
        arms.append(arm)
    roots = [arm.get("run_root") for arm in arms]
    if (
        any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots)
        or roots[0] == roots[1]
    ):
        raise RouteQualificationError(
            "qualification-pair attestation requires distinct absolute run roots"
        )
    process_ids = arms[1].get("process_instance_ids")
    if (
        not isinstance(process_ids, Sequence)
        or isinstance(process_ids, (str, bytes))
        or len(process_ids) != 2
        or any(not isinstance(item, str) or not item for item in process_ids)
        or process_ids[0] == process_ids[1]
    ):
        raise RouteQualificationError(
            "qualification-pair attestation lacks distinct producer/restore processes"
        )
    from .long_horizon_evidence_materialization import (
        LongHorizonEvidenceMaterializationError,
        validate_b136_5000tick_qualification_pair_evidence_v1,
    )

    if canonical_revalidation:
        try:
            validate_b136_5000tick_qualification_pair_evidence_v1(evidence)
        except LongHorizonEvidenceMaterializationError as exc:
            raise RouteQualificationError(
                f"qualification-pair canonical revalidation failed: {exc}"
            ) from exc


_OBLIGATION_SENTINEL_TRANSITION_V1 = {
    "generation_boundary": "generation_boundary",
    "first_post_boundary_tick": "first_tick_after_boundary",
    "second_post_boundary_tick": "second_tick_after_boundary",
    "post_boundary_policy_cache_refill": ("first_policy_cache_refill_after_boundary"),
    "checkpoint_export": "checkpoint_export_boundary",
    "terminal_settlement": "terminal_settlement",
}
_SEMANTIC_ROUTE_OBLIGATIONS_V1 = frozenset(
    {"ordinary_birth_disabled", "turnover_refill"}
)
_ACS_ASYNC_ROUTE_OBLIGATIONS_V1 = frozenset(
    {
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
        "acs2_host_only_capture",
        "durable_frontier_publication",
    }
)
_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1 = ACS_FOCUSED_ROUTE_OBLIGATIONS_V1
_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1 = frozenset(
    {
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
        "compact_science_host_only_capture",
        "compact_science_checkpoint_prefix_drain",
    }
)
_COMPACT_SCIENCE_FOCUSED_EVIDENCE_OBLIGATIONS_V1 = frozenset(
    {
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
    }
)
_TICK_NATIVE_ROUTE_OBLIGATIONS_V1 = frozenset(
    {
        "non_generation_divisible_terminal_tick",
        "non_generation_divisible_checkpoint_tick",
        "non_lattice_divisible_terminal_tick",
        "non_lattice_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_generation_restore",
        "fresh_owner_partial_tick_restore",
        "first_committed_tick_after_partial_restore",
        "first_post_restore_refill",
    }
)
_GV2_MUTATION_RATE_Q10_KEY = "genome_upgrade_v2.mutation_rate_q10"
_GV2_MUTATION_STEP_Q10_KEY = "genome_upgrade_v2.mutation_step_q10"


def bounded_route_qualification_preflight_v1(
    provider: Any,
    plan: Mapping[str, Any],
    *,
    profile: Any | None = None,
) -> dict[str, Any]:
    """Grant only explicit bounded-canary dispatch, never campaign authority."""

    from .preflight import preflight_campaign
    from .providers import ExecutionProviderRegistryV1

    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    receipt = preflight_campaign(
        plan,
        provider_registry=registry,
        profile=profile,
    )
    qualification_only_reason_codes = {
        "execution_route_obligation_uncovered",
        "execution_route_hash_mismatch",
        "execution_route_profile_mismatch",
        "execution_route_qualification_missing",
        # A profile awaiting its first route receipt must be able to run the
        # bounded canary that produces that receipt.  This exception is local
        # to qualification mode; ordinary preflight remains fail-closed.
        "profile_not_execution_qualified",
    }
    resources = receipt.get("resources")
    cuda = resources.get("cuda") if isinstance(resources, Mapping) else None
    if (
        isinstance(cuda, Mapping)
        and cuda.get("free_covers_admission_bound_and_headroom") is True
        and cuda.get("live_admission_bound_source")
        == "provider_conservative_projection"
        and type(cuda.get("live_admission_bound_bytes")) is int
    ):
        qualification_only_reason_codes.add(
            "provider_vram_bytes_upper_bound_unavailable"
        )
    other_authority_reasons = [
        row
        for row in receipt.get("execution_authority_reasons", [])
        if not isinstance(row, Mapping)
        or row.get("code") not in qualification_only_reason_codes
    ]
    route = receipt.get("execution_route")
    if isinstance(route, dict) and (
        route.get("authority_state") == "legacy_unbound"
        or route.get("route_hash_exact") is not True
    ):
        # A receipt for a different route never grants ordinary execution, but
        # it must not prevent the bounded canary that proves this exact route.
        route["authority_state"] = "qualification_only"
    receipt["qualification_mode"] = True
    receipt["ready_for_route_qualification"] = (
        receipt.get("ok") is True and not other_authority_reasons
    )
    receipt["ready_for_execution"] = False
    return receipt


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteQualificationError(f"{name} must be an object")
    return dict(value)


def _rows(value: Any, *, name: str) -> list[dict[str, Any]]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not all(isinstance(row, Mapping) for row in value)
    ):
        raise RouteQualificationError(f"{name} must be an array of objects")
    return [dict(row) for row in value]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouteQualificationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _canary_cases(canary: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = _mapping(canary.get("qualification_plan"), name="qualification_plan")
    cases = _rows(plan.get("cases"), name="qualification_plan.cases")
    if not cases:
        raise RouteQualificationError("qualification plan has no cases")
    identifiers = [row.get("id") for row in cases]
    if any(
        not isinstance(case_id, str) or not case_id for case_id in identifiers
    ) or len(identifiers) != len(set(identifiers)):
        raise RouteQualificationError("qualification plan case IDs are invalid")
    return cases


def _bounded_cells(
    campaign_plan: Mapping[str, Any],
    *,
    count: int,
    case_id: str,
    preserve_distinct_source_cells: bool = False,
) -> list[dict[str, Any]]:
    source = _rows(campaign_plan.get("cells"), name="campaign plan cells")
    if not source:
        raise RouteQualificationError("campaign plan has no source cell")
    if preserve_distinct_source_cells:
        # Partition parity is a logical-world identity check.  It must use
        # distinct source cells even when several cells share the same
        # route-affecting mutation regime.  Replacing them with repeated
        # representatives would turn a two-world partition test into a
        # duplicate-world test and can hide seed/identity binding defects.
        representatives = _route_identity_representative_cells(
            source, minimum_count=count
        )
        if len(representatives) <= count:
            source = [dict(source[index % len(source)]) for index in range(count)]
        else:
            # A bounded case may intentionally carry more logical cells than
            # fit in one physical batch so that every route-affecting regime
            # remains represented.  Retain that established invariant when
            # the requested physical extent is smaller than the regime union.
            source = representatives
    else:
        source = _route_identity_representative_cells(source, minimum_count=count)
    cells: list[dict[str, Any]] = []
    for ordinal, source_row in enumerate(source):
        row = deepcopy(source_row)
        source_id = str(row.get("cell_id", ordinal))
        row.update(
            {
                "ordinal": ordinal,
                "cell_id": _stable_id("route-cell", case_id, source_id, str(ordinal)),
                "branch_id": _stable_id(
                    "route-branch", case_id, source_id, str(ordinal)
                ),
                "rng_lineage_id": _stable_id(
                    "route-rng", case_id, source_id, str(ordinal)
                ),
            }
        )
        cells.append(row)
    return cells


def _genome_v2_mutation_regime_q10(
    cell: Mapping[str, Any],
) -> tuple[int, int] | None:
    parameters = cell.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    rate = parameters.get(_GV2_MUTATION_RATE_Q10_KEY)
    step = parameters.get(_GV2_MUTATION_STEP_Q10_KEY)
    if rate is None and step is None:
        return None
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int)
        or isinstance(step, bool)
        or not isinstance(step, int)
    ):
        raise RouteQualificationError(
            "route-affecting Genome V2 mutation regime must be exact Q10 integers"
        )
    return int(rate), int(step)


def _route_identity_representative_cells(
    source: Sequence[Mapping[str, Any]], *, minimum_count: int
) -> list[dict[str, Any]]:
    """Keep representatives for per-cell values that affect route identity."""

    representatives: list[Mapping[str, Any]] = []
    seen_regimes: set[tuple[int, int]] = set()
    for cell in source:
        regime = _genome_v2_mutation_regime_q10(cell)
        if regime is None or regime in seen_regimes:
            continue
        seen_regimes.add(regime)
        representatives.append(cell)

    selected: list[Mapping[str, Any]] = list(representatives)
    target_count = max(minimum_count, len(representatives), 1)
    ordinal = 0
    while len(selected) < target_count:
        selected.append(source[ordinal % len(source)])
        ordinal += 1
    return [dict(row) for row in selected[:target_count]]


def _bounded_batches(
    cells: Sequence[Mapping[str, Any]], *, batch_cells: int, partition: Any
) -> list[dict[str, Any]]:
    if partition == "alternate" and len(cells) > 1:
        return [
            {
                "ordinal": ordinal,
                "batch_id": f"route-batch-{ordinal:04d}",
                "cell_ids": [str(cell["cell_id"])],
            }
            for ordinal, cell in enumerate(cells)
        ]
    batches: list[dict[str, Any]] = []
    for ordinal, start in enumerate(range(0, len(cells), batch_cells)):
        batches.append(
            {
                "ordinal": ordinal,
                "batch_id": f"route-batch-{ordinal:04d}",
                "cell_ids": [
                    str(cell["cell_id"]) for cell in cells[start : start + batch_cells]
                ],
            }
        )
    return batches


def _comparison_branches(
    case: Mapping[str, Any], *, source_cell_id: str, generations: int | None
) -> list[dict[str, Any]]:
    assignments = _mapping(case.get("assignments"), name="case assignments")
    if assignments.get("time_authority") == "absolute_tick_v1":
        # Tick-native continuation and reporting parity are paired canary
        # campaigns.  They must not re-enter the generation-shaped comparison
        # adapter merely to express a split or observation boundary.
        return []
    mode = assignments.get("mode")
    case_id = str(case["id"])
    if case_id == "fresh-repeat-b":
        return []
    if mode == "disk-continuation":
        kind = "checkpoint_continuation"
        split = assignments.get("continuation_split_generation")
        if (
            isinstance(split, bool)
            or not isinstance(split, int)
            or split <= 0
            or split >= generations
        ):
            raise RouteQualificationError(
                "disk-continuation case lacks a valid split generation"
            )
        repeat = None
    elif case_id == "reporting-enabled":
        kind = "reporting_parity"
        split = None
        repeat = None
    else:
        return []
    return [
        {
            "ordinal": 0,
            "branch_id": _stable_id("route-compare", case_id, kind),
            "kind": kind,
            "source_cell_id": source_cell_id,
            "repeat_ordinal": repeat,
            "split_generation": split,
        }
    ]


def _bounded_resource_shape_v1(
    plan: Mapping[str, Any],
    *,
    cells: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    generations: int | None,
    ticks_per_generation: int | None,
    total_ticks: int | None = None,
    tick_native: bool = False,
) -> dict[str, Any]:
    """Rebind planning-only cardinalities to the derived canary case.

    The target plan's B48 planning estimate is an upper bound for the target,
    not the shape of a B2 route-qualification case.  Providers consume these
    dimensions when calculating their own conservative admission projections,
    so retaining the target shape can both misstate the case and select an
    inapplicable resource envelope.
    """

    raw_estimate = plan.get("resource_estimate")
    # Unit-level route mechanics and non-resource providers are permitted to
    # omit the compiler estimate.  Populate only the bounded cardinalities in
    # that case; a resource-owning provider still validates its own inputs.
    estimate = dict(raw_estimate) if isinstance(raw_estimate, Mapping) else {}
    physical_batch_cells = max((len(row["cell_ids"]) for row in batches), default=0)
    if physical_batch_cells <= 0:
        raise RouteQualificationError("bounded case has no physical batch cells")
    comparison_trajectories = max(1, len(branches))
    logical_trajectories = len(cells) + comparison_trajectories
    if tick_native:
        if type(total_ticks) is not int or total_ticks < 1:
            raise RouteQualificationError(
                "tick-native bounded case lacks an exact total_ticks horizon"
            )
        execution = plan.get("execution")
        control = (
            execution.get("tick_native_control")
            if isinstance(execution, Mapping)
            else None
        )
        if not isinstance(control, Mapping):
            control = {
                "resident_owner_span_ticks": 1_120,
                "native_dispatch_window_ticks": 1_120,
                "source_window_ticks": 160,
                "event_window_capacity": 41_804,
            }
        estimate.update(
            {
                "logical_cells": len(cells),
                "logical_trajectories": logical_trajectories,
                "comparison_trajectories": comparison_trajectories,
                "physical_batch_cells": physical_batch_cells,
                "logical_agent_ticks": logical_trajectories * total_ticks,
                "time_authority": "absolute_tick_v1",
                "total_ticks": total_ticks,
                "resident_owner_span_ticks": control.get("resident_owner_span_ticks"),
                "native_dispatch_window_ticks": control.get(
                    "native_dispatch_window_ticks"
                ),
                "source_window_ticks": control.get("source_window_ticks"),
                "event_window_capacity": control.get("event_window_capacity"),
                "resident_event_bytes_horizon_invariant": True,
                "resident_source_bytes_horizon_invariant": True,
                "batch_admission": "bounded_route_qualification_case_shape",
            }
        )
    else:
        if type(generations) is not int or type(ticks_per_generation) is not int:
            raise RouteQualificationError(
                "legacy bounded case lacks generation dimensions"
            )
        estimate.update(
            {
                "logical_cells": len(cells),
                "logical_trajectories": logical_trajectories,
                "comparison_trajectories": comparison_trajectories,
                "physical_batch_cells": physical_batch_cells,
                "logical_agent_ticks": (
                    logical_trajectories * generations * ticks_per_generation
                ),
                "batch_admission": "bounded_route_qualification_case_shape",
            }
        )
    return estimate


def _tick_native_direct_control_aliases_v1(
    control: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the canonical absolute-tick control into execution fields."""

    aliases = {
        name: control[name]
        for name in (
            "total_ticks",
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "source_window_ticks",
            "event_window_capacity",
            "report_every_ticks",
            "science_sample_interval_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "checkpoint_every_ticks",
        )
    }
    aliases["start_next_tick"] = control["next_absolute_tick"]
    aliases["checkpoint_ticks"] = list(control["checkpoint_ticks"])
    return aliases


def _validated_tick_native_prior_control_v1(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    forbidden = sorted(set(value).intersection(FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1))
    if forbidden:
        raise RouteQualificationError(
            "tick-native control contains forbidden compatibility aliases: "
            f"{forbidden}"
        )
    return value


def derive_bounded_route_campaigns_v1(
    campaign_plan: Mapping[str, Any], canary: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Derive deterministic case campaigns without changing route identity."""

    base = _mapping(campaign_plan, name="campaign plan")
    route_hash = canary.get("execution_route_sha256")
    if not isinstance(route_hash, str) or len(route_hash) != 64:
        raise RouteQualificationError("canary lacks an execution route SHA-256")
    derived: list[dict[str, Any]] = []
    for case in _canary_cases(canary):
        case_id = str(case["id"])
        assignments = _mapping(case.get("assignments"), name="case assignments")
        tick_native_case = (
            assignments.get("time_authority") == "absolute_tick_v1"
            or assignments.get("mode") == "tick-native-continuation"
        )
        generations = assignments.get("generations")
        ticks = assignments.get("ticks_per_generation")
        batch_cells = assignments.get("physical_batch_cells")
        reporting_enabled = assignments.get("reporting_enabled")
        bounded_values = (
            (batch_cells,) if tick_native_case else (generations, ticks, batch_cells)
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in bounded_values
        ) or not isinstance(reporting_enabled, bool):
            raise RouteQualificationError(
                f"canary case {case_id!r} has invalid bounded dimensions"
            )
        plan = deepcopy(base)
        partition = assignments.get("partition")
        # The default/alternate partition cases must run identical logical
        # cells. Only their physical batching may differ.
        cell_identity_case = (
            "partition-parity"
            if case_id in {"partition-default", "partition-alternate"}
            else (
                "fresh-repeat"
                if case_id in {"fresh-repeat-a", "fresh-repeat-b"}
                else (
                    "continuation-parity"
                    if case_id in {"uninterrupted", "disk-continuation"}
                    else (
                        "reporting-parity"
                        if case_id in {"reporting-disabled", "reporting-enabled"}
                        else case_id
                    )
                )
            )
        )
        cells = _bounded_cells(
            plan,
            count=batch_cells,
            case_id=cell_identity_case,
            preserve_distinct_source_cells=(cell_identity_case == "partition-parity"),
        )
        batches = _bounded_batches(cells, batch_cells=batch_cells, partition=partition)
        execution = _mapping(plan.get("execution"), name="campaign plan execution")
        if tick_native_case:
            checkpoint_ticks = assignments.get("checkpoint_ticks")
            total_ticks = assignments.get("total_ticks")
            deliberate_restore_tick = assignments.get("deliberate_restore_tick")
            report_every_ticks = assignments.get("report_every_ticks")
            if (
                not isinstance(checkpoint_ticks, str)
                or type(total_ticks) is not int
                or (
                    deliberate_restore_tick not in (None, "none")
                    and type(deliberate_restore_tick) is not int
                )
                or (
                    report_every_ticks not in (None, "none")
                    and type(report_every_ticks) is not int
                )
            ):
                raise RouteQualificationError(
                    "tick-native canary case has invalid absolute-tick assignments"
                )
            try:
                parsed_checkpoint_ticks = [
                    int(value) for value in checkpoint_ticks.split(",")
                ]
            except ValueError as exc:
                raise RouteQualificationError(
                    "tick-native canary checkpoint schedule is invalid"
                ) from exc
            if (
                not parsed_checkpoint_ticks
                or parsed_checkpoint_ticks != sorted(set(parsed_checkpoint_ticks))
                or parsed_checkpoint_ticks[-1] != total_ticks
                or any(
                    value <= 0 or value > total_ticks
                    for value in parsed_checkpoint_ticks
                )
            ):
                raise RouteQualificationError(
                    "tick-native canary checkpoint schedule is invalid"
                )
            restore_tick = (
                None
                if deliberate_restore_tick in (None, "none")
                else deliberate_restore_tick
            )
            report_interval = (
                None if report_every_ticks in (None, "none") else report_every_ticks
            )
            execution.pop("generations", None)
            execution.pop("ticks_per_generation", None)
            execution.pop("checkpoints", None)
            execution.pop("tick_schedule", None)
            execution.pop("internal_lifecycle_schedule", None)
            execution.pop("physical_window_generations", None)
            execution.pop("max_internal_lifecycle_segments_per_dispatch", None)
            prior_control = execution.get("tick_native_control")
            if not isinstance(prior_control, Mapping):
                prior_control = {}
            prior_control = _validated_tick_native_prior_control_v1(prior_control)
            owner_span = int(prior_control.get("resident_owner_span_ticks", 1_120))
            dispatch_window = int(
                prior_control.get("native_dispatch_window_ticks", 1_120)
            )
            source_window = int(prior_control.get("source_window_ticks", 160))
            native_control = TickNativeControlV1(
                total_ticks=total_ticks,
                start_next_tick=0,
                resident_owner_span_ticks=owner_span,
                native_dispatch_window_ticks=dispatch_window,
                report_every_ticks=report_interval,
                science_sample_interval_ticks=prior_control.get(
                    "science_sample_interval_ticks"
                ),
                reducer_every_ticks=prior_control.get("reducer_every_ticks"),
                archive_every_ticks=prior_control.get("archive_every_ticks"),
                checkpoint_ticks=tuple(parsed_checkpoint_ticks),
                checkpoint_every_ticks=None,
                event_window_capacity=int(
                    prior_control.get("event_window_capacity", 41_804)
                ),
                source_window_ticks=source_window,
            ).payload()
            execution.update(
                {
                    "execution_route_identity": "tick_native_execution_route_v1",
                    **_tick_native_direct_control_aliases_v1(native_control),
                    "deliberate_restore_tick": restore_tick,
                    "deliberate_restore_ticks": (
                        [restore_tick] if restore_tick is not None else []
                    ),
                    "time_authority": "absolute_tick_v1",
                    "tick_native_control": native_control,
                    "checkpoint_storage_mode": "disk",
                    "reporting_enabled": reporting_enabled,
                }
            )
        else:
            async_staging = bool(
                set(canary.get("missing_obligations", ()))
                & _ACS_ASYNC_ROUTE_OBLIGATIONS_V1
            )
            requirements = canary.get("canary_requirements")
            minimum_segment_generations = (
                requirements.get("resident_segment_generation_min", 1)
                if isinstance(requirements, Mapping)
                else 1
            )
            if (
                type(minimum_segment_generations) is not int
                or minimum_segment_generations < 1
            ):
                raise RouteQualificationError(
                    "canary resident segment minimum is invalid"
                )
            async_checkpoints = [generations]
            if async_staging and generations >= 2 * minimum_segment_generations:
                async_checkpoints = [minimum_segment_generations, generations]
            checkpoint_generations = (
                [int(assignments["continuation_split_generation"])]
                if assignments.get("mode") == "disk-continuation"
                else async_checkpoints
            )
            if execution.get("execution_route_identity") == (
                "tick_native_execution_route_v1"
            ):
                # Keep route identity; project the bounded G/T canary onto
                # absolute ticks so dual time authority cannot enter execution.
                execution.pop("generations", None)
                execution.pop("ticks_per_generation", None)
                execution.pop("checkpoints", None)
                execution.pop("tick_schedule", None)
                execution.pop("internal_lifecycle_schedule", None)
                execution.pop("physical_window_generations", None)
                execution.pop("max_internal_lifecycle_segments_per_dispatch", None)
                total_ticks = int(generations) * int(ticks)
                checkpoint_ticks = sorted(
                    {
                        int(generation) * int(ticks)
                        for generation in checkpoint_generations
                        if type(generation) is int and generation > 0
                    }
                    | {total_ticks}
                )
                prior_control = execution.get("tick_native_control")
                if not isinstance(prior_control, Mapping):
                    prior_control = {}
                prior_control = _validated_tick_native_prior_control_v1(prior_control)
                owner_span = int(prior_control.get("resident_owner_span_ticks", 1_120))
                dispatch_window = int(
                    prior_control.get("native_dispatch_window_ticks", 1_120)
                )
                source_window = int(prior_control.get("source_window_ticks", 160))
                native_control = TickNativeControlV1(
                    total_ticks=total_ticks,
                    start_next_tick=0,
                    resident_owner_span_ticks=owner_span,
                    native_dispatch_window_ticks=dispatch_window,
                    report_every_ticks=prior_control.get("report_every_ticks"),
                    science_sample_interval_ticks=prior_control.get(
                        "science_sample_interval_ticks"
                    ),
                    reducer_every_ticks=prior_control.get("reducer_every_ticks"),
                    archive_every_ticks=prior_control.get("archive_every_ticks"),
                    checkpoint_ticks=tuple(checkpoint_ticks),
                    checkpoint_every_ticks=None,
                    event_window_capacity=int(
                        prior_control.get("event_window_capacity", 41_804)
                    ),
                    source_window_ticks=source_window,
                ).payload()
                execution.update(
                    {
                        **_tick_native_direct_control_aliases_v1(native_control),
                        "deliberate_restore_tick": None,
                        "deliberate_restore_ticks": [],
                        "time_authority": "absolute_tick_v1",
                        "tick_native_control": native_control,
                        "checkpoint_storage_mode": "disk",
                        "reporting_enabled": reporting_enabled,
                    }
                )
            else:
                execution.update(
                    {
                        "generations": generations,
                        "ticks_per_generation": ticks,
                        "checkpoints": checkpoint_generations,
                        "checkpoint_storage_mode": "disk",
                        "reporting_enabled": reporting_enabled,
                    }
                )
        # A campaign technical gate names cells and batches from the full
        # campaign plan. Route qualification derives its own bounded cells and
        # batches, so retaining that gate would make the canary inspect
        # unrelated manifests. The canary itself is the execution-route gate.
        execution.pop("technical_gate", None)
        branches = _comparison_branches(
            case, source_cell_id=cells[0]["cell_id"], generations=generations
        )
        reporting_required = any(row["kind"] == "reporting_parity" for row in branches)
        resource_estimate = _bounded_resource_shape_v1(
            plan,
            cells=cells,
            batches=batches,
            branches=branches,
            generations=generations,
            ticks_per_generation=ticks,
            total_ticks=(
                execution.get("total_ticks")
                if execution.get("execution_route_identity")
                == "tick_native_execution_route_v1"
                else None
            ),
            tick_native=tick_native_case,
        )
        plan.update(
            {
                "campaign_id": _stable_id(
                    "route-canary", str(base.get("campaign_id")), case_id
                ),
                "execution": execution,
                "cells": cells,
                "physical_batches": batches,
                "comparisons": {
                    "branches": branches,
                    "reporting_parity": {
                        "required": reporting_required,
                        "accepted_evidence_modes": [
                            "enabled_disabled_exact_terminal_state"
                        ],
                    },
                },
                "resource_estimate": resource_estimate,
                "execution_route_sha256": route_hash,
                "route_qualification_case_id": case_id,
            }
        )
        from .run_contract_v1 import refresh_resolved_run_contract_v1

        refreshed_contract = refresh_resolved_run_contract_v1(plan)
        if refreshed_contract is not None:
            plan["run_contract"] = refreshed_contract
        plan.pop("plan_sha256", None)
        plan["plan_sha256"] = canonical_sha256(plan)
        derived.append(plan)
    return tuple(derived)


def _rebind_bounded_route_case_hashes_v1(
    plans: Sequence[dict[str, Any]],
    *,
    provider_registry: Any | None,
) -> tuple[dict[str, Any], ...]:
    """Bind each transformed case to the provider's executable route hash.

    A bounded canary is derived from a parent route contract, but its absolute
    tick horizon and checkpoint schedule can be route-affecting inputs for the
    provider.  The parent hash therefore cannot be copied into every derived
    plan.  Re-resolve the provider contract after the transformation and keep
    the parent route hash as the qualification receipt's authority identity.
    """

    if provider_registry is None:
        return tuple(plans)
    from .run_contract_v1 import refresh_resolved_run_contract_v1

    rebound: list[dict[str, Any]] = []
    for plan in plans:
        profile_id = plan.get("resolved_profile")
        if not isinstance(profile_id, str) or not profile_id:
            raise RouteQualificationError(
                "bounded route case lacks a resolved provider profile"
            )
        provider = provider_registry.get(profile_id)
        if provider is None:
            raise RouteQualificationError(
                f"bounded route case profile {profile_id!r} has no provider"
            )
        resolver = getattr(provider, "execution_route_contract", None)
        if not callable(resolver):
            raise RouteQualificationError(
                f"provider {getattr(provider, 'provider_id', profile_id)!r} "
                "cannot resolve a bounded case route"
            )
        contract = resolver(plan)
        case_route_hash = getattr(contract, "execution_route_sha256", None)
        if isinstance(contract, Mapping):
            case_route_hash = contract.get("execution_route_sha256", case_route_hash)
        if (
            not isinstance(case_route_hash, str)
            or len(case_route_hash) != 64
            or any(char not in "0123456789abcdef" for char in case_route_hash)
        ):
            raise RouteQualificationError(
                "provider returned an invalid bounded case execution route hash"
            )
        plan["execution_route_sha256"] = case_route_hash
        refreshed_contract = refresh_resolved_run_contract_v1(plan)
        if refreshed_contract is not None:
            plan["run_contract"] = refreshed_contract
        plan.pop("plan_sha256", None)
        plan["plan_sha256"] = canonical_sha256(plan)
        rebound.append(plan)
    return tuple(rebound)


def _default_orchestration_runner(
    provider_registry: Any,
) -> OrchestrationRunnerV1:
    def run(plan: Mapping[str, Any], run_root: Path) -> Mapping[str, Any]:
        execution = execute_campaign(
            plan,
            run_root=run_root,
            provider_registry=provider_registry,
            qualification_mode=True,
        )
        plan_execution = _mapping(plan.get("execution"), name="plan execution")
        if (
            execution.get("storage_wait") is True
            or execution.get("status") == "WAIT_STORAGE"
            or isinstance(execution.get("storage_wait_receipt_path"), str)
        ):
            # WAIT_STORAGE is a normal fail-closed lifecycle boundary.  Do not
            # retry the same attempt or inspect pending batch manifests: the
            # batch result is intentionally absent until storage admission is
            # available and the exact attempt can be resumed.
            return execution
        if (
            plan_execution.get("execution_route_identity")
            == "tick_native_execution_route_v1"
            and execution.get("status") == "interrupted"
        ):
            execution = execute_campaign(
                plan,
                run_root=run_root,
                provider_registry=provider_registry,
                qualification_mode=True,
            )
        exact_comparisons: list[dict[str, Any]] = []
        comparisons = _mapping(plan.get("comparisons", {}), name="plan comparisons")
        for branch in _rows(
            comparisons.get("branches", []), name="plan comparison branches"
        ):
            path = run_root / "comparisons" / str(branch["branch_id"]) / "manifest.json"
            manifest = _mapping(
                json.loads(path.read_text(encoding="utf-8")),
                name="comparison manifest",
            )
            comparison = _mapping(manifest.get("comparison"), name="comparison receipt")
            exact_comparisons.append(
                {
                    "comparison_id": str(branch["branch_id"]),
                    "match": comparison.get("match") is True,
                    "first_difference": comparison.get("first_difference"),
                    "execution_route_sha256": comparison.get("execution_route_sha256"),
                    "compared_surfaces": comparison.get("compared_surfaces", []),
                }
            )
        observed_transitions: set[str] = set()
        terminal_by_cell: dict[str, str] = {}
        route_obligation_evidence: list[dict[str, Any]] = []
        tick_native_evidence: list[dict[str, Any]] = []
        acs_async_evidence: list[dict[str, Any]] = []
        compact_science_async_evidence: list[dict[str, Any]] = []
        for batch in _rows(plan.get("physical_batches"), name="physical batches"):
            path = run_root / "batches" / str(batch["batch_id"]) / "manifest.json"
            manifest = _mapping(
                json.loads(path.read_text(encoding="utf-8")),
                name="batch manifest",
            )
            result = _mapping(manifest.get("result"), name="batch result")
            staging = result.get("acs_checkpoint_staging")
            if isinstance(staging, Mapping):
                acs_async_evidence.append(
                    {
                        "case_id": str(plan["route_qualification_case_id"]),
                        "batch_id": str(batch["batch_id"]),
                        "execution_route_sha256": plan.get("execution_route_sha256"),
                        "telemetry": dict(staging),
                    }
                )
            compact_publication = result.get("compact_science_async_publication")
            if isinstance(compact_publication, Mapping):
                compact_science_async_evidence.append(
                    {
                        "case_id": str(plan["route_qualification_case_id"]),
                        "batch_id": str(batch["batch_id"]),
                        "execution_route_sha256": plan.get("execution_route_sha256"),
                        "terminal_next_absolute_tick": result.get(
                            "next_absolute_tick"
                        ),
                        "telemetry": dict(compact_publication),
                    }
                )
            if (
                plan_execution.get("execution_route_identity")
                == "tick_native_execution_route_v1"
            ):
                tick_native_evidence.append(
                    _tick_native_evidence_row_v1(
                        plan_execution=plan_execution,
                        result=result,
                        batch_id=str(batch["batch_id"]),
                    )
                )
            sentinels = _mapping(
                result.get("transition_sentinels"),
                name="batch transition sentinels",
            )
            expected_cells = {str(cell_id) for cell_id in batch.get("cell_ids", [])}
            if set(sentinels) != expected_cells:
                raise RouteQualificationError(
                    "batch transition sentinels do not cover every expected cell"
                )
            for cell_id in sorted(expected_cells):
                rows = _rows(
                    sentinels[cell_id],
                    name=f"cell {cell_id} transition sentinels",
                )
                if not rows:
                    raise RouteQualificationError(
                        f"cell {cell_id!r} has no transition sentinels"
                    )
                for row in rows:
                    sentinel = RouteTransitionSentinelV1.from_mapping(row)
                    observed_transitions.add(sentinel.transition_name)
            # Partition equivalence is a trajectory-state claim.  Complete
            # Bundle/source-projection digests also bind batch-local overlay
            # and Chronicle audit history, which legitimately changes when
            # the same cells are repartitioned.
            raw_digests = result.get("mechanics_checkpoint_digests")
            if raw_digests is None:
                raw_digests = result.get("terminal_cell_digests")
            if raw_digests is None:
                checkpoint = manifest.get("checkpoint")
                raw_digests = (
                    checkpoint.get("mechanics_checkpoint_digests")
                    if isinstance(checkpoint, Mapping)
                    else None
                )
                if raw_digests is None and isinstance(checkpoint, Mapping):
                    raw_digests = checkpoint.get("terminal_cell_digests")
            digests = _mapping(
                raw_digests,
                name="batch terminal cell digests",
            )
            if set(digests) != expected_cells or not all(
                isinstance(value, str) and value for value in digests.values()
            ):
                raise RouteQualificationError(
                    "batch terminal digests do not cover every expected cell"
                )
            terminal_by_cell.update(
                {str(key): str(value) for key, value in digests.items()}
            )
            route_obligation_evidence.extend(
                _batch_route_obligation_evidence_v1(
                    plan=plan,
                    batch=batch,
                    manifest=manifest,
                )
            )
        ordered_terminal_digests = [
            terminal_by_cell[str(cell["cell_id"])]
            for cell in _rows(plan.get("cells"), name="plan cells")
        ]
        return {
            **dict(execution),
            "exact": execution.get("status") == "completed",
            "execution_route_sha256": execution.get("execution_route_sha256"),
            "transition_sentinels_exact": (
                bool(observed_transitions)
                and all(row["match"] for row in exact_comparisons)
            ),
            "observed_transition_names": sorted(observed_transitions),
            "terminal_cell_digests_by_cell_id": dict(sorted(terminal_by_cell.items())),
            "terminal_cell_digests_by_ordinal": ordered_terminal_digests,
            "exact_comparisons": exact_comparisons,
            "route_obligation_evidence": {
                "schema_version": "eiel_route_obligation_evidence_v1",
                "cells": route_obligation_evidence,
            },
            "tick_native_evidence": tick_native_evidence,
            "acs_async_evidence": acs_async_evidence,
            "compact_science_async_evidence": compact_science_async_evidence,
        }

    return run


def _tick_native_evidence_row_v1(
    *,
    plan_execution: Mapping[str, Any],
    result: Mapping[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    """Project tick-native runtime evidence without generation-era aliases."""

    checkpoint_catalog = result.get("checkpoint_catalog")
    restore_receipt = result.get("fresh_owner_restore_receipt")
    return {
        "batch_id": batch_id,
        "requested_terminal_tick": plan_execution.get("total_ticks"),
        "requested_checkpoint_ticks": list(
            plan_execution.get("checkpoint_ticks") or []
        ),
        "deliberate_restore_tick": plan_execution.get("deliberate_restore_tick"),
        "actual_terminal_tick": result.get("terminal_next_absolute_tick"),
        "actual_checkpoint_ticks": sorted(
            int(value)
            for value in (
                checkpoint_catalog.keys()
                if isinstance(checkpoint_catalog, Mapping)
                else ()
            )
        ),
        "fresh_owner_restore_receipt": (
            dict(restore_receipt) if isinstance(restore_receipt, Mapping) else None
        ),
    }


def _plan_cell_parameters_by_id(
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    cells = _rows(plan.get("cells"), name="plan cells")
    indexed: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        cell_id = cell.get("cell_id")
        parameters = cell.get("parameters")
        if not isinstance(cell_id, str) or not isinstance(parameters, Mapping):
            raise RouteQualificationError("plan cell parameters are incomplete")
        if cell_id in indexed:
            raise RouteQualificationError("plan cell identifiers are duplicated")
        indexed[cell_id] = parameters
    return indexed


def _nested_int(row: Mapping[str, Any], path: tuple[str, ...]) -> int | None:
    current: Any = row
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    if isinstance(current, bool) or not isinstance(current, int):
        return None
    return int(current)


def _generation_row_evidence(row: Mapping[str, Any]) -> dict[str, Any] | None:
    generation = row.get("generation")
    cell_id = row.get("cell_id")
    if (
        not isinstance(cell_id, str)
        or isinstance(generation, bool)
        or not isinstance(generation, int)
    ):
        return None
    population_alive = _nested_int(row, ("population", "alive"))
    if population_alive is None:
        population_alive = _nested_int(row, ("metrics", "population.alive"))
    if population_alive is None:
        population_alive = _nested_int(row, ("alive",))
    ordinary_birth_count = _nested_int(row, ("reproduction", "ordinary_birth_count"))
    if ordinary_birth_count is None:
        ordinary_birth_count = _nested_int(row, ("ordinary_birth_count",))
    refill_child_count = _nested_int(row, ("reproduction", "refill_child_count"))
    if refill_child_count is None:
        refill_child_count = _nested_int(row, ("turnover_refill_child_count",))
    if refill_child_count is None:
        refill_child_count = _nested_int(row, ("refill_child_count",))
    if ordinary_birth_count is not None and refill_child_count is not None:
        successful_births = ordinary_birth_count + refill_child_count
    else:
        successful_births = _nested_int(row, ("reproduction", "successful_births"))
        if successful_births is None:
            successful_births = _nested_int(row, ("metrics", "population.births"))
        if successful_births is None:
            successful_births = _nested_int(row, ("successful_births",))
    return {
        "cell_id": cell_id,
        "generation": int(generation),
        "population_alive": population_alive,
        "successful_births": successful_births,
        "ordinary_birth_count": ordinary_birth_count,
        "refill_child_count": refill_child_count,
    }


def _batch_route_obligation_evidence_v1(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    parameters_by_cell = _plan_cell_parameters_by_id(plan)
    raw_cell_ids = batch.get("cell_ids")
    if (
        isinstance(raw_cell_ids, (str, bytes))
        or not isinstance(raw_cell_ids, Sequence)
        or not all(isinstance(cell_id, str) for cell_id in raw_cell_ids)
    ):
        raise RouteQualificationError("physical batch cell_ids are invalid")
    rows_by_cell: dict[str, list[dict[str, Any]]] = {
        str(cell_id): [] for cell_id in raw_cell_ids
    }
    raw_rows = manifest.get("metrics_rows", [])
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise RouteQualificationError("batch metrics_rows must be an array")
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise RouteQualificationError("batch metrics_rows must contain objects")
        nested = raw.get("generation_metrics")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            parent_cell_id = raw.get("cell_id")
            for nested_row in nested:
                if not isinstance(nested_row, Mapping):
                    raise RouteQualificationError(
                        "batch generation_metrics must contain objects"
                    )
                combined = {
                    **dict(nested_row),
                    "cell_id": parent_cell_id,
                }
                evidence = _generation_row_evidence(combined)
                if evidence is None:
                    continue
                cell_id = evidence["cell_id"]
                if cell_id in rows_by_cell:
                    rows_by_cell[cell_id].append(evidence)
        else:
            evidence = _generation_row_evidence(raw)
            if evidence is None:
                continue
            cell_id = evidence["cell_id"]
            if cell_id in rows_by_cell:
                rows_by_cell[cell_id].append(evidence)
    evidence_rows: list[dict[str, Any]] = []
    for cell_id in raw_cell_ids:
        parameters = parameters_by_cell[str(cell_id)]
        target_population = parameters.get("patch.initial_population")
        resident_capacity = parameters.get("patch.capacity")
        evidence_rows.append(
            {
                "schema_version": "eiel_route_obligation_cell_evidence_v1",
                "cell_id": str(cell_id),
                "source": "batch_manifest_metrics_rows_and_compiled_cell_parameters",
                "profile_ordinary_births_enabled": parameters.get(
                    "production_lifecycle.ordinary_births_enabled"
                ),
                "profile_turnover_enabled": parameters.get(
                    "production_lifecycle.turnover_enabled"
                ),
                "turnover_cull_fraction": parameters.get(
                    "production_lifecycle.turnover_cull_fraction"
                ),
                "turnover_parent_fraction": parameters.get(
                    "production_lifecycle.turnover_parent_fraction"
                ),
                "target_population": (
                    int(target_population) if type(target_population) is int else None
                ),
                "resident_capacity": (
                    int(resident_capacity) if type(resident_capacity) is int else None
                ),
                "generation_rows": sorted(
                    rows_by_cell[str(cell_id)],
                    key=lambda row: int(row["generation"]),
                ),
            }
        )
    return tuple(evidence_rows)


def _validate_case_evidence(
    cases: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    expected = {str(case["id"]) for case in cases}
    route_by_case = {
        str(plan["route_qualification_case_id"]): str(plan["execution_route_sha256"])
        for plan in plans
    }
    expected_comparisons = {
        str(plan["route_qualification_case_id"]): {
            str(branch["branch_id"])
            for branch in _rows(
                _mapping(plan.get("comparisons"), name="plan comparisons").get(
                    "branches", []
                ),
                name="plan comparison branches",
            )
        }
        for plan in plans
    }
    by_id = {str(row.get("case_id")): dict(row) for row in results}
    if set(by_id) != expected:
        raise RouteQualificationError(
            "route qualification did not return every generated case exactly once"
        )
    comparisons: list[dict[str, Any]] = []
    for case_id in sorted(expected):
        row = by_id[case_id]
        if (
            row.get("status") != "completed"
            or row.get("exact") is not True
            or row.get("transition_sentinels_exact") is not True
            or row.get("execution_route_sha256") != route_by_case[case_id]
        ):
            raise RouteQualificationError(
                f"route qualification case {case_id!r} is incomplete or non-exact"
            )
        raw_comparisons = _rows(
            row.get("exact_comparisons", []),
            name=f"case {case_id} exact_comparisons",
        )
        comparison_ids = {
            str(comparison.get("comparison_id")) for comparison in raw_comparisons
        }
        if comparison_ids != expected_comparisons[case_id]:
            raise RouteQualificationError(
                f"route qualification case {case_id!r} lacks exact generated comparisons"
            )
        for comparison in raw_comparisons:
            if (
                comparison.get("match") is not True
                or comparison.get("first_difference") is not None
                or comparison.get("execution_route_sha256") != route_by_case[case_id]
            ):
                raise RouteQualificationError(
                    f"route qualification case {case_id!r} has a comparison mismatch"
                )
            comparisons.append({"case_id": case_id, **comparison})
    if {
        "partition-default",
        "partition-alternate",
    }.issubset(expected):
        default = by_id["partition-default"].get("terminal_cell_digests_by_cell_id")
        alternate = by_id["partition-alternate"].get("terminal_cell_digests_by_cell_id")
        if (
            not isinstance(default, Mapping)
            or not isinstance(alternate, Mapping)
            or not default
            or default != alternate
        ):
            raise RouteQualificationError(
                "alternate batch partition terminal state is not exact"
            )
    if {"fresh-repeat-a", "fresh-repeat-b"}.issubset(expected):
        first = by_id["fresh-repeat-a"].get("terminal_cell_digests_by_cell_id")
        repeat = by_id["fresh-repeat-b"].get("terminal_cell_digests_by_cell_id")
        if (
            not isinstance(first, Mapping)
            or not isinstance(repeat, Mapping)
            or not first
            or first != repeat
        ):
            raise RouteQualificationError("fresh repeat terminal state is not exact")
    for first_id, second_id, label in (
        ("uninterrupted", "disk-continuation", "checkpoint continuation"),
        ("reporting-disabled", "reporting-enabled", "reporting parity"),
    ):
        if {first_id, second_id}.issubset(expected):
            first = by_id[first_id].get("terminal_cell_digests_by_cell_id")
            second = by_id[second_id].get("terminal_cell_digests_by_cell_id")
            if (
                not isinstance(first, Mapping)
                or not isinstance(second, Mapping)
                or not first
                or first != second
            ):
                raise RouteQualificationError(
                    f"tick-native {label} terminal state is not exact"
                )
    return tuple(by_id[key] for key in sorted(by_id)), tuple(comparisons)


def _route_obligation_cells(
    row: Mapping[str, Any], *, case_id: str
) -> dict[str, Mapping[str, Any]]:
    evidence = _mapping(
        row.get("route_obligation_evidence"),
        name=f"case {case_id} route_obligation_evidence",
    )
    if evidence.get("schema_version") != "eiel_route_obligation_evidence_v1":
        raise RouteQualificationError(
            f"case {case_id!r} route obligation evidence has an invalid schema"
        )
    cells = _rows(
        evidence.get("cells"),
        name=f"case {case_id} route obligation evidence cells",
    )
    by_cell: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in by_cell:
            raise RouteQualificationError(
                f"case {case_id!r} route obligation evidence has duplicate cells"
            )
        by_cell[cell_id] = cell
    return by_cell


def _validate_semantic_route_obligations_v1(
    *,
    covered: set[str],
    plans: Sequence[Mapping[str, Any]],
    exact_cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    obligations = covered & set(_SEMANTIC_ROUTE_OBLIGATIONS_V1)
    if not obligations:
        return ()
    plan_by_case = {str(plan["route_qualification_case_id"]): plan for plan in plans}
    reporting_evidence_available = any(
        _mapping(plan.get("execution"), name="plan execution").get("reporting_enabled")
        is True
        for plan in plans
    )
    rows: list[dict[str, Any]] = []
    turnover_refill_proven = False
    for case in exact_cases:
        case_id = str(case.get("case_id"))
        plan = plan_by_case.get(case_id)
        if plan is None:
            raise RouteQualificationError(
                f"case {case_id!r} has no matching bounded plan"
            )
        execution = _mapping(plan.get("execution"), name="plan execution")
        if reporting_evidence_available and execution.get("reporting_enabled") is False:
            # Reporting-disabled runs deliberately retain no native per-generation
            # rows. Their state equivalence is checked by the reporting-parity
            # comparison; exact lifecycle evidence is supplied by the paired
            # reporting-enabled route case.
            continue
        cell_evidence = _route_obligation_cells(case, case_id=case_id)
        plan_parameters = _plan_cell_parameters_by_id(plan)
        if set(cell_evidence) != set(plan_parameters):
            raise RouteQualificationError(
                f"case {case_id!r} route obligation evidence does not cover every cell"
            )
        for cell_id, parameters in sorted(plan_parameters.items()):
            evidence = cell_evidence[cell_id]
            generation_rows = _rows(
                evidence.get("generation_rows"),
                name=f"case {case_id} cell {cell_id} generation_rows",
            )
            boundary_rows = [
                row
                for row in generation_rows
                if _strict_int(row.get("generation"), name="generation") > 0
            ]
            if not boundary_rows:
                raise RouteQualificationError(
                    f"case {case_id!r} cell {cell_id!r} lacks generation-boundary evidence"
                )
            ordinary_enabled = parameters.get(
                "production_lifecycle.ordinary_births_enabled"
            )
            turnover_enabled = parameters.get("production_lifecycle.turnover_enabled")
            target_population = _strict_int(
                parameters.get("patch.initial_population"),
                name="patch.initial_population",
                minimum=1,
            )
            _strict_int(
                parameters.get("patch.capacity"),
                name="patch.capacity",
                minimum=target_population,
            )
            expected_refill = target_population // 2
            if "ordinary_birth_disabled" in obligations:
                if (
                    ordinary_enabled is not False
                    or evidence.get("profile_ordinary_births_enabled") is not False
                ):
                    raise RouteQualificationError(
                        "ordinary_birth_disabled requires explicit disabled lifecycle control"
                    )
                for generation_row in boundary_rows:
                    if generation_row.get("ordinary_birth_count") != 0:
                        raise RouteQualificationError(
                            "ordinary_birth_disabled cannot be proven without exact zero ordinary-birth counts"
                        )
            if "turnover_refill" in obligations:
                if (
                    turnover_enabled is not True
                    or evidence.get("profile_turnover_enabled") is not True
                    or evidence.get("turnover_cull_fraction") != 0.5
                    or evidence.get("turnover_parent_fraction") != 1.0
                ):
                    raise RouteQualificationError(
                        "turnover_refill requires exact turnover lifecycle control"
                    )
                if any(
                    row.get("population_alive") == target_population
                    and row.get("refill_child_count") == expected_refill
                    and row.get("successful_births") == expected_refill
                    for row in boundary_rows
                ):
                    turnover_refill_proven = True
            rows.append(
                {
                    "case_id": case_id,
                    "cell_id": cell_id,
                    "covered_obligations": sorted(obligations),
                    "boundary_generation_count": len(boundary_rows),
                }
            )
    if "turnover_refill" in obligations and not turnover_refill_proven:
        raise RouteQualificationError(
            "turnover_refill lacks exact generation-boundary refill evidence"
        )
    return tuple(rows)


def _validate_acs_async_route_obligations_v1(
    *,
    covered: set[str],
    exact_cases: Sequence[Mapping[str, Any]],
    execution_route_sha256: str,
    expected_case_batches: set[tuple[str, str]],
    immutable_snapshot_slots: int,
) -> tuple[dict[str, Any], ...]:
    """Validate ordinary staging without promoting it to behavioral proof."""

    obligations = covered & set(_ACS_ASYNC_ROUTE_OBLIGATIONS_V1)
    if not obligations:
        return ()
    if type(immutable_snapshot_slots) is not int or immutable_snapshot_slots not in {
        1,
        2,
        3,
    }:
        raise RouteQualificationError(
            "ACS route evidence requires one, two, or three immutable slots"
        )
    evidence: list[tuple[str, Mapping[str, Any]]] = []
    for case in exact_cases:
        enclosing_case_id = case.get("case_id")
        if not isinstance(enclosing_case_id, str) or not enclosing_case_id:
            raise RouteQualificationError(
                "ACS asynchronous route evidence has no enclosing case identity"
            )
        evidence.extend(
            (enclosing_case_id, row)
            for row in _rows(
                case.get("acs_async_evidence", []),
                name="ACS asynchronous route evidence",
            )
        )
    if not evidence:
        raise RouteQualificationError(
            "asynchronous checkpoint route qualification has no staging evidence"
        )
    case_route_hashes = {
        str(case["case_id"]): case.get("execution_route_sha256")
        for case in exact_cases
        if isinstance(case.get("case_id"), str)
        and isinstance(case.get("execution_route_sha256"), str)
    }
    validated: list[dict[str, Any]] = []
    observed_case_batches: set[tuple[str, str]] = set()
    for enclosing_case_id, row in evidence:
        telemetry = _mapping(row.get("telemetry"), name="ACS staging telemetry")
        case_id = row.get("case_id")
        batch_id = row.get("batch_id")
        route_sha256 = row.get("execution_route_sha256")
        slots = telemetry.get("slots")
        high_water = telemetry.get("queue_occupancy_high_water")
        compute = telemetry.get("compute_frontier")
        durable = telemetry.get("durable_checkpoint_frontier")
        blocked_time_ns = telemetry.get("producer_blocked_time_ns")
        telemetry_schema = telemetry.get("schema_version")
        legacy_telemetry = telemetry_schema == "eiel_acs_checkpoint_staging_v1"
        detached_telemetry = (
            telemetry_schema == "eiel_acs_native_detached_slots_v1"
            and telemetry.get("capture_mode") == "detached_file_backed_native_v1"
            and telemetry.get("writer_count") == 1
            and telemetry.get("worker_accepts_cuda_objects") is False
            and telemetry.get("worker_accepts_callbacks") is False
            and telemetry.get("capture_pool_count") == 1
            and telemetry.get("capture_pool_model")
            == "one_shared_two_slab_pool_not_per_slot"
        )
        if "acs2_host_only_capture" in obligations and not detached_telemetry:
            raise RouteQualificationError(
                "ACS2 host-only capture obligation lacks detached tensor-free "
                "worker telemetry"
            )
        if (
            not (legacy_telemetry or detached_telemetry)
            or not isinstance(case_id, str)
            or not case_id
            or not isinstance(batch_id, str)
            or not batch_id
            or type(slots) is not int
            or slots != immutable_snapshot_slots
            or type(high_water) is not int
            or not 1 <= high_water <= slots
            or type(compute) is not int
            or type(durable) is not int
            or (
                blocked_time_ns is not None
                and (type(blocked_time_ns) is not int or blocked_time_ns < 0)
            )
            or compute < 0
            or durable != compute
            or telemetry.get("frontier_lag") != 0
            or telemetry.get("failure_latched") is not False
            or telemetry.get("closed") is not True
            or telemetry.get("writer_alive") is not False
        ):
            raise RouteQualificationError(
                "asynchronous checkpoint staging telemetry is not durable, "
                "closed, and writer-settled"
            )
        case_batch = (case_id, batch_id)
        expected_case_route = case_route_hashes.get(
            enclosing_case_id, execution_route_sha256
        )
        if route_sha256 != expected_case_route:
            raise RouteQualificationError(
                "asynchronous checkpoint staging evidence does not match the "
                "current execution route for its enclosing case"
            )
        if case_id != enclosing_case_id:
            raise RouteQualificationError(
                "asynchronous checkpoint staging evidence does not match its "
                "enclosing route qualification case"
            )
        if case_batch in observed_case_batches:
            raise RouteQualificationError(
                "asynchronous checkpoint staging evidence contains a duplicate "
                "case/batch row"
            )
        observed_case_batches.add(case_batch)

        validated.append(
            {
                "case_id": case_id,
                "batch_id": batch_id,
                "execution_route_sha256": route_sha256,
                "slots": slots,
                "queue_occupancy_high_water": high_water,
                "producer_blocked_time_ns": blocked_time_ns,
                "compute_frontier": compute,
                "durable_checkpoint_frontier": durable,
                "writer_alive": False,
                "persistence_mode": (
                    "detached_file_backed_native_v1"
                    if detached_telemetry
                    else "legacy_callable_snapshot_v1"
                ),
                "worker_accepts_cuda_objects": (
                    False if detached_telemetry else None
                ),
                "worker_accepts_callbacks": (
                    False if detached_telemetry else None
                ),
                "exact": True,
            }
        )
    if observed_case_batches != expected_case_batches:
        missing = sorted(expected_case_batches - observed_case_batches)
        unexpected = sorted(observed_case_batches - expected_case_batches)
        raise RouteQualificationError(
            "asynchronous checkpoint staging evidence does not exactly cover "
            f"the expected case/batch rows (missing={missing!r}, "
            f"unexpected={unexpected!r})"
        )
    unqualified = obligations & set(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
    qualified = obligations - unqualified
    return (
        {
            "schema_version": "eiel_acs_async_route_obligation_evidence_v1",
            "covered_obligations": sorted(qualified),
            "missing_obligations": sorted(unqualified),
            "behavioral_evidence_authority": (
                "ordinary_success_telemetry_is_not_fault_evidence_v1"
            ),
            "batches": validated,
            "exact": not unqualified,
        },
    )


def _compact_science_async_component_v1(
    route_contract: ExecutionRouteContractV1,
) -> Mapping[str, Any]:
    """Return the one declared compact-publication component, fail-closed."""

    components = [
        component
        for component in route_contract.components
        if component.get("component_id")
        == "cuda_full_evolution_v2_compact_science_async_publication"
    ]
    if len(components) != 1:
        raise RouteQualificationError(
            "compact science async obligations require exactly one compact "
            "publication route component"
        )
    component = components[0]
    workers = component.get("workers")
    pending_slots = component.get("pending_slots")
    if (
        type(workers) is not int
        or workers not in {1, 2}
        or type(pending_slots) is not int
        or pending_slots not in {1, 2}
        or workers > pending_slots
        or component.get("worker_boundary")
        != "immutable_tensor_free_host_capsule_v1"
        or component.get("worker_accepts_cuda_objects") is not False
        or component.get("worker_accepts_runtime_owner") is not False
        or component.get("fifo_publication") is not True
        or component.get("checkpoint_frontier_requires_durable_prefix") is not True
    ):
        raise RouteQualificationError(
            "compact science async route component does not declare the bounded "
            "tensor-free FIFO publication contract"
        )
    return component


def _validate_compact_science_async_route_obligations_v1(
    *,
    covered: set[str],
    exact_cases: Sequence[Mapping[str, Any]],
    route_contract: ExecutionRouteContractV1,
    execution_route_sha256: str,
    expected_case_batches: set[tuple[str, str]],
) -> tuple[dict[str, Any], ...]:
    """Project exact terminal compact-publication facts without fault claims.

    Ordinary completed batches establish only host-only capture and a drained
    durable absolute-tick prefix.  Saturation and injected-failure behavior
    deliberately remain unqualified until separately focused fault evidence is
    introduced and validated.
    """

    obligations = covered & set(_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1)
    if not obligations:
        return ()
    component = _compact_science_async_component_v1(route_contract)
    evidence: list[tuple[str, Mapping[str, Any]]] = []
    for case in exact_cases:
        enclosing_case_id = case.get("case_id")
        if not isinstance(enclosing_case_id, str) or not enclosing_case_id:
            raise RouteQualificationError(
                "compact science async route evidence has no enclosing case identity"
            )
        evidence.extend(
            (enclosing_case_id, row)
            for row in _rows(
                case.get("compact_science_async_evidence", []),
                name="compact science asynchronous route evidence",
            )
        )
    if not evidence:
        raise RouteQualificationError(
            "compact science asynchronous route qualification has no publication evidence"
        )
    case_route_hashes = {
        str(case["case_id"]): case.get("execution_route_sha256")
        for case in exact_cases
        if isinstance(case.get("case_id"), str)
        and isinstance(case.get("execution_route_sha256"), str)
    }
    validated: list[dict[str, Any]] = []
    observed_case_batches: set[tuple[str, str]] = set()
    for enclosing_case_id, row in evidence:
        telemetry = _mapping(
            row.get("telemetry"), name="compact science asynchronous telemetry"
        )
        case_id = row.get("case_id")
        batch_id = row.get("batch_id")
        route_sha256 = row.get("execution_route_sha256")
        terminal_tick = row.get("terminal_next_absolute_tick")
        workers = telemetry.get("workers")
        pending_slots = telemetry.get("pending_slots")
        compute = telemetry.get("compute_frontier")
        durable = telemetry.get("durable_frontier")
        submitted = telemetry.get("submitted_jobs")
        completed = telemetry.get("completed_jobs")
        failed = telemetry.get("failed_jobs")
        pending = telemetry.get("pending_jobs")
        high_water = telemetry.get("occupancy_high_water")
        captured_host_bytes = telemetry.get("captured_host_bytes")
        queued_cuda_references = telemetry.get("queued_cuda_references")
        expected_case_route = case_route_hashes.get(
            enclosing_case_id, execution_route_sha256
        )
        if (
            telemetry.get("schema_version")
            != "eiel_compact_science_async_publication_v1"
            or not isinstance(case_id, str)
            or not case_id
            or not isinstance(batch_id, str)
            or not batch_id
            or type(terminal_tick) is not int
            or terminal_tick < 0
            or route_sha256 != expected_case_route
            or case_id != enclosing_case_id
            or workers != component.get("workers")
            or pending_slots != component.get("pending_slots")
            or type(compute) is not int
            or type(durable) is not int
            or compute != terminal_tick
            or durable != compute
            or type(submitted) is not int
            or type(completed) is not int
            or submitted < 1
            or completed != submitted
            or type(failed) is not int
            or failed != 0
            or type(pending) is not int
            or pending != 0
            or type(high_water) is not int
            or not 1 <= high_water <= pending_slots
            or type(captured_host_bytes) is not int
            or captured_host_bytes != 0
            or queued_cuda_references != 0
            or telemetry.get("fifo_publication") is not True
            or telemetry.get("closed") is not True
            or telemetry.get("aborted") is not False
        ):
            raise RouteQualificationError(
                "compact science async telemetry is not an exact closed, FIFO, "
                "tensor-free durable absolute-tick prefix"
            )
        case_batch = (case_id, batch_id)
        if case_batch in observed_case_batches:
            raise RouteQualificationError(
                "compact science async evidence contains a duplicate case/batch row"
            )
        observed_case_batches.add(case_batch)
        validated.append(
            {
                "case_id": case_id,
                "batch_id": batch_id,
                "execution_route_sha256": route_sha256,
                "terminal_next_absolute_tick": terminal_tick,
                "workers": workers,
                "pending_slots": pending_slots,
                "compute_frontier": compute,
                "durable_frontier": durable,
                "submitted_jobs": submitted,
                "completed_jobs": completed,
                "queued_cuda_references": 0,
                "fifo_publication": True,
                "exact": True,
            }
        )
    if observed_case_batches != expected_case_batches:
        missing = sorted(expected_case_batches - observed_case_batches)
        unexpected = sorted(observed_case_batches - expected_case_batches)
        raise RouteQualificationError(
            "compact science async evidence does not exactly cover the expected "
            f"case/batch rows (missing={missing!r}, unexpected={unexpected!r})"
        )
    unqualified = obligations & set(_COMPACT_SCIENCE_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
    qualified = obligations - unqualified
    return (
        {
            "schema_version": "eiel_compact_science_async_route_obligation_evidence_v1",
            "covered_obligations": sorted(qualified),
            "missing_obligations": sorted(unqualified),
            "behavioral_evidence_authority": (
                "ordinary_success_telemetry_is_not_backpressure_or_failure_evidence_v1"
            ),
            "batches": validated,
            "exact": not unqualified,
        },
    )


def _acs_staging_source_sha256_v1() -> str:
    return acs_checkpoint_staging_source_sha256_v1()


def _acs_immutable_snapshot_slots_v1(
    route_contract: ExecutionRouteContractV1,
) -> int:
    components = [
        component
        for component in route_contract.components
        if component.get("component_id") == "acs_checkpoint_persistence"
    ]
    if len(components) != 1:
        raise RouteQualificationError(
            "ACS route evidence requires one checkpoint persistence component"
        )
    slots = components[0].get("immutable_snapshot_slots")
    if type(slots) is not int or slots not in {1, 2, 3}:
        raise RouteQualificationError(
            "ACS route evidence requires one, two, or three immutable slots"
        )
    return slots


def _acs_persistence_route_v1(
    route_contract: ExecutionRouteContractV1,
) -> str | None:
    components = [
        component
        for component in route_contract.components
        if component.get("component_id") == "acs_checkpoint_persistence"
    ]
    if len(components) != 1:
        raise RouteQualificationError(
            "ACS route evidence requires one checkpoint persistence component"
        )
    component = components[0]
    persistence_route = component.get("persistence_route")
    contract_version = component.get("contract_version")
    if (
        isinstance(persistence_route, str)
        and isinstance(contract_version, str)
        and persistence_route != contract_version
    ):
        raise RouteQualificationError(
            "ACS checkpoint persistence route and contract version disagree"
        )
    if isinstance(persistence_route, str):
        return persistence_route
    return contract_version if isinstance(contract_version, str) else None


def _execute_acs_focused_route_probe_v1(
    execution_route_sha256: str,
    obligations: frozenset[str],
    immutable_snapshot_slots: int,
) -> dict[str, Any]:
    """Exercise staging backpressure and failure latching without GPU dispatch."""

    from .acs_checkpoint_staging import (
        ACSCheckpointStaging,
        ACSCheckpointStagingError,
    )

    requested = obligations & _ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1
    if type(immutable_snapshot_slots) is not int or immutable_snapshot_slots not in {
        1,
        2,
        3,
    }:
        raise RouteQualificationError(
            "ACS focused probe requires one, two, or three immutable slots"
        )
    result: dict[str, Any] = {
        "schema_version": "eiel_acs_checkpoint_focused_route_evidence_v1",
        "source_kind": "current_process_cpu_staging_probe_v1",
        "implementation": (
            "engine.campaigns.acs_checkpoint_staging.ACSCheckpointStaging"
        ),
        "implementation_source_sha256": _acs_staging_source_sha256_v1(),
        "execution_route_sha256": execution_route_sha256,
        "target_immutable_snapshot_slots": immutable_snapshot_slots,
        "covered_obligations": [],
        "missing_obligations": sorted(requested),
        "exact": False,
    }
    covered: set[str] = set()

    if "async_checkpoint_backpressure" in requested:
        writer_started = threading.Event()
        release_writer = threading.Event()
        submit_finished = threading.Event()
        submit_errors: list[BaseException] = []
        blocked_tokens: list[Any] = []
        persist_order: list[str] = []
        staging = ACSCheckpointStaging[str](slots=immutable_snapshot_slots)

        def persist_first() -> str:
            persist_order.append("first")
            writer_started.set()
            if not release_writer.wait(2):
                raise TimeoutError("ACS focused backpressure writer release timed out")
            return "first"

        def persist_queued(frontier: int) -> str:
            value = f"queued-{frontier}"
            persist_order.append(value)
            return value

        def persist_blocked() -> str:
            persist_order.append("blocked")
            return "blocked"

        def submit_blocked() -> None:
            try:
                blocked_tokens.append(
                    staging.submit(
                        compute_frontier=immutable_snapshot_slots + 1,
                        captured_bytes=1,
                        persist=persist_blocked,
                    )
                )
            except BaseException as exc:
                submit_errors.append(exc)
            finally:
                submit_finished.set()

        submit_thread: threading.Thread | None = None
        occupied_tokens: list[Any] = []
        observed_results: list[str] = []
        try:
            first = staging.submit(
                compute_frontier=1,
                captured_bytes=1,
                persist=persist_first,
            )
            if not writer_started.wait(2):
                raise RouteQualificationError(
                    "ACS focused backpressure writer did not start"
                )
            occupied_tokens.append(first)
            for frontier in range(2, immutable_snapshot_slots + 1):
                occupied_tokens.append(
                    staging.submit(
                        compute_frontier=frontier,
                        captured_bytes=1,
                        persist=lambda frontier=frontier: persist_queued(frontier),
                    )
                )
            occupied = staging.telemetry()
            all_slots_occupied = (
                occupied.get("queue_occupancy") == immutable_snapshot_slots
                and occupied.get("queue_occupancy_high_water")
                == immutable_snapshot_slots
            )
            submit_thread = threading.Thread(
                target=submit_blocked,
                name="eiel-acs-focused-backpressure-producer",
                daemon=True,
            )
            submit_thread.start()
            blocked_before_release = not submit_finished.wait(0.05)
            release_writer.set()
            first_result = staging.result(first)
            if first_result != "first":
                raise RouteQualificationError(
                    "ACS focused backpressure first result changed"
                )
            observed_results.append(first_result)
            staging.acknowledge_durable(first)
            if not submit_finished.wait(2):
                raise RouteQualificationError(
                    "ACS focused backpressure producer did not resume"
                )
            submit_thread.join(timeout=2)
            if submit_thread.is_alive() or submit_errors or len(blocked_tokens) != 1:
                raise RouteQualificationError(
                    "ACS focused backpressure producer did not settle exactly"
                )
            for frontier, token in enumerate(occupied_tokens[1:], start=2):
                value = staging.result(token)
                if value != f"queued-{frontier}":
                    raise RouteQualificationError(
                        "ACS focused backpressure queued result changed"
                    )
                observed_results.append(value)
                staging.acknowledge_durable(token)
            blocked = blocked_tokens[0]
            blocked_result = staging.result(blocked)
            if blocked_result != "blocked":
                raise RouteQualificationError(
                    "ACS focused backpressure blocked result changed"
                )
            observed_results.append(blocked_result)
            staging.acknowledge_durable(blocked)
            staging.close()
            telemetry = staging.telemetry()
            blocked_time_ns = telemetry.get("producer_blocked_time_ns")
            expected_order = [
                "first",
                *[
                    f"queued-{frontier}"
                    for frontier in range(2, immutable_snapshot_slots + 1)
                ],
                "blocked",
            ]
            exact = (
                blocked_before_release
                and type(blocked_time_ns) is int
                and blocked_time_ns > 0
                and all_slots_occupied
                and telemetry.get("slots") == immutable_snapshot_slots
                and telemetry.get("queue_occupancy_high_water")
                == immutable_snapshot_slots
                and telemetry.get("compute_frontier") == immutable_snapshot_slots + 1
                and telemetry.get("durable_checkpoint_frontier")
                == immutable_snapshot_slots + 1
                and telemetry.get("frontier_lag") == 0
                and telemetry.get("failure_latched") is False
                and telemetry.get("queue_occupancy") == 0
                and telemetry.get("slot_states") == ["FREE"] * immutable_snapshot_slots
                and telemetry.get("abandoned_snapshot_count") == 0
                and telemetry.get("aborted") is False
                and telemetry.get("closed") is True
                and telemetry.get("writer_alive") is False
                and persist_order == expected_order
                and observed_results == expected_order
            )
            if not exact:
                raise RouteQualificationError(
                    "ACS focused backpressure evidence is not exact"
                )
            result["backpressure"] = {
                "blocked_before_release": True,
                "post_release_submit_completed": True,
                "producer_thread_joined": True,
                "all_slots_occupied_before_block": True,
                "blocked_submission_compute_frontier": (immutable_snapshot_slots + 1),
                "fifo_results": observed_results,
                "fifo_persist_order": persist_order,
                "no_overwrite_or_drop": len(observed_results)
                == immutable_snapshot_slots + 1,
                "producer_blocked_time_ns": blocked_time_ns,
                "terminal_telemetry": telemetry,
                "exact": True,
            }
            covered.add("async_checkpoint_backpressure")
        finally:
            release_writer.set()
            staging.abort()
            if submit_thread is not None:
                submit_thread.join(timeout=2)
                if submit_thread.is_alive():
                    raise RouteQualificationError(
                        "ACS focused backpressure producer did not join after abort"
                    )

    if "async_checkpoint_failure_latch" in requested:
        writer_started = threading.Event()
        release_writer = threading.Event()
        queued_callback_started = threading.Event()
        staging = ACSCheckpointStaging[str](slots=immutable_snapshot_slots)

        def inject_failure() -> str:
            writer_started.set()
            if not release_writer.wait(2):
                raise TimeoutError("ACS focused failure writer release timed out")
            raise OSError("eiel-acs-focused-injected-writer-failure")

        failure_observed = False
        post_latch_rejected = False
        tokens: list[Any] = []
        try:
            tokens.append(
                staging.submit(
                    compute_frontier=1,
                    captured_bytes=1,
                    persist=inject_failure,
                )
            )
            if not writer_started.wait(2):
                raise RouteQualificationError(
                    "ACS focused failure writer did not start"
                )
            for frontier in range(2, immutable_snapshot_slots + 1):
                tokens.append(
                    staging.submit(
                        compute_frontier=frontier,
                        captured_bytes=1,
                        persist=lambda: (
                            queued_callback_started.set(),
                            "must-not-run",
                        )[-1],
                    )
                )
            occupied = staging.telemetry()
            all_slots_occupied = (
                occupied.get("queue_occupancy") == immutable_snapshot_slots
                and occupied.get("queue_occupancy_high_water")
                == immutable_snapshot_slots
            )
            release_writer.set()
            try:
                staging.result(tokens[0])
            except ACSCheckpointStagingError as exc:
                failure_observed = (
                    isinstance(exc.__cause__, OSError)
                    and str(exc.__cause__) == "eiel-acs-focused-injected-writer-failure"
                )
            try:
                staging.submit(
                    compute_frontier=2,
                    captured_bytes=1,
                    persist=lambda: "unreachable",
                )
            except ACSCheckpointStagingError as exc:
                post_latch_rejected = "eiel-acs-focused-injected-writer-failure" in str(
                    exc
                )
        finally:
            release_writer.set()
            staging.abort()
        telemetry = staging.telemetry()
        all_tokens_inaccessible = True
        for token in tokens:
            for operation in (
                lambda token=token: staging.ready(token),
                lambda token=token: staging.result(token),
                lambda token=token: staging.acknowledge_durable(token),
            ):
                try:
                    operation()
                except ACSCheckpointStagingError:
                    continue
                all_tokens_inaccessible = False
        exact = (
            failure_observed
            and post_latch_rejected
            and all_slots_occupied
            and not queued_callback_started.is_set()
            and all_tokens_inaccessible
            and telemetry.get("slots") == immutable_snapshot_slots
            and telemetry.get("queue_occupancy_high_water") == immutable_snapshot_slots
            and telemetry.get("abandoned_snapshot_count") == immutable_snapshot_slots
            and telemetry.get("queue_occupancy") == 0
            and telemetry.get("slot_states") == ["FREE"] * immutable_snapshot_slots
            and telemetry.get("failure_latched") is True
            and telemetry.get("closed") is True
            and telemetry.get("aborted") is True
            and telemetry.get("writer_alive") is False
            and telemetry.get("durable_checkpoint_frontier") == -1
        )
        if not exact:
            raise RouteQualificationError(
                "ACS focused failure-latch evidence is not exact"
            )
        result["failure_latch"] = {
            "injected_failure": "eiel-acs-focused-injected-writer-failure",
            "failure_observed": True,
            "post_latch_capture_rejected": True,
            "abort_joined_writer": True,
            "all_slots_occupied_before_failure": True,
            "queued_token_count": immutable_snapshot_slots - 1,
            "queued_callbacks_never_started": True,
            "all_tokens_inaccessible_after_abort": True,
            "terminal_telemetry": telemetry,
            "exact": True,
        }
        covered.add("async_checkpoint_failure_latch")

    result["covered_obligations"] = sorted(covered)
    result["missing_obligations"] = sorted(requested - covered)
    result["exact"] = covered == requested
    return result


def _validate_acs_focused_route_evidence_v1(
    *,
    execution_route_sha256: str,
    requested: set[str],
    evidence: Mapping[str, Any] | None,
    immutable_snapshot_slots: int,
) -> tuple[set[str], dict[str, Any]]:
    focused = requested & set(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
    if not focused or evidence is None:
        return set(), {
            "schema_version": "eiel_acs_checkpoint_focused_route_evidence_v1",
            "execution_route_sha256": execution_route_sha256,
            "covered_obligations": [],
            "missing_obligations": sorted(focused),
            "exact": False,
        }
    row = dict(evidence)
    details = [
        row.get(key)
        for obligation, key in (
            ("async_checkpoint_backpressure", "backpressure"),
            ("async_checkpoint_failure_latch", "failure_latch"),
        )
        if obligation in focused
    ]
    slots_exact = row.get(
        "target_immutable_snapshot_slots"
    ) == immutable_snapshot_slots and all(
        isinstance(detail, Mapping)
        and isinstance(detail.get("terminal_telemetry"), Mapping)
        and detail["terminal_telemetry"].get("slots") == immutable_snapshot_slots
        and detail["terminal_telemetry"].get("queue_occupancy_high_water")
        == immutable_snapshot_slots
        for detail in details
    )
    if not slots_exact:
        return set(), {
            **row,
            "covered_obligations": [],
            "missing_obligations": sorted(focused),
            "exact": False,
        }
    covered, validated = validate_execution_route_acs_focused_evidence_v1(
        execution_route_sha256=execution_route_sha256,
        covered_obligations=focused,
        route_obligation_results=[row],
        immutable_snapshot_slots=immutable_snapshot_slots,
    )
    if covered != focused or validated is None:
        return set(), {
            **row,
            "covered_obligations": [],
            "missing_obligations": sorted(focused),
            "exact": False,
        }
    return covered, validated


def _validate_tick_native_route_obligations_v1(
    *,
    covered: set[str],
    exact_cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    obligations = covered & set(_TICK_NATIVE_ROUTE_OBLIGATIONS_V1)
    if not obligations:
        return ()
    case = next(
        (
            row
            for row in exact_cases
            if row.get("case_id") == "tick-native-partial-restore"
        ),
        None,
    )
    if not isinstance(case, Mapping):
        raise RouteQualificationError(
            "tick-native route qualification lacks its partial-restore case"
        )
    evidence = _rows(
        case.get("tick_native_evidence"),
        name="tick-native partial-restore evidence",
    )
    if not evidence:
        raise RouteQualificationError("tick-native qualification evidence is empty")
    requires_restore_projection = "first_post_restore_refill" in obligations
    observed_transitions = set(case.get("observed_transition_names") or ())
    for row in evidence:
        terminal = row.get("requested_terminal_tick")
        requested_checkpoints = row.get("requested_checkpoint_ticks")
        restore_tick = row.get("deliberate_restore_tick")
        actual_checkpoints = row.get("actual_checkpoint_ticks")
        restore = row.get("fresh_owner_restore_receipt")
        if (
            type(terminal) is not int
            or terminal != 161
            or terminal % 160 == 0
            or requested_checkpoints != [17, 160, 161]
            or type(restore_tick) is not int
            or restore_tick != 17
            or 17 % 160 == 0
            or row.get("actual_terminal_tick") != terminal
            or actual_checkpoints != requested_checkpoints
            or not isinstance(restore, Mapping)
            or restore.get("restore_next_tick") != restore_tick
            or restore.get("first_executed_absolute_tick") != restore_tick
            or restore.get("first_committed_next_tick") != restore_tick + 1
            or restore.get("off_by_one_check") != "PASS"
        ):
            raise RouteQualificationError(
                "tick-native partial checkpoint/restore evidence is not exact"
            )
        if requires_restore_projection:
            cell_ids = restore.get("cell_ids")
            parent_digests = restore.get(
                "parent_checkpoint_projection_digest_sha256s"
            )
            source_digests = restore.get("source_checkpoint_cell_digests")
            terminal_digests = restore.get(
                "restored_terminal_projection_digest_sha256s"
            )
            restored_cell_digests = restore.get("restored_terminal_cell_digests")
            reload_receipts = restore.get("reload_receipts")
            valid_cells = (
                isinstance(cell_ids, Sequence)
                and not isinstance(cell_ids, (str, bytes))
                and bool(cell_ids)
                and all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids)
                and len(set(cell_ids)) == len(cell_ids)
            )
            valid_digest_maps = (
                valid_cells
                and isinstance(parent_digests, Mapping)
                and isinstance(source_digests, Mapping)
                and isinstance(terminal_digests, Mapping)
                and isinstance(restored_cell_digests, Mapping)
                and set(parent_digests) == set(cell_ids)
                and source_digests == parent_digests
                and set(terminal_digests) == set(cell_ids)
                and restored_cell_digests == terminal_digests
                and all(
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                    for digest_map in (parent_digests, terminal_digests)
                    for value in digest_map.values()
                )
            )
            reload_by_cell = (
                {
                    receipt.get("cell_id"): receipt
                    for receipt in reload_receipts
                    if isinstance(receipt, Mapping)
                    and isinstance(receipt.get("cell_id"), str)
                }
                if isinstance(reload_receipts, Sequence)
                and not isinstance(reload_receipts, (str, bytes))
                else {}
            )
            reloads_exact = (
                valid_digest_maps
                and set(reload_by_cell) == set(cell_ids)
                and len(reload_by_cell) == len(reload_receipts)
                and all(
                    reload_by_cell[cell_id].get("source_projection_sha256")
                    == parent_digests[cell_id]
                    and reload_by_cell[cell_id].get("loaded_projection_sha256")
                    == parent_digests[cell_id]
                    for cell_id in cell_ids
                )
            )
            if (
                restore.get("schema_version")
                != "eiel_patch_fixed_fresh_owner_restore_receipt_v1"
                or restore.get("time_authority") != "absolute_tick_v1"
                or restore.get("projection_bundles_reloaded") is not True
                or restore.get("source_checkpoint_absolute_tick") != restore_tick
                or restore.get("source_checkpoint_next_absolute_tick")
                != restore_tick
                or "first_policy_cache_refill_after_restore"
                not in observed_transitions
                or not reloads_exact
            ):
                raise RouteQualificationError(
                    "tick-native fresh-owner policy-cache restore evidence is not exact"
                )
    return (
        {
            "schema_version": "eiel_tick_native_route_obligation_result_v1",
            "case_id": "tick-native-partial-restore",
            "covered_obligations": sorted(obligations),
            "requested_terminal_tick": 161,
            "requested_checkpoint_ticks": [17, 160, 161],
            "restore_tick": 17,
            "first_committed_next_tick": 18,
            "batch_count": len(evidence),
            "exact": True,
        },
    )


def _validated_route_qualification_inheritance_v1(
    *,
    campaign_plan: Mapping[str, Any],
    route_contract: ExecutionRouteContractV1,
) -> tuple[set[str], dict[str, Any] | None]:
    raw = campaign_plan.get("route_qualification_inheritance")
    if raw is None:
        return set(), None
    prior = _mapping(raw, name="route qualification inheritance")
    current_required = set(route_contract.required_obligations)
    inherited_required = current_required - set(_TICK_NATIVE_ROUTE_OBLIGATIONS_V1)
    # New compact publication behavior must be demonstrated on the current
    # route; a route receipt predating (or merely naming) this lane is not
    # evidence for its FIFO, host-only, or fault behavior.
    inherited_required.difference_update(_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1)
    covered = prior.get("covered_obligations")
    covered_set = (
        {value for value in covered if isinstance(value, str)}
        if isinstance(covered, Sequence) and not isinstance(covered, (str, bytes))
        else set()
    )
    qualification_key = campaign_plan.get("qualification_key")
    artifact_hashes = (
        qualification_key.get("artifact_hashes")
        if isinstance(qualification_key, Mapping)
        else None
    )
    current_artifact = (
        artifact_hashes.get("cuda_extension")
        if isinstance(artifact_hashes, Mapping)
        else None
    )
    prior_contract = prior.get("route_contract")
    prior_components = (
        prior_contract.get("components")
        if isinstance(prior_contract, Mapping)
        else None
    )
    current_components = [
        dict(component)
        for component in route_contract.payload()["components"]
        if component.get("component_id") != "absolute_tick_time_contract"
    ]
    prior_variability = prior.get("variability_envelope")
    prior_required = (
        prior_contract.get("required_obligations")
        if isinstance(prior_contract, Mapping)
        else None
    )
    current_variability = (
        qualification_key.get("variability_envelope")
        if isinstance(qualification_key, Mapping)
        else None
    )
    if (
        prior.get("schema_version") != EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
        or prior.get("authority_state") != "route_qualified"
        or prior.get("provider_id") != route_contract.provider_id
        or prior.get("profile_id") != route_contract.profile_id
        or prior.get("artifact_sha256") != current_artifact
        or covered_set != inherited_required
        or not isinstance(prior_contract, Mapping)
        or not isinstance(prior_required, list)
        or set(prior_required) != covered_set
        or canonical_sha256(prior_contract) != prior.get("execution_route_sha256")
        or canonical_sha256({"components": prior_components})
        != canonical_sha256({"components": current_components})
        or prior_variability != current_variability
    ):
        raise RouteQualificationError(
            "route qualification inheritance does not exactly bind the current "
            "artifact and unchanged route components"
        )
    return covered_set, {
        "schema_version": "eiel_execution_route_qualification_inheritance_v1",
        "source_execution_route_sha256": prior["execution_route_sha256"],
        "artifact_sha256": current_artifact,
        "inherited_obligations": sorted(covered_set),
        "excluded_new_obligations": sorted(
            set(_TICK_NATIVE_ROUTE_OBLIGATIONS_V1)
            | set(_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1)
        ),
        "component_contract_exact": True,
        "variability_envelope_exact": True,
    }


def execute_route_qualification_v1(
    *,
    campaign_plan: Mapping[str, Any],
    canary: Mapping[str, Any],
    run_root: Path,
    route_contract: ExecutionRouteContractV1,
    qualification_receipt_path: str | Path,
    preflight_runner: PreflightRunnerV1,
    provider_registry: Any | None = None,
    orchestration_runner: OrchestrationRunnerV1 | None = None,
    acs_focused_probe_runner: ACSFocusedProbeRunnerV1 | None = None,
    long_horizon_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute every generated case and publish authority only after exactness.

    Providers call this helper from their ``execute_route_qualification``
    method, supplying their contract, mechanic receipt path, and a
    qualification-aware preflight seam.  Until ordinary preflight has an
    explicit qualification mode, that seam must grant only bounded canary
    execution and must not grant scientific execution.
    """

    if canary.get("execution_route_sha256") != route_contract.execution_route_sha256:
        raise RouteQualificationError("canary route hash does not match provider route")
    covered = canary.get("missing_obligations")
    if (
        isinstance(covered, (str, bytes))
        or not isinstance(covered, Sequence)
        or not all(isinstance(item, str) for item in covered)
    ):
        raise RouteQualificationError("canary missing_obligations is invalid")
    sidecar = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path,
        route_contract.execution_route_sha256,
    )
    previous = read_execution_route_qualification_sidecar_v1(
        qualification_receipt_path,
        route_contract.execution_route_sha256,
    )
    previous_covered: set[str] = set()
    if previous is not None:
        previous_binding_exact = (
            previous.get("schema_version")
            == EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
            and previous.get("execution_route_sha256")
            == route_contract.execution_route_sha256
            and previous.get("provider_id") == route_contract.provider_id
            and previous.get("profile_id") == route_contract.profile_id
        )
        raw_previous_covered = previous.get("covered_obligations")
        if (
            previous_binding_exact
            and isinstance(raw_previous_covered, Sequence)
            and not isinstance(raw_previous_covered, (str, bytes))
        ):
            previous_covered = {
                item for item in raw_previous_covered if isinstance(item, str)
            }
            previous_covered.difference_update(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
            previous_covered.difference_update(
                _COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1
            )
    inherited_covered, inheritance_result = (
        _validated_route_qualification_inheritance_v1(
            campaign_plan=campaign_plan,
            route_contract=route_contract,
        )
    )
    inherited_covered.difference_update(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
    inherited_covered.difference_update(_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1)
    if inheritance_result is not None:
        inheritance_result["inherited_obligations"] = sorted(inherited_covered)
        inheritance_result["excluded_unproven_obligations"] = sorted(
            set(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
            | set(_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1)
        )
    cases = _canary_cases(canary)
    plans = _rebind_bounded_route_case_hashes_v1(
        derive_bounded_route_campaigns_v1(campaign_plan, canary),
        provider_registry=provider_registry,
    )
    runner = orchestration_runner
    if runner is None:
        if provider_registry is None:
            raise RouteQualificationError(
                "provider_registry is required for the default orchestration runner"
            )
        runner = _default_orchestration_runner(provider_registry)

    root = Path(run_root).expanduser().resolve()
    case_results: list[dict[str, Any]] = []
    for plan in plans:
        case_id = str(plan["route_qualification_case_id"])
        case_root = root / "cases" / case_id
        preflight = _mapping(preflight_runner(plan), name=f"case {case_id} preflight")
        ready_for_qualification = preflight.get("ready_for_route_qualification")
        if ready_for_qualification is None and orchestration_runner is not None:
            ready_for_qualification = preflight.get("ready_for_execution")
        if ready_for_qualification is not True or preflight.get("reasons") not in (
            None,
            [],
        ):
            raise RouteQualificationError(
                f"route qualification case {case_id!r} failed bounded preflight"
            )
        atomic_write_json(case_root / "preflight_receipt.json", preflight)
        result = _mapping(
            runner(plan, case_root), name=f"case {case_id} execution result"
        )
        if (
            result.get("storage_wait") is True
            or result.get("status") == "WAIT_STORAGE"
            or isinstance(result.get("storage_wait_receipt_path"), str)
        ):
            receipt_path = result.get("storage_wait_receipt_path")
            suffix = (
                f"; receipt={receipt_path}" if isinstance(receipt_path, str) else ""
            )
            raise RouteQualificationError(
                f"route qualification case {case_id!r} reached WAIT_STORAGE; "
                f"resume the same attempt after storage admission{suffix}"
            )
        result["case_id"] = case_id
        case_results.append(result)

    exact_cases, exact_comparisons = _validate_case_evidence(cases, plans, case_results)
    observed_transitions = {
        transition
        for row in exact_cases
        for transition in row.get("observed_transition_names", [])
        if isinstance(transition, str)
    }
    missing_transitions = sorted(
        {
            transition
            for obligation, transition in _OBLIGATION_SENTINEL_TRANSITION_V1.items()
            if obligation in covered and transition not in observed_transitions
        }
    )
    if missing_transitions:
        raise RouteQualificationError(
            "route qualification lacks required transition sentinels: "
            + ", ".join(missing_transitions)
        )
    requested_acs = set(covered) & set(_ACS_ASYNC_ROUTE_OBLIGATIONS_V1)
    requested_focused = requested_acs & set(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
    immutable_snapshot_slots = (
        _acs_immutable_snapshot_slots_v1(route_contract) if requested_acs else 0
    )
    focused_raw: Mapping[str, Any] | None = None
    if requested_focused:
        if acs_focused_probe_runner is not None:
            focused_raw = acs_focused_probe_runner(
                route_contract.execution_route_sha256,
                frozenset(requested_focused),
                immutable_snapshot_slots,
            )
        elif _acs_persistence_route_v1(route_contract) == (
            "acs_detached_file_backed_native_v1"
        ):
            from .acs_native_detached_slots_v1 import (
                execute_detached_slots_focused_probe_v1,
            )

            focused_raw = execute_detached_slots_focused_probe_v1(
                execution_route_sha256=route_contract.execution_route_sha256,
                obligations=frozenset(requested_focused),
                immutable_snapshot_slots=immutable_snapshot_slots,
            )
        else:
            focused_raw = _execute_acs_focused_route_probe_v1(
                route_contract.execution_route_sha256,
                frozenset(requested_focused),
                immutable_snapshot_slots,
            )
    focused_covered, focused_result = _validate_acs_focused_route_evidence_v1(
        execution_route_sha256=route_contract.execution_route_sha256,
        requested=requested_focused,
        evidence=focused_raw,
        immutable_snapshot_slots=immutable_snapshot_slots,
    )
    expected_acs_case_batches = {
        (
            str(plan["route_qualification_case_id"]),
            str(batch["batch_id"]),
        )
        for plan in plans
        for batch in _rows(
            plan.get("physical_batches"),
            name="route qualification physical batches",
        )
    }
    acs_async_results = _validate_acs_async_route_obligations_v1(
        covered=set(covered),
        exact_cases=exact_cases,
        execution_route_sha256=route_contract.execution_route_sha256,
        expected_case_batches=expected_acs_case_batches,
        immutable_snapshot_slots=immutable_snapshot_slots,
    )
    acs_async_covered = {
        obligation
        for result in acs_async_results
        for obligation in result.get("covered_obligations", ())
        if isinstance(obligation, str)
    }
    compact_science_async_results = _validate_compact_science_async_route_obligations_v1(
        covered=set(covered),
        exact_cases=exact_cases,
        route_contract=route_contract,
        execution_route_sha256=route_contract.execution_route_sha256,
        expected_case_batches=expected_acs_case_batches,
    )
    compact_science_async_covered = {
        obligation
        for result in compact_science_async_results
        for obligation in result.get("covered_obligations", ())
        if isinstance(obligation, str)
    }
    semantic_obligation_results = (
        *_validate_semantic_route_obligations_v1(
            covered=set(covered),
            plans=plans,
            exact_cases=exact_cases,
        ),
        *acs_async_results,
        *compact_science_async_results,
        *_validate_tick_native_route_obligations_v1(
            covered=set(covered),
            exact_cases=exact_cases,
        ),
        *([focused_result] if requested_focused else []),
        *([inheritance_result] if inheritance_result is not None else []),
    )
    ordinary_route_covered = (
        set(covered)
        - set(_ACS_ASYNC_ROUTE_OBLIGATIONS_V1)
        - set(_COMPACT_SCIENCE_ASYNC_ROUTE_OBLIGATIONS_V1)
    )
    final_covered = (
        previous_covered
        | inherited_covered
        | ordinary_route_covered
        | acs_async_covered
        | compact_science_async_covered
        | focused_covered
    )
    required = set(route_contract.required_obligations)
    missing_obligations = required - final_covered
    unproven_without_focused_evidence = (
        set(_ACS_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
        | set(_COMPACT_SCIENCE_FOCUSED_EVIDENCE_OBLIGATIONS_V1)
    )
    if missing_obligations - unproven_without_focused_evidence:
        raise RouteQualificationError(
            "successful cases do not cover every required route obligation"
        )
    qualification_plan = _mapping(
        canary.get("qualification_plan"), name="qualification_plan"
    )
    artifact_hashes = campaign_plan.get("qualification_key", {})
    artifact_hashes = (
        artifact_hashes.get("artifact_hashes")
        if isinstance(artifact_hashes, Mapping)
        else {}
    )
    artifact_sha256 = (
        artifact_hashes.get("cuda_extension")
        if isinstance(artifact_hashes, Mapping)
        else None
    )
    qualification_key = campaign_plan.get("qualification_key")
    variability_envelope = (
        qualification_key.get("variability_envelope", {})
        if isinstance(qualification_key, Mapping)
        else {}
    )
    if not isinstance(variability_envelope, Mapping):
        raise RouteQualificationError(
            "qualification key variability envelope is malformed"
        )
    long_horizon_scope = qualification_scope_for_plan_v1(campaign_plan)
    qualification_dimensions: tuple[str, ...] = (
        (BOUNDED_ROUTE_QUALIFIED,) if not missing_obligations else ()
    )
    if long_horizon_evidence is not None:
        if missing_obligations:
            raise RouteQualificationError(
                "long-horizon evidence cannot extend an incomplete bounded route"
            )
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise RouteQualificationError(
                "long-horizon qualification requires the exact CUDA artifact identity"
            )
        expected_identity: dict[str, Any] = {
            "execution_route_sha256": route_contract.execution_route_sha256,
            "provider_id": route_contract.provider_id,
            "profile_id": route_contract.profile_id,
            "plan_sha256": campaign_plan.get("plan_sha256"),
        }
        expected_identity["artifact_sha256s"] = {
            str(name): str(digest)
            for name, digest in artifact_hashes.items()
            if isinstance(name, str) and isinstance(digest, str)
        }
        try:
            validated_long_horizon = validate_long_horizon_evidence_v1(
                long_horizon_evidence,
                expected_identity=expected_identity,
            )
        except LongHorizonQualificationError as exc:
            raise RouteQualificationError(
                f"long-horizon qualification evidence is invalid: {exc}"
            ) from exc
        _validate_b136_5000tick_qualification_pair_v1(
            long_horizon_evidence,
            qualification_dimensions=validated_long_horizon["qualification_dimensions"],
        )
        qualification_dimensions = (
            BOUNDED_ROUTE_QUALIFIED,
            *validated_long_horizon["qualification_dimensions"],
        )
        long_horizon_scope = {
            **long_horizon_scope,
            "dimensions": list(qualification_dimensions),
            "validated_evidence": dict(long_horizon_evidence),
            "evidence_binding": validated_long_horizon["evidence_binding"],
        }
    receipt = ExecutionRouteQualificationReceiptV1(
        schema_version=EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
        execution_route_sha256=route_contract.execution_route_sha256,
        provider_id=route_contract.provider_id,
        profile_id=route_contract.profile_id,
        covered_obligations=tuple(sorted(final_covered)),
        qualification_plan_ids=(str(qualification_plan["plan_id"]),),
        case_ids=tuple(str(case["id"]) for case in cases),
        artifact_sha256=(artifact_sha256 if isinstance(artifact_sha256, str) else None),
        checkpoint_results={
            "exact": True,
            "case_count": len(exact_cases),
        },
        reporting_results={
            "exact": True,
            "case_count": len(exact_cases),
        },
        route_obligation_results=semantic_obligation_results,
        exact_comparisons=exact_comparisons,
        variability_envelope=dict(variability_envelope),
        authority_state=(
            "qualification_only" if missing_obligations else "route_qualified"
        ),
        qualification_dimensions=qualification_dimensions,
        long_horizon_scope=long_horizon_scope,
    ).payload()
    receipt["missing_obligations"] = sorted(missing_obligations)
    receipt["route_contract"] = route_contract.payload()
    receipt["route_contract_sha256"] = route_contract.execution_route_sha256
    atomic_write_json(sidecar, receipt)
    legacy_sidecar = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path
    )
    if legacy_sidecar != sidecar:
        atomic_write_json(legacy_sidecar, receipt)
    return {
        "qualification_succeeded": not missing_obligations,
        "execution_route_sha256": route_contract.execution_route_sha256,
        "qualification_receipt_path": str(sidecar),
        "covered_obligations": sorted(final_covered),
        "missing_obligations": sorted(missing_obligations),
        "case_count": len(exact_cases),
        "exact_comparison_count": len(exact_comparisons),
    }


def promote_execution_route_long_horizon_evidence_v1(
    *,
    campaign_plan: Mapping[str, Any],
    route_contract: ExecutionRouteContractV1,
    qualification_receipt_path: str | Path,
    long_horizon_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend an exact bounded-route receipt without rerunning its canaries."""

    sidecar = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path,
        route_contract.execution_route_sha256,
    )
    if not sidecar.is_file() or sidecar.is_symlink():
        raise RouteQualificationError(
            "long-horizon promotion requires the exact hash-specific route receipt"
        )
    try:
        current_sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    except OSError as exc:
        raise RouteQualificationError(
            "long-horizon promotion cannot read the exact route receipt"
        ) from exc
    receipt = read_execution_route_qualification_sidecar_v1(
        qualification_receipt_path,
        route_contract.execution_route_sha256,
    )
    if not isinstance(receipt, Mapping):
        raise RouteQualificationError("bounded route qualification receipt is absent")
    value = dict(receipt)
    required = list(route_contract.payload()["required_obligations"])
    covered_raw = value.get("covered_obligations")
    if (
        value.get("schema_version") != EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
        or value.get("authority_state") != "route_qualified"
        or value.get("execution_route_sha256") != route_contract.execution_route_sha256
        or value.get("route_contract_sha256") != route_contract.execution_route_sha256
        or value.get("route_contract") != route_contract.payload()
        or value.get("provider_id") != route_contract.provider_id
        or value.get("profile_id") != route_contract.profile_id
        or not isinstance(covered_raw, list)
        or covered_raw != required
        or value.get("missing_obligations") != []
        or BOUNDED_ROUTE_QUALIFIED not in value.get("qualification_dimensions", [])
    ):
        raise RouteQualificationError(
            "long-horizon promotion requires a complete exact bounded-route receipt"
        )
    qualification_key = campaign_plan.get("qualification_key")
    artifact_hashes = (
        qualification_key.get("artifact_hashes")
        if isinstance(qualification_key, Mapping)
        else None
    )
    artifact_sha256 = (
        artifact_hashes.get("cuda_extension")
        if isinstance(artifact_hashes, Mapping)
        else None
    )
    if (
        not isinstance(artifact_sha256, str)
        or value.get("artifact_sha256") != artifact_sha256
    ):
        raise RouteQualificationError(
            "bounded route receipt does not bind the current CUDA artifact"
        )
    expected_identity = {
        "execution_route_sha256": route_contract.execution_route_sha256,
        "provider_id": route_contract.provider_id,
        "profile_id": route_contract.profile_id,
        "artifact_sha256s": {
            str(name): str(digest)
            for name, digest in artifact_hashes.items()
            if isinstance(name, str) and isinstance(digest, str)
        },
        "plan_sha256": campaign_plan.get("plan_sha256"),
    }
    try:
        validated = validate_long_horizon_evidence_v1(
            long_horizon_evidence,
            expected_identity=expected_identity,
        )
    except LongHorizonQualificationError as exc:
        raise RouteQualificationError(
            f"long-horizon qualification evidence is invalid: {exc}"
        ) from exc
    paired_bindings = long_horizon_evidence.get("paired_run_bindings")
    arms = (
        (
            paired_bindings.get("uninterrupted"),
            paired_bindings.get("restored"),
        )
        if isinstance(paired_bindings, Mapping)
        else ()
    )
    bound_qualification_hashes = {
        arm.get("qualification_receipt_sha256")
        for arm in arms
        if isinstance(arm, Mapping)
    }
    if (
        len(bound_qualification_hashes) != 1
        or not all(
            isinstance(item, str)
            and len(item) == 64
            and all(char in "0123456789abcdef" for char in item)
            for item in bound_qualification_hashes
        )
    ):
        raise RouteQualificationError(
            "long-horizon evidence does not bind one exact source route receipt"
        )
    bound_source_receipt_sha256 = next(iter(bound_qualification_hashes))
    already_promoted = LONG_HORIZON_RUNABILITY_QUALIFIED in value.get(
        "qualification_dimensions", []
    )
    lineage_backfill = (
        already_promoted and value.get("promotion_source_receipt_sha256") is None
    )
    if already_promoted:
        existing_scope = value.get("long_horizon_scope")
        existing_evidence = (
            existing_scope.get("validated_evidence")
            if isinstance(existing_scope, Mapping)
            else None
        )
        if (
            not isinstance(existing_evidence, Mapping)
            or existing_evidence.get("evidence_sha256")
            != long_horizon_evidence.get("evidence_sha256")
            or value.get("promotion_source_receipt_sha256")
            not in {None, bound_source_receipt_sha256}
        ):
            raise RouteQualificationError(
                "existing long-horizon promotion differs from supplied evidence"
            )
        promotion_source_receipt_sha256 = bound_source_receipt_sha256
    else:
        if current_sidecar_sha256 != bound_source_receipt_sha256:
            raise RouteQualificationError(
                "bounded route receipt differs from the paired-run source binding"
            )
        promotion_source_receipt_sha256 = current_sidecar_sha256
    _validate_b136_5000tick_qualification_pair_v1(
        long_horizon_evidence,
        qualification_dimensions=validated["qualification_dimensions"],
        canonical_revalidation=not lineage_backfill,
    )
    dimensions = [
        BOUNDED_ROUTE_QUALIFIED,
        *validated["qualification_dimensions"],
    ]
    value["artifact_sha256s"] = expected_identity["artifact_sha256s"]
    value["promotion_source_receipt_sha256"] = promotion_source_receipt_sha256
    value["qualification_dimensions"] = sorted(set(dimensions))
    value["long_horizon_scope"] = {
        **qualification_scope_for_plan_v1(campaign_plan),
        "dimensions": list(dimensions),
        "validated_evidence": dict(long_horizon_evidence),
        "evidence_binding": validated["evidence_binding"],
        "promotion_mode": "existing_bounded_route_evidence_reused_v1",
    }
    atomic_write_json(sidecar, value)
    legacy_sidecar = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path
    )
    if legacy_sidecar != sidecar:
        atomic_write_json(legacy_sidecar, value)
    return {
        "qualification_succeeded": True,
        "execution_route_sha256": route_contract.execution_route_sha256,
        "qualification_receipt_path": str(sidecar),
        "qualification_dimensions": value["qualification_dimensions"],
        "long_horizon_evidence_sha256": long_horizon_evidence.get("evidence_sha256"),
        "bounded_canaries_rerun": False,
    }


def promote_execution_route_science_evidence_v1(
    *,
    campaign_plan: Mapping[str, Any],
    route_contract: ExecutionRouteContractV1,
    qualification_receipt_path: str | Path,
    long_horizon_evidence: Mapping[str, Any],
    canonical_full_production_route_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach a separate Gate27 science authority to a qualified route.

    The existing bounded and long-horizon scopes remain byte-for-byte evidence
    inputs.  Gate27 is written to a separate science projection; the exact
    pre-promotion route receipt is never rewritten and remains an immutable
    evidence input to the digest-bound attestation.
    """

    sidecar = execution_route_qualification_sidecar_path_v1(
        qualification_receipt_path,
        route_contract.execution_route_sha256,
    )
    projection = execution_route_science_qualification_projection_path_v1(
        qualification_receipt_path,
        route_contract.execution_route_sha256,
    )
    if not sidecar.is_file() or sidecar.is_symlink():
        raise RouteQualificationError(
            "science-route promotion requires the exact hash-specific route receipt"
        )
    try:
        source_route_receipt_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    except OSError as exc:
        raise RouteQualificationError(
            "science-route promotion cannot read the exact route receipt"
        ) from exc
    try:
        receipt = json.loads(sidecar.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteQualificationError(
            "bounded route qualification receipt is unreadable"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise RouteQualificationError("bounded route qualification receipt is absent")
    value = dict(receipt)
    if "science_route_scope" in value:
        raise RouteQualificationError(
            "route source receipt is already mutated; immutable pre-promotion evidence is unavailable"
        )
    required = list(route_contract.payload()["required_obligations"])
    covered_raw = value.get("covered_obligations")
    declared_dimensions = value.get("qualification_dimensions")
    if (
        value.get("schema_version") != EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION
        or value.get("authority_state") != "route_qualified"
        or value.get("execution_route_sha256") != route_contract.execution_route_sha256
        or value.get("route_contract_sha256") != route_contract.execution_route_sha256
        or value.get("route_contract") != route_contract.payload()
        or value.get("provider_id") != route_contract.provider_id
        or value.get("profile_id") != route_contract.profile_id
        or not isinstance(covered_raw, list)
        or covered_raw != required
        or value.get("missing_obligations") != []
        or not isinstance(declared_dimensions, Sequence)
        or isinstance(declared_dimensions, (str, bytes))
        or not {
            BOUNDED_ROUTE_QUALIFIED,
            LONG_HORIZON_RUNABILITY_QUALIFIED,
        }.issubset(declared_dimensions)
    ):
        raise RouteQualificationError(
            "science-route promotion requires a complete exact long-horizon route receipt"
        )
    qualification_key = campaign_plan.get("qualification_key")
    artifact_hashes = (
        qualification_key.get("artifact_hashes")
        if isinstance(qualification_key, Mapping)
        else None
    )
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise RouteQualificationError(
            "science-route promotion requires the current artifact identity map"
        )
    artifact_sha256 = artifact_hashes.get("cuda_extension")
    if (
        not isinstance(artifact_sha256, str)
        or value.get("artifact_sha256") != artifact_sha256
    ):
        raise RouteQualificationError(
            "bounded route receipt does not bind the current CUDA artifact"
        )
    expected_identity = {
        "execution_route_sha256": route_contract.execution_route_sha256,
        "provider_id": route_contract.provider_id,
        "profile_id": route_contract.profile_id,
        "artifact_sha256s": {
            str(name): str(digest)
            for name, digest in artifact_hashes.items()
            if isinstance(name, str) and isinstance(digest, str)
        },
        "plan_sha256": campaign_plan.get("plan_sha256"),
    }
    try:
        validated_long = validate_long_horizon_evidence_v1(
            long_horizon_evidence,
            expected_identity=expected_identity,
        )
    except LongHorizonQualificationError as exc:
        raise RouteQualificationError(
            f"long-horizon qualification evidence is invalid: {exc}"
        ) from exc
    if validated_long["qualification_dimensions"] != [
        LONG_HORIZON_RUNABILITY_QUALIFIED
    ]:
        raise RouteQualificationError(
            "science-route promotion requires the unchanged long-horizon evidence scope"
        )
    existing_scope = value.get("long_horizon_scope")
    if not isinstance(existing_scope, Mapping):
        raise RouteQualificationError(
            "science-route promotion requires the promoted long-horizon scope"
        )
    if (
        existing_scope.get("validated_evidence", {}).get("evidence_sha256")
        if isinstance(existing_scope.get("validated_evidence"), Mapping)
        else None
    ) != long_horizon_evidence.get("evidence_sha256"):
        raise RouteQualificationError(
            "route receipt long-horizon evidence differs from supplied evidence"
        )
    if existing_scope.get("evidence_binding") != validated_long["evidence_binding"]:
        raise RouteQualificationError(
            "route receipt long-horizon binding differs from supplied evidence"
        )
    paired_bindings = long_horizon_evidence.get("paired_run_bindings")
    if not isinstance(paired_bindings, Mapping):
        raise RouteQualificationError(
            "science-route promotion requires paired long-horizon lineage"
        )
    expected_attestation_identity = validated_long["evidence_identity"]
    expected_evidence_sha256 = long_horizon_evidence["evidence_sha256"]
    expected_evidence_binding_sha256 = validated_long["evidence_binding"][
        "binding_sha256"
    ]
    existing_projection: dict[str, Any] | None = None
    if projection.exists():
        if projection.is_symlink() or not projection.is_file():
            raise RouteQualificationError(
                "existing science-route projection is not a regular file"
            )
        try:
            loaded_projection = json.loads(
                projection.read_bytes().decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouteQualificationError(
                "existing science-route projection is unreadable"
            ) from exc
        if not isinstance(loaded_projection, Mapping):
            raise RouteQualificationError(
                "existing science-route projection is not an object"
            )
        existing_projection = dict(loaded_projection)
        for name, item in value.items():
            if name in {"qualification_dimensions", "science_route_scope"}:
                continue
            if existing_projection.get(name) != item:
                raise RouteQualificationError(
                    "science-route projection no longer matches its immutable source"
                )
    existing_science_scope = (
        existing_projection.get("science_route_scope")
        if existing_projection is not None
        else None
    )
    existing_attestation = (
        existing_science_scope.get("canonical_full_production_route_attestation")
        if isinstance(existing_science_scope, Mapping)
        else None
    )
    if existing_science_scope is not None and not isinstance(
        existing_attestation, Mapping
    ):
        raise RouteQualificationError(
            "existing science-route scope lacks its canonical attestation"
        )
    if isinstance(existing_attestation, Mapping):
        try:
            attestation = validate_canonical_full_production_route_attestation_v1(
                existing_attestation,
                expected_identity=expected_attestation_identity,
                expected_evidence_sha256=expected_evidence_sha256,
                expected_evidence_binding_sha256=expected_evidence_binding_sha256,
                expected_source_route_receipt_sha256=source_route_receipt_sha256,
            )
            if canonical_full_production_route_attestation is not None:
                supplied = validate_canonical_full_production_route_attestation_v1(
                    canonical_full_production_route_attestation,
                    expected_identity=expected_attestation_identity,
                    expected_evidence_sha256=expected_evidence_sha256,
                    expected_evidence_binding_sha256=expected_evidence_binding_sha256,
                    expected_source_route_receipt_sha256=source_route_receipt_sha256,
                )
                if supplied["attestation_sha256"] != attestation["attestation_sha256"]:
                    raise LongHorizonQualificationError(
                        "idempotent science-route promotion changes the attestation"
                    )
        except (KeyError, LongHorizonQualificationError) as exc:
            raise RouteQualificationError(
                f"existing canonical full production route attestation is invalid: {exc}"
            ) from exc
        validated_science = validate_long_horizon_evidence_v1(
            long_horizon_evidence,
            expected_identity=expected_identity,
            canonical_full_production_route_attestation=attestation,
        )
        _validate_b136_5000tick_qualification_pair_v1(
            long_horizon_evidence,
            qualification_dimensions=validated_long["qualification_dimensions"],
            canonical_revalidation=False,
        )
        return {
            "qualification_succeeded": True,
            "execution_route_sha256": route_contract.execution_route_sha256,
            "qualification_receipt_path": str(projection),
            "qualification_dimensions": existing_projection["qualification_dimensions"],
            "long_horizon_evidence_sha256": expected_evidence_sha256,
            "science_route_attestation_sha256": attestation["attestation_sha256"],
            "bounded_canaries_rerun": False,
            "idempotent": True,
        }

    if not isinstance(canonical_full_production_route_attestation, Mapping):
        raise RouteQualificationError(
            "science-route promotion requires a canonical evidence-ref attestation"
        )
    raw_refs = canonical_full_production_route_attestation.get("evidence_refs")
    if not isinstance(raw_refs, Sequence) or isinstance(raw_refs, (str, bytes)):
        raise RouteQualificationError(
            "science-route promotion requires structured evidence_refs"
        )
    refs: list[dict[str, Any]] = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, Mapping):
            raise RouteQualificationError("science-route evidence ref is malformed")
        ref = dict(raw_ref)
        if ref.get("role") == "promoted_route_receipt":
            if (
                ref.get("path") != str(sidecar)
                or ref.get("file_sha256") != source_route_receipt_sha256
                or ref.get("source_route_receipt_sha256")
                != source_route_receipt_sha256
            ):
                raise RouteQualificationError(
                    "science-route route ref must bind the exact pre-promotion receipt"
                )
        refs.append(ref)
    try:
        materialized = materialize_canonical_full_production_route_attestation_v1(
            evidence_refs=refs,
            expected_identity=expected_attestation_identity,
            expected_evidence_sha256=expected_evidence_sha256,
            expected_evidence_binding_sha256=expected_evidence_binding_sha256,
            expected_source_route_receipt_sha256=source_route_receipt_sha256,
        )
        attestation = validate_canonical_full_production_route_attestation_v1(
            canonical_full_production_route_attestation,
            expected_identity=expected_attestation_identity,
            expected_evidence_sha256=expected_evidence_sha256,
            expected_evidence_binding_sha256=expected_evidence_binding_sha256,
            expected_source_route_receipt_sha256=source_route_receipt_sha256,
        )
        if attestation != materialized:
            raise LongHorizonQualificationError(
                "science-route attestation is not the canonical materialization"
            )
    except (KeyError, LongHorizonQualificationError) as exc:
        raise RouteQualificationError(
            f"canonical full production route attestation is invalid: {exc}"
        ) from exc
    validated_science = validate_long_horizon_evidence_v1(
        long_horizon_evidence,
        expected_identity=expected_identity,
        canonical_full_production_route_attestation=attestation,
    )
    _validate_b136_5000tick_qualification_pair_v1(
        long_horizon_evidence,
        qualification_dimensions=validated_long["qualification_dimensions"],
        canonical_revalidation=False,
    )
    science_scope = {
        "route_scope": CANONICAL_FULL_PRODUCTION_ROUTE_SCOPE,
        "dimensions": [
            BOUNDED_ROUTE_QUALIFIED,
            LONG_HORIZON_RUNABILITY_QUALIFIED,
            SCIENCE_ROUTE_QUALIFIED,
        ],
        "validated_evidence": dict(long_horizon_evidence),
        "evidence_binding": validated_science["evidence_binding"],
        "canonical_full_production_route_attestation": attestation,
        "attestation_sha256": attestation["attestation_sha256"],
        "source_route_receipt_sha256": source_route_receipt_sha256,
        "promotion_mode": "canonical_full_production_route_attested_v1",
    }
    value["qualification_dimensions"] = [
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
        SCIENCE_ROUTE_QUALIFIED,
    ]
    value["science_route_scope"] = science_scope
    if projection.exists():
        raise RouteQualificationError(
            "science-route projection appeared during promotion"
        )
    atomic_write_json(projection, value)
    return {
        "qualification_succeeded": True,
        "execution_route_sha256": route_contract.execution_route_sha256,
        "qualification_receipt_path": str(projection),
        "qualification_dimensions": value["qualification_dimensions"],
        "long_horizon_evidence_sha256": long_horizon_evidence.get("evidence_sha256"),
        "science_route_attestation_sha256": attestation["attestation_sha256"],
        "bounded_canaries_rerun": False,
    }


__all__ = [
    "RouteQualificationError",
    "bounded_route_qualification_preflight_v1",
    "derive_bounded_route_campaigns_v1",
    "execute_route_qualification_v1",
    "promote_execution_route_long_horizon_evidence_v1",
    "promote_execution_route_science_evidence_v1",
]
