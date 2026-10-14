#!/usr/bin/env python3
"""Project one completed B48 stage through the existing WRC reducer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Final
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from engine.campaigns.wrc_adaptive_rebinding import (  # noqa: E402
    WRCCheckpointAdaptiveRebindingError,
    validate_wrc_checkpoint_adaptive_rebinding_receipt_v1,
)
from engine.campaigns.wrc_variability import (  # noqa: E402
    INVARIANT_NAME,
    WRCVariabilityEnvelopeError,
    validate_wrc_parameter_variability_envelope_v1,
)
from engine.wrc.reducer import (  # noqa: E402
    Observation,
    SourceLocator,
    WorldIdentity,
    WorldReducer,
)
from tools.scripts.eiel_wrc_b48_adaptive_stage import (  # noqa: E402
    validate_design_binding,
)


STAGE_WINDOWS: Final[dict[str, tuple[int, int, int, int]]] = {
    "C1": (0, 25, 0, 4_000),
    "C2": (25, 50, 4_000, 8_000),
    "C3": (50, 75, 8_000, 12_000),
    "C4": (75, 100, 12_000, 16_000),
    "C5": (100, 125, 16_000, 20_000),
}
WORLD_FIELDS: Final[tuple[str, ...]] = (
    "stage",
    "tick_start",
    "tick_end",
    "config_id",
    "seed_tuple",
    "cell_id",
    "branch_id",
    "rng_lineage_id",
    "active_bundle_sha256",
    "observation_count",
    "metric_count",
    "exploratory_viability_band",
    "alive_first",
    "alive_terminal",
    "alive_minimum",
    "alive_trajectory_mean",
    "births_total",
    "failed_births_total",
    "deaths_total",
    "moved_total",
    "gathered_total",
    "food_stock_terminal",
    "energy_collected_terminal",
    "policy_energy_terminal",
    "metric_summaries_json",
    "reducer_result_sha256",
)
CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "stage",
    "tick_start",
    "tick_end",
    "config_id",
    "replicate_count",
    "viable_count",
    "fragile_count",
    "collapsed_count",
    "extinct_count",
    "alive_terminal_mean",
    "alive_terminal_median",
    "alive_terminal_minimum",
    "alive_terminal_maximum",
    "alive_terminal_range",
    "births_total_mean",
    "deaths_total_mean",
    "metric_aggregate_json",
)
DIVERGENCE_FIELDS: Final[tuple[str, ...]] = (
    "stage",
    "config_id",
    "metric",
    "replicate_count",
    "terminal_mean",
    "terminal_median",
    "terminal_minimum",
    "terminal_maximum",
    "terminal_range",
    "terminal_relative_range",
    "terminal_population_stddev",
    "agreement_band",
)
CHECKPOINT_FIELDS: Final[tuple[str, ...]] = (
    "stage",
    "absolute_tick",
    "generation",
    "checkpoint_schema_version",
    "checkpoint_sha256",
    "plan_sha256",
    "qualification_key_sha256",
    "qualification_receipt_sha256",
    "variability_envelope_id",
    "variability_envelope_sha256",
    "execution_route_sha256",
    "physical_cell_count",
    "terminal_cell_digests_sha256",
    "checkpoint_identity_contract_sha256",
    "adaptive_rebinding_receipt_sha256",
    "adaptive_runtime_readback_receipt_sha256",
    "continuity_status",
    "qualification_receipt_action",
)
_ROW_IDENTITY_FIELDS = frozenset(
    {
        "cell_id",
        "cell",
        "generation",
        "schema_version",
        "world",
        "production_control_bindings_sha256",
        "production_control_readback_sha256",
        "production_control_readback_status",
    }
)


class AdaptiveAnalysisError(ValueError):
    """A completed stage cannot support exact B48 interval analysis."""


def _verified_payload(
    value: Mapping[str, Any], *, hash_field: str, label: str
) -> dict[str, Any]:
    payload = dict(value)
    declared = payload.pop(hash_field, None)
    if not isinstance(declared, str) or canonical_sha256(payload) != declared:
        raise AdaptiveAnalysisError(f"{label} identity hash differs")
    return {**payload, hash_field: declared}


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _row_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    supplied = row.get("metrics")
    source = supplied if isinstance(supplied, Mapping) else row
    metrics = {
        str(name): float(value)
        for name, value in source.items()
        if name not in _ROW_IDENTITY_FIELDS and _finite_number(value)
    }
    if "alive" not in metrics and "population.alive" in metrics:
        metrics["alive"] = metrics["population.alive"]
    if "alive" not in metrics:
        raise AdaptiveAnalysisError("generation metric row lacks alive")
    return dict(sorted(metrics.items()))


def _metric_value(
    result: Mapping[str, Any], metric: str, field: str, default: float = 0.0
) -> float:
    metrics = result.get("metrics")
    summary = metrics.get(metric) if isinstance(metrics, Mapping) else None
    if not isinstance(summary, Mapping):
        return default
    value = summary.get(field)
    if field in {"first", "last", "minimum", "maximum"}:
        value = value.get("value") if isinstance(value, Mapping) else None
    return float(value) if _finite_number(value) else default


def _viability(result: Mapping[str, Any]) -> str:
    first = _metric_value(result, "alive", "first")
    terminal = _metric_value(result, "alive", "last")
    if terminal <= 0.0:
        return "EXTINCT"
    if first > 0.0 and terminal <= max(1.0, first * 0.10):
        return "COLLAPSED"
    if first > 0.0 and terminal <= first * 0.50:
        return "FRAGILE"
    return "VIABLE"


def _summary_projection(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw = result.get("metrics")
    if not isinstance(raw, Mapping):
        raise AdaptiveAnalysisError("reducer result has no metric summaries")
    projected: dict[str, dict[str, float]] = {}
    for name, value in sorted(raw.items()):
        if not isinstance(value, Mapping):
            continue
        projected[str(name)] = {
            "first": _metric_value(result, str(name), "first"),
            "terminal": _metric_value(result, str(name), "last"),
            "minimum": _metric_value(result, str(name), "minimum"),
            "maximum": _metric_value(result, str(name), "maximum"),
            "trajectory_mean": _metric_value(
                result, str(name), "trajectory_mean"
            ),
            "net_change": _metric_value(result, str(name), "net_change"),
            "slope_per_tick": _metric_value(result, str(name), "slope_per_tick"),
            "information_score": _metric_value(
                result, str(name), "information_score"
            ),
        }
    return projected


def _validate_checkpoint(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    stage: str,
    cell_ids: list[str],
    transition: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    start_generation, end_generation, _tick_start, _tick_end = STAGE_WINDOWS[stage]
    if (
        manifest.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or manifest.get("campaign_id") != plan.get("campaign_id")
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or manifest.get("batch_id") != "batch-0000"
        or manifest.get("cell_ids") != cell_ids
        or manifest.get("status") != "completed"
    ):
        raise AdaptiveAnalysisError("completed B48 batch manifest binding differs")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("generation") != end_generation:
        raise AdaptiveAnalysisError("terminal checkpoint generation differs")
    checkpoint = dict(checkpoint)
    binding = checkpoint.get("checkpoint_binding")
    route = manifest.get("execution_route_sha256")
    if (
        not isinstance(binding, Mapping)
        or binding.get("plan_sha256") != plan.get("plan_sha256")
        or binding.get("qualification_key_sha256")
        != plan.get("qualification_key_sha256")
        or binding.get("cell_ids") != cell_ids
        or binding.get("execution_route_sha256") != route
        or checkpoint.get("execution_route_sha256") != route
    ):
        raise AdaptiveAnalysisError("checkpoint plan, route, or cell binding differs")
    bundles = checkpoint.get("cell_bundles")
    if (
        not isinstance(bundles, list)
        or len(bundles) != 48
        or {row.get("cell_id") for row in bundles if isinstance(row, Mapping)}
        != set(cell_ids)
    ):
        raise AdaptiveAnalysisError("checkpoint does not bind 48 cell bundles")
    terminal_digests = checkpoint.get("terminal_cell_digests")
    identity = checkpoint.get("checkpoint_identity_contract")
    if (
        not isinstance(terminal_digests, Mapping)
        or set(terminal_digests) != set(cell_ids)
        or not isinstance(identity, Mapping)
        or identity.get("terminal_projection_digest_sha256s") != terminal_digests
    ):
        raise AdaptiveAnalysisError("terminal checkpoint identity is incomplete")
    for field in (
        "canonical_terminal_state_digest_sha256s",
        "canonical_checkpoint_storage_digest_sha256s",
    ):
        value = identity.get(field)
        if not isinstance(value, Mapping) or set(value) != set(cell_ids):
            raise AdaptiveAnalysisError(f"checkpoint identity lacks {field}")
    parents = identity.get("parent_checkpoint_projection_digest_sha256s")
    if start_generation == 0:
        if parents is not None:
            raise AdaptiveAnalysisError("C1 checkpoint unexpectedly has a parent")
    elif not isinstance(parents, Mapping) or set(parents) != set(cell_ids):
        raise AdaptiveAnalysisError("continued checkpoint parent identity is incomplete")

    adaptive: dict[str, Any] | None = None
    readback: dict[str, Any] | None = None
    if stage == "C1":
        if transition is not None:
            raise AdaptiveAnalysisError("C1 must not have an adaptive transition")
    else:
        if not isinstance(transition, Mapping):
            raise AdaptiveAnalysisError(f"{stage} adaptive transition receipt is absent")
        transition = _verified_payload(
            transition,
            hash_field="receipt_sha256",
            label="adaptive transition receipt",
        )
        if (
            transition.get("schema_version")
            != "eiel_wrc_adaptive_campaign_transition_v1"
            or transition.get("restore_plan_sha256") != plan.get("plan_sha256")
            or transition.get("restore_generation") != end_generation
            or transition.get("source_generation") != start_generation
            or transition.get("cell_ids") != cell_ids
            or transition.get("qualification_key_sha256")
            != plan.get("qualification_key_sha256")
            or transition.get("execution_route_sha256") != route
            or transition.get("qualification_receipt_action") != "unchanged"
            or transition.get("status") != "READY_TO_RESUME"
        ):
            raise AdaptiveAnalysisError("adaptive transition binding differs")
        result = manifest.get("result")
        fresh = (
            result.get("fresh_owner_restore_receipt")
            if isinstance(result, Mapping)
            else None
        )
        readback = (
            result.get("wrc_checkpoint_adaptive_runtime_readback")
            if isinstance(result, Mapping)
            else None
        )
        adaptive = (
            fresh.get("wrc_checkpoint_adaptive_rebinding")
            if isinstance(fresh, Mapping)
            else None
        )
        if not isinstance(adaptive, Mapping) or not isinstance(readback, Mapping):
            raise AdaptiveAnalysisError("adaptive restore/readback evidence is absent")
        try:
            adaptive = validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
                adaptive
            )
        except WRCCheckpointAdaptiveRebindingError as exc:
            raise AdaptiveAnalysisError(str(exc)) from exc
        readback = _verified_payload(
            readback,
            hash_field="receipt_sha256",
            label="adaptive runtime readback",
        )
        adaptive_sha = transition.get("adaptive_rebinding_receipt_sha256")
        if (
            adaptive.get("receipt_sha256") != adaptive_sha
            or readback.get("adaptive_rebinding_receipt_sha256") != adaptive_sha
            or readback.get("status") != "READBACK_COMPLETE"
            or readback.get("start_generation") != start_generation
        ):
            raise AdaptiveAnalysisError("adaptive restore/readback identity differs")
    return checkpoint, adaptive, readback


def analyze_completed_stage(
    plan_value: Mapping[str, Any],
    design_binding: Mapping[str, Any],
    batch_manifest: Mapping[str, Any],
    *,
    stage: str,
    transition_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce an exact completed 48-world interval without new semantics."""

    if stage not in STAGE_WINDOWS:
        raise AdaptiveAnalysisError(f"unsupported stage {stage!r}")
    plan = _verified_payload(plan_value, hash_field="plan_sha256", label="plan")
    start_generation, end_generation, tick_start, tick_end = STAGE_WINDOWS[stage]
    execution = plan.get("execution")
    cells = plan.get("cells")
    batches = plan.get("physical_batches")
    tick_schedule = (
        execution.get("tick_schedule") if isinstance(execution, Mapping) else None
    )
    if (
        not isinstance(execution, Mapping)
        or execution.get("generations") != end_generation
        or execution.get("ticks_per_generation") != 160
        or end_generation not in execution.get("checkpoints", [])
        or not isinstance(tick_schedule, Mapping)
        or tick_schedule.get("total_ticks") != tick_end
        or not isinstance(cells, list)
        or len(cells) != 48
        or not isinstance(batches, list)
        or len(batches) != 1
    ):
        raise AdaptiveAnalysisError("plan is not the exact B48/T160 stage")
    cell_ids = [
        str(cell.get("cell_id")) for cell in cells if isinstance(cell, Mapping)
    ]
    if len(cell_ids) != 48 or len(set(cell_ids)) != 48:
        raise AdaptiveAnalysisError("plan cell identity is malformed")
    if batches[0].get("batch_id") != "batch-0000" or batches[0].get(
        "cell_ids"
    ) != cell_ids:
        raise AdaptiveAnalysisError("plan physical B48 mapping differs")
    qualification = plan.get("qualification_key")
    if (
        not isinstance(qualification, Mapping)
        or canonical_sha256(qualification) != plan.get("qualification_key_sha256")
    ):
        raise AdaptiveAnalysisError("plan qualification identity differs")
    envelope_payload = qualification.get("variability_envelope")
    fixed_invariants = qualification.get("fixed_invariants")
    if not isinstance(envelope_payload, Mapping) or not isinstance(
        fixed_invariants, Mapping
    ):
        raise AdaptiveAnalysisError("plan variability envelope binding is absent")
    try:
        envelope_contract = validate_wrc_parameter_variability_envelope_v1(
            envelope_payload
        )
    except WRCVariabilityEnvelopeError as exc:
        raise AdaptiveAnalysisError(f"plan variability envelope is invalid: {exc}") from exc
    if fixed_invariants.get(INVARIANT_NAME) != envelope_contract.envelope_sha256:
        raise AdaptiveAnalysisError("plan variability envelope invariant differs")
    binding_rows = validate_design_binding(design_binding)
    binding_by_cell = {str(row["cell_id"]): row for row in binding_rows}
    if set(binding_by_cell) != set(cell_ids):
        raise AdaptiveAnalysisError("design binding does not cover plan cells")
    cells_by_id = {str(cell["cell_id"]): dict(cell) for cell in cells}
    for cell_id in cell_ids:
        binding = binding_by_cell[cell_id]
        cell = cells_by_id[cell_id]
        if (
            binding.get("branch_id") != cell.get("branch_id")
            or binding.get("rng_lineage_id") != cell.get("rng_lineage_id")
        ):
            raise AdaptiveAnalysisError("plan lineage differs from design binding")

    checkpoint, adaptive, readback = _validate_checkpoint(
        plan,
        batch_manifest,
        stage=stage,
        cell_ids=cell_ids,
        transition=transition_receipt,
    )
    raw_rows = batch_manifest.get("metrics_rows")
    if not isinstance(raw_rows, list):
        raise AdaptiveAnalysisError("batch manifest metrics rows are absent")
    interval: dict[str, dict[int, tuple[int, dict[str, float]]]] = defaultdict(dict)
    expected_generations = set(range(start_generation, end_generation))
    for row_number, raw in enumerate(raw_rows, 1):
        if not isinstance(raw, Mapping):
            raise AdaptiveAnalysisError("batch metric row is malformed")
        cell_id = raw.get("cell_id", raw.get("cell"))
        generation = raw.get("generation")
        if cell_id not in cells_by_id or type(generation) is not int:
            raise AdaptiveAnalysisError("batch metric row identity is invalid")
        if generation not in expected_generations:
            continue
        if generation in interval[str(cell_id)]:
            raise AdaptiveAnalysisError(
                f"duplicate interval metric row: {cell_id}/G{generation}"
            )
        interval[str(cell_id)][generation] = (row_number, _row_metrics(raw))
    if any(set(interval[cell_id]) != expected_generations for cell_id in cell_ids):
        raise AdaptiveAnalysisError("interval does not cover exact 48 x 25 rows")

    reductions: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        cell = cells_by_id[cell_id]
        binding = binding_by_cell[cell_id]
        parameters = cell.get("parameters")
        seeds = cell.get("seeds")
        if not isinstance(parameters, Mapping) or not isinstance(seeds, Mapping):
            raise AdaptiveAnalysisError("plan cell parameters or seeds are absent")
        bundle_sha = canonical_sha256(parameters)
        reducer = WorldReducer(
            WorldIdentity(
                world_id=cell_id,
                parameter_id="adaptive_bundle_sha256",
                parameter_value=bundle_sha,
                seed=int(seeds["ecology"]),
                ticks_per_observation=160,
                metadata={
                    "stage": stage,
                    "config_id": binding["config_id"],
                    "seed_tuple": binding["seed_tuple"],
                    "branch_id": cell["branch_id"],
                    "rng_lineage_id": cell["rng_lineage_id"],
                },
            )
        )
        raw_values: dict[str, list[float]] = defaultdict(list)
        for sequence, generation in enumerate(range(start_generation, end_generation)):
            row_number, metrics = interval[cell_id][generation]
            for name, value in metrics.items():
                raw_values[name].append(value)
            reducer.ingest(
                Observation(
                    sequence=sequence,
                    tick=(generation + 1) * 160,
                    metrics=metrics,
                    source=SourceLocator(
                        kind="campaign_batch_manifest",
                        uri=(
                            f"campaign://{plan['campaign_id']}/"
                            f"{plan['plan_sha256']}/batch-0000"
                        ),
                        row_number=row_number,
                        event_id=f"{cell_id}:G{generation}",
                    ),
                )
            )
        result = reducer.finalize()
        reductions.append(result)
        projection = _summary_projection(result)
        world_rows.append(
            {
                "stage": stage,
                "tick_start": tick_start,
                "tick_end": tick_end,
                "config_id": binding["config_id"],
                "seed_tuple": binding["seed_tuple"],
                "cell_id": cell_id,
                "branch_id": cell["branch_id"],
                "rng_lineage_id": cell["rng_lineage_id"],
                "active_bundle_sha256": bundle_sha,
                "observation_count": result["observation_count"],
                "metric_count": result["metric_count"],
                "exploratory_viability_band": _viability(result),
                "alive_first": _metric_value(result, "alive", "first"),
                "alive_terminal": _metric_value(result, "alive", "last"),
                "alive_minimum": _metric_value(result, "alive", "minimum"),
                "alive_trajectory_mean": _metric_value(
                    result, "alive", "trajectory_mean"
                ),
                "births_total": sum(raw_values.get("births", [])),
                "failed_births_total": sum(raw_values.get("failed_births", [])),
                "deaths_total": sum(raw_values.get("deaths", [])),
                "moved_total": sum(raw_values.get("moved", [])),
                "gathered_total": sum(raw_values.get("gathered", [])),
                "food_stock_terminal": _metric_value(
                    result, "patch_food_stock_after_turnover", "last"
                ),
                "energy_collected_terminal": _metric_value(
                    result,
                    "energy_collected_exact_sum_after_turnover",
                    "last",
                ),
                "policy_energy_terminal": _metric_value(
                    result, "policy_energy_live_sum_after_turnover", "last"
                ),
                "metric_summaries_json": json.dumps(
                    projection, sort_keys=True, separators=(",", ":")
                ),
                "reducer_result_sha256": result["result_sha256"],
            }
        )

    world_rows.sort(key=lambda row: (row["config_id"], row["seed_tuple"]))
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reduction_by_cell = {
        str(result["identity"]["world_id"]): result for result in reductions
    }
    for row in world_rows:
        by_config[str(row["config_id"])].append(row)
    if set(by_config) != {f"C{index:02d}" for index in range(1, 17)} or any(
        len(rows) != 3 for rows in by_config.values()
    ):
        raise AdaptiveAnalysisError("analysis does not preserve 16 x 3 replication")

    config_rows: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, Any]] = []
    for config_id, rows in sorted(by_config.items()):
        alive = [float(row["alive_terminal"]) for row in rows]
        bands = Counter(str(row["exploratory_viability_band"]) for row in rows)
        metric_names = set.intersection(
            *(
                set(reduction_by_cell[str(row["cell_id"])]["metrics"])
                for row in rows
            )
        )
        aggregate: dict[str, dict[str, float]] = {}
        for metric in sorted(metric_names):
            terminal = [
                _metric_value(reduction_by_cell[str(row["cell_id"])], metric, "last")
                for row in rows
            ]
            trajectory = [
                _metric_value(
                    reduction_by_cell[str(row["cell_id"])],
                    metric,
                    "trajectory_mean",
                )
                for row in rows
            ]
            net = [
                _metric_value(
                    reduction_by_cell[str(row["cell_id"])], metric, "net_change"
                )
                for row in rows
            ]
            terminal_range = max(terminal) - min(terminal)
            scale = max(abs(statistics.fmean(terminal)), 1.0e-12)
            relative_range = terminal_range / scale
            aggregate[metric] = {
                "terminal_mean": statistics.fmean(terminal),
                "terminal_median": statistics.median(terminal),
                "terminal_minimum": min(terminal),
                "terminal_maximum": max(terminal),
                "terminal_range": terminal_range,
                "trajectory_mean": statistics.fmean(trajectory),
                "net_change_mean": statistics.fmean(net),
            }
            divergence_rows.append(
                {
                    "stage": stage,
                    "config_id": config_id,
                    "metric": metric,
                    "replicate_count": 3,
                    "terminal_mean": statistics.fmean(terminal),
                    "terminal_median": statistics.median(terminal),
                    "terminal_minimum": min(terminal),
                    "terminal_maximum": max(terminal),
                    "terminal_range": terminal_range,
                    "terminal_relative_range": relative_range,
                    "terminal_population_stddev": statistics.pstdev(terminal),
                    "agreement_band": (
                        "HIGH" if relative_range <= 0.10 else "MODERATE"
                        if relative_range <= 0.50
                        else "LOW"
                    ),
                }
            )
        config_rows.append(
            {
                "stage": stage,
                "tick_start": tick_start,
                "tick_end": tick_end,
                "config_id": config_id,
                "replicate_count": 3,
                "viable_count": bands["VIABLE"],
                "fragile_count": bands["FRAGILE"],
                "collapsed_count": bands["COLLAPSED"],
                "extinct_count": bands["EXTINCT"],
                "alive_terminal_mean": statistics.fmean(alive),
                "alive_terminal_median": statistics.median(alive),
                "alive_terminal_minimum": min(alive),
                "alive_terminal_maximum": max(alive),
                "alive_terminal_range": max(alive) - min(alive),
                "births_total_mean": statistics.fmean(
                    float(row["births_total"]) for row in rows
                ),
                "deaths_total_mean": statistics.fmean(
                    float(row["deaths_total"]) for row in rows
                ),
                "metric_aggregate_json": json.dumps(
                    aggregate, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    artifacts = plan.get("artifacts")
    qualification_receipt = (
        artifacts.get("qualification_receipt")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(qualification_receipt, Mapping):
        raise AdaptiveAnalysisError("plan qualification binding is incomplete")
    terminal_digests = checkpoint["terminal_cell_digests"]
    identity_contract = checkpoint["checkpoint_identity_contract"]
    checkpoint_row = {
        "stage": stage,
        "absolute_tick": tick_end,
        "generation": end_generation,
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "checkpoint_sha256": canonical_sha256(checkpoint),
        "plan_sha256": plan["plan_sha256"],
        "qualification_key_sha256": plan["qualification_key_sha256"],
        "qualification_receipt_sha256": qualification_receipt.get("sha256"),
        "variability_envelope_id": envelope_contract.envelope_id,
        "variability_envelope_sha256": envelope_contract.envelope_sha256,
        "execution_route_sha256": batch_manifest.get("execution_route_sha256"),
        "physical_cell_count": 48,
        "terminal_cell_digests_sha256": canonical_sha256(terminal_digests),
        "checkpoint_identity_contract_sha256": canonical_sha256(identity_contract),
        "adaptive_rebinding_receipt_sha256": (
            None if adaptive is None else adaptive["receipt_sha256"]
        ),
        "adaptive_runtime_readback_receipt_sha256": (
            None if readback is None else readback["receipt_sha256"]
        ),
        "continuity_status": "VERIFIED_EXACT_48",
        "qualification_receipt_action": "unchanged",
    }
    viability_counts = Counter(
        str(row["exploratory_viability_band"]) for row in world_rows
    )
    result = {
        "schema_version": "eiel_wrc_b48_adaptive_stage_analysis_v1",
        "stage": stage,
        "source_generation": start_generation,
        "terminal_generation": end_generation,
        "tick_start": tick_start,
        "tick_end": tick_end,
        "plan_sha256": plan["plan_sha256"],
        "design_binding_receipt_sha256": design_binding.get("receipt_sha256"),
        "physical_cell_count": 48,
        "configuration_count": 16,
        "seed_replicates_per_configuration": 3,
        "interval_observation_count": 48 * 25,
        "reducer_version": reductions[0]["reducer_version"],
        "reducer_config_sha256": reductions[0]["config_sha256"],
        "world_reduction_set_sha256": canonical_sha256(
            [row["result_sha256"] for row in reductions]
        ),
        "world_rows": world_rows,
        "configuration_rows": config_rows,
        "seed_divergence_rows": divergence_rows,
        "world_reductions": reductions,
        "checkpoint_row": checkpoint_row,
        "b48_summary": {
            "viability_counts": dict(sorted(viability_counts.items())),
            "alive_terminal_minimum": min(
                float(row["alive_terminal"]) for row in world_rows
            ),
            "alive_terminal_maximum": max(
                float(row["alive_terminal"]) for row in world_rows
            ),
            "alive_terminal_mean": statistics.fmean(
                float(row["alive_terminal"]) for row in world_rows
            ),
        },
        "interpretation": "EXPLORATORY_ADAPTIVE_INTERVENTION",
        "viability_heuristic": (
            "EXTINCT terminal=0; COLLAPSED terminal<=10% first; "
            "FRAGILE terminal<=50% first; otherwise VIABLE"
        ),
        "limitations": [
            "Uses the existing WRC reducer over retained compact generation rows.",
            "Reduced summaries do not replace authoritative raw/checkpoint sources.",
            "Path-dependent stages do not identify isolated knob causality.",
            "Unavailable metrics remain absent rather than inferred.",
        ],
    }
    material = dict(result)
    material.pop("world_reductions")
    material.pop("world_rows")
    material.pop("configuration_rows")
    material.pop("seed_divergence_rows")
    return {**result, "analysis_sha256": canonical_sha256(material)}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def materialize_analysis_outputs(
    analysis: Mapping[str, Any], *, output_dir: Path
) -> dict[str, Any]:
    """Write canonical reducer projections and a hash-bound output receipt."""

    root = output_dir.resolve()
    outputs = {
        "world_summary": (
            "B48_EXPLORATORY_WORLD_SUMMARY.csv",
            _csv_text(analysis["world_rows"], WORLD_FIELDS),
        ),
        "configuration_summary": (
            "B48_EXPLORATORY_CONFIGURATION_SUMMARY.csv",
            _csv_text(analysis["configuration_rows"], CONFIG_FIELDS),
        ),
        "seed_divergence": (
            "B48_EXPLORATORY_SEED_DIVERGENCE.csv",
            _csv_text(analysis["seed_divergence_rows"], DIVERGENCE_FIELDS),
        ),
        "checkpoints": (
            "B48_EXPLORATORY_CHECKPOINTS.csv",
            _csv_text([analysis["checkpoint_row"]], CHECKPOINT_FIELDS),
        ),
        "world_reductions": (
            "B48_EXPLORATORY_WORLD_REDUCTIONS.jsonl",
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in analysis["world_reductions"]
            ),
        ),
    }
    descriptors: dict[str, dict[str, Any]] = {}
    for name, (filename, text) in outputs.items():
        path = root / filename
        _atomic_write_text(path, text)
        descriptors[name] = {
            "path": str(path),
            "sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
            "bytes": len(text.encode("utf-8")),
        }
    receipt = {
        "schema_version": "eiel_wrc_b48_adaptive_analysis_outputs_v1",
        "stage": analysis["stage"],
        "plan_sha256": analysis["plan_sha256"],
        "design_binding_receipt_sha256": analysis[
            "design_binding_receipt_sha256"
        ],
        "terminal_generation": analysis["terminal_generation"],
        "tick_end": analysis["tick_end"],
        "checkpoint_sha256": analysis["checkpoint_row"]["checkpoint_sha256"],
        "reducer_version": analysis["reducer_version"],
        "reducer_config_sha256": analysis["reducer_config_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "outputs": descriptors,
        "qualification_receipt_action": "unchanged",
    }
    receipt = {**receipt, "receipt_sha256": canonical_sha256(receipt)}
    atomic_write_json(root / "analysis_receipt.json", receipt)
    return receipt


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdaptiveAnalysisError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise AdaptiveAnalysisError(f"{label} must be an object")
    return dict(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--design-binding", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path)
    parser.add_argument("--stage", choices=tuple(STAGE_WINDOWS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transition = (
        None
        if args.transition_receipt is None
        else _load_json(args.transition_receipt, label="transition receipt")
    )
    analysis = analyze_completed_stage(
        _load_json(args.plan, label="plan"),
        _load_json(args.design_binding, label="design binding"),
        _load_json(args.batch_manifest, label="batch manifest"),
        stage=args.stage,
        transition_receipt=transition,
    )
    receipt = materialize_analysis_outputs(analysis, output_dir=args.output_dir)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
