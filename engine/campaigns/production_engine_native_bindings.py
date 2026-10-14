"""Engine-owned native integration bindings for configured domain mechanics.

The binding composes world and controller mechanics with one engine lifecycle.
It contains no campaign matrix and exposes no execution-provider identity.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Final, Mapping, Sequence

from .providers import CampaignProviderError


@dataclass(frozen=True, slots=True)
class NativeMechanicsBindingV1:
    binding_id: str
    artifact_abi: str
    batch_max: int
    generation_min: int
    generation_max: int
    ticks_per_generation: int
    policy_cache_refill: bool
    terminal_snapshot_metrics: bool
    runner_builder: Callable[..., Any]
    request_builder: Callable[..., Any]
    preparer: Callable[..., Any]
    executor: Callable[..., Any]
    projector: Callable[..., Any]
    generation_zero_checkpoint_projector: Callable[..., Any] | None = None
    terminal_snapshot_metrics_merger: (
        Callable[..., Sequence[Mapping[str, Any]]] | None
    ) = None
    source_preparer: Callable[..., Any] | None = None
    artifact_loader: Callable[..., Any] | None = None
    component_checkpoint_exporter: Callable[..., Mapping[str, Any]] | None = None
    component_checkpoint_restorer: Callable[..., int] | None = None
    component_checkpoint_source_restorer: Callable[..., Sequence[Mapping[str, Any]]] | None = None
    required_artifact_ids: tuple[str, ...] = (
        "cuda_extension",
        "extension_manifest",
        "qualification_receipt",
        "reporting_receipt",
    )
    abi_bound_artifact_ids: tuple[str, ...] = (
        "cuda_extension",
        "extension_manifest",
        "qualification_receipt",
        "reporting_receipt",
    )
    required_artifact_exports: tuple[str, ...] = ()
    admitted_ticks_per_generation: tuple[int, ...] = ()

    def adapters(self) -> dict[str, Callable[..., Any] | None]:
        return {
            "runner_builder": self.runner_builder,
            "request_builder": self.request_builder,
            "preparer": self.preparer,
            "executor": self.executor,
            "projector": self.projector,
            "generation_zero_checkpoint_projector": (
                self.generation_zero_checkpoint_projector
            ),
            "terminal_snapshot_metrics_merger": (
                self.terminal_snapshot_metrics_merger
            ),
            "source_preparer": self.source_preparer,
            "artifact_loader": self.artifact_loader,
            "component_checkpoint_exporter": (
                self.component_checkpoint_exporter
            ),
            "component_checkpoint_restorer": (
                self.component_checkpoint_restorer
            ),
            "component_checkpoint_source_restorer": (
                self.component_checkpoint_source_restorer
            ),
        }


@dataclass(frozen=True, slots=True)
class PreparedNativeMechanicsOwnerV1:
    """Engine-owned dispatch adapter around one configured mechanics owner."""

    physical_window_arguments: Any
    source_projections: tuple[Any, ...]
    cell_ids: tuple[str, ...]
    request_descriptors: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class NativeRequestDescriptorV1:
    """Compact per-cell request identity retained after streamed preparation."""

    cell_id: str
    cuda_device: str
    cuda_capability: Mapping[str, Any]
    event_capacity: int
    source_import_projection_digest_sha256: str | None


PATCH_FIXED_MLP_BINDING_ID: Final[str] = "patch_fixed_mlp_native_binding_v1"
WDL_FIXED_MLP_BINDING_ID: Final[str] = "wdl_fixed_mlp_native_binding_v1"
EED_FIXED_MLP_BINDING_ID: Final[str] = "eed_fixed_mlp_native_binding_v1"
JUNGLE_FIXED_MLP_BINDING_ID: Final[str] = "jungle_fixed_mlp_native_binding_v1"
WDL_JUNGLE_FIXED_MLP_BINDING_ID: Final[str] = (
    "wdl_jungle_fixed_mlp_native_binding_v1"
)
WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID: Final[str] = (
    "wdl_jungle_eed_fixed_mlp_native_binding_v1"
)
WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID: Final[str] = (
    "wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1"
)
PATCH_STABLE_GRU_BINDING_ID: Final[str] = "patch_stable_gru_native_binding_v1"
WDL_STABLE_GRU_BINDING_ID: Final[str] = "wdl_stable_gru_native_binding_v1"
GMNW_POLICY_ARMS_BINDING_ID: Final[str] = "gmnw_policy_arms_native_binding_v1"
E1_NATIVE_METRIC_SCHEMA_VERSION: Final[str] = (
    "eiel_genome_v2_e1_native_cell_metrics_v1"
)
EED_NATIVE_METRIC_SCHEMA_VERSION: Final[str] = (
    "eiel_eed_native_cell_metrics_v2"
)

_GENERATION_SUMMARY_COLUMNS_V1: Final[tuple[str, ...]] = (
    "generation",
    "event_cursor_start",
    "event_cursor_pre_turnover",
    "event_cursor_end",
    "alive_after_turnover",
    "energy_sum_after_turnover",
    "births",
    "deaths",
    "failed_births",
    "action_total",
    "move_total",
    "gather_total",
    "moved_total",
    "gathered_total",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "peak_injury",
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "policy_cache_hit_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
    "patch_food_stock_after_turnover",
    "energy_collected_exact_sum_after_turnover",
    "policy_energy_live_sum_after_turnover",
)
_WDL_GENERATION_METRIC_NAMES_V2: Final[tuple[str, ...]] = (
    "tick_count",
    "event_count",
    "cue_cell_count_sum",
    "active_cell_count_sum",
    "cue_without_active_count",
    "active_without_prior_cue_count",
    "cue_to_active_delay_sum",
    "cue_to_active_delay_count",
    "agent_cue_exposure_count",
    "agent_active_exposure_count",
    "lethal_exposure_count",
    "cue_move_away_count",
    "cue_move_toward_count",
    "cue_stayed_count",
    "pre_hazard_exit_count",
    "hazard_enter_count",
    "hazard_exposure_after_cue_count",
    "damage_total",
    "damage_max",
)
_WDL_FIXED_EVENT_CAPACITY_PROMOTION_POLICY_V1: Final[str] = (
    "wdl_fixed_multicell_event_capacity_batch_max_v1"
)
_EED_FIXED_EVENT_CAPACITY_PROMOTION_POLICY_V1: Final[str] = (
    "eed_fixed_multicell_event_capacity_batch_max_v1"
)


def _exact_nonnegative_metric(value: Any, *, name: str) -> int:
    floating = float(value)
    integer = int(floating)
    if floating != float(integer) or integer < 0:
        raise CampaignProviderError(f"{name} must be an exact nonnegative integer")
    return integer


def _generation_summary_columns_from_capability_v1(
    capability: Any,
) -> tuple[str, ...] | None:
    if not isinstance(capability, Mapping):
        return None
    columns = capability.get("generation_summary_scalar_columns")
    width = capability.get("generation_summary_scalar_layout_width")
    if columns is None and width is None:
        return None
    if (
        not isinstance(columns, (list, tuple))
        or type(width) is not int
        or width != len(columns)
    ):
        raise CampaignProviderError(
            "native generation summary capability has inconsistent layout metadata"
        )
    return tuple(columns)


def _project_native_cell_metrics_v1(
    projections: Sequence[Mapping[str, Any]],
    *,
    cell_ids: Sequence[str],
    direct_result: Mapping[str, Any],
    wdl_generation_metrics: Any | None = None,
    generation_summary_columns: Sequence[str] | None = None,
    turnover_counts: Any | None = None,
    turnover_count_generations: Sequence[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Project bounded native generation authorities to one terminal row per cell."""

    import math
    import torch

    columns = (
        _GENERATION_SUMMARY_COLUMNS_V1
        if generation_summary_columns is None
        else tuple(generation_summary_columns)
    )
    if (
        any(type(name) is not str or not name for name in columns)
        or not set(_GENERATION_SUMMARY_COLUMNS_V1) <= set(columns)
    ):
        raise CampaignProviderError(
            "native generation summary columns do not satisfy the bounded E1 layout"
        )
    summary = direct_result.get("generation_summary_scalar")
    completed = direct_result.get("completed_generations")
    if (
        not isinstance(summary, torch.Tensor)
        or summary.dtype != torch.float64
        or summary.ndim != 3
        or not summary.is_contiguous()
        or type(completed) is not int
        or completed < 1
        or tuple(summary.shape)
        != (completed, len(cell_ids), len(columns))
    ):
        raise CampaignProviderError(
            "native generation summary does not match the bounded E1 metric layout"
        )
    if len(projections) != len(cell_ids):
        raise CampaignProviderError(
            "native E1 metric projection count differs from cell identities"
        )
    wdl = None
    if wdl_generation_metrics is not None:
        if (
            not isinstance(wdl_generation_metrics, torch.Tensor)
            or wdl_generation_metrics.dtype != torch.float64
            or wdl_generation_metrics.ndim != 3
            or tuple(wdl_generation_metrics.shape)
            != (len(cell_ids), completed, len(_WDL_GENERATION_METRIC_NAMES_V2))
        ):
            raise CampaignProviderError(
                "WDL generation metrics do not match the bounded v2 layout"
            )
        wdl = wdl_generation_metrics.detach().to(device="cpu").contiguous()
    compact = summary.detach().to(device="cpu").contiguous()
    turnover = None
    turnover_by_generation: dict[int, Any] = {}
    if turnover_counts is not None:
        if (
            not hasattr(turnover_counts, "detach")
            or turnover_counts.ndim not in {2, 3}
        ):
            raise CampaignProviderError(
                "turnover count metrics do not match the bounded v1 layout"
            )
        turnover = turnover_counts.detach().to(device="cpu").contiguous()
        if turnover.ndim == 2:
            if tuple(turnover.shape) != (len(cell_ids), 8):
                raise CampaignProviderError(
                    "turnover count metrics do not match the bounded v1 layout"
                )
        else:
            if (
                turnover_count_generations is None
                or tuple(turnover.shape[1:]) != (len(cell_ids), 8)
                or len(turnover_count_generations) != int(turnover.shape[0])
                or any(
                    type(generation) is not int or generation < 0
                    for generation in turnover_count_generations
                )
                or len(set(turnover_count_generations))
                != len(turnover_count_generations)
            ):
                raise CampaignProviderError(
                    "turnover count generation metrics do not match the bounded v1 layout"
                )
            turnover_by_generation = {
                int(generation): turnover[index]
                for index, generation in enumerate(turnover_count_generations)
            }
    column = {name: index for index, name in enumerate(columns)}
    wdl_column = {
        name: index for index, name in enumerate(_WDL_GENERATION_METRIC_NAMES_V2)
    }
    rows: list[dict[str, Any]] = []
    for cell, (cell_id, projection) in enumerate(
        zip(cell_ids, projections, strict=True)
    ):
        generation_rows: list[dict[str, Any]] = []
        previous_generation: int | None = None
        for generation_offset in range(completed):
            source = compact[generation_offset, cell]
            generation = _exact_nonnegative_metric(
                source[column["generation"]].item(),
                name="generation",
            )
            if previous_generation is not None and generation != previous_generation + 1:
                raise CampaignProviderError(
                    "native generation metric rows are not a contiguous prefix"
                )
            previous_generation = generation
            generation_row: dict[str, Any] = {
                "generation": generation,
                "alive": _exact_nonnegative_metric(
                    source[column["alive_after_turnover"]].item(),
                    name="alive_after_turnover",
                ),
                "successful_births": _exact_nonnegative_metric(
                    source[column["births"]].item(),
                    name="births",
                ),
            }
            generation_row["ordinary_birth_count"] = generation_row[
                "successful_births"
            ]
            turnover_row = None
            if turnover is not None and turnover.ndim == 2 and generation_offset == completed - 1:
                turnover_row = turnover[cell]
            elif turnover_by_generation:
                window_turnover = turnover_by_generation.get(generation)
                if window_turnover is not None:
                    turnover_row = window_turnover[cell]
            if turnover_row is not None:
                generation_row["turnover_refill_child_count"] = (
                    _exact_nonnegative_metric(
                        turnover_row[6].item(),
                        name="turnover_child_committed_count",
                    )
                )
                generation_row["turnover_final_live_count"] = (
                    _exact_nonnegative_metric(
                        turnover_row[7].item(),
                        name="turnover_final_live_count",
                    )
                )
            if wdl is not None:
                wdl_row = wdl[cell, generation_offset]
                lethal = _exact_nonnegative_metric(
                    wdl_row[wdl_column["lethal_exposure_count"]].item(),
                    name="wdl.lethal_exposure_count",
                )
                damage = float(
                    wdl_row[wdl_column["damage_total"]].item()
                )
                if not math.isfinite(damage) or damage < 0.0:
                    raise CampaignProviderError(
                        "wdl.damage_total must be finite and nonnegative"
                    )
                generation_row["wdl"] = {
                    "lethal_exposure_count": lethal,
                    "damage_total": damage,
                }
            generation_rows.append(generation_row)
        terminal_alive = generation_rows[-1]["alive"]
        successful_births = sum(
            row["successful_births"] for row in generation_rows
        )
        metrics: dict[str, int | float] = {
            "population.alive": terminal_alive,
            "population.terminal_alive": terminal_alive,
            "population.survival_auc": sum(
                row["alive"] for row in generation_rows
            ),
            "population.successful_births_cumulative": successful_births,
            "population.births": successful_births,
            "wdl.lethal_exposure_total": sum(
                row.get("wdl", {}).get("lethal_exposure_count", 0)
                for row in generation_rows
            ),
            "wdl.damage_total": sum(
                row.get("wdl", {}).get("damage_total", 0.0)
                for row in generation_rows
            ),
        }
        turnover_generation_rows = [
            row for row in generation_rows if "turnover_refill_child_count" in row
        ]
        if turnover_generation_rows:
            metrics["population.turnover_refill_children"] = sum(
                int(row["turnover_refill_child_count"])
                for row in turnover_generation_rows
            )
            metrics["population.turnover_final_live"] = int(
                turnover_generation_rows[-1]["turnover_final_live_count"]
            )
        from engine.evolution.genome_upgrade_v2_telemetry import (
            project_genome_upgrade_v2_e1_metric_attribution_v1,
        )

        genome_v2 = project_genome_upgrade_v2_e1_metric_attribution_v1(
            projection,
            expected_cell_id=cell_id,
        )
        if genome_v2 is not None:
            metrics.update(genome_v2["metrics"])
        rows.append(
            {
                "schema_version": E1_NATIVE_METRIC_SCHEMA_VERSION,
                "cell_id": cell_id,
                "metrics": metrics,
                "generation_metrics": generation_rows,
                "genome_v2_attribution": genome_v2,
                "raw_tick_arrays_retained": False,
                "reporting_influence": "forbidden",
            }
        )
    return tuple(rows)


def _compact_terminal_native_metric_rows_v1(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Keep terminal metrics while discarding reporting-only generation rows."""

    return tuple(
        {
            "schema_version": row["schema_version"],
            "cell_id": row["cell_id"],
            "metrics": dict(row["metrics"]),
            "genome_v2_attribution": row["genome_v2_attribution"],
            "raw_tick_arrays_retained": False,
            "reporting_influence": "forbidden",
        }
        for row in rows
    )


def _tick_native_terminal_metric_rows_v1(
    projections: Sequence[Mapping[str, Any]],
    *,
    cell_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Project exact terminal tick snapshots without inventing generation rows."""

    import torch

    if len(projections) != len(cell_ids):
        raise CampaignProviderError(
            "tick-native terminal projections differ from the cell identities"
        )
    rows: list[dict[str, Any]] = []
    for cell_id, projection in zip(cell_ids, projections, strict=True):
        metadata = projection.get("metadata")
        tensors = projection.get("tensors")
        if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
            raise CampaignProviderError(
                "tick-native terminal metric projection is incomplete"
            )
        generation = metadata.get("generation")
        tick = metadata.get("tick")
        interval = metadata.get("configured_steps_per_generation")
        alive = tensors.get("alive_mask")
        if (
            type(generation) is not int
            or generation < 0
            or type(tick) is not int
            or type(interval) is not int
            or interval < 1
            or not 0 <= tick < interval
            or not isinstance(alive, torch.Tensor)
            or alive.dtype != torch.bool
            or alive.ndim != 1
        ):
            raise CampaignProviderError(
                "tick-native terminal metric authority is malformed"
            )
        terminal_alive = int(alive.count_nonzero().item())
        if metadata.get("alive_count") != terminal_alive:
            raise CampaignProviderError(
                "tick-native terminal alive count contradicts its checkpoint"
            )
        from engine.evolution.genome_upgrade_v2_telemetry import (
            project_genome_upgrade_v2_e1_metric_attribution_v1,
        )

        genome_v2 = project_genome_upgrade_v2_e1_metric_attribution_v1(
            projection,
            expected_cell_id=cell_id,
        )
        rows.append(
            {
                "schema_version": E1_NATIVE_METRIC_SCHEMA_VERSION,
                "cell_id": cell_id,
                "metrics": {
                    "population.alive": terminal_alive,
                    "population.terminal_alive": terminal_alive,
                    "runtime.terminal_absolute_tick": generation * interval + tick,
                },
                "genome_v2_attribution": genome_v2,
                "raw_tick_arrays_retained": False,
                "reporting_influence": "forbidden",
            }
        )
    return tuple(rows)


def _prepared_turnover_counts_v1(prepared: Any) -> Any | None:
    patch = getattr(prepared, "patch", None)
    source = getattr(patch, "source", None)
    outputs = getattr(source, "outputs", None)
    if not isinstance(outputs, Mapping):
        return None
    return outputs.get("turnover_counts")


def _turnover_count_authority_v1(
    prepared: Any, direct_result: Mapping[str, Any]
) -> tuple[Any | None, Sequence[int] | None]:
    counts_by_window = direct_result.get("turnover_counts_by_window")
    count_generations = direct_result.get("turnover_count_terminal_generations")
    if counts_by_window is not None or count_generations is not None:
        return counts_by_window, count_generations
    return _prepared_turnover_counts_v1(prepared), None


def _merge_native_cell_metric_snapshots_v1(
    prior_rows: Sequence[Mapping[str, Any]],
    projected_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Merge exact generation prefixes into resumed terminal metric snapshots."""

    if not prior_rows:
        return tuple(dict(row) for row in projected_rows)
    if len(prior_rows) != len(projected_rows):
        raise CampaignProviderError(
            "native terminal metric continuation changed the cell count"
        )
    merged: list[dict[str, Any]] = []

    def merge_cumulative_metrics(
        prior: Mapping[str, Any], projected: Mapping[str, Any]
    ) -> dict[str, Any]:
        prior_metrics = prior.get("metrics")
        projected_metrics = projected.get("metrics")
        if not isinstance(prior_metrics, Mapping) or not isinstance(
            projected_metrics, Mapping
        ):
            raise CampaignProviderError(
                "compact terminal metric continuation lacks cumulative metrics"
            )
        terminal_metrics = dict(projected_metrics)
        for name in (
            "population.survival_auc",
            "population.successful_births_cumulative",
            "population.births",
            "population.deaths",
            "wdl.lethal_exposure_total",
            "wdl.damage_total",
            "population.turnover_refill_children",
        ):
            prior_value = prior_metrics.get(name, 0)
            projected_value = projected_metrics.get(name, 0)
            if (
                isinstance(prior_value, bool)
                or not isinstance(prior_value, (int, float))
                or isinstance(projected_value, bool)
                or not isinstance(projected_value, (int, float))
            ):
                raise CampaignProviderError(
                    "compact terminal metric continuation has invalid cumulative metrics"
                )
            if name in prior_metrics or name in projected_metrics:
                terminal_metrics[name] = prior_value + projected_value
        return terminal_metrics

    for prior_raw, projected_raw in zip(prior_rows, projected_rows, strict=True):
        prior = dict(prior_raw)
        projected = dict(projected_raw)
        schema_version = projected.get("schema_version")
        if (
            schema_version
            not in {E1_NATIVE_METRIC_SCHEMA_VERSION, EED_NATIVE_METRIC_SCHEMA_VERSION}
            or prior.get("schema_version") != schema_version
            or prior.get("cell_id") != projected.get("cell_id")
        ):
            raise CampaignProviderError(
                "native terminal metric continuation identity is invalid"
            )
        prior_series = prior.get("generation_metrics")
        projected_series = projected.get("generation_metrics")
        if prior_series is None and projected_series is None:
            projected["metrics"] = merge_cumulative_metrics(prior, projected)
            merged.append(projected)
            continue
        if (
            isinstance(prior_series, list)
            and prior_series
            and all(isinstance(row, Mapping) for row in prior_series)
            and projected_series is None
            and "generation_metrics" not in projected
            and projected.get("reporting_influence") == "forbidden"
            and projected.get("raw_tick_arrays_retained") is False
        ):
            # Reporting-disabled continuation intentionally projects only a
            # compact terminal snapshot. Preserve exact cumulative metrics
            # without fabricating generation rows that were not retained.
            projected["metrics"] = merge_cumulative_metrics(prior, projected)
            merged.append(projected)
            continue
        if (
            not isinstance(prior_series, list)
            or not isinstance(projected_series, list)
            or not prior_series
            or not projected_series
            or any(not isinstance(row, Mapping) for row in prior_series)
            or any(not isinstance(row, Mapping) for row in projected_series)
        ):
            raise CampaignProviderError(
                "native terminal metric continuation series is invalid"
            )
        generation_series = [
            *(dict(row) for row in prior_series),
            *(dict(row) for row in projected_series),
        ]
        generations = [row.get("generation") for row in generation_series]
        if (
            any(type(generation) is not int for generation in generations)
            or any(
                right != left + 1
                for left, right in zip(generations, generations[1:])
            )
        ):
            raise CampaignProviderError(
                "native terminal metric continuation is not contiguous"
            )
        metrics = projected.get("metrics")
        if not isinstance(metrics, Mapping):
            raise CampaignProviderError(
                "native terminal metric continuation lacks cumulative metrics"
            )
        alive_values = [row.get("alive") for row in generation_series]
        birth_values = [
            row.get("successful_births") for row in generation_series
        ]
        death_values = [row.get("death_rows", 0) for row in generation_series]
        if (
            any(type(value) is not int or value < 0 for value in alive_values)
            or any(type(value) is not int or value < 0 for value in birth_values)
            or any(type(value) is not int or value < 0 for value in death_values)
        ):
            raise CampaignProviderError(
                "native terminal metric continuation population values are invalid"
            )
        successful_births = sum(birth_values)
        deaths = sum(death_values)
        terminal_metrics = dict(metrics)
        terminal_metrics.update(
            {
                "population.alive": alive_values[-1],
                "population.terminal_alive": alive_values[-1],
                "population.survival_auc": sum(alive_values),
                "population.successful_births_cumulative": successful_births,
                "population.births": successful_births,
                "population.deaths": deaths,
                "wdl.lethal_exposure_total": sum(
                    row.get("wdl", {}).get("lethal_exposure_count", 0)
                    for row in generation_series
                ),
                "wdl.damage_total": sum(
                    row.get("wdl", {}).get("damage_total", 0.0)
                    for row in generation_series
                ),
            }
        )
        if any("turnover_refill_child_count" in row for row in generation_series):
            terminal_metrics["population.turnover_refill_children"] = sum(
                row.get("turnover_refill_child_count", 0)
                for row in generation_series
            )
        if schema_version == EED_NATIVE_METRIC_SCHEMA_VERSION:
            from engine.campaigns.eed_opportunity_metrics_v1 import (
                EEDOpportunityMeasurementError,
                merge_eed_opportunity_metrics_v1,
            )

            try:
                opportunity_metrics, opportunity_audit = (
                    merge_eed_opportunity_metrics_v1(
                        prior,
                        projected,
                        generation_series,
                    )
                )
            except EEDOpportunityMeasurementError as exc:
                raise CampaignProviderError(str(exc)) from exc
            terminal_metrics.update(
                {
                    **opportunity_metrics,
                    "eed.reproductive_opportunity_successful_births": (
                        successful_births
                    ),
                    "eed.reproductive_opportunity_survival_auc": (
                        sum(alive_values)
                    ),
                }
            )
            projected["eed_opportunity_measurement"] = opportunity_audit
        projected["metrics"] = terminal_metrics
        projected["generation_metrics"] = generation_series
        merged.append(projected)
    return tuple(merged)


def _patch_fixed_request_adapter(runner: Any, **kwargs: Any) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
        build_direct_resident_source_allocation_request_v1,
    )

    kwargs.pop("runner_build", None)
    kwargs.pop("cell", None)
    return build_direct_resident_source_allocation_request_v1(runner, **kwargs)


def _patch_fixed_prepare_adapter(**kwargs: Any) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        prepare_direct_resident_patch_multicell_v1,
    )

    kwargs.pop("plan", None)
    kwargs.pop("cells", None)
    return prepare_direct_resident_patch_multicell_v1(**kwargs)


def _patch_fixed_project_adapter(
    prepared: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    **kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        project_direct_resident_patch_multicell_checkpoints_v1,
    )

    settlement = project_direct_resident_patch_multicell_checkpoints_v1(
        prepared,
        direct_result=direct_result,
        reporting_enabled=reporting_enabled,
        **kwargs,
    )
    turnover_counts, turnover_count_generations = _turnover_count_authority_v1(
        prepared, direct_result
    )
    native_rows = _project_native_cell_metrics_v1(
        settlement.checkpoint_projections,
        cell_ids=settlement.cell_ids,
        direct_result=direct_result,
        generation_summary_columns=_generation_summary_columns_from_capability_v1(
            kwargs.get("capability")
        ),
        turnover_counts=turnover_counts,
        turnover_count_generations=turnover_count_generations,
    )
    rows = (
        native_rows
        if reporting_enabled
        else _compact_terminal_native_metric_rows_v1(native_rows)
    )
    return replace(settlement, compact_result_rows=rows)


_PATCH_FIXED_CPU_CAPABILITY_MASK_V1: Final[Mapping[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
}
_WDL_JUNGLE_FIXED_CPU_CAPABILITY_MASK_V1: Final[Mapping[str, bool]] = {
    **_PATCH_FIXED_CPU_CAPABILITY_MASK_V1,
    "wdl": True,
    "wdl_storage": True,
    "jungle": True,
}


def _cpu_capability_mask_for_runner_v1(runner: Any) -> Mapping[str, bool]:
    env_cfg = getattr(runner, "env_cfg", None)
    sim_cfg = getattr(runner, "sim_cfg", None)
    if bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)) or bool(
        getattr(env_cfg, "jungle_predator_pressure_enable", False)
    ):
        mask = dict(_WDL_JUNGLE_FIXED_CPU_CAPABILITY_MASK_V1)
    else:
        mask = dict(_PATCH_FIXED_CPU_CAPABILITY_MASK_V1)
    if bool(getattr(sim_cfg, "genome_upgrade_v2_enable", False)):
        mask["genome_v2"] = True
    return mask


def _patch_fixed_cpu_generation(projection: Mapping[str, Any]) -> int:
    metadata = projection.get("metadata")
    generation = metadata.get("generation") if isinstance(metadata, Mapping) else None
    if type(generation) is not int or generation < 0:
        raise CampaignProviderError("CPU-native Patch projection lacks generation")
    tick = metadata.get("tick", 0)
    if tick != 0:
        raise CampaignProviderError("CPU-native Patch projection is not at a boundary")
    return generation


def _patch_fixed_cpu_artifact_paths_v1() -> tuple[str, str, str]:
    from engine.perf.cxxbackbone10_resident_state_owner import (
        resident_state_owner_extension_status,
    )

    status = resident_state_owner_extension_status()
    status_payload = status.to_dict()
    extension_path = (
        status_payload.get("extension_path", "")
        if status_payload.get("available") is True
        else ""
    )
    if not isinstance(extension_path, str):
        extension_path = ""
    return (extension_path, "", "")


def _patch_fixed_cpu_artifact_loader(
    artifact_path: str = "",
    qualification_receipt_path: str = "",
    reporting_receipt_path: str = "",
) -> Any:
    from engine.perf.cxxbackbone10_resident_state_owner import (
        resident_state_owner_extension_status,
    )

    if qualification_receipt_path or reporting_receipt_path:
        raise CampaignProviderError(
            "CPU-native Patch cannot bind CUDA qualification or reporting receipts"
        )
    status = resident_state_owner_extension_status()
    if not status.available:
        raise CampaignProviderError(
            "CPU-native Patch production requires a verified prebuilt resident "
            f"owner extension: {status.error}"
        )
    status_payload = status.to_dict()
    expected_path = status_payload.get("extension_path")
    if (
        status_payload.get("available") is not True
        or status_payload.get("abi_compatible") is not True
        or status_payload.get("output_sha256_verified") is not True
        or status_payload.get("source_inventory_verified") is not True
        or not isinstance(expected_path, str)
        or not expected_path
        or artifact_path != expected_path
    ):
        raise CampaignProviderError(
            "CPU-native Patch artifact path differs from its verified resident "
            "owner extension"
        )
    return SimpleNamespace(
        module=SimpleNamespace(
            cpu_native_profile=True,
            resident_owner_extension=status_payload,
        ),
        production_authority=True,
    )


def _patch_fixed_cpu_runner_builder_adapter(*args: Any, **kwargs: Any) -> Any:
    from engine.campaigns.patch_runner_factory import build_patch_fixed_runner_v1

    kwargs["execution_backend"] = "cpu_native"
    return build_patch_fixed_runner_v1(*args, **kwargs)


def _wdl_jungle_fixed_cpu_runner_builder_adapter(
    *args: Any, **kwargs: Any
) -> Any:
    from engine.campaigns.patch_runner_factory import (
        build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
        build_patch_wdl_jungle_fixed_mlp_runner_v1,
    )

    cell = args[0] if args else None
    parameters = (
        cell.get("parameters")
        if isinstance(cell, Mapping)
        else None
    )
    if (
        isinstance(parameters, Mapping)
        and "eed.resource_impact_apply_enable" in parameters
    ):
        builder = build_patch_wdl_jungle_eed_fixed_mlp_runner_v1
    else:
        builder = build_patch_wdl_jungle_fixed_mlp_runner_v1

    kwargs["execution_backend"] = "cpu_native"
    return builder(*args, **kwargs)


def _patch_fixed_cpu_source_preparer(
    _plan: Mapping[str, Any],
    request: Any,
    context: Any,
    *,
    runner_builder: Callable[..., Any],
    cells: Sequence[Mapping[str, Any]],
    artifact: str,
    qualification: str,
    reporting: str,
    segment_generations: int,
    **_kwargs: Any,
) -> tuple[Mapping[str, Any], ...]:
    from engine.perf.cxxbackbone10_resident_state_owner import ResidentRuntimeState

    projections: list[Mapping[str, Any]] = []
    for cell, cell_id in zip(cells, request.cell_ids, strict=True):
        build = runner_builder(
            cell,
            context.batch_root
            / "cells"
            / cell_id
            / "segment-000000-000000-source",
            generations=segment_generations,
            ticks_per_generation=request.ticks_per_generation,
            artifact_path=artifact,
            qualification_receipt_path=qualification,
            reporting_receipt_path=reporting,
            execution_backend="cpu_native",
            **(
                {
                    "genome_v2_checkpoint_authority_binding": dict(
                        request.genome_v2_checkpoint_authority_binding
                    )
                }
                if getattr(request, "genome_v2_checkpoint_authority_binding", ())
                else {}
            ),
        )
        owner = ResidentRuntimeState.from_runner(
            build.runner,
            _cpu_capability_mask_for_runner_v1(build.runner),
            generation=0,
            tick=0,
            owner_device="cpu",
        )
        projection = owner.export_checkpoint_projection()
        if _patch_fixed_cpu_generation(projection) != 0:
            raise CampaignProviderError(
                "CPU-native Patch source projection is not generation zero"
            )
        projections.append(projection)
    return tuple(projections)


def _patch_fixed_cpu_request_adapter(
    runner: Any,
    *,
    logical_generation_count: int,
    source_checkpoint_projection: Mapping[str, Any] | None,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cxxbackbone10_resident_state_owner import ResidentRuntimeState

    if source_checkpoint_projection is None:
        owner = ResidentRuntimeState.from_runner(
            runner,
            _cpu_capability_mask_for_runner_v1(runner),
            generation=0,
            tick=0,
            owner_device="cpu",
        )
    else:
        owner = ResidentRuntimeState.from_checkpoint_projection(
            source_checkpoint_projection,
            owner_device="cpu",
        )
    projection = owner.export_checkpoint_projection()
    ticks_per_generation = int(
        getattr(getattr(runner, "sim_cfg", None), "steps_per_generation", 0)
        or 0
    )
    if ticks_per_generation < 1:
        raise CampaignProviderError(
            "CPU-native Patch request lacks ticks_per_generation"
        )
    ordinary_births_enabled = bool(
        getattr(getattr(runner, "sim_cfg", None), "ordinary_births_enabled", True)
    )
    max_births_per_generation = int(
        getattr(
            getattr(runner, "sim_cfg", None),
            "native_runtime_max_births_per_generation",
            64,
        )
        or 64
    )
    if not ordinary_births_enabled:
        max_births_per_generation = 0
    return SimpleNamespace(
        owner=owner,
        import_projection=projection,
        source_checkpoint_projection=source_checkpoint_projection,
        logical_generation_count=int(logical_generation_count),
        ticks_per_generation=ticks_per_generation,
        max_births_per_generation=max_births_per_generation,
        ordinary_births_enabled=ordinary_births_enabled,
        generation_turnover_enabled=bool(
            getattr(
                getattr(runner, "sim_cfg", None),
                "generation_turnover_enabled",
                False,
            )
        ),
        population_size=int(
            getattr(getattr(runner, "sim_cfg", None), "population_size", 0) or 0
        ),
        cuda_capability={"cpu_native": True},
        cuda_device="cpu",
    )


def _patch_fixed_cpu_prepare_adapter(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any],
    logical_generation_count: int,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    **_kwargs: Any,
) -> PreparedNativeMechanicsOwnerV1:
    if len(cell_ids) != 1 or len(per_cell_requests) != 1:
        raise CampaignProviderError("CPU-native Patch production is B1 only")
    request = per_cell_requests[0]
    source = request.source_checkpoint_projection
    if resume_projections is not None:
        if len(resume_projections) != 1 or not isinstance(source, Mapping):
            raise CampaignProviderError(
                "CPU-native Patch resume projection was substituted"
            )
        resumed = resume_projections[0]
        if not isinstance(resumed, Mapping):
            raise CampaignProviderError(
                "CPU-native Patch resume projection was substituted"
            )
        source_restore_count = source.get("restore_count")
        resumed_restore_count = resumed.get("restore_count")
        if (
            type(source_restore_count) is not int
            or source_restore_count < 0
            or type(resumed_restore_count) is not int
            or resumed_restore_count != source_restore_count + 1
        ):
            raise CampaignProviderError(
                "CPU-native Patch resume projection restore counter is not authoritative"
            )
        from engine.native_runtime.checkpoint_digests import (
            checkpoint_source_projection_digest_manifest,
        )

        normalized_resumed = dict(resumed)
        normalized_resumed["restore_count"] = source_restore_count
        try:
            source_digest = checkpoint_source_projection_digest_manifest(source)[
                "digest"
            ]
            resumed_digest = checkpoint_source_projection_digest_manifest(
                normalized_resumed
            )["digest"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                "CPU-native Patch resume projection could not be verified"
            ) from exc
        if source_digest != resumed_digest:
            raise CampaignProviderError(
                "CPU-native Patch resume projection was substituted"
            )
    source_projection = dict(request.import_projection)
    initial_generation = _patch_fixed_cpu_generation(source_projection)
    owner = SimpleNamespace(
        owners=(request.owner,),
        cell_ids=tuple(cell_ids),
        source_projections=(source_projection,),
        initial_generation=initial_generation,
        logical_generation_count=int(logical_generation_count),
        ticks_per_generation=int(request.ticks_per_generation),
        max_births_per_generation=int(request.max_births_per_generation),
        ordinary_births_enabled=bool(request.ordinary_births_enabled),
        generation_turnover_enabled=bool(request.generation_turnover_enabled),
        population_size=int(request.population_size),
    )
    return PreparedNativeMechanicsOwnerV1(
        physical_window_arguments=owner,
        source_projections=(source_projection,),
        cell_ids=tuple(cell_ids),
    )


def _patch_fixed_cpu_execute_adapter(
    _module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    prepared = physical_window_arguments
    owners = tuple(getattr(prepared, "owners", ()))
    if len(owners) != 1:
        raise CampaignProviderError("CPU-native Patch execution requires B1")
    owner = owners[0]
    initial_generation = int(prepared.initial_generation)
    ticks_per_generation = int(prepared.ticks_per_generation)
    if ticks_per_generation < 1:
        raise CampaignProviderError(
            "CPU-native Patch execution lacks ticks_per_generation"
        )
    ordinary_births_enabled = bool(prepared.ordinary_births_enabled)
    max_births = int(prepared.max_births_per_generation)
    if not ordinary_births_enabled:
        max_births = 0
    generation_turnover_enabled = bool(prepared.generation_turnover_enabled)
    population_size = int(prepared.population_size)
    source_projections = tuple(getattr(prepared, "source_projections", ()))
    source_metadata = (
        source_projections[0].get("metadata")
        if source_projections and isinstance(source_projections[0], Mapping)
        else None
    )
    mate_required = (
        bool(source_metadata.get("reproduce_requires_mate", False))
        if isinstance(source_metadata, Mapping)
        else False
    )
    raw_results: list[dict[str, Any]] = []
    for offset in range(int(logical_generation_count)):
        generation = initial_generation + offset
        if mate_required:
            context_result = dict(owner.execute_wdl_jungle_context_cpu())
            if context_result.get("hidden_fallback") is not False:
                raise CampaignProviderError(
                    "CPU-native WDL/Jungle context did not prove no hidden fallback"
                )
            birth = dict(
                owner.execute_population_asexual_resident_birth_commit_cpu(
                    max_births
                )
            )
            if birth.get("hidden_fallback") is not False:
                raise CampaignProviderError(
                    "CPU-native mate-required birth did not prove no hidden fallback"
                )
            turnover = dict(owner.execute_population_generation_turnover_cpu())
            if turnover.get("hidden_fallback") is not False:
                raise CampaignProviderError(
                    "CPU-native mate-required hardcap turnover did not prove no "
                    "hidden fallback"
                )
            if _strict_nonnegative_int_mapping_value_v1(
                turnover, "next_generation"
            ) != generation + 1 or _strict_nonnegative_int_mapping_value_v1(
                turnover, "next_tick"
            ) != 0:
                raise CampaignProviderError(
                    "CPU-native mate-required hardcap turnover advanced to an "
                    "invalid boundary"
                )
            alive_after = _strict_nonnegative_int_mapping_value_v1(
                turnover, "alive_after"
            )
            if population_size > 0 and alive_after != population_size:
                raise CampaignProviderError(
                    "CPU-native mate-required hardcap turnover did not restore "
                    "population size"
                )
            raw = {
                **birth,
                "generation_index": generation,
                "alive_count": alive_after,
                "death_rows": _strict_nonnegative_int_mapping_value_v1(
                    turnover, "culled_rows"
                ),
                "wdl_jungle_context": context_result,
                "population_generation_turnover": turnover,
            }
        else:
            raw = dict(
                owner.execute_generation_patch_with_population_birth_commit_cpu(
                    generation,
                    ticks_per_generation,
                    max_births,
                )
            )
        if raw.get("hidden_fallback") is not False:
            raise CampaignProviderError(
                "CPU-native Patch owner did not prove no hidden fallback"
            )
        committed_birth_rows = _strict_nonnegative_int_mapping_value_v1(
            raw, "committed_birth_rows"
        )
        if not ordinary_births_enabled and committed_birth_rows != 0:
            raise CampaignProviderError(
                "CPU-native Patch turnover-only execution committed ordinary births"
            )
        if generation_turnover_enabled and not mate_required:
            turnover = dict(owner.execute_population_generation_turnover_cpu())
            if turnover.get("hidden_fallback") is not False:
                raise CampaignProviderError(
                    "CPU-native Patch turnover did not prove no hidden fallback"
                )
            if _strict_nonnegative_int_mapping_value_v1(
                turnover, "next_generation"
            ) != generation + 1 or _strict_nonnegative_int_mapping_value_v1(
                turnover, "next_tick"
            ) != 0:
                raise CampaignProviderError(
                    "CPU-native Patch turnover advanced to an invalid boundary"
                )
            alive_after = _strict_nonnegative_int_mapping_value_v1(
                turnover, "alive_after"
            )
            if population_size > 0 and alive_after != population_size:
                raise CampaignProviderError(
                    "CPU-native Patch turnover did not restore population size"
                )
            raw["population_generation_turnover"] = turnover
        raw_results.append(raw)
    projection = owner.export_checkpoint_projection()
    terminal_generation = _patch_fixed_cpu_generation(projection)
    expected_terminal = initial_generation + int(logical_generation_count)
    if terminal_generation != expected_terminal:
        raise CampaignProviderError(
            "CPU-native Patch terminal generation differs from request"
        )
    return {
        "owner": "cpu_native_extension",
        "hidden_fallback": False,
        "runtime_jit": False,
        "completed_generations": int(logical_generation_count),
        "initial_generation": initial_generation,
        "terminal_generation": terminal_generation,
        "checkpoint_projections": (projection,),
        "cpu_generation_results": tuple(raw_results),
        "cross_path_hidden_fallback": False,
        # The CPU owner completes synchronously and has no CUDA stream or
        # device-to-host settlement.  Publish the same closed ACS completion
        # surface as the native CUDA route, with explicit not-applicable/zero
        # values, so ACS can keep its fail-closed receipt validation without
        # treating an omitted CPU field as an implementation failure.
        "final_stream_synchronization_time_ns": 0,
        "last_useful_kernel_completion_observed_host_time_ns": time.time_ns(),
        "final_host_synchronizations": 0,
        "final_device_to_host_operations": 0,
        "compact_settlement_d2h_operations": 0,
        "compact_summary_reporting_storage": {
            "schema_version": (
                "eiel_cuda_architecture_b_compact_summary_reporting_integration_v1"
            ),
            "enabled": False,
            "native_producer_supported": False,
            "final_transfer": {
                "boundary": "not_applicable_cpu",
                "contract_host_transfer_count": 0,
                "observed_host_transfer_count": 0,
                "source_tensor": "not_applicable_cpu",
                "dtype": "not_applicable",
                "shape": [],
                "bytes": 0,
                "scratch_included": False,
                "event_cursors_included": False,
                "status": "disabled",
            },
        },
    }


def _strict_nonnegative_int_mapping_value_v1(
    row: Mapping[str, Any], name: str
) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignProviderError(f"CPU-native Patch result {name} is invalid")
    return int(value)


def _patch_fixed_cpu_project_adapter(
    prepared_adapter: PreparedNativeMechanicsOwnerV1,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        DirectResidentPatchMulticellSettlementV1,
    )

    projections = tuple(direct_result.get("checkpoint_projections", ()))
    cell_ids = tuple(prepared_adapter.cell_ids)
    completed = direct_result.get("completed_generations")
    terminal_generation = direct_result.get("terminal_generation")
    if (
        len(projections) != len(cell_ids)
        or type(completed) is not int
        or type(terminal_generation) is not int
    ):
        raise CampaignProviderError("CPU-native Patch result is malformed")
    raw_results = direct_result.get("cpu_generation_results")
    if (
        isinstance(raw_results, (str, bytes))
        or not isinstance(raw_results, Sequence)
        or len(raw_results) != completed
        or not all(isinstance(row, Mapping) for row in raw_results)
    ):
        raise CampaignProviderError("CPU-native Patch generation results are malformed")
    generation_rows: list[dict[str, Any]] = []
    for raw in raw_results:
        generation = _strict_nonnegative_int_mapping_value_v1(raw, "generation_index")
        ordinary_births = _strict_nonnegative_int_mapping_value_v1(
            raw, "committed_birth_rows"
        )
        generation_row: dict[str, Any] = {
            "generation": generation,
            "alive": _strict_nonnegative_int_mapping_value_v1(raw, "alive_count"),
            "successful_births": ordinary_births,
            "ordinary_birth_count": ordinary_births,
            "death_rows": _strict_nonnegative_int_mapping_value_v1(raw, "death_rows"),
        }
        turnover = raw.get("population_generation_turnover")
        if turnover is not None:
            if not isinstance(turnover, Mapping):
                raise CampaignProviderError("CPU-native Patch turnover result is malformed")
            generation_row["turnover_refill_child_count"] = (
                _strict_nonnegative_int_mapping_value_v1(turnover, "committed_refill_rows")
            )
            generation_row["turnover_final_live_count"] = (
                _strict_nonnegative_int_mapping_value_v1(turnover, "alive_after")
            )
            generation_row["alive"] = generation_row["turnover_final_live_count"]
        generation_rows.append(generation_row)
    ordinary_birth_total = sum(row["ordinary_birth_count"] for row in generation_rows)
    turnover_refill_total = sum(
        row.get("turnover_refill_child_count", 0) for row in generation_rows
    )
    alive_total = sum(row["alive"] for row in generation_rows)
    rows = tuple(
        {
            "cell_id": cell_id,
            "schema_version": E1_NATIVE_METRIC_SCHEMA_VERSION,
            "generation": terminal_generation,
            "metrics": {
                "population.alive": int(projection["tensors"]["alive_mask"].sum().item()),
                "population.terminal_alive": int(projection["tensors"]["alive_mask"].sum().item()),
                "population.survival_auc": alive_total,
                "population.births": ordinary_birth_total,
                "population.successful_births_cumulative": ordinary_birth_total,
                "population.turnover_refill_children": turnover_refill_total,
                "population.deaths": sum(
                    _strict_nonnegative_int_mapping_value_v1(raw, "death_rows")
                    for raw in raw_results
                ),
                "wdl.lethal_exposure_total": 0,
                "wdl.damage_total": 0.0,
            },
            **({"generation_metrics": generation_rows} if reporting_enabled else {}),
        }
        for cell_id, projection in zip(cell_ids, projections, strict=True)
    )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=cell_ids,
        checkpoint_projections=projections,
        compact_result_rows=rows,
        completed_generation_count=completed,
        terminal_generation=terminal_generation,
    )


def _patch_fixed_cpu_generation_zero_projector(
    prepared_adapter: PreparedNativeMechanicsOwnerV1,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(prepared_adapter.source_projections)


def patch_fixed_mlp_cpu_native_adapters_v1() -> dict[str, Any]:
    """Return CPU-native B1 adapters for the Patch fixed-MLP route."""

    return {
        "artifact_loader": _patch_fixed_cpu_artifact_loader,
        "artifact_paths": _patch_fixed_cpu_artifact_paths_v1(),
        "runner_builder": _patch_fixed_cpu_runner_builder_adapter,
        "request_builder": _patch_fixed_cpu_request_adapter,
        "preparer": _patch_fixed_cpu_prepare_adapter,
        "executor": _patch_fixed_cpu_execute_adapter,
        "projector": _patch_fixed_cpu_project_adapter,
        "source_preparer": _patch_fixed_cpu_source_preparer,
        "generation_zero_checkpoint_projector": (
            _patch_fixed_cpu_generation_zero_projector
        ),
        "terminal_snapshot_metrics_merger": _merge_native_cell_metric_snapshots_v1,
    }


def wdl_jungle_fixed_mlp_cpu_native_adapters_v1() -> dict[str, Any]:
    """Return CPU-native B1 adapters for the integrated WDL+Jungle route."""

    adapters = patch_fixed_mlp_cpu_native_adapters_v1()
    adapters["runner_builder"] = _wdl_jungle_fixed_cpu_runner_builder_adapter
    return adapters


def wdl_jungle_eed_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose one shared B1..B256 Patch/WDL/Jungle/EED CUDA owner."""

    from engine.campaigns.patch_runner_factory import (
        build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        project_direct_resident_wdl_jungle_generation_zero_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_wdl_jungle_eed_fixed_mlp_native_v1",
        batch_max=256,
        generation_min=3,
        generation_max=128,
        ticks_per_generation=160,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
        request_builder=_wdl_fixed_request_adapter,
        preparer=_wdl_jungle_eed_prepare_adapter,
        executor=_wdl_jungle_execute_adapter,
        projector=_wdl_jungle_project_adapter,
        generation_zero_checkpoint_projector=(
            project_direct_resident_wdl_jungle_generation_zero_v1
        ),
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1",
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2",
        ),
        admitted_ticks_per_generation=(160,),
    )


def _wdl_jungle_eed_hapb_current_source_artifact_loader(
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
) -> Any:
    """Load a HAPB remint bound by the campaign spec for QualPrep Launchpad.

    Historical qualification receipts keep ABI/provider identity. They do not
    pin the reminted instance hash; Genome V2 package identity does that.
    """

    import hashlib
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    artifact = Path(artifact_path).expanduser().resolve()
    if not artifact.is_file():
        raise CampaignProviderError("HAPB current-source artifact is missing")
    try:
        qualification = json.loads(
            Path(qualification_receipt_path).expanduser().read_text(encoding="utf-8")
        )
        reporting = json.loads(
            Path(reporting_receipt_path).expanduser().read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            f"HAPB current-source qualification companion is unreadable: {exc}"
        ) from exc
    if not isinstance(qualification, Mapping) or qualification.get("ok") is not True:
        raise CampaignProviderError(
            "HAPB current-source qualification companion is invalid"
        )
    if not isinstance(reporting, Mapping) or reporting.get("ok") is not True:
        raise CampaignProviderError(
            "HAPB current-source reporting companion is invalid"
        )
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise CampaignProviderError(
            "PyTorch is required to load the HAPB current-source artifact"
        ) from exc
    spec = importlib.util.spec_from_file_location(
        "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1", artifact
    )
    if spec is None or spec.loader is None:
        raise CampaignProviderError("HAPB current-source artifact import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1",
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1",
        "eiel_cuda_architecture_b_hapb_ordinary_birth_mutate_rows_v1",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    if missing:
        raise CampaignProviderError(
            "HAPB current-source artifact is missing exports: " + ", ".join(missing)
        )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return SimpleNamespace(
        module=module,
        candidate_authority_sha256=digest,
        production_authority=False,
        qualification_receipt=dict(qualification),
        reporting_receipt=dict(reporting),
    )


def wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose the separate host-authoritative B48 WDL/Jungle/EED route."""

    from engine.campaigns.patch_runner_factory import (
        build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        export_direct_resident_wdl_jungle_hapb_component_checkpoint_v1,
        project_direct_resident_wdl_jungle_generation_zero_v1,
        restore_direct_resident_wdl_jungle_hapb_component_checkpoint_v1,
        restore_direct_resident_wdl_jungle_hapb_component_sources_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_wdl_jungle_eed_hapb_fixed_mlp_native_v1",
        batch_max=256,
        generation_min=3,
        generation_max=128,
        ticks_per_generation=160,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
        request_builder=_wdl_fixed_request_adapter,
        preparer=_wdl_jungle_eed_hapb_birth_prepare_adapter,
        executor=_wdl_jungle_eed_hapb_birth_execute_adapter,
        projector=_wdl_jungle_project_adapter,
        generation_zero_checkpoint_projector=(
            project_direct_resident_wdl_jungle_generation_zero_v1
        ),
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        component_checkpoint_exporter=(
            export_direct_resident_wdl_jungle_hapb_component_checkpoint_v1
        ),
        component_checkpoint_restorer=(
            restore_direct_resident_wdl_jungle_hapb_component_checkpoint_v1
        ),
        component_checkpoint_source_restorer=(
            restore_direct_resident_wdl_jungle_hapb_component_sources_v1
        ),
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1",
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1",
            "eiel_cuda_architecture_b_hapb_ordinary_birth_mutate_rows_v1",
        ),
        artifact_loader=_wdl_jungle_eed_hapb_current_source_artifact_loader,
        abi_bound_artifact_ids=(
            "cuda_extension",
            "extension_manifest",
        ),
        admitted_ticks_per_generation=(160,),
    )


def patch_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose existing Patch and fixed-MLP mechanics under the engine."""

    from engine.campaigns.patch_runner_factory import build_patch_fixed_runner_v1
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        project_direct_resident_patch_multicell_generation_zero_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
        execute_direct_resident_windows_v1,
    )
    return NativeMechanicsBindingV1(
        binding_id=PATCH_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_patch_p2_stage3_v2",
        batch_max=65535,
        generation_min=3,
        generation_max=128,
        ticks_per_generation=80,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_fixed_runner_v1,
        request_builder=_patch_fixed_request_adapter,
        preparer=_patch_fixed_prepare_adapter,
        executor=execute_direct_resident_windows_v1,
        projector=_patch_fixed_project_adapter,
        generation_zero_checkpoint_projector=(
            project_direct_resident_patch_multicell_generation_zero_v1
        ),
        terminal_snapshot_metrics_merger=(
            _merge_native_cell_metric_snapshots_v1
        ),
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_windows_v1",
        ),
        admitted_ticks_per_generation=(80, 160),
    )


_TICK_NATIVE_FORBIDDEN_SOURCE_COMPATIBILITY_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "generations",
        "generation_count",
        "logical_generation_count",
        "physical_window_generations",
        "ticks_per_generation",
        "refill_chunk_ticks",
        "legacy_turnover_interval_ticks",
    }
)


def _tick_native_private_source_capacity_window_count_v1(
    control: Mapping[str, Any],
) -> int:
    """Validate native control and derive only private source-buffer width.

    The returned integer is retained solely for the generation-shaped direct
    CUDA allocation ABI.  It is not a runtime generation count or a time/RNG
    authority; absolute tick spans remain authoritative at the execute seam.
    """

    contaminated = sorted(
        _TICK_NATIVE_FORBIDDEN_SOURCE_COMPATIBILITY_FIELDS_V1.intersection(control)
    )
    if contaminated:
        raise CampaignProviderError(
            "WDL/fixed-MLP tick-native control contains retired compatibility "
            f"fields: {contaminated}"
        )
    if control.get("time_authority") != "absolute_tick_v1":
        raise CampaignProviderError(
            "WDL/fixed-MLP tick-native request control has the wrong time authority"
        )
    owner = control.get("resident_owner_span_ticks")
    dispatch = control.get("native_dispatch_window_ticks")
    source = control.get("source_window_ticks")
    if any(type(value) is not int or value < 1 for value in (owner, dispatch, source)):
        raise CampaignProviderError(
            "WDL/fixed-MLP tick-native source allocation spans are invalid"
        )
    if dispatch > owner or source > owner:
        raise CampaignProviderError(
            "WDL/fixed-MLP tick-native source allocation spans exceed owner span"
        )
    return max(2, (dispatch + source - 1) // source)


def _wdl_fixed_request_adapter(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int,
    physical_window_generations: int = 2,
    agent_id_allocation_generation_count: int,
    source_checkpoint_projection: Mapping[str, Any] | None,
    source_capability_mask: Mapping[str, bool] | None = None,
    allow_short_technical_gate_generation_count: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    direct_batch_host_import_enable: bool = False,
    **_kwargs: Any,
) -> Any:
    if (
        type(physical_window_generations) is not int
        or physical_window_generations not in {1, 2}
    ):
        raise CampaignProviderError(
            "WDL/fixed-MLP physical window generations must be exact integer 1 or 2"
        )
    if source_capability_mask is not None and not isinstance(
        source_capability_mask, Mapping
    ):
        raise CampaignProviderError(
            "WDL/fixed-MLP source capability mask must be a mapping"
        )
    if type(direct_batch_host_import_enable) is not bool:
        raise CampaignProviderError(
            "WDL/fixed-MLP direct batch host import enable must be an exact bool"
        )
    native_control = dict(_kwargs.get("tick_native_control") or {})
    if native_control:
        _tick_native_private_source_capacity_window_count_v1(native_control)
    return SimpleNamespace(
        runner=runner,
        qualified_cuda_module=qualified_cuda_module,
        logical_generation_count=logical_generation_count,
        physical_window_generations=physical_window_generations,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        source_checkpoint_projection=source_checkpoint_projection,
        source_capability_mask=(
            None
            if source_capability_mask is None
            else dict(source_capability_mask)
        ),
        allow_short_technical_gate_generation_count=(
            bool(allow_short_technical_gate_generation_count)
        ),
        allow_partial_generation_source_cursor=(
            bool(allow_partial_generation_source_cursor)
        ),
        # These fields are carried through the request adapter so source
        # preparation and the native owner receive the same plan-bound
        # active-resident contract.  The legacy logical-generation members
        # above remain an ABI projection only.
        tick_native_control=native_control,
        resident_owner_span_ticks=native_control.get("resident_owner_span_ticks"),
        native_dispatch_window_ticks=native_control.get(
            "native_dispatch_window_ticks"
        ),
        source_window_ticks=native_control.get("source_window_ticks"),
        event_window_capacity=native_control.get("event_window_capacity"),
        event_history_sink_path=_kwargs.get("event_history_sink_path"),
        event_history_sink_cell_index=_kwargs.get(
            "event_history_sink_cell_index", 0
        ),
        # Streamed source continuation owns exactly one detached payload at a
        # time.  Preserve the provider's release/failure callbacks on the
        # thin wrapper so the streamed preparer can close that payload before
        # requesting the successor cell.
        source_import_release=_kwargs.get("source_import_release"),
        source_import_fail=_kwargs.get("source_import_fail"),
        direct_batch_host_import_enable=direct_batch_host_import_enable,
        cuda_device="cuda",
        cuda_capability={},
    )


def _wdl_fixed_request_event_capacity_v1(request: Any, *, cell: int) -> int:
    event_capacity = getattr(request, "event_capacity", None)
    if type(event_capacity) is not int or event_capacity <= 0:
        raise CampaignProviderError(
            "WDL/fixed-MLP source request event_capacity must be a positive "
            f"exact integer for cell {cell}"
        )
    return event_capacity


def _wdl_fixed_projection_identity_signature_v1(projection: Any) -> tuple[Any, ...]:
    metadata = (
        projection.get("metadata")
        if isinstance(projection, Mapping)
        else getattr(projection, "metadata", None)
    )
    if not isinstance(metadata, Mapping):
        return (id(projection), None)
    return (
        id(projection),
        metadata.get("generation"),
        metadata.get("tick"),
        metadata.get("event_cursor"),
        metadata.get("runtime_event_cursor"),
        metadata.get("event_tape_active_count"),
        metadata.get("import_digest"),
        metadata.get("checkpoint_digest"),
        metadata.get("source_projection_digest"),
        metadata.get("active_event_prefix_sha256"),
        metadata.get("event_prefix_sha256"),
        metadata.get("event_prefix_digest"),
    )


def _eed_fixed_projection_semantic_signature_v1(projection: Any) -> tuple[Any, ...]:
    metadata = (
        projection.get("metadata")
        if isinstance(projection, Mapping)
        else getattr(projection, "metadata", None)
    )
    if not isinstance(metadata, Mapping):
        return (None,)
    event_cursor = (
        projection.get("event_cursor")
        if isinstance(projection, Mapping)
        else getattr(projection, "event_cursor", None)
    )
    runtime_event_cursor = (
        projection.get("runtime_event_cursor")
        if isinstance(projection, Mapping)
        else getattr(projection, "runtime_event_cursor", None)
    )
    return (
        metadata.get("generation"),
        metadata.get("tick"),
        event_cursor,
        runtime_event_cursor,
        metadata.get("event_tape_active_count"),
        metadata.get("typed_lifecycle_event_active_count"),
        metadata.get("event_tape_cursor"),
        metadata.get("typed_lifecycle_event_next_sequence"),
        metadata.get("import_digest"),
        metadata.get("checkpoint_digest"),
        metadata.get("source_projection_digest"),
        metadata.get("active_event_prefix_sha256"),
        metadata.get("event_prefix_sha256"),
        metadata.get("event_prefix_digest"),
    )


def _wdl_fixed_record_event_capacity_promotion_v1(
    prepared: Any,
    *,
    cell_ids: Sequence[str],
    original_event_capacities: Sequence[int],
    batch_event_capacity: int,
) -> None:
    promoted_count = sum(
        1 for value in original_event_capacities if value != batch_event_capacity
    )
    receipt = {
        "schema_version": "wdl_fixed_event_capacity_promotion_receipt_v1",
        "promotion_policy": _WDL_FIXED_EVENT_CAPACITY_PROMOTION_POLICY_V1,
        "cell_ids": tuple(cell_ids),
        "original_event_capacities": tuple(original_event_capacities),
        "batch_event_capacity": batch_event_capacity,
        "promoted_request_count": promoted_count,
        "promoted": promoted_count > 0,
    }
    config = getattr(prepared, "config", None)
    if isinstance(config, dict):
        config["wdl_event_capacity_promotion"] = receipt
        return
    try:
        setattr(prepared, "wdl_event_capacity_promotion", receipt)
    except (AttributeError, TypeError):
        pass


def _eed_fixed_request_event_capacity_v1(request: Any, *, cell: int) -> int:
    event_capacity = getattr(request, "event_capacity", None)
    if type(event_capacity) is not int or event_capacity <= 0:
        raise CampaignProviderError(
            "EED/fixed-MLP source request event_capacity must be a positive "
            f"exact integer for cell {cell}"
        )
    return event_capacity


def _eed_fixed_record_event_capacity_promotion_v1(
    prepared: Any,
    *,
    cell_ids: Sequence[str],
    original_event_capacities: Sequence[int],
    batch_event_capacity: int,
) -> None:
    promoted_count = sum(
        1 for value in original_event_capacities if value != batch_event_capacity
    )
    receipt = {
        "schema_version": "eed_fixed_event_capacity_promotion_receipt_v1",
        "promotion_policy": _EED_FIXED_EVENT_CAPACITY_PROMOTION_POLICY_V1,
        "cell_ids": tuple(cell_ids),
        "original_event_capacities": tuple(original_event_capacities),
        "batch_event_capacity": batch_event_capacity,
        "promoted_request_count": promoted_count,
        "promoted": promoted_count > 0,
    }
    config = getattr(prepared, "config", None)
    if isinstance(config, dict):
        config["eed_event_capacity_promotion"] = receipt
        return
    try:
        setattr(prepared, "eed_event_capacity_promotion", receipt)
    except (AttributeError, TypeError):
        pass


def _wdl_fixed_prepare_adapter(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any] | None = None,
    per_cell_request_factory: Callable[[int], Any] | None = None,
    logical_generation_count: int,
    physical_window_generations: int = 2,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    source_projection_cursor_factory: Callable[[], Any] | None = None,
    source_import_cursor_receipt: Mapping[str, Any] | None = None,
    jungle_overlay: bool = False,
    eed_overlay: bool = False,
    hapb_host_policy_enable: bool = False,
    hapb_policy_tile_rows: int | None = None,
    **kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
        build_prepared_direct_resident_wdl_v2,
        build_prepared_direct_resident_wdl_v2_streaming,
        _patch_envelope_with_wdl_masked,
        _wdl_source_capabilities_v2,
    )
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        build_prepared_direct_resident_wdl_jungle_v1,
        patch_envelope_with_wdl_jungle_masked_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
        build_direct_resident_source_allocation_request_v1,
    )

    if per_cell_request_factory is not None:
        if per_cell_requests is not None:
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed preparation cannot receive request tuples"
            )
        if resume_projections is not None:
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed preparation cannot receive a projection tuple"
            )
        ordered_cell_ids = tuple(cell_ids)
        if (
            not ordered_cell_ids
            or len(set(ordered_cell_ids)) != len(ordered_cell_ids)
            or any(type(cell_id) is not str or not cell_id for cell_id in ordered_cell_ids)
        ):
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed cell IDs must be unique nonempty strings"
            )
        if hapb_host_policy_enable:
            raise CampaignProviderError(
                "HAPB WDL/fixed-MLP preparation is not admitted on the streamed route"
            )
        if source_projection_cursor_factory is None or not callable(
            source_projection_cursor_factory
        ):
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed preparation requires a source projection cursor"
            )

        pending_first: list[Any] = [per_cell_request_factory(0)]
        first_wrapper = pending_first[0]
        if first_wrapper is None:
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed request factory returned no first request"
            )

        def wrapper_for_cell(index: int) -> Any:
            if index == 0 and pending_first:
                return pending_first.pop()
            return per_cell_request_factory(index)

        first_source = getattr(first_wrapper, "source_checkpoint_projection", None)
        if first_source is None:
            initial_generation = 0
        else:
            first_metadata = (
                first_source.get("metadata")
                if isinstance(first_source, Mapping)
                else None
            )
            if not isinstance(first_metadata, Mapping) or type(
                first_metadata.get("generation")
            ) is not int:
                raise CampaignProviderError(
                    "WDL/fixed-MLP streamed first source generation is invalid"
                )
            initial_generation = int(first_metadata["generation"])

        tick_native_control = dict(kwargs.get("tick_native_control") or {})
        if tick_native_control and tick_native_control.get(
            "time_authority"
        ) != "absolute_tick_v1":
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed control has the wrong time authority"
            )
        request_physical_window = getattr(
            first_wrapper, "physical_window_generations", None
        )
        if (
            type(request_physical_window) is not int
            or request_physical_window not in {1, 2}
            or request_physical_window != physical_window_generations
        ):
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed physical window authority differs"
            )
        first_short_extent = getattr(
            first_wrapper, "allow_short_technical_gate_generation_count", False
        )
        first_partial_cursor = getattr(
            first_wrapper, "allow_partial_generation_source_cursor", False
        )
        if type(first_short_extent) is not bool or type(first_partial_cursor) is not bool:
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed extent authorities must be exact booleans"
            )
        agent_id_allocation_generation_count = kwargs.get(
            "agent_id_allocation_generation_count"
        )
        if agent_id_allocation_generation_count is None:
            agent_id_allocation_generation_count = getattr(
                first_wrapper, "agent_id_allocation_generation_count", None
            )
        if agent_id_allocation_generation_count is None and tick_native_control:
            # Private width for the legacy allocator ABI; source-window
            # capacity is the only input besides the bounded dispatch span.
            agent_id_allocation_generation_count = (
                _tick_native_private_source_capacity_window_count_v1(
                    tick_native_control
                )
            )
        if type(agent_id_allocation_generation_count) is not int:
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed preparation lacks allocation authority"
            )

        source_projection_capture = kwargs.get("source_projection_capture")
        source_import_receipt = source_import_cursor_receipt
        if source_import_receipt is not None and not isinstance(
            source_import_receipt, Mapping
        ):
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed source-import receipt is malformed"
            )
        direct_descriptors: list[NativeRequestDescriptorV1] = []
        initial_generations: set[int] = set()
        expected_event_capacity: int | None = None
        expected_short_extent: bool | None = first_short_extent
        expected_partial_cursor: bool | None = first_partial_cursor

        def build_one_direct_request(index: int) -> Any:
            nonlocal expected_event_capacity
            nonlocal expected_short_extent
            nonlocal expected_partial_cursor
            item = wrapper_for_cell(index)
            if item is None:
                raise CampaignProviderError(
                    f"WDL/fixed-MLP streamed request factory returned no request for cell {index}"
                )
            item_window = getattr(item, "physical_window_generations", None)
            if item_window != physical_window_generations:
                raise CampaignProviderError(
                    f"WDL/fixed-MLP streamed cell {index} has a different physical window"
                )
            short_extent = getattr(
                item, "allow_short_technical_gate_generation_count", False
            )
            partial_cursor = getattr(
                item, "allow_partial_generation_source_cursor", False
            )
            if type(short_extent) is not bool or type(partial_cursor) is not bool:
                raise CampaignProviderError(
                    "WDL/fixed-MLP streamed extent authorities must be exact booleans"
                )
            if expected_short_extent is None:
                expected_short_extent = short_extent
                expected_partial_cursor = partial_cursor
            elif (
                short_extent != expected_short_extent
                or partial_cursor != expected_partial_cursor
            ):
                raise CampaignProviderError(
                    "WDL/fixed-MLP streamed extent authorities are mixed"
                )
            source = getattr(item, "source_checkpoint_projection", None)
            if source is None:
                source_generation = 0
            else:
                metadata = source.get("metadata") if isinstance(source, Mapping) else None
                if not isinstance(metadata, Mapping) or type(
                    metadata.get("generation")
                ) is not int:
                    raise CampaignProviderError(
                        f"WDL/fixed-MLP streamed source generation is invalid for cell {index}"
                    )
                source_generation = int(metadata["generation"])
            initial_generations.add(source_generation)
            build_kwargs: dict[str, Any] = {
                "qualified_cuda_module": item.qualified_cuda_module,
                "logical_generation_count": logical_generation_count,
                "physical_window_generations": physical_window_generations,
                "agent_id_allocation_generation_count": (
                    agent_id_allocation_generation_count
                ),
                "source_checkpoint_projection": source,
                "source_capability_mask": _wdl_source_capabilities_v2(
                    item.source_capability_mask,
                    jungle_overlay=jungle_overlay,
                    eed_overlay=eed_overlay,
                ),
                "allow_short_technical_gate_generation_count": short_extent,
                "allow_partial_generation_source_cursor": partial_cursor,
                "resident_owner_span_ticks": (
                    tick_native_control.get("resident_owner_span_ticks")
                    if tick_native_control
                    else getattr(item, "resident_owner_span_ticks", None)
                ),
                "native_dispatch_window_ticks": (
                    tick_native_control.get("native_dispatch_window_ticks")
                    if tick_native_control
                    else getattr(item, "native_dispatch_window_ticks", None)
                ),
                "source_window_ticks": (
                    tick_native_control.get("source_window_ticks")
                    if tick_native_control
                    else getattr(item, "source_window_ticks", None)
                ),
                "event_window_capacity": (
                    tick_native_control.get("event_window_capacity")
                    if tick_native_control
                    else getattr(item, "event_window_capacity", None)
                ),
                "tick_native_control": tick_native_control,
                "event_history_sink_path": getattr(
                    item, "event_history_sink_path", None
                ),
                "event_history_sink_cell_index": getattr(
                    item, "event_history_sink_cell_index", index
                ),
                "runner_envelope_deriver": (
                    lambda runner: (
                        patch_envelope_with_wdl_jungle_masked_v1(
                            runner,
                            allow_short_technical_gate_generation_count=short_extent,
                            eed_overlay=eed_overlay,
                            tick_native_control=(tick_native_control or None),
                        )
                        if jungle_overlay
                        else _patch_envelope_with_wdl_masked(
                            runner,
                            allow_short_technical_gate_generation_count=short_extent,
                            tick_native_control=(tick_native_control or None),
                        )
                    )
                ),
            }
            direct_request = build_direct_resident_source_allocation_request_v1(
                item.runner,
                **build_kwargs,
            )
            if hapb_policy_tile_rows is not None:
                raise CampaignProviderError(
                    "HAPB policy tiles are not admitted on the streamed route"
                )
            if direct_request.batch_cells != 1:
                raise CampaignProviderError(
                    f"WDL/fixed-MLP streamed direct request {index} is not B1"
                )
            if direct_request.cpu_request_staging_enable is not False:
                raise CampaignProviderError(
                    "WDL/fixed-MLP streamed route enabled CPU request staging"
                )
            event_capacity = _wdl_fixed_request_event_capacity_v1(
                direct_request, cell=index
            )
            if expected_event_capacity is None:
                expected_event_capacity = event_capacity
            elif event_capacity != expected_event_capacity:
                raise CampaignProviderError(
                    "WDL/fixed-MLP streamed route requires one source-derived event capacity"
                )
            captured_source_digest = None
            if callable(source_projection_capture):
                captured_source_digest = source_projection_capture(
                    index, direct_request.import_projection
                )
            release_source = getattr(item, "source_import_release", None)
            try:
                if callable(release_source):
                    release_source(direct_request)
            except BaseException:
                fail_source = getattr(item, "source_import_fail", None)
                if callable(fail_source):
                    fail_source("streamed direct request source release failed")
                raise
            direct_descriptors.append(
                NativeRequestDescriptorV1(
                    cell_id=ordered_cell_ids[index],
                    cuda_device=str(direct_request.cuda_device),
                    cuda_capability=dict(direct_request.cuda_capability),
                    event_capacity=event_capacity,
                    source_import_projection_digest_sha256=(
                        direct_request.source_import_projection_digest_sha256
                        if isinstance(
                            direct_request.source_import_projection_digest_sha256,
                            str,
                        )
                        else (
                            str(captured_source_digest)
                            if isinstance(captured_source_digest, str)
                            else None
                        )
                    ),
                )
            )
            return direct_request

        stream_kwargs = {
            "qualified_cuda_module": getattr(
                first_wrapper, "qualified_cuda_module", None
            ),
            "generation_rows": logical_generation_count,
            "initial_generation": initial_generation,
            "cell_ids": ordered_cell_ids,
            "agent_id_allocation_generation_count": (
                agent_id_allocation_generation_count
            ),
            "allow_short_technical_gate_generation_count": (
                bool(expected_short_extent)
            ),
            "allow_partial_generation_source_cursor": (
                bool(expected_partial_cursor)
            ),
            "source_projection_cursor_factory": source_projection_cursor_factory,
            "source_import_cursor_receipt": source_import_receipt,
            "source_capability_mask": _wdl_source_capabilities_v2(
                getattr(first_wrapper, "source_capability_mask", None),
                jungle_overlay=jungle_overlay,
                eed_overlay=eed_overlay,
            ),
            "eed_overlay": eed_overlay,
        }
        if jungle_overlay:
            prepared = build_prepared_direct_resident_wdl_jungle_v1(
                per_cell_request_factory=build_one_direct_request,
                **stream_kwargs,
            )
            prepared_source_projections = tuple(
                getattr(getattr(prepared, "wdl", None), "source_projections", ())
            )
        else:
            prepared = build_prepared_direct_resident_wdl_v2_streaming(
                request_factory=build_one_direct_request,
                **stream_kwargs,
            )
            prepared_source_projections = tuple(
                getattr(prepared, "source_projections", ())
            )
        if initial_generations != {initial_generation}:
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed source generations are mixed"
            )
        if len(direct_descriptors) != len(ordered_cell_ids):
            raise CampaignProviderError(
                "WDL/fixed-MLP streamed preparation did not cover every cell"
            )
        return PreparedNativeMechanicsOwnerV1(
            physical_window_arguments=prepared,
            source_projections=prepared_source_projections,
            cell_ids=ordered_cell_ids,
            request_descriptors=tuple(direct_descriptors),
        )

    if per_cell_requests is None:
        raise CampaignProviderError(
            "WDL/fixed-MLP preparation requires request tuples or a request factory"
        )
    if len(cell_ids) != len(per_cell_requests):
        raise CampaignProviderError(
            "WDL/fixed-MLP cell IDs and request count differ"
        )
    if not per_cell_requests:
        raise CampaignProviderError("WDL/fixed-MLP execution request is invalid")
    if (
        type(physical_window_generations) is not int
        or physical_window_generations not in {1, 2}
    ):
        raise CampaignProviderError(
            "WDL/fixed-MLP physical window generations must be exact integer 1 or 2"
        )
    request_physical_windows = tuple(
        getattr(item, "physical_window_generations", None)
        for item in per_cell_requests
    )
    if any(
        type(value) is not int or value not in {1, 2}
        for value in request_physical_windows
    ):
        raise CampaignProviderError(
            "WDL/fixed-MLP requests lack exact physical window authority"
        )
    if any(
        value != physical_window_generations for value in request_physical_windows
    ):
        raise CampaignProviderError(
            "WDL/fixed-MLP request and preparation physical windows differ"
        )
    if eed_overlay and not jungle_overlay:
        raise CampaignProviderError("EED WDL preparation requires the shared Jungle owner")
    if type(hapb_host_policy_enable) is not bool:
        raise CampaignProviderError("HAPB WDL preparation enable must be an exact bool")
    if hapb_host_policy_enable and not (jungle_overlay and eed_overlay):
        raise CampaignProviderError("HAPB preparation requires the integrated WDL/Jungle/EED route")
    if hapb_policy_tile_rows is not None and (
        type(hapb_policy_tile_rows) is not int
        or hapb_policy_tile_rows not in {512, 4096}
    ):
        raise CampaignProviderError(
            "HAPB policy tile rows must be exactly 512 or 4096"
        )
    if hapb_policy_tile_rows is not None and not hapb_host_policy_enable:
        raise CampaignProviderError(
            "HAPB policy tile rows require host-authoritative policy preparation"
        )
    request = per_cell_requests[0]
    tick_native_control = dict(kwargs.get("tick_native_control") or {})
    if not tick_native_control:
        # The lifecycle passes the canonical control separately from the thin
        # request wrappers.  A wrapper may predate the absolute-tick fields,
        # so recover the control only from an identically bound per-cell
        # request; never infer it from logical generation compatibility data.
        per_cell_controls = tuple(
            dict(getattr(item, "tick_native_control", {}) or {})
            for item in per_cell_requests
        )
        nonempty_controls = tuple(
            control for control in per_cell_controls if control
        )
        if nonempty_controls:
            tick_native_control = dict(nonempty_controls[0])
            if any(control != tick_native_control for control in nonempty_controls[1:]):
                raise CampaignProviderError(
                    "WDL/fixed-MLP mixed tick-native control authority is not representable"
                )
    if tick_native_control and tick_native_control.get("time_authority") != "absolute_tick_v1":
        raise CampaignProviderError(
            "WDL/fixed-MLP preparation control has the wrong time authority"
        )
    agent_id_allocation_generation_count = kwargs.get(
        "agent_id_allocation_generation_count"
    )
    if agent_id_allocation_generation_count is None:
        per_cell_allocation_counts = tuple(
            getattr(item, "agent_id_allocation_generation_count", None)
            for item in per_cell_requests
        )
        exact_counts = tuple(
            value
            for value in per_cell_allocation_counts
            if type(value) is int
        )
        if exact_counts and len(exact_counts) == len(per_cell_allocation_counts):
            if len(set(exact_counts)) != 1:
                raise CampaignProviderError(
                    "WDL/fixed-MLP mixed agent-ID allocation authority is not representable"
                )
            agent_id_allocation_generation_count = exact_counts[0]
        elif tick_native_control:
            # Private width for the legacy allocator ABI; absolute ticks
            # remain the sole active horizon and RNG authority.
            agent_id_allocation_generation_count = (
                _tick_native_private_source_capacity_window_count_v1(
                    tick_native_control
                )
            )
        # Legacy callers may leave this compatibility field absent; preserve
        # that existing contract and let the legacy allocator issue its own
        # fail-closed validation.  A tick-native control above always derives
        # the bounded compatibility envelope before reaching this point.
    if len(per_cell_requests) >= 1:
        short_extent_authorities = tuple(
            getattr(
                item,
                "allow_short_technical_gate_generation_count",
                False,
            )
            for item in per_cell_requests
        )
        if any(type(value) is not bool for value in short_extent_authorities):
            raise CampaignProviderError(
                "WDL/fixed-MLP short-extent authority must be an exact bool"
            )
        if len(set(short_extent_authorities)) != 1:
            raise CampaignProviderError(
                "WDL/fixed-MLP mixed short-extent authority is not representable"
            )
        allow_short_technical_gate_generation_count = short_extent_authorities[0]
        partial_cursor_authorities = tuple(
            getattr(item, "allow_partial_generation_source_cursor", False)
            for item in per_cell_requests
        )
        if any(type(value) is not bool for value in partial_cursor_authorities):
            raise CampaignProviderError(
                "WDL/fixed-MLP partial-cursor authority must be an exact bool"
            )
        if len(set(partial_cursor_authorities)) != 1:
            raise CampaignProviderError(
                "WDL/fixed-MLP mixed partial-cursor authority is not representable"
            )
        allow_partial_generation_source_cursor = partial_cursor_authorities[0]
        direct_batch_import_authorities = tuple(
            getattr(item, "direct_batch_host_import_enable", False)
            for item in per_cell_requests
        )
        if any(type(value) is not bool for value in direct_batch_import_authorities):
            raise CampaignProviderError(
                "WDL/fixed-MLP direct batch host import enable must be an exact bool"
            )
        if len(set(direct_batch_import_authorities)) != 1:
            raise CampaignProviderError(
                "WDL/fixed-MLP mixed direct batch host import authority is not representable"
            )
        direct_batch_host_import_enable = direct_batch_import_authorities[0]
        if resume_projections is not None and (
            len(resume_projections) != len(per_cell_requests)
            or any(
                item.source_checkpoint_projection is not source
                for item, source in zip(
                    per_cell_requests,
                    resume_projections,
                    strict=True,
                )
            )
        ):
            raise CampaignProviderError(
                "WDL/fixed-MLP ordered resume projections were substituted"
            )
        initial_generations: set[int] = set()

        def build_direct_requests(
            *, event_capacity_override: int | None = None
        ) -> tuple[Any, ...]:
            direct_requests = []
            for item in per_cell_requests:
                source = item.source_checkpoint_projection
                if source is None:
                    initial_generations.add(0)
                else:
                    metadata = source.get("metadata")
                    if (
                        not isinstance(metadata, Mapping)
                        or type(metadata.get("generation")) is not int
                    ):
                        raise CampaignProviderError(
                            "WDL/fixed-MLP resume boundary is invalid"
                        )
                    initial_generations.add(metadata["generation"])
                build_kwargs = {
                    "qualified_cuda_module": item.qualified_cuda_module,
                    "logical_generation_count": logical_generation_count,
                    "physical_window_generations": physical_window_generations,
                    "agent_id_allocation_generation_count": (
                        agent_id_allocation_generation_count
                    ),
                    "source_checkpoint_projection": source,
                    "source_capability_mask": _wdl_source_capabilities_v2(
                        item.source_capability_mask,
                        jungle_overlay=jungle_overlay,
                        eed_overlay=eed_overlay,
                    ),
                    "allow_short_technical_gate_generation_count": (
                        allow_short_technical_gate_generation_count
                    ),
                    "allow_partial_generation_source_cursor": (
                        allow_partial_generation_source_cursor
                    ),
                    "direct_batch_host_import_enable": (
                        direct_batch_host_import_enable
                    ),
                    "resident_owner_span_ticks": (
                        tick_native_control.get("resident_owner_span_ticks")
                        if tick_native_control
                        else getattr(item, "resident_owner_span_ticks", None)
                    ),
                    "native_dispatch_window_ticks": (
                        tick_native_control.get("native_dispatch_window_ticks")
                        if tick_native_control
                        else getattr(item, "native_dispatch_window_ticks", None)
                    ),
                    "source_window_ticks": (
                        tick_native_control.get("source_window_ticks")
                        if tick_native_control
                        else getattr(item, "source_window_ticks", None)
                    ),
                    "event_window_capacity": (
                        tick_native_control.get("event_window_capacity")
                        if tick_native_control
                        else getattr(item, "event_window_capacity", None)
                    ),
                    "tick_native_control": dict(tick_native_control),
                    "event_history_sink_path": getattr(
                        item, "event_history_sink_path", None
                    ),
                    "event_history_sink_cell_index": getattr(
                        item, "event_history_sink_cell_index", 0
                    ),
                    "runner_envelope_deriver": (
                        lambda runner: (
                            patch_envelope_with_wdl_jungle_masked_v1(
                                runner,
                                allow_short_technical_gate_generation_count=(
                                    item.allow_short_technical_gate_generation_count
                                ),
                                eed_overlay=eed_overlay,
                                tick_native_control=(
                                    tick_native_control or None
                                ),
                            )
                            if jungle_overlay
                            else _patch_envelope_with_wdl_masked(
                                runner,
                                allow_short_technical_gate_generation_count=(
                                    item.allow_short_technical_gate_generation_count
                                ),
                                tick_native_control=(
                                    tick_native_control or None
                                ),
                            )
                        )
                    ),
                }
                if hapb_host_policy_enable:
                    build_kwargs["host_authoritative_tensor_names"] = (
                        "mlp_pack",
                    )
                if event_capacity_override is not None:
                    build_kwargs["event_capacity_override"] = event_capacity_override
                direct_request = build_direct_resident_source_allocation_request_v1(
                    item.runner,
                    **build_kwargs,
                )
                if hapb_policy_tile_rows is not None:
                    boundary_metadata = getattr(
                        getattr(direct_request, "cuda_boundary", None),
                        "metadata",
                        None,
                    )
                    if not isinstance(boundary_metadata, dict):
                        raise CampaignProviderError(
                            "HAPB tile timing requires mutable CUDA-boundary metadata"
                        )
                    # The WDL adapter wraps the native request; bind the tile
                    # candidate only after native boundary construction and
                    # repeat this on any event-capacity rebuild.
                    boundary_metadata[
                        "hapb_policy_tile_rows"
                    ] = hapb_policy_tile_rows
                direct_requests.append(direct_request)
            return tuple(direct_requests)

        direct_requests = build_direct_requests()
        original_event_capacities = tuple(
            _wdl_fixed_request_event_capacity_v1(item, cell=cell)
            for cell, item in enumerate(direct_requests)
        )
        batch_event_capacity = max(original_event_capacities)
        projection_signatures = tuple(
            _wdl_fixed_projection_identity_signature_v1(item.import_projection)
            for item in direct_requests
        )
        if any(value != batch_event_capacity for value in original_event_capacities):
            direct_requests = build_direct_requests(
                event_capacity_override=batch_event_capacity
            )
            rebuilt_event_capacities = tuple(
                _wdl_fixed_request_event_capacity_v1(item, cell=cell)
                for cell, item in enumerate(direct_requests)
            )
            if rebuilt_event_capacities != (batch_event_capacity,) * len(
                direct_requests
            ):
                raise CampaignProviderError(
                    "WDL/fixed-MLP request builder did not honor the batch "
                    "event-capacity override"
                )
            rebuilt_projection_signatures = tuple(
                _wdl_fixed_projection_identity_signature_v1(item.import_projection)
                for item in direct_requests
            )
            if rebuilt_projection_signatures != projection_signatures:
                raise CampaignProviderError(
                    "WDL/fixed-MLP event-capacity promotion changed source "
                    "projection identity"
                )
        if len(initial_generations) != 1:
            raise CampaignProviderError(
                "WDL/fixed-MLP mixed resume generations are not representable"
            )
        build_kwargs = {
            "qualified_cuda_module": request.qualified_cuda_module,
            "generation_rows": logical_generation_count,
            "initial_generation": initial_generations.pop(),
            "per_cell_requests": direct_requests,
            "source_checkpoint_projections": tuple(
                item.import_projection for item in direct_requests
            ),
            "cell_ids": cell_ids,
            "agent_id_allocation_generation_count": (
                agent_id_allocation_generation_count
            ),
            "allow_short_technical_gate_generation_count": (
                allow_short_technical_gate_generation_count
            ),
            "allow_partial_generation_source_cursor": (
                allow_partial_generation_source_cursor
            ),
            "eed_overlay": eed_overlay,
            "hapb_host_policy_enable": hapb_host_policy_enable,
        }
        if jungle_overlay:
            from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
                build_prepared_direct_resident_wdl_jungle_v1,
            )

            prepared = build_prepared_direct_resident_wdl_jungle_v1(**build_kwargs)
        else:
            prepared = build_prepared_direct_resident_wdl_v2(None, **build_kwargs)
        _wdl_fixed_record_event_capacity_promotion_v1(
            prepared.wdl if jungle_overlay else prepared,
            cell_ids=cell_ids,
            original_event_capacities=original_event_capacities,
            batch_event_capacity=batch_event_capacity,
        )
        return PreparedNativeMechanicsOwnerV1(
            physical_window_arguments=prepared,
            source_projections=prepared.source_projections,
            cell_ids=tuple(cell_ids),
        )


def _wdl_fixed_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
        execute_direct_resident_wdl_v2,
    )

    prepared = physical_window_arguments
    return execute_direct_resident_wdl_v2(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=(
            "wdl-fixed-mlp:"
            f"{prepared.patch.source.generation_base}:"
            f"{logical_generation_count}"
        ),
    )


def _wdl_fixed_project_adapter(
    prepared_adapter: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        DirectResidentPatchMulticellSettlementV1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
        project_direct_resident_wdl_multicell_checkpoints_v2,
    )

    prepared = prepared_adapter.physical_window_arguments
    completed = direct_result.get("completed_generations")
    if type(completed) is not int:
        raise CampaignProviderError(
            "WDL/fixed-MLP result lacks completed generations"
        )
    cell_ids = getattr(prepared_adapter, "cell_ids", ())
    if not isinstance(cell_ids, (tuple, list)) or not cell_ids:
        raise CampaignProviderError(
            "WDL/fixed-MLP projection requires stable cell IDs"
        )
    if any(type(cell_id) is not str or not cell_id for cell_id in cell_ids):
        raise CampaignProviderError(
            "WDL/fixed-MLP projection requires stable cell IDs"
        )
    projections = project_direct_resident_wdl_multicell_checkpoints_v2(
        prepared,
        direct_result=direct_result,
        completed_generation_count=completed,
        reporting_enabled=reporting_enabled,
    )
    turnover_counts, turnover_count_generations = _turnover_count_authority_v1(
        prepared, direct_result
    )
    native_rows = _project_native_cell_metrics_v1(
        projections,
        cell_ids=tuple(cell_ids),
        direct_result=direct_result,
        wdl_generation_metrics=prepared.reporting_tensors[5][:, :completed],
        turnover_counts=turnover_counts,
        turnover_count_generations=turnover_count_generations,
    )
    rows = (
        native_rows
        if reporting_enabled
        else _compact_terminal_native_metric_rows_v1(native_rows)
    )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=tuple(cell_ids),
        checkpoint_projections=projections,
        compact_result_rows=rows,
        completed_generation_count=completed,
        terminal_generation=prepared.patch.source.generation_base + completed,
    )


def wdl_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose WDL world mechanics and fixed MLP under the engine."""

    from engine.campaigns.patch_runner_factory import (
        build_patch_wdl_fixed_mlp_runner_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
        project_direct_resident_wdl_generation_zero_v2,
    )

    return NativeMechanicsBindingV1(
        binding_id=WDL_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_wdl_fixed_mlp_v2",
        batch_max=72,
        generation_min=3,
        generation_max=32,
        ticks_per_generation=80,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_wdl_fixed_mlp_runner_v1,
        request_builder=_wdl_fixed_request_adapter,
        preparer=_wdl_fixed_prepare_adapter,
        executor=_wdl_fixed_execute_adapter,
        projector=_wdl_fixed_project_adapter,
        generation_zero_checkpoint_projector=(
            project_direct_resident_wdl_generation_zero_v2
        ),
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_v3",
        ),
    )


def _eed_fixed_request_adapter(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    source_checkpoint_projection: Mapping[str, Any] | None,
    source_capability_mask: Mapping[str, bool] | None = None,
    **_kwargs: Any,
) -> Any:
    required_capabilities = {
        "cross_path_overlay": False,
        "genome_v2": True,
    }
    if source_capability_mask is not None:
        if not isinstance(source_capability_mask, Mapping):
            raise CampaignProviderError(
                "EED/fixed-MLP source capability mask must be a mapping"
            )
        for name, expected in required_capabilities.items():
            actual = source_capability_mask.get(name, expected)
            if type(actual) is not bool or actual is not expected:
                raise CampaignProviderError(
                    f"EED/fixed-MLP requires {name}={expected}"
                )
    return SimpleNamespace(
        runner=runner,
        qualified_cuda_module=qualified_cuda_module,
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=(
            agent_id_allocation_generation_count
        ),
        source_checkpoint_projection=source_checkpoint_projection,
        source_capability_mask=required_capabilities,
        cuda_device="cuda",
        cuda_capability={},
    )


def _eed_checkpoint_projection_mapping_v1(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        projection = dict(value)
    else:
        metadata = getattr(value, "metadata", None)
        tensors = getattr(value, "tensors", None)
        if not (isinstance(metadata, Mapping) and isinstance(tensors, Mapping)):
            raise CampaignProviderError(
                "EED/fixed-MLP source projection is not canonical"
            )
        projection = {
            "metadata": metadata,
            "tensors": tensors,
        }
        for name, child in vars(value).items():
            if name not in projection:
                projection[name] = child
    metadata = projection.get("metadata")
    if isinstance(metadata, Mapping):
        active_event_count = metadata.get(
            "typed_lifecycle_event_active_count",
            metadata.get(
                "event_tape_cursor",
                metadata.get("event_tape_active_count", 0),
            ),
        )
        if type(active_event_count) is not int or active_event_count < 0:
            raise CampaignProviderError(
                "EED/fixed-MLP source projection has invalid lifecycle event count"
            )
        event_cursor = projection.get("event_cursor", active_event_count)
        runtime_event_cursor = projection.get("runtime_event_cursor", event_cursor)
        if (
            type(event_cursor) is not int
            or event_cursor < 0
            or type(runtime_event_cursor) is not int
            or runtime_event_cursor < event_cursor
        ):
            raise CampaignProviderError(
                "EED/fixed-MLP source projection has invalid event cursors"
            )
        projection.setdefault("event_cursor", event_cursor)
        projection.setdefault("runtime_event_cursor", runtime_event_cursor)
        if metadata.get("generation") == 0 and metadata.get("tick") == 0:
            projection.setdefault("restore_count", 0)
            projection.setdefault("generation_calls", 0)
            projection.setdefault("last_applied_actions", 0)
            projection.setdefault("last_moved", 0)
            projection.setdefault("last_gathered", 0)
            projection.setdefault("last_policy_refill_calls", 0)
            projection.setdefault("last_policy_refill_rows", 0)
            projection.setdefault("pending_generation_free_agent_ids", [])
            projection.setdefault("pending_generation_free_slots", [])
            projection.setdefault("last_world", "none")
    return _eed_checkpoint_contract_projection_v1(projection)


def _eed_checkpoint_contract_projection_v1(
    projection: Mapping[str, Any],
) -> Mapping[str, Any]:
    from engine.perf.cxxbackbone10_resident_state_owner import (
        _with_checkpoint_contract_metadata,
    )

    return _with_checkpoint_contract_metadata(projection)


def _eed_fixed_prepare_adapter(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any],
    logical_generation_count: int,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
        build_direct_resident_eed_source_request_v1,
        build_prepared_direct_resident_eed_v1,
    )

    if len(cell_ids) != len(per_cell_requests) or not per_cell_requests:
        raise CampaignProviderError(
            "EED/fixed-MLP cell IDs and request count differ"
        )
    if len(per_cell_requests) > 48:
        raise CampaignProviderError("EED/fixed-MLP physical batch exceeds B48")
    if len(per_cell_requests) > 1:
        if resume_projections is not None and (
            len(resume_projections) != len(per_cell_requests)
            or any(
                item.source_checkpoint_projection is not source
                for item, source in zip(
                    per_cell_requests,
                    resume_projections,
                    strict=True,
                )
            )
        ):
            raise CampaignProviderError(
                "EED/fixed-MLP ordered resume projections were substituted"
            )
        initial_generations: set[int] = set()
        validated_sources: list[tuple[Any, Mapping[str, Any] | None, int]] = []
        for item in per_cell_requests:
            source = item.source_checkpoint_projection
            if source is None:
                initial_generation = 0
            else:
                metadata = source.get("metadata")
                if (
                    not isinstance(metadata, Mapping)
                    or type(metadata.get("generation")) is not int
                    or metadata.get("tick") != 0
                ):
                    raise CampaignProviderError(
                        "EED/fixed-MLP resume boundary is invalid"
                    )
                initial_generation = metadata["generation"]
            initial_generations.add(initial_generation)
            validated_sources.append((item, source, initial_generation))
        if len(initial_generations) != 1:
            raise CampaignProviderError(
                "EED/fixed-MLP mixed resume generations are not representable"
            )
        def build_direct_requests(
            *, event_capacity_override: int | None = None
        ) -> tuple[Any, ...]:
            direct_requests = []
            for item, source, initial_generation in validated_sources:
                build_kwargs = {
                    "qualified_cuda_module": item.qualified_cuda_module,
                    "logical_generation_count": logical_generation_count,
                    "initial_generation": initial_generation,
                    "source_checkpoint_projection": source,
                    "source_capability_mask": item.source_capability_mask,
                    "agent_id_allocation_generation_count": (
                        item.agent_id_allocation_generation_count
                    ),
                }
                if event_capacity_override is not None:
                    build_kwargs["event_capacity_override"] = event_capacity_override
                direct_requests.append(
                    build_direct_resident_eed_source_request_v1(
                        item.runner,
                        **build_kwargs,
                    )
                )
            return tuple(direct_requests)

        direct_requests = build_direct_requests()
        original_event_capacities = tuple(
            _eed_fixed_request_event_capacity_v1(item, cell=cell)
            for cell, item in enumerate(direct_requests)
        )
        batch_event_capacity = max(original_event_capacities)
        projection_signatures = tuple(
            _eed_fixed_projection_semantic_signature_v1(item.import_projection)
            for item in direct_requests
        )
        if any(value != batch_event_capacity for value in original_event_capacities):
            direct_requests = build_direct_requests(
                event_capacity_override=batch_event_capacity
            )
            rebuilt_event_capacities = tuple(
                _eed_fixed_request_event_capacity_v1(item, cell=cell)
                for cell, item in enumerate(direct_requests)
            )
            if rebuilt_event_capacities != (batch_event_capacity,) * len(
                direct_requests
            ):
                raise CampaignProviderError(
                    "EED/fixed-MLP request builder did not honor the batch "
                    "event-capacity override"
                )
            rebuilt_projection_signatures = tuple(
                _eed_fixed_projection_semantic_signature_v1(item.import_projection)
                for item in direct_requests
            )
            if rebuilt_projection_signatures != projection_signatures:
                raise CampaignProviderError(
                    "EED/fixed-MLP event-capacity promotion changed source "
                    "projection identity"
                )
        request = per_cell_requests[0]
        prepared = build_prepared_direct_resident_eed_v1(
            None,
            qualified_cuda_module=request.qualified_cuda_module,
            logical_generation_count=logical_generation_count,
            initial_generation=initial_generations.pop(),
            per_cell_requests=tuple(direct_requests),
            source_checkpoint_projections=tuple(
                item.import_projection for item in direct_requests
            ),
            cell_ids=cell_ids,
            agent_id_allocation_generation_count=(
                request.agent_id_allocation_generation_count
            ),
        )
        source_projections = tuple(
            _eed_checkpoint_projection_mapping_v1(projection)
            for projection in prepared.source_projections
        )
        _eed_fixed_record_event_capacity_promotion_v1(
            prepared,
            cell_ids=cell_ids,
            original_event_capacities=original_event_capacities,
            batch_event_capacity=batch_event_capacity,
        )
        return PreparedNativeMechanicsOwnerV1(
            physical_window_arguments=prepared,
            source_projections=source_projections,
            cell_ids=tuple(cell_ids),
        )
    request = per_cell_requests[0]
    projection = request.source_checkpoint_projection
    if resume_projections is not None and (
        len(resume_projections) != 1 or projection is not resume_projections[0]
    ):
        raise CampaignProviderError(
            "EED/fixed-MLP resume projection was substituted"
        )
    initial_generation = 0
    if projection is not None:
        metadata = projection.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("generation")) is not int
            or metadata.get("tick") != 0
        ):
            raise CampaignProviderError(
                "EED/fixed-MLP resume boundary is invalid"
            )
        initial_generation = metadata["generation"]
    prepared = build_prepared_direct_resident_eed_v1(
        request.runner,
        qualified_cuda_module=request.qualified_cuda_module,
        cell_id=str(cell_ids[0]),
        logical_generation_count=logical_generation_count,
        initial_generation=initial_generation,
        source_checkpoint_projection=projection,
        source_capability_mask=request.source_capability_mask,
        agent_id_allocation_generation_count=(
            request.agent_id_allocation_generation_count
        ),
    )
    return PreparedNativeMechanicsOwnerV1(
        physical_window_arguments=prepared,
        source_projections=(
            _eed_checkpoint_projection_mapping_v1(prepared.initial_projection),
        ),
        cell_ids=tuple(cell_ids),
    )


def _eed_fixed_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
        execute_direct_resident_eed_v1,
    )

    prepared = physical_window_arguments
    return execute_direct_resident_eed_v1(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=(
            "eed-fixed-mlp:"
            f"{prepared.patch.source.generation_base}:"
            f"{logical_generation_count}"
        ),
    )


def _eed_fixed_project_adapter(
    prepared_adapter: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    **_kwargs: Any,
) -> Any:
    import torch

    from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
        EED_STATE_FIELDS_V1,
        project_direct_resident_eed_multicell_settlement_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        DirectResidentPatchMulticellSettlementV1,
    )

    prepared = prepared_adapter.physical_window_arguments
    completed = direct_result.get("completed_generations")
    if type(completed) is not int:
        raise CampaignProviderError(
            "EED/fixed-MLP result lacks completed generations"
        )
    cell_ids = tuple(prepared_adapter.cell_ids)
    eed_settlement = project_direct_resident_eed_multicell_settlement_v1(
        prepared,
        direct_result=direct_result,
        completed_generation_count=completed,
        reporting_enabled=reporting_enabled,
    )
    projections = tuple(
        _eed_checkpoint_projection_mapping_v1(projection)
        for projection in eed_settlement.checkpoint_projections
    )
    if len(projections) != len(cell_ids):
        raise CampaignProviderError(
            "EED checkpoint count differs from physical cell count"
        )
    base_rows = (
        _project_native_cell_metrics_v1(
            projections,
            cell_ids=cell_ids,
            direct_result=direct_result,
        )
        if reporting_enabled
        else ()
    )
    rows: list[dict[str, Any]] = []
    for cell, (cell_id, projection) in enumerate(
        zip(cell_ids, projections, strict=True)
    ):
        tensors = projection.get("tensors")
        if not isinstance(tensors, Mapping):
            raise CampaignProviderError(
                f"EED checkpoint {cell} lacks tensor authority"
            )
        missing = [name for name in EED_STATE_FIELDS_V1 if name not in tensors]
        if missing:
            raise CampaignProviderError(
                f"EED checkpoint {cell} lost authoritative state tensors: "
                + ", ".join(missing)
            )
        alive = tensors.get("alive_mask")
        axes = tensors.get("genome_resource_handling_axis")
        if (
            not isinstance(alive, torch.Tensor)
            or alive.dtype != torch.bool
            or not isinstance(axes, torch.Tensor)
            or axes.dtype != torch.float32
            or tuple(axes.shape) != tuple(alive.shape)
        ):
            raise CampaignProviderError(
                f"EED checkpoint {cell} lacks exact alive/phenotype authority"
            )
        if not reporting_enabled:
            continue
        base = base_rows[cell]
        base_metrics = dict(base["metrics"])
        generation_metrics = base.get("generation_metrics", [])
        if not isinstance(generation_metrics, list) or any(
            not isinstance(row, Mapping) for row in generation_metrics
        ):
            raise CampaignProviderError(
                f"EED generation metric projection {cell} is invalid"
            )
        from engine.campaigns.eed_opportunity_metrics_v1 import (
            EEDOpportunityMeasurementError,
            project_eed_opportunity_metrics_v1,
        )

        try:
            opportunity = project_eed_opportunity_metrics_v1(
                projection,
                generation_metrics,
            )
        except EEDOpportunityMeasurementError as exc:
            raise CampaignProviderError(
                f"EED opportunity measurement failed for cell {cell}: {exc}"
            ) from exc
        generation_metrics = list(opportunity.generation_metrics)
        active_axes = axes[alive]

        def total(name: str) -> float:
            value = tensors[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
            ):
                raise CampaignProviderError(
                    f"EED checkpoint {cell} field {name} is not exact float32"
                )
            return float(value.detach().to(device="cpu").sum().item())

        alive_count = int(alive.sum().item())
        resource_a_stock_total = total("eed_resource_a_amount")
        resource_b_stock_total = total("eed_resource_b_amount")
        resource_a_impact_load_total = total("eed_resource_a_impact_load")
        resource_b_impact_load_total = total("eed_resource_b_impact_load")

        def per_capita(value: float) -> float | None:
            return value / alive_count if alive_count else None

        rows.append(
            {
                "schema_version": EED_NATIVE_METRIC_SCHEMA_VERSION,
                "cell_id": cell_id,
                "metrics": {
                    **base_metrics,
                    **opportunity.terminal_metrics,
                    "population.alive": alive_count,
                    "eed.resource_a_stock_total": resource_a_stock_total,
                    "eed.resource_b_stock_total": resource_b_stock_total,
                    "eed.resource_a_impact_load_total": (
                        resource_a_impact_load_total
                    ),
                    "eed.resource_b_impact_load_total": (
                        resource_b_impact_load_total
                    ),
                    "eed.resource_a_stock_per_capita": per_capita(
                        resource_a_stock_total
                    ),
                    "eed.resource_b_stock_per_capita": per_capita(
                        resource_b_stock_total
                    ),
                    "eed.resource_a_impact_load_per_capita": per_capita(
                        resource_a_impact_load_total
                    ),
                    "eed.resource_b_impact_load_per_capita": per_capita(
                        resource_b_impact_load_total
                    ),
                    "eed.reproductive_opportunity_successful_births": (
                        base_metrics.get(
                            "population.successful_births_cumulative"
                        )
                    ),
                    "eed.reproductive_opportunity_survival_auc": (
                        base_metrics.get("population.survival_auc")
                    ),
                    "eed.resource_axis_mean": (
                        float(active_axes.mean().item())
                        if active_axes.numel()
                        else None
                    ),
                    "eed.resource_axis_positive_count": int(
                        (active_axes > 0).sum().item()
                    ),
                    "eed.resource_axis_negative_count": int(
                        (active_axes < 0).sum().item()
                    ),
                    "eed.resource_axis_neutral_count": int(
                        (active_axes == 0).sum().item()
                    ),
                },
                "generation_metrics": generation_metrics,
                "eed_opportunity_measurement": dict(opportunity.audit),
                "reporting_influence": "forbidden",
            }
        )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=cell_ids,
        checkpoint_projections=projections,
        compact_result_rows=tuple(rows),
        completed_generation_count=completed,
        terminal_generation=prepared.patch.source.generation_base + completed,
        segment_artifacts=tuple(eed_settlement.segment_artifacts),
        analysis_source_rows=tuple(eed_settlement.analysis_source_rows),
        lifecycle_qacct_receipts=tuple(eed_settlement.lifecycle_qacct_receipts),
        settlement_event_rows=tuple(eed_settlement.settlement_event_rows),
    )


def eed_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose reciprocal-impact world mechanics and fixed MLP under the engine."""

    from engine.campaigns.patch_runner_factory import (
        build_patch_eed_fixed_mlp_runner_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
        project_direct_resident_eed_generation_zero_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=EED_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_eed_fixed_mlp_v2",
        batch_max=48,
        generation_min=3,
        generation_max=128,
        ticks_per_generation=80,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_eed_fixed_mlp_runner_v1,
        request_builder=_eed_fixed_request_adapter,
        preparer=_eed_fixed_prepare_adapter,
        executor=_eed_fixed_execute_adapter,
        projector=_eed_fixed_project_adapter,
        generation_zero_checkpoint_projector=(
            project_direct_resident_eed_generation_zero_v1
        ),
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_eed_v1",
            "eiel_cuda_architecture_b_flattened_eed_capability_v1",
        ),
        admitted_ticks_per_generation=(80, 160),
    )


def _jungle_request_adapter(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    source_checkpoint_projection: Mapping[str, Any] | None,
    **_kwargs: Any,
) -> Any:
    return SimpleNamespace(
        runner=runner,
        qualified_cuda_module=qualified_cuda_module,
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        source_checkpoint_projection=source_checkpoint_projection,
        cuda_device="cuda",
        cuda_capability={},
    )


def _jungle_prepare_adapter(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any],
    logical_generation_count: int,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
        build_prepared_direct_resident_jungle_v4,
    )

    if len(cell_ids) != 1 or len(per_cell_requests) != 1:
        raise CampaignProviderError("Jungle/fixed-MLP execution is B1-only")
    request = per_cell_requests[0]
    projection = request.source_checkpoint_projection
    if resume_projections is not None and (
        len(resume_projections) != 1 or projection is not resume_projections[0]
    ):
        raise CampaignProviderError(
            "Jungle/fixed-MLP resume projection was substituted"
        )
    initial_generation = 0
    if projection is not None:
        metadata = projection.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("generation")) is not int
            or metadata.get("tick") != 0
        ):
            raise CampaignProviderError(
                "Jungle/fixed-MLP resume boundary is invalid"
            )
        initial_generation = metadata["generation"]
    prepared = build_prepared_direct_resident_jungle_v4(
        request.runner,
        qualified_cuda_module=request.qualified_cuda_module,
        logical_generation_count=logical_generation_count,
        initial_generation=initial_generation,
        source_checkpoint_projection=projection,
    )
    return PreparedNativeMechanicsOwnerV1(
        physical_window_arguments=prepared,
        source_projections=(prepared.initial_projection,),
        cell_ids=tuple(cell_ids),
    )


def _jungle_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
        execute_direct_resident_jungle_v4,
    )

    prepared = physical_window_arguments
    return execute_direct_resident_jungle_v4(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=(
            "jungle-fixed-mlp:"
            f"{prepared.patch.source.generation_base}:"
            f"{logical_generation_count}"
        ),
    )


def _jungle_project_adapter(
    prepared_adapter: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    **_kwargs: Any,
) -> Any:
    import torch

    from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
        JUNGLE_STATE_FIELDS_V4,
        project_direct_resident_jungle_checkpoint_v4,
    )
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        DirectResidentPatchMulticellSettlementV1,
    )

    prepared = prepared_adapter.physical_window_arguments
    completed = direct_result.get("completed_generations")
    if type(completed) is not int:
        raise CampaignProviderError(
            "Jungle/fixed-MLP result lacks completed generations"
        )
    projection = project_direct_resident_jungle_checkpoint_v4(
        prepared,
        direct_result=direct_result,
        completed_generation_count=completed,
    )
    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping):
        raise CampaignProviderError("Jungle checkpoint lacks tensor authority")
    missing = [name for name in JUNGLE_STATE_FIELDS_V4 if name not in tensors]
    if missing:
        raise CampaignProviderError(
            "Jungle checkpoint lost authoritative state tensors: "
            + ", ".join(missing)
        )
    alive = tensors.get("alive_mask")
    if not isinstance(alive, torch.Tensor) or alive.dtype != torch.bool:
        raise CampaignProviderError(
            "Jungle checkpoint lacks exact boolean alive authority"
        )
    alive_count = int(alive.detach().sum().cpu().item())
    rows = (
        (
            {
                "cell_id": prepared_adapter.cell_ids[0],
                "metrics": {"population.alive": alive_count},
            },
        )
        if reporting_enabled
        else ()
    )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=tuple(prepared_adapter.cell_ids),
        checkpoint_projections=(projection,),
        compact_result_rows=rows,
        completed_generation_count=completed,
        terminal_generation=prepared.patch.source.generation_base + completed,
    )


def jungle_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose Jungle world mechanics and fixed MLP under the engine."""

    from engine.campaigns.patch_runner_factory import (
        build_patch_jungle_fixed_mlp_runner_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=JUNGLE_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_jungle_fixed_mlp_v4",
        batch_max=1,
        generation_min=3,
        generation_max=32,
        ticks_per_generation=80,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_jungle_fixed_mlp_runner_v1,
        request_builder=_jungle_request_adapter,
        preparer=_jungle_prepare_adapter,
        executor=_jungle_execute_adapter,
        projector=_jungle_project_adapter,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_jungle_v4",
        ),
    )


def _wdl_jungle_prepare_adapter(**kwargs: Any) -> Any:
    return _wdl_fixed_prepare_adapter(jungle_overlay=True, **kwargs)


def _wdl_jungle_eed_prepare_adapter(**kwargs: Any) -> Any:
    return _wdl_fixed_prepare_adapter(jungle_overlay=True, eed_overlay=True, **kwargs)


def _wdl_jungle_eed_hapb_policy_prepare_adapter(**kwargs: Any) -> Any:
    """Prepare the B48 policy-only HAPB route through the normal owner seam."""

    return _wdl_fixed_prepare_adapter(
        jungle_overlay=True,
        eed_overlay=True,
        hapb_host_policy_enable=True,
        **kwargs,
    )


def _validated_tick_native_abi_span_v1(
    prepared: Any,
    absolute_tick_span: object,
    *,
    absolute_start_next_tick: object | None = None,
    authoritative_next_absolute_tick: object | None = None,
) -> int:
    """Fail closed before entering the absolute-tick CUDA ABI."""

    if (
        type(absolute_start_next_tick) is not int
        or absolute_start_next_tick < 0
        or type(authoritative_next_absolute_tick) is not int
        or authoritative_next_absolute_tick < 0
        or absolute_start_next_tick != authoritative_next_absolute_tick
        or type(absolute_tick_span) is not int
        or absolute_tick_span < 1
    ):
        raise RuntimeError(
            "tick-native dispatch requires one exact authoritative absolute cursor "
            "and a positive span"
        )
    return absolute_tick_span


def _wdl_jungle_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    absolute_tick_span: int | None = None,
    next_absolute_tick: int | None = None,
    authoritative_next_absolute_tick: int | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2,
        execute_direct_resident_wdl_jungle_v1,
    )

    prepared = physical_window_arguments
    if absolute_tick_span is not None:
        span_ticks = _validated_tick_native_abi_span_v1(
            prepared,
            absolute_tick_span,
            absolute_start_next_tick=next_absolute_tick,
            authoritative_next_absolute_tick=authoritative_next_absolute_tick,
        )
        return execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2(
            module,
            prepared,
            span_ticks=span_ticks,
            absolute_start_next_tick=next_absolute_tick,
            source_window_ticks=dict(tick_native_control or {}).get(
                "source_window_ticks"
            ),
        )
    return execute_direct_resident_wdl_jungle_v1(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=(
            "wdl-jungle-fixed-mlp:"
            f"{prepared.patch.source.generation_base}:"
            f"{logical_generation_count}"
        ),
    )


def _wdl_jungle_eed_hapb_policy_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    """Dispatch only the guarded no-birth/no-turnover HAPB policy trajectory."""

    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        execute_direct_resident_wdl_jungle_eed_hapb_policy_v1,
    )

    prepared = physical_window_arguments
    return execute_direct_resident_wdl_jungle_eed_hapb_policy_v1(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=(
            "wdl-jungle-eed-hapb-policy:"
            f"{prepared.patch.source.generation_base}:"
            f"{logical_generation_count}"
        ),
    )


def _wdl_jungle_eed_hapb_birth_prepare_adapter(**kwargs: Any) -> Any:
    """Prepare the explicit ordinary-birth HAPB owner; turnover stays barred."""

    return _wdl_jungle_eed_hapb_policy_prepare_adapter(**kwargs)


def _wdl_jungle_eed_hapb_birth_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    absolute_tick_span: int | None = None,
    next_absolute_tick: int | None = None,
    authoritative_next_absolute_tick: int | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        execute_direct_resident_wdl_jungle_eed_hapb_birth_v1,
        execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1,
    )

    prepared = physical_window_arguments
    if absolute_tick_span is not None:
        span_ticks = _validated_tick_native_abi_span_v1(
            prepared,
            absolute_tick_span,
            absolute_start_next_tick=next_absolute_tick,
            authoritative_next_absolute_tick=authoritative_next_absolute_tick,
        )
        return execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1(
            module,
            prepared,
            span_ticks=span_ticks,
            absolute_start_next_tick=next_absolute_tick,
            source_window_ticks=dict(tick_native_control or {}).get(
                "source_window_ticks"
            ),
        )
    return execute_direct_resident_wdl_jungle_eed_hapb_birth_v1(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=(
            "wdl-jungle-eed-hapb-birth:"
            f"{prepared.patch.source.generation_base}:"
            f"{logical_generation_count}"
        ),
    )


def _tick_native_private_projection_cursor_v1(
    direct_result: Mapping[str, Any],
    *,
    source: Any | None = None,
) -> tuple[dict[str, Any], int, int, int]:
    """Derive one private settlement cursor from absolute ticks.

    ``source`` is supplied by the Patch projector when its private settlement
    ABI validates the prepared reusable-source capacity envelope.  Its exact
    cursor and capacity are re-derived from absolute-tick fields before the
    legacy aliases are admitted.  The source origin is an absolute-tick lower
    bound and the reusable ring is bounded by the nominal dispatch span; it is
    not an absolute interval that expires after the first owner epoch.  The
    generation-shaped aliases are never inferred from the native result and
    never returned to the public provider boundary.
    """

    absolute_start = direct_result.get("tick_native_start_next_tick")
    absolute_end = direct_result.get("tick_native_end_next_tick")
    absolute_span = direct_result.get("tick_span_count")
    span_start_alias = direct_result.get("tick_span_start")
    span_end_alias = direct_result.get("tick_span_end")
    source_window_ticks = direct_result.get("source_window_ticks")
    source_window_count = direct_result.get("source_window_count")
    if (
        type(absolute_start) is not int
        or absolute_start < 0
        or type(absolute_end) is not int
        or absolute_end <= absolute_start
        or type(absolute_span) is not int
        or absolute_span < 1
        or absolute_end - absolute_start != absolute_span
        or type(span_start_alias) is not int
        or span_start_alias != absolute_start
        or type(span_end_alias) is not int
        or span_end_alias != absolute_end
        or type(source_window_ticks) is not int
        or source_window_ticks < 1
        or type(source_window_count) is not int
        or source_window_count
        != (
            absolute_start % source_window_ticks
            + absolute_span
            + source_window_ticks
            - 1
        )
        // source_window_ticks
    ):
        raise CampaignProviderError(
            "tick-native result has inconsistent absolute "
            "start/end/span authority"
        )

    terminal_generation, terminal_tick = divmod(
        absolute_end, source_window_ticks
    )
    private_generation_base = (
        terminal_generation if terminal_tick else terminal_generation - 1
    )
    private_span_end = terminal_tick or source_window_ticks
    # The generation-era projector consumes only the final private-window
    # prefix represented by this absolute span.  A restored suffix such as
    # [3359,3360) therefore owns private rows [159,160), not [0,160).
    # Longer native spans still collapse to their final complete/partial
    # window exactly as before.  These aliases never escape this adapter.
    private_span_start = max(0, private_span_end - absolute_span)
    if private_generation_base < 0 or not 1 <= private_span_end <= source_window_ticks:
        raise CampaignProviderError(
            "absolute cursor cannot form one private settlement window"
        )

    projector_result = dict(direct_result)
    prefixes = projector_result.get("tick_native_source_prefix_by_cell")
    if prefixes is not None:
        if (
            not isinstance(prefixes, Sequence)
            or isinstance(prefixes, (str, bytes, bytearray))
            or any(
                not isinstance(prefix, Mapping)
                or not isinstance(prefix.get("metadata"), Mapping)
                for prefix in prefixes
            )
        ):
            raise CampaignProviderError(
                "WDL/Jungle tick-native source prefix authority is incomplete"
            )
        for prefix in prefixes:
            metadata = prefix["metadata"]
            prefix_generation = metadata.get("generation")
            prefix_tick = metadata.get("tick")
            if (
                type(prefix_generation) is not int
                or prefix_generation < 0
                or type(prefix_tick) is not int
                or not 0 <= prefix_tick < source_window_ticks
                or prefix_generation * source_window_ticks + prefix_tick
                != absolute_start
            ):
                raise CampaignProviderError(
                    "WDL/Jungle tick-native source prefix cursor differs from "
                    "the requested absolute-tick start"
                )
        private_prefixes = []
        for prefix in prefixes:
            private_prefix = dict(prefix)
            metadata = dict(prefix["metadata"])
            metadata.update(
                {
                    "generation": private_generation_base,
                    "tick": private_span_start,
                    "prediction_tick": private_span_start,
                    "agent_grid_runner_phase": (
                        "generation_boundary"
                        if private_span_start == 0
                        else "partial_generation"
                    ),
                }
            )
            private_prefix["metadata"] = metadata
            private_prefixes.append(private_prefix)
        projector_result["tick_native_source_prefix_by_cell"] = tuple(
            private_prefixes
        )

    if source is not None:
        source_time_authority = getattr(source, "time_authority", None)
        source_window_ticks_value = getattr(source, "source_window_ticks", None)
        source_abi_base = getattr(source, "generation_base", None)
        source_capacity_window_count = getattr(
            source, "logical_generation_count", None
        )
        source_physical_window_count = getattr(
            source, "physical_window_generations", None
        )
        source_owner_span_ticks = getattr(
            source, "resident_owner_span_ticks", None
        )
        source_dispatch_span_ticks = getattr(
            source, "native_dispatch_window_ticks", None
        )
        source_allows_partial_cursor = getattr(
            source, "allow_partial_generation_source_cursor", None
        )
        checkpoint_inputs = getattr(source, "checkpoint_projection_inputs", None)
        canonical_source = (
            checkpoint_inputs.get("pre_generation_canonical_projection")
            if isinstance(checkpoint_inputs, Mapping)
            else None
        )
        canonical_metadata = (
            canonical_source.get("metadata")
            if isinstance(canonical_source, Mapping)
            else None
        )
        source_cursor_abi_base = (
            canonical_metadata.get("generation")
            if isinstance(canonical_metadata, Mapping)
            else None
        )
        source_cursor_tick = (
            canonical_metadata.get("tick")
            if isinstance(canonical_metadata, Mapping)
            else None
        )
        if source_time_authority != "absolute_tick_v1":
            raise CampaignProviderError(
                "prepared tick-native source time authority is not "
                "absolute-tick authoritative"
            )
        if (
            type(source_window_ticks_value) is not int
            or source_window_ticks_value < 1
            or source_window_ticks_value != source_window_ticks
        ):
            raise CampaignProviderError(
                "prepared tick-native source window does not match the "
                "absolute-tick result"
            )
        if (
            type(source_abi_base) is not int
            or source_abi_base < 0
            or type(source_capacity_window_count) is not int
            or not 1 <= source_capacity_window_count <= 128
            or type(source_physical_window_count) is not int
            or source_physical_window_count not in {1, 2}
            or type(source_owner_span_ticks) is not int
            or source_owner_span_ticks < 1
            or type(source_dispatch_span_ticks) is not int
            or not 1 <= source_dispatch_span_ticks <= source_owner_span_ticks
            or type(source_allows_partial_cursor) is not bool
            or type(source_cursor_abi_base) is not int
            or source_cursor_abi_base != source_abi_base
            or type(source_cursor_tick) is not int
            or not 0 <= source_cursor_tick < source_window_ticks_value
            or (source_cursor_tick != 0 and not source_allows_partial_cursor)
        ):
            raise CampaignProviderError(
                "prepared tick-native source has an inconsistent absolute-tick "
                "settlement capacity envelope"
            )
        expected_source_capacity_window_count = (
            source_dispatch_span_ticks
            + source_window_ticks_value
            - 1
        ) // source_window_ticks_value
        if source_capacity_window_count != expected_source_capacity_window_count:
            raise CampaignProviderError(
                "prepared tick-native source has an inconsistent absolute-tick "
                "settlement capacity envelope"
            )
        source_origin_absolute_tick = (
            source_cursor_abi_base * source_window_ticks_value + source_cursor_tick
        )
        if (
            source_origin_absolute_tick > absolute_start
            or absolute_span > source_dispatch_span_ticks
        ):
            raise CampaignProviderError(
                "prepared tick-native source has an inconsistent absolute-tick "
                "settlement capacity envelope"
            )
        projector_result.update(
            {
                # These are private projector aliases only.  The public
                # absolute-tick direct result remains unchanged at the caller.
                "physical_window_generations": source_physical_window_count,
                "logical_generation_count": source_capacity_window_count,
                "window_count": (
                    (
                        source_abi_base % source_physical_window_count
                        + source_capacity_window_count
                        + source_physical_window_count
                        - 1
                    )
                    // source_physical_window_count
                ),
                "physical_window_alignment": "absolute_generation_lattice_v1",
            }
        )

    return (
        {
            **projector_result,
            # These aliases enter only the private generation-era projector.
            # The retained direct result remains absolute-tick authoritative.
            "completed_generations": 1,
            "tick_span_start": private_span_start,
            "tick_span_end": private_span_end,
            "tick_span_count": private_span_end - private_span_start,
            "terminal_generation_boundary": terminal_tick == 0,
        },
        private_generation_base,
        terminal_generation,
        terminal_tick,
    )


def _wdl_jungle_project_adapter(
    prepared_adapter: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    bounded_projection_cursor: bool = False,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        project_direct_resident_wdl_jungle_multicell_settlement_v1,
    )

    prepared = prepared_adapter.physical_window_arguments
    cell_ids = tuple(prepared_adapter.cell_ids)
    if not cell_ids or any(type(cell_id) is not str or not cell_id for cell_id in cell_ids):
        raise CampaignProviderError("WDL/Jungle projection requires stable cell IDs")
    tick_native_owner = direct_result.get("absolute_tick_span_native_owner") is True
    if tick_native_owner:
        source = getattr(getattr(prepared, "patch", None), "source", None)
        (
            projector_result,
            generation_base_override,
            terminal_generation,
            terminal_tick,
        ) = _tick_native_private_projection_cursor_v1(
            direct_result,
            source=source,
        )
        if source is None:
            raise CampaignProviderError(
                "prepared tick-native owner lacks its private settlement source"
            )
        completed = 1
    else:
        completed = direct_result.get("completed_generations")
        if type(completed) is not int:
            raise CampaignProviderError(
                "WDL/Jungle result lacks completed generations"
            )
        projector_result = direct_result
        generation_base_override = direct_result.get(
            "tick_native_generation_base"
        )
        if generation_base_override is not None and (
            type(generation_base_override) is not int
            or generation_base_override < 0
        ):
            raise CampaignProviderError(
                "WDL/Jungle result has an invalid private generation cursor"
            )
    settlement = project_direct_resident_wdl_jungle_multicell_settlement_v1(
        prepared,
        direct_result=projector_result,
        completed_generation_count=completed,
        reporting_enabled=reporting_enabled,
        generation_base_override=generation_base_override,
        bounded_projection_cursor=bounded_projection_cursor,
    )
    projections = tuple(settlement.checkpoint_projections)
    if tick_native_owner:
        if bounded_projection_cursor:
            cursor = settlement.checkpoint_projection_cursor
            if cursor is None:
                raise CampaignProviderError(
                    "tick-native bounded terminal projection lacks a cursor"
                )

            def attach_terminal_metrics(
                detached: Any,
            ) -> Any:
                rows = _tick_native_terminal_metric_rows_v1(
                    (detached.require_projection(),),
                    cell_ids=(detached.cell_id,),
                )
                return replace(detached, compact_result_rows=rows)

            cursor = cursor.map(attach_terminal_metrics)
            return replace(
                settlement,
                compact_result_rows=(),
                checkpoint_projection_cursor=cursor,
                completed_generation_count=completed,
                terminal_generation=terminal_generation,
                terminal_tick=terminal_tick,
            )
        return replace(
            settlement,
            compact_result_rows=_tick_native_terminal_metric_rows_v1(
                projections,
                cell_ids=cell_ids,
            ),
            completed_generation_count=completed,
            terminal_generation=terminal_generation,
            terminal_tick=terminal_tick,
        )
    turnover_counts, turnover_count_generations = _turnover_count_authority_v1(
        prepared.wdl, direct_result
    )
    native_rows = _project_native_cell_metrics_v1(
        projections,
        cell_ids=cell_ids,
        direct_result=direct_result,
        wdl_generation_metrics=prepared.wdl.reporting_tensors[5][:, :completed],
        turnover_counts=turnover_counts,
        turnover_count_generations=turnover_count_generations,
    )
    return replace(
        settlement,
        compact_result_rows=(
            native_rows
            if reporting_enabled
            else _compact_terminal_native_metric_rows_v1(native_rows)
        ),
        completed_generation_count=completed,
        terminal_generation=prepared.patch.source.generation_base + completed,
    )


def wdl_jungle_fixed_mlp_native_binding_v1() -> NativeMechanicsBindingV1:
    """Compose one non-EED Patch/WDL owner with a Jungle-only overlay."""

    from engine.campaigns.patch_runner_factory import (
        build_patch_wdl_jungle_fixed_mlp_runner_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
        project_direct_resident_wdl_jungle_generation_zero_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=WDL_JUNGLE_FIXED_MLP_BINDING_ID,
        artifact_abi="architecture_b_wdl_jungle_fixed_mlp_native_v1",
        batch_max=48,
        generation_min=3,
        generation_max=128,
        ticks_per_generation=160,
        policy_cache_refill=True,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_wdl_jungle_fixed_mlp_runner_v1,
        request_builder=_wdl_fixed_request_adapter,
        preparer=_wdl_jungle_prepare_adapter,
        executor=_wdl_jungle_execute_adapter,
        projector=_wdl_jungle_project_adapter,
        generation_zero_checkpoint_projector=(
            project_direct_resident_wdl_jungle_generation_zero_v1
        ),
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_v1",
        ),
        admitted_ticks_per_generation=(160,),
    )


# Compatibility import for callers that have not yet refreshed their binding name.
wdl_jungle_fixed_mlp_native_binding_candidate_v1 = wdl_jungle_fixed_mlp_native_binding_v1


def _gru_request_adapter(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    source_checkpoint_projection: Mapping[str, Any] | None,
    runner_build: Any,
    cell: Mapping[str, Any],
    wdl: bool,
    **extra_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
        GRU_CAPABILITIES_V3,
    )
    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
        build_direct_resident_source_allocation_request_v1,
    )

    kwargs: dict[str, Any] = {
        "qualified_cuda_module": qualified_cuda_module,
        "policy_family": "gru",
        "source_capability_mask": GRU_CAPABILITIES_V3,
        "logical_generation_count": logical_generation_count,
        "agent_id_allocation_generation_count": (
            agent_id_allocation_generation_count
        ),
        "source_checkpoint_projection": source_checkpoint_projection,
    }
    event_capacity_override = extra_kwargs.get("event_capacity_override")
    if event_capacity_override is not None:
        kwargs["event_capacity_override"] = event_capacity_override
    receipt = dict(runner_build.receipt)
    task_seed: int | None = None
    intensity: float | None = None
    if wdl:
        from engine.perf.cuda_architecture_b_direct_resident_wdl_gru_v3 import (
            WDL_GRU_CAPABILITIES_V3,
            derive_direct_resident_wdl_gru_runner_envelope_v3,
        )

        kwargs["source_capability_mask"] = WDL_GRU_CAPABILITIES_V3
        kwargs["runner_envelope_deriver"] = (
            derive_direct_resident_wdl_gru_runner_envelope_v3
        )
        wdl_receipt = receipt.get("wdl")
        if not isinstance(wdl_receipt, Mapping):
            raise CampaignProviderError("WDL/GRU runner receipt lacks world identity")
        task_seed = wdl_receipt.get("task_seed")
        intensity = wdl_receipt.get("intensity")
        if type(task_seed) is not int or type(intensity) is not float:
            raise CampaignProviderError(
                "WDL/GRU runner receipt lacks task seed or intensity"
            )
    native_request = build_direct_resident_source_allocation_request_v1(
        runner, **kwargs
    )
    return SimpleNamespace(
        native_request=native_request,
        cuda_capability=native_request.cuda_capability,
        cuda_device=native_request.cuda_device,
        source_checkpoint_projection=source_checkpoint_projection,
        task_seed=task_seed,
        intensity=intensity,
        cell=cell,
    )


def _patch_gru_request_adapter(runner: Any, **kwargs: Any) -> Any:
    return _gru_request_adapter(runner, wdl=False, **kwargs)


def _wdl_gru_request_adapter(runner: Any, **kwargs: Any) -> Any:
    return _gru_request_adapter(runner, wdl=True, **kwargs)


def _initial_generation(
    projections: Sequence[Any] | None,
) -> int:
    if not projections:
        return 0
    projection = projections[0]
    metadata = (
        projection.get("metadata")
        if isinstance(projection, Mapping)
        else getattr(projection, "metadata", None)
    )
    generation = metadata.get("generation") if isinstance(metadata, Mapping) else None
    if type(generation) is not int:
        raise CampaignProviderError("GRU resume projection has no generation")
    return generation


def _patch_gru_prepare_adapter(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any],
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
        build_prepared_direct_resident_gru_v3,
    )

    native_requests = tuple(row.native_request for row in per_cell_requests)
    prepared = build_prepared_direct_resident_gru_v3(
        qualified_cuda_module=native_requests[0].qualified_cuda_module,
        per_cell_requests=native_requests,
        source_checkpoint_projections=resume_projections,
        cell_ids=tuple(cell_ids),
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        initial_generation=_initial_generation(resume_projections),
    )
    return PreparedNativeMechanicsOwnerV1(
        physical_window_arguments=prepared,
        source_projections=prepared.source_projections,
        cell_ids=tuple(cell_ids),
    )


def _wdl_gru_prepare_adapter(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any],
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_gru_v3 import (
        build_prepared_direct_resident_wdl_gru_v3,
    )

    native_requests = tuple(row.native_request for row in per_cell_requests)
    prepared = build_prepared_direct_resident_wdl_gru_v3(
        qualified_cuda_module=native_requests[0].qualified_cuda_module,
        per_cell_requests=native_requests,
        source_checkpoint_projections=resume_projections,
        cell_ids=tuple(cell_ids),
        task_seeds=tuple(int(row.task_seed) for row in per_cell_requests),
        intensities=tuple(float(row.intensity) for row in per_cell_requests),
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        initial_generation=_initial_generation(resume_projections),
    )
    return PreparedNativeMechanicsOwnerV1(
        physical_window_arguments=prepared,
        source_projections=prepared.source_projections,
        cell_ids=tuple(cell_ids),
    )


def _patch_gru_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
        execute_direct_resident_gru_physical_v3,
        execute_direct_resident_gru_v3,
    )

    prepared = physical_window_arguments
    if logical_generation_count <= 2:
        if len(prepared.cell_ids) != 1:
            raise CampaignProviderError(
                "physical Patch/GRU g1/g2 segments currently require B1"
            )
        return execute_direct_resident_gru_physical_v3(
            module,
            prepared,
            generation_count=logical_generation_count,
            ticks_per_generation=80,
        )
    return execute_direct_resident_gru_v3(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=f"patch-gru:{_initial_generation(prepared.source_projections)}",
    )


def _wdl_gru_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_wdl_gru_v3 import (
        execute_direct_resident_wdl_gru_physical_v3,
        execute_direct_resident_wdl_gru_v3,
    )

    prepared = physical_window_arguments
    if logical_generation_count <= 2:
        if len(prepared.cell_ids) != 1:
            raise CampaignProviderError(
                "physical WDL/GRU g1/g2 segments currently require B1"
            )
        return execute_direct_resident_wdl_gru_physical_v3(
            module,
            prepared,
            generation_count=logical_generation_count,
            ticks_per_generation=80,
        )
    return execute_direct_resident_wdl_gru_v3(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=f"wdl-gru:{_initial_generation(prepared.source_projections)}",
    )


def _gru_metric_rows(
    projections: Sequence[Mapping[str, Any]],
    *,
    cell_ids: Sequence[str],
    direct_result: Mapping[str, Any],
    wdl_generation_metrics: Any | None,
) -> tuple[dict[str, Any], ...]:
    rows = _project_native_cell_metrics_v1(
        projections,
        cell_ids=cell_ids,
        direct_result=direct_result,
        wdl_generation_metrics=wdl_generation_metrics,
    )
    for row, projection in zip(rows, projections, strict=True):
        tensors = projection.get("tensors")
        alive = tensors.get("alive_mask") if isinstance(tensors, Mapping) else None
        if hasattr(alive, "detach"):
            values = alive.detach().cpu().tolist()
        elif hasattr(alive, "tolist"):
            values = alive.tolist()
        else:
            raise CampaignProviderError(
                "GRU checkpoint has no native alive_mask authority"
            )
        if not isinstance(values, list) or any(type(value) is not bool for value in values):
            raise CampaignProviderError(
                "GRU checkpoint alive_mask must be exact boolean state"
            )
        if row["metrics"]["population.alive"] != sum(values):
            raise CampaignProviderError(
                "GRU terminal alive metric differs from checkpoint authority"
            )
    return rows


def _gru_project_adapter(
    prepared_adapter: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    wdl: bool,
    **_kwargs: Any,
) -> Any:
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        DirectResidentPatchMulticellSettlementV1,
    )

    prepared = prepared_adapter.physical_window_arguments
    completed = direct_result.get("completed_generations")
    if type(completed) is not int:
        raise CampaignProviderError("GRU result lacks completed generations")
    if wdl:
        from engine.perf.cuda_architecture_b_direct_resident_wdl_gru_v3 import (
            project_direct_resident_wdl_gru_multicell_checkpoints_v3,
        )

        projected = project_direct_resident_wdl_gru_multicell_checkpoints_v3(
            prepared,
            direct_result=direct_result,
            completed_generation_count=completed,
            ordered_source_projections=prepared.source_projections,
        )
        projections = tuple(
            projected.checkpoint_projections
            if hasattr(projected, "checkpoint_projections")
            else projected
        )
    else:
        from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
            project_direct_resident_gru_multicell_checkpoints_v3,
        )

        projections = tuple(
            project_direct_resident_gru_multicell_checkpoints_v3(
                prepared,
                direct_result=direct_result,
                completed_generation_count=completed,
                ordered_source_projections=prepared.source_projections,
            )
        )
    metadata = projections[0].get("metadata")
    terminal = metadata.get("generation") if isinstance(metadata, Mapping) else None
    if type(terminal) is not int:
        raise CampaignProviderError("GRU projection lacks terminal generation")
    rows = (
        _gru_metric_rows(
            projections,
            cell_ids=prepared_adapter.cell_ids,
            direct_result=direct_result,
            wdl_generation_metrics=(
                prepared.wdl_reporting_tensors[5][:, :completed]
                if wdl
                else None
            ),
        )
        if reporting_enabled
        else ()
    )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=tuple(prepared_adapter.cell_ids),
        checkpoint_projections=projections,
        compact_result_rows=rows,
        completed_generation_count=completed,
        terminal_generation=terminal,
    )


def _patch_gru_project_adapter(prepared_adapter: Any, **kwargs: Any) -> Any:
    return _gru_project_adapter(prepared_adapter, wdl=False, **kwargs)


def _wdl_gru_project_adapter(prepared_adapter: Any, **kwargs: Any) -> Any:
    return _gru_project_adapter(prepared_adapter, wdl=True, **kwargs)


def patch_stable_gru_native_binding_v1() -> NativeMechanicsBindingV1:
    from engine.campaigns.patch_runner_factory import (
        build_patch_stable_gru_runner_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=PATCH_STABLE_GRU_BINDING_ID,
        artifact_abi="architecture_b_gru_v3",
        batch_max=65535,
        generation_min=1,
        generation_max=32,
        ticks_per_generation=80,
        policy_cache_refill=False,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_stable_gru_runner_v1,
        request_builder=_patch_gru_request_adapter,
        preparer=_patch_gru_prepare_adapter,
        executor=_patch_gru_execute_adapter,
        projector=_patch_gru_project_adapter,
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_gru_v3",
        ),
    )


def wdl_stable_gru_native_binding_v1() -> NativeMechanicsBindingV1:
    from engine.campaigns.patch_runner_factory import (
        build_patch_wdl_stable_gru_runner_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=WDL_STABLE_GRU_BINDING_ID,
        artifact_abi="architecture_b_wdl_gru_v3",
        batch_max=65535,
        generation_min=1,
        generation_max=32,
        ticks_per_generation=80,
        policy_cache_refill=False,
        terminal_snapshot_metrics=True,
        runner_builder=build_patch_wdl_stable_gru_runner_v1,
        request_builder=_wdl_gru_request_adapter,
        preparer=_wdl_gru_prepare_adapter,
        executor=_wdl_gru_execute_adapter,
        projector=_wdl_gru_project_adapter,
        terminal_snapshot_metrics_merger=_merge_native_cell_metric_snapshots_v1,
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_gru_v3",
        ),
    )


def _gmnw_runner_builder(
    cell: Mapping[str, Any],
    run_dir: Any,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    **_kwargs: Any,
) -> Any:
    return SimpleNamespace(
        runner=SimpleNamespace(
            cell=dict(cell),
            run_dir=run_dir,
            generations=generations,
            ticks_per_generation=ticks_per_generation,
            artifact_path=artifact_path,
            qualification_receipt_path=qualification_receipt_path,
            reporting_receipt_path=reporting_receipt_path,
        ),
        receipt={
            "schema_version": "eiel_gmnw_engine_source_request_v1",
            "cell_id": cell.get("cell_id"),
        },
    )


def _gmnw_request_adapter(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    source_checkpoint_projection: Mapping[str, Any] | None,
    **_kwargs: Any,
) -> Any:
    return SimpleNamespace(
        source=runner,
        qualified_cuda_module=qualified_cuda_module,
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        source_checkpoint_projection=source_checkpoint_projection,
        cuda_device="cuda",
    )


def _gmnw_configs(cells: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
        GmnwConfigurationV1,
    )

    result: list[Any] = []
    for cell in cells:
        parameters = dict(cell.get("parameters", {}))
        seeds = dict(cell.get("seeds", {}))
        arm = str(parameters["gmnw.policy_arm"])
        cue_visible = bool(
            parameters.get("gmnw.cue_visible_at_decision", False)
        )
        if arm == "reset_gru" and cue_visible:
            raise CampaignProviderError(
                "reset_gru does not admit cue_visible_at_decision=true"
            )
        founder_count = parameters.get("gmnw.gru_founder_count")
        if arm == "mixed_selection":
            if type(founder_count) is not int or not 0 <= founder_count <= 64:
                raise CampaignProviderError(
                    "mixed_selection requires gmnw.gru_founder_count in [0,64]"
                )
        else:
            founder_count = None
        profile = parameters.get("gmnw.venv_consequence_profile", "legacy")
        terminal = parameters.get(
            "gmnw.terminal_low_population_no_reinit", False
        )
        if profile not in {"legacy", "P1", "P2", "P3"} or type(terminal) is not bool:
            raise CampaignProviderError("GMNW consequence profile is invalid")
        result.append(
            GmnwConfigurationV1(
                cue_alphabet=int(parameters.get("gmnw.cue_alphabet", 2)),
                cue_ticks=int(parameters.get("gmnw.cue_ticks", 1)),
                delay_ticks=int(parameters.get("gmnw.delay_ticks", 4)),
                consequence_ticks=int(
                    parameters.get("gmnw.consequence_ticks", 1)
                ),
                distractors_enabled=bool(
                    parameters.get("gmnw.distractors_enabled", False)
                ),
                cue_visible_at_decision=cue_visible,
                reset_hidden_each_tick=arm == "reset_gru",
                policy_arm=arm,
                gru_founder_count=founder_count,
                task_seed=int(seeds.get("task", seeds.get("ecology", 7001))),
                venv_consequence_profile=str(profile),
                terminal_low_population_no_reinit=terminal,
            )
        )
    return tuple(result)


def _gmnw_prepare_adapter(
    *,
    plan: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[Any],
    logical_generation_count: int,
    agent_id_allocation_generation_count: int,
    resume_projections: Sequence[Mapping[str, Any]] | None,
    **_kwargs: Any,
) -> Any:
    from pathlib import Path

    from engine.campaigns.patch_runner_factory import (
        build_patch_fixed_runner_v1,
        build_patch_stable_gru_runner_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
        build_prepared_direct_resident_gmnw_gru_v1,
        gmnw_allocation_generation_capacity_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
        GRU_CAPABILITIES_V3,
    )
    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
        QUALIFIED_CAPABILITY_MASK_V1,
        build_direct_resident_source_allocation_request_v1,
    )

    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CampaignProviderError("GMNW engine binding lacks artifacts")

    def artifact_path(name: str) -> str:
        row = artifacts.get(name)
        path = row.get("path") if isinstance(row, Mapping) else None
        if not isinstance(path, str) or not path:
            raise CampaignProviderError(f"GMNW artifact is absent: {name}")
        return path

    module = per_cell_requests[0].qualified_cuda_module
    initial_generation = _initial_generation(resume_projections)
    source_capacity = gmnw_allocation_generation_capacity_v1(
        logical_generation_count
    )
    source_requests: list[Any] = []
    patch_names = (
        "patch.grid_size",
        "patch.initial_population",
        "patch.capacity",
        "patch.founder_placement",
    )
    for index, (cell, request) in enumerate(
        zip(cells, per_cell_requests, strict=True)
    ):
        parameters = dict(cell.get("parameters", {}))
        source_projection = (
            None if resume_projections is None else resume_projections[index]
        )
        metadata = (
            source_projection.get("metadata")
            if isinstance(source_projection, Mapping)
            else None
        )
        policy_family = (
            str(metadata.get("native_runtime_fixed_policy_family"))
            if isinstance(metadata, Mapping)
            else "gru"
        )
        root = Path(request.source.run_dir)
        if policy_family == "gru":
            source_cell = {
                **dict(cell),
                "parameters": {
                    **{name: parameters[name] for name in patch_names},
                    "policy.hidden_width": 64,
                    "policy.reset_hidden_each_tick": False,
                },
            }
            build = build_patch_stable_gru_runner_v1(
                source_cell,
                root,
                generations=source_capacity,
                ticks_per_generation=80,
                artifact_path=artifact_path("cuda_extension"),
                qualification_receipt_path=artifact_path(
                    "source_qualification_receipt"
                ),
                reporting_receipt_path=artifact_path(
                    "source_reporting_receipt"
                ),
            )
            capabilities = GRU_CAPABILITIES_V3
        elif policy_family == "mlp":
            source_cell = {
                **dict(cell),
                "parameters": {
                    name: parameters[name] for name in patch_names
                },
            }
            build = build_patch_fixed_runner_v1(
                source_cell,
                root,
                generations=source_capacity,
                ticks_per_generation=80,
                artifact_path=artifact_path("cuda_extension"),
                qualification_receipt_path=artifact_path(
                    "source_qualification_receipt"
                ),
                reporting_receipt_path=artifact_path(
                    "source_reporting_receipt"
                ),
                max_births_per_generation=1,
            )
            capabilities = QUALIFIED_CAPABILITY_MASK_V1
        else:
            raise CampaignProviderError(
                "GMNW resume supports only GRU or matched-MLP sources"
            )
        source_requests.append(
            build_direct_resident_source_allocation_request_v1(
                build.runner,
                qualified_cuda_module=module,
                policy_family=policy_family,
                source_capability_mask=capabilities,
                logical_generation_count=source_capacity,
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
                source_checkpoint_projection=source_projection,
            )
        )
    owner = build_prepared_direct_resident_gmnw_gru_v1(
        qualified_cuda_module=module,
        cell_configs=_gmnw_configs(cells),
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        initial_generation=initial_generation,
        gmnw_generation_base=0,
        per_cell_requests=tuple(source_requests),
        source_checkpoint_projections=(
            None if resume_projections is None else tuple(resume_projections)
        ),
        cell_ids=tuple(cell_ids),
        witness_path=Path(artifact_path("gmnw_witness")),
        matched_mlp_path=Path(artifact_path("matched_mlp_artifact")),
        selection_metrics_enabled=True,
    )
    sources = getattr(owner, "source_projections", None)
    if sources is None:
        sources = getattr(getattr(owner, "patch", None), "source_projections", None)
    if not isinstance(sources, Sequence):
        raise CampaignProviderError("GMNW owner lacks source projections")
    return PreparedNativeMechanicsOwnerV1(
        physical_window_arguments=owner,
        source_projections=tuple(sources),
        cell_ids=tuple(cell_ids),
    )


def _gmnw_execute_adapter(
    module: Any,
    *,
    physical_window_arguments: Any,
    logical_generation_count: int,
    **_kwargs: Any,
) -> Mapping[str, Any]:
    from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
        execute_direct_resident_gmnw_gru_v1,
    )

    return execute_direct_resident_gmnw_gru_v1(
        module,
        physical_window_arguments,
        logical_generation_count=logical_generation_count,
        run_id=f"gmnw-engine:{logical_generation_count}",
    )


def _gmnw_project_adapter(
    prepared_adapter: Any,
    *,
    direct_result: Mapping[str, Any],
    reporting_enabled: bool,
    **_kwargs: Any,
) -> Any:
    import torch

    from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
        project_direct_resident_gmnw_gru_multicell_checkpoints_v1,
    )
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        DirectResidentPatchMulticellSettlementV1,
    )

    owner = prepared_adapter.physical_window_arguments
    completed = direct_result.get("completed_generations")
    if type(completed) is not int:
        raise CampaignProviderError("GMNW result lacks completed generations")
    projections = tuple(
        project_direct_resident_gmnw_gru_multicell_checkpoints_v1(
            owner,
            direct_result=direct_result,
            completed_generation_count=completed,
            ordered_source_projections=prepared_adapter.source_projections,
        )
    )
    selection = direct_result.get("gmnw_selection_metrics")
    selection = selection if isinstance(selection, Mapping) else {}
    rows: list[dict[str, Any]] = []
    for index, (cell_id, projection) in enumerate(
        zip(prepared_adapter.cell_ids, projections, strict=True)
    ):
        tensors = projection.get("tensors")
        alive = tensors.get("alive_mask") if isinstance(tensors, Mapping) else None
        alive_count = (
            int(alive.detach().sum().cpu().item())
            if isinstance(alive, torch.Tensor) and alive.dtype == torch.bool
            else None
        )
        decisions = selection.get("decision_count")
        correct = selection.get("correct_count")
        deaths = selection.get("task_death_count")
        upkeep = selection.get("recurrent_upkeep_charge_total")
        decision_count = (
            int(decisions[index].detach().sum().cpu().item())
            if isinstance(decisions, torch.Tensor)
            else None
        )
        correct_count = (
            int(correct[index].detach().sum().cpu().item())
            if isinstance(correct, torch.Tensor)
            else None
        )
        rows.append(
            {
                "cell_id": cell_id,
                "metrics": {
                    "population.alive": alive_count,
                    "policy.task_accuracy": (
                        correct_count / decision_count
                        if decision_count
                        else None
                    ),
                    "policy.recurrent_cost": (
                        float(upkeep[index].detach().sum().cpu().item())
                        if isinstance(upkeep, torch.Tensor)
                        else None
                    ),
                    "outcome.decision_count": decision_count,
                    "outcome.correct_count": correct_count,
                    "outcome.task_death_count": (
                        int(deaths[index].detach().sum().cpu().item())
                        if isinstance(deaths, torch.Tensor)
                        else None
                    ),
                },
            }
        )
    metadata = projections[0].get("metadata")
    terminal = metadata.get("generation") if isinstance(metadata, Mapping) else None
    if type(terminal) is not int:
        raise CampaignProviderError("GMNW projection lacks terminal generation")
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=tuple(prepared_adapter.cell_ids),
        checkpoint_projections=projections,
        compact_result_rows=tuple(rows) if reporting_enabled else (),
        completed_generation_count=completed,
        terminal_generation=terminal,
    )


def gmnw_policy_arms_native_binding_v1() -> NativeMechanicsBindingV1:
    from engine.perf.cuda_architecture_b_direct_resident_gmnw_checkpoint_v1 import (
        continue_direct_resident_gmnw_multicell_checkpoint_v1,
        export_direct_resident_gmnw_multicell_checkpoint_v1,
    )

    return NativeMechanicsBindingV1(
        binding_id=GMNW_POLICY_ARMS_BINDING_ID,
        artifact_abi="architecture_b_gmnw_gru_v1",
        batch_max=65535,
        generation_min=4,
        generation_max=100,
        ticks_per_generation=80,
        policy_cache_refill=False,
        terminal_snapshot_metrics=True,
        runner_builder=_gmnw_runner_builder,
        request_builder=_gmnw_request_adapter,
        preparer=_gmnw_prepare_adapter,
        executor=_gmnw_execute_adapter,
        projector=_gmnw_project_adapter,
        component_checkpoint_exporter=(
            export_direct_resident_gmnw_multicell_checkpoint_v1
        ),
        component_checkpoint_restorer=(
            continue_direct_resident_gmnw_multicell_checkpoint_v1
        ),
        required_artifact_ids=(
            "cuda_extension",
            "extension_manifest",
            "gmnw_qualification_receipt",
            "gmnw_reporting_receipt",
            "gmnw_witness",
            "matched_mlp_artifact",
            "source_qualification_receipt",
            "source_reporting_receipt",
        ),
        abi_bound_artifact_ids=(
            "cuda_extension",
            "extension_manifest",
            "gmnw_qualification_receipt",
            "gmnw_reporting_receipt",
        ),
        required_artifact_exports=(
            "eiel_cuda_architecture_b_execute_direct_resident_gmnw_gru_v1",
        ),
    )


__all__ = [
    "EED_FIXED_MLP_BINDING_ID",
    "NativeMechanicsBindingV1",
    "JUNGLE_FIXED_MLP_BINDING_ID",
    "GMNW_POLICY_ARMS_BINDING_ID",
    "PATCH_STABLE_GRU_BINDING_ID",
    "PATCH_FIXED_MLP_BINDING_ID",
    "WDL_FIXED_MLP_BINDING_ID",
    "WDL_JUNGLE_FIXED_MLP_BINDING_ID",
    "WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID",
    "WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID",
    "WDL_STABLE_GRU_BINDING_ID",
    "patch_fixed_mlp_native_binding_v1",
    "wdl_jungle_fixed_mlp_cpu_native_adapters_v1",
    "jungle_fixed_mlp_native_binding_v1",
    "gmnw_policy_arms_native_binding_v1",
    "eed_fixed_mlp_native_binding_v1",
    "patch_stable_gru_native_binding_v1",
    "wdl_jungle_fixed_mlp_native_binding_v1",
    "wdl_jungle_fixed_mlp_native_binding_candidate_v1",
    "wdl_jungle_eed_fixed_mlp_native_binding_v1",
    "wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1",
    "wdl_stable_gru_native_binding_v1",
    "wdl_fixed_mlp_native_binding_v1",
]
