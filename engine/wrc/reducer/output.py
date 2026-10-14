"""Deterministic, compact WRC reducer output materialization."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_sha256, pretty_json

from .config import REDUCER_VERSION, ReducerConfig, reducer_schema


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_text(path, pretty_json(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        for row in rows
    ]
    _atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    temporary.replace(path)


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _detail_metrics(parameter_result: Mapping[str, Any], config: ReducerConfig) -> list[str]:
    ranked = [
        str(row["metric"])
        for row in parameter_result.get("ranked_metrics", [])
        if isinstance(row, Mapping) and row.get("metric")
    ]
    parameter = parameter_result.get("parameter")
    activation = (
        [str(value) for value in parameter.get("activation_metrics", [])]
        if isinstance(parameter, Mapping)
        else []
    )
    standardized = parameter_result.get("standardized_summary")
    layers = (
        standardized.get("layers")
        if isinstance(standardized, Mapping)
        and isinstance(standardized.get("layers"), Mapping)
        else {}
    )
    layer_metrics: list[str] = []
    for layer in ("activation", "direct", "ecology", "organism", "population", "other"):
        payload = layers.get(layer)
        outcomes = payload.get("top_outcomes") if isinstance(payload, Mapping) else None
        if not isinstance(outcomes, list):
            continue
        for row in outcomes:
            if isinstance(row, Mapping) and row.get("metric"):
                layer_metrics.append(str(row["metric"]))

    output: list[str] = []
    for metric in [*activation, *layer_metrics, *ranked]:
        if metric not in output:
            output.append(metric)
        if len(output) >= config.max_detail_metrics_per_parameter:
            break
    return output


def _compact_world_rows(
    world_results: Sequence[Mapping[str, Any]],
    detail_metrics: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    compact_worlds: list[dict[str, Any]] = []
    selected_points: list[dict[str, Any]] = []
    raw_rows: dict[tuple[str, int | None, str | None], dict[str, Any]] = {}
    for world in world_results:
        metrics = world.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        selected_metrics = {
            metric: metrics[metric]
            for metric in sorted(detail_metrics)
            if metric in metrics
        }
        compact_worlds.append(
            {
                "schema_version": "eiel_wrc_compact_world_summary_v1",
                "source_world_result_sha256": world.get("result_sha256"),
                "identity": world.get("identity"),
                "identity_sha256": world.get("identity_sha256"),
                "observation_count": world.get("observation_count"),
                "first_sequence": world.get("first_sequence"),
                "last_sequence": world.get("last_sequence"),
                "last_tick": world.get("last_tick"),
                "source_kinds": world.get("source_kinds"),
                "metric_count_available": world.get("metric_count"),
                "metric_count_included": len(selected_metrics),
                "metrics": selected_metrics,
            }
        )
        identity = world.get("identity") if isinstance(world.get("identity"), Mapping) else {}
        for metric, summary in selected_metrics.items():
            for point in summary.get("selected_points", []):
                if not isinstance(point, Mapping):
                    continue
                source = point.get("source") if isinstance(point.get("source"), Mapping) else {}
                row = {
                    "parameter_id": identity.get("parameter_id"),
                    "parameter_value": _json_cell(identity.get("parameter_value")),
                    "seed": identity.get("seed"),
                    "world_id": identity.get("world_id"),
                    "metric": metric,
                    "sequence": point.get("sequence"),
                    "tick": point.get("tick"),
                    "value": point.get("value"),
                    "score": point.get("score"),
                    "reasons": _json_cell(point.get("reasons", [])),
                    "source_kind": source.get("kind"),
                    "source_uri": source.get("uri"),
                    "source_row_number": source.get("row_number"),
                    "source_event_id": source.get("event_id"),
                }
                selected_points.append(row)
                key = (
                    str(source.get("uri") or ""),
                    int(source["row_number"]) if source.get("row_number") is not None else None,
                    str(source["event_id"]) if source.get("event_id") is not None else None,
                )
                indexed = raw_rows.setdefault(
                    key,
                    {
                        "source_kind": source.get("kind"),
                        "source_uri": source.get("uri"),
                        "source_row_number": source.get("row_number"),
                        "source_event_id": source.get("event_id"),
                        "world_id": identity.get("world_id"),
                        "parameter_value": _json_cell(identity.get("parameter_value")),
                        "seed": identity.get("seed"),
                        "sequence": point.get("sequence"),
                        "tick": point.get("tick"),
                        "metrics": {},
                    },
                )
                indexed["metrics"][metric] = point.get("reasons", [])
    selected_points.sort(
        key=lambda row: (
            str(row["world_id"]),
            int(row["sequence"] or 0),
            str(row["metric"]),
        )
    )
    raw_row_output = []
    for row in sorted(
        raw_rows.values(),
        key=lambda value: (
            str(value["world_id"]),
            int(value["sequence"] or 0),
            str(value["source_uri"]),
        ),
    ):
        copy = dict(row)
        copy["metrics"] = _json_cell(copy["metrics"])
        raw_row_output.append(copy)
    return compact_worlds, selected_points, raw_row_output


def _effect_csv_rows(parameter_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for effect in parameter_result.get("metric_effects", []):
        if not isinstance(effect, Mapping):
            continue
        row = dict(effect)
        for name in ("domain_tags", "baseline_level", "treatment_level", "paired_seeds"):
            row[name] = _json_cell(row.get(name))
        rows.append(row)
    return rows


def _curve_csv_rows(parameter_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for curve in parameter_result.get("response_curves", []):
        if not isinstance(curve, Mapping):
            continue
        rows.append(
            {
                "metric": curve.get("metric"),
                "metric_role": curve.get("metric_role"),
                "feature": curve.get("feature"),
                "domain": curve.get("domain"),
                "shape": curve.get("shape"),
                "first_effective_level": _json_cell(curve.get("first_effective_level")),
                "strongest_response_level": _json_cell(curve.get("strongest_response_level")),
                "largest_step_level": _json_cell(curve.get("largest_step_level")),
                "threshold_level": _json_cell(curve.get("threshold_level")),
                "saturation_level": _json_cell(curve.get("saturation_level")),
                "collapse_level": _json_cell(curve.get("collapse_level")),
                "relative_range": curve.get("relative_range"),
                "signal_to_noise": curve.get("signal_to_noise"),
                "largest_adjacent_step": curve.get("largest_adjacent_step"),
                "largest_adjacent_step_fraction": curve.get("largest_adjacent_step_fraction"),
                "ordered_levels": _json_cell(curve.get("ordered_levels", [])),
                "ordered_means": _json_cell(curve.get("ordered_means", [])),
                "ordered_standard_deviations": _json_cell(
                    curve.get("ordered_standard_deviations", [])
                ),
                "reason": curve.get("reason"),
            }
        )
    return rows


def _format_scalar(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return _json_cell(value)
    return str(value)


def _parameter_card_markdown(parameter_result: Mapping[str, Any]) -> str:
    parameter = parameter_result.get("parameter") if isinstance(parameter_result.get("parameter"), Mapping) else {}
    coverage = parameter_result.get("coverage") if isinstance(parameter_result.get("coverage"), Mapping) else {}
    standardized = parameter_result.get("standardized_summary") if isinstance(parameter_result.get("standardized_summary"), Mapping) else {}
    headline = standardized.get("headline") if isinstance(standardized.get("headline"), Mapping) else {}
    control = standardized.get("control_transport") if isinstance(standardized.get("control_transport"), Mapping) else {}
    semantics = standardized.get("interpretation_semantics") if isinstance(standardized.get("interpretation_semantics"), Mapping) else {}
    layers = standardized.get("layers") if isinstance(standardized.get("layers"), Mapping) else {}
    dynamics = standardized.get("dynamics") if isinstance(standardized.get("dynamics"), Mapping) else {}
    landmarks = standardized.get("response_landmarks") if isinstance(standardized.get("response_landmarks"), Mapping) else {}

    lines = [
        f"# {parameter.get('id', '')} — `{parameter.get('parameter', '')}`",
        "",
        "## Headline",
        "",
        "| Field | Result |",
        "|---|---|",
        f"| Screening | {_format_scalar(headline.get('screening_classification'))} |",
        f"| Control transport | {_format_scalar(control.get('status'))} |",
        f"| Control-path activation | {_format_scalar(semantics.get('control_path_activation'))} |",
        f"| Mechanism response | {_format_scalar(semantics.get('mechanism_response'))} |",
        f"| Mechanism metric | `{_format_scalar(semantics.get('mechanism_metric'))}` |",
        f"| Declared activation metric response | {_format_scalar(headline.get('activation_status'))} |",
        f"| Sensitivity | {_format_scalar(headline.get('sensitivity'))} |",
        f"| Response shape | {_format_scalar(headline.get('response_shape'))} |",
        f"| Primary consequence domain | {_format_scalar(headline.get('primary_consequence_domain'))} |",
        f"| Top scientific metric | `{_format_scalar(headline.get('top_scientific_metric'))}` |",
        f"| Seed consistency | {_format_scalar(headline.get('seed_consistency'))} |",
        f"| Coverage | {coverage.get('complete_worlds', 0)} / {coverage.get('expected_worlds', 0)} worlds |",
        "",
        "## Standard layers",
        "",
        "| Layer | Status | Top displayed metric | Direction | Effect score | Seed consistency |",
        "|---|---|---|---|---:|---:|",
    ]
    for layer in ("activation", "direct", "ecology", "organism", "population"):
        payload = layers.get(layer) if isinstance(layers.get(layer), Mapping) else {}
        outcomes = payload.get("top_outcomes") if isinstance(payload.get("top_outcomes"), list) else []
        representative_metric = payload.get("top_effect_metric")
        top = next(
            (
                row
                for row in outcomes
                if isinstance(row, Mapping)
                and row.get("metric") == representative_metric
            ),
            outcomes[0] if outcomes and isinstance(outcomes[0], Mapping) else {},
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    layer.title(),
                    _format_scalar(payload.get("status")),
                    f"`{_format_scalar(top.get('metric'))}`",
                    _format_scalar(top.get("direction")),
                    _format_scalar(top.get("effect_score")),
                    _format_scalar(top.get("seed_consistency")),
                )
            )
            + " |"
        )
    lines.extend(["", "## Representative outcomes", ""])
    for layer in ("direct", "ecology", "organism", "population"):
        payload = layers.get(layer) if isinstance(layers.get(layer), Mapping) else {}
        outcomes = payload.get("top_outcomes") if isinstance(payload.get("top_outcomes"), list) else []
        lines.append(f"### {layer.title()}")
        lines.append("")
        if not outcomes:
            lines.append("No ranked metric was available.")
        else:
            for row in outcomes[:3]:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "- `"
                    + _format_scalar(row.get("metric"))
                    + "`: "
                    + _format_scalar(row.get("direction"))
                    + ", effect score "
                    + _format_scalar(row.get("effect_score"))
                    + ", seed consistency "
                    + _format_scalar(row.get("seed_consistency"))
                )
        lines.append("")

    lines.extend(
        [
            "## Dynamics",
            "",
            "| Field | Result |",
            "|---|---|",
            f"| Metric | `{_format_scalar(dynamics.get('metric'))}` |",
            f"| Pattern | {_format_scalar(dynamics.get('pattern'))} |",
            f"| Timing | {_format_scalar(dynamics.get('timing'))} |",
            f"| Timing basis | {_format_scalar(dynamics.get('timing_basis'))} |",
            f"| Treatment level | {_format_scalar(dynamics.get('treatment_level'))} |",
            f"| Treatment onset median tick | {_format_scalar(dynamics.get('treatment_first_significant_tick_median'))} |",
            f"| Horizon tick | {_format_scalar(dynamics.get('horizon_tick'))} |",
            "",
            "## Response landmarks",
            "",
            "| Landmark | Level |",
            "|---|---|",
            f"| First effective | {_format_scalar(landmarks.get('first_effective_level'))} |",
            f"| Strongest tested effect | {_format_scalar(landmarks.get('strongest_effect_level'))} |",
            f"| Largest adjacent step | {_format_scalar(landmarks.get('largest_step_level'))} |",
            f"| Threshold | {_format_scalar(landmarks.get('threshold_level'))} |",
            f"| Saturation | {_format_scalar(landmarks.get('saturation_level'))} |",
            f"| Collapse | {_format_scalar(landmarks.get('collapse_level'))} |",
            f"| Operating-range recommendation | {_format_scalar(landmarks.get('recommendation_status'))} |",
            "",
            "## Control readback",
            "",
            f"Metric: `{_format_scalar(control.get('metric'))}`  ",
            f"Coverage: {_format_scalar(control.get('coverage'))}  ",
            f"Match fraction: {_format_scalar(control.get('match_fraction'))}",
            "",
            "## Interpretation limits",
            "",
            "- This is a screening reduction, not qualification or decision authority.",
            "- Response labels describe only the tested levels and available seeds.",
            "- Raw source locators remain the drill-back authority.",
            "",
        ]
    )
    return "\n".join(lines)


def master_row(
    parameter_result: Mapping[str, Any],
    *,
    summary_path: str,
    card_path: str,
    card_markdown_path: str,
) -> dict[str, Any]:
    parameter = parameter_result.get("parameter") if isinstance(parameter_result.get("parameter"), Mapping) else {}
    coverage = parameter_result.get("coverage") if isinstance(parameter_result.get("coverage"), Mapping) else {}
    screening = parameter_result.get("screening") if isinstance(parameter_result.get("screening"), Mapping) else {}
    standardized = parameter_result.get("standardized_summary") if isinstance(parameter_result.get("standardized_summary"), Mapping) else {}
    layers = standardized.get("layers") if isinstance(standardized.get("layers"), Mapping) else {}
    dynamics = standardized.get("dynamics") if isinstance(standardized.get("dynamics"), Mapping) else {}
    control = standardized.get("control_transport") if isinstance(standardized.get("control_transport"), Mapping) else {}
    semantics = standardized.get("interpretation_semantics") if isinstance(standardized.get("interpretation_semantics"), Mapping) else {}

    def layer_status(name: str) -> Any:
        value = layers.get(name)
        return value.get("status") if isinstance(value, Mapping) else None

    def layer_metric(name: str) -> Any:
        value = layers.get(name)
        if isinstance(value, Mapping) and value.get("top_effect_metric"):
            return value.get("top_effect_metric")
        return None

    return {
        "id": parameter.get("id"),
        "parameter": parameter.get("parameter"),
        "runtime_parameter": parameter.get("runtime_parameter"),
        "family": parameter.get("family"),
        "type": parameter.get("type"),
        "reference_default": _json_cell(parameter.get("reference_default")),
        "levels": _json_cell(parameter.get("levels", [])),
        "worlds_complete": coverage.get("complete_worlds"),
        "worlds_expected": coverage.get("expected_worlds"),
        "coverage_complete": coverage.get("complete"),
        "control_transport_status": control.get("status"),
        "control_path_activation": semantics.get("control_path_activation"),
        "mechanism_response": semantics.get("mechanism_response"),
        "mechanism_metric": semantics.get("mechanism_metric"),
        "activation_status": screening.get("activation_status"),
        "direct_status": layer_status("direct"),
        "ecology_status": layer_status("ecology"),
        "organism_status": layer_status("organism"),
        "population_status": layer_status("population"),
        "sensitivity": screening.get("sensitivity"),
        "response_shape": screening.get("response_shape"),
        "dynamics_pattern": dynamics.get("pattern"),
        "dynamics_timing": dynamics.get("timing"),
        "primary_domain": screening.get("primary_domain"),
        "top_metric": screening.get("top_metric"),
        "top_direct_metric": layer_metric("direct"),
        "top_ecology_metric": layer_metric("ecology"),
        "top_organism_metric": layer_metric("organism"),
        "top_population_metric": layer_metric("population"),
        "top_effect_score": screening.get("top_effect_score"),
        "seed_consistency": screening.get("top_seed_consistency"),
        "downstream_response": screening.get("downstream_response"),
        "screening_classification": screening.get("classification"),
        "parameter_card_path": card_path,
        "parameter_card_markdown_path": card_markdown_path,
        "parameter_summary_path": summary_path,
        "result_sha256": parameter_result.get("result_sha256"),
    }


def write_row_outputs(
    output_dir: Path,
    *,
    selected_row: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    world_results: Sequence[Mapping[str, Any]],
    parameter_result: Mapping[str, Any],
    incomplete_worlds: Sequence[Mapping[str, Any]],
    config: ReducerConfig,
    source_description: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_metrics = set(_detail_metrics(parameter_result, config))
    compact_worlds, selected_points, raw_rows = _compact_world_rows(
        world_results, detail_metrics
    )
    effect_rows = _effect_csv_rows(parameter_result)
    curve_rows = _curve_csv_rows(parameter_result)

    schema_path = output_dir / "reducer_schema.json"
    selected_row_path = output_dir / "selected_atlas_row.json"
    plan_path = output_dir / "launch_plan.json"
    summary_path = output_dir / "parameter_summary.json"
    card_path = output_dir / "parameter_card.json"
    card_markdown_path = output_dir / "parameter_card.md"
    worlds_path = output_dir / "world_summaries.jsonl"
    effects_path = output_dir / "metric_effects.csv"
    curves_path = output_dir / "response_curves.csv"
    points_path = output_dir / "selected_points.csv"
    raw_index_path = output_dir / "raw_row_index.csv"
    incomplete_path = output_dir / "incomplete_worlds.json"

    write_json(schema_path, reducer_schema(config))
    write_json(selected_row_path, dict(selected_row))
    write_json(plan_path, dict(launch_plan))
    compact_parameter = dict(parameter_result)
    compact_parameter.pop("metric_effects", None)
    compact_parameter["metric_effects_artifact"] = effects_path.name
    compact_parameter["metric_effect_count"] = len(effect_rows)
    compact_parameter["parameter_card_artifact"] = card_path.name
    compact_parameter["parameter_card_markdown_artifact"] = card_markdown_path.name
    compact_parameter["world_summaries_artifact"] = worlds_path.name
    compact_parameter["selected_points_artifact"] = points_path.name
    compact_parameter["raw_row_index_artifact"] = raw_index_path.name
    write_json(summary_path, compact_parameter)
    write_json(
        card_path,
        {
            "schema_version": "eiel_wrc_parameter_card_document_v1",
            "reducer_version": REDUCER_VERSION,
            "parameter": parameter_result.get("parameter"),
            "coverage": parameter_result.get("coverage"),
            "summary": parameter_result.get("standardized_summary"),
            "source_parameter_result_sha256": parameter_result.get("result_sha256"),
        },
    )
    _atomic_text(card_markdown_path, _parameter_card_markdown(parameter_result))
    write_jsonl(worlds_path, compact_worlds)
    write_json(incomplete_path, {"worlds": list(incomplete_worlds)})
    write_csv(
        effects_path,
        effect_rows,
        (
            "metric",
            "metric_role",
            "feature",
            "domain",
            "domain_tags",
            "baseline_level",
            "treatment_level",
            "world_coverage",
            "paired_seed_count",
            "paired_seeds",
            "baseline_mean",
            "treatment_mean",
            "mean_delta",
            "median_delta",
            "minimum_delta",
            "maximum_delta",
            "delta_standard_deviation",
            "relative_mean_delta",
            "paired_standardized_effect",
            "normalized_effect",
            "absolute_normalized_effect",
            "positive_seed_count",
            "negative_seed_count",
            "zero_seed_count",
            "seed_direction_consistency",
            "mean_direction",
            "seed_majority_direction",
            "direction",
            "effect_score",
            "scientific_effect_score",
            "scientific_materiality",
            "scientific_materiality_weight",
            "scientific_confidence_weight",
            "scientific_relevance_weight",
        ),
    )
    write_csv(
        curves_path,
        curve_rows,
        (
            "metric",
            "metric_role",
            "feature",
            "domain",
            "shape",
            "first_effective_level",
            "strongest_response_level",
            "largest_step_level",
            "threshold_level",
            "saturation_level",
            "collapse_level",
            "relative_range",
            "signal_to_noise",
            "largest_adjacent_step",
            "largest_adjacent_step_fraction",
            "ordered_levels",
            "ordered_means",
            "ordered_standard_deviations",
            "reason",
        ),
    )
    write_csv(
        points_path,
        selected_points,
        (
            "parameter_id",
            "parameter_value",
            "seed",
            "world_id",
            "metric",
            "sequence",
            "tick",
            "value",
            "score",
            "reasons",
            "source_kind",
            "source_uri",
            "source_row_number",
            "source_event_id",
        ),
    )
    write_csv(
        raw_index_path,
        raw_rows,
        (
            "source_kind",
            "source_uri",
            "source_row_number",
            "source_event_id",
            "world_id",
            "parameter_value",
            "seed",
            "sequence",
            "tick",
            "metrics",
        ),
    )

    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "reduction_manifest.json":
            artifacts.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema_version": "eiel_wrc_row_reduction_manifest_v1",
        "reducer_version": REDUCER_VERSION,
        "config_sha256": config.sha256,
        "parameter_id": parameter_result.get("parameter", {}).get("id"),
        "source": dict(source_description),
        "complete_world_count": len(world_results),
        "incomplete_world_count": len(incomplete_worlds),
        "detail_metrics": sorted(detail_metrics),
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_dir / "reduction_manifest.json", manifest)
    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
        "master_row": master_row(
            parameter_result,
            # Keep reducer artifacts content-addressable and portable.  The
            # campaign writer prefixes these row-local names with the stable
            # ``rows/<parameter-id>/`` location.  Absolute output roots are
            # an operator concern and must not perturb scientific artifacts.
            summary_path=summary_path.name,
            card_path=card_path.name,
            card_markdown_path=card_markdown_path.name,
        ),
    }


MASTER_FIELDS = (
    "id",
    "parameter",
    "runtime_parameter",
    "family",
    "type",
    "reference_default",
    "levels",
    "worlds_complete",
    "worlds_expected",
    "coverage_complete",
    "control_transport_status",
    "control_path_activation",
    "mechanism_response",
    "mechanism_metric",
    "activation_status",
    "direct_status",
    "ecology_status",
    "organism_status",
    "population_status",
    "sensitivity",
    "response_shape",
    "dynamics_pattern",
    "dynamics_timing",
    "primary_domain",
    "top_metric",
    "top_direct_metric",
    "top_ecology_metric",
    "top_organism_metric",
    "top_population_metric",
    "top_effect_score",
    "seed_consistency",
    "downstream_response",
    "screening_classification",
    "parameter_card_path",
    "parameter_card_markdown_path",
    "parameter_summary_path",
    "result_sha256",
)


def _master_markdown(master_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# WRC Master Screening Table",
        "",
        "| ID | Parameter | Screening | Control path | Mechanism | Downstream | Direct | Ecology | Organism | Population | Shape | Top metric |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in master_rows:
        values = (
            row.get("id"),
            f"`{row.get('parameter', '')}`",
            row.get("screening_classification"),
            row.get("control_path_activation"),
            row.get("mechanism_response"),
            row.get("downstream_response"),
            row.get("direct_status"),
            row.get("ecology_status"),
            row.get("organism_status"),
            row.get("population_status"),
            row.get("response_shape"),
            f"`{row.get('top_metric', '')}`",
        )
        lines.append("| " + " | ".join(_format_scalar(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "This is a screening table. Parameter cards and raw-row indexes contain the drill-down evidence.",
            "",
        ]
    )
    return "\n".join(lines)


FAMILY_FIELDS = (
    "family",
    "parameter_count",
    "complete_count",
    "incomplete_count",
    "active_count",
    "weak_count",
    "no_effect_count",
    "unresolved_count",
    "high_sensitivity_count",
    "dominant_shapes",
    "active_parameters",
    "no_effect_parameters",
    "summary",
)


def _family_summary_rows(master_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in master_rows:
        family = str(row.get("family") or "UNSPECIFIED")
        grouped.setdefault(family, []).append(row)
    output: list[dict[str, Any]] = []
    for family in sorted(grouped):
        rows = sorted(grouped[family], key=lambda row: str(row.get("id") or ""))
        classifications = [str(row.get("screening_classification") or "UNRESOLVED") for row in rows]
        active = [row for row in rows if row.get("screening_classification") == "ACTIVE"]
        no_effect = [row for row in rows if row.get("screening_classification") == "NO_EFFECT_DETECTED"]
        shape_counts: dict[str, int] = {}
        for row in rows:
            shape = str(row.get("response_shape") or "UNRESOLVED")
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        dominant = sorted(shape_counts, key=lambda shape: (-shape_counts[shape], shape))[:4]
        active_names = [f"{row.get('id')}:{row.get('parameter')}" for row in active]
        no_effect_names = [f"{row.get('id')}:{row.get('parameter')}" for row in no_effect]
        complete = sum(bool(row.get("coverage_complete")) for row in rows)
        high = sum(row.get("sensitivity") == "HIGH" for row in rows)
        pieces = [
            f"{len(rows)} parameter(s); {complete} complete",
            f"{len(active)} active",
        ]
        if no_effect:
            pieces.append(f"{len(no_effect)} no-effect")
        if high:
            pieces.append(f"{high} high-sensitivity")
        if dominant:
            pieces.append("dominant shapes: " + ", ".join(dominant))
        output.append(
            {
                "family": family,
                "parameter_count": len(rows),
                "complete_count": complete,
                "incomplete_count": len(rows) - complete,
                "active_count": classifications.count("ACTIVE"),
                "weak_count": classifications.count("WEAK"),
                "no_effect_count": classifications.count("NO_EFFECT_DETECTED"),
                "unresolved_count": classifications.count("UNRESOLVED") + classifications.count("INCOMPLETE"),
                "high_sensitivity_count": high,
                "dominant_shapes": _json_cell(dominant),
                "active_parameters": _json_cell(active_names),
                "no_effect_parameters": _json_cell(no_effect_names),
                "summary": "; ".join(pieces) + ".",
            }
        )
    return output


def _family_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# WRC Family Summaries",
        "",
        "| Family | Parameters | Complete | Active | No effect | High sensitivity | Dominant shapes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        shapes = json.loads(str(row.get("dominant_shapes") or "[]"))
        lines.append(
            "| " + " | ".join(
                [
                    str(row.get("family") or ""),
                    str(row.get("parameter_count") or 0),
                    str(row.get("complete_count") or 0),
                    str(row.get("active_count") or 0),
                    str(row.get("no_effect_count") or 0),
                    str(row.get("high_sensitivity_count") or 0),
                    ", ".join(shapes),
                ]
            ) + " |"
        )
    lines.extend(["", "## Compact interpretations", ""] )
    for row in rows:
        lines.append(f"- **{row.get('family')}**: {row.get('summary')}")
    lines.append("")
    return "\n".join(lines)


def write_atlas_outputs(
    output_dir: Path,
    *,
    master_rows: Sequence[Mapping[str, Any]],
    row_results: Sequence[Mapping[str, Any]],
    skipped_rows: Sequence[Mapping[str, Any]],
    failed_rows: Sequence[Mapping[str, Any]],
    config: ReducerConfig,
    source_description: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "wrc_master.csv"
    master_markdown_path = output_dir / "wrc_master.md"
    status_path = output_dir / "atlas_reduction_status.json"
    schema_path = output_dir / "reducer_schema.json"
    family_csv_path = output_dir / "wrc_family_summaries.csv"
    family_json_path = output_dir / "wrc_family_summaries.json"
    family_markdown_path = output_dir / "wrc_family_summaries.md"
    write_csv(master_path, list(master_rows), MASTER_FIELDS)
    _atomic_text(master_markdown_path, _master_markdown(master_rows))
    family_rows = _family_summary_rows(master_rows)
    write_csv(family_csv_path, family_rows, FAMILY_FIELDS)
    write_json(
        family_json_path,
        {
            "schema_version": "eiel_wrc_family_summaries_v1",
            "reducer_version": REDUCER_VERSION,
            "families": family_rows,
        },
    )
    _atomic_text(family_markdown_path, _family_markdown(family_rows))
    write_json(schema_path, reducer_schema(config))
    status = {
        "schema_version": "eiel_wrc_atlas_reduction_status_v1",
        "reducer_version": REDUCER_VERSION,
        "source": dict(source_description),
        "row_result_count": len(row_results),
        "skipped_row_count": len(skipped_rows),
        "failed_row_count": len(failed_rows),
        "rows": list(row_results),
        "skipped_rows": list(skipped_rows),
        "failed_rows": list(failed_rows),
    }
    status["status_sha256"] = canonical_sha256(status)
    write_json(status_path, status)
    artifacts = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "atlas_reduction_manifest.json"
    ]
    manifest = {
        "schema_version": "eiel_wrc_atlas_reduction_manifest_v1",
        "reducer_version": REDUCER_VERSION,
        "config_sha256": config.sha256,
        "source": dict(source_description),
        "master_row_count": len(master_rows),
        "failed_row_count": len(failed_rows),
        "skipped_row_count": len(skipped_rows),
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_dir / "atlas_reduction_manifest.json", manifest)
    return manifest


__all__ = [
    "FAMILY_FIELDS",
    "MASTER_FIELDS",
    "master_row",
    "sha256_file",
    "write_atlas_outputs",
    "write_csv",
    "write_json",
    "write_jsonl",
    "write_row_outputs",
]
