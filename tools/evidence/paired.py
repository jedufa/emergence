"""Generic paired descriptive comparison engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Iterable, Mapping

from tools.evidence.schema import parse_number


Filter = Mapping[str, Any] | Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    metric_names: tuple[str, ...]
    group_by_dimensions: tuple[str, ...]
    pair_by_dimensions: tuple[str, ...]
    left_filter: Filter
    right_filter: Filter
    delta_label: str


def _matches(row: Mapping[str, Any], flt: Filter) -> bool:
    if callable(flt):
        return bool(flt(row))
    return all(row.get(key) == value for key, value in flt.items())


def _key(row: Mapping[str, Any], dimensions: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(dim) for dim in dimensions)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _direction_consistency(values: list[float]) -> float | None:
    if not values:
        return None
    positives = sum(1 for value in values if value > 0)
    negatives = sum(1 for value in values if value < 0)
    zeros = sum(1 for value in values if value == 0)
    return max(positives, negatives, zeros) / len(values)


def compare_paired(rows: Iterable[Mapping[str, Any]], spec: ComparisonSpec) -> dict[str, Any]:
    source_rows = [dict(row) for row in rows]
    groups = sorted({_key(row, spec.group_by_dimensions) for row in source_rows}, key=lambda item: tuple(str(v) for v in item))
    per_pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for group in groups:
        group_rows = [row for row in source_rows if _key(row, spec.group_by_dimensions) == group]
        left_index = {_key(row, spec.pair_by_dimensions): row for row in group_rows if _matches(row, spec.left_filter)}
        right_index = {_key(row, spec.pair_by_dimensions): row for row in group_rows if _matches(row, spec.right_filter)}
        pair_keys = sorted(set(left_index) | set(right_index), key=lambda item: tuple(str(v) for v in item))
        group_dict = dict(zip(spec.group_by_dimensions, group))
        metric_deltas: dict[str, list[float]] = {metric: [] for metric in spec.metric_names}
        missing_pairs: list[dict[str, Any]] = []

        for pair_key in pair_keys:
            pair_dict = dict(zip(spec.pair_by_dimensions, pair_key))
            left = left_index.get(pair_key)
            right = right_index.get(pair_key)
            row: dict[str, Any] = {
                "comparison": spec.name,
                **group_dict,
                **pair_dict,
                "status": "paired" if left and right else "missing_pair",
                "left_run_id": left.get("run_id", "") if left else "",
                "right_run_id": right.get("run_id", "") if right else "",
            }
            if not left or not right:
                missing_pairs.append({**group_dict, **pair_dict})
            for metric in spec.metric_names:
                left_value = parse_number(left.get(metric)) if left else None
                right_value = parse_number(right.get(metric)) if right else None
                delta = left_value - right_value if left_value is not None and right_value is not None else None
                row[f"left_{metric}"] = left_value
                row[f"right_{metric}"] = right_value
                row[f"{spec.delta_label}_{metric}"] = delta
                if delta is not None:
                    metric_deltas[metric].append(delta)
            per_pair_rows.append(row)

        summary: dict[str, Any] = {
            "comparison": spec.name,
            **group_dict,
            "paired_count": sum(1 for row in per_pair_rows if row.get("comparison") == spec.name and all(row.get(k) == v for k, v in group_dict.items()) and row.get("status") == "paired"),
            "missing_pairs": missing_pairs,
        }
        for metric in spec.metric_names:
            deltas = metric_deltas[metric]
            summary[metric] = {
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "median_delta": median(deltas) if deltas else None,
                "min_delta": min(deltas) if deltas else None,
                "max_delta": max(deltas) if deltas else None,
                "std_delta": _std(deltas) if deltas else None,
                "direction_consistency": _direction_consistency(deltas),
                "paired_seed_coverage": len(deltas),
            }
        summaries.append(summary)

    return {
        "comparison": spec.name,
        "delta_label": spec.delta_label,
        "summary": summaries,
        "per_pair_rows": per_pair_rows,
    }
