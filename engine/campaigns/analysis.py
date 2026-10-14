"""Provider-neutral typed analysis for compiled campaign plans."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from engine.authoritative_state.admission import require_sha256
from engine.authoritative_state.conversions import (
    checked_finite_float32,
    checked_integer,
    checked_lineage_founder_id,
)
from engine.authoritative_state.projections import validate_projection_envelope
from engine.authoritative_state.schema import load_schema_v1

from .canonical import canonical_bytes
from .contracts import CampaignPlanV1


SCHEMA_VERSION = "eiel_campaign_analysis_v1"
DECISIONS = frozenset({"pass", "fail", "not_measured"})
AUTHORITATIVE_BINDING_KEYS = frozenset(
    {
        "authoritative_schema_hash",
        "binary_schema_hash",
        "checkpoint_schema_hash",
    }
)


class CampaignAnalysisError(ValueError):
    """A compiled plan or cell-metric row is malformed or inconsistent."""


def authoritative_analysis_projection_v1(
    *,
    source_values: Mapping[str, Sequence[Any]],
    source_record_hashes: Mapping[str, str],
    provider_projection: Mapping[str, Any],
    identity_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Project schema-bound reporting rows after validating provider identity."""

    try:
        schema = load_schema_v1()
        if (
            not isinstance(identity_bindings, Mapping)
            or set(identity_bindings) != AUTHORITATIVE_BINDING_KEYS
        ):
            raise CampaignAnalysisError(
                "authoritative analysis identity binding keys differ"
            )
        base = {
            key: require_sha256(
                value, path=f"analysis.identity_bindings.{key}"
            )
            for key, value in identity_bindings.items()
        }
        expected_local = {
            "authoritative_schema_hash": schema.schema_hash,
            "checkpoint_schema_hash": schema.components[
                "checkpoint_schema.json"
            ]["contract_hash"],
        }
        for key, expected in expected_local.items():
            if base[key] != expected:
                raise CampaignAnalysisError(
                    f"authoritative analysis identity mismatch: {key}"
                )

        provider_contract = schema.components["provider_projection_schema.json"]
        chronicle_contract = schema.components["chronicle_schema.json"]
        provider_bindings = {
            **base,
            "chronicle_schema_hash": chronicle_contract["contract_hash"],
        }
        validate_projection_envelope(
            provider_projection,
            contract=provider_contract,
            expected_bindings=provider_bindings,
        )

        source_ids = (
            "reporting.injury_start.v1",
            "reporting.injury_end.v1",
            "identity.lineage_founder_id.v1",
            "reporting.offspring_count.v1",
        )
        if set(source_values) != set(source_ids) or set(
            source_record_hashes
        ) != set(source_ids):
            raise CampaignAnalysisError(
                "authoritative analysis source field IDs differ"
            )
        rows = {
            field_id: list(values)
            if isinstance(values, Sequence)
            and not isinstance(values, (str, bytes, bytearray))
            else None
            for field_id, values in source_values.items()
        }
        if any(values is None for values in rows.values()):
            raise CampaignAnalysisError(
                "authoritative analysis source values must be arrays"
            )
        row_counts = {len(values) for values in rows.values() if values is not None}
        if len(row_counts) != 1 or not row_counts or next(iter(row_counts)) < 1:
            raise CampaignAnalysisError(
                "authoritative analysis source row counts differ"
            )
        hashes = {
            field_id: require_sha256(
                digest, path=f"analysis.source_record_hashes.{field_id}"
            )
            for field_id, digest in source_record_hashes.items()
        }
        starts = rows["reporting.injury_start.v1"]
        ends = rows["reporting.injury_end.v1"]
        lineages = rows["identity.lineage_founder_id.v1"]
        offspring = rows["reporting.offspring_count.v1"]
        assert starts is not None
        assert ends is not None
        assert lineages is not None
        assert offspring is not None
        values = {
            "analysis.injury_delta.v1": [
                checked_finite_float32(
                    checked_finite_float32(
                        end, path=f"reporting.injury_end.v1[{index}]"
                    )
                    - checked_finite_float32(
                        start, path=f"reporting.injury_start.v1[{index}]"
                    ),
                    path=f"analysis.injury_delta.v1[{index}]",
                )
                for index, (start, end) in enumerate(
                    zip(starts, ends, strict=True)
                )
            ],
            "analysis.lineage_id.v1": [
                str(
                    checked_lineage_founder_id(
                        value,
                        path=f"identity.lineage_founder_id.v1[{index}]",
                    )
                )
                for index, value in enumerate(lineages)
            ],
            "analysis.offspring_count.v1": [
                _checked_nonnegative_int64(
                    value, path=f"reporting.offspring_count.v1[{index}]"
                )
                for index, value in enumerate(offspring)
            ],
        }
        analysis_contract = schema.components["analysis_projection_schema.json"]
        provenance = {}
        for row in analysis_contract["fields"]:
            field_id = row["field_id"]
            sources = list(row["source_field_ids"])
            provenance[field_id] = {
                "transform": row["transform"],
                "source_field_ids": sources,
                "source_record_hashes": [hashes[source] for source in sources],
            }
        expected_bindings = {
            **base,
            "provider_projection_hash": provider_contract["contract_hash"],
        }
        envelope = {
            "projection_schema_version": analysis_contract["schema_version"],
            "projection_schema_hash": analysis_contract["contract_hash"],
            "bindings": expected_bindings,
            "output_manifest": [
                dict(row) for row in analysis_contract["fields"]
            ],
            "values": values,
            "provenance": provenance,
        }
        validate_projection_envelope(
            envelope,
            contract=analysis_contract,
            expected_bindings=expected_bindings,
        )
        canonical_bytes(envelope)
        return envelope
    except CampaignAnalysisError:
        raise
    except (TypeError, ValueError) as exc:
        raise CampaignAnalysisError(
            f"authoritative analysis projection rejected: {exc}"
        ) from exc


def _checked_nonnegative_int64(value: Any, *, path: str) -> int:
    """Admit a count only when its persisted representation is signed int64."""

    result = checked_integer(value, bits=64, signed=True, path=path)
    if result < 0:
        raise CampaignAnalysisError(f"{path} must be nonnegative")
    return result


def _extension_decision_labels(extension: Mapping[str, Any]) -> dict[str, str]:
    """Return the frozen decision vocabulary declared by an extension contract."""

    contract = _mapping(
        extension.get("contract"), path="plan.analysis.extension.contract"
    )
    if "decisions" in contract:
        decisions = _mapping(
            contract["decisions"],
            path="plan.analysis.extension.contract.decisions",
        )
        raw_labels = decisions.get("labels")
        labels_path = "plan.analysis.extension.contract.decisions.labels"
    else:
        raw_labels = contract.get("decision_labels")
        labels_path = "plan.analysis.extension.contract.decision_labels"
    labels = _mapping(raw_labels, path=labels_path)
    if not labels or any(
        not key or not isinstance(value, str) or not value
        for key, value in labels.items()
    ):
        raise CampaignAnalysisError(
            f"{labels_path} must declare nonempty decision labels"
        )
    return {str(key): value for key, value in labels.items()}


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CampaignAnalysisError(f"{path} must be an object with string keys")
    return dict(value)


def _plan_payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    return _mapping(plan, path="plan")


def _cell_payload(value: Any, *, path: str) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    cell = _mapping(value, path=path)
    cell_id = cell.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise CampaignAnalysisError(f"{path}.cell_id must be a nonempty string")
    dimensions = _mapping(cell.get("dimensions"), path=f"{path}.dimensions")
    seeds = _mapping(cell.get("seeds"), path=f"{path}.seeds")
    return {"cell_id": cell_id, "dimensions": dimensions, "seeds": seeds}


def _metric_value(value: Any, *, path: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignAnalysisError(f"{path} must be a finite number or null")
    if isinstance(value, float) and not math.isfinite(value):
        raise CampaignAnalysisError(f"{path} must be finite or null")
    return value


def _rows(
    values: Sequence[Mapping[str, Any]], *, known_cells: set[str]
) -> dict[str, dict[str, int | float | None]]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise CampaignAnalysisError("cell_metric_rows must be an array")
    result: dict[str, dict[str, int | float | None]] = {}
    for index, raw in enumerate(values):
        row = _mapping(raw, path=f"cell_metric_rows[{index}]")
        unknown = sorted(set(row) - {"cell_id", "metrics"})
        if unknown:
            raise CampaignAnalysisError(
                f"cell_metric_rows[{index}] contains unsupported fields: {unknown}"
            )
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in known_cells:
            raise CampaignAnalysisError(
                f"cell_metric_rows[{index}].cell_id is not a compiled cell"
            )
        if cell_id in result:
            raise CampaignAnalysisError(f"duplicate metric row for cell {cell_id!r}")
        metrics = _mapping(row.get("metrics"), path=f"cell_metric_rows[{index}].metrics")
        result[cell_id] = {
            name: _metric_value(value, path=f"cell_metric_rows[{index}].metrics.{name}")
            for name, value in sorted(metrics.items())
        }
    return result


def _dimension_value(cell: Mapping[str, Any], key: str) -> Any:
    dimensions = cell["dimensions"]
    seeds = cell["seeds"]
    if key.startswith("dimension."):
        name = key.removeprefix("dimension.")
        source = dimensions
    elif key.startswith("seed."):
        name = key.removeprefix("seed.")
        source = seeds
    elif key in dimensions:
        name = key
        source = dimensions
    elif key in seeds:
        name = key
        source = seeds
    else:
        raise CampaignAnalysisError(
            f"analysis key {key!r} is absent from compiled cell dimensions and seeds"
        )
    if name not in source:
        raise CampaignAnalysisError(f"analysis key {key!r} is absent from compiled cells")
    value = source[name]
    try:
        canonical_bytes(value)
    except ValueError as exc:
        raise CampaignAnalysisError(f"analysis key {key!r} is not canonical JSON data") from exc
    return value


def _key_payload(cell: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: _dimension_value(cell, key) for key in keys}


def _key_token(value: Mapping[str, Any]) -> bytes:
    return canonical_bytes(dict(value))


def _coverage(measured: int, expected: int) -> float:
    return float(measured) / float(expected) if expected else 0.0


def _compare(value: int | float, operator: str, threshold: Any) -> bool:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise CampaignAnalysisError("analysis rule threshold must be numeric")
    if isinstance(threshold, float) and not math.isfinite(threshold):
        raise CampaignAnalysisError("analysis rule threshold must be finite")
    operations = {
        ">": lambda: value > threshold,
        ">=": lambda: value >= threshold,
        "<": lambda: value < threshold,
        "<=": lambda: value <= threshold,
        "==": lambda: value == threshold,
        "!=": lambda: value != threshold,
    }
    try:
        return operations[operator]()
    except KeyError as exc:
        raise CampaignAnalysisError(f"unsupported analysis operator {operator!r}") from exc


def _aggregate(values: list[int | float], aggregate: str) -> int | float:
    if aggregate == "sum":
        return sum(values)
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "count":
        return len(values)
    raise CampaignAnalysisError(f"unsupported analysis aggregate {aggregate!r}")


def _ordinary_rule(
    rule: Mapping[str, Any],
    *,
    cells: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, int | float | None]],
) -> dict[str, Any]:
    metric = str(rule["metric"])
    group_by = list(rule.get("group_by", []))
    grouped: dict[bytes, tuple[dict[str, Any], list[Mapping[str, Any]]]] = {}
    for cell in cells:
        key = _key_payload(cell, group_by)
        token = _key_token(key)
        grouped.setdefault(token, (key, []))[1].append(cell)

    groups: list[dict[str, Any]] = []
    for token in sorted(grouped):
        key, members = grouped[token]
        values = [
            rows[cell["cell_id"]][metric]
            for cell in members
            if cell["cell_id"] in rows
            and metric in rows[cell["cell_id"]]
            and rows[cell["cell_id"]][metric] is not None
        ]
        expected = len(members)
        measured = len(values)
        coverage = _coverage(measured, expected)
        minimum = float(rule.get("minimum_coverage", 1.0))
        if coverage < minimum or not values:
            result = None
            status = "not_measured"
        else:
            result = _aggregate(values, str(rule["aggregate"]))
            status = "pass" if _compare(result, str(rule["operator"]), rule["threshold"]) else "fail"
        groups.append(
            {
                "key": key,
                "expected": expected,
                "measured": measured,
                "coverage": coverage,
                "value": result,
                "status": status,
            }
        )
    statuses = {group["status"] for group in groups}
    status = "fail" if "fail" in statuses else "not_measured" if "not_measured" in statuses else "pass"
    return {"status": status, "groups": groups}


def _paired_rule(
    rule: Mapping[str, Any],
    *,
    cells: Sequence[Mapping[str, Any]],
    rows: Mapping[str, Mapping[str, int | float | None]],
) -> dict[str, Any]:
    metric = str(rule["metric"])
    group_by = list(rule.get("group_by", []))
    pair_by = list(rule.get("pair_by", []))
    if not group_by or not pair_by:
        return {
            "status": "not_measured",
            "groups": [],
            "reason": "paired_delta_requires_nonempty_group_by_and_pair_by",
        }

    arms: dict[bytes, dict[str, Any]] = {}
    pair_cells: dict[bytes, tuple[dict[str, Any], dict[bytes, list[Mapping[str, Any]]]]] = {}
    for cell in cells:
        arm = _key_payload(cell, group_by)
        arm_token = _key_token(arm)
        arms[arm_token] = arm
        pair = _key_payload(cell, pair_by)
        pair_token = _key_token(pair)
        pair_cells.setdefault(pair_token, (pair, {}))[1].setdefault(arm_token, []).append(cell)
    arm_tokens = sorted(arms)
    if len(arm_tokens) != 2:
        return {
            "status": "not_measured",
            "groups": [],
            "reason": "paired_delta_requires_exactly_two_group_arms",
            "observed_arm_count": len(arm_tokens),
        }

    left_token, right_token = arm_tokens
    invalid_pairs: list[dict[str, Any]] = []
    deltas: list[int | float] = []
    expected_pairs = len(pair_cells)
    for pair_token in sorted(pair_cells):
        pair, by_arm = pair_cells[pair_token]
        if set(by_arm) != set(arm_tokens) or any(len(by_arm[token]) != 1 for token in arm_tokens if token in by_arm):
            invalid_pairs.append(pair)
            continue
        left_cell = by_arm[left_token][0]["cell_id"]
        right_cell = by_arm[right_token][0]["cell_id"]
        left = rows.get(left_cell, {}).get(metric)
        right = rows.get(right_cell, {}).get(metric)
        if left is not None and right is not None:
            deltas.append(right - left)

    coverage = _coverage(len(deltas), expected_pairs)
    minimum = float(rule.get("minimum_coverage", 1.0))
    if invalid_pairs or coverage < minimum or not deltas:
        value = None
        status = "not_measured"
    else:
        value = sum(deltas) / len(deltas)
        status = "pass" if _compare(value, str(rule["operator"]), rule["threshold"]) else "fail"
    group = {
        "key": {"left": arms[left_token], "right": arms[right_token]},
        "expected": expected_pairs,
        "measured": len(deltas),
        "coverage": coverage,
        "value": value,
        "status": status,
    }
    result: dict[str, Any] = {"status": status, "groups": [group]}
    if invalid_pairs:
        result["reason"] = "pair_keys_do_not_identify_exactly_one_cell_per_arm"
        result["invalid_pairs"] = invalid_pairs
    return result


def analyze_campaign(
    plan: CampaignPlanV1 | Mapping[str, Any],
    cell_metric_rows: Sequence[Mapping[str, Any]],
    *,
    generation_metric_rows: Sequence[Mapping[str, Any]] | None = None,
    run_root: Path | None = None,
    extension_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate compiled typed rules without depending on a runtime profile."""

    payload = _plan_payload(plan)
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)) or not raw_cells:
        raise CampaignAnalysisError("plan.cells must be a nonempty array")
    cells = [_cell_payload(cell, path=f"plan.cells[{index}]") for index, cell in enumerate(raw_cells)]
    cell_ids = [cell["cell_id"] for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise CampaignAnalysisError("plan.cells contains duplicate cell IDs")
    rows = _rows(cell_metric_rows, known_cells=set(cell_ids))
    analysis = _mapping(payload.get("analysis"), path="plan.analysis")
    required = analysis.get("required_metrics", [])
    optional = analysis.get("optional_metrics", [])
    rules = analysis.get("rules", [])
    if not all(isinstance(value, list) for value in (required, optional, rules)):
        raise CampaignAnalysisError("plan.analysis lists are malformed")

    metric_names = sorted(set(required) | set(optional) | {str(rule.get("metric")) for rule in rules if isinstance(rule, Mapping)})
    metric_coverage: dict[str, Any] = {}
    for metric in metric_names:
        measured = sum(
            1
            for cell_id in cell_ids
            if cell_id in rows and metric in rows[cell_id] and rows[cell_id][metric] is not None
        )
        required_metric = metric in required
        metric_coverage[metric] = {
            "required": required_metric,
            "expected": len(cells),
            "measured": measured,
            "coverage": _coverage(measured, len(cells)),
            "status": "measured" if measured == len(cells) else "not_measured",
        }

    rule_results: list[dict[str, Any]] = []
    for index, raw_rule in enumerate(rules):
        rule = _mapping(raw_rule, path=f"plan.analysis.rules[{index}]")
        base = {
            "id": rule.get("id"),
            "metric": rule.get("metric"),
            "aggregate": rule.get("aggregate"),
            "group_by": list(rule.get("group_by", [])),
            "pair_by": list(rule.get("pair_by", [])),
            "operator": rule.get("operator"),
            "threshold": rule.get("threshold"),
            "minimum_coverage": float(rule.get("minimum_coverage", 1.0)),
        }
        evaluated = (
            _paired_rule(rule, cells=cells, rows=rows)
            if rule.get("aggregate") == "paired_delta"
            else _ordinary_rule(rule, cells=cells, rows=rows)
        )
        rule_results.append({**base, **evaluated})

    rule_statuses = {rule["status"] for rule in rule_results}
    required_missing = any(metric_coverage[str(metric)]["status"] == "not_measured" for metric in required)
    if "fail" in rule_statuses:
        decision = "fail"
    elif required_missing or "not_measured" in rule_statuses:
        decision = "not_measured"
    else:
        decision = "pass"
    result = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": payload.get("campaign_id"),
        "plan_sha256": payload.get("plan_sha256", ""),
        "expected_cells": len(cells),
        "observed_cell_rows": len(rows),
        "metric_coverage": metric_coverage,
        "rules": rule_results,
        "decision": decision,
    }
    extension = analysis.get("extension")
    if extension is not None:
        if not isinstance(extension, Mapping):
            raise CampaignAnalysisError("plan.analysis.extension must be an object")
        extension_id = extension.get("id")
        if not isinstance(extension_id, str) or not extension_id:
            raise CampaignAnalysisError("plan.analysis.extension.id is invalid")
        if generation_metric_rows is None:
            raise CampaignAnalysisError(
                "registered analysis extension requires generation metric rows"
            )
        from .analysis_extensions import (
            CampaignAnalysisExtensionError,
            default_analysis_extension_registry,
        )

        try:
            registered = default_analysis_extension_registry().resolve(extension_id)
            scientific = registered.analyze(
                payload,
                generation_metric_rows,
                run_root,
                dict(extension_context or {}),
            )
        except CampaignAnalysisExtensionError as exc:
            raise CampaignAnalysisError(str(exc)) from exc
        scientific_decision = scientific.get("decision")
        decision_labels = _extension_decision_labels(extension)
        if scientific_decision not in decision_labels:
            raise CampaignAnalysisError(
                "analysis extension returned a decision outside its frozen contract"
            )
        result = {
            **result,
            "generic_decision": decision,
            "decision": scientific_decision,
            "analysis_extension": {
                "id": extension_id,
                "contract_sha256": extension.get("contract_sha256"),
            },
            "scientific_analysis": scientific,
        }
    canonical_bytes(result)
    return result


def render_analysis_markdown(analysis: Mapping[str, Any]) -> str:
    """Render a concise deterministic Markdown summary."""

    value = _mapping(analysis, path="analysis")
    extension = value.get("analysis_extension")
    if isinstance(extension, Mapping):
        extension_id = extension.get("id")
        if not isinstance(extension_id, str) or not extension_id:
            raise CampaignAnalysisError("analysis extension identity is invalid")
        from .analysis_extensions import (
            CampaignAnalysisExtensionError,
            default_analysis_extension_registry,
        )

        try:
            return default_analysis_extension_registry().resolve(
                extension_id
            ).render_markdown(value)
        except CampaignAnalysisExtensionError as exc:
            raise CampaignAnalysisError(str(exc)) from exc
    decision = value.get("decision")
    if decision not in DECISIONS:
        raise CampaignAnalysisError("analysis.decision is invalid")
    lines = [
        "# Campaign Analysis",
        "",
        f"Campaign: `{value.get('campaign_id', '')}`",
        "",
        f"Decision: **{decision}**",
        "",
        f"Coverage: `{value.get('observed_cell_rows', 0)}` / `{value.get('expected_cells', 0)}` cell rows.",
        "",
        "## Rules",
        "",
    ]
    rules = value.get("rules", [])
    if not rules:
        lines.append("- No decision rules were configured.")
    else:
        for rule in rules:
            groups = list(rule.get("groups", []))
            measured = sum(int(group.get("measured", 0)) for group in groups)
            expected = sum(int(group.get("expected", 0)) for group in groups)
            lines.append(
                f"- `{rule.get('id')}`: **{rule.get('status')}** "
                f"(`{rule.get('aggregate')}` of `{rule.get('metric')}`, coverage {measured}/{expected})."
            )
    lines.extend(["", "## Metric availability", ""])
    coverage = value.get("metric_coverage", {})
    if not coverage:
        lines.append("- No metrics were declared.")
    else:
        for metric, row in sorted(coverage.items()):
            label = "required" if row.get("required") else "optional"
            lines.append(
                f"- `{metric}` ({label}): `{row.get('measured', 0)}` / "
                f"`{row.get('expected', 0)}` measured."
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "CampaignAnalysisError",
    "SCHEMA_VERSION",
    "analyze_campaign",
    "authoritative_analysis_projection_v1",
    "render_analysis_markdown",
]
