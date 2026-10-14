"""Pure, contract-driven analysis for paired inheritance paths.

This module contains no campaign matrix, scientific threshold, inheritance
identity, locus registry, checkpoint schedule, or decision label.  Those are
validated inputs supplied by an immutable campaign analysis contract.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Hashable, Mapping, Sequence

from engine.campaigns.canonical import canonical_sha256


class InheritanceCrossPathAnalysisError(ValueError):
    """The supplied contract or observations are incomplete or inconsistent."""


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InheritanceCrossPathAnalysisError(f"{name} must be an object")
    return value


def _array(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InheritanceCrossPathAnalysisError(f"{name} must be an array")
    return list(value)


def _strings(value: Any, *, name: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in _array(value, name=name))
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise InheritanceCrossPathAnalysisError(
            f"{name} must contain unique nonempty strings"
        )
    return result


def _identities(value: Any, *, name: str) -> tuple[str, ...]:
    raw = _array(value, name=name)
    if (
        not raw
        or any(not isinstance(item, str) or not item for item in raw)
        or len(set(raw)) != len(raw)
    ):
        raise InheritanceCrossPathAnalysisError(
            f"{name} must contain unique nonempty string identities"
        )
    return tuple(raw)


def _integers(value: Any, *, name: str) -> tuple[int, ...]:
    raw = _array(value, name=name)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise InheritanceCrossPathAnalysisError(f"{name} must contain integers")
    result = tuple(raw)
    if not result or len(set(result)) != len(result):
        raise InheritanceCrossPathAnalysisError(
            f"{name} must contain unique integers"
        )
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise InheritanceCrossPathAnalysisError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InheritanceCrossPathAnalysisError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise InheritanceCrossPathAnalysisError(f"{name} must be finite")
    return result


def _at(value: Mapping[str, Any], path: Sequence[str], *, name: str) -> Any:
    current: Any = value
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise InheritanceCrossPathAnalysisError(
                f"{name} lacks {'.'.join(path)}"
            )
        current = current[component]
    return current


def _path(value: Any, *, name: str) -> tuple[str, ...]:
    result = tuple(str(item) for item in _array(value, name=name))
    if not result or any(not item for item in result):
        raise InheritanceCrossPathAnalysisError(
            f"{name} must contain nonempty path components"
        )
    return result


def _paired_values(value: Any, *, name: str) -> tuple[Hashable, ...]:
    raw = _array(value, name=name)
    result: list[Hashable] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise InheritanceCrossPathAnalysisError(
                f"{name} must contain integer or string identities"
            )
        if isinstance(item, str) and not item:
            raise InheritanceCrossPathAnalysisError(
                f"{name} cannot contain an empty identity"
            )
        result.append(item)
    if not result or len(set(result)) != len(result):
        raise InheritanceCrossPathAnalysisError(
            f"{name} must contain unique identities"
        )
    return tuple(result)


@dataclass(frozen=True)
class InheritanceAnalysisContrast:
    """One directional comparison between two declared analysis arms."""

    name: str
    baseline_arm: str
    treatment_arm: str


@dataclass(frozen=True)
class InheritanceCrossPathContract:
    """Validated immutable wrapper around a canonical analysis contract."""

    payload: Mapping[str, Any]
    arms: tuple[str, ...]
    contrasts: tuple[InheritanceAnalysisContrast, ...]
    confirmation_contrasts: tuple[str, ...]
    paths: tuple[str, ...]
    patterns: tuple[str, ...]
    pairs: tuple[Hashable, ...]
    generations: tuple[int, ...]
    milestones: tuple[int, ...]
    loci: tuple[str, ...]
    metric_names: tuple[str, ...]
    gate_names: tuple[str, ...]
    decision_labels: Mapping[str, str]

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any]
    ) -> "InheritanceCrossPathContract":
        contract = copy.deepcopy(dict(_object(raw, name="analysis_contract")))
        identity = _object(contract.get("identity"), name="identity")
        has_arms = "arms" in identity
        has_legacy_arms = (
            "baseline_mode" in identity or "treatment_mode" in identity
        )
        if has_arms and has_legacy_arms:
            raise InheritanceCrossPathAnalysisError(
                "identity cannot mix arms with legacy baseline/treatment modes"
            )
        if has_arms:
            arms = _identities(identity.get("arms"), name="identity.arms")
            raw_contrasts = _array(contract.get("contrasts"), name="contrasts")
        else:
            baseline = identity.get("baseline_mode")
            treatment = identity.get("treatment_mode")
            if (
                not isinstance(baseline, str)
                or not baseline
                or not isinstance(treatment, str)
                or not treatment
                or baseline == treatment
            ):
                raise InheritanceCrossPathAnalysisError(
                    "identity requires arms or distinct legacy baseline/treatment modes"
                )
            arms = (baseline, treatment)
            if "contrasts" in contract:
                raise InheritanceCrossPathAnalysisError(
                    "legacy baseline/treatment identity cannot declare contrasts"
                )
            raw_contrasts = [
                {
                    "name": "treatment_vs_baseline",
                    "baseline_arm": baseline,
                    "treatment_arm": treatment,
                }
            ]
        contrasts: list[InheritanceAnalysisContrast] = []
        contrast_names: set[str] = set()
        contrast_pairs: set[tuple[str, str]] = set()
        for index, raw_contrast in enumerate(raw_contrasts):
            item = _object(raw_contrast, name=f"contrasts[{index}]")
            allowed = {"name", "baseline_arm", "treatment_arm"}
            unexpected = set(item) - allowed
            if unexpected:
                raise InheritanceCrossPathAnalysisError(
                    f"contrasts[{index}] has unsupported fields: "
                    + ", ".join(sorted(str(value) for value in unexpected))
                )
            name = item.get("name")
            baseline_arm = item.get("baseline_arm")
            treatment_arm = item.get("treatment_arm")
            if not isinstance(name, str) or not name:
                raise InheritanceCrossPathAnalysisError(
                    f"contrasts[{index}].name must be nonempty"
                )
            if not isinstance(baseline_arm, str) or not isinstance(
                treatment_arm, str
            ):
                raise InheritanceCrossPathAnalysisError(
                    f"contrast {name!r} arm identities must be strings"
                )
            if name in contrast_names:
                raise InheritanceCrossPathAnalysisError(
                    f"contrast name is duplicated: {name!r}"
                )
            if (
                baseline_arm not in arms
                or treatment_arm not in arms
                or baseline_arm == treatment_arm
            ):
                raise InheritanceCrossPathAnalysisError(
                    f"contrast {name!r} must reference two distinct declared arms"
                )
            pair = (baseline_arm, treatment_arm)
            if pair in contrast_pairs:
                raise InheritanceCrossPathAnalysisError(
                    f"contrast arm pair is duplicated: {pair!r}"
                )
            contrast_names.add(name)
            contrast_pairs.add(pair)
            contrasts.append(
                InheritanceAnalysisContrast(
                    name=name,
                    baseline_arm=baseline_arm,
                    treatment_arm=treatment_arm,
                )
            )
        if not contrasts:
            raise InheritanceCrossPathAnalysisError("contrasts cannot be empty")
        paths = _strings(identity.get("paths"), name="identity.paths")
        patterns = _strings(identity.get("patterns"), name="identity.patterns")
        pairs = _paired_values(
            identity.get("paired_values"), name="identity.paired_values"
        )
        for name in ("mode", "pattern", "path", "paired", "generation"):
            path_key = f"{name}_path"
            field_key = f"{name}_field"
            if path_key in identity and field_key in identity:
                raise InheritanceCrossPathAnalysisError(
                    f"identity cannot declare both {path_key} and {field_key}"
                )
            if path_key in identity:
                _path(identity[path_key], name=f"identity.{path_key}")
            elif field_key in identity:
                field = identity[field_key]
                if not isinstance(field, str) or not field:
                    raise InheritanceCrossPathAnalysisError(
                        f"identity.{field_key} must be a nonempty string"
                    )
        generations = _integers(contract.get("generations"), name="generations")
        if tuple(sorted(generations)) != generations:
            raise InheritanceCrossPathAnalysisError("generations must be increasing")
        milestones = _integers(contract.get("milestones"), name="milestones")
        if any(item not in generations for item in milestones):
            raise InheritanceCrossPathAnalysisError(
                "milestones must be members of generations"
            )
        loci = _strings(contract.get("loci"), name="loci")

        pattern_contracts = _object(contract.get("pattern_contracts"), name="pattern_contracts")
        if set(pattern_contracts) != set(patterns):
            raise InheritanceCrossPathAnalysisError(
                "pattern_contracts must exactly match identity.patterns"
            )
        metric_sources = _object(contract.get("metric_sources"), name="metric_sources")
        metric_names = tuple(sorted(str(name) for name in metric_sources))
        if not metric_names:
            raise InheritanceCrossPathAnalysisError("metric_sources cannot be empty")
        for name, source in metric_sources.items():
            source_object = _object(source, name=f"metric_sources.{name}")
            _path(source_object.get("path"), name=f"metric_sources.{name}.path")
        for pattern, details in pattern_contracts.items():
            item = _object(details, name=f"pattern_contracts.{pattern}")
            primary = str(item.get("primary_locus") or "")
            if primary not in loci:
                raise InheritanceCrossPathAnalysisError(
                    f"pattern {pattern!r} primary_locus is not registered"
                )
            metrics = _strings(item.get("benefit_metrics"), name=f"{pattern}.benefit_metrics")
            if any(metric not in metric_sources for metric in metrics):
                raise InheritanceCrossPathAnalysisError(
                    f"pattern {pattern!r} references an unknown benefit metric"
                )
            homogenization_loci = _strings(
                item.get("homogenization_loci"),
                name=f"{pattern}.homogenization_loci",
            )
            if any(locus not in loci for locus in homogenization_loci):
                raise InheritanceCrossPathAnalysisError(
                    f"pattern {pattern!r} has an unknown homogenization locus"
                )

        homogenization = _object(contract.get("homogenization"), name="homogenization")
        threshold = _finite(
            homogenization.get("threshold_ratio_exclusive"),
            name="homogenization.threshold_ratio_exclusive",
        )
        consecutive = homogenization.get("consecutive_generations")
        if not 0.0 <= threshold <= 1.0:
            raise InheritanceCrossPathAnalysisError(
                "homogenization threshold must be between zero and one"
            )
        if isinstance(consecutive, bool) or not isinstance(consecutive, int) or consecutive < 1:
            raise InheritanceCrossPathAnalysisError(
                "homogenization consecutive_generations must be positive"
            )
        variance_path = _path(
            homogenization.get("variance_path"), name="homogenization.variance_path"
        )
        if "$locus" not in variance_path:
            raise InheritanceCrossPathAnalysisError(
                "homogenization.variance_path must contain $locus"
            )

        gate_names = _strings(contract.get("required_gates"), name="required_gates")
        row_gates = _array(contract.get("row_gates", []), name="row_gates")
        for index, gate in enumerate(row_gates):
            item = _object(gate, name=f"row_gates[{index}]")
            if str(item.get("operator") or "") not in {"equals", "is_true", "positive"}:
                raise InheritanceCrossPathAnalysisError(
                    f"row_gates[{index}] has an unsupported operator"
                )
            _path(item.get("path"), name=f"row_gates[{index}].path")

        benefit = _object(contract.get("benefit"), name="benefit")
        if benefit.get("strict_positive_delta") is not True:
            raise InheritanceCrossPathAnalysisError(
                "benefit.strict_positive_delta must be explicit"
            )
        if benefit.get("treatment_value_positive") is not True:
            raise InheritanceCrossPathAnalysisError(
                "benefit.treatment_value_positive must be explicit"
            )
        if benefit.get("milestones_required") is not True:
            raise InheritanceCrossPathAnalysisError(
                "benefit.milestones_required must be explicit"
            )
        if not isinstance(benefit.get("homogenization_later_required"), bool):
            raise InheritanceCrossPathAnalysisError(
                "benefit.homogenization_later_required must be boolean"
            )
        if not isinstance(benefit.get("path_reversal_invalidates"), bool):
            raise InheritanceCrossPathAnalysisError(
                "benefit.path_reversal_invalidates must be boolean"
            )
        if benefit.get("terminal_generation") not in generations:
            raise InheritanceCrossPathAnalysisError(
                "benefit.terminal_generation must be a declared generation"
            )
        confirmation_contrasts = (
            _strings(
                benefit.get("confirmation_contrasts"),
                name="benefit.confirmation_contrasts",
            )
            if "confirmation_contrasts" in benefit
            else tuple(contrast.name for contrast in contrasts)
        )
        if any(name not in contrast_names for name in confirmation_contrasts):
            raise InheritanceCrossPathAnalysisError(
                "benefit.confirmation_contrasts references an unknown contrast"
            )

        locking = contract.get("locking")
        if locking is not None:
            locking_object = _object(locking, name="locking")
            locking_contrast = locking_object.get("contrast")
            if locking_contrast is None and len(contrasts) != 1:
                raise InheritanceCrossPathAnalysisError(
                    "locking.contrast is required when multiple contrasts exist"
                )
            if (
                locking_contrast is not None
                and str(locking_contrast) not in contrast_names
            ):
                raise InheritanceCrossPathAnalysisError(
                    "locking.contrast references an unknown contrast"
                )

        decisions = _object(contract.get("decisions"), name="decisions")
        rules = _array(decisions.get("rules"), name="decisions.rules")
        labels = dict(_object(decisions.get("labels"), name="decisions.labels"))
        if not rules or not labels:
            raise InheritanceCrossPathAnalysisError(
                "decisions requires ordered rules and labels"
            )
        for index, rule in enumerate(rules):
            item = _object(rule, name=f"decisions.rules[{index}]")
            decision = str(item.get("decision") or "")
            when = str(item.get("when") or "")
            if decision not in labels:
                raise InheritanceCrossPathAnalysisError(
                    f"decision rule {index} references an unknown decision"
                )
            if when not in {
                "integrity_failure",
                "all_paths_confirmed_and_locking",
                "all_paths_confirmed",
                "confirmed_paths_exact",
                "otherwise",
            }:
                raise InheritanceCrossPathAnalysisError(
                    f"decision rule {index} has an unsupported predicate"
                )
            if when == "confirmed_paths_exact":
                selected = _strings(item.get("paths"), name=f"decisions.rules[{index}].paths")
                if any(path not in paths for path in selected):
                    raise InheritanceCrossPathAnalysisError(
                        f"decision rule {index} references an unknown path"
                    )
        return cls(
            payload=contract,
            arms=arms,
            contrasts=tuple(contrasts),
            confirmation_contrasts=confirmation_contrasts,
            paths=paths,
            patterns=patterns,
            pairs=pairs,
            generations=generations,
            milestones=milestones,
            loci=loci,
            metric_names=metric_names,
            gate_names=gate_names,
            decision_labels={str(key): str(value) for key, value in labels.items()},
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload)

    @property
    def modes(self) -> tuple[str, ...]:
        """Compatibility alias for row identities historically named modes."""

        return self.arms

    @property
    def baseline_mode(self) -> str:
        """Compatibility alias for single-contrast consumers."""

        return self.contrasts[0].baseline_arm

    @property
    def treatment_mode(self) -> str:
        """Compatibility alias for single-contrast consumers."""

        return self.contrasts[0].treatment_arm

    def contrast(self, name: str) -> InheritanceAnalysisContrast:
        for contrast in self.contrasts:
            if contrast.name == name:
                return contrast
        raise InheritanceCrossPathAnalysisError(f"unknown contrast: {name!r}")

    @property
    def expected_cell_count(self) -> int:
        return (
            len(self.modes)
            * len(self.paths)
            * len(self.patterns)
            * len(self.pairs)
        )

    @property
    def expected_row_count(self) -> int:
        return self.expected_cell_count * len(self.generations)


Identity = tuple[str, str, str, Hashable]
Index = dict[Identity, dict[int, Mapping[str, Any]]]


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: InheritanceCrossPathContract,
) -> tuple[Index, list[str]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise InheritanceCrossPathAnalysisError("generation_rows must be an array")
    indexed: Index = {}
    reasons: set[str] = set()
    identity_fields = _object(contract.payload["identity"], name="identity")

    def identity_path(name: str, default: str) -> tuple[str, ...]:
        path_key = f"{name}_path"
        if path_key in identity_fields:
            return _path(identity_fields[path_key], name=f"identity.{path_key}")
        return (str(identity_fields.get(f"{name}_field", default)),)

    mode_path = identity_path("mode", "mode")
    pattern_path = identity_path("pattern", "pattern")
    reproduction_path = identity_path("path", "reproduction_path")
    pair_path = identity_path("paired", "paired_value")
    generation_path = identity_path("generation", "generation")
    for ordinal, raw in enumerate(rows):
        row = _object(raw, name=f"generation_rows[{ordinal}]")
        try:
            mode = str(_at(row, mode_path, name=f"generation_rows[{ordinal}]") or "")
            pattern = str(
                _at(row, pattern_path, name=f"generation_rows[{ordinal}]") or ""
            )
            path = str(
                _at(row, reproduction_path, name=f"generation_rows[{ordinal}]") or ""
            )
            pair = _at(row, pair_path, name=f"generation_rows[{ordinal}]")
            generation = _at(
                row, generation_path, name=f"generation_rows[{ordinal}]"
            )
        except InheritanceCrossPathAnalysisError:
            reasons.add(f"row_{ordinal}:unexpected_identity")
            continue
        if (
            mode not in contract.modes
            or pattern not in contract.patterns
            or path not in contract.paths
            or pair not in contract.pairs
            or generation not in contract.generations
        ):
            reasons.add(f"row_{ordinal}:unexpected_identity")
            continue
        identity = (mode, pattern, path, pair)
        generations = indexed.setdefault(identity, {})
        if generation in generations:
            reasons.add(
                "duplicate_generation_row:"
                f"{mode}/{pattern}/{path}/{pair}/{generation}"
            )
        else:
            generations[int(generation)] = row
    expected = {
        (mode, pattern, path, pair)
        for mode in contract.modes
        for pattern in contract.patterns
        for path in contract.paths
        for pair in contract.pairs
    }
    if set(indexed) != expected:
        reasons.add(f"missing_cell_identities:{len(expected - set(indexed))}")
        if set(indexed) - expected:
            reasons.add(f"unexpected_cell_identities:{len(set(indexed) - expected)}")
    for identity, generations in indexed.items():
        if set(generations) != set(contract.generations):
            reasons.add(
                "generation_coverage_incomplete:"
                + "/".join((*identity[:3], str(identity[3])))
            )
    if len(rows) != contract.expected_row_count:
        reasons.add(
            f"generation_row_count:{len(rows)}:expected:{contract.expected_row_count}"
        )
    return indexed, sorted(reasons)


def _row_gate_reasons(
    indexed: Index,
    contract: InheritanceCrossPathContract,
) -> list[str]:
    reasons: set[str] = set()
    row_gates = _array(contract.payload.get("row_gates", []), name="row_gates")
    initial_generation = contract.generations[0]
    for identity, generations in indexed.items():
        mode, pattern, path, pair = identity
        for generation, row in generations.items():
            tag = f"{mode}/{pattern}/{path}/{pair}/{generation}"
            for gate_index, raw_gate in enumerate(row_gates):
                gate = _object(raw_gate, name=f"row_gates[{gate_index}]")
                modes = tuple(str(item) for item in gate.get("modes", contract.modes))
                paths = tuple(str(item) for item in gate.get("paths", contract.paths))
                patterns = tuple(
                    str(item) for item in gate.get("patterns", contract.patterns)
                )
                scope = str(gate.get("generation_scope", "all"))
                if (
                    mode not in modes
                    or path not in paths
                    or pattern not in patterns
                    or (scope == "initial" and generation != initial_generation)
                    or (scope == "non_initial" and generation == initial_generation)
                ):
                    continue
                name = str(gate.get("name") or f"row_gate_{gate_index}")
                try:
                    value = _at(
                        row,
                        _path(gate.get("path"), name=f"{name}.path"),
                        name=tag,
                    )
                    operator = str(gate.get("operator"))
                    passed = (
                        value == gate.get("expected")
                        if operator == "equals"
                        else value is True
                        if operator == "is_true"
                        else _finite(value, name=f"{tag}.{name}") > 0.0
                    )
                except (InheritanceCrossPathAnalysisError, TypeError, ValueError):
                    passed = False
                if not passed:
                    reasons.add(f"{tag}:row_gate_failed:{name}")
    return sorted(reasons)


def _resolve_metric_path(
    path: Sequence[str],
    *,
    primary_locus: str,
) -> tuple[str, ...]:
    return tuple(primary_locus if item == "$primary_locus" else item for item in path)


def _metric(
    row: Mapping[str, Any],
    *,
    contract: InheritanceCrossPathContract,
    pattern: str,
    metric: str,
) -> float:
    source = _object(
        _object(contract.payload["metric_sources"], name="metric_sources").get(metric),
        name=f"metric_sources.{metric}",
    )
    pattern_contract = _object(
        _object(contract.payload["pattern_contracts"], name="pattern_contracts").get(
            pattern
        ),
        name=f"pattern_contracts.{pattern}",
    )
    path = _resolve_metric_path(
        _path(source.get("path"), name=f"metric_sources.{metric}.path"),
        primary_locus=str(pattern_contract["primary_locus"]),
    )
    return _finite(_at(row, path, name=metric), name=metric)


def _paired_summaries(
    indexed: Index,
    contract: InheritanceCrossPathContract,
) -> list[dict[str, Any]]:
    pattern_contracts = _object(
        contract.payload["pattern_contracts"], name="pattern_contracts"
    )
    summaries: list[dict[str, Any]] = []
    for contrast in contract.contrasts:
        for path in contract.paths:
            for pattern in contract.patterns:
                metrics = _strings(
                    _object(pattern_contracts[pattern], name=pattern).get(
                        "benefit_metrics"
                    ),
                    name=f"{pattern}.benefit_metrics",
                )
                for metric in metrics:
                    for generation in contract.milestones:
                        pairs = []
                        for pair in contract.pairs:
                            baseline = _metric(
                                indexed[
                                    (
                                        contrast.baseline_arm,
                                        pattern,
                                        path,
                                        pair,
                                    )
                                ][generation],
                                contract=contract,
                                pattern=pattern,
                                metric=metric,
                            )
                            treatment = _metric(
                                indexed[
                                    (
                                        contrast.treatment_arm,
                                        pattern,
                                        path,
                                        pair,
                                    )
                                ][generation],
                                contract=contract,
                                pattern=pattern,
                                metric=metric,
                            )
                            pairs.append(
                                {
                                    "paired_value": pair,
                                    "baseline": baseline,
                                    "treatment": treatment,
                                    "delta_treatment_minus_baseline": (
                                        treatment - baseline
                                    ),
                                }
                            )
                        deltas = [
                            item["delta_treatment_minus_baseline"] for item in pairs
                        ]
                        summaries.append(
                            {
                                "contrast": contrast.name,
                                "baseline_arm": contrast.baseline_arm,
                                "treatment_arm": contrast.treatment_arm,
                                "path": path,
                                "pattern": pattern,
                                "metric": metric,
                                "generation": generation,
                                "paired_count": len(pairs),
                                "delta_mean": float(sum(deltas) / len(deltas)),
                                "delta_median": float(median(deltas)),
                                "delta_minimum": float(min(deltas)),
                                "delta_maximum": float(max(deltas)),
                                "positive_count": sum(value > 0.0 for value in deltas),
                                "strictly_positive_all_pairs": all(
                                    value > 0.0 for value in deltas
                                ),
                                "pairs": pairs,
                            }
                        )
    return summaries


def _time_to_homogenization(
    values: Sequence[tuple[int, float]],
    *,
    threshold: float,
    consecutive: int,
) -> dict[str, Any]:
    initial = values[0][1]
    if initial <= 0.0:
        return {
            "status": "not_evaluable",
            "first_generation": None,
            "confirmation_generation": None,
            "initial_variance": initial,
        }
    streak: list[int] = []
    for generation, variance in values[1:]:
        below = variance / initial < threshold
        if below and (not streak or generation == streak[-1] + 1):
            streak.append(generation)
        elif below:
            streak = [generation]
        else:
            streak = []
        if len(streak) >= consecutive:
            return {
                "status": "reached",
                "first_generation": streak[0],
                "confirmation_generation": streak[consecutive - 1],
                "initial_variance": initial,
            }
    return {
        "status": "not_reached",
        "first_generation": None,
        "confirmation_generation": None,
        "initial_variance": initial,
    }


def _homogenization(
    indexed: Index,
    contract: InheritanceCrossPathContract,
) -> list[dict[str, Any]]:
    settings = _object(contract.payload["homogenization"], name="homogenization")
    threshold = _finite(
        settings["threshold_ratio_exclusive"], name="homogenization.threshold"
    )
    consecutive = int(settings["consecutive_generations"])
    variance_template = _path(
        settings["variance_path"], name="homogenization.variance_path"
    )
    pattern_contracts = _object(
        contract.payload["pattern_contracts"], name="pattern_contracts"
    )
    result = []
    for (mode, pattern, path, pair), generations in sorted(
        indexed.items(), key=lambda item: repr(item[0])
    ):
        loci = _strings(
            _object(pattern_contracts[pattern], name=pattern).get(
                "homogenization_loci"
            ),
            name=f"{pattern}.homogenization_loci",
        )
        for locus in loci:
            metric_path = tuple(
                locus if item == "$locus" else item for item in variance_template
            )
            series = [
                (
                    generation,
                    _finite(
                        _at(generations[generation], metric_path, name="variance"),
                        name="variance",
                    ),
                )
                for generation in contract.generations
            ]
            result.append(
                {
                    "mode": mode,
                    "pattern": pattern,
                    "path": path,
                    "paired_value": pair,
                    "locus": locus,
                    **_time_to_homogenization(
                        series, threshold=threshold, consecutive=consecutive
                    ),
                }
            )
    return result


def _path_reversals(
    paired: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str, Hashable], list[tuple[int, float]]
    ] = defaultdict(
        list
    )
    for summary in paired:
        for pair in summary["pairs"]:
            grouped[
                (
                    str(summary["contrast"]),
                    str(summary["path"]),
                    str(summary["pattern"]),
                    str(summary["metric"]),
                    pair["paired_value"],
                )
            ].append(
                (
                    int(summary["generation"]),
                    float(pair["delta_treatment_minus_baseline"]),
                )
            )
    return [
        {
            "contrast": key[0],
            "path": key[1],
            "pattern": key[2],
            "metric": key[3],
            "paired_value": key[4],
            "milestone_deltas": [
                {"generation": generation, "delta": delta}
                for generation, delta in sorted(values)
            ],
        }
        for key, values in sorted(grouped.items(), key=lambda item: repr(item[0]))
        if any(delta > 0.0 for _generation, delta in values)
        and any(delta < 0.0 for _generation, delta in values)
    ]


def _path_confirmations(
    indexed: Index,
    paired: Sequence[Mapping[str, Any]],
    homogeneous: Sequence[Mapping[str, Any]],
    contract: InheritanceCrossPathContract,
) -> dict[str, dict[str, Any]]:
    pattern_contracts = _object(
        contract.payload["pattern_contracts"], name="pattern_contracts"
    )
    summary_index = {
        (
            row["contrast"],
            row["path"],
            row["pattern"],
            row["metric"],
            row["generation"],
        ): row
        for row in paired
    }
    homogeneous_index = {
        (
            row["mode"],
            row["pattern"],
            row["path"],
            row["paired_value"],
            row["locus"],
        ): row
        for row in homogeneous
    }
    benefit = _object(contract.payload["benefit"], name="benefit")
    terminal_generation = benefit["terminal_generation"]
    result = {}
    for path in contract.paths:
        later_required = benefit["homogenization_later_required"] is True
        contrast_results: dict[str, dict[str, Any]] = {}
        for contrast in contract.contrasts:
            milestone_positive = all(
                summary_index[
                    (contrast.name, path, pattern, metric, generation)
                ]["strictly_positive_all_pairs"]
                for pattern in contract.patterns
                for metric in _strings(
                    _object(pattern_contracts[pattern], name=pattern).get(
                        "benefit_metrics"
                    ),
                    name=f"{pattern}.benefit_metrics",
                )
                for generation in contract.milestones
            )
            treatment_positive = all(
                _metric(
                    indexed[
                        (
                            contrast.treatment_arm,
                            pattern,
                            path,
                            pair,
                        )
                    ][terminal_generation],
                    contract=contract,
                    pattern=pattern,
                    metric=metric,
                )
                > 0.0
                for pattern in contract.patterns
                for metric in _strings(
                    _object(pattern_contracts[pattern], name=pattern).get(
                        "benefit_metrics"
                    ),
                    name=f"{pattern}.benefit_metrics",
                )
                for pair in contract.pairs
            )
            later = True
            for pattern in contract.patterns:
                loci = _strings(
                    _object(pattern_contracts[pattern], name=pattern).get(
                        "homogenization_loci"
                    ),
                    name=f"{pattern}.homogenization_loci",
                )
                for pair in contract.pairs:
                    for locus in loci:
                        baseline = homogeneous_index[
                            (
                                contrast.baseline_arm,
                                pattern,
                                path,
                                pair,
                                locus,
                            )
                        ]
                        treatment = homogeneous_index[
                            (
                                contrast.treatment_arm,
                                pattern,
                                path,
                                pair,
                                locus,
                            )
                        ]
                        pair_later = treatment["status"] != "reached" or (
                            baseline["status"] == "reached"
                            and treatment["first_generation"]
                            > baseline["first_generation"]
                        )
                        later = later and pair_later
            contrast_results[contrast.name] = {
                "confirmed": bool(
                    milestone_positive
                    and treatment_positive
                    and (later or not later_required)
                ),
                "baseline_arm": contrast.baseline_arm,
                "treatment_arm": contrast.treatment_arm,
                "milestone_benefits_strictly_positive": milestone_positive,
                "terminal_treatment_values_positive": treatment_positive,
                "treatment_homogenizes_later_or_not": later,
                "homogenization_later_required": later_required,
            }
        required = tuple(contract.confirmation_contrasts)
        milestone_positive = all(
            contrast_results[name]["milestone_benefits_strictly_positive"]
            for name in required
        )
        treatment_positive = all(
            contrast_results[name]["terminal_treatment_values_positive"]
            for name in required
        )
        later = all(
            contrast_results[name]["treatment_homogenizes_later_or_not"]
            for name in required
        )
        result[path] = {
            "confirmed": all(contrast_results[name]["confirmed"] for name in required),
            "milestone_benefits_strictly_positive": milestone_positive,
            "terminal_treatment_values_positive": treatment_positive,
            "treatment_homogenizes_later_or_not": later,
            "homogenization_later_required": later_required,
            "required_contrasts": list(required),
            "contrasts": contrast_results,
        }
    return result


def _cross_path_consistency(
    paired: Sequence[Mapping[str, Any]],
    contract: InheritanceCrossPathContract,
) -> dict[str, Any]:
    grouped: dict[
        tuple[str, str, str, int, Hashable], dict[str, float]
    ] = defaultdict(dict)
    for summary in paired:
        for pair in summary["pairs"]:
            grouped[
                (
                    str(summary["contrast"]),
                    str(summary["pattern"]),
                    str(summary["metric"]),
                    int(summary["generation"]),
                    pair["paired_value"],
                )
            ][str(summary["path"])] = float(
                pair["delta_treatment_minus_baseline"]
            )
    mismatches = [
        {
            "contrast": key[0],
            "pattern": key[1],
            "metric": key[2],
            "generation": key[3],
            "paired_value": key[4],
            "path_deltas": dict(sorted(values.items())),
        }
        for key, values in sorted(grouped.items(), key=lambda item: repr(item[0]))
        if set(values) != set(contract.paths)
        or any(value <= 0.0 for value in values.values())
    ]
    return {"consistent_positive": not mismatches, "mismatches": mismatches}


def _locking(
    indexed: Index,
    contract: InheritanceCrossPathContract,
) -> dict[str, Any]:
    raw = contract.payload.get("locking")
    if raw is None:
        return {"enabled": False, "proven": False, "paths": [], "details": []}
    settings = _object(raw, name="locking")
    contrast = contract.contrast(
        str(settings.get("contrast") or contract.contrasts[0].name)
    )
    pattern = str(settings.get("pattern") or "")
    generation = int(settings.get("generation"))
    if pattern not in contract.patterns or generation not in contract.generations:
        raise InheritanceCrossPathAnalysisError("locking identity is invalid")
    effective_path = _path(settings.get("effective_path"), name="locking.effective_path")
    concentration_path = _path(
        settings.get("concentration_path"), name="locking.concentration_path"
    )
    retention_metric = str(settings.get("retention_metric") or "")
    effective_multiplier = _finite(
        settings.get("effective_treatment_lte_baseline_multiplier"),
        name="locking.effective_multiplier",
    )
    concentration_multiplier = _finite(
        settings.get("concentration_treatment_gte_baseline_multiplier"),
        name="locking.concentration_multiplier",
    )
    retention_minimum = _finite(
        settings.get("treatment_retention_gte"), name="locking.retention_minimum"
    )
    proven_paths = []
    details = []
    for path in contract.paths:
        pairs = []
        all_locked = True
        for pair in contract.pairs:
            baseline = indexed[
                (contrast.baseline_arm, pattern, path, pair)
            ][generation]
            treatment = indexed[
                (contrast.treatment_arm, pattern, path, pair)
            ][generation]
            baseline_effective = _finite(
                _at(baseline, effective_path, name="locking.baseline_effective"),
                name="locking.baseline_effective",
            )
            treatment_effective = _finite(
                _at(treatment, effective_path, name="locking.treatment_effective"),
                name="locking.treatment_effective",
            )
            baseline_concentration = _finite(
                _at(
                    baseline,
                    concentration_path,
                    name="locking.baseline_concentration",
                ),
                name="locking.baseline_concentration",
            )
            treatment_concentration = _finite(
                _at(
                    treatment,
                    concentration_path,
                    name="locking.treatment_concentration",
                ),
                name="locking.treatment_concentration",
            )
            retention = _metric(
                treatment,
                contract=contract,
                pattern=pattern,
                metric=retention_metric,
            )
            locked = (
                treatment_effective <= effective_multiplier * baseline_effective
                and treatment_concentration
                >= concentration_multiplier * baseline_concentration
                and retention >= retention_minimum
            )
            all_locked = all_locked and locked
            pairs.append({"paired_value": pair, "locked": locked})
        if all_locked:
            proven_paths.append(path)
        details.append({"path": path, "pairs": pairs})
    return {
        "enabled": True,
        "contrast": contrast.name,
        "baseline_arm": contrast.baseline_arm,
        "treatment_arm": contrast.treatment_arm,
        "proven": bool(proven_paths),
        "paths": proven_paths,
        "details": details,
    }


def _contribution_report(
    indexed: Index,
    contract: InheritanceCrossPathContract,
) -> dict[str, Any]:
    raw = contract.payload.get("contribution_reporting")
    if raw is None:
        return {"enabled": False, "rows": []}
    settings = _object(raw, name="contribution_reporting")
    paths = _strings(settings.get("paths"), name="contribution_reporting.paths")
    if any(path not in contract.paths for path in paths):
        raise InheritanceCrossPathAnalysisError(
            "contribution_reporting references an unknown path"
        )
    hhi_path = _path(settings.get("hhi_path"), name="contribution_reporting.hhi_path")
    effective_path = _path(
        settings.get("effective_path"), name="contribution_reporting.effective_path"
    )
    opportunities_path = _path(
        settings.get("opportunities_path"),
        name="contribution_reporting.opportunities_path",
    )
    successes_path = _path(
        settings.get("successes_path"), name="contribution_reporting.successes_path"
    )
    non_initial = contract.generations[1:]
    rows = []
    for path in paths:
        for pattern in contract.patterns:
            for mode in contract.modes:
                for pair in contract.pairs:
                    generations = indexed[(mode, pattern, path, pair)]
                    hhi = [
                        _finite(_at(generations[g], hhi_path, name="hhi"), name="hhi")
                        for g in non_initial
                    ]
                    effective = [
                        _finite(
                            _at(generations[g], effective_path, name="effective"),
                            name="effective",
                        )
                        for g in non_initial
                    ]
                    opportunities = sum(
                        _finite(
                            _at(generations[g], opportunities_path, name="opportunities"),
                            name="opportunities",
                        )
                        for g in non_initial
                    )
                    successes = sum(
                        _finite(
                            _at(generations[g], successes_path, name="successes"),
                            name="successes",
                        )
                        for g in non_initial
                    )
                    rows.append(
                        {
                            "path": path,
                            "pattern": pattern,
                            "mode": mode,
                            "paired_value": pair,
                            "mean_hhi": float(sum(hhi) / len(hhi)),
                            "mean_effective_contributors": float(
                                sum(effective) / len(effective)
                            ),
                            "successes_per_opportunity": float(
                                successes / opportunities if opportunities else 0.0
                            ),
                        }
                    )
    return {"enabled": True, "observational_only": True, "rows": rows}


def _select_decision(
    contract: InheritanceCrossPathContract,
    *,
    integrity_failed: bool,
    confirmed_paths: set[str],
    locking: bool,
) -> str:
    rules = _array(
        _object(contract.payload["decisions"], name="decisions").get("rules"),
        name="decisions.rules",
    )
    for raw_rule in rules:
        rule = _object(raw_rule, name="decision_rule")
        when = str(rule["when"])
        matches = (
            integrity_failed
            if when == "integrity_failure"
            else confirmed_paths == set(contract.paths) and locking
            if when == "all_paths_confirmed_and_locking"
            else confirmed_paths == set(contract.paths)
            if when == "all_paths_confirmed"
            else confirmed_paths == set(str(item) for item in rule.get("paths", ()))
            if when == "confirmed_paths_exact"
            else True
        )
        if matches:
            return str(rule["decision"])
    raise InheritanceCrossPathAnalysisError("decision rules are not exhaustive")


def _result(
    contract: InheritanceCrossPathContract,
    decision: str,
    *,
    reasons: Sequence[str],
    paired: Sequence[Mapping[str, Any]] = (),
    homogeneous: Sequence[Mapping[str, Any]] = (),
    confirmations: Mapping[str, Any] | None = None,
    cross_path: Mapping[str, Any] | None = None,
    reversals: Sequence[Mapping[str, Any]] = (),
    locking: Mapping[str, Any] | None = None,
    contributions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    integrity_decision = _select_decision(
        contract,
        integrity_failed=True,
        confirmed_paths=set(),
        locking=False,
    )
    return {
        "schema_version": str(
            contract.payload.get(
                "result_schema_version", "inheritance_cross_path_analysis_v1"
            )
        ),
        "contract_sha256": contract.sha256,
        "arms": list(contract.arms),
        "contrasts": [
            {
                "name": contrast.name,
                "baseline_arm": contrast.baseline_arm,
                "treatment_arm": contrast.treatment_arm,
            }
            for contrast in contract.contrasts
        ],
        "expected_cells": contract.expected_cell_count,
        "expected_generation_rows": contract.expected_row_count,
        "decision": decision,
        "decision_label": contract.decision_labels[decision],
        "interpretable": decision != integrity_decision,
        "fail_closed_reasons": list(reasons),
        "paired_milestone_summaries": list(paired),
        "time_to_homogenization": list(homogeneous),
        "path_confirmations": dict(confirmations or {}),
        "cross_path_consistency": dict(cross_path or {}),
        "path_reversals": list(reversals),
        "locking": dict(locking or {}),
        "contribution_inequality": dict(contributions or {}),
    }


def analyze_inheritance_cross_path(
    generation_rows: Sequence[Mapping[str, Any]],
    *,
    analysis_contract: InheritanceCrossPathContract | Mapping[str, Any],
    external_gates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Analyze a complete paired matrix using only the supplied frozen contract."""

    contract = (
        analysis_contract
        if isinstance(analysis_contract, InheritanceCrossPathContract)
        else InheritanceCrossPathContract.from_mapping(analysis_contract)
    )
    try:
        gates = _object(external_gates, name="external_gates")
        gate_reasons = [
            f"external_gate_failed:{name}"
            for name in contract.gate_names
            if gates.get(name) is not True
        ]
        indexed, identity_reasons = _index_rows(generation_rows, contract)
        reasons = sorted(set(gate_reasons + identity_reasons))
        integrity_decision = _select_decision(
            contract,
            integrity_failed=True,
            confirmed_paths=set(),
            locking=False,
        )
        if reasons:
            return _result(contract, integrity_decision, reasons=reasons)
        row_reasons = _row_gate_reasons(indexed, contract)
        if row_reasons:
            return _result(contract, integrity_decision, reasons=row_reasons)

        paired = _paired_summaries(indexed, contract)
        homogeneous = _homogenization(indexed, contract)
        reversals = _path_reversals(paired)
        confirmations = _path_confirmations(
            indexed, paired, homogeneous, contract
        )
        cross_path = _cross_path_consistency(paired, contract)
        locking = _locking(indexed, contract)
        contributions = _contribution_report(indexed, contract)
        benefit = _object(contract.payload["benefit"], name="benefit")
        gating_reversals = [
            reversal
            for reversal in reversals
            if reversal["contrast"] in contract.confirmation_contrasts
        ]
        if gating_reversals and benefit["path_reversal_invalidates"] is True:
            return _result(
                contract,
                integrity_decision,
                reasons=["path_reversal"],
                paired=paired,
                homogeneous=homogeneous,
                confirmations=confirmations,
                cross_path=cross_path,
                reversals=reversals,
                locking=locking,
                contributions=contributions,
            )
        confirmed_paths = {
            path for path, result in confirmations.items() if result["confirmed"]
        }
        decision = _select_decision(
            contract,
            integrity_failed=False,
            confirmed_paths=confirmed_paths,
            locking=bool(locking["proven"]),
        )
        reasons = (
            ["no_nonfailure_decision_predicate_proven"]
            if decision == integrity_decision
            else []
        )
        return _result(
            contract,
            decision,
            reasons=reasons,
            paired=paired,
            homogeneous=homogeneous,
            confirmations=confirmations,
            cross_path=cross_path,
            reversals=reversals,
            locking=locking,
            contributions=contributions,
        )
    except (KeyError, TypeError, ValueError) as exc:
        failure = _select_decision(
            contract,
            integrity_failed=True,
            confirmed_paths=set(),
            locking=False,
        )
        return _result(
            contract,
            failure,
            reasons=[f"analysis_input_invalid:{type(exc).__name__}:{exc}"],
        )


__all__ = [
    "InheritanceAnalysisContrast",
    "InheritanceCrossPathAnalysisError",
    "InheritanceCrossPathContract",
    "analyze_inheritance_cross_path",
]
