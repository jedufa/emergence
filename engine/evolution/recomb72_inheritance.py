from __future__ import annotations

"""Frozen, audit-only inheritance semantics for RECOMB-72.

This module is deliberately independent of campaign orchestration and runtime
owners.  It defines the six measured scalar loci, four inheritance modes, their
draw budgets, the founder packages, and the genotype-blind parent schedule used
by the neutral audit fixture.
"""

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "recomb72_inheritance_v1"
REGISTRY_VERSION = "recomb72_audited_scalar_registry_v1"
PARENT_PAIR_SCHEDULE_VERSION = "recomb72_genotype_blind_parent_pairs_v1"
ANCESTRY_MERGE_VERSION = "recomb72_equal_parent_ancestry_v1"
MUTATION_OFF_CONTRACT_VERSION = "recomb72_post_inheritance_mutation_off_v1"

CURRENT_AUTHORITATIVE = "current_authoritative"
PARENT_COPY_PER_LOCUS = "parent_copy_per_locus"
BOUNDED_INTERPOLATION = "bounded_interpolation"
WHOLE_PACKAGE_SINGLE_PARENT = "whole_package_single_parent"

INHERITANCE_MODES = (
    CURRENT_AUTHORITATIVE,
    PARENT_COPY_PER_LOCUS,
    BOUNDED_INTERPOLATION,
    WHOLE_PACKAGE_SINGLE_PARENT,
)

AUDITED_LOCI = (
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
    "resource_handling_axis",
    "drive_inertia",
    "pred_lr",
)

LOCUS_BOUNDS: dict[str, tuple[float, float]] = {
    "exploration_bias": (-1.0, 1.0),
    "aggression_bias": (-2.0, 2.0),
    "action_temperature": (0.25, 2.5),
    "resource_handling_axis": (-1.0, 1.0),
    "drive_inertia": (0.0, 0.999),
    "pred_lr": (0.0, 1.0),
}

NEUTRAL_PACKAGE: dict[str, float] = {
    "exploration_bias": 0.0,
    "aggression_bias": 0.0,
    "action_temperature": 1.0,
    "resource_handling_axis": 0.0,
    "drive_inertia": 0.85,
    "pred_lr": 0.08,
}

MODE_DRAW_BUDGETS: dict[str, int] = {
    CURRENT_AUTHORITATIVE: 0,
    PARENT_COPY_PER_LOCUS: len(AUDITED_LOCI),
    BOUNDED_INTERPOLATION: len(AUDITED_LOCI),
    WHOLE_PACKAGE_SINGLE_PARENT: 1,
}

MUTATION_OFF_CONTRACT: dict[str, Any] = {
    "version": MUTATION_OFF_CONTRACT_VERSION,
    "scalar_mutation": False,
    "mutation_rate_mutation": False,
    "memory_mutation": False,
    "policy_weight_mutation": False,
    "structural_mutation": False,
    "mutation_events_required": 0,
    "mutation_rng_draws_required": 0,
}


class Recomb72InheritanceError(ValueError):
    """Raised when the frozen audit contract is violated."""


@dataclass(frozen=True)
class InheritanceResult:
    mode: str
    child: dict[str, float]
    draw_count: int
    donor_by_locus: dict[str, str]
    package_donor: str | None
    mode_sha256: str
    registry_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_float(value: Any, *, locus: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise Recomb72InheritanceError(
            f"{locus} must be a finite number"
        ) from exc
    if not math.isfinite(out):
        raise Recomb72InheritanceError(f"{locus} must be a finite number")
    return out


def canonical_package(raw: Mapping[str, Any]) -> dict[str, float]:
    """Validate and canonicalize one complete audited package."""

    if not isinstance(raw, Mapping):
        raise Recomb72InheritanceError("audited package must be a mapping")
    missing = [locus for locus in AUDITED_LOCI if locus not in raw]
    extra = sorted(str(key) for key in raw if key not in AUDITED_LOCI)
    if missing or extra:
        raise Recomb72InheritanceError(
            f"audited package fields mismatch: missing={missing!r} extra={extra!r}"
        )
    out: dict[str, float] = {}
    for locus in AUDITED_LOCI:
        value = _finite_float(raw[locus], locus=locus)
        lo, hi = LOCUS_BOUNDS[locus]
        if value < lo or value > hi:
            raise Recomb72InheritanceError(
                f"{locus}={value!r} is outside [{lo}, {hi}]"
            )
        out[locus] = float(value)
    return out


def _pattern_package(
    *,
    overrides: Mapping[str, float],
) -> dict[str, float]:
    package = dict(NEUTRAL_PACKAGE)
    package.update({str(key): float(value) for key, value in overrides.items()})
    return canonical_package(package)


FOUNDER_PATTERNS: dict[str, dict[str, dict[str, float]]] = {
    "single_extreme_scalar": {
        "a": _pattern_package(overrides={"exploration_bias": -1.0}),
        "b": _pattern_package(overrides={"exploration_bias": 1.0}),
    },
    "resource_specialization_extremes": {
        "a": _pattern_package(overrides={"resource_handling_axis": -1.0}),
        "b": _pattern_package(overrides={"resource_handling_axis": 1.0}),
    },
    "correlated_multi_trait_packages": {
        "a": canonical_package(
            {locus: LOCUS_BOUNDS[locus][0] for locus in AUDITED_LOCI}
        ),
        "b": canonical_package(
            {locus: LOCUS_BOUNDS[locus][1] for locus in AUDITED_LOCI}
        ),
    },
}


def registry_metadata() -> dict[str, Any]:
    """Return the complete hashable contract without its self-hash."""

    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "audited_loci": [
            {
                "name": locus,
                "bounds": list(LOCUS_BOUNDS[locus]),
                "value_type": "bounded_float",
            }
            for locus in AUDITED_LOCI
        ],
        "modes": [
            {
                "id": mode,
                "draw_budget": {"inheritance.py": MODE_DRAW_BUDGETS[mode]},
            }
            for mode in INHERITANCE_MODES
        ],
        "founder_patterns": copy.deepcopy(FOUNDER_PATTERNS),
        "neutral_package": dict(NEUTRAL_PACKAGE),
        "mutation_off_contract": copy.deepcopy(MUTATION_OFF_CONTRACT),
        "parent_pair_schedule_version": PARENT_PAIR_SCHEDULE_VERSION,
        "ancestry_merge_version": ANCESTRY_MERGE_VERSION,
    }


def registry_sha256() -> str:
    return _sha256(registry_metadata())


def mode_metadata(mode: str) -> dict[str, Any]:
    selected = validate_mode(mode)
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "registry_sha256": registry_sha256(),
        "mode": selected,
        "draw_budget": {"inheritance.py": MODE_DRAW_BUDGETS[selected]},
        "audited_loci": list(AUDITED_LOCI),
        "mutation_off_contract_version": MUTATION_OFF_CONTRACT_VERSION,
    }


def mode_sha256(mode: str) -> str:
    return _sha256(mode_metadata(mode))


def validate_mode(mode: Any) -> str:
    selected = str(mode or "").strip()
    if selected not in INHERITANCE_MODES:
        raise Recomb72InheritanceError(
            f"unsupported RECOMB-72 inheritance mode: {selected!r}"
        )
    return selected


def inherit_package(
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    *,
    mode: str,
    rng: random.Random,
) -> InheritanceResult:
    """Apply one frozen inheritance mode to the complete audited package."""

    selected = validate_mode(mode)
    if rng is None or not callable(getattr(rng, "random", None)):
        raise Recomb72InheritanceError("inheritance requires a random.Random-compatible RNG")
    a = canonical_package(parent_a)
    b = canonical_package(parent_b)
    child: dict[str, float] = {}
    donor_by_locus: dict[str, str] = {}
    package_donor: str | None = None
    draws = 0

    if selected == CURRENT_AUTHORITATIVE:
        for locus in AUDITED_LOCI:
            lo, hi = LOCUS_BOUNDS[locus]
            child[locus] = min(hi, max(lo, 0.5 * (a[locus] + b[locus])))
    elif selected == PARENT_COPY_PER_LOCUS:
        for locus in AUDITED_LOCI:
            donor = "a" if float(rng.random()) < 0.5 else "b"
            draws += 1
            donor_by_locus[locus] = donor
            child[locus] = a[locus] if donor == "a" else b[locus]
    elif selected == BOUNDED_INTERPOLATION:
        for locus in AUDITED_LOCI:
            alpha = float(rng.random())
            draws += 1
            if not 0.0 <= alpha < 1.0:
                raise Recomb72InheritanceError(
                    "random.Random-compatible RNG returned a value outside [0, 1)"
                )
            lo, hi = LOCUS_BOUNDS[locus]
            value = a[locus] + alpha * (b[locus] - a[locus])
            child[locus] = min(hi, max(lo, value))
    else:
        package_donor = "a" if float(rng.random()) < 0.5 else "b"
        draws = 1
        donor = a if package_donor == "a" else b
        child = dict(donor)
        donor_by_locus = {locus: package_donor for locus in AUDITED_LOCI}

    child = canonical_package(child)
    expected = MODE_DRAW_BUDGETS[selected]
    if draws != expected:
        raise RuntimeError(
            f"RECOMB-72 inheritance draw-budget mismatch: mode={selected} "
            f"expected={expected} observed={draws}"
        )
    return InheritanceResult(
        mode=selected,
        child=child,
        draw_count=draws,
        donor_by_locus=donor_by_locus,
        package_donor=package_donor,
        mode_sha256=mode_sha256(selected),
        registry_sha256=registry_sha256(),
    )


def founder_packages(pattern: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return detached canonical A/B packages for one frozen founder pattern."""

    selected = str(pattern or "").strip()
    if selected not in FOUNDER_PATTERNS:
        raise Recomb72InheritanceError(
            f"unsupported RECOMB-72 founder pattern: {selected!r}"
        )
    row = FOUNDER_PATTERNS[selected]
    return dict(row["a"]), dict(row["b"])


def _schedule_word(*parts: Any) -> int:
    payload = {
        "version": PARENT_PAIR_SCHEDULE_VERSION,
        "parts": list(parts),
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(payload)).digest()[:8], "big")


def deterministic_parent_pairs(
    eligible_parent_ids: Sequence[int],
    *,
    seed: int,
    generation: int,
    child_count: int,
) -> tuple[tuple[int, int], ...]:
    """Return a genotype-blind, RNG-state-independent two-parent schedule."""

    supplied_parent_ids = tuple(int(value) for value in eligible_parent_ids)
    if len(set(supplied_parent_ids)) != len(supplied_parent_ids):
        raise Recomb72InheritanceError(
            "eligible parent IDs must be unique"
        )
    parent_ids = tuple(sorted(supplied_parent_ids))
    if len(parent_ids) < 2:
        raise Recomb72InheritanceError("parent-pair schedule requires two eligible parents")
    if int(generation) < 0:
        raise Recomb72InheritanceError("generation must be non-negative")
    if int(child_count) < 0:
        raise Recomb72InheritanceError("child_count must be non-negative")

    pairs: list[tuple[int, int]] = []
    count = len(parent_ids)
    for child_ordinal in range(int(child_count)):
        first_index = _schedule_word(
            int(seed), int(generation), child_ordinal, "parent_a"
        ) % count
        mate_offset = 1 + (
            _schedule_word(int(seed), int(generation), child_ordinal, "parent_b")
            % (count - 1)
        )
        second_index = (first_index + mate_offset) % count
        pairs.append((parent_ids[first_index], parent_ids[second_index]))
    return tuple(pairs)


def merge_ancestry(
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
) -> dict[str, float]:
    """Merge normalized parental ancestry at equal 1/2 contribution."""

    def normalized(raw: Mapping[str, Any], *, parent: str) -> dict[str, float]:
        if not isinstance(raw, Mapping) or not raw:
            raise Recomb72InheritanceError(f"{parent} ancestry must be non-empty")
        values: dict[str, float] = {}
        for key, value in raw.items():
            weight = _finite_float(value, locus=f"{parent}.{key}")
            if weight < 0.0:
                raise Recomb72InheritanceError(
                    f"{parent} ancestry weights must be non-negative"
                )
            values[str(key)] = weight
        total = sum(values.values())
        if total <= 0.0:
            raise Recomb72InheritanceError(
                f"{parent} ancestry weights must have positive mass"
            )
        return {key: value / total for key, value in values.items()}

    a = normalized(parent_a, parent="parent_a")
    b = normalized(parent_b, parent="parent_b")
    merged: dict[str, float] = {}
    for source in (a, b):
        for key, value in source.items():
            merged[key] = merged.get(key, 0.0) + 0.5 * value
    return {key: merged[key] for key in sorted(merged)}


__all__ = [
    "ANCESTRY_MERGE_VERSION",
    "AUDITED_LOCI",
    "BOUNDED_INTERPOLATION",
    "CURRENT_AUTHORITATIVE",
    "FOUNDER_PATTERNS",
    "INHERITANCE_MODES",
    "InheritanceResult",
    "LOCUS_BOUNDS",
    "MODE_DRAW_BUDGETS",
    "MUTATION_OFF_CONTRACT",
    "MUTATION_OFF_CONTRACT_VERSION",
    "NEUTRAL_PACKAGE",
    "PARENT_COPY_PER_LOCUS",
    "PARENT_PAIR_SCHEDULE_VERSION",
    "Recomb72InheritanceError",
    "SCHEMA_VERSION",
    "WHOLE_PACKAGE_SINGLE_PARENT",
    "canonical_package",
    "deterministic_parent_pairs",
    "founder_packages",
    "inherit_package",
    "merge_ancestry",
    "mode_metadata",
    "mode_sha256",
    "registry_metadata",
    "registry_sha256",
    "validate_mode",
]
