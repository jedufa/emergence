#!/usr/bin/env python3
"""Validate and analyze the frozen 48-track GMNW capability matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Final


INPUT_SCHEMA_VERSION: Final[str] = "eiel_gmnw_capability_matrix_results_v1"
ANALYSIS_SCHEMA_VERSION: Final[str] = "eiel_gmnw_capability_matrix_analysis_v1"
MATRIX_SCHEMA_VERSION: Final[str] = "eiel_gmnw_capability_matrix_contract_v1"

CUE_ALPHABETS: Final[tuple[int, ...]] = (2, 4)
DELAYS: Final[tuple[int, ...]] = (4, 12)
DISTRACTOR_ARMS: Final[tuple[bool, ...]] = (False, True)
POLICY_ARMS: Final[tuple[str, ...]] = (
    "persistent_gru",
    "reset_gru",
    "matched_mlp",
)
SEEDS: Final[tuple[int, ...]] = (101, 202)

RATIO_METRICS: Final[tuple[str, ...]] = (
    "held_out_accuracy",
    "native_live_accuracy",
    "chance_accuracy",
    "cue_balanced_accuracy",
    "survival_fraction",
    "reproduction_eligibility_fraction",
    "leakage_accuracy",
)
COUNT_METRICS: Final[tuple[str, ...]] = ("incorrect_decisions", "deaths")
NONNEGATIVE_METRICS: Final[tuple[str, ...]] = ("recurrent_cost",)
EXACTNESS_FIELDS: Final[tuple[str, ...]] = ("repeat_exact", "parity_exact")

PERSISTENT_ACCURACY_MIN: Final[float] = 0.90
CONTROL_ACCURACY_MARGIN: Final[float] = 0.10
LEAKAGE_MARGIN: Final[float] = 0.05
PERSISTENT_SURVIVAL_MIN: Final[float] = 0.80
CONTROL_SURVIVAL_MAX: Final[float] = 0.30

DECISION_A: Final[str] = "GMNW-CAPABILITY-A/PERSISTENT-GRU-ONLY-SOLVES-DELAYED-TASK"
NON_A_DECISIONS: Final[dict[str, str]] = {
    "persistent_held_out_accuracy_min": (
        "GMNW-CAPABILITY-NON-A/PERSISTENT-GRU-ACCURACY-BELOW-0.90"
    ),
    "persistent_native_live_accuracy_min": (
        "GMNW-CAPABILITY-NON-A/PERSISTENT-GRU-LIVE-ACCURACY-BELOW-0.90"
    ),
    "reset_gru_held_out_accuracy_max": (
        "GMNW-CAPABILITY-NON-A/RESET-GRU-ACCURACY-ABOVE-CHANCE-PLUS-0.10"
    ),
    "reset_gru_native_live_accuracy_max": (
        "GMNW-CAPABILITY-NON-A/RESET-GRU-LIVE-ACCURACY-ABOVE-CHANCE-PLUS-0.10"
    ),
    "matched_mlp_held_out_accuracy_max": (
        "GMNW-CAPABILITY-NON-A/MATCHED-MLP-ACCURACY-ABOVE-CHANCE-PLUS-0.10"
    ),
    "matched_mlp_native_live_accuracy_max": (
        "GMNW-CAPABILITY-NON-A/MATCHED-MLP-LIVE-ACCURACY-ABOVE-CHANCE-PLUS-0.10"
    ),
    "leakage_accuracy_max": ("GMNW-CAPABILITY-NON-A/LEAKAGE-ABOVE-CHANCE-PLUS-0.05"),
    "persistent_survival_fraction_min": (
        "GMNW-CAPABILITY-NON-A/PERSISTENT-GRU-SURVIVAL-BELOW-0.80"
    ),
    "reset_gru_survival_fraction_max": (
        "GMNW-CAPABILITY-NON-A/RESET-GRU-SURVIVAL-ABOVE-0.30"
    ),
    "matched_mlp_survival_fraction_max": (
        "GMNW-CAPABILITY-NON-A/MATCHED-MLP-SURVIVAL-ABOVE-0.30"
    ),
    "persistent_recurrent_cost_min": (
        "GMNW-CAPABILITY-NON-A/PERSISTENT-GRU-RECURRENT-COST-NOT-POSITIVE"
    ),
    "exactness_all_green": "GMNW-CAPABILITY-NON-A/EXACTNESS-FAILURE",
}


class CapabilityMatrixValidationError(ValueError):
    """The capability result is incomplete, mixed, or outside the contract."""


@dataclass(frozen=True, slots=True)
class Track:
    """One frozen capability track identity."""

    track_id: str
    cue_alphabet: int
    delay: int
    distractors: bool
    policy_arm: str
    seed: int


def make_track_id(
    *,
    cue_alphabet: int,
    delay: int,
    distractors: bool,
    policy_arm: str,
    seed: int,
) -> str:
    """Return the stable human-readable ID for one frozen track."""

    distractor_label = "on" if distractors else "off"
    policy_label = policy_arm.replace("_", "-")
    return (
        f"gmnw-cap-c{cue_alphabet}-d{delay:02d}-x{distractor_label}-"
        f"{policy_label}-s{seed}"
    )


def _build_tracks() -> tuple[Track, ...]:
    tracks: list[Track] = []
    for cue_alphabet in CUE_ALPHABETS:
        for delay in DELAYS:
            for distractors in DISTRACTOR_ARMS:
                for policy_arm in POLICY_ARMS:
                    for seed in SEEDS:
                        tracks.append(
                            Track(
                                track_id=make_track_id(
                                    cue_alphabet=cue_alphabet,
                                    delay=delay,
                                    distractors=distractors,
                                    policy_arm=policy_arm,
                                    seed=seed,
                                ),
                                cue_alphabet=cue_alphabet,
                                delay=delay,
                                distractors=distractors,
                                policy_arm=policy_arm,
                                seed=seed,
                            )
                        )
    return tuple(tracks)


TRACKS: Final[tuple[Track, ...]] = _build_tracks()
TRACKS_BY_ID: Final[dict[str, Track]] = {track.track_id: track for track in TRACKS}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CapabilityMatrixValidationError(
            f"value is not canonical strict JSON: {exc}"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def canonical_matrix() -> list[dict[str, Any]]:
    """Return the frozen matrix projection used to derive its canonical hash."""

    return [asdict(track) for track in TRACKS]


MATRIX_SHA256: Final[str] = (
    "3f65a85747d22b738c3527b06ab3d7f50c5d8e5777b449895cca14361ebb060c"
)

if len(TRACKS) != 48 or len(TRACKS_BY_ID) != 48:  # pragma: no cover
    raise RuntimeError("the frozen GMNW matrix must contain 48 unique tracks")
if _canonical_sha256(canonical_matrix()) != MATRIX_SHA256:  # pragma: no cover
    raise RuntimeError("the frozen GMNW matrix changed without a hash update")


def _fail(path: str, message: str) -> None:
    raise CapabilityMatrixValidationError(f"{path}: {message}")


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _string(value: Any, *, path: str) -> str:
    if type(value) is not str or not value:
        _fail(path, "must be a non-empty string")
    return value


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, *, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        _fail(path, "must be a finite number")
    return float(value)


def _ratio(value: Any, *, path: str) -> float:
    result = _finite_number(value, path=path)
    if not 0.0 <= result <= 1.0:
        _fail(path, "must be within [0, 1]")
    return result


def _identity(value: Any, *, path: str) -> tuple[Any, str]:
    if type(value) is str:
        if not value:
            _fail(path, "must not be empty")
    elif isinstance(value, Mapping):
        if not value:
            _fail(path, "must not be an empty object")
    else:
        _fail(path, "must be a non-empty string or object")
    canonical = _canonical_json(value)
    return json.loads(canonical), canonical


def _identity_integer(row: Mapping[str, Any], field: str, *, path: str) -> int:
    return _integer(row.get(field), path=f"{path}.{field}")


def _validate_track_row(
    value: Any,
    *,
    index: int,
) -> tuple[dict[str, Any], str, str]:
    path = f"$.results[{index}]"
    row = _mapping(value, path=path)
    required = {
        "track_id",
        "cue_alphabet",
        "delay",
        "distractors",
        "policy_arm",
        "seed",
        "artifact_identity",
        "source_identity",
        *RATIO_METRICS,
        *COUNT_METRICS,
        *NONNEGATIVE_METRICS,
        *EXACTNESS_FIELDS,
        "first_difference",
    }
    missing = sorted(required - set(row))
    if missing:
        _fail(path, f"missing required field(s): {missing}")

    track_id = _string(row["track_id"], path=f"{path}.track_id")
    expected = TRACKS_BY_ID.get(track_id)
    if expected is None:
        _fail(f"{path}.track_id", f"unknown frozen track {track_id!r}")

    actual_identity = (
        _identity_integer(row, "cue_alphabet", path=path),
        _identity_integer(row, "delay", path=path),
        row["distractors"],
        _string(row["policy_arm"], path=f"{path}.policy_arm"),
        _identity_integer(row, "seed", path=path),
    )
    if type(actual_identity[2]) is not bool:
        _fail(f"{path}.distractors", "must be a boolean")
    expected_identity = (
        expected.cue_alphabet,
        expected.delay,
        expected.distractors,
        expected.policy_arm,
        expected.seed,
    )
    if actual_identity != expected_identity:
        _fail(
            path,
            f"track identity does not match {track_id!r}: "
            f"expected {expected_identity}, got {actual_identity}",
        )

    artifact_identity, artifact_canonical = _identity(
        row["artifact_identity"], path=f"{path}.artifact_identity"
    )
    source_identity, source_canonical = _identity(
        row["source_identity"], path=f"{path}.source_identity"
    )

    normalized: dict[str, Any] = {
        **asdict(expected),
        "artifact_identity": artifact_identity,
        "source_identity": source_identity,
    }
    for metric in RATIO_METRICS:
        normalized[metric] = _ratio(row[metric], path=f"{path}.{metric}")
    for metric in COUNT_METRICS:
        normalized[metric] = _integer(row[metric], path=f"{path}.{metric}")
    for metric in NONNEGATIVE_METRICS:
        number = _finite_number(row[metric], path=f"{path}.{metric}")
        if number < 0.0:
            _fail(f"{path}.{metric}", "must be >= 0")
        normalized[metric] = number

    expected_chance = 1.0 / expected.cue_alphabet
    if not math.isclose(
        normalized["chance_accuracy"], expected_chance, rel_tol=0.0, abs_tol=1e-12
    ):
        _fail(
            f"{path}.chance_accuracy",
            f"must equal frozen chance {expected_chance}",
        )

    for field in EXACTNESS_FIELDS:
        if row[field] is not True:
            _fail(f"{path}.{field}", "must be exactly true")
        normalized[field] = True
    if row["first_difference"] is not None:
        _fail(f"{path}.first_difference", "must be null")
    normalized["first_difference"] = None
    return normalized, artifact_canonical, source_canonical


def validate_input(payload: Any) -> list[dict[str, Any]]:
    """Validate and canonically order all 48 input results."""

    root = _mapping(payload, path="$")
    if root.get("schema_version") != INPUT_SCHEMA_VERSION:
        _fail("$.schema_version", f"must equal {INPUT_SCHEMA_VERSION!r}")
    if root.get("matrix_sha256") != MATRIX_SHA256:
        _fail("$.matrix_sha256", f"must equal {MATRIX_SHA256}")
    results = root.get("results")
    if type(results) is not list:
        _fail("$.results", "must be an array")
    if len(results) != len(TRACKS):
        _fail("$.results", f"must contain exactly {len(TRACKS)} rows")

    by_id: dict[str, dict[str, Any]] = {}
    artifact_identities: set[str] = set()
    source_identities: set[str] = set()
    for index, value in enumerate(results):
        row, artifact_identity, source_identity = _validate_track_row(
            value, index=index
        )
        if row["track_id"] in by_id:
            _fail(
                f"$.results[{index}].track_id",
                f"duplicate frozen track {row['track_id']!r}",
            )
        by_id[row["track_id"]] = row
        artifact_identities.add(artifact_identity)
        source_identities.add(source_identity)

    missing = [track.track_id for track in TRACKS if track.track_id not in by_id]
    if missing:
        _fail("$.results", f"missing frozen track(s): {missing}")
    if len(artifact_identities) != 1:
        _fail("$.results", "mixed artifact_identity values are forbidden")
    if len(source_identities) != 1:
        _fail("$.results", "mixed source_identity values are forbidden")
    return [by_id[track.track_id] for track in TRACKS]


def _metric_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows]
    result: dict[str, Any] = {
        "min": min(values),
        "max": max(values),
        "mean": fmean(values),
    }
    if metric in COUNT_METRICS:
        result["sum"] = int(sum(values))
    return result


def _arm_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "metrics": {
            metric: _metric_summary(rows, metric)
            for metric in (*RATIO_METRICS, *COUNT_METRICS, *NONNEGATIVE_METRICS)
        },
        "exactness": {
            "repeat_exact": all(row["repeat_exact"] is True for row in rows),
            "parity_exact": all(row["parity_exact"] is True for row in rows),
            "first_difference_absent": all(
                row["first_difference"] is None for row in rows
            ),
        },
    }


def _check(
    check_id: str,
    *,
    observed: float | bool,
    comparator: str,
    threshold: float | bool,
    passes: bool,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
        "passes": passes,
    }


def _configuration_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    by_arm = {
        arm: sorted(
            (row for row in rows if row["policy_arm"] == arm),
            key=lambda row: int(row["seed"]),
        )
        for arm in POLICY_ARMS
    }
    if any(len(arm_rows) != len(SEEDS) for arm_rows in by_arm.values()):
        raise AssertionError("validated matrix lost an arm or seed")
    aggregates = {arm: _arm_aggregate(by_arm[arm]) for arm in POLICY_ARMS}
    persistent = aggregates["persistent_gru"]["metrics"]
    reset = aggregates["reset_gru"]["metrics"]
    mlp = aggregates["matched_mlp"]["metrics"]
    chance = 1.0 / int(first["cue_alphabet"])
    leakage_max = max(
        aggregates[arm]["metrics"]["leakage_accuracy"]["max"] for arm in POLICY_ARMS
    )
    exactness_all_green = all(
        all(aggregate["exactness"].values()) for aggregate in aggregates.values()
    )
    checks = [
        _check(
            "persistent_held_out_accuracy_min",
            observed=persistent["held_out_accuracy"]["min"],
            comparator=">=",
            threshold=PERSISTENT_ACCURACY_MIN,
            passes=persistent["held_out_accuracy"]["min"] >= PERSISTENT_ACCURACY_MIN,
        ),
        _check(
            "reset_gru_held_out_accuracy_max",
            observed=reset["held_out_accuracy"]["max"],
            comparator="<=",
            threshold=chance + CONTROL_ACCURACY_MARGIN,
            passes=reset["held_out_accuracy"]["max"]
            <= chance + CONTROL_ACCURACY_MARGIN,
        ),
        _check(
            "matched_mlp_held_out_accuracy_max",
            observed=mlp["held_out_accuracy"]["max"],
            comparator="<=",
            threshold=chance + CONTROL_ACCURACY_MARGIN,
            passes=mlp["held_out_accuracy"]["max"] <= chance + CONTROL_ACCURACY_MARGIN,
        ),
        _check(
            "persistent_native_live_accuracy_min",
            observed=persistent["native_live_accuracy"]["min"],
            comparator=">=",
            threshold=PERSISTENT_ACCURACY_MIN,
            passes=persistent["native_live_accuracy"]["min"]
            >= PERSISTENT_ACCURACY_MIN,
        ),
        _check(
            "reset_gru_native_live_accuracy_max",
            observed=reset["native_live_accuracy"]["max"],
            comparator="<=",
            threshold=chance + CONTROL_ACCURACY_MARGIN,
            passes=reset["native_live_accuracy"]["max"]
            <= chance + CONTROL_ACCURACY_MARGIN,
        ),
        _check(
            "matched_mlp_native_live_accuracy_max",
            observed=mlp["native_live_accuracy"]["max"],
            comparator="<=",
            threshold=chance + CONTROL_ACCURACY_MARGIN,
            passes=mlp["native_live_accuracy"]["max"]
            <= chance + CONTROL_ACCURACY_MARGIN,
        ),
        _check(
            "leakage_accuracy_max",
            observed=leakage_max,
            comparator="<=",
            threshold=chance + LEAKAGE_MARGIN,
            passes=leakage_max <= chance + LEAKAGE_MARGIN,
        ),
        _check(
            "persistent_survival_fraction_min",
            observed=persistent["survival_fraction"]["min"],
            comparator=">=",
            threshold=PERSISTENT_SURVIVAL_MIN,
            passes=persistent["survival_fraction"]["min"] >= PERSISTENT_SURVIVAL_MIN,
        ),
        _check(
            "reset_gru_survival_fraction_max",
            observed=reset["survival_fraction"]["max"],
            comparator="<=",
            threshold=CONTROL_SURVIVAL_MAX,
            passes=reset["survival_fraction"]["max"] <= CONTROL_SURVIVAL_MAX,
        ),
        _check(
            "matched_mlp_survival_fraction_max",
            observed=mlp["survival_fraction"]["max"],
            comparator="<=",
            threshold=CONTROL_SURVIVAL_MAX,
            passes=mlp["survival_fraction"]["max"] <= CONTROL_SURVIVAL_MAX,
        ),
        _check(
            "persistent_recurrent_cost_min",
            observed=persistent["recurrent_cost"]["min"],
            comparator=">",
            threshold=0.0,
            passes=persistent["recurrent_cost"]["min"] > 0.0,
        ),
        _check(
            "exactness_all_green",
            observed=exactness_all_green,
            comparator="is",
            threshold=True,
            passes=exactness_all_green,
        ),
    ]
    failed_checks = [check["check_id"] for check in checks if not check["passes"]]
    return {
        "task_config": {
            "cue_alphabet": int(first["cue_alphabet"]),
            "delay": int(first["delay"]),
            "distractors": bool(first["distractors"]),
        },
        "chance_accuracy": chance,
        "track_count": len(rows),
        "arm_aggregates": aggregates,
        "qualification": {
            "qualifies": not failed_checks,
            "checks": checks,
            "failed_checks": failed_checks,
        },
    }


def _selection_key(analysis: Mapping[str, Any]) -> tuple[int, bool, int]:
    config = analysis["task_config"]
    return (
        bool(config["distractors"]),
        int(config["cue_alphabet"]),
        int(config["delay"]),
    )


def _primary_blocker(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    check_order = [
        check["check_id"] for check in analyses[0]["qualification"]["checks"]
    ]
    failures: dict[str, list[dict[str, Any]]] = {name: [] for name in check_order}
    for analysis in analyses:
        checks = {
            check["check_id"]: check for check in analysis["qualification"]["checks"]
        }
        for name in check_order:
            if not checks[name]["passes"]:
                failures[name].append(dict(analysis["task_config"]))
    selected_id = min(
        (name for name in check_order if failures[name]),
        key=lambda name: (-len(failures[name]), check_order.index(name)),
    )
    return {
        "check_id": selected_id,
        "failed_configuration_count": len(failures[selected_id]),
        "total_configuration_count": len(analyses),
        "failed_task_configs": sorted(
            failures[selected_id],
            key=lambda config: (
                config["delay"],
                config["distractors"],
                config["cue_alphabet"],
            ),
        ),
        "rationale": (
            f"No task configuration passed every frozen threshold. "
            f"{selected_id} failed in {len(failures[selected_id])}/"
            f"{len(analyses)} configurations; it is the most frequent "
            "deterministically tie-broken blocker."
        ),
    }


def analyze(payload: Any) -> dict[str, Any]:
    """Validate a complete matrix and return deterministic threshold analysis."""

    rows = validate_input(payload)
    grouped: dict[tuple[int, int, bool], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["cue_alphabet"], row["delay"], row["distractors"])
        grouped.setdefault(key, []).append(row)
    analyses = sorted(
        (_configuration_analysis(group_rows) for group_rows in grouped.values()),
        key=_selection_key,
    )
    qualifiers = [
        analysis for analysis in analyses if analysis["qualification"]["qualifies"]
    ]
    artifact_identity = rows[0]["artifact_identity"]
    source_identity = rows[0]["source_identity"]
    common: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "matrix_contract": {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "matrix_sha256": MATRIX_SHA256,
            "track_count": len(TRACKS),
        },
        "artifact_identity": artifact_identity,
        "source_identity": source_identity,
        "validated_results_sha256": _canonical_sha256(rows),
        "result_count": len(rows),
        "configuration_count": len(analyses),
        "qualifier_count": len(qualifiers),
        "thresholds": {
            "persistent_held_out_accuracy_min": PERSISTENT_ACCURACY_MIN,
            "control_accuracy_margin_above_chance_max": CONTROL_ACCURACY_MARGIN,
            "leakage_margin_above_chance_max": LEAKAGE_MARGIN,
            "persistent_survival_fraction_min": PERSISTENT_SURVIVAL_MIN,
            "control_survival_fraction_max": CONTROL_SURVIVAL_MAX,
            "persistent_recurrent_cost": "strictly_positive",
            "exactness": "repeat_and_parity_true_first_difference_null",
        },
        "configuration_analyses": analyses,
    }
    if qualifiers:
        selected = qualifiers[0]
        config = dict(selected["task_config"])
        rationale = (
            f"Selected delay={config['delay']}, distractors="
            f"{str(config['distractors']).lower()}, cue_alphabet="
            f"{config['cue_alphabet']} from {len(qualifiers)} qualifier(s): "
            "no distractors first, then cue alphabet 2 before 4, then shortest "
            "delay. Selection used capability results only."
        )
        common.update(
            {
                "ok": True,
                "decision": DECISION_A,
                "selected_configuration": config,
                "selection_rationale": rationale,
                "primary_blocker": None,
                "human_readable": (
                    f"{DECISION_A}. Validated 48/48 tracks under one artifact "
                    f"and source identity; {len(qualifiers)}/8 task "
                    f"configurations qualified. {rationale}"
                ),
            }
        )
    else:
        blocker = _primary_blocker(analyses)
        decision = NON_A_DECISIONS[blocker["check_id"]]
        common.update(
            {
                "ok": False,
                "decision": decision,
                "selected_configuration": None,
                "selection_rationale": None,
                "primary_blocker": blocker,
                "human_readable": (
                    f"{decision}. Validated 48/48 tracks under one "
                    "artifact and source identity; 0/8 task configurations "
                    f"qualified. {blocker['rationale']}"
                ),
            }
        )
    return common


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CapabilityMatrixValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load strict JSON, rejecting duplicate keys and non-finite constants."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CapabilityMatrixValidationError(
                    f"non-finite JSON constant is forbidden: {value}"
                )
            ),
        )
    except CapabilityMatrixValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapabilityMatrixValidationError(
            f"cannot load strict JSON {path}: {exc}"
        ) from exc


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic, sorted strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result = analyze(load_json(args.input))
        write_json(args.output, result)
    except CapabilityMatrixValidationError as exc:
        print(f"GMNW capability analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
