from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tools.scripts import eiel_gmnw_capability_matrix_analyzer_v1 as analyzer


def _valid_payload() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for track in analyzer.TRACKS:
        chance = 1.0 / track.cue_alphabet
        persistent = track.policy_arm == "persistent_gru"
        rows.append(
            {
                "track_id": track.track_id,
                "cue_alphabet": track.cue_alphabet,
                "delay": track.delay,
                "distractors": track.distractors,
                "policy_arm": track.policy_arm,
                "seed": track.seed,
                "artifact_identity": {"extension_sha256": "a" * 64},
                "source_identity": {"git_head": "b" * 40, "dirty": False},
                "held_out_accuracy": 0.95 if persistent else chance + 0.02,
                "native_live_accuracy": 0.95 if persistent else chance + 0.02,
                "chance_accuracy": chance,
                "cue_balanced_accuracy": 0.95 if persistent else chance + 0.01,
                "survival_fraction": 0.90 if persistent else 0.20,
                "reproduction_eligibility_fraction": 0.85 if persistent else 0.10,
                "incorrect_decisions": 1 if persistent else 40,
                "deaths": 2 if persistent else 80,
                "recurrent_cost": 1.0 if persistent else 0.0,
                "leakage_accuracy": chance,
                "repeat_exact": True,
                "parity_exact": True,
                "first_difference": None,
            }
        )
    return {
        "schema_version": analyzer.INPUT_SCHEMA_VERSION,
        "matrix_sha256": analyzer.MATRIX_SHA256,
        "results": rows,
    }


def _rows(
    payload: dict[str, Any],
    *,
    cue: int | None = None,
    delay: int | None = None,
    distractors: bool | None = None,
    arm: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in payload["results"]
        if (cue is None or row["cue_alphabet"] == cue)
        and (delay is None or row["delay"] == delay)
        and (distractors is None or row["distractors"] is distractors)
        and (arm is None or row["policy_arm"] == arm)
    ]


def _set(
    payload: dict[str, Any],
    field: str,
    value: Any,
    **filters: Any,
) -> None:
    selected = _rows(payload, **filters)
    assert selected
    for row in selected:
        row[field] = value


def _target_check(
    result: dict[str, Any],
    check_id: str,
    filters: dict[str, Any],
) -> dict[str, Any]:
    analyses = result["configuration_analyses"]
    for analysis in analyses:
        config = analysis["task_config"]
        if (
            ("cue" not in filters or config["cue_alphabet"] == filters["cue"])
            and ("delay" not in filters or config["delay"] == filters["delay"])
            and (
                "distractors" not in filters
                or config["distractors"] is filters["distractors"]
            )
        ):
            return next(
                check
                for check in analysis["qualification"]["checks"]
                if check["check_id"] == check_id
            )
    raise AssertionError(f"no configuration matched {filters}")


def test_frozen_matrix_has_48_unique_stable_tracks_and_hash() -> None:
    assert len(analyzer.TRACKS) == 48
    assert len({track.track_id for track in analyzer.TRACKS}) == 48
    assert analyzer.TRACKS[0].track_id == ("gmnw-cap-c2-d04-xoff-persistent-gru-s101")
    assert analyzer.TRACKS[-1].track_id == ("gmnw-cap-c4-d12-xon-matched-mlp-s202")
    assert analyzer.MATRIX_SHA256 == (
        "3f65a85747d22b738c3527b06ab3d7f50c5d8e5777b449895cca14361ebb060c"
    )
    assert (
        analyzer._canonical_sha256(analyzer.canonical_matrix())
        == analyzer.MATRIX_SHA256
    )


def test_valid_matrix_selects_simplest_qualifier_and_aggregates_arms() -> None:
    result = analyzer.analyze(_valid_payload())

    assert result["ok"] is True
    assert result["decision"] == analyzer.DECISION_A
    assert result["selected_configuration"] == {
        "cue_alphabet": 2,
        "delay": 4,
        "distractors": False,
    }
    assert result["qualifier_count"] == 8
    assert "no distractors first" in result["selection_rationale"]
    simplest = result["configuration_analyses"][0]
    assert simplest["track_count"] == 6
    assert simplest["arm_aggregates"]["persistent_gru"]["seed_count"] == 2
    assert simplest["arm_aggregates"]["persistent_gru"]["metrics"][
        "held_out_accuracy"
    ] == {"min": 0.95, "max": 0.95, "mean": 0.95}


def test_selection_order_is_simplest_then_shortest_delay() -> None:
    payload = _valid_payload()
    _set(
        payload,
        "held_out_accuracy",
        0.89,
        cue=2,
        delay=4,
        distractors=False,
        arm="persistent_gru",
    )

    result = analyzer.analyze(payload)

    assert result["selected_configuration"] == {
        "cue_alphabet": 2,
        "delay": 12,
        "distractors": False,
    }


@pytest.mark.parametrize(
    ("field", "boundary", "failure", "filters", "check_id"),
    [
        (
            "held_out_accuracy",
            0.90,
            0.899999,
            {"arm": "persistent_gru"},
            "persistent_held_out_accuracy_min",
        ),
        (
            "held_out_accuracy",
            0.60,
            0.600001,
            {"cue": 2, "arm": "reset_gru"},
            "reset_gru_held_out_accuracy_max",
        ),
        (
            "held_out_accuracy",
            0.60,
            0.600001,
            {"cue": 2, "arm": "matched_mlp"},
            "matched_mlp_held_out_accuracy_max",
        ),
        (
            "native_live_accuracy",
            0.90,
            0.899999,
            {"arm": "persistent_gru"},
            "persistent_native_live_accuracy_min",
        ),
        (
            "native_live_accuracy",
            0.60,
            0.600001,
            {"cue": 2, "arm": "reset_gru"},
            "reset_gru_native_live_accuracy_max",
        ),
        (
            "native_live_accuracy",
            0.60,
            0.600001,
            {"cue": 2, "arm": "matched_mlp"},
            "matched_mlp_native_live_accuracy_max",
        ),
        (
            "leakage_accuracy",
            0.55,
            0.550001,
            {"cue": 2},
            "leakage_accuracy_max",
        ),
        (
            "survival_fraction",
            0.80,
            0.799999,
            {"arm": "persistent_gru"},
            "persistent_survival_fraction_min",
        ),
        (
            "survival_fraction",
            0.30,
            0.300001,
            {"arm": "reset_gru"},
            "reset_gru_survival_fraction_max",
        ),
        (
            "survival_fraction",
            0.30,
            0.300001,
            {"arm": "matched_mlp"},
            "matched_mlp_survival_fraction_max",
        ),
    ],
)
def test_inclusive_threshold_boundaries(
    field: str,
    boundary: float,
    failure: float,
    filters: dict[str, Any],
    check_id: str,
) -> None:
    boundary_payload = _valid_payload()
    _set(boundary_payload, field, boundary, **filters)
    boundary_result = analyzer.analyze(boundary_payload)
    assert _target_check(boundary_result, check_id, filters)["passes"] is True

    failing_payload = _valid_payload()
    _set(failing_payload, field, failure, **filters)
    failing_result = analyzer.analyze(failing_payload)
    assert _target_check(failing_result, check_id, filters)["passes"] is False


def test_recurrent_cost_is_strictly_positive() -> None:
    positive = _valid_payload()
    _set(positive, "recurrent_cost", 1e-12, arm="persistent_gru")
    assert analyzer.analyze(positive)["ok"] is True

    zero = _valid_payload()
    _set(zero, "recurrent_cost", 0.0, arm="persistent_gru")
    result = analyzer.analyze(zero)
    assert result["ok"] is False
    assert result["primary_blocker"]["check_id"] == ("persistent_recurrent_cost_min")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.pop("schema_version"), "schema_version"),
        (
            lambda payload: payload.update(matrix_sha256="0" * 64),
            "matrix_sha256",
        ),
        (lambda payload: payload["results"].pop(), "exactly 48"),
        (
            lambda payload: payload["results"][0].pop("held_out_accuracy"),
            "missing required field",
        ),
        (
            lambda payload: payload["results"].__setitem__(
                1, copy.deepcopy(payload["results"][0])
            ),
            "duplicate frozen track",
        ),
        (
            lambda payload: payload["results"][0].update(track_id="unknown"),
            "unknown frozen track",
        ),
        (
            lambda payload: payload["results"][0].update(held_out_accuracy=1.01),
            r"within \[0, 1\]",
        ),
        (
            lambda payload: payload["results"][0].update(recurrent_cost=float("nan")),
            "finite number",
        ),
        (
            lambda payload: payload["results"][0].update(
                artifact_identity={"extension_sha256": "c" * 64}
            ),
            "mixed artifact_identity",
        ),
        (
            lambda payload: payload["results"][0].update(
                source_identity={"git_head": "c" * 40}
            ),
            "mixed source_identity",
        ),
        (
            lambda payload: payload["results"][0].update(chance_accuracy=0.3),
            "must equal frozen chance",
        ),
    ],
)
def test_malformed_or_mixed_inputs_fail_closed(mutation: Any, match: str) -> None:
    payload = _valid_payload()
    mutation(payload)
    with pytest.raises(analyzer.CapabilityMatrixValidationError, match=match):
        analyzer.analyze(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeat_exact", False),
        ("parity_exact", False),
        ("first_difference", {"field": "action"}),
    ],
)
def test_any_exactness_failure_fails_closed(field: str, value: Any) -> None:
    payload = _valid_payload()
    payload["results"][17][field] = value
    with pytest.raises(analyzer.CapabilityMatrixValidationError, match=field):
        analyzer.analyze(payload)


def test_no_qualifier_returns_one_precise_non_a_blocker() -> None:
    payload = _valid_payload()
    _set(payload, "held_out_accuracy", 0.89, arm="persistent_gru")

    result = analyzer.analyze(payload)

    assert result["ok"] is False
    assert result["decision"] == (
        "GMNW-CAPABILITY-NON-A/PERSISTENT-GRU-ACCURACY-BELOW-0.90"
    )
    assert result["selected_configuration"] is None
    assert result["primary_blocker"] == {
        "check_id": "persistent_held_out_accuracy_min",
        "failed_configuration_count": 8,
        "total_configuration_count": 8,
        "failed_task_configs": [
            {"cue_alphabet": cue, "delay": delay, "distractors": distractors}
            for delay in (4, 12)
            for distractors in (False, True)
            for cue in (2, 4)
        ],
        "rationale": (
            "No task configuration passed every frozen threshold. "
            "persistent_held_out_accuracy_min failed in 8/8 configurations; "
            "it is the most frequent deterministically tie-broken blocker."
        ),
    }
    assert "0/8 task configurations qualified" in result["human_readable"]


def test_cli_writes_deterministic_sorted_json(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    input_path.write_text(json.dumps(_valid_payload()), encoding="utf-8")

    assert (
        analyzer.main(["--input", str(input_path), "--output", str(first_output)]) == 0
    )
    assert (
        analyzer.main(["--input", str(input_path), "--output", str(second_output)]) == 0
    )
    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_output.read_text(encoding="ascii").startswith(
        '{\n  "artifact_identity"'
    )
    assert json.loads(first_output.read_text())["decision"] == analyzer.DECISION_A


def test_strict_json_loader_rejects_duplicate_keys_and_nan(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"results": [], "results": []}', encoding="utf-8")
    with pytest.raises(analyzer.CapabilityMatrixValidationError, match="duplicate"):
        analyzer.load_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(analyzer.CapabilityMatrixValidationError, match="non-finite"):
        analyzer.load_json(nonfinite)
