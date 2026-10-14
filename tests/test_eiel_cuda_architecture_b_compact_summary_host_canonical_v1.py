from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import random
import struct
import subprocess
import sysconfig
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pybind11
import pytest

from engine.native_runtime.compact_summary_digests import (
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
)
from simulation.metrics_pkg.lineage import (
    LineageMetricsState,
    NativeLineageAgentRecord,
    NativeLineageGenomeRecord,
    NativeLineageStatsRecord,
    compute_lineage_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_compact_summary_host_canonical_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
BINDING = (
    ROOT
    / "tests"
    / "perf"
    / "test_cuda_architecture_b_compact_summary_host_canonical_v1_binding.cpp"
)
MODULE_NAME = "cuda_architecture_b_compact_summary_host_canonical_v1_probe"

DEFINED_ACTIVE = 1 << 7
DEFINED_DOMINANT_TEXT = 1 << 8
DEFINED_DOMINANT_POPULATION_SHARE = 1 << 9
DEFINED_TOP3_SHARE = 1 << 10
DEFINED_POPULATION_ENTROPY = 1 << 11
DEFINED_REPRODUCTION_ENTROPY = 1 << 12
DEFINED_DOMINANT_REPRODUCTION_SHARE = 1 << 13
DEFINED_RARE_PERSISTENT = 1 << 14
DEFINED_COLLAPSE_WARNING = 1 << 15
DEFINED_COLLAPSE_STREAK = 1 << 16
DEFINED_LINEAGE_BASE = (
    DEFINED_ACTIVE
    | DEFINED_DOMINANT_POPULATION_SHARE
    | DEFINED_TOP3_SHARE
    | DEFINED_POPULATION_ENTROPY
    | DEFINED_RARE_PERSISTENT
    | DEFINED_COLLAPSE_WARNING
    | DEFINED_COLLAPSE_STREAK
)


def _f32(value: float) -> float:
    return float(np.float32(value))


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    build = tmp_path_factory.mktemp("compact_summary_host_canonical_v1")
    extension = build / f"{MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    completed = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{ROOT / 'engine' / 'perf'}",
            str(BINDING),
            "-o",
            str(extension),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _behavior_rows_signed_zero() -> list[dict[str, object]]:
    return [
        {
            "generation": 0,
            "agent_id": 0,
            "cohort": "start",
            "move_n": 0,
            "move_s": 0,
            "move_e": 0,
            "move_w": 0,
            "gather": 0,
            "rest": 0,
            "attack": 0,
            "lifespan_steps": 0,
            "energy_collected": -0.0,
            "offspring": 0,
            "attacks_made": 0,
            "fitness": 0.0,
            "alive_end": 1,
            "eligible_for_selection": 1,
            "kept_next": 1,
            "injury_end": -0.0,
            "injury_delta": 0.0,
            "aggression_bias": -0.0,
        }
    ]


def _lineage_record(
    *,
    agent_id: int,
    founder_id: int,
    energy: float,
    health: float,
    injury: float,
    drives: list[float],
    y: int,
    x: int,
    birth_tick: int,
    lineage_depth: int,
    offspring_count: int,
    survival_steps: float,
    energy_collected: float,
    first_repro_step: int | None,
    pred_err_count: int,
    pred_err_sum: float,
    episode0_pred_err_mean: float,
    identity_drift_ema: float,
    drive_jitter: float,
    hidden_size: int,
    memory_dim: int,
    use_gru: bool,
    aggression_bias: float,
    exploration_bias: float,
    patch_lineage: int,
) -> dict[str, object]:
    return {
        "agent_id": agent_id,
        "energy": energy,
        "health": health,
        "injury": injury,
        "drives": drives,
        "y": y,
        "x": x,
        "stats": {
            "lineage_id": f"L{founder_id:08d}",
            "founder_id": founder_id,
            "parent_ids": [],
            "parent_lineage_ids": [],
            "birth_generation": 0,
            "birth_tick": birth_tick,
            "lineage_depth": lineage_depth,
            "offspring_count": offspring_count,
            "survival_steps": survival_steps,
            "energy_collected": energy_collected,
            "first_repro_step": first_repro_step,
            "pred_err_count": pred_err_count,
            "pred_err_sum": pred_err_sum,
            "episode0_pred_err_mean": episode0_pred_err_mean,
            "identity_drift_ema": identity_drift_ema,
            "drive_jitter": drive_jitter,
            "ecological_role": "general_agent",
            "role_source": "default",
            "role_confidence": None,
            "patch_habitat_lineage_id": patch_lineage,
        },
        "genome": {
            "hidden_size": hidden_size,
            "memory_dim": memory_dim,
            "use_gru": use_gru,
            "aggression_bias": aggression_bias,
            "exploration_bias": exploration_bias,
        },
    }


def _lineage_signed_zero() -> list[dict[str, object]]:
    return [
        _lineage_record(
            agent_id=0,
            founder_id=0,
            energy=-0.0,
            health=-0.0,
            injury=-0.0,
            drives=[-0.0, 0.0, -0.0, 0.0],
            y=0,
            x=0,
            birth_tick=0,
            lineage_depth=0,
            offspring_count=0,
            survival_steps=0.0,
            energy_collected=-0.0,
            first_repro_step=None,
            pred_err_count=0,
            pred_err_sum=-0.0,
            episode0_pred_err_mean=0.0,
            identity_drift_ema=0.0,
            drive_jitter=0.0,
            hidden_size=0,
            memory_dim=0,
            use_gru=False,
            aggression_bias=-0.0,
            exploration_bias=-0.0,
            patch_lineage=0,
        )
    ]


def _behavior_rows_real_shaped() -> list[dict[str, object]]:
    rows = [
        (10, "start", (2, 0, 0, 0, 0, 0, 0), 3, 0.75, 0, 0, 1.5, 0, 0, 0, -0.0, 0.5, 0.375),
        (11, "start", (0, 0, 0, 0, 0, 0, 1), 6, 2.5, 1, 1, 8.25, 1, 1, 1, 0.25, 0.125, -0.25),
        (12, "born", (0, 0, 0, 0, 1, 0, 0), 5, 1.5, 0, 0, 5.125, 1, 0, 0, 0.0, 0.0, 0.42961761),
    ]
    output: list[dict[str, object]] = []
    for (
        agent_id,
        cohort,
        actions,
        lifespan,
        energy,
        offspring,
        attacks,
        fitness,
        alive,
        eligible,
        kept,
        injury_start,
        injury_end,
        aggression,
    ) in rows:
        output.append(
            {
                "generation": 4,
                "agent_id": agent_id,
                "cohort": cohort,
                "move_n": actions[0],
                "move_s": actions[1],
                "move_e": actions[2],
                "move_w": actions[3],
                "gather": actions[4],
                "rest": actions[5],
                "attack": actions[6],
                "lifespan_steps": lifespan,
                "energy_collected": float(energy),
                "offspring": offspring,
                "attacks_made": attacks,
                "fitness": float(fitness),
                "alive_end": alive,
                "eligible_for_selection": eligible,
                "kept_next": kept,
                "injury_end": _f32(injury_end),
                "injury_delta": _f32(injury_end) - _f32(injury_start),
                "aggression_bias": float(aggression),
            }
        )
    return output


def _lineage_real_shaped() -> list[dict[str, object]]:
    return [
        _lineage_record(
            agent_id=11,
            founder_id=11,
            energy=41.0,
            health=1.0,
            injury=0.125,
            drives=[_f32(1.1), _f32(1.2), _f32(1.3), _f32(1.4)],
            y=3,
            x=4,
            birth_tick=0,
            lineage_depth=0,
            offspring_count=1,
            survival_steps=6.0,
            energy_collected=2.5,
            first_repro_step=4,
            pred_err_count=2,
            pred_err_sum=1.0,
            episode0_pred_err_mean=0.25,
            identity_drift_ema=0.03125,
            drive_jitter=0.0625,
            hidden_size=9,
            memory_dim=5,
            use_gru=True,
            aggression_bias=-0.25,
            exploration_bias=-0.25,
            patch_lineage=8,
        ),
        _lineage_record(
            agent_id=12,
            founder_id=12,
            energy=40.0,
            health=1.0,
            injury=0.0,
            drives=[_f32(0.1), _f32(0.2), _f32(0.3), _f32(0.4)],
            y=2,
            x=3,
            birth_tick=3,
            lineage_depth=1,
            offspring_count=0,
            survival_steps=5.0,
            energy_collected=1.5,
            first_repro_step=None,
            pred_err_count=1,
            pred_err_sum=0.5,
            episode0_pred_err_mean=0.25,
            identity_drift_ema=0.015625,
            drive_jitter=0.03125,
            hidden_size=8,
            memory_dim=4,
            use_gru=False,
            aggression_bias=round(_f32(0.123456789), 8),
            exploration_bias=-0.42961761,
            patch_lineage=7,
        ),
    ]


def _native_agent(row: dict[str, int]) -> NativeLineageAgentRecord:
    founder_id = row["founder_id"]
    stats = NativeLineageStatsRecord(
        lineage_id=f"L{founder_id:08d}",
        founder_id=founder_id,
        parent_ids=(),
        parent_lineage_ids=(),
        birth_generation=0,
        birth_tick=0,
        lineage_depth=row["lineage_depth"],
        offspring_count=row["offspring_count"],
        survival_steps=0.0,
        energy_collected=0.0,
        first_repro_step=None,
        pred_err_count=0,
        pred_err_sum=0.0,
        episode0_pred_err_mean=0.0,
        identity_drift_ema=0.0,
        drive_jitter=0.0,
        ecological_role="general_agent",
        role_source="default",
        role_confidence=None,
    )
    genome = NativeLineageGenomeRecord(
        hidden_size=0,
        memory_dim=0,
        use_gru=False,
        aggression_bias=0.0,
        exploration_bias=0.0,
    )
    return NativeLineageAgentRecord(
        agent_id=row["agent_id"],
        energy=0.0,
        health=0.0,
        injury=0.0,
        drives=(0.0, 0.0, 0.0, 0.0),
        genome=genome,
        stats=stats,
    )


def _python_lineage_snapshot(
    generation: int,
    lineage_rows: list[dict[str, int]],
    state_values: dict[str, int],
    config: dict[str, float | int],
) -> tuple[dict[str, object], dict[str, int]]:
    state = LineageMetricsState(**state_values)
    summary, _, _, state_after = compute_lineage_snapshot(
        generation=generation,
        agents=[_native_agent(row) for row in lineage_rows],
        sim_cfg=SimpleNamespace(**config),
        state=state,
    )
    return summary, {
        "collapse_streak": state_after.collapse_streak,
        "population_dominance_streak": state_after.population_dominance_streak,
        "reproduction_dominance_streak": state_after.reproduction_dominance_streak,
    }


def _assert_float_bits_equal(
    native: float,
    python: float,
    *,
    field: str,
    evidence: object,
) -> None:
    native_bits = struct.pack("=d", native)
    python_bits = struct.pack("=d", python)
    if native_bits != python_bits:
        pytest.fail(
            f"first aggregate float mismatch field={field}: "
            f"native={native!r}/{native_bits.hex()} "
            f"python={python!r}/{python_bits.hex()} evidence={evidence!r}"
        )


def _assert_summary_matches_python(
    actual: dict[str, object],
    expected: dict[str, object],
    expected_state: dict[str, int],
    *,
    evidence: object,
) -> None:
    assert actual["status"] == 0, evidence
    lineage = actual["lineage"]
    assert lineage["record_count"] == evidence["lineage_count"]
    assert lineage["active_lineages"] == expected["active_lineages"]
    assert lineage["dominant_lineage_id"] == (
        expected["dominant_lineage_id"] or None
    )
    assert (
        lineage["rare_persistent_lineages"]
        == expected["rare_persistent_lineages"]
    )
    assert lineage["collapse_streak"] == expected["lineage_collapse_streak"]
    assert lineage["collapse_warning_code"] == (
        2 if expected["lineage_collapse_warning"] else 1
    )
    assert actual["lineage_state_after"] == expected_state

    float_fields = {
        "dominant_population_share": "dominant_lineage_population_share",
        "top3_population_share": "top3_lineage_population_share",
        "population_entropy": "lineage_population_entropy",
        "reproduction_entropy": "lineage_reproduction_entropy",
        "dominant_reproduction_share": "dominant_lineage_reproduction_share",
    }
    for native_name, python_name in float_fields.items():
        native_value = lineage[native_name]
        python_value = expected[python_name]
        if python_value is None:
            assert native_value is None, (native_name, evidence)
        else:
            assert native_value is not None, (native_name, evidence)
            _assert_float_bits_equal(
                native_value,
                python_value,
                field=native_name,
                evidence=evidence,
            )

    expected_mask = DEFINED_LINEAGE_BASE
    dominant = expected["dominant_lineage_id"]
    if dominant:
        expected_mask |= DEFINED_DOMINANT_TEXT
    if expected["lineage_reproduction_entropy"] is not None:
        expected_mask |= (
            DEFINED_REPRODUCTION_ENTROPY
            | DEFINED_DOMINANT_REPRODUCTION_SHARE
        )
    assert lineage["defined_mask"] == expected_mask
    text = (dominant or "").encode("ascii")
    expected_record = bytes([len(text)]) + text + bytes(64 - len(text))
    assert lineage["dominant_lineage_text_record"] == expected_record


def _assert_behavior_aggregates(
    actual: dict[str, int],
    rows: list[dict[str, object]],
    start_id_ceiling: int,
) -> None:
    totals = [sum(row["actions"][index] for row in rows) for index in range(7)]
    cohort = [row for row in rows if row["agent_id"] < start_id_ceiling]
    assert actual == {
        "row_count": len(rows),
        "cohort_size": len(cohort),
        "cohort_survivors": sum(bool(row["alive"]) for row in cohort),
        "action_total": sum(totals),
        "move_total": sum(totals[:4]),
        "move_n": totals[0],
        "move_s": totals[1],
        "move_e": totals[2],
        "move_w": totals[3],
        "gather": totals[4],
        "rest": totals[5],
        "attack": totals[6],
    }


@pytest.mark.parametrize(
    "name,generation,behavior,lineage",
    [
        ("signed_zero", 0, _behavior_rows_signed_zero(), _lineage_signed_zero()),
        ("real_shaped", 4, _behavior_rows_real_shaped(), _lineage_real_shaped()),
    ],
)
def test_native_digests_match_python_exactly(
    probe: ModuleType,
    name: str,
    generation: int,
    behavior: list[dict[str, object]],
    lineage: list[dict[str, object]],
) -> None:
    actual = probe.run_fixture(name)
    assert actual["status"] == 0
    assert actual["behavior_digest"].hex() == behavior_rows_digest_sha256(behavior)
    assert actual["lineage_digest"].hex() == lineage_records_digest_sha256(
        generation, lineage
    )
    if name == "signed_zero":
        assert struct.pack("=d", actual["lineage"]["population_entropy"]) == (
            struct.pack("=d", 0.0)
        )
        assert actual["lineage"]["reproduction_entropy"] is None
        assert actual["lineage"]["dominant_reproduction_share"] is None


def test_randomized_behavior_and_lineage_aggregates_match_python_exactly(
    probe: ModuleType,
) -> None:
    rng = random.Random(0xC0A57A1)
    founder_pool = [0, 1, 7, 42, 99_999_999, 100_000_000]
    thresholds = [-0.0, 0.0, 0.1, 0.5, 0.8, 1.0, 1.25]
    for case in range(160):
        behavior_count = rng.randrange(13)
        agent_ids = sorted(rng.sample(range(0, 80), behavior_count))
        start_id_ceiling = rng.randrange(0, 82)
        behavior_rows = [
            {
                "agent_id": agent_id,
                "alive": bool(rng.getrandbits(1)),
                "actions": [rng.randrange(256) for _ in range(7)],
            }
            for agent_id in agent_ids
        ]
        lineage_rows = [
            {
                "agent_id": row["agent_id"],
                "founder_id": rng.choice(founder_pool),
                "lineage_depth": rng.randrange(5),
                "offspring_count": rng.randrange(8),
            }
            for row in behavior_rows
            if row["alive"]
        ]
        state = {
            "collapse_streak": rng.randrange(5),
            "population_dominance_streak": rng.randrange(5),
            "reproduction_dominance_streak": rng.randrange(5),
        }
        config = {
            "lineage_dominance_threshold": rng.choice(thresholds),
            "lineage_dominance_streak_required": rng.randrange(-1, 6),
            "lineage_min_active_lineages": rng.randrange(-1, 6),
            "lineage_entropy_minimum": rng.choice(thresholds),
            "lineage_reproduction_dominance_threshold": rng.choice(thresholds),
            "lineage_reproduction_dominance_streak_required": rng.randrange(-1, 6),
        }
        expected, expected_state = _python_lineage_snapshot(
            case, lineage_rows, state, config
        )
        actual = probe.run_rows(
            case,
            start_id_ceiling,
            behavior_rows,
            lineage_rows,
            state,
            config,
        )
        evidence = {
            "case": case,
            "lineage_count": len(lineage_rows),
            "lineage_rows": lineage_rows,
            "state": state,
            "config": config,
        }
        _assert_behavior_aggregates(
            actual["behavior"], behavior_rows, start_id_ceiling
        )
        _assert_summary_matches_python(
            actual, expected, expected_state, evidence=evidence
        )


def test_real_shaped_two_generation_state_advance_and_lexical_tie_break(
    probe: ModuleType,
) -> None:
    config = {
        "lineage_dominance_threshold": 0.50,
        "lineage_dominance_streak_required": 2,
        "lineage_min_active_lineages": 2,
        "lineage_entropy_minimum": -0.0,
        "lineage_reproduction_dominance_threshold": 0.60,
        "lineage_reproduction_dominance_streak_required": 2,
    }
    generations = [
        [
            (99_999_999, 2, 3),
            (99_999_999, 1, 0),
            (100_000_000, 0, 2),
            (100_000_000, 3, 0),
        ],
        [
            (100_000_000, 3, 2),
            (100_000_000, 3, 1),
            (100_000_000, 2, 1),
            (100_000_000, 2, 0),
            (100_000_000, 1, 0),
            (100_000_000, 1, 0),
            (100_000_000, 0, 0),
            (99_999_999, 2, 0),
            (7, 4, 0),
            (8, 0, 0),
        ],
    ]
    native_state = {
        "collapse_streak": 0,
        "population_dominance_streak": 0,
        "reproduction_dominance_streak": 0,
    }
    python_state = LineageMetricsState()
    for generation, shaped_rows in enumerate(generations):
        lineage_rows = [
            {
                "agent_id": index,
                "founder_id": founder,
                "lineage_depth": depth,
                "offspring_count": offspring,
            }
            for index, (founder, depth, offspring) in enumerate(shaped_rows)
        ]
        behavior_rows = [
            {
                "agent_id": row["agent_id"],
                "alive": True,
                "actions": [
                    (row["agent_id"] + generation + action) % 11
                    for action in range(7)
                ],
            }
            for row in lineage_rows
        ]
        expected, _, _, python_state = compute_lineage_snapshot(
            generation=generation,
            agents=[_native_agent(row) for row in lineage_rows],
            sim_cfg=SimpleNamespace(**config),
            state=python_state,
        )
        expected_state = {
            "collapse_streak": python_state.collapse_streak,
            "population_dominance_streak": python_state.population_dominance_streak,
            "reproduction_dominance_streak": python_state.reproduction_dominance_streak,
        }
        actual = probe.run_rows(
            generation,
            6,
            behavior_rows,
            lineage_rows,
            native_state,
            config,
        )
        evidence = {
            "generation": generation,
            "lineage_count": len(lineage_rows),
            "lineage_rows": lineage_rows,
        }
        _assert_behavior_aggregates(actual["behavior"], behavior_rows, 6)
        _assert_summary_matches_python(
            actual, expected, expected_state, evidence=evidence
        )
        if generation == 0:
            assert expected["dominant_lineage_id"] == "L100000000"
            assert actual["lineage"]["dominant_lineage_id"] == "L100000000"
        native_state = actual["lineage_state_after"]

    assert generations[0][0][0] < generations[0][2][0]
    assert expected_state == native_state


def test_lineage_null_flags_and_state_passthrough_when_metrics_are_undefined(
    probe: ModuleType,
) -> None:
    behavior_rows = [
        {"agent_id": 1, "alive": True, "actions": [0, 1, 2, 3, 4, 5, 6]}
    ]
    lineage_rows = [
        {
            "agent_id": 1,
            "founder_id": 1,
            "lineage_depth": 2,
            "offspring_count": 3,
        }
    ]
    state = {
        "collapse_streak": 4,
        "population_dominance_streak": 3,
        "reproduction_dominance_streak": 2,
    }
    config = {
        "lineage_dominance_threshold": 0.8,
        "lineage_dominance_streak_required": 5,
        "lineage_min_active_lineages": 2,
        "lineage_entropy_minimum": 0.1,
        "lineage_reproduction_dominance_threshold": 0.8,
        "lineage_reproduction_dominance_streak_required": 5,
    }
    actual = probe.run_rows(
        0, 2, behavior_rows, lineage_rows, state, config, metrics_defined=False
    )
    assert actual["status"] == 0
    _assert_behavior_aggregates(actual["behavior"], behavior_rows, 2)
    assert actual["lineage"]["record_count"] == 1
    assert actual["lineage"]["defined_mask"] == 0
    assert actual["lineage"]["dominant_lineage_id"] is None
    assert actual["lineage"]["dominant_lineage_text_record"] == bytes(65)
    assert actual["lineage_state_after"] == state


def test_empty_lineage_summary_and_nonfinite_config_contract(probe: ModuleType) -> None:
    state = {
        "collapse_streak": 0,
        "population_dominance_streak": 0,
        "reproduction_dominance_streak": 0,
    }
    config = {
        "lineage_dominance_threshold": 0.8,
        "lineage_dominance_streak_required": 5,
        "lineage_min_active_lineages": 2,
        "lineage_entropy_minimum": 0.1,
        "lineage_reproduction_dominance_threshold": 0.8,
        "lineage_reproduction_dominance_streak_required": 5,
    }
    expected, expected_state = _python_lineage_snapshot(0, [], state, config)
    actual = probe.run_rows(0, 0, [], [], state, config)
    _assert_summary_matches_python(
        actual,
        expected,
        expected_state,
        evidence={"case": "empty", "lineage_count": 0},
    )
    assert actual["lineage"]["dominant_lineage_text_record"] == bytes(65)

    config["lineage_entropy_minimum"] = math.inf
    invalid = probe.run_rows(0, 0, [], [], state, config)
    assert invalid["status"] == 18
    assert invalid["failed_field"] == 24
    assert invalid["behavior_digest"] == bytes(32)
    assert invalid["lineage_digest"] == bytes(32)


def test_to_chars_lexemes_match_python_at_first_difference(probe: ModuleType) -> None:
    values = [
        -0.0,
        0.0,
        -1.0,
        1.0,
        0.125,
        -0.25,
        _f32(0.123456789),
        *[_f32(value) for value in (0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4)],
        1e-7,
        1e-4,
        1e15,
        1e16,
    ]
    for index, value in enumerate(values):
        native = probe.format_float(value)
        python = json.dumps(value, allow_nan=False, separators=(",", ":"))
        if native != python:
            pytest.fail(
                f"first float mismatch at index={index} value={value!r}: "
                f"native={native!r} python={python!r}"
            )


@pytest.mark.parametrize(
    "fixture,status",
    [
        ("unsealed", 5),
        ("count", 6),
        ("inactive_nonzero", 7),
        ("behavior_order", 8),
        ("behavior_nonfinite", 11),
        ("lineage_order", 12),
        ("lineage_nonfinite", 15),
        ("lineage_mismatch", 16),
        ("alive_count_mismatch", 16),
    ],
)
def test_retained_contracts_fail_closed(
    probe: ModuleType, fixture: str, status: int
) -> None:
    result = probe.run_fixture(fixture)
    assert result["status"] == status
    assert result["behavior_digest"] == bytes(32)
    assert result["lineage_digest"] == bytes(32)


def test_callback_pod_matches_direct_result(probe: ModuleType) -> None:
    assert probe.run_fixture("real_shaped", callback=True) == probe.run_fixture(
        "real_shaped", callback=False
    )
    assert probe.pod_receipt() == {
        "authority_header_bytes": 16,
        "authority_behavior_bytes": 56,
        "authority_lineage_bytes": 136,
        "result_bytes": 360,
        "request_bytes": 128,
        "lineage_input_bytes": 80,
        "behavior_output_bytes": 96,
        "lineage_output_bytes": 160,
        "lineage_state_bytes": 24,
        "dominant_text_record_bytes": 65,
        "digest_bytes": 32,
    }


@pytest.mark.parametrize("payload", [b"", b"abc", b"x" * 55, b"x" * 64, b"x" * 1000])
def test_host_sha256_matches_known_implementation(
    probe: ModuleType, payload: bytes
) -> None:
    assert probe.sha256_bytes(payload) == hashlib.sha256(payload).digest()


def test_production_sources_are_host_only_and_compile_portably(tmp_path: Path) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    assert "cuda_runtime" not in source + header
    assert "cudaLaunch" not in source + header
    assert "pybind" not in source + header
    assert "Python.h" not in source + header
    assert "std::vector" not in source
    assert "operator new" not in source
    assert "throw " not in source
    completed = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O3",
            "-fno-exceptions",
            "-fno-rtti",
            f"-I{HEADER.parent}",
            "-c",
            str(SOURCE),
            "-o",
            str(tmp_path / "host_canonical.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_separate_host_object_links_with_real_authority_pods(tmp_path: Path) -> None:
    caller = tmp_path / "compact_summary_host_canonical_v1_link.cpp"
    caller.write_text(
        """
#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"
#include "cuda_architecture_b_compact_summary_host_canonical_v1.hpp"

int main() {
    namespace authority =
        eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;
    namespace canonical =
        eiel::cuda_architecture_b::compact_summary_host_canonical_v1;
    authority::GenerationAuthorityHeaderV1 header{0, 0, 0, 2};
    authority::BehaviorAuthorityRowV1 behavior[1]{};
    authority::LineageAuthorityRowV1 lineage[1]{};
    canonical::HostCanonicalResultV1 result{};
    return canonical::canonicalize_retained_digests_v1(
        0, &header, behavior, lineage, 1, &result
    ) == canonical::HostCanonicalStatusV1::kSuccess ? 0 : 1;
}
""".lstrip(),
        encoding="ascii",
    )
    executable = tmp_path / "compact_summary_host_canonical_v1_link"
    compiled = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{HEADER.parent}",
            str(caller),
            str(SOURCE),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    linked = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True
    )
    assert linked.returncode == 0, linked.stderr
