from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scripts.eiel_option_d_m15_live_rng_inventory_classification_v1 import (
    ACTIVE_PHYSICAL_STREAMS,
    build_receipt,
    classify_hit,
)


def test_m15_classifier_splits_generated_and_active_route_state() -> None:
    generated = classify_hit(
        {
            "path": "engine/authoritative_state/generated/schema_v1.hpp",
            "line": 1,
            "kind": "pcg64_state",
            "text": "runner_pcg64_state_words",
        }
    )
    assert generated.category == "generated_schema_echo"
    assert generated.route_relevance == (
        "derived_from_authoritative_contracts_not_independent_stream"
    )

    active = classify_hit(
        {
            "path": "engine/authoritative_state/contracts/full_tick_event_v1.py",
            "line": 50,
            "kind": "pcg64_state",
            "text": '("runner_pcg64_state_words", "uint64", "[B,4]")',
        }
    )
    assert active.category == "active_route_state_contract"
    assert active.stream_id == "numpy.runner"


def test_m15_classifier_names_live_population_standard_normal_gap() -> None:
    hit = classify_hit(
        {
            "path": "engine/perf/cuda_architecture_b_persistent_turnover_drives.cuh",
            "line": 321,
            "kind": "pcg64_state",
            "text": "pcg64_numpy_standard_normal(commit_rng)",
        }
    )
    assert hit.category == "active_route_operator_gap"
    assert hit.stream_id == "numpy.population"
    assert hit.operator_id == "population.newborn_drive.standard_normal"


def test_m15_receipt_is_route_bound_and_fails_on_unresolved_engine_hit(
    tmp_path: Path,
) -> None:
    scan = {
        "hits_total": 4,
        "classified_total": 2,
        "unknown_total": 1,
        "blocked_total": 1,
        "suspicious_device_semantics": [],
        "classified": [
            {
                "path": "engine/authoritative_state/contracts/full_tick_event_v1.py",
                "line": 50,
                "kind": "pcg64_state",
                "text": "runner_pcg64_state_words",
                "context_sha256": "a",
            },
            {
                "path": "engine/authoritative_state/generated/schema_v1.hpp",
                "line": 1,
                "kind": "pcg64_state",
                "text": "runner_pcg64_state_words",
                "context_sha256": "b",
            },
        ],
        "unknown": [
            {
                "path": "engine/new_rng_surface.py",
                "line": 7,
                "kind": "python.random",
                "text": "random.random()",
                "context_sha256": "c",
            }
        ],
        "blocked": [
            {
                "path": "engine/perf/cuda_architecture_b_persistent_turnover_drives.cuh",
                "line": 321,
                "kind": "pcg64_state",
                "text": "pcg64_numpy_standard_normal(commit_rng)",
                "context_sha256": "d",
            }
        ],
    }
    stream_inventory = {
        "accepted_live_reconciliation": False,
        "streams": [{"stream_id": row["physical_stream_id"]} for row in ACTIVE_PHYSICAL_STREAMS],
        "unique_physical_state_count_if_alias_preserved": 6,
    }
    operator_inventory = {
        "accepted_live_reconciliation": False,
        "operators": [{"operator_id": "runner.drive_noise.standard_normal"}],
    }
    scan_path = tmp_path / "scan.json"
    stream_path = tmp_path / "streams.json"
    operator_path = tmp_path / "operators.json"
    scan_path.write_text(json.dumps(scan), encoding="utf-8")
    stream_path.write_text(json.dumps(stream_inventory), encoding="utf-8")
    operator_path.write_text(json.dumps(operator_inventory), encoding="utf-8")

    receipt = build_receipt(
        repo_root=Path.cwd(),
        scan_path=scan_path,
        stream_inventory_path=stream_path,
        operator_inventory_path=operator_path,
    )

    assert receipt["passed"] is False
    assert receipt["m15_counts"]["unresolved_engine_hit_count"] == 1
    assert receipt["m15_counts"]["active_operator_gap_hit_count"] == 1
    assert (
        "population.newborn_drive.standard_normal"
        in receipt["promotion_blockers"]["missing_operator_gap_ids_from_v31_inventory"]
    )


def test_m15_receipt_passes_when_every_engine_hit_is_classified(tmp_path: Path) -> None:
    scan = {
        "hits_total": 3,
        "classified_total": 3,
        "unknown_total": 0,
        "blocked_total": 0,
        "suspicious_device_semantics": [],
        "classified": [
            {
                "path": "engine/authoritative_state/contracts/full_tick_event_v1.py",
                "line": 50,
                "kind": "pcg64_state",
                "text": "runner_pcg64_state_words",
            },
            {
                "path": "engine/authoritative_state/generated/schema_v1.hpp",
                "line": 1,
                "kind": "pcg64_state",
                "text": "runner_pcg64_state_words",
            },
            {
                "path": "engine/agents/directed_inheritance.py",
                "line": 335,
                "kind": "python.random",
                "text": "rng.random()",
            },
        ],
        "unknown": [],
        "blocked": [],
    }
    stream_inventory = {
        "accepted_live_reconciliation": False,
        "streams": [{"stream_id": row["physical_stream_id"]} for row in ACTIVE_PHYSICAL_STREAMS],
        "unique_physical_state_count_if_alias_preserved": 6,
    }
    operator_inventory = {
        "accepted_live_reconciliation": False,
        "operators": [],
    }
    scan_path = tmp_path / "scan.json"
    stream_path = tmp_path / "streams.json"
    operator_path = tmp_path / "operators.json"
    scan_path.write_text(json.dumps(scan), encoding="utf-8")
    stream_path.write_text(json.dumps(stream_inventory), encoding="utf-8")
    operator_path.write_text(json.dumps(operator_inventory), encoding="utf-8")

    receipt = build_receipt(
        repo_root=Path.cwd(),
        scan_path=scan_path,
        stream_inventory_path=stream_path,
        operator_inventory_path=operator_path,
    )

    assert receipt["passed"] is True
    assert receipt["physical_stream_count_if_environment_policy_alias_preserved"] == 6
    assert receipt["m15_counts"]["unresolved_engine_hit_count"] == 0
