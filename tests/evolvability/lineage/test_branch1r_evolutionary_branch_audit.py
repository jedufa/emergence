from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/branch1r_evolutionary_branch_audit.py"
SPEC = importlib.util.spec_from_file_location("branch1r_evolutionary_branch_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
branch1r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(branch1r)


def _founder(agent_id: int, founder: str, generation: int = 0, *, strategy: str = "s0") -> dict[str, object]:
    return {
        "event_type": "founder",
        "generation": generation,
        "agent_id": agent_id,
        "founder_ancestry_id": founder,
        "genome_signature": f"g-{founder}",
        "behavior_signature": f"b-{strategy}",
        "ecology_signature": f"e-{strategy}",
        "reproduction_signature": f"r-{strategy}",
        "cognition_signature": f"c-{strategy}",
        "strategy_id": strategy,
    }


def _birth(
    parent: int,
    child: int,
    generation: int,
    founder: str,
    *,
    genome: str,
    behavior: str,
    ecology: str,
    reproduction: str = "",
    cognition: str = "",
    strategy: str = "s0",
) -> dict[str, object]:
    return {
        "event_type": "birth_success",
        "generation": generation,
        "event_id": f"e-{generation}-{parent}-{child}",
        "parent_agent_id": parent,
        "child_agent_id": child,
        "parent_biological_lineage_id": founder,
        "child_biological_lineage_id": founder,
        "genome_signature": genome,
        "behavior_signature": behavior,
        "ecology_signature": ecology,
        "reproduction_signature": reproduction or f"r-{strategy}",
        "cognition_signature": cognition or f"c-{strategy}",
        "strategy_id": strategy,
    }


def _classification(rows: list[dict[str, object]]) -> str:
    return branch1r.analyze_event_rows(rows)["three_layer_gate"]["classification"]


def test_several_founders_without_descendant_branching_are_not_oversplit() -> None:
    rows = [_founder(1, "L1"), _founder(2, "L2"), _founder(3, "L3")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-L1", behavior="b-s0", ecology="e-s0"),
            _birth(2, 21, 1, "L2", genome="g-L2", behavior="b-s0", ecology="e-s0"),
            _birth(3, 31, 1, "L3", genome="g-L3", behavior="b-s0", ecology="e-s0"),
        ]
    )

    result = branch1r.analyze_event_rows(rows)

    assert result["tree"]["root_branch_count"] == 3
    assert result["tree"]["persistent_branch_count"] == 0
    assert all(row["active_descendant_branches"] == 3 for row in result["branches_by_generation"])


def test_one_founder_dominates_but_splits_into_three_persistent_branches() -> None:
    rows = [_founder(1, "L1")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(11, 12, 2, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(1, 21, 1, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="sb"),
            _birth(21, 22, 2, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="sb"),
            _birth(1, 31, 1, "L1", genome="g-c", behavior="b-c", ecology="e-c", strategy="sc"),
            _birth(31, 32, 2, "L1", genome="g-c", behavior="b-c", ecology="e-c", strategy="sc"),
        ]
    )

    result = branch1r.analyze_event_rows(rows)

    assert result["tree"]["persistent_branch_count"] == 3
    assert result["branches_by_generation"][-1]["active_descendant_branches"] == 3
    assert result["branches_by_generation"][-1]["active_strategies"] == 3
    assert result["three_layer_gate"]["classification"] == "healthy_branching_search"


def test_several_founders_can_converge_to_one_strategy() -> None:
    rows = [_founder(1, "L1", strategy="same"), _founder(2, "L2", strategy="same")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="same"),
            _birth(11, 12, 2, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="same"),
            _birth(2, 21, 1, "L2", genome="g-b", behavior="b-b", ecology="e-b", strategy="same"),
            _birth(21, 22, 2, "L2", genome="g-b", behavior="b-b", ecology="e-b", strategy="same"),
        ]
    )

    assert _classification(rows) == "founder_diverse_but_strategy_converged"


def test_transient_mutations_do_not_count_as_persistent_branches() -> None:
    rows = [_founder(1, "L1")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(1, 12, 2, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="sb"),
        ]
    )

    result = branch1r.analyze_event_rows(rows, min_persistence_generations=2)

    assert result["tree"]["persistent_branch_count"] == 0
    assert {b["confidence_classification"] for b in result["tree"]["branches"] if b["parent_branch_id"]} == {
        "transient_descriptor_split"
    }


def test_branch_can_split_die_and_split_again() -> None:
    rows = [_founder(1, "L1")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(11, 12, 2, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(11, 13, 2, "L1", genome="g-dead", behavior="b-dead", ecology="e-dead", strategy="sd"),
            _birth(12, 14, 3, "L1", genome="g-c", behavior="b-c", ecology="e-c", strategy="sc"),
            _birth(14, 15, 4, "L1", genome="g-c", behavior="b-c", ecology="e-c", strategy="sc"),
        ]
    )

    result = branch1r.analyze_event_rows(rows)

    assert result["tree"]["persistent_branch_count"] == 2
    assert any(event["event_type"] == "branch_split" for event in result["branch_events"])


def test_two_branches_can_independently_converge_to_one_strategy() -> None:
    rows = [_founder(1, "L1")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="same"),
            _birth(11, 12, 2, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="same"),
            _birth(1, 21, 1, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="same"),
            _birth(21, 22, 2, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="same"),
        ]
    )

    assert _classification(rows) == "founder_diverse_but_strategy_converged"


def test_repeat_parity_produces_stable_branch_tree_json() -> None:
    rows = [_founder(1, "L1")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(11, 12, 2, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
        ]
    )
    shuffled = [rows[1], rows[0], rows[2]]

    a = branch1r.analyze_event_rows(rows)["tree"]
    b = branch1r.analyze_event_rows(shuffled)["tree"]

    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_missing_descriptors_fail_closed() -> None:
    rows = [
        {"event_type": "founder", "generation": 0, "agent_id": 1, "founder_ancestry_id": "L1"},
        {"event_type": "birth_success", "generation": 1, "parent_agent_id": 1, "child_agent_id": 2},
    ]

    result = branch1r.analyze_event_rows(rows)

    assert result["coverage"]["missing_descriptors"] is True
    assert result["three_layer_gate"]["classification"] == "insufficient_branch_data"


def test_candidate_a_like_founder_and_strategy_collapse() -> None:
    rows = [_founder(1, "L1", strategy="same")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-L1", behavior="b-same", ecology="e-same", strategy="same"),
            _birth(11, 12, 2, "L1", genome="g-L1", behavior="b-same", ecology="e-same", strategy="same"),
        ]
    )

    assert _classification(rows) == "strategy_collapse"


def test_candidate_b_like_founder_drift_with_healthy_branch_diversity() -> None:
    rows = [_founder(1, "L1")]
    rows.extend(
        [
            _birth(1, 11, 1, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(11, 12, 2, "L1", genome="g-a", behavior="b-a", ecology="e-a", strategy="sa"),
            _birth(1, 21, 1, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="sb"),
            _birth(21, 22, 2, "L1", genome="g-b", behavior="b-b", ecology="e-b", strategy="sb"),
        ]
    )

    assert _classification(rows) == "healthy_branching_search"


def test_archive_without_parent_child_events_is_insufficient_branch_data(tmp_path: Path) -> None:
    archive = tmp_path / "compact.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "campaign/r/r0001/metrics/lineage_summary.csv",
            "generation,active_lineages,dominant_lineage_population_share\n0,2,0.75\n",
        )
        zf.writestr(
            "campaign/r/r0001/metrics/lineage_descriptors.jsonl",
            json.dumps(
                {
                    "generation": 0,
                    "lineages": [
                        {
                            "lineage_id": "L1",
                            "population_count": 3,
                            "mean_offspring_count": 1,
                            "mean_memory_dim": 0,
                        }
                    ],
                }
            )
            + "\n",
        )

    rows, index = branch1r.analyze_archive_zip(archive, family="fixture")

    assert index["branch_reconstruction_status"] == "insufficient_branch_data"
    assert rows[0]["classification"] == "insufficient_branch_data"
    assert rows[0]["coverage_gap"] == "parent_child_birth_event_rows_absent_or_pruned"
