from __future__ import annotations

import json
from types import SimpleNamespace

from simulation.metrics_pkg.birth_skew_mechanism import (
    maybe_write_birth_skew_mechanism_report,
    summarize_generation,
)


def _agent(lineage_id: str) -> SimpleNamespace:
    return SimpleNamespace(stats=SimpleNamespace(lineage_id=lineage_id))


def test_birth_skew_mechanism_summary_tracks_lineage_concentration() -> None:
    record, state = summarize_generation(
        generation=2,
        selected_parent_agents=[_agent("L1"), _agent("L1"), _agent("L2")],
        alive_agents=[_agent("L1"), _agent("L1"), _agent("L1"), _agent("L2")],
        birth_contexts=[{"lineage_id": "L1"}, {"lineage_id": "L1"}, {"lineage_id": "L1"}],
        state={},
        top_n=2,
    )

    assert record["reporting_only"] is True
    assert record["mechanics_intervention"] is False
    assert record["selected_parent_concentration"]["top1_lineage_id"] == "L1"
    assert record["selected_parent_concentration"]["top1_share"] == 2 / 3
    assert record["birth_opportunity"]["top1_share"] == 1.0
    assert record["birth_opportunity"]["first_birth_dominance_generation"] == 2
    assert record["alive_context"]["top1_share"] == 0.75
    assert state["first_birth_dominance_generation"] == 2
    assert state["nonzero_birth_lineages"] == ["L1"]


def test_maybe_write_birth_skew_mechanism_report_is_default_off(tmp_path) -> None:
    sim_cfg = SimpleNamespace(birth_skew_mechanism_reporting_enable=False)

    result = maybe_write_birth_skew_mechanism_report(
        run_dir=tmp_path,
        generation=1,
        sim_cfg=sim_cfg,
        selected_parent_agents=[_agent("L1")],
        alive_agents=[_agent("L1")],
        birth_contexts=[{"lineage_id": "L1"}],
    )

    assert result is None
    assert not (tmp_path / "metrics" / "birth_skew_mechanism.jsonl").exists()


def test_maybe_write_birth_skew_mechanism_report_writes_jsonl_when_enabled(tmp_path) -> None:
    sim_cfg = SimpleNamespace(
        birth_skew_mechanism_reporting_enable=True,
        birth_skew_mechanism_top_n=1,
    )

    result = maybe_write_birth_skew_mechanism_report(
        run_dir=tmp_path,
        generation=3,
        sim_cfg=sim_cfg,
        selected_parent_agents=[_agent("L1"), _agent("L2")],
        alive_agents=[_agent("L1")],
        birth_contexts=[{"lineage_id": "L2"}],
        state={"first_birth_dominance_generation": None},
    )

    assert result is not None
    path = tmp_path / "metrics" / "birth_skew_mechanism.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "birth_skew_mechanism_reporting_v1"
    assert rows[0]["birth_opportunity"]["top1_lineage_id"] == "L2"
    assert rows[0]["birth_opportunity"]["first_birth_dominance_generation"] == 3
