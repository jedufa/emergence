from __future__ import annotations

from pathlib import Path

from simulation.mem_max_pre_cue_checkpoint import (
    execute_lineage_producing_generation_resume,
    execute_lineage_resume_repeat_parity,
)
from tests.evolvability.wdl.test_mem_max_23_pre_cue_checkpoint import _make_sim


def _snapshot(tmp_path: Path):
    sim = _make_sim(tmp_path)
    sim.env.patch_world_step(0)
    return sim.mem_max_pre_cue_checkpoint_recorder.snapshots[0]


def test_mem_max_25_resume_reaches_generation_end_and_records_real_lineage(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    result = execute_lineage_producing_generation_resume(snap["snapshot_path"], label="test_resume")

    assert result["generation_end_reached"] is True
    assert result["production_resume_api"] == "runner.run(start_gen=0, skip_init=True)"
    assert result["birth_event_count"] >= 1
    assert result["policy_stage_record_count"] >= 1
    assert result["policy_stage_hashes_match"] is True
    assert result["common_state_pair_count"] == 4
    assert result["common_state_all_available"] is True


def test_mem_max_25_lineage_resume_repeat_parity(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path)
    parity = execute_lineage_resume_repeat_parity(snap["snapshot_path"])

    assert parity["passed"] is True
    assert parity["left"]["birth_event_count"] == parity["right"]["birth_event_count"]
    assert parity["left"]["policy_stage_reconciliation"] == parity["right"]["policy_stage_reconciliation"]
