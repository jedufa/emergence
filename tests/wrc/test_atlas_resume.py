from __future__ import annotations

import json

import pytest

from engine.campaigns.wrc_atlas_resume import (
    AtlasResumeError,
    append_resume_history,
    initialize_campaign,
    publish_world_complete,
    scan_campaign,
)
from engine.wrc.reducer.canonical import canonical_sha256
from engine.wrc.reducer import Observation, SourceLocator, WorldIdentity, WorldReducer, save_checkpoint
from tools.scripts.eiel_wrc_atlas import main


def _identity() -> dict[str, object]:
    return {"atlas_sha256": "atlas", "plan_sha256": "plan", "repository_commit": "commit", "route": {"id": "cpu-test"}}


def _cell() -> dict[str, object]:
    return {"world_id": "p011-seed-101", "parameter_id": "env_cfg.move_cost", "parameter_value": 1.0, "seed": 101, "expected_rows": 2, "route_identity": {"route": "cpu-test"}}


def _finalized_reducer(path, cell: dict[str, object]) -> None:
    reducer = WorldReducer(
        WorldIdentity(
            str(cell["world_id"]),
            str(cell["parameter_id"]),
            cell["parameter_value"],
            int(cell["seed"]),
        )
    )
    reducer.ingest(
        Observation(
            sequence=0,
            tick=1,
            metrics={"signal": 1.0},
            source=SourceLocator("fixture", "fixture://atlas", row_number=1),
        )
    )
    result = reducer.finalize()
    save_checkpoint(path, reducer, finalized=True, finalized_result=result)


def _outputs(root, cell: dict[str, object] | None = None):
    cell = _cell() if cell is None else cell
    world = root / "worlds" / "p011-seed-101"
    world.mkdir(parents=True)
    metrics = world / "metrics.csv"
    metrics.write_text("tick,alive,signal\n1,4,1.0\n2,3,2.0\n")
    reducer = world / "reducer_state.json"
    _finalized_reducer(reducer, cell)
    return metrics, reducer


def test_resume_requires_hash_bound_identity_and_stable_scan(tmp_path) -> None:
    root = tmp_path / "campaign"
    initialize_campaign(root, identity=_identity(), cells=[_cell()])
    metrics, reducer = _outputs(root)
    publish_world_complete(root, world_id="p011-seed-101", metrics=metrics, reducer_state=reducer)
    first = scan_campaign(root)
    second = scan_campaign(root)
    assert first == second
    assert first["counts"] == {"complete": 1, "invalid": 0, "missing": 0, "rerun": 0}
    with pytest.raises(AtlasResumeError, match="identity differs"):
        initialize_campaign(root, identity={**_identity(), "plan_sha256": "other"}, cells=[_cell()])


def test_corruption_and_header_only_metrics_rerun_fail_closed(tmp_path) -> None:
    root = tmp_path / "campaign"
    initialize_campaign(root, identity=_identity(), cells=[_cell()])
    metrics, reducer = _outputs(root)
    publish_world_complete(root, world_id="p011-seed-101", metrics=metrics, reducer_state=reducer)
    metrics.write_text("tick,alive,signal\n")
    scan = scan_campaign(root)
    assert scan["rows"] == [{"world_id": "p011-seed-101", "status": "invalid", "reason": "outputs_invalid"}]
    assert scan["counts"]["rerun"] == 1
    with pytest.raises(AtlasResumeError, match="not reusable"):
        publish_world_complete(root, world_id="p011-seed-101", metrics=metrics, reducer_state=reducer)


def test_retention_proof_allows_missing_raw_metrics_only_when_hash_valid(tmp_path) -> None:
    root = tmp_path / "campaign"
    cell = _cell()
    initialize_campaign(root, identity=_identity(), cells=[cell])
    world = root / "worlds" / "p011-seed-101"
    world.mkdir(parents=True)
    reducer = world / "reducer_state.json"
    _finalized_reducer(reducer, cell)
    retained = {"complete": True, "world_identity_sha256": canonical_sha256(cell)}
    retained["manifest_sha256"] = canonical_sha256(retained)
    retention = world / "retention_manifest.json"
    retention.write_text(json.dumps(retained))
    publish_world_complete(root, world_id="p011-seed-101", reducer_state=reducer, retention_manifest=retention)
    assert scan_campaign(root)["rows"][0]["status"] == "complete"
    retained["complete"] = False
    retention.write_text(json.dumps(retained))
    assert scan_campaign(root)["rows"][0]["status"] == "invalid"


def test_missing_marker_never_becomes_complete_from_directory_or_history(tmp_path) -> None:
    root = tmp_path / "campaign"
    initialize_campaign(root, identity=_identity(), cells=[_cell()])
    _outputs(root)
    scan = scan_campaign(root)
    append_resume_history(root, scan)
    assert scan["rows"][0]["status"] == "missing"
    assert (root / "resume_history.jsonl").read_text().count("status_sha256") == 1


def test_operational_scan_command_is_status_only(tmp_path, capsys) -> None:
    root = tmp_path / "campaign"
    initialize_campaign(root, identity=_identity(), cells=[_cell()])
    assert main(["scan", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["missing"] == 1
    assert result["rows"][0]["status"] == "missing"
