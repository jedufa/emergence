from __future__ import annotations

import json

import pytest

from engine.campaigns.wrc_atlas_resume import AtlasResumeError, initialize_campaign, publish_world_complete, scan_campaign
from engine.wrc.reducer import Observation, SourceLocator, WorldIdentity, WorldReducer, save_checkpoint


def _cell(world_id: str, seed: int) -> dict[str, object]:
    return {"world_id": world_id, "parameter_id": "p", "parameter_value": 1, "seed": seed, "expected_rows": 1, "route_identity": {"id": "cpu"}}


def _identity() -> dict[str, object]:
    return {"atlas": "a", "source": "s", "configuration": "c", "route": "cpu"}


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


def test_duplicate_plans_and_unplanned_worlds_are_rejected(tmp_path) -> None:
    with pytest.raises(AtlasResumeError, match="duplicate"):
        initialize_campaign(tmp_path / "duplicate", identity=_identity(), cells=[_cell("w", 1), _cell("w", 2)])
    root = tmp_path / "campaign"
    cell = _cell("w", 1)
    initialize_campaign(root, identity=_identity(), cells=[cell])
    reducer = tmp_path / "reducer.json"
    reducer.write_text("{}")
    with pytest.raises(AtlasResumeError, match="not planned"):
        publish_world_complete(root, world_id="other", reducer_state=reducer)


def test_marker_is_atomic_and_reducer_hash_is_rechecked(tmp_path) -> None:
    root = tmp_path / "campaign"
    cell = _cell("w", 1)
    initialize_campaign(root, identity=_identity(), cells=[cell])
    world = root / "worlds" / "w"
    world.mkdir(parents=True)
    (world / "metrics.csv").write_text("tick,signal\n1,1\n")
    reducer = world / "reducer.json"
    _finalized_reducer(reducer, cell)
    marker = publish_world_complete(root, world_id="w", reducer_state=reducer)
    assert marker["world_complete_sha256"]
    assert (world / "world_complete.json").is_file()
    reducer.write_text('{"state":"tampered"}')
    scan = scan_campaign(root)
    assert scan["counts"] == {"complete": 0, "invalid": 1, "missing": 0, "rerun": 1}
    broken = json.loads((world / "world_complete.json").read_text())
    broken["cell"]["seed"] = 2
    (world / "world_complete.json").write_text(json.dumps(broken))
    assert scan_campaign(root)["rows"][0]["reason"] == "invalid_marker"


def test_reducer_completion_requires_final_identity_bound_checkpoint(tmp_path) -> None:
    root = tmp_path / "campaign"
    cell = _cell("w", 1)
    initialize_campaign(root, identity=_identity(), cells=[cell])
    world = root / "worlds" / "w"
    world.mkdir(parents=True)
    (world / "metrics.csv").write_text("tick,signal\n1,1\n")
    reducer = world / "reducer.json"

    reducer.write_text("{}")
    with pytest.raises(AtlasResumeError, match="finalized checkpoint"):
        publish_world_complete(root, world_id="w", reducer_state=reducer)

    wrong_cell = _cell("other", 1)
    _finalized_reducer(reducer, wrong_cell)
    with pytest.raises(AtlasResumeError, match="finalized checkpoint"):
        publish_world_complete(root, world_id="w", reducer_state=reducer)
