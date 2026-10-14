from __future__ import annotations

import json
from pathlib import Path

from engine.wrc.reducer import WorldIdentity, WrcLiveReducer
from tools.scripts.eiel_wrc_reduce import main


def test_operational_status_without_retention_manifest_is_explicitly_unmaterialized(
    tmp_path: Path, capsys
) -> None:
    reducer = WrcLiveReducer(
        WorldIdentity(
            world_id="status-only-world",
            parameter_id="fixture.parameter",
            parameter_value=False,
            seed=101,
            ticks_per_observation=80,
        )
    )
    snapshot = tmp_path / "reducer_state.json"
    reducer.write_snapshot(snapshot)

    assert main(["status", "--snapshot", str(snapshot)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status == {
        "enabled": True,
        "finalized": False,
        "ok": True,
        "reducer_version": "eiel_wrc_reducer_v1_1",
        "retained_bytes": None,
        "retained_count": None,
        "retention_mode": "retain_all",
        "retention_outcome": "not_materialized",
        "schema_version": "eiel_wrc_live_integration_v1",
        "source_bytes": None,
        "source_count": None,
    }
