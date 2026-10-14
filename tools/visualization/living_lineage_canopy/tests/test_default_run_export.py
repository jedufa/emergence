from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.visualization.living_lineage_canopy.python.default_run_export import (
    DefaultCanopyExportError,
    materialize_default_terminal_canopy,
    materialize_default_terminal_canopy_from_agents,
)
from tools.visualization.living_lineage_canopy.python.provenance import validate_provenance_binding
from tools.visualization.living_lineage_canopy.python.relay import validate_relay_manifest
from tools.visualization.living_lineage_canopy.python.schema import validate_playback


def _projection() -> dict:
    return {
        "metadata": {"generation": 4},
        "tensors": {
            "alive_mask": torch.tensor([True, False, True], dtype=torch.bool),
            "slot_to_agent_id": torch.tensor([10, -1, 14], dtype=torch.int64),
            "lineage_founder_id": torch.tensor([2, -1, 7], dtype=torch.int64),
        },
    }


def test_default_terminal_export_writes_verified_compact_canopy_bundle(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"checkpoint")

    result = materialize_default_terminal_canopy(
        run_root=tmp_path,
        run_id="run-42",
        terminal_generation=4,
        projection=_projection(),
        source_files=(source,),
    )

    relay_root = tmp_path / "lineage-playback-relay"
    manifest = validate_relay_manifest(relay_root / "relay_manifest.json")
    playback = json.loads((relay_root / manifest["playback_file_name"]).read_text(encoding="utf-8"))
    provenance = json.loads((relay_root / manifest["provenance_file_name"]).read_text(encoding="utf-8"))
    validate_playback(playback)
    assert validate_provenance_binding(playback, provenance)["ok"] is True
    assert result["status"] == "generated"
    assert result["agent_count"] == 2
    assert {agent["agent_id"] for agent in playback["agents"]} == {"A10", "A14"}
    assert playback["metadata"]["time_unit"] == "terminal_snapshot"


def test_default_terminal_export_is_idempotent_and_rejects_bad_identity_vectors(tmp_path: Path) -> None:
    first = materialize_default_terminal_canopy(
        run_root=tmp_path,
        run_id="run-42",
        terminal_generation=4,
        projection=_projection(),
    )
    second = materialize_default_terminal_canopy(
        run_root=tmp_path,
        run_id="run-42",
        terminal_generation=4,
        projection=_projection(),
    )
    assert first["status"] == "generated"
    assert second["status"] == "existing"

    with pytest.raises(DefaultCanopyExportError, match="inconsistent alive identity"):
        materialize_default_terminal_canopy(
            run_root=tmp_path / "invalid",
            run_id="invalid",
            terminal_generation=1,
            projection={
                "tensors": {
                    "alive_mask": [True],
                    "slot_to_agent_id": [-1],
                    "lineage_founder_id": [1],
                }
            },
        )


def test_python_runner_export_reads_only_live_agent_identities(tmp_path: Path) -> None:
    class Stats:
        founder_id = 3

    class Agent:
        def __init__(self, agent_id: int, founder_id: int) -> None:
            self.agent_id = agent_id
            self.stats = Stats()
            self.stats.founder_id = founder_id

    result = materialize_default_terminal_canopy_from_agents(
        run_root=tmp_path / "python-run-root",
        run_id="python-run",
        terminal_generation=9,
        agents=[Agent(8, 3), Agent(4, 2)],
    )

    payload = json.loads(
        (tmp_path / "python-run-root" / "lineage-playback-relay" / "playback_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "generated"
    assert payload["metadata"]["source_kind"] == "python_terminal_identity_snapshot"
    assert [row["agent_id"] for row in payload["agents"]] == ["A4", "A8"]


def test_default_export_explicit_operational_opt_out_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_CANOPY_DEFAULT_PLAYBACK", "false")
    result = materialize_default_terminal_canopy(run_root=tmp_path, run_id="disabled", terminal_generation=1, projection=_projection())
    policy = json.loads((tmp_path / "canopy_playback_policy.json").read_text(encoding="utf-8"))
    assert result["status"] == "disabled"
    assert policy["enabled"] is False
    assert not (tmp_path / "lineage-playback-relay").exists()


def test_default_export_marks_interrupted_source_without_claiming_completion(tmp_path: Path) -> None:
    materialize_default_terminal_canopy(run_root=tmp_path, run_id="interrupted", terminal_generation=2, projection=_projection(), source_state="source_interrupted")
    manifest = json.loads((tmp_path / "lineage-playback-relay" / "relay_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "lineage-playback-relay" / "relay_status.json").read_text(encoding="utf-8"))
    assert manifest["source_state"] == "source_interrupted"
    assert status["state"] == "source_interrupted"


def test_completed_resume_replaces_interrupted_derived_snapshot(tmp_path: Path) -> None:
    materialize_default_terminal_canopy(run_root=tmp_path, run_id="resume", terminal_generation=2, projection=_projection(), source_state="source_interrupted")
    resumed = _projection()
    resumed["tensors"]["slot_to_agent_id"] = torch.tensor([21, -1, 25], dtype=torch.int64)
    result = materialize_default_terminal_canopy(run_root=tmp_path, run_id="resume", terminal_generation=4, projection=resumed)
    relay = tmp_path / "lineage-playback-relay"
    manifest = json.loads((relay / "relay_manifest.json").read_text(encoding="utf-8"))
    playback = json.loads((relay / "playback_snapshot.json").read_text(encoding="utf-8"))
    assert result["status"] == "generated"
    assert manifest["source_state"] == "source_complete"
    assert {agent["agent_id"] for agent in playback["agents"]} == {"A21", "A25"}
