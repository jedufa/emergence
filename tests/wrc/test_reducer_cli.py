from __future__ import annotations

import csv
import json
from pathlib import Path

from engine.wrc.reducer import WorldIdentity, WrcLiveReducer
from engine.wrc.reducer.canonical import canonical_sha256
from tools.scripts.eiel_wrc_reduce import main


def test_extract_defaults_to_indexed_metrics_not_full_wide_rows(
    tmp_path: Path,
    capsys,
) -> None:
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "signal", "unrelated", "large_json"])
        writer.writerow([0, 1.5, 99, json.dumps([{"x": "a" * 1000}])])

    index = tmp_path / "raw_row_index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_uri", "source_row_number", "metrics"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_uri": str(metrics.resolve()),
                "source_row_number": 2,
                "metrics": json.dumps({"signal": ["stage1:first"]}),
            }
        )

    output = tmp_path / "rows.jsonl"
    assert main(["extract", "--index", str(index), "--output", str(output)]) == 0
    capsys.readouterr()
    row = json.loads(output.read_text().splitlines()[0])
    assert row["selected_metrics"] == ["signal"]
    assert row["row"] == {"generation": "0", "signal": "1.5"}
    assert "large_json" not in row["row"]


def test_status_reports_bounded_live_and_retention_state(tmp_path: Path, capsys) -> None:
    reducer = WrcLiveReducer(
        WorldIdentity(
            world_id="status-world",
            parameter_id="fixture.parameter",
            parameter_value=False,
            seed=101,
            ticks_per_observation=80,
        )
    )
    reducer.ingest(sequence=0, tick=80, metrics={"alive": 64.0})
    snapshot = tmp_path / "state.json"
    reducer.write_snapshot(snapshot)
    manifest = tmp_path / "retention_manifest.json"
    manifest_value = {
        "complete": True,
        "mode": "retain_compact",
        "source_bytes": 1000,
        "retained_bytes": 400,
        "source_count": 10,
        "retained_count": 4,
        "raw_prune_complete": False,
    }
    manifest_value["manifest_sha256"] = canonical_sha256(manifest_value)
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

    assert main(
        [
            "status",
            "--snapshot",
            str(snapshot),
            "--retention-manifest",
            str(manifest),
        ]
    ) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ok"] is True
    assert status["enabled"] is True
    assert status["finalized"] is False
    assert status["retention_mode"] == "retain_all"
    assert status["retention_outcome"]["status"] == "complete"
    assert status["source_bytes"] == 1000
    assert status["retained_bytes"] == 400
    assert "observation_count" not in status


def test_status_rejects_incomplete_retention_manifest(tmp_path: Path, capsys) -> None:
    reducer = WrcLiveReducer(
        WorldIdentity(
            world_id="status-world",
            parameter_id="fixture.parameter",
            parameter_value=False,
            seed=101,
            ticks_per_observation=80,
        )
    )
    snapshot = tmp_path / "state.json"
    reducer.write_snapshot(snapshot)
    manifest = tmp_path / "retention_manifest.json"
    manifest.write_text(json.dumps({"complete": False}), encoding="utf-8")

    assert main(
        ["status", "--snapshot", str(snapshot), "--retention-manifest", str(manifest)]
    ) == 2
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert "not complete" in error["error"]
