from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
for required in (str(REPO_ROOT), str(ENGINE_ROOT)):
    while required in sys.path:
        sys.path.remove(required)
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from infra.webui.server_chronicle import handle_get_request  # noqa: E402
from simulation.chronicle.documentary import DOCUMENTARY_ARTIFACTS, verify_documentary_script, write_documentary_outputs  # noqa: E402
from simulation.chronicle.storage import read_json, read_jsonl, sha256_file  # noqa: E402
from tests.scientific_chronicle.test_observer import _chronicle, _tree_bytes, _web_config  # noqa: E402


def test_documentary_generation_is_deterministic_source_linked_and_read_only(tmp_path: Path) -> None:
    run = tmp_path / "run"
    chronicle = _chronicle(run)
    before = _tree_bytes(chronicle)

    result_a = write_documentary_outputs(run, run / "documentary")
    bytes_a = _tree_bytes(run / "documentary")
    result_b = write_documentary_outputs(run, tmp_path / "documentary-b")

    assert before == _tree_bytes(chronicle)
    assert result_a["source_immutable"] is True
    assert result_a["verifier_accepted"] is True
    assert result_a["word_count"] >= 1300
    assert sha256_file(run / "documentary" / "documentary_script.json") == sha256_file(tmp_path / "documentary-b" / "documentary_script.json")
    assert bytes_a["documentary_transcript.md"].startswith(b"# EIEL Two-Host Documentary Transcript")
    assert b"Field Scientist" in bytes_a["documentary_transcript.html"]
    assert b"Cosmic Creator" in bytes_a["documentary_audio_script.txt"]
    for relative in DOCUMENTARY_ARTIFACTS.values():
        assert (run / "documentary" / relative).exists(), relative

    script = read_json(run / "documentary" / "documentary_script.json")
    assert script["schema_version"] == "eiel_two_host_documentary_script_v1"
    assert all(line["source_event_ids"] for line in script["utterances"] if line["line_type"] != "transition")
    assert {line["speaker"] for line in script["utterances"]} == {"field_scientist", "cosmic_creator"}
    assert read_json(run / "documentary" / "documentary_verifier_report.json")["unsupported_count"] == 0


def test_documentary_verifier_rejects_unsupported_and_unsafe_imported_line(tmp_path: Path) -> None:
    run = tmp_path / "run"
    chronicle = _chronicle(run)
    events = read_jsonl(chronicle / "formal_events.jsonl")
    entries = read_jsonl(chronicle / "scientific_chronicle.jsonl")
    episodes = read_json(chronicle / "episodes.json")
    bad_script = {
        "schema_version": "eiel_two_host_documentary_script_v1",
        "utterances": [
            {
                "utterance_id": "podutt_bad",
                "sequence": 1,
                "speaker": "cosmic_creator",
                "line_type": "fact",
                "text": "Agent 99 planned the whole collapse because a mutation caused intelligence.",
                "source_event_ids": [],
                "source_entry_ids": [],
                "source_episode_ids": [],
                "confidence": "measured",
                "causality": "not_assessed",
            }
        ],
    }

    report = verify_documentary_script(bad_script, events=events, entries=entries, episodes=episodes)

    assert report["accepted"] is False
    assert report["unsupported_count"] == 1 or report["unsafe_wording_count"] == 1
    assert report["findings"][0]["status"] in {"unsupported", "unsafe_wording", "causal_inflation"}


def test_documentary_cli_and_explorer_endpoint(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "run-a"
    _chronicle(run, run_id="pod1-run")
    out = run / "documentary"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "projects" / "emergence" / "tools" / "scientific_chronicle.py"),
        "documentary-generate",
        str(run),
        str(out),
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)

    assert payload["verifier_accepted"] is True
    cfg = _web_config(tmp_path / "runs")
    status, runs = handle_get_request(cfg=cfg, path="/api/chronicle/runs", qs={})
    assert status == 200
    ready = [row for row in runs["runs"] if row["status"] == "ready"]
    assert ready[0]["documentary_available"] is True
    status, doc = handle_get_request(cfg=cfg, path="/api/chronicle/documentary", qs={"run_key": [ready[0]["run_key"]]})
    assert status == 200
    assert doc["schema_version"] == "eiel_two_host_documentary_explorer_v1"
    assert doc["verifier"]["accepted"] is True
    assert doc["script_summary"]["word_count"] >= 1300
    assert "Field Scientist" in doc["transcript"]
