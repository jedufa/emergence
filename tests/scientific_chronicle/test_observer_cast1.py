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

from simulation.chronicle.cast import (  # noqa: E402
    CAST_ARTIFACTS,
    build_name_registries,
    build_retrospective_cast,
    observer_cast_regenerate,
    reconstruct_live_cast,
)
from simulation.chronicle.observer import ObserverConfig, write_observer_outputs  # noqa: E402
from simulation.chronicle.storage import read_json, read_jsonl, sha256_file  # noqa: E402
from tests.scientific_chronicle.test_observer import _chronicle, _tree_bytes  # noqa: E402


def _event(event_id: str, agent_id: int, lineage_id: str, generation: int, event_type: str = "notable_birth") -> dict[str, object]:
    return {
        "event_id": event_id,
        "generation": generation,
        "step": generation * 10,
        "event_type": event_type,
        "event_category": "lifecycle",
        "agent_ids": [agent_id],
        "lineage_ids": [lineage_id],
        "historical_importance": 4,
        "scientific_importance": 3,
        "public_interest": 2,
        "visual_interest": 1,
        "live_priority": "high",
    }


def test_agent_and_lineage_names_are_deterministic_unique_and_branch_stable() -> None:
    events_a = [_event("evt_a", 29, "L00000008", 0), _event("evt_b", 30, "L00000008", 1)]
    events_b = [dict(row, branch_id="alternate") for row in events_a]
    agents_a, lineages_a = build_name_registries(events_a)
    agents_b, lineages_b = build_name_registries(events_b)

    assert agents_a == agents_b
    assert lineages_a == lineages_b
    names = [row["display_name"] for row in agents_a["items"]]
    assert len(names) == len(set(names))
    assert all("agent_id" in row and "display_name" in row for row in agents_a["items"])
    assert lineages_a["items"][0]["lineage_id"] == "L00000008"


def test_retrospective_cast_ranking_retains_components_and_roles() -> None:
    events = [
        _event("evt_a", 29, "L00000008", 0),
        _event("evt_b", 29, "L00000008", 2, "mutation_observation"),
        _event("evt_c", 29, "L00000008", 4, "notable_death"),
        _event("evt_d", 30, "L00000009", 1),
    ]
    agents, _lineages = build_name_registries(events)
    cast = build_retrospective_cast(
        events,
        [{"episode_id": "epi_a", "event_ids": ["evt_a", "evt_b", "evt_c"]}],
        agent_registry=agents,
        max_agents=1,
    )

    assert cast["score_version"] == "retrospective_cast_score_v1"
    assert cast["featured_agent_ids"] == [29]
    top = cast["items"][0]
    assert top["agent_id"] == 29
    assert top["score_components"]["mutation_occurrence_links"] == 1
    assert top["score_components"]["source_episode_ids"] == ["epi_a"]
    assert "mutation_record_carrier" in top["roles"]
    assert "Cast score is presentation ranking" in top["role_limitations"][1]


def test_live_cast_reconstruction_uses_prefix_only_and_promotes_stably() -> None:
    events = [_event(f"evt_{i}", 7, "L00000007", i) for i in range(4)]
    agents, _lineages = build_name_registries(events)
    rows = reconstruct_live_cast(events, agent_registry=agents, max_featured=1)

    assert [row["score_components"]["prefix_notable_event_count"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["state"] in {"noticed", "watchlist"}
    assert rows[-1]["state"] in {"featured", "currently_central"}
    assert all("no future events inspected" in row["promotion_reason"] for row in rows)
    assert all(row["future_leakage_guard"].startswith("computed in chronological order") for row in rows)


def test_cast_regeneration_writes_required_outputs_and_preserves_source(tmp_path: Path) -> None:
    run = tmp_path / "run"
    chronicle = _chronicle(run)
    before = _tree_bytes(chronicle)
    observer_dir = tmp_path / "observer"
    write_observer_outputs(
        run,
        observer_dir,
        config=ObserverConfig(audience_profile="cosmic_creator", storage_profile="observer_full", acknowledge_large_output=True),
    )

    result_a = observer_cast_regenerate(run, tmp_path / "cast-a", observer_dir=observer_dir)
    result_b = observer_cast_regenerate(run, tmp_path / "cast-b", observer_dir=observer_dir)

    assert before == _tree_bytes(chronicle)
    assert result_a["source_immutable"] is True
    for filename in CAST_ARTIFACTS.values():
        assert (tmp_path / "cast-a" / filename).exists(), filename
    assert read_json(tmp_path / "cast-a" / "cast_manifest.json")["source_immutable"] is True
    assert read_jsonl(tmp_path / "cast-a" / "observer_cast_feed.jsonl")
    assert read_jsonl(tmp_path / "cast-a" / "live_cast_reconstruction.jsonl")
    dialogue = (tmp_path / "cast-a" / "observer_cast_dialogue.txt").read_text(encoding="utf-8").splitlines()
    assert dialogue
    assert all(line.startswith("[2000-01-") and "] [Creator] " in line for line in dialogue)
    assert not any(line.startswith("#") or "source" in line.lower() for line in dialogue[:10])
    assert sha256_file(tmp_path / "cast-a" / "observer_cast_feed.jsonl") == sha256_file(tmp_path / "cast-b" / "observer_cast_feed.jsonl")
    assert sha256_file(tmp_path / "cast-a" / "observer_cast_dialogue.txt") == sha256_file(tmp_path / "cast-b" / "observer_cast_dialogue.txt")
    assert result_a["featured_agent_count"] == result_b["featured_agent_count"]


def test_cast_cli_writes_verified_package(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _chronicle(run)
    out = tmp_path / "cli-cast"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "projects" / "emergence" / "tools" / "scientific_chronicle.py"),
        "observer-cast-regenerate",
        str(run),
        str(out),
        "--max-featured-agents",
        "3",
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)

    assert payload["source_immutable"] is True
    assert payload["output_dir"] == str(out.resolve())
    assert (out / "agent_name_registry.json").exists()
    assert read_json(out / "observer_cast_index.json")["future_leakage_audit"]["live_reconstruction_uses_prefix_only"] is True
