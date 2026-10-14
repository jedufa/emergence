from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
for required in (str(REPO_ROOT), str(ENGINE_ROOT)):
    while required in sys.path:
        sys.path.remove(required)
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from simulation.chronicle.live_observer import (  # noqa: E402
    LIVE1_ARTIFACTS,
    LiveObserverConfig,
    LiveObserverError,
    catch_up_approved_root,
    catch_up_live_observer,
    discover_chronicle_runs,
    follow_live_observer,
)
from simulation.chronicle.storage import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from infra.webui.server_chronicle import (  # noqa: E402
    discover_live_runs,
    handle_get_request,
    live_feed,
    live_branch_replay,
    live_overview,
    live_replay,
)
from tests.scientific_chronicle.test_observer import _chronicle, _tree_bytes  # noqa: E402


def _config(*profiles: str) -> LiveObserverConfig:
    return LiveObserverConfig(audience_profiles=profiles or ("public_plain", "field_observer", "dry_humour", "documentary"))


def _web_config(scientific: Path, live: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runs_root=scientific,
        chronicle_roots=(scientific,),
        chronicle_live_roots=(live,),
        chronicle_scan_depth=8,
    )


def test_one_shot_restart_traceability_and_source_immutability(tmp_path: Path) -> None:
    chronicle = _chronicle(tmp_path / "source")
    before = _tree_bytes(chronicle)
    output = tmp_path / "live" / "one"
    status = catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config())
    events = read_jsonl(chronicle / "formal_events.jsonl")
    feed = read_jsonl(output / LIVE1_ARTIFACTS["feed"])
    cursor = json.loads((output / LIVE1_ARTIFACTS["cursor"]).read_text())
    assert status["status"] == "ready"
    assert len(feed) == len(events) * 4
    assert cursor["emitted_observer_ids"] == [row["observer_item_id"] for row in feed]
    assert len(cursor["emitted_observer_ids"]) == len(set(cursor["emitted_observer_ids"]))
    assert all(row["source_links"]["event_ids"] and row["source_links"]["entry_ids"] for row in feed)
    assert any(row["map_anchor"] for row in feed)
    assert any(row["global_fallback"] for row in feed)
    assert catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config())["status"] == "idle"
    assert before == _tree_bytes(chronicle)


def test_complete_prefix_append_partial_tail_and_replacement_detection(tmp_path: Path) -> None:
    chronicle = _chronicle(tmp_path / "source")
    events_path = chronicle / "formal_events.jsonl"
    entries_path = chronicle / "scientific_chronicle.jsonl"
    manifest_path = chronicle / "chronicle_manifest.json"
    full_events = read_jsonl(events_path)
    full_entries = read_jsonl(entries_path)
    full_manifest = json.loads(manifest_path.read_text())
    split = max(1, len(full_events) // 2)
    write_jsonl(events_path, full_events[:split])
    write_jsonl(entries_path, full_entries[:split])
    initial_manifest = {**full_manifest, "event_count": split}
    write_json(manifest_path, initial_manifest)
    output = tmp_path / "live" / "one"
    catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config("public_plain"))
    write_jsonl(events_path, full_events)
    write_jsonl(entries_path, full_entries)
    write_json(manifest_path, full_manifest)
    status = catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config("public_plain"))
    feed = read_jsonl(output / LIVE1_ARTIFACTS["feed"])
    assert status["source_event_count"] == len(full_events)
    assert len(feed) == len(full_events)
    assert len({row["observer_item_id"] for row in feed}) == len(feed)
    events_path.write_bytes(events_path.read_bytes() + b'{"incomplete":')
    status = catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config("public_plain"))
    assert status["status"] == "partial_tail_wait"
    write_jsonl(events_path, full_events)
    assert catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config("public_plain"))["status"] == "ready"
    changed = [dict(row) for row in full_events]
    changed[0]["measurements"] = {**dict(changed[0].get("measurements", {})), "live1_test_replacement": 1}
    write_jsonl(events_path, changed)
    with pytest.raises(LiveObserverError, match="replaced"):
        catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config("public_plain"))


def test_paths_discovery_and_controlled_follow(tmp_path: Path) -> None:
    _chronicle(tmp_path / "sources" / "a")
    assert discover_chronicle_runs(tmp_path / "sources") == [tmp_path / "sources" / "a"]
    results = catch_up_approved_root(tmp_path / "sources", tmp_path / "live", config=_config("public_plain"))
    assert results[0]["result"]["status"] == "ready"
    followed = follow_live_observer(
        tmp_path / "sources" / "a",
        tmp_path / "followed",
        approved_roots=(tmp_path / "sources",),
        config=_config("public_plain"),
        maximum_polls=2,
    )
    assert followed["poll_count"] == 2
    with pytest.raises(LiveObserverError, match="separate"):
        catch_up_live_observer(
            tmp_path / "sources" / "a",
            tmp_path / "sources" / "a" / "live",
            approved_roots=(tmp_path / "sources",),
            config=_config("public_plain"),
        )
    with pytest.raises(LiveObserverError, match="outside"):
        catch_up_live_observer(
            tmp_path / "sources" / "a",
            tmp_path / "elsewhere",
            approved_roots=(tmp_path / "not-approved",),
            config=_config("public_plain"),
        )


def test_get_only_transport_replay_and_hash_failure(tmp_path: Path) -> None:
    _chronicle(tmp_path / "runs" / "a")
    output = tmp_path / "live" / "a"
    catch_up_live_observer(tmp_path / "runs" / "a", output, approved_roots=(tmp_path / "runs",), config=_config())
    cfg = _web_config(tmp_path / "runs", tmp_path / "live")
    run = discover_live_runs(cfg)[0]
    key = run["live_key"]
    overview = live_overview(cfg, key)
    assert overview["simulation_control_available"] is False
    page = live_feed(cfg, key, {"profile": ["field_observer"], "limit": ["2"]})
    assert len(page["items"]) == 2 and page["unread"] > 0
    next_page = live_feed(cfg, key, {"profile": ["field_observer"], "after": [str(page["next_cursor"])], "limit": ["2"]})
    assert {row["observer_item_id"] for row in page["items"]}.isdisjoint(
        {row["observer_item_id"] for row in next_page["items"]}
    )
    replay = live_replay(cfg, key, {"profile": ["dry_humour"], "limit": ["200"]})
    assert replay["label"] == "Chronicle replay, not simulation replay."
    assert replay["episodes"]
    branch = live_branch_replay(
        cfg,
        {"run_a_live_key": [key], "run_b_live_key": [key], "profile": ["public_plain"], "generation": ["1"]},
    )
    assert branch["run_a"]["items"] == branch["run_b"]["items"]
    status, payload = handle_get_request(cfg=cfg, path="/api/chronicle/live/control", qs={"live_key": [key]})
    assert status == 404 and payload["error_code"] == "not_found"
    feed_path = output / LIVE1_ARTIFACTS["feed"]
    rows = read_jsonl(feed_path)
    rows[0]["text"] = "tampered"
    write_jsonl(feed_path, rows)
    status, payload = handle_get_request(cfg=cfg, path="/api/chronicle/live/feed", qs={"live_key": [key]})
    assert status == 422 and payload["error_code"] == "live_hash_mismatch"


def test_observer_watch_cli(tmp_path: Path) -> None:
    _chronicle(tmp_path / "source")
    output = tmp_path / "live"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "projects" / "emergence" / "tools" / "scientific_chronicle.py"),
            "observer-watch",
            str(tmp_path / "source"),
            str(output),
            "--approved-root",
            str(tmp_path),
            "--audience-profile",
            "documentary",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout)["status"] == "ready"
    assert all(row["audience_profile"] == "documentary" for row in read_jsonl(output / LIVE1_ARTIFACTS["feed"]))


def test_live_manifest_inventory_hashes_match(tmp_path: Path) -> None:
    _chronicle(tmp_path / "source")
    output = tmp_path / "live"
    catch_up_live_observer(tmp_path / "source", output, approved_roots=(tmp_path,), config=_config("public_plain"))
    manifest = json.loads((output / LIVE1_ARTIFACTS["manifest"]).read_text())
    assert {
        row["relative_path"]: row["sha256"] for row in manifest["artifacts"]
    } == {
        row["relative_path"]: sha256_file(output / row["relative_path"])
        for row in manifest["artifacts"]
    }


def test_chronicle_page_is_reduced_to_dialogue_controls_and_bubbles() -> None:
    html = (REPO_ROOT / "infra" / "webui" / "static" / "chronicle.html").read_text(encoding="utf-8")
    script = (REPO_ROOT / "infra" / "webui" / "static" / "chronicle.js").read_text(encoding="utf-8")
    assert 'id="simulationSelect"' in html
    assert 'id="audienceSelect"' in html
    assert 'id="dialogue"' in html
    assert "/api/chronicle/dialogue" in script
    assert 'id="liveGeneration"' not in html
    assert 'class="chronicle-tabs"' not in html
