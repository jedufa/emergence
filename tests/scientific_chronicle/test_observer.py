from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
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

import simulation.chronicle.observer as observer_module
from simulation.chronicle.observer import (
    OBSERVER_ARTIFACTS,
    ObserverConfig,
    ObserverError,
    _bounded_spoken_lines,
    _card_summary,
    load_scientific_source,
    observer_storage_preflight,
    render_observer_event,
    select_observer_events,
    select_observer_episodes,
    validate_observer_wording,
    write_observer_outputs,
)
from simulation.chronicle.observer_templates import AUDIENCE_PROFILES, FORBIDDEN_CLAIM_PATTERNS
from simulation.chronicle.recorder import ScientificChronicleRecorder
from simulation.chronicle.storage import read_jsonl, sha256_file, write_json, write_jsonl
from infra.webui.server_chronicle import (
    discover_runs,
    handle_get_request,
    observer_feed,
    observer_overview,
    observer_story,
    observer_stories,
)


@dataclass
class _Stats:
    lineage_id: str
    founder_id: int


@dataclass
class _Agent:
    agent_id: int
    stats: _Stats


def _agent(agent_id: int, lineage: str) -> _Agent:
    return _Agent(agent_id=agent_id, stats=_Stats(lineage_id=lineage, founder_id=agent_id))


def _chronicle(run_root: Path, *, run_id: str = "observer-run") -> Path:
    cfg = SimpleNamespace(
        chronicle_enable=True,
        chronicle_profile="scientific_full",
        chronicle_run_id=run_id,
        chronicle_max_stored_events_per_generation=128,
        chronicle_max_events_per_category_per_generation=64,
        chronicle_minimum_persistence_generations=1,
        chronicle_subject_cooldown_generations=0,
        chronicle_region_cooldown_generations=0,
        campaign_id="observer-campaign",
        branch_id="observer-branch",
        generations=3,
        allow_large_output_profile=True,
    )
    recorder = ScientificChronicleRecorder(SimpleNamespace(sim_cfg=cfg, run_dir=str(run_root)))
    recorder.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(2, "L2"), _agent(3, "L2")],
        deaths=0,
        births=1,
        failed_births=0,
        cohort_size=3,
        patch_metrics={"patch_food_depleted_frac": 0.0},
        birth_contexts=[{"agent_id": 3, "biological_lineage_id": "L2", "y": 2, "x": 3}],
    )
    recorder.observe_generation(
        generation=1,
        step=19,
        alive_agents=[_agent(1, "L1")],
        deaths=2,
        births=0,
        failed_births=1,
        cohort_size=3,
        patch_metrics={
            "patch_food_depleted_frac": 0.45,
            "patch_density_hotspot_frac": 0.4,
            "occupied_patch_waste_mean": 0.5,
            "patch_food_regrowth_suppressed_total": 2,
            "jungle_predator_events": 3,
            "jungle_predator_damage_total": 0.75,
            "jungle_predator_deaths": 1,
            "wdl_delayed_hazard_event_count": 2,
            "wdl_delayed_hazard_agents_damaged": 1,
            "wdl_delayed_hazard_damage_total": 0.2,
        },
        death_contexts=[
            {"agent_id": 2, "biological_lineage_id": "L2", "y": 2, "x": 3},
            {"agent_id": 3, "biological_lineage_id": "L2", "y": 2, "x": 3},
        ],
        predator_contexts=[{"agent_id": 2, "biological_lineage_id": "L2", "y": 2, "x": 3}],
        wdl_contexts=[{"phase": "active"}],
    )
    recorder.observe_generation(
        generation=2,
        step=29,
        alive_agents=[_agent(1, "L1"), _agent(4, "L3")],
        deaths=0,
        births=1,
        failed_births=0,
        cohort_size=3,
        patch_metrics={
            "patch_food_depleted_frac": 0.05,
            "patch_density_hotspot_frac": 0.0,
            "occupied_patch_waste_mean": 0.0,
            "patch_food_recovery_positive_total": 2,
        },
        birth_contexts=[{"agent_id": 4, "biological_lineage_id": "L3", "y": 1, "x": 1}],
    )
    recorder.finalize()
    return recorder.root


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _gen1(root: Path, chronicle_root: Path) -> Path:
    event = next(
        row
        for row in read_jsonl(chronicle_root / "formal_events.jsonl")
        if 2 in {int(value) for value in row.get("agent_ids", []) or []}
    )
    mutation_id = "rmut_observer_test"
    write_json(
        root / "chronicle_companion_index.json",
        {
            "schema_version": "eiel_gen1_chronicle_companion_index_v1",
            "runs": {
                "observer-run": {
                    "agent_mutations": {"2": [mutation_id]},
                    "lineage_mutations": {"L2": [mutation_id]},
                    "chronicle_event_mutations": {event["event_id"]: [mutation_id]},
                }
            },
        },
    )
    write_jsonl(
        root / "mutation_event_inventory.jsonl",
        [
            {
                "schema_version": "eiel_gen1_mutation_event_inventory_v1",
                "mutation_event_id": mutation_id,
                "run_id": "observer-run",
                "child_agent_id": "2",
                "lineage_id": "L2",
                "changed_fields": ["/policy/test"],
                "evidence_strength": "observed",
                "source_completeness": "bounded_test_occurrence_only",
                "causal_status": "not_assessed",
            }
        ],
    )
    return root


def _web_config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(runs_root=root, chronicle_roots=(root,), chronicle_scan_depth=6)


def test_all_profiles_generate_separate_source_linked_outputs(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "run")
    source_before = _tree_bytes(chronicle_root)
    for profile in AUDIENCE_PROFILES:
        output = tmp_path / "observer" / profile
        manifest = write_observer_outputs(
            tmp_path / "run",
            output,
            config=ObserverConfig(
                audience_profile=profile,
                storage_profile="observer_notable",
                repeated_theme_cooldown=0,
            ),
        )
        assert set(path.name for path in output.iterdir()) == set(OBSERVER_ARTIFACTS.values())
        assert manifest["audience_profile"] == profile
        assert manifest["presentation_only"] is True
        assert manifest["scientific_source_immutability"] is True
        assert manifest["rng_immutability"] is True
        assert manifest["wall_clock_fields_emitted"] is False
        feed = read_jsonl(output / "observer_feed.jsonl")
        assert feed
        assert all(row["source_links"]["event_ids"] for row in feed)
        assert all(row["source_links"]["entry_ids"] for row in feed)
        if profile in {"dry_humour", "cosmic_creator"}:
            assert all(row["humour_line"] for row in feed)
        else:
            assert all(row["humour_line"] is None for row in feed)
    assert source_before == _tree_bytes(chronicle_root)


def test_byte_determinism_template_stability_and_rng_immutability(tmp_path: Path) -> None:
    _chronicle(tmp_path / "run")
    source = load_scientific_source(tmp_path / "run")
    config = ObserverConfig(audience_profile="dry_humour", repeated_theme_cooldown=0)
    selected = select_observer_events(source, config)
    first_render = render_observer_event(
        selected[0][0],
        selected[0][1],
        audience_profile="dry_humour",
        episode_family=selected[0][2],
        source_completeness="complete",
    )
    second_render = render_observer_event(
        selected[0][0],
        selected[0][1],
        audience_profile="dry_humour",
        episode_family=selected[0][2],
        source_completeness="complete",
    )
    assert first_render == second_render
    python_rng_before = random.getstate()
    write_observer_outputs(tmp_path / "run", tmp_path / "a", config=config)
    write_observer_outputs(tmp_path / "run", tmp_path / "b", config=config)
    assert python_rng_before == random.getstate()
    assert _tree_bytes(tmp_path / "a") == _tree_bytes(tmp_path / "b")


def test_filters_caps_cooldown_and_independent_dimensions(tmp_path: Path) -> None:
    _chronicle(tmp_path / "run")
    source = load_scientific_source(tmp_path / "run")
    selected = select_observer_events(
        source,
        ObserverConfig(
            audience_profile="field_observer",
            minimum_public_interest=2,
            minimum_visual_interest=2,
            minimum_live_priority="high",
            max_messages_per_generation=1,
            repeated_theme_cooldown=2,
        ),
    )
    assert selected
    counts = {}
    for event, _entry, _family in selected:
        counts[int(event["generation"])] = counts.get(int(event["generation"]), 0) + 1
        assert int(event["public_interest"]) >= 2
        assert int(event["visual_interest"]) >= 2
        assert event["live_priority"] in {"critical", "high"}
    assert max(counts.values()) == 1
    manifest = write_observer_outputs(
        tmp_path / "run",
        tmp_path / "filtered",
        config=ObserverConfig(
            categories=("hazard",),
            event_types=("delayed_hazard_cycle",),
            repeated_theme_cooldown=0,
        ),
    )
    feed = read_jsonl(tmp_path / "filtered" / "observer_feed.jsonl")
    assert feed
    assert {row["event_category"] for row in feed} == {"hazard"}
    assert {row["event_type"] for row in feed} == {"delayed_hazard_cycle"}
    assert manifest["independent_scientific_dimensions_preserved"] is True
    empty_subject = ObserverConfig(agent_ids=(999999,), repeated_theme_cooldown=0)
    assert select_observer_events(source, empty_subject) == []
    assert select_observer_episodes(source, empty_subject) == []


@pytest.mark.parametrize("phrase", FORBIDDEN_CLAIM_PATTERNS)
def test_wording_contract_rejects_unsupported_claims(phrase: str) -> None:
    text = phrase.replace(".*", " this record ")
    with pytest.raises(ObserverError, match="claims contract"):
        validate_observer_wording(f"The observer {text}.")


def test_generated_text_is_safe_and_technical_events_remain_separate(tmp_path: Path) -> None:
    _chronicle(tmp_path / "run")
    write_observer_outputs(
        tmp_path / "run",
        tmp_path / "observer",
        config=ObserverConfig(audience_profile="dry_humour", repeated_theme_cooldown=0),
    )
    for filename in ("observer_feed.jsonl", "observer_episodes.jsonl"):
        for row in read_jsonl(tmp_path / "observer" / filename):
            for name in ("headline", "text", "humour_line"):
                validate_observer_wording(str(row.get(name) or ""))
    source = load_scientific_source(tmp_path / "run")
    technical_event = dict(source.events[0])
    technical_event.update(event_type="chronicle_integrity", event_category="integrity")
    technical = render_observer_event(
        technical_event,
        source.entries[0],
        audience_profile="dry_humour",
        source_completeness="partial",
    )
    assert technical["technical_class"] is True
    assert "technical Chronicle limitation" in technical["plain_factual_text"]
    assert technical["event_category"] == "integrity"
    lineage_event = dict(next(event for event in source.events if event["event_type"] == "lineage_appearance"))
    lineage_event["measurements"] = {**lineage_event["measurements"], "alive_count": 1}
    lineage_text = render_observer_event(
        lineage_event,
        source.entries[source.events.index(next(event for event in source.events if event["event_type"] == "lineage_appearance"))],
        audience_profile="public_plain",
        source_completeness="complete",
    )["plain_factual_text"]
    assert "1 living member and" in lineage_text
    assert "1 living members" not in lineage_text


def test_story_cards_audio_history_and_preflight(tmp_path: Path) -> None:
    _chronicle(tmp_path / "run")
    write_observer_outputs(
        tmp_path / "run",
        tmp_path / "observer",
        config=ObserverConfig(
            audience_profile="documentary",
            repeated_theme_cooldown=0,
            maximum_audio_sentence_length=80,
        ),
    )
    assert read_jsonl(tmp_path / "observer" / "observer_agents.jsonl")
    assert read_jsonl(tmp_path / "observer" / "observer_lineages.jsonl")
    regions = read_jsonl(tmp_path / "observer" / "observer_regions.jsonl")
    assert regions
    assert all(row["source_event_ids"] for row in regions)
    assert all(
        set(row["source_event_ids"])
        == set(row["direct_source_event_ids"]).union(row["episode_source_event_ids"])
        for row in regions
    )
    assert all("1 selected event records" not in row["summary"] for row in regions)
    assert all("1 episode stories" not in row["summary"] for row in regions)
    history = (tmp_path / "observer" / "observer_world_history.md").read_text(encoding="utf-8")
    assert "## Episode History" in history
    assert "Source episode:" in history
    audio_lines = (tmp_path / "observer" / "observer_audio_script.txt").read_text(encoding="utf-8").splitlines()
    spoken = [line for line in audio_lines if line and not line.startswith("EIEL ") and not line.startswith("Audience ") and not line.startswith("Presentation ") and not line.startswith("[Generation")]
    assert spoken
    assert max(len(line) for line in spoken) <= 80
    assert max(len(line) for line in _bounded_spoken_lines("X" * 190, 60)) <= 60
    preflight = json.loads((tmp_path / "observer" / "observer_storage_preflight.json").read_text())
    assert preflight["scientific_storage_accounting_affected"] is False
    assert preflight["risk_classification"] == "compact"
    with pytest.raises(ObserverError, match="acknowledge_large_output"):
        ObserverConfig(storage_profile="observer_full").validate()
    accepted = observer_storage_preflight(
        source_entry_count=10,
        source_episode_count=2,
        selected_entry_count=10,
        selected_episode_count=2,
        estimated_card_count=3,
        config=ObserverConfig(storage_profile="observer_full", acknowledge_large_output=True),
    )
    assert accepted["accepted"] is True
    assert accepted["risk_classification"] == "large"


def test_card_sentences_inflect_and_humour_varies_by_subject_id() -> None:
    summaries = [
        _card_summary(
            audience_profile="dry_humour",
            kind="agent",
            identifier=str(identifier),
            event_count=1,
            first_generation=11,
            last_generation=11,
        )
        for identifier in (29, 37, 52, 153, 161)
    ]
    assert all("1 selected Observer record in generation 11" in summary for summary, _template_id in summaries)
    assert all("1 selected Observer records" not in summary for summary, _template_id in summaries)
    assert len({template_id for _summary, template_id in summaries}) >= 2


def test_gen1_boundary_is_observed_only_and_never_explains_success(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "run")
    gen1 = _gen1(tmp_path / "gen1", chronicle_root)
    manifest = write_observer_outputs(
        tmp_path / "run",
        tmp_path / "observer",
        config=ObserverConfig(audience_profile="public_plain", repeated_theme_cooldown=0),
        gen1_root=gen1,
    )
    agent_cards = read_jsonl(tmp_path / "observer" / "observer_agents.jsonl")
    lineage_cards = read_jsonl(tmp_path / "observer" / "observer_lineages.jsonl")
    mutations = [
        mutation
        for card in [*agent_cards, *lineage_cards]
        for mutation in card.get("mutation_occurrences", [])
    ]
    assert mutations
    assert {row["evidence_strength"] for row in mutations} == {"observed"}
    assert {row["inheritance_status"] for row in mutations} == {"unavailable"}
    assert {row["association_status"] for row in mutations} == {"unavailable"}
    assert {row["causal_status"] for row in mutations} == {"not_assessed"}
    assert all("affected an outcome" in row["public_statement"] for row in mutations)
    assert manifest["gen1_r1"]["maximum_real_evidence_strength"] == "observed"


def test_fail_closed_corruption_path_safety_and_output_separation(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "run")
    with pytest.raises(ObserverError, match="outside the scientific Chronicle"):
        write_observer_outputs(
            tmp_path / "run",
            chronicle_root / "observer",
            config=ObserverConfig(),
        )
    symlink = tmp_path / "source-link"
    symlink.symlink_to(tmp_path / "run", target_is_directory=True)
    with pytest.raises(ObserverError, match="must not be a symlink"):
        write_observer_outputs(symlink, tmp_path / "observer", config=ObserverConfig())
    reserved = tmp_path / "reserved-output.obs1-staging"
    reserved.mkdir()
    marker = reserved / "unrelated.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ObserverError, match="reserved Observer publication path already exists"):
        write_observer_outputs(tmp_path / "run", tmp_path / "reserved-output", config=ObserverConfig())
    assert marker.read_text(encoding="utf-8") == "preserve"
    manifest_path = chronicle_root / "chronicle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["finalized"] = False
    write_json(manifest_path, manifest)
    with pytest.raises(ObserverError, match="requires a finalized"):
        write_observer_outputs(tmp_path / "run", tmp_path / "observer-provisional", config=ObserverConfig())
    manifest["finalized"] = True
    write_json(manifest_path, manifest)
    entries_path = chronicle_root / "scientific_chronicle.jsonl"
    entries = read_jsonl(entries_path)
    entries[0]["source_event_ids"] = ["evt_missing"]
    write_jsonl(entries_path, entries)
    with pytest.raises(ObserverError, match="not traceable"):
        write_observer_outputs(tmp_path / "run", tmp_path / "observer-corrupt", config=ObserverConfig())


def test_regeneration_failure_leaves_previous_observer_tree_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _chronicle(tmp_path / "run")
    output = tmp_path / "observer"
    write_observer_outputs(tmp_path / "run", output, config=ObserverConfig())
    before = _tree_bytes(output)
    real_hashes = observer_module._scientific_hashes
    calls = 0

    def changing_hashes(root: Path) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        rows = real_hashes(root)
        if calls == 2:
            rows[0] = {**rows[0], "sha256": "0" * 64}
        return rows

    monkeypatch.setattr(observer_module, "_scientific_hashes", changing_hashes)
    with pytest.raises(ObserverError, match="source bytes changed"):
        write_observer_outputs(tmp_path / "run", output, config=ObserverConfig())
    assert _tree_bytes(output) == before
    assert not output.with_name("observer.obs1-staging").exists()
    assert not output.with_name("observer.obs1-previous").exists()


def test_explorer_observer_is_read_only_profiled_source_linked_and_bounded(tmp_path: Path) -> None:
    _chronicle(tmp_path / "runs" / "run-a")
    for profile in ("public_plain", "dry_humour"):
        write_observer_outputs(
            tmp_path / "runs" / "run-a",
            tmp_path / "runs" / "run-a" / "observer" / profile,
            config=ObserverConfig(audience_profile=profile, repeated_theme_cooldown=0),
        )
    cfg = _web_config(tmp_path / "runs")
    row = discover_runs(cfg)[0]
    assert row["observer_profiles"] == ["public_plain", "dry_humour"]
    assert row["run_path"] == str(tmp_path / "runs" / "run-a")
    run_key = row["run_key"]
    overview = observer_overview(cfg, run_key, "dry_humour")
    assert overview["presentation_only"] is True
    assert overview["source_evidence"]["scientific_source_hashes_verified"] is True
    assert overview["available_profiles"] == ["public_plain", "dry_humour"]
    page = observer_feed(
        cfg,
        run_key,
        "dry_humour",
        {"minimum_live_priority": ["high"], "limit": ["2"]},
    )
    assert page["items"]
    assert len(page["items"]) <= 2
    assert all(item["presentation_dimensions"]["live_priority"] in {"critical", "high"} for item in page["items"])
    unfiltered_page = observer_feed(cfg, run_key, "dry_humour", {"limit": ["200"]})
    available_family = next(
        item["episode_family"] for item in unfiltered_page["items"] if item.get("episode_family")
    )
    family_page = observer_feed(
        cfg,
        run_key,
        "dry_humour",
        {"episode_family": [available_family], "limit": ["200"]},
    )
    assert family_page["items"]
    assert {item["episode_family"] for item in family_page["items"]} == {available_family}
    missing_family_page = observer_feed(
        cfg,
        run_key,
        "dry_humour",
        {"episode_family": ["definitely_missing_family"], "limit": ["200"]},
    )
    assert missing_family_page["items"] == []
    episode = read_jsonl(
        tmp_path / "runs" / "run-a" / "observer" / "dry_humour" / "observer_episodes.jsonl"
    )[0]
    source_episode_id = episode["source_links"]["episode_id"]
    detail = observer_story(cfg, run_key, "dry_humour", "episode", source_episode_id)
    assert detail["story"]["source_links"]["event_ids"]
    assert detail["authoritative_source"] == "Scientific Chronicle"
    cards = observer_stories(cfg, run_key, "dry_humour", "lineage", {"limit": ["10"]})
    assert cards["items"]
    assert cards["presentation_only"] is True
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/observer/control",
        qs={"run_key": [run_key]},
    )
    assert status == 404
    assert payload["error_code"] == "not_found"
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/observer/feed",
        qs={"run_key": [run_key], "profile": ["dry_humour"], "limit": ["201"]},
    )
    assert status == 400
    assert payload["error_code"] == "invalid_paging"


def test_discovery_keeps_compact_dialogue_ready_runs_visible(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "runs" / "compact-run")
    (chronicle_root / "chronicle_storage_summary.json").unlink()

    row = discover_runs(_web_config(tmp_path / "runs"))[0]

    assert row["status"] == "ready"
    assert row["storage_summary_available"] is False


def test_explorer_observer_fails_closed_when_scientific_source_hash_changes(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "runs" / "run-a")
    write_observer_outputs(
        tmp_path / "runs" / "run-a",
        tmp_path / "runs" / "run-a" / "observer" / "public_plain",
        config=ObserverConfig(),
    )
    cfg = _web_config(tmp_path / "runs")
    run_key = discover_runs(cfg)[0]["run_key"]
    summary_path = chronicle_root / "scientific_chronicle_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["limitations"] = ["changed after Observer generation"]
    write_json(summary_path, summary)
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/observer",
        qs={"run_key": [run_key], "profile": ["public_plain"]},
    )
    assert status == 422
    assert payload["error_code"] == "source_hash_mismatch"


def test_explorer_observer_verifies_derived_artifacts_and_source_ids(tmp_path: Path) -> None:
    _chronicle(tmp_path / "runs" / "run-a")
    observer_root = tmp_path / "runs" / "run-a" / "observer" / "public_plain"
    write_observer_outputs(
        tmp_path / "runs" / "run-a",
        observer_root,
        config=ObserverConfig(repeated_theme_cooldown=0),
    )
    cfg = _web_config(tmp_path / "runs")
    run_key = discover_runs(cfg)[0]["run_key"]
    feed_path = observer_root / "observer_feed.jsonl"
    feed = read_jsonl(feed_path)
    feed[0]["text"] = "Tampered Observer sentence."
    write_jsonl(feed_path, feed)
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/observer/feed",
        qs={"run_key": [run_key], "profile": ["public_plain"]},
    )
    assert status == 422
    assert payload["error_code"] == "observer_hash_mismatch"
    manifest_path = observer_root / "observer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["relative_path"] == "observer_feed.jsonl":
            artifact["sha256"] = sha256_file(feed_path)
            artifact["bytes"] = feed_path.stat().st_size
    feed[0]["source_links"]["event_ids"] = ["evt_missing"]
    write_jsonl(feed_path, feed)
    for artifact in manifest["artifacts"]:
        if artifact["relative_path"] == "observer_feed.jsonl":
            artifact["sha256"] = sha256_file(feed_path)
            artifact["bytes"] = feed_path.stat().st_size
    write_json(manifest_path, manifest)
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/observer/feed",
        qs={"run_key": [run_key], "profile": ["public_plain"]},
    )
    assert status == 422
    assert payload["error_code"] == "traceability_mismatch"


def test_explorer_requires_complete_scientific_source_inventory(tmp_path: Path) -> None:
    _chronicle(tmp_path / "runs" / "run-a")
    observer_root = tmp_path / "runs" / "run-a" / "observer" / "public_plain"
    write_observer_outputs(tmp_path / "runs" / "run-a", observer_root, config=ObserverConfig())
    manifest_path = observer_root / "observer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_chronicle_artifacts"] = []
    write_json(manifest_path, manifest)
    cfg = _web_config(tmp_path / "runs")
    run_key = discover_runs(cfg)[0]["run_key"]
    status, payload = handle_get_request(
        cfg=cfg,
        path="/api/chronicle/observer",
        qs={"run_key": [run_key], "profile": ["public_plain"]},
    )
    assert status == 422
    assert payload["error_code"] == "traceability_mismatch"


def test_checkpoint_branch_prefix_is_valid_but_unrelated_mixing_is_not(tmp_path: Path) -> None:
    chronicle_root = _chronicle(tmp_path / "run")
    events_path = chronicle_root / "formal_events.jsonl"
    entries_path = chronicle_root / "scientific_chronicle.jsonl"
    manifest_path = chronicle_root / "chronicle_manifest.json"
    events = read_jsonl(events_path)
    entries = read_jsonl(entries_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefix_count = max(1, len(events) // 3)
    for event, entry in zip(events[:prefix_count], entries[:prefix_count]):
        event["run_id"] = "parent-run"
        event["branch_id"] = "parent"
        entry["run_id"] = "parent-run"
        entry["branch_id"] = "parent"
    manifest["parent_checkpoint_id"] = "checkpoint-test"
    write_jsonl(events_path, events)
    write_jsonl(entries_path, entries)
    write_json(manifest_path, manifest)
    source = load_scientific_source(tmp_path / "run")
    assert {event["run_id"] for event in source.events} == {"parent-run", "observer-run"}
    manifest["parent_checkpoint_id"] = None
    write_json(manifest_path, manifest)
    with pytest.raises(ObserverError, match="without checkpoint ancestry"):
        load_scientific_source(tmp_path / "run")


def test_explorer_static_creator_scientist_dialogue_is_wired() -> None:
    html = (REPO_ROOT / "infra" / "webui" / "static" / "chronicle.html").read_text(encoding="utf-8")
    script = (REPO_ROOT / "infra" / "webui" / "static" / "chronicle.js").read_text(encoding="utf-8")
    assert 'id="simulationSelect"' in html
    assert 'id="runSearch"' in html
    assert 'id="audienceSelect"' in html
    assert 'value="dry_humour" selected' in html
    assert 'class="dialogue"' in html
    assert "/api/chronicle/dialogue" in script
    assert "matchingRuns" in script
    assert "bubble ${speaker}" in script
    assert 'speaker_label' in script
    assert 'id="observerView"' not in html


def test_observer_generate_cli_writes_a_verified_package(tmp_path: Path) -> None:
    _chronicle(tmp_path / "run")
    output = tmp_path / "cli-observer"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "projects" / "emergence" / "tools" / "scientific_chronicle.py"),
            "observer-generate",
            str(tmp_path / "run"),
            str(output),
            "--audience-profile",
            "field_observer",
            "--repeated-theme-cooldown",
            "0",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(completed.stdout)
    assert manifest["audience_profile"] == "field_observer"
    assert manifest["observer_feed_count"] > 0
    assert set(path.name for path in output.iterdir()) == set(OBSERVER_ARTIFACTS.values())
