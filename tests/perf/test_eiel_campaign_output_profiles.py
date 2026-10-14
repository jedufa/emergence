from __future__ import annotations

from simulation.runner_pkg.runner_config import SimConfig

from experiments.harness.output_storage import (
    COMPACT_DEFAULT,
    COMPACT_FUNNEL,
    DEBUG_EVENT_TRACE,
    FULL_REPLAY,
    campaign_output_profile_knobs,
    classify_storage_risk,
    resolve_output_settings,
)


def _knob_cli(knobs: list[str]) -> list[str]:
    cli: list[str] = []
    for knob in knobs:
        cli.extend(["--knob", knob])
    return cli


def test_sim_config_default_campaign_output_is_compact_and_no_generation_checkpoint() -> None:
    cfg = SimConfig()

    assert cfg.campaign_output_profile == COMPACT_DEFAULT
    assert cfg.checkpoint_bundle_v2_period_gen == 0
    assert cfg.lineage_reproduction_funnel_write_events is False


def test_compact_profile_overrides_dangerous_registry_output_knobs() -> None:
    registry_cli = _knob_cli(
        [
            "sim_cfg.checkpoint_bundle_v2_period_gen=1",
            "sim_cfg.lineage_reproduction_funnel_write_events=true",
            "sim_cfg.lineage_reproduction_funnel_write_per_generation=true",
            "sim_cfg.behavior_log_enable=true",
        ]
    )
    compact_cli = registry_cli + _knob_cli(campaign_output_profile_knobs(profile=COMPACT_DEFAULT))

    settings = resolve_output_settings(compact_cli)
    risk = classify_storage_risk(settings, planned_cells=40)

    assert settings.profile == COMPACT_DEFAULT
    assert settings.checkpoint_period_gen == 0
    assert settings.lineage_funnel_write_events is False
    assert settings.lineage_funnel_write_per_generation is False
    assert settings.behavior_log_enable is False
    assert risk["ok"] is True


def test_compact_funnel_profile_preserves_generation_funnel_without_event_jsonl() -> None:
    registry_cli = _knob_cli(
        [
            "sim_cfg.lineage_reproduction_funnel_reporting_enable=true",
            "sim_cfg.lineage_reproduction_funnel_write_events=true",
            "sim_cfg.lineage_reproduction_funnel_write_per_generation=false",
            "sim_cfg.behavior_log_enable=true",
        ]
    )
    compact_cli = registry_cli + _knob_cli(campaign_output_profile_knobs(profile=COMPACT_FUNNEL))

    settings = resolve_output_settings(compact_cli)
    risk = classify_storage_risk(settings, planned_cells=80)

    assert settings.profile == COMPACT_FUNNEL
    assert settings.checkpoint_period_gen == 0
    assert settings.lineage_funnel_write_events is False
    assert settings.lineage_funnel_write_per_generation is True
    assert settings.lineage_funnel_include_patch_context is True
    assert settings.lineage_funnel_include_basin_context is True
    assert settings.behavior_log_enable is False
    assert risk["ok"] is True
    assert risk["risk"] == "moderate"


def test_compact_profile_preserves_explicit_buffered_developmental_trace() -> None:
    registry_cli = _knob_cli(
        [
            "sim_cfg.runnercore47_developmental_trace_buffered_emit_enable=true",
            *campaign_output_profile_knobs(profile=COMPACT_DEFAULT),
        ]
    )

    settings = resolve_output_settings(registry_cli)

    assert settings.profile == COMPACT_DEFAULT
    assert settings.runnercore47_developmental_trace_buffered_emit_enable is True


def test_full_replay_allows_generation_checkpointing_only_with_acknowledgement() -> None:
    cli = _knob_cli(
        [
            "sim_cfg.checkpoint_bundle_v2_period_gen=1",
            *campaign_output_profile_knobs(
                profile=FULL_REPLAY,
                allow_large_output=True,
                allow_generation_checkpointing=True,
            ),
        ]
    )

    risk = classify_storage_risk(resolve_output_settings(cli), planned_cells=1)

    assert risk["risk"] == "full_replay_heavy"
    assert risk["ok"] is True


def test_period_one_outside_full_replay_fails_storage_preflight() -> None:
    cli = _knob_cli(
        [
            "sim_cfg.checkpoint_bundle_v2_period_gen=1",
            "sim_cfg.campaign_output_profile=debug_event_trace",
            "sim_cfg.allow_large_output_profile=true",
            "sim_cfg.allow_generation_checkpointing=true",
        ]
    )

    risk = classify_storage_risk(resolve_output_settings(cli), planned_cells=1)

    assert risk["ok"] is False
    assert "checkpoint_period_1_requires_full_replay_profile" in risk["errors"]


def test_event_jsonl_requires_debug_or_full_profile_and_large_output_ack() -> None:
    compact_cli = _knob_cli(
        [
            "sim_cfg.campaign_output_profile=compact_default",
            "sim_cfg.lineage_reproduction_funnel_write_events=true",
        ]
    )
    debug_cli = _knob_cli(
        [
            "sim_cfg.lineage_reproduction_funnel_write_events=true",
            *campaign_output_profile_knobs(profile=DEBUG_EVENT_TRACE, allow_large_output=True),
        ]
    )

    compact_risk = classify_storage_risk(resolve_output_settings(compact_cli), planned_cells=1)
    debug_risk = classify_storage_risk(resolve_output_settings(debug_cli), planned_cells=1)

    assert compact_risk["ok"] is False
    assert "event_jsonl_requires_debug_event_trace_or_full_replay_profile" in compact_risk["errors"]
    assert debug_risk["ok"] is True


def test_notable_chronicle_is_bounded_and_estimated_in_campaign_preflight() -> None:
    cli = ["--gen", "10", *_knob_cli([
        "sim_cfg.chronicle_enable=true",
        "sim_cfg.chronicle_profile=notable",
        "sim_cfg.chronicle_max_stored_events_per_generation=12",
        "sim_cfg.chronicle_estimated_event_bytes=1024",
    ])]

    risk = classify_storage_risk(resolve_output_settings(cli), planned_cells=6)

    assert risk["ok"] is True
    assert risk["risk"] == "moderate"
    assert risk["chronicle_estimated_events"] == 720
    assert risk["chronicle_estimated_bytes"] == 720 * 1024
    assert "scientific_chronicle_enabled" in risk["warnings"]


def test_heavy_chronicle_requires_campaign_storage_acknowledgement() -> None:
    unacknowledged = _knob_cli([
        "sim_cfg.chronicle_enable=true",
        "sim_cfg.chronicle_profile=scientific_full",
    ])
    acknowledged = [*unacknowledged, *_knob_cli(["sim_cfg.allow_large_output_profile=true"])]

    rejected = classify_storage_risk(resolve_output_settings(unacknowledged), planned_cells=1)
    accepted = classify_storage_risk(resolve_output_settings(acknowledged), planned_cells=1)

    assert rejected["ok"] is False
    assert "heavy_chronicle_profile_requires_allow_large_output" in rejected["errors"]
    assert accepted["ok"] is True
    assert accepted["risk"] == "large"
