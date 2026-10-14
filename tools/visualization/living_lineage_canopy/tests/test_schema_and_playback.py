from __future__ import annotations

import copy

import pytest

from tools.visualization.living_lineage_canopy.python.generator import balanced
from tools.visualization.living_lineage_canopy.python.playback_model import PlaybackModel
from tools.visualization.living_lineage_canopy.python.schema import SchemaError, validate_playback


def test_valid_playback_is_accepted() -> None:
    validate_playback(balanced())


def test_unsupported_schema_is_rejected() -> None:
    data = balanced()
    data["metadata"]["schema_version"] = "future_schema"
    with pytest.raises(SchemaError, match="unsupported schema_version"):
        validate_playback(data)


def test_duplicate_agent_id_is_rejected() -> None:
    data = balanced()
    data["agents"][1]["agent_id"] = data["agents"][0]["agent_id"]
    with pytest.raises(SchemaError, match="duplicate agent_id"):
        validate_playback(data)


def test_invalid_birth_death_ordering_is_rejected() -> None:
    data = balanced()
    data["agents"][0]["death_tick"] = data["agents"][0]["birth_tick"] - 1
    with pytest.raises(SchemaError, match="invalid death_tick"):
        validate_playback(data)


def test_missing_lineage_is_rejected() -> None:
    data = balanced()
    data["agents"][0]["lineage_id"] = "missing"
    with pytest.raises(SchemaError, match="missing lineage"):
        validate_playback(data)


def test_invalid_branch_ancestry_is_rejected() -> None:
    data = balanced()
    data["branches"][1]["parent_branch_id"] = "missing"
    with pytest.raises(SchemaError, match="missing parent"):
        validate_playback(data)


def test_missing_parent_reference_policy() -> None:
    data = balanced()
    data["agents"][-1]["parent_agent_ids"] = ["external-parent"]
    with pytest.raises(SchemaError, match="missing parent"):
        validate_playback(data)
    allowed = copy.deepcopy(data)
    allowed["metadata"]["parent_reference_policy"] = "allow_missing_external"
    result = validate_playback(allowed)
    assert result.warnings


def test_playback_counts_and_scrub_reconstruction() -> None:
    model = PlaybackModel(balanced())
    start = model.state_at(0)
    mid = model.state_at(120)
    back = model.state_at(0)
    assert start.living_agents == back.living_agents
    assert len(mid.living_agents) > len(start.living_agents)
    assert mid.active_lineage_ids
    assert mid.active_branch_ids


def test_reset_returns_initial_state() -> None:
    model = PlaybackModel(balanced())
    assert model.state_at(0).living_agents == model.state_at(-500).living_agents
