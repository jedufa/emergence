from __future__ import annotations

from typing import Any

import pytest

from agents.agent import Agent
from native_runtime.flat_population_reporting import native_flat_population_reporting_records
from native_runtime.public_metrics_overlay import (
    NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS,
    public_metrics_overlay,
)
from simulation.metrics_pkg.cognitive_status import (
    compute_cognitive_status_metrics_from_records,
)
from tests.perf.test_eiel_native_flat_reporting_projection import (
    _behavior_rows,
    _configs,
    _owner_projection,
)


def test_native_flat_reporting_adapter_and_overlay_do_not_construct_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_attempts = 0

    def forbid_agent_construction(*_args: Any, **_kwargs: Any) -> None:
        nonlocal construction_attempts
        construction_attempts += 1
        raise AssertionError("native Flat reporting must not construct Agent objects")

    monkeypatch.setattr(Agent, "__init__", forbid_agent_construction)

    projection = _owner_projection([41])
    behavior_rows = _behavior_rows([41], selected_ids={41})
    env_cfg, sim_cfg = _configs()
    sim_cfg.lineage_observability_enable = False

    records = native_flat_population_reporting_records(
        tensors=projection["tensors"],
        behavior_rows=behavior_rows,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metadata=projection["metadata"],
    )
    reduced = compute_cognitive_status_metrics_from_records(
        records,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=4,
    )
    adapter_metrics = {
        name: reduced[name] for name in NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS
    }

    overlay = public_metrics_overlay(
        {"behavior_rows": behavior_rows},
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )
    overlay_metrics = {
        name: overlay[name] for name in NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS
    }

    assert len(adapter_metrics) == 66
    assert len(overlay_metrics) == 66
    assert overlay_metrics == adapter_metrics
    assert construction_attempts == 0
