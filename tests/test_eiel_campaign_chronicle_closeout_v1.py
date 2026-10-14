from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.campaigns.chronicle_closeout import (
    CampaignChronicleCloseoutError,
    materialize_completed_campaign_chronicles,
)
from engine.campaigns.production_engine_identity import PRODUCTION_PROVIDER_ID


def _plan(tmp_path: Path, *, profile: str = "notable") -> dict:
    return {
        "campaign_id": "campaign-a",
        "cells": [{"cell_id": "world-0"}, {"cell_id": "world-1"}],
        "execution": {"generations": 3},
        "output": {
            "chronicle_profile": profile,
            "chronicle_history_root": str(tmp_path / "history"),
            "chronicle_archive_enabled": True,
        },
    }


def _metrics_manifest(*, provider_id: str | None = PRODUCTION_PROVIDER_ID) -> dict:
    manifest = {
        "metrics_rows": [
            {
                "cell_id": "world-0",
                "generation": 1,
                "metrics": {"population.alive": 5, "population.births": 2},
            },
            {
                "cell_id": "world-0",
                "generation": 3,
                "metrics": {"population.alive": 4, "population.deaths": 1},
            },
            {
                "cell_id": "world-1",
                "metrics": {"population.alive": 7},
            },
        ]
    }
    if provider_id is not None:
        manifest["provider_id"] = provider_id
    return manifest


def test_completed_campaign_materializes_one_finalized_chronicle_per_world(
    tmp_path: Path,
) -> None:
    worlds = materialize_completed_campaign_chronicles(
        _plan(tmp_path),
        run_root=tmp_path / "run",
        batch_manifests=[_metrics_manifest()],
    )

    assert set(worlds) == {"world-0", "world-1"}
    for world_id, chronicle_root in worlds.items():
        manifest = json.loads(
            (chronicle_root / "chronicle_manifest.json").read_text()
        )
        assert manifest["campaign_id"] == "campaign-a"
        assert manifest["run_id"] == world_id
        assert manifest["profile"] == "notable"
        assert manifest["finalized"] is True


@pytest.mark.parametrize(
    "provider_id",
    [
        None,
        "recomb72-live-simulation-runner-provider-v1",
        "recomb72-gpu-native-cuda-resident-provider-v1",
        "arbitrary-provider-v9",
    ],
)
def test_completed_campaign_rejects_nonproduction_provider_before_closeout(
    tmp_path: Path,
    provider_id: str | None,
) -> None:
    run_root = tmp_path / "run"

    with pytest.raises(
        CampaignChronicleCloseoutError,
        match="production Chronicle closeout requires only provider",
    ):
        materialize_completed_campaign_chronicles(
            _plan(tmp_path),
            run_root=run_root,
            batch_manifests=[_metrics_manifest(provider_id=provider_id)],
        )

    assert not (run_root / "chronicle_worlds").exists()


def test_completed_campaign_rejects_mixed_provider_manifests(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        CampaignChronicleCloseoutError,
        match="production Chronicle closeout requires only provider",
    ):
        materialize_completed_campaign_chronicles(
            _plan(tmp_path),
            run_root=tmp_path / "run",
            batch_manifests=[
                _metrics_manifest(),
                _metrics_manifest(provider_id="historical-provider-v1"),
            ],
        )


def test_completed_campaign_fails_closed_when_a_world_has_no_metrics(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        CampaignChronicleCloseoutError, match="world-1.*no Chronicle-capable metrics"
    ):
        materialize_completed_campaign_chronicles(
            _plan(tmp_path),
            run_root=tmp_path / "run",
            batch_manifests=[
                {
                    "provider_id": PRODUCTION_PROVIDER_ID,
                    "metrics_rows": [
                        {
                            "cell_id": "world-0",
                            "metrics": {"population.alive": 5},
                        }
                    ],
                }
            ],
        )
