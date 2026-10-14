from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ACTIVE_SCRIPT = Path(
    "tools/scripts/recomb1_hereditary_persistence_campaign.py"
).resolve()
HISTORICAL_SCRIPT = Path(
    "tools/historical/recomb1_hereditary_persistence_campaign.py"
).resolve()
SPEC = importlib.util.spec_from_file_location(
    "recomb1_hereditary_persistence_campaign",
    HISTORICAL_SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAMPAIGN
SPEC.loader.exec_module(CAMPAIGN)


def test_recomb1_campaign_is_not_in_the_active_tool_namespace() -> None:
    assert ACTIVE_SCRIPT.exists() is False
    assert HISTORICAL_SCRIPT.is_file()


def test_historical_campaign_cannot_execute_retired_profile() -> None:
    with pytest.raises(ValueError, match="P12_RETIRED_INHERITANCE_PROFILE"):
        CAMPAIGN.run_campaign(
            seeds=(7100,),
            population_size=8,
            generations=2,
            checkpoint_generation=1,
        )


def test_historical_cli_fails_without_writing_output(tmp_path: Path) -> None:
    output = tmp_path / "retired-campaign.json"
    result = subprocess.run(
        [
            sys.executable,
            str(HISTORICAL_SCRIPT),
            "--output",
            str(output),
            "--seeds",
            "7100",
            "--population-size",
            "8",
            "--generations",
            "2",
            "--checkpoint-generation",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "P12_RETIRED_INHERITANCE_PROFILE" in (
        result.stdout + result.stderr
    )
    assert output.exists() is False
