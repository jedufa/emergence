from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from pathlib import Path

import pytest

from tools.visualization.living_lineage_canopy.python.adapters.p12b import adapt_p12b, classify_p12b
from tools.visualization.living_lineage_canopy.python.schema import validate_playback


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "p12b_partial_birth_events"
EXTERNAL_P12B_ROOT = Path(public_runtime_path('evidence_raw/p12b_wdl_genome_v2_successor_science_20260728'))


def test_p12b_adapter_classifies_partial_individual_data() -> None:
    result = classify_p12b(FIXTURE_ROOT)
    assert result["classification"] == "B"
    assert result["semantics_mode"] == "partial_agents"
    assert "death" in " ".join(result["missing_fields"])


def test_p12b_adapter_emits_normalized_partial_playback() -> None:
    data = adapt_p12b(FIXTURE_ROOT, limit_rows=6)
    validate_playback(data)
    assert data["metadata"]["semantics_mode"] == "partial_agents"
    assert data["metadata"]["adapter_classification"]["classification"] == "B"
    assert len(data["agents"]) == 6
    assert data["metadata"]["source_path"] == str(FIXTURE_ROOT)


@pytest.mark.skipif(
    not EXTERNAL_P12B_ROOT.exists(),
    reason="optional integration test requires local P12B runtime evidence",
)
def test_p12b_adapter_external_runtime_source_when_available() -> None:
    data = adapt_p12b(EXTERNAL_P12B_ROOT, limit_rows=32)
    validate_playback(data)
    assert data["metadata"]["semantics_mode"] == "partial_agents"
    assert data["metadata"]["adapter_classification"]["classification"] == "B"
    assert len(data["agents"]) == 32
