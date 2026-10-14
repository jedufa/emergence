from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "tools/scripts/eiel_eed0_resident_checkpoint_gate_v1.py"


def test_checkpoint_gate_covers_two_axes_and_stays_unpromoted() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "for axis in (1.0, -1.0)" in source
    assert '"production_route_integrated": False' in source
    assert '"campaign_authorized": False' in source
    assert "selection, route qualification, and EED-1" in source


def test_checkpoint_gate_requires_explicit_sidecar_identity() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "sidecar[\"identity\"].get(\"cell_id\") != CELL_ID" in source
    assert "cursor checkpoint width differs from slot capacity" in source
