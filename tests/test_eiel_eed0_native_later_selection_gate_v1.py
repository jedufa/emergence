from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT / "tools/scripts/eiel_eed0_native_later_selection_gate_v1.py"
)


def test_native_selection_gate_requires_both_reciprocal_directions() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'for resource in ("a", "b")' in source
    assert "on[\"later_stock\"] < off[\"later_stock\"]" in source
    assert "on[\"later_effective_try_prob\"]" in source
    assert "compute_lro1_reproductive_opportunity" in source
    assert 'result.get("owner") != "native_cpp_cuda"' in source


def test_native_selection_gate_composes_all_prior_authorities() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "EED-0-RESIDENT-PHYSICAL-PARITY-A" in source
    assert "EED-0-RESIDENT-CHECKPOINT-A" in source
    assert "EED-0-NATIVE-PARITY-A" in source
    assert '"production_route_integrated": False' in source
    assert '"campaign_authorized": False' in source
