from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scripts.recomb1_build_surface_audit import build


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _shards(tmp_path: Path, *, unknown_count: int = 0) -> tuple[Path, Path, Path]:
    scalar = _write(
        tmp_path / "scalar.json",
        {
            "coverage_assertion": {"unknown_current_surface_count": unknown_count},
            "hereditary_surfaces": [{"surface_id": "flat.example"}],
            "runtime_state_exclusions": [{"field": "lifetime_state"}],
        },
    )
    structural = _write(
        tmp_path / "structural.json",
        {
            "audit_outcome": {"status": "complete_for_current_surface_classification"},
            "surfaces": [{"surface": "modules"}],
        },
    )
    plan = _write(
        tmp_path / "plan.json",
        {
            "audit_decision": "fixture_implementation_required_before_scientific_decision",
            "smallest_deterministic_fixture_matrix": {"selection": "none"},
        },
    )
    return scalar, structural, plan


def test_surface_audit_joins_complete_shards_and_keeps_runtime_state_excluded(
    tmp_path: Path,
) -> None:
    scalar, structural, plan = _shards(tmp_path)
    result = build(scalar_path=scalar, structural_path=structural, plan_path=plan)

    assert result["coverage"]["unknown_production_surface_count"] == 0
    assert result["coverage"]["total_surface_family_count"] == 2
    assert result["runtime_state_exclusions"] == [{"field": "lifetime_state"}]
    assert result["audit_conclusion"]["production_decision"] == "not_yet_frozen"
    assert result["scope"]["cuda_or_accelerated_work_started"] is False


def test_surface_audit_rejects_any_unknown_production_surface(tmp_path: Path) -> None:
    scalar, structural, plan = _shards(tmp_path, unknown_count=1)
    with pytest.raises(ValueError, match="unknown scalar/policy production surfaces remain"):
        build(scalar_path=scalar, structural_path=structural, plan_path=plan)
