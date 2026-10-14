from __future__ import annotations

from pathlib import Path

from tools.scripts.eiel_option_d_m24_pcg64_route_promotion_decision import (
    DECISION_WITHHELD,
    decide,
)


def test_m24_withholds_route_promotion_with_bound_operation_receipts(
    tmp_path: Path,
) -> None:
    m15 = {
        "active_physical_streams": [
            {"physical_stream_id": "torch.mutation.policy_birth.cpu"},
            {"physical_stream_id": "numpy.environment_policy"},
            {"physical_stream_id": "numpy.runner"},
            {"physical_stream_id": "numpy.population"},
            {"physical_stream_id": "numpy.birth_spawn_environment"},
        ]
    }
    m22 = {
        "pcg64_promotion_authority": {
            "runner_pcg64_operation_receipts": {
                "receipts": [
                    {
                        "operator_id": "runner.drive_noise.standard_normal",
                        "promotion_qualified": True,
                        "raw_pcg64_delta_available": True,
                        "promotion_blockers": [],
                    }
                ]
            },
            "turnover_spawn_coordinate_operation_receipts": {
                "receipts": [
                    {
                        "operator_id": "turnover.spawn_coordinate",
                        "promotion_qualified": True,
                        "raw_pcg64_delta_available": True,
                        "promotion_blockers": [],
                    }
                ]
            },
        }
    }
    m23 = {
        "operator_id": "population.newborn_drive.standard_normal",
        "promotion_qualified": True,
        "raw_pcg64_delta_available": True,
        "promotion_blockers": [],
    }
    paths = {}
    for name in ("m15", "m22", "m23"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path

    result = decide(
        m15_receipt=m15,
        m22_receipt=m22,
        m23_receipt=m23,
        paths=paths,
    )

    assert result["decision"] == DECISION_WITHHELD
    assert result["bounded_operation_receipts_qualified"] is True
    assert result["pcg64_route_promotion_qualified"] is False
    assert result["qualified_operations"] == [
        "population.newborn_drive.standard_normal",
        "runner.drive_noise.standard_normal",
        "turnover.spawn_coordinate",
    ]
    assert result["missing_required_streams"] == ["numpy.environment_policy"]
    assert result["unqualified_live_streams"] == ["numpy.environment_policy"]
    assert "pcg64_physical_streams_without_operation_receipts" in result[
        "promotion_blockers"
    ]
    assert len(result["receipt_sha256"]) == 64


def test_m24_fails_closed_when_population_raw_receipt_missing(tmp_path: Path) -> None:
    m15 = {
        "active_physical_streams": [
            {"physical_stream_id": "numpy.runner"},
            {"physical_stream_id": "numpy.population"},
            {"physical_stream_id": "numpy.birth_spawn_environment"},
        ]
    }
    m22 = {"pcg64_promotion_authority": {}}
    m23 = {
        "operator_id": "population.newborn_drive.standard_normal",
        "promotion_qualified": False,
        "raw_pcg64_delta_available": False,
        "promotion_blockers": ["raw_missing"],
    }
    path = tmp_path / "empty.json"
    path.write_text("{}\n", encoding="utf-8")

    result = decide(
        m15_receipt=m15,
        m22_receipt=m22,
        m23_receipt=m23,
        paths={"empty": path},
    )

    assert result["bounded_operation_receipts_qualified"] is False
    assert result["pcg64_route_promotion_qualified"] is False
    assert result["qualified_operations"] == []
    assert result["unqualified_live_streams"] == [
        "numpy.birth_spawn_environment",
        "numpy.population",
        "numpy.runner",
    ]
