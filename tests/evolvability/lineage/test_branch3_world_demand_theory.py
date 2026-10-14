from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "tools/bin/branch3_world_demand_theory_report.py"
SPEC = importlib.util.spec_from_file_location("branch3_world_demand_theory_report", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
branch3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(branch3)


def _row(strategy_id: str, *, energy: float, lifetime: int = 100, children: int = 1, alive: bool = True) -> dict:
    return {
        "strategy_id": strategy_id,
        "memory_dim": 8,
        "sensor_radius": 2,
        "successful_children_count": children,
        "lifetime_ticks": lifetime,
        "alive": alive,
        "food_use_summary": {"energy_collected": energy},
        "damage_summary": {"injury": 0.0, "health": 1.0},
        "patch_use_summary": {"coverage": "not_available"},
    }


def test_compressed_strategy_signature_is_deterministic() -> None:
    row = _row("raw_a", energy=3.0, lifetime=128)

    assert branch3.compressed_strategy_signature(row, profile="nominal") == branch3.compressed_strategy_signature(
        dict(reversed(list(row.items()))), profile="nominal"
    )


def test_strategy_validation_collapses_raw_individual_noise_but_fails_closed() -> None:
    rows = [_row(f"raw_{i}", energy=2.0 + (i % 2), lifetime=96 + (i % 3)) for i in range(20)]

    validation = branch3.strategy_validation(rows)

    assert validation["strategy_layer_scientifically_discriminating"] is False
    assert validation["reporting_only_compression_attempted"] is True
    assert validation["reporting_only_compression_reduces_noise"] is True
    assert validation["nominal_compressed_strategy_ids"] < validation["raw_strategy_ids"]
    assert "patch-use and niche exposure coverage is missing" in validation["blocker_reasons"]


def test_memory_and_reproduction_bins_are_bounded() -> None:
    assert branch3.memory_bin(0) == "memory_none"
    assert branch3.memory_bin(8) == "memory_low"
    assert branch3.memory_bin(32) == "memory_medium"
    assert branch3.memory_bin(64) == "memory_high"
    assert branch3.reproduction_bin(0) == "children_0"
    assert branch3.reproduction_bin(1) == "children_1"
    assert branch3.reproduction_bin(2) == "children_2"
    assert branch3.reproduction_bin(3) == "children_3plus"


def test_compression_sensitivity_reports_noise_audit_only() -> None:
    rows = [_row(f"raw_{i}", energy=float(i % 4), lifetime=80 + i) | {"source_label": "cell_a"} for i in range(12)]
    out = branch3.compression_sensitivity(rows, [{"label": "cell_a"}])

    assert {row["profile"] for row in out} == {"coarse", "nominal", "fine"}
    assert all(row["scientific_use"] == "noise_audit_only" for row in out)
    assert all(row["patch_use_coverage"] == "missing" for row in out)
