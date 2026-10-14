from __future__ import annotations

import json

import numpy as np

from agents.cognition import compute_env_cost_charge_scalar_from_surface, express_cost_charge_surface
from env.env_config import EnvConfig
from perf.costsurface_soa import (
    COSTSURFACE_SOA_FLOAT_FIELDS,
    CostSurfaceSoADiagnostic,
    costsurface_soa_row,
    costsurface_soa_scalar_from_row,
)
from simulation.runner_pkg.runner_config import SimConfig

from tests.perf.test_eiel_costsurf1_cost_charge_surface import _agent, _charged_cfg


def test_costsoa1_row_matches_cost_charge_surface_fields() -> None:
    sim_cfg = _charged_cfg()
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    surface = express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)

    row = costsurface_soa_row(surface)

    assert row.dtype == np.float64
    assert row.shape == (len(COSTSURFACE_SOA_FLOAT_FIELDS),)
    np.testing.assert_allclose(
        row,
        [float(getattr(surface, field_name)) for field_name in COSTSURFACE_SOA_FLOAT_FIELDS],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        costsurface_soa_scalar_from_row(
            row,
            charge_channels=sim_cfg.enable_experimental_perception_cost_charging,
            charge_radius=sim_cfg.enable_experimental_sensor_radius_cost_charging,
        ),
        compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg),
        rtol=0.0,
        atol=1e-12,
    )


def test_costsoa1_diagnostic_summary_serializes_field_parity(tmp_path) -> None:
    sim_cfg = _charged_cfg()
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    surface = express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    scalar = compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg)
    diag = CostSurfaceSoADiagnostic(run_dir=str(tmp_path), enabled=True)

    diag.record_surface(surface, current_scalar=scalar, current_scalar_s=0.125, sim_cfg=sim_cfg)
    out_path = diag.write_summary()
    payload = json.loads((tmp_path / "metrics" / "costsurface_soa_diagnostic_summary.json").read_text(encoding="utf-8"))

    assert out_path.endswith("metrics/costsurface_soa_diagnostic_summary.json")
    assert payload["field_names"] == list(COSTSURFACE_SOA_FLOAT_FIELDS)
    assert payload["dtype"] == "float64"
    assert payload["array_shape"] == [1, len(COSTSURFACE_SOA_FLOAT_FIELDS)]
    assert payload["rows_produced"] == 1
    assert payload["active_agents_covered"] == 1
    assert payload["missing_fallback_rows"] == 0
    assert payload["field_mismatch_counts"] == {field_name: 0 for field_name in COSTSURFACE_SOA_FLOAT_FIELDS}
    assert payload["max_numeric_diff_by_field"] == {field_name: 0.0 for field_name in COSTSURFACE_SOA_FLOAT_FIELDS}
    assert payload["scalar_mismatch_count"] == 0
    assert payload["max_scalar_abs_diff"] == 0.0
    assert payload["current_scalar_extraction_s"] == 0.125
    assert payload["estimated_avoidable_object_reads"] == len(COSTSURFACE_SOA_FLOAT_FIELDS)


def test_costsoa1_diagnostic_accumulates_multiple_records(tmp_path) -> None:
    sim_cfg = _charged_cfg()
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    surface = express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    scalar = compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg)
    diag = CostSurfaceSoADiagnostic(run_dir=str(tmp_path), enabled=True)

    diag.record_surface(surface, current_scalar=scalar, current_scalar_s=0.1, sim_cfg=sim_cfg)
    diag.record_surface(surface, current_scalar=scalar, current_scalar_s=0.2, sim_cfg=sim_cfg)
    payload = diag.summary_payload()

    assert payload["array_shape"] == [2, len(COSTSURFACE_SOA_FLOAT_FIELDS)]
    assert payload["rows_produced"] == 2
    assert payload["active_agents_covered"] == 2
    assert payload["current_scalar_extraction_s"] == 0.30000000000000004
    assert payload["estimated_avoidable_object_reads"] == 2 * len(COSTSURFACE_SOA_FLOAT_FIELDS)


def test_costsoa1_diagnostic_records_fallback_rows(tmp_path) -> None:
    diag = CostSurfaceSoADiagnostic(run_dir=str(tmp_path), enabled=True)

    diag.record_fallback("exception")
    payload = diag.summary_payload()

    assert payload["array_shape"] == [0, len(COSTSURFACE_SOA_FLOAT_FIELDS)]
    assert payload["missing_fallback_rows"] == 1
    assert payload["skipped_reason_counts"] == {"exception": 1}


def test_costsoa1_row_scalar_respects_disabled_ledger_flag() -> None:
    sim_cfg = SimConfig()
    row = np.ones((len(COSTSURFACE_SOA_FLOAT_FIELDS),), dtype=np.float64)

    assert (
        costsurface_soa_scalar_from_row(
            row,
            enable_cognitive_budget_ledger=sim_cfg.enable_cognitive_budget_ledger,
            charge_channels=True,
            charge_radius=True,
        )
        > 0.0
    )
    assert (
        costsurface_soa_scalar_from_row(
            row,
            enable_cognitive_budget_ledger=False,
            charge_channels=True,
            charge_radius=True,
        )
        == 0.0
    )
