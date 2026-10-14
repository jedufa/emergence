from __future__ import annotations

from pathlib import Path

import pytest

from engine.campaigns.wrc_adapters import (
    CONSTRUCTION_CONTEXT,
    GENERIC_RUNTIME,
    SPECIALIZED_CAMPAIGN,
    AtlasRowV1,
    GenericRuntimeParameterAdapter,
    WRCAtlasAdapterError,
    adapter_for_row_v1,
    load_authoritative_atlas_rows_v1,
    plan_authoritative_atlas_v1,
)


REGISTRY = Path(__file__).resolve().parents[2] / "_spec" / "registry" / "v1"


def test_authoritative_atlas_inventory_preserves_all_launcher_classes() -> None:
    rows = load_authoritative_atlas_rows_v1(REGISTRY)

    assert len(rows) == 95
    assert {row.launcher_profile for row in rows} == {
        "STANDARD_CUDA", "SPECIALIZED_MANUAL", "CONTEXT_MANUAL"
    }
    assert sum(row.launcher_profile == "STANDARD_CUDA" for row in rows) == 73
    assert sum(row.launcher_profile == "SPECIALIZED_MANUAL" for row in rows) == 15
    assert sum(row.launcher_profile == "CONTEXT_MANUAL" for row in rows) == 7


def test_adapter_dispatch_is_explicit_and_manual_rows_cannot_fall_through() -> None:
    rows = {row.row_id: row for row in load_authoritative_atlas_rows_v1(REGISTRY)}

    assert adapter_for_row_v1(rows["P011"]).kind == GENERIC_RUNTIME
    assert adapter_for_row_v1(rows["P001"]).kind == SPECIALIZED_CAMPAIGN
    assert adapter_for_row_v1(rows["P089"]).kind == CONSTRUCTION_CONTEXT
    with pytest.raises(WRCAtlasAdapterError, match="expected STANDARD_CUDA"):
        GenericRuntimeParameterAdapter().plan_cells(rows["P001"])


def test_normalized_plan_identity_is_deterministic_and_covers_every_row() -> None:
    first = plan_authoritative_atlas_v1(REGISTRY)
    second = plan_authoritative_atlas_v1(REGISTRY)

    assert first == second
    assert [cell.identity_sha256 for cell in first] == [cell.identity_sha256 for cell in second]
    assert {cell.row_id for cell in first} == {row.row_id for row in load_authoritative_atlas_rows_v1(REGISTRY)}
    assert all(cell.expected_metrics for cell in first)
    assert all(cell.preflight_mode for cell in first)
    assert {
        cell.row_id for cell in first if cell.adapter_kind == CONSTRUCTION_CONTEXT and cell.level is None
    } == {"P089", "P090", "P091"}


def test_normalized_cell_exposes_distinct_configuration_output_and_reducer_identities() -> None:
    cells = plan_authoritative_atlas_v1(REGISTRY)
    first = next(cell for cell in cells if cell.row_id == "P011")
    sibling = next(
        cell
        for cell in cells
        if cell.row_id == "P011" and cell.level == first.level and cell.seed != first.seed
    )

    assert first.configuration_identity_payload()["seed"] == first.seed
    assert first.configuration_sha256 != sibling.configuration_sha256
    assert first.output_identity_payload()["configuration_sha256"] == first.configuration_sha256
    assert first.reducer_identity_payload()["output_sha256"] == first.output_sha256
    assert first.reducer_identity_payload()["reducer_protocol"] == "eiel_wrc_reducer_v1"
    assert first.reducer_sha256 in first.identity_payload().values()


def test_normalized_plan_retains_authoritative_overrides_and_preflight_assertions() -> None:
    cells = plan_authoritative_atlas_v1(REGISTRY)
    p008 = next(cell for cell in cells if cell.row_id == "P008")
    p056 = next(cell for cell in cells if cell.row_id == "P056")
    p011 = next(cell for cell in cells if cell.row_id == "P011")

    assert p008.required_overrides == {"eed.resource_impact_apply_enable": True}
    assert p056.activation_overrides == {"env_cfg.patch_movement_cost_default": 2.0}
    assert p011.preflight_assertions == ({
        "aggregate": "max",
        "allow_missing": True,
        "metric": "patch_recovery_activity_mean",
        "operator": ">",
        "value": 0.0,
    },)
    assert p008.configuration_sha256 != p056.configuration_sha256
    assert p011.configuration_identity_payload()["preflight_assertions"] == [
        dict(p011.preflight_assertions[0])
    ]
