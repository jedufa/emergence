from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from engine.campaigns.production_control_path_v1 import (
    PRODUCTION_CONTROL_READBACK_SCHEMA_V1,
    apply_primitive_overrides_v1,
    build_segment_control_readback_v1,
    primitive_parameter_contracts_v1,
)
from engine.campaigns.registry import default_profile_registry
from engine.env.env_config import EnvConfig
from engine.evolution.population_pkg.population_config import PopConfig
from engine.perf.cxxbackbone10_resident_state_owner import projection_digest
from engine.simulation.runner_pkg.runner_config import SimConfig


def _production_contracts(names: set[str]):
    profile = next(
        profile
        for profile in default_profile_registry().profiles()
        if profile.profile_id == "eiel-production-runtime-v1"
    )
    return {name: profile.parameter_contracts[name] for name in names}


def _readback_fixture(*, runtime_regrowth: float = 0.125):
    env_cfg = EnvConfig(width=16, height=16)
    env_cfg.patch_food_regrow_per_step = 0.125
    pop_cfg = PopConfig(population_size=64, max_slots_cap=256)
    sim_cfg = SimConfig()
    runner = SimpleNamespace(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg)
    build = SimpleNamespace(runner=runner, receipt={})
    source_projection = {
        "metadata": {
            "width": 16,
            "height": 16,
            "population_size": 64,
            "capacity": 256,
            "patch_food_regrow_per_step": 0.125,
        }
    }
    prepared = SimpleNamespace(
        source_projections=(source_projection,),
        source=SimpleNamespace(
            patch_p2_config={"patch_food_regrow_per_step": 0.125}
        ),
        checkpoint_projection_inputs={},
    )
    request = SimpleNamespace(
        configuration=SimpleNamespace(patch_food_regrow_per_step=0.125),
        envelope=SimpleNamespace(grid=(16, 16)),
    )
    checkpoint = {
        "metadata": {
            "width": 16,
            "height": 16,
            "population_size": 64,
            "capacity": 256,
            "patch_food_regrow_per_step": 0.125,
        }
    }
    post_dispatch = {
        "receipt_sha256": "a" * 64,
        "outer_arguments": [
            {
                "name": "patch_p2_config",
                "kind": "mapping",
                "scalar_values": {
                    "patch_food_regrow_per_step": runtime_regrowth,
                },
            }
        ],
    }
    cell = {
        "cell_id": "cell-0",
        "parameters": {
            "patch.grid_size": 16,
            "patch.initial_population": 64,
            "patch.capacity": 256,
            "env_cfg.patch_food_regrow_per_step": 0.125,
        },
    }
    metric_row = {
        "cell_id": "cell-0",
        "generation": 1,
        "patch_food_amount_mean": 0.75,
    }
    names = set(cell["parameters"])
    return {
        "cells": (cell,),
        "builds": (build,),
        "requests": (request,),
        "prepared": prepared,
        "checkpoint_projections": (checkpoint,),
        "compact_result_rows": (metric_row,),
        "parameter_contracts": _production_contracts(names),
        "post_dispatch_abi_receipt": post_dispatch,
    }


def test_primitive_contracts_are_typed_and_apply_after_profile_defaults() -> None:
    contracts = primitive_parameter_contracts_v1()
    control = "env_cfg.patch_food_regrow_per_step"
    assert contracts[control].accepts(0.125) is True
    assert contracts[control].accepts(1) is False

    movement_default = "env_cfg.patch_movement_cost_default"
    assert contracts[movement_default].accepts(2.0) is True
    assert contracts[movement_default].accepts(-0.1) is False

    env_cfg = EnvConfig()
    sim_cfg = SimConfig()
    env_cfg.patch_food_regrow_per_step = 0.006
    receipt = apply_primitive_overrides_v1(
        env_cfg,
        sim_cfg,
        {control: 0.125},
        allowed_parameter_names=frozenset({control}),
    )

    assert env_cfg.patch_food_regrow_per_step == 0.125
    assert receipt["requested"] == {control: 0.125}
    assert receipt["applied"] == {control: 0.125}


def test_segment_readback_binds_request_runtime_checkpoint_and_metrics() -> None:
    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **_readback_fixture(),
    )

    assert receipt["schema_version"] == PRODUCTION_CONTROL_READBACK_SCHEMA_V1
    assert receipt["receipt_sha256"]
    controls = {
        row["control"]: row for row in receipt["cells"][0]["controls"]
    }
    regrowth = controls["env_cfg.patch_food_regrow_per_step"]
    assert regrowth["requested"] == 0.125
    assert regrowth["compiled"] == 0.125
    assert regrowth["applied"] == 0.125
    assert regrowth["readback"] == 0.125
    assert regrowth["checkpoint_projection_value"] == 0.125
    assert regrowth["status"] == "READBACK_COMPLETE"
    assert receipt["cells"][0]["metric_row_sha256"]


def test_segment_readback_binds_tensor_bearing_checkpoint_projection() -> None:
    fixture = _readback_fixture()
    checkpoint = dict(fixture["checkpoint_projections"][0])
    checkpoint["tensors"] = {
        "resident_move_cost_by_cell": torch.tensor([0.2, 3.2], dtype=torch.float64)
    }
    fixture["checkpoint_projections"] = (checkpoint,)

    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **fixture,
    )

    assert receipt["cells"][0]["authoritative_state_projection_sha256"] == (
        projection_digest(checkpoint)
    )


def test_segment_readback_uses_per_cell_move_cost_over_shared_pre_metadata() -> None:
    fixture = _single_control_readback_fixture("env_cfg.move_cost", 0.2)
    fixture["prepared"].checkpoint_projection_inputs = {
        "pre_generation_canonical_projection": {"metadata": {"move_cost": 3.2}}
    }

    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **fixture,
    )

    control = receipt["cells"][0]["controls"][0]
    assert control["readback"] == 0.2
    assert control["runtime_sources"]["source_import_metadata"] == 0.2
    assert "prepared_checkpoint_metadata" not in control["runtime_sources"]


def test_segment_readback_uses_resident_move_cost_tensor_for_heterogeneous_cells() -> None:
    first_value, second_value = 0.2, 3.2
    first = _single_control_readback_fixture("env_cfg.move_cost", first_value)
    second_runner = SimpleNamespace(
        env_cfg=EnvConfig(width=16, height=16),
        pop_cfg=PopConfig(population_size=64, max_slots_cap=256),
        sim_cfg=SimConfig(),
    )
    second_runner.env_cfg.move_cost = second_value
    first["cells"] = (
        first["cells"][0],
        {"cell_id": "cell-1", "parameters": {"env_cfg.move_cost": second_value}},
    )
    first["builds"] = (
        first["builds"][0],
        SimpleNamespace(runner=second_runner, receipt={}),
    )
    first["requests"] = (
        first["requests"][0],
        SimpleNamespace(configuration=SimpleNamespace(move_cost=second_value)),
    )
    first["prepared"].cell_ids = ("cell-0", "cell-1")
    first["prepared"].source_projections = (
        {"metadata": {"move_cost": first_value}},
        {"metadata": {"move_cost": second_value}},
    )
    first["prepared"].source = SimpleNamespace(
        configuration=SimpleNamespace(move_cost=first_value),
        move_cost_by_cell=torch.tensor(
            [first_value, second_value], dtype=torch.float64
        ),
        patch_p2_config={},
    )
    first["checkpoint_projections"] = (
        {"metadata": {"move_cost": first_value}},
        {"metadata": {"move_cost": second_value}},
    )
    first["compact_result_rows"] = (
        {"cell_id": "cell-0", "generation": 1},
        {"cell_id": "cell-1", "generation": 1},
    )
    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **first,
    )

    second = receipt["cells"][1]["controls"][0]
    assert second["runtime_sources"]["resident_move_cost_by_cell"] == second_value
    assert "native_source_configuration" not in second["runtime_sources"]


def test_segment_readback_uses_resident_wdl_radius_tensor_for_heterogeneous_cells() -> None:
    control = "env_cfg.wdl_delayed_hazard_radius"
    first_value, second_value = 0, 3
    first = _single_control_readback_fixture(control, first_value)
    second_runner = SimpleNamespace(
        env_cfg=EnvConfig(width=16, height=16),
        pop_cfg=PopConfig(population_size=64, max_slots_cap=256),
        sim_cfg=SimConfig(),
    )
    second_runner.env_cfg.wdl_delayed_hazard_radius = second_value
    first["cells"] = (
        first["cells"][0],
        {"cell_id": "cell-1", "parameters": {control: second_value}},
    )
    first["builds"] = (
        first["builds"][0],
        SimpleNamespace(runner=second_runner, receipt={}),
    )
    first["requests"] = (
        first["requests"][0],
        SimpleNamespace(
            configuration=SimpleNamespace(wdl_delayed_hazard_radius=second_value)
        ),
    )
    first["prepared"].cell_ids = ("cell-0", "cell-1")
    first["prepared"].source_projections = (
        {"metadata": {"wdl_delayed_hazard_radius": first_value}},
        {"metadata": {"wdl_delayed_hazard_radius": second_value}},
    )
    first["prepared"].config = {
        "radius": first_value,
        "radius_by_cell": torch.tensor([first_value, second_value], dtype=torch.int32),
    }
    first["checkpoint_projections"] = (
        {"metadata": {"wdl_delayed_hazard_radius": first_value}},
        {"metadata": {"wdl_delayed_hazard_radius": second_value}},
    )
    first["compact_result_rows"] = (
        {"cell_id": "cell-0", "generation": 1},
        {"cell_id": "cell-1", "generation": 1},
    )

    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **first,
    )

    second = receipt["cells"][1]["controls"][0]
    assert second["readback"] == second_value
    assert second["runtime_sources"]["resident_wdl_radius_by_cell"] == second_value
    assert "specialized_runtime_config" not in second["runtime_sources"]


def test_segment_readback_uses_aligned_projection_for_varying_batch_control() -> None:
    control = "env_cfg.patch_movement_cost_energy_scale"
    first_value, second_value = 1.0, 16.0
    first = _single_control_readback_fixture(control, first_value)
    second_runner = SimpleNamespace(
        env_cfg=EnvConfig(width=16, height=16),
        pop_cfg=PopConfig(population_size=64, max_slots_cap=256),
        sim_cfg=SimConfig(),
    )
    setattr(second_runner.env_cfg, "patch_movement_cost_energy_scale", second_value)
    first_projection = first["prepared"].source_projections[0]
    second_projection = {"metadata": {"patch_movement_cost_energy_scale": second_value}}
    first["cells"] = (
        first["cells"][0],
        {"cell_id": "cell-1", "parameters": {control: second_value}},
    )
    first["builds"] = (
        first["builds"][0],
        SimpleNamespace(runner=second_runner, receipt={}),
    )
    first["requests"] = (
        first["requests"][0],
        SimpleNamespace(
            configuration=SimpleNamespace(patch_movement_cost_energy_scale=second_value),
            envelope=SimpleNamespace(grid=(16, 16)),
        ),
    )
    first["prepared"].source_projections = (
        first_projection,
        second_projection,
    )
    first["prepared"].source.patch_p2_config = {
        "patch_movement_cost_energy_scale": first_value,
        "patch_movement_cost_energy_scale_by_cell": torch.tensor(
            [first_value, second_value], dtype=torch.float64
        ),
    }
    first["checkpoint_projections"] = (
        first["checkpoint_projections"][0],
        second_projection,
    )
    first["compact_result_rows"] = (
        first["compact_result_rows"][0],
        {"cell_id": "cell-1", "generation": 1, "alive": 64},
    )
    first["post_dispatch_abi_receipt"] = {
        "receipt_sha256": "d" * 64,
        "outer_arguments": [{
            "name": "patch_p2_config",
            "kind": "mapping",
            "scalar_values": {"patch_movement_cost_energy_scale": first_value},
            "scalar_vectors": {
                "patch_movement_cost_energy_scale_by_cell": {
                    "dtype": "float64",
                    "shape": [2],
                    "values": [first_value, second_value],
                    "values_sha256": "e" * 64,
                }
            },
        }],
    }

    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **first,
    )

    controls = [cell["controls"][0] for cell in receipt["cells"]]
    assert [control_row["readback"] for control_row in controls] == [first_value, second_value]
    assert [cell["cell_ordinal"] for cell in receipt["cells"]] == [0, 1]
    assert [
        row["runtime_sources"]["post_dispatch_abi.patch_p2_config_by_cell"]
        for row in controls
    ] == [first_value, second_value]
    assert [
        row["runtime_sources"]["retained_patch_abi_config_by_cell"]
        for row in controls
    ] == [first_value, second_value]

    first["post_dispatch_abi_receipt"]["outer_arguments"][0]["scalar_vectors"][
        "patch_movement_cost_energy_scale_by_cell"
    ]["values"] = [second_value, first_value]
    with pytest.raises(ValueError, match="native production control differs"):
        build_segment_control_readback_v1(
            profile_id="eiel-production-runtime-v1",
            segment_ordinal=1,
            start_generation=0,
            end_generation=1,
            **first,
        )
    first["post_dispatch_abi_receipt"]["outer_arguments"][0]["scalar_vectors"][
        "patch_movement_cost_energy_scale_by_cell"
    ]["values"] = [first_value, second_value]

    first["prepared"].source_projections = (first_projection, first_projection)
    with pytest.raises(ValueError, match="native production control differs"):
        build_segment_control_readback_v1(
            profile_id="eiel-production-runtime-v1",
            segment_ordinal=1,
            start_generation=0,
            end_generation=1,
            **first,
        )


def test_segment_readback_fails_closed_on_post_dispatch_mismatch() -> None:
    with pytest.raises(ValueError, match="native production control differs"):
        build_segment_control_readback_v1(
            profile_id="eiel-production-runtime-v1",
            segment_ordinal=1,
            start_generation=0,
            end_generation=1,
            **_readback_fixture(runtime_regrowth=0.25),
        )


def test_segment_readback_fails_closed_when_runtime_value_is_absent() -> None:
    fixture = _readback_fixture()
    fixture["post_dispatch_abi_receipt"] = {
        "receipt_sha256": "b" * 64,
        "outer_arguments": [],
    }
    fixture["requests"] = (
        SimpleNamespace(
            configuration=SimpleNamespace(),
            envelope=SimpleNamespace(grid=(16, 16)),
        ),
    )
    fixture["prepared"].source.patch_p2_config = {}
    fixture["prepared"].source_projections = (
        {
            "metadata": {
                "width": 16,
                "height": 16,
                "population_size": 64,
                "capacity": 256,
            }
        },
    )

    with pytest.raises(ValueError, match="no native runtime readback"):
        build_segment_control_readback_v1(
            profile_id="eiel-production-runtime-v1",
            segment_ordinal=1,
            start_generation=0,
            end_generation=1,
            **fixture,
        )


def test_segment_readback_uses_wrapped_resident_patch_grid_without_envelope() -> None:
    fixture = _readback_fixture()
    fixture["requests"] = (SimpleNamespace(configuration=SimpleNamespace()),)
    fixture["prepared"] = SimpleNamespace(
        physical_window_arguments=SimpleNamespace(
            patch=SimpleNamespace(
                source=SimpleNamespace(
                    patch_p2_config={},
                    patch_p2_state_tensors=(
                        torch.zeros((1, 16, 16), dtype=torch.float64),
                    ),
                ),
                patch=(),
            )
        ),
        source_projections=fixture["prepared"].source_projections,
    )
    fixture["post_dispatch_abi_receipt"] = {
        "receipt_sha256": "b" * 64,
        "outer_arguments": [],
    }

    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **fixture,
    )

    control = next(
        row
        for row in receipt["cells"][0]["controls"]
        if row["control"] == "patch.grid_size"
    )
    assert control["runtime_sources"] == {"resident_patch_state_tensor": 16}
    assert control["status"] == "READBACK_COMPLETE"


def test_segment_readback_interprets_hybrid_lifecycle_from_native_source() -> None:
    """Public lifecycle admission is not the legacy cull/refill ABI switch."""

    env_cfg = EnvConfig(width=16, height=16)
    pop_cfg = PopConfig(population_size=64, max_slots_cap=256)
    sim_cfg = SimConfig()
    sim_cfg.lifecycle_mode = 1
    sim_cfg.generation_turnover_enabled = False
    sim_cfg.cull_fraction = 0.5
    sim_cfg.ordinary_births_enabled = True
    sim_cfg.life_history_profile_id = "stage1s_overlap_tick_life_history_v1"
    runner = SimpleNamespace(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg)
    parameters = {
        "production_lifecycle.turnover_enabled": True,
        "production_lifecycle.ordinary_births_enabled": True,
        "production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
        "production_lifecycle.turnover_cull_fraction": 0.0,
        "production_lifecycle.life_history_profile": (
            "stage1s_overlap_tick_life_history_v1"
        ),
    }
    source_configuration = SimpleNamespace(
        lifecycle_mode=1,
        generation_turnover_enabled=False,
        cull_fraction=0.5,
        ordinary_births_enabled=True,
        life_history_profile_id="stage1s_overlap_tick_life_history_v1",
    )
    prepared = SimpleNamespace(
        source_projections=({"metadata": {}},),
        source=SimpleNamespace(
            configuration=source_configuration,
            patch_p2_config={},
        ),
        checkpoint_projection_inputs={},
    )
    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        cells=({"cell_id": "cell-0", "parameters": parameters},),
        builds=(SimpleNamespace(runner=runner, receipt={}),),
        requests=(SimpleNamespace(),),
        prepared=prepared,
        checkpoint_projections=({"metadata": {"lifecycle_mode_abi": 1}},),
        compact_result_rows=({"cell_id": "cell-0", "generation": 1},),
        parameter_contracts=_production_contracts(set(parameters)),
        post_dispatch_abi_receipt={"receipt_sha256": "a" * 64, "outer_arguments": []},
    )

    controls = {row["control"]: row for row in receipt["cells"][0]["controls"]}
    assert controls["production_lifecycle.turnover_enabled"]["runtime_sources"] == {
        "native_lifecycle_mode_abi": True
    }
    assert controls["production_lifecycle.turnover_cull_fraction"]["runtime_sources"] == {
        "native_lifecycle_mode_abi": 0.0
    }
    assert controls["production_lifecycle.ordinary_births_enabled"]["runtime_sources"] == {
        "native_source_configuration": True
    }
    assert controls["production_lifecycle.lifecycle_mode"]["status"] == (
        "READBACK_COMPLETE"
    )


def test_post_dispatch_abi_mapping_readback_preserves_nested_scalars() -> None:
    from engine.authoritative_state.runtime_abi import _mapping_scalar_values_v1

    projected = _mapping_scalar_values_v1(
        {
            "regrowth": 0.125,
            "enabled": True,
            "nested": {"radius": 2, "profile": "wdl"},
            "vector": [1, 2, 3],
            "opaque": object(),
        }
    )

    assert projected == {
        "enabled": True,
        "nested": {"profile": "wdl", "radius": 2},
        "regrowth": 0.125,
        "vector": [1, 2, 3],
    }


def test_production_registry_declares_true_axis_roles() -> None:
    profile = next(
        profile
        for profile in default_profile_registry().profiles()
        if profile.profile_id == "eiel-production-runtime-v1"
    )

    assert profile.parameter_contracts[
        "env_cfg.patch_food_regrow_per_step"
    ].control_role == "independent_primitive"
    assert profile.parameter_contracts[
        "env_cfg.patch_food_regrow_per_step"
    ].wrc_axis is True
    assert profile.parameter_contracts["patch.initial_live_count"].alias_of == (
        "patch.initial_population"
    )
    assert profile.parameter_contracts["patch.initial_live_count"].wrc_axis is False
    assert profile.parameter_contracts["policy.hidden_width"].control_role == (
        "route_constant"
    )
    assert profile.parameter_contracts["policy.hidden_width"].wrc_axis is False
    assert "env_cfg.wdl_delayed_hazard_regrowth_suppression_enable" not in (
        profile.parameter_contracts
    )


def _single_control_readback_fixture(control: str, value: object) -> dict[str, object]:
    env_cfg = EnvConfig(width=16, height=16)
    pop_cfg = PopConfig(population_size=64, max_slots_cap=256)
    sim_cfg = SimConfig()
    section, field_name = control.split(".", 1)
    target = env_cfg if section == "env_cfg" else sim_cfg
    setattr(target, field_name, value)
    runner = SimpleNamespace(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg)
    build = SimpleNamespace(runner=runner, receipt={})
    source_projection = {"metadata": {field_name: value}}
    prepared = SimpleNamespace(
        source_projections=(source_projection,),
        source=SimpleNamespace(patch_p2_config={field_name: value}),
        checkpoint_projection_inputs={},
    )
    request = SimpleNamespace(
        configuration=SimpleNamespace(**{field_name: value}),
        envelope=SimpleNamespace(grid=(16, 16)),
    )
    checkpoint = {"metadata": {field_name: value}}
    post_dispatch = {
        "receipt_sha256": "c" * 64,
        "outer_arguments": [
            {
                "name": "production_config",
                "kind": "mapping",
                "scalar_values": {field_name: value},
            }
        ],
    }
    cell = {"cell_id": "cell-0", "parameters": {control: value}}
    metric_row = {"cell_id": "cell-0", "generation": 1, "alive": 64}
    return {
        "cells": (cell,),
        "builds": (build,),
        "requests": (request,),
        "prepared": prepared,
        "checkpoint_projections": (checkpoint,),
        "compact_result_rows": (metric_row,),
        "parameter_contracts": _production_contracts({control}),
        "post_dispatch_abi_receipt": post_dispatch,
    }


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("env_cfg.patch_food_regrow_per_step", 0.125),
        ("env_cfg.jungle_predator_base_risk", 0.04),
        ("env_cfg.eed_resource_a_capacity", 2.0),
        ("env_cfg.wdl_delayed_hazard_event_period_steps", 40),
        ("env_cfg.reproduce_try_prob", 0.3),
        ("sim_cfg.resource_maintenance_energy_cost_per_tick", 0.02),
    ],
)
def test_representative_control_families_bind_end_to_end(
    control: str,
    value: object,
) -> None:
    receipt = build_segment_control_readback_v1(
        profile_id="eiel-production-runtime-v1",
        segment_ordinal=1,
        start_generation=0,
        end_generation=1,
        **_single_control_readback_fixture(control, value),
    )

    row = receipt["cells"][0]["controls"][0]
    assert row["control"] == control
    assert row["requested"] == value
    assert row["compiled"] == value
    assert row["applied"] == value
    assert row["checkpoint_projection_value"] == value
    assert row["readback"] == value
    assert row["status"] == "READBACK_COMPLETE"
    assert receipt["cells"][0]["metric_binding_status"] == "BOUND"
    assert receipt["cells"][0]["authoritative_state_projection_sha256"]
