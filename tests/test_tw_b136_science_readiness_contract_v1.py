from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.patch_runner_factory import (
    build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
)
from engine.campaigns.providers import default_execution_provider_registry
from engine.campaigns.tick_native import TICK_NATIVE_EXECUTION_ROUTE_V1
from engine.perf import cuda_architecture_b_direct_resident_wdl_jungle_v1 as wdl_jungle

ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "_spec/campaigns/v1/tw_rngv2"

WRC_ENVELOPE_SHA256 = "0db593666df84bfa200d6bed62cf1eef8c5713bfb67bce6db5e476e0e2034b46"
ACS_NATIVE_RUNTIME_SHA256 = (
    "9edde818faa5d33ea1dcc9481dc1ade77782a172404e9fdc04219dd91cd0c63e"
)
ACS_NATIVE_RUNTIME_MANIFEST_SHA256 = (
    "81d0de158743ad679ae489050e695eb20e2743ed0e3a07ea2a128ead5c782eed"
)

PROGRAMME_SPECS = (
    ("science_representative_3360tick_b136_acs0_v1.yaml", 0),
    ("science_representative_3360tick_b136_acs0_fresh_restore_v1.yaml", 0),
    ("science_representative_3360tick_b136_acs2_v1.yaml", 2),
    ("science_representative_3360tick_b136_acs2_fresh_restore_v1.yaml", 2),
    ("science_representative_5000tick_b136_acs2_v1.yaml", 2),
    ("science_representative_5000tick_b2_diagnostic_acs1_v1.yaml", 1),
    ("science_representative_5000tick_b2_diagnostic_acs2_v1.yaml", 2),
    ("science_representative_5000tick_b2_diagnostic_control_v1.yaml", 0),
    ("science_representative_5000tick_qualification_uninterrupted_v1.yaml", 2),
    ("science_representative_5000tick_restore_v1.yaml", 2),
    ("science_representative_5000tick_terminal_benchmark_v1.yaml", 0),
    ("science_representative_5000tick_v1.yaml", 0),
)

_GENERATION_AUTHORITY_KEYS = frozenset(
    {
        "generation",
        "generations",
        "ticks_per_generation",
        "physical_window_generation",
        "physical_window_generations",
    }
)


def _generation_authority_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            key for key, nested in value.items() if key in _GENERATION_AUTHORITY_KEYS
        } | {
            key
            for nested in value.values()
            for key in _generation_authority_keys(nested)
        }
    if isinstance(value, (list, tuple)):
        return {key for nested in value for key in _generation_authority_keys(nested)}
    return set()


@pytest.mark.parametrize(
    ("spec_name", "expected_acs_slots"),
    PROGRAMME_SPECS,
    ids=[name.removesuffix(".yaml") for name, _ in PROGRAMME_SPECS],
)
def test_tick_native_tw_science_readiness_contract(
    spec_name: str, expected_acs_slots: int
) -> None:
    source = load_campaign_spec(SPEC_ROOT / spec_name)
    plan = compile_campaign(source)
    payload = plan.payload(include_hash=False)
    execution = payload["execution"]
    source_execution = source["execution"]
    tick_native_control = execution["tick_native_control"]

    assert source_execution["execution_route_identity"] == (
        TICK_NATIVE_EXECUTION_ROUTE_V1
    )
    assert type(source_execution["total_ticks"]) is int
    assert source_execution["total_ticks"] > 0
    assert execution["total_ticks"] == source_execution["total_ticks"]
    assert tick_native_control["total_ticks"] == execution["total_ticks"]
    assert execution["time_authority"] == "absolute_tick_v1"
    assert tick_native_control["time_authority"] == "absolute_tick_v1"
    assert not _generation_authority_keys(source_execution)
    assert not _generation_authority_keys(execution)

    assert tick_native_control["event_window_capacity"] == 41_804
    assert (
        source["parameters"]["cuda_architecture_b.cpu_request_staging_enable"] is False
    )
    assert source["parameters"]["tw_rngv2.semantics_version"] == 2

    qualification_key = plan.qualification_key
    fixed_invariants = qualification_key["fixed_invariants"]
    assert (
        fixed_invariants["wrc.parameter_variability_envelope_sha256"]
        == WRC_ENVELOPE_SHA256
    )
    assert (
        fixed_invariants["parameter.cuda_architecture_b.cpu_request_staging_enable"]
        is False
    )
    assert fixed_invariants["parameter.tw_rngv2.semantics_version"] == 2

    provider = default_execution_provider_registry().get(plan.profile_id)
    assert provider is not None
    route = provider.execution_route_contract(payload)
    route_components = {
        component["component_id"]: component
        for component in route.payload()["components"]
    }
    route_qualification = route_components["qualification_identity"]
    assert (
        route_qualification["fixed_invariants"][
            "wrc.parameter_variability_envelope_sha256"
        ]
        == WRC_ENVELOPE_SHA256
    )

    artifacts = source["artifacts"]
    assert artifacts["acs_native_runtime_extension"]["sha256"] == (
        ACS_NATIVE_RUNTIME_SHA256
    )
    assert artifacts["acs_native_runtime_extension_manifest"]["sha256"] == (
        ACS_NATIVE_RUNTIME_MANIFEST_SHA256
    )
    assert artifacts["acs_native_runtime_extension"]["path"]
    assert artifacts["acs_native_runtime_extension_manifest"]["path"]

    assert source["checkpoints"]["acs_slots"] == expected_acs_slots
    assert plan.acs_checkpoint_slots == expected_acs_slots
    assert execution["acs_checkpoint_slots"] == expected_acs_slots
    assert source["checkpoints"]["native_streamed_checkpoint"] is True
    assert plan.native_streamed_checkpoint is True
    assert execution["native_streamed_checkpoint"] is True

    if spec_name in {
        "science_representative_5000tick_qualification_uninterrupted_v1.yaml",
        "science_representative_5000tick_restore_v1.yaml",
    }:
        assert source["artifact_lifecycle"]["lifecycle_profile"] == "QUALIFICATION"
        assert (
            plan.payload()["artifact_lifecycle"]["lifecycle_profile"] == "QUALIFICATION"
        )
        assert source_execution["total_ticks"] == 5000
        assert source_execution["checkpoint_ticks"] == [2500, 5000]
        assert len(plan.cells) == 136
        assert [len(batch.cell_ids) for batch in plan.batches] == [136]
        assert plan.resource_estimate["compiled_batch_count"] == 1
        if "restore" in spec_name:
            assert source_execution["deliberate_restore_tick"] == 2500
        else:
            assert source_execution.get("deliberate_restore_tick") is None


def test_physical_b136_acs0_fresh_restore_preserves_acs2_restore_contract() -> None:
    spec_name = "science_representative_3360tick_b136_acs0_fresh_restore_v1.yaml"
    acs0 = load_campaign_spec(SPEC_ROOT / spec_name)
    acs2 = load_campaign_spec(
        SPEC_ROOT / "science_representative_3360tick_b136_acs2_fresh_restore_v1.yaml"
    )

    execution = acs0["execution"]
    assert execution["total_ticks"] == 3360
    assert execution["checkpoint_ticks"] == [3359, 3360]
    assert execution["deliberate_restore_tick"] == 3359
    assert execution["batch"] == {"mode": "explicit", "cells": 136}
    assert acs0["checkpoints"]["acs_slots"] == 0
    assert acs0["checkpoints"]["acs_capture_instrumentation"] is True
    assert acs0["checkpoints"]["native_streamed_checkpoint"] is True

    for section in (
        "run_contract",
        "world",
        "policy",
        "selection",
        "parameters",
        "matrix",
        "artifact_lifecycle",
        "artifacts",
        "analysis",
        "comparisons",
    ):
        assert acs0[section] == acs2[section]

    assert acs0["execution"] == acs2["execution"]
    assert {
        key: value for key, value in acs0["checkpoints"].items() if key != "acs_slots"
    } == {
        key: value for key, value in acs2["checkpoints"].items() if key != "acs_slots"
    }


def test_physical_b136_sync_async_restore_share_one_heterogeneous_wrc_matrix() -> None:
    spec_names = (
        "science_representative_3360tick_b136_acs0_v1.yaml",
        "science_representative_3360tick_b136_acs0_fresh_restore_v1.yaml",
        "science_representative_3360tick_b136_acs2_v1.yaml",
        "science_representative_3360tick_b136_acs2_fresh_restore_v1.yaml",
    )
    controls = (
        "env_cfg.jungle_predator_base_risk",
        "env_cfg.patch_food_gather_amount",
        "env_cfg.patch_movement_cost_energy_scale",
        "env_cfg.wdl_delayed_hazard_radius",
    )
    matrix_digests = set()
    for spec_name in spec_names:
        plan = compile_campaign(load_campaign_spec(SPEC_ROOT / spec_name))
        rows = [
            {
                "cell_id": cell.cell_id,
                "controls": {control: cell.parameters[control] for control in controls},
                "seeds": dict(cell.seeds),
            }
            for cell in plan.cells
        ]
        matrix_digests.add(canonical_sha256({"controls": controls, "rows": rows}))
        assert len(rows) == 136
        assert len({canonical_sha256(row["controls"]) for row in rows}) == 136
        assert {
            cell.parameters["env_cfg.jungle_predator_base_risk"] for cell in plan.cells
        } == {0.0, 0.02}
        assert {
            cell.parameters["wdl.cue_visible_at_decision"] for cell in plan.cells
        } == {False}
        assert {
            cell.parameters["env_cfg.patch_food_gather_amount"] for cell in plan.cells
        } == {0.5, 1.0}
        assert {
            cell.parameters["env_cfg.patch_movement_cost_energy_scale"]
            for cell in plan.cells
        } == {value / 4 for value in range(17)}
        assert {
            cell.parameters["env_cfg.wdl_delayed_hazard_radius"] for cell in plan.cells
        } == {0, 3}

        assert matrix_digests == {
            "adceffe3e29bcd5fe78168a88e36a1c108f17545de1f6610782fd14c3033f863"
        }


@pytest.mark.parametrize(
    "spec_name",
    (
        "science_representative_3360tick_b136_acs0_v1.yaml",
        "science_representative_3360tick_b136_acs0_fresh_restore_v1.yaml",
        "science_representative_3360tick_b136_acs2_v1.yaml",
        "science_representative_3360tick_b136_acs2_fresh_restore_v1.yaml",
        "science_representative_5000tick_b136_acs2_v1.yaml",
    ),
    ids=lambda name: name.removesuffix(".yaml"),
)
def test_all_physical_b136_cells_build_through_wdl_jungle_eed_factory_cpu(
    tmp_path: Path, spec_name: str
) -> None:
    source = load_campaign_spec(SPEC_ROOT / spec_name)
    plan = compile_campaign(source)
    assert len(plan.cells) == 136

    artifacts = source["artifacts"]
    assert "predator_base_risk" in (
        wdl_jungle._JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
    )
    jungle_abi = (
        ROOT / "engine/perf/cuda_architecture_b_flattened_jungle_controller_v4.hpp"
    ).read_text(encoding="utf-8")
    assert "predator_base_risk_by_cell" in jungle_abi
    for index, cell in enumerate(plan.cells):
        built = build_patch_wdl_jungle_eed_fixed_mlp_runner_v1(
            cell,
            tmp_path / spec_name / str(index),
            generations=1,
            ticks_per_generation=160,
            artifact_path=artifacts["cuda_extension"]["path"],
            qualification_receipt_path=artifacts["qualification_receipt"]["path"],
            reporting_receipt_path=artifacts["reporting_receipt"]["path"],
            execution_backend="cpu_native",
            runner_constructor=lambda **kwargs: SimpleNamespace(**kwargs),
        )

        env_cfg = built.runner.env_cfg
        assert (
            env_cfg.wdl_delayed_hazard_cue_enable
            is cell.parameters["wdl.cue_visible_at_decision"]
        )
        assert env_cfg.jungle_predator_pressure_enable is True
        assert env_cfg.jungle_juvenile_predator_vulnerability_enable is True
        assert env_cfg.jungle_terrain_protection_enable is True
        if "env_cfg.jungle_predator_base_risk" in cell.parameters:
            applied = built.receipt["jungle"]["primitive_controls"]["applied"]
            assert (
                applied["env_cfg.jungle_predator_base_risk"]
                == cell.parameters["env_cfg.jungle_predator_base_risk"]
            )
        assert built.receipt["world_profile"] == ("patch-p2-wdl-v2-jungle-v4-eed")
        assert built.runner.sim_cfg.native_runtime_use_precomputed_drive_noise is False
