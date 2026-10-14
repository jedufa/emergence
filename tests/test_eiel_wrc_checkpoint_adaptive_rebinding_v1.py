from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.providers import (
    CampaignProviderError,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
    WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1,
)
from engine.campaigns.registry import default_profile_registry
from engine.campaigns.wrc_variability import (
    INVARIANT_NAME,
    build_wrc_parameter_variability_envelope_v1,
)
from engine.campaigns.wrc_adaptive_rebinding import (
    WRCCheckpointAdaptiveRebindingError,
    materialize_wrc_adaptive_restore_plan_v1,
    materialize_wrc_adaptive_runtime_projections_v1,
    validate_wrc_adaptive_runtime_readback_v1,
)


CELL_IDS = ("cell-a", "cell-b")
ROUTE_SHA = "6" * 64


def _profile():
    return next(
        profile
        for profile in default_profile_registry().profiles()
        if profile.profile_id == "eiel-production-runtime-v1"
    )


def _parameters(*, move_cost: tuple[float, float], apply: bool = True):
    return {
        "env_cfg.move_cost": move_cost,
        "env_cfg.patch_movement_cost_apply_enable": (apply, apply),
        "env_cfg.patch_movement_cost_energy_scale": (1.0, 1.0),
        "patch.capacity": (512, 512),
    }


def _request(
    *,
    batch_id: str,
    generations: int,
    parameters: dict[str, tuple[Any, Any]],
) -> ResidentBatchRequestV1:
    return ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id=batch_id,
        cell_ids=CELL_IDS,
        generations=generations,
        ticks_per_generation=160,
        checkpoint_milestones=(generations,),
        parameter_arrays=tuple(sorted(parameters.items())),
        seed_arrays=(("ecology", (101, 202)), ("task", (1001, 2002))),
        agent_id_allocation_generation_count=generations,
        checkpoint_storage_mode="disk",
        reporting_enabled=True,
        execution_route_sha256=ROUTE_SHA,
    )


def _plan(
    *,
    generations: int,
    parameters: dict[str, tuple[Any, Any]],
    envelope_names: set[str] | None = None,
) -> dict[str, Any]:
    profile = _profile()
    envelope = build_wrc_parameter_variability_envelope_v1(
        profile,
        world_base="patch",
        world_overlays=("eed_resource_impact", "jungle", "wdl_delayed_hazard"),
        requested_parameter_names=envelope_names or {"env_cfg.move_cost"},
    )
    assert envelope is not None
    qualification = {
        "fixed_invariants": {
            INVARIANT_NAME: envelope.envelope_sha256,
            "runtime.world_base": "patch",
            "runtime.policy_family": "fixed_mlp",
        },
        "variability_envelope": envelope.payload(),
    }
    plan: dict[str, Any] = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "adaptive-rebind-test",
        "description": "test",
        "no_claims": "test only",
        "resolved_profile": profile.profile_id,
        "qualification_key": qualification,
        "qualification_key_sha256": canonical_sha256(qualification),
        "execution": {
            "generations": generations,
            "ticks_per_generation": 160,
            "checkpoints": [generations],
            "checkpoint_storage_mode": "disk",
            "budgets": {},
            "tick_schedule": {
                "schema_version": "eiel-tick-schedule-v1",
                "total_ticks": generations * 160,
                "science_sample_interval_ticks": None,
                "legacy_turnover_interval_ticks": None,
                "legacy_projection": {
                    "generations": generations,
                    "ticks_per_generation": 160,
                },
            },
        },
        "cells": [
            {
                "ordinal": ordinal,
                "cell_id": cell_id,
                "branch_id": f"branch-{ordinal}",
                "rng_lineage_id": f"rng-{ordinal}",
                "dimensions": {"configuration": ordinal},
                "parameters": {
                    name: values[ordinal] for name, values in parameters.items()
                },
                "seeds": {"ecology": (101, 202)[ordinal], "task": (1001, 2002)[ordinal]},
            }
            for ordinal, cell_id in enumerate(CELL_IDS)
        ],
        "physical_batches": [
            {"ordinal": 0, "batch_id": "batch-0000", "cell_ids": list(CELL_IDS)}
        ],
        "comparisons": {"branches": []},
        "analysis": {},
        "artifacts": {
            "cuda_extension": {"path": "/qualified/runtime.so", "sha256": "3" * 64},
            "qualification_receipt": {"path": "/qualified/q.json", "sha256": "4" * 64},
            "reporting_receipt": {"path": "/qualified/r.json", "sha256": "5" * 64},
        },
        "output": {"runtime_root": "/runtime"},
        "resource_estimate": {},
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _rebind(
    *,
    source_parameters: dict[str, tuple[Any, Any]] | None = None,
    restore_parameters: dict[str, tuple[Any, Any]] | None = None,
    envelope_names: set[str] | None = None,
):
    source_parameters = source_parameters or _parameters(move_cost=(0.2, 0.2))
    restore_parameters = restore_parameters or _parameters(move_cost=(0.4, 0.8))
    provider = ProductionEngineLifecycleV1()
    source_plan = _plan(
        generations=25,
        parameters=source_parameters,
        envelope_names=envelope_names,
    )
    restore_plan = _plan(
        generations=50,
        parameters=restore_parameters,
        envelope_names=envelope_names,
    )
    source_request = _request(
        batch_id="stage-c1", generations=25, parameters=source_parameters
    )
    restore_request = _request(
        batch_id="stage-c2", generations=50, parameters=restore_parameters
    )
    checkpoint = {
        "schema_version": provider.checkpoint_schema_version,
        "generation": 25,
        "checkpoint_binding": provider._checkpoint_binding(
            source_plan, source_request
        ),
        "world_state": {"must": "remain exact"},
        "terminal_cell_digests": {cell_id: str(index) * 64 for index, cell_id in enumerate(CELL_IDS, 1)},
    }
    checkpoint["checkpoint_identity_contract"] = {
        "terminal_projection_digest_sha256s": checkpoint[
            "terminal_cell_digests"
        ]
    }
    rebound = provider.prepare_wrc_checkpoint_adaptive_rebinding_v1(
        source_plan,
        restore_plan,
        source_request,
        restore_request,
        checkpoint,
    )
    return provider, source_plan, restore_plan, source_request, restore_request, checkpoint, rebound


def test_approved_rebinding_preserves_checkpoint_and_binds_declared_changes():
    provider, _source_plan, restore_plan, _source_request, restore_request, checkpoint, rebound = _rebind()

    assert rebound["world_state"] is checkpoint["world_state"]
    assert rebound["terminal_cell_digests"] == checkpoint["terminal_cell_digests"]
    assert rebound["checkpoint_binding"] == provider._checkpoint_binding(
        restore_plan, restore_request
    )
    receipt = rebound[WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1]
    assert receipt["capability_id"] == "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1"
    assert receipt["source_generation"] == 25
    assert receipt["restore_generation"] == 50
    assert receipt["changed_parameter_names"] == ["env_cfg.move_cost"]
    assert [row["cell_id"] for row in receipt["changed_cells"]] == list(CELL_IDS)


def test_runtime_projection_overlay_changes_only_declared_metadata():
    import torch

    *_prefix, rebound = _rebind()
    adaptive = rebound[WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1]
    tensors = (
        {"state": torch.tensor([1, 2])},
        {"state": torch.tensor([3, 4])},
    )
    projections = tuple(
        {
            "schema_version": "test-projection-v1",
            "metadata": {
                "cell_id": cell_id,
                "move_cost": 0.2,
                "preserved": "exact",
                "import_digest": "source",
            },
            "tensors": cell_tensors,
        }
        for cell_id, cell_tensors in zip(CELL_IDS, tensors, strict=True)
    )

    runtime = materialize_wrc_adaptive_runtime_projections_v1(
        adaptive_rebinding=adaptive,
        source_projections=projections,
    )

    assert [row["metadata"]["move_cost"] for row in runtime] == [0.4, 0.8]
    assert all(row["metadata"]["preserved"] == "exact" for row in runtime)
    assert all(
        after["tensors"] is before["tensors"]
        for before, after in zip(projections, runtime, strict=True)
    )
    assert all(before["metadata"]["move_cost"] == 0.2 for before in projections)
    assert all(after is not before for before, after in zip(projections, runtime, strict=True))


def test_runtime_projection_overlay_rebinds_only_inheritance_base_identity():
    import torch

    from engine.perf.cxxbackbone10_resident_state_owner import projection_digest
    from engine.perf.inheritance_sidecar_identity_v1 import (
        INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
        inheritance_sidecar_payload_sha256_v1,
    )

    *_prefix, rebound = _rebind()
    adaptive = rebound[WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1]
    projections = []
    for ordinal, cell_id in enumerate(CELL_IDS):
        projection = {
            "schema_version": "test-projection-v1",
            "metadata": {
                "cell_id": cell_id,
                "seed": 100 + ordinal,
                "generation": 25,
                "tick": 0,
                "move_cost": 0.2,
                "import_digest": "source",
            },
            "tensors": {"state": torch.tensor([ordinal, ordinal + 1])},
        }
        payload = {"storage": torch.tensor([ordinal + 2])}
        projection["genome_upgrade_v2_checkpoint_projection"] = {
            **payload,
            "identity": {
                "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
                "cell_id": cell_id,
                "seed": 100 + ordinal,
                "generation": 25,
                "tick": 0,
                "base_projection_sha256": projection_digest(projection),
                "payload_sha256": inheritance_sidecar_payload_sha256_v1(payload),
            },
        }
        projections.append(projection)

    runtime = materialize_wrc_adaptive_runtime_projections_v1(
        adaptive_rebinding=adaptive,
        source_projections=projections,
    )

    for source, restored in zip(projections, runtime, strict=True):
        source_sidecar = source["genome_upgrade_v2_checkpoint_projection"]
        restored_sidecar = restored["genome_upgrade_v2_checkpoint_projection"]
        restored_base = dict(restored)
        restored_base.pop("genome_upgrade_v2_checkpoint_projection")
        assert restored_sidecar["storage"] is source_sidecar["storage"]
        assert restored_sidecar["identity"] == {
            **source_sidecar["identity"],
            "base_projection_sha256": projection_digest(restored_base),
        }
        assert source_sidecar["identity"]["base_projection_sha256"] != (
            restored_sidecar["identity"]["base_projection_sha256"]
        )


def test_stage_plan_materialization_is_deterministic_and_preserves_identity():
    source_parameters = _parameters(move_cost=(0.2, 0.2))
    source = _plan(generations=25, parameters=source_parameters)
    restored = materialize_wrc_adaptive_restore_plan_v1(
        source,
        restore_generation=50,
        parameter_arrays={"env_cfg.move_cost": (0.4, 0.8)},
    )
    repeated = materialize_wrc_adaptive_restore_plan_v1(
        source,
        restore_generation=50,
        parameter_arrays={"env_cfg.move_cost": (0.4, 0.8)},
    )

    assert restored == repeated
    assert restored["plan_sha256"] != source["plan_sha256"]
    assert restored["execution"]["generations"] == 50
    assert restored["execution"]["checkpoints"] == [25, 50]
    assert restored["execution"]["tick_schedule"]["total_ticks"] == 8_000
    assert restored["execution"]["tick_schedule"]["legacy_projection"] == {
        "generations": 50,
        "ticks_per_generation": 160,
    }
    assert restored["qualification_key"] == source["qualification_key"]
    assert restored["artifacts"] == source["artifacts"]
    for before, after in zip(source["cells"], restored["cells"], strict=True):
        for field in (
            "ordinal",
            "cell_id",
            "branch_id",
            "rng_lineage_id",
            "dimensions",
            "seeds",
        ):
            assert after[field] == before[field]


def test_unchanged_rebinding_is_identical_to_legacy_restore():
    parameters = _parameters(move_cost=(0.2, 0.2))
    provider = ProductionEngineLifecycleV1()
    plan = _plan(generations=25, parameters=parameters)
    source = _request(batch_id="source", generations=25, parameters=parameters)
    restore = replace(source, batch_id="restore")
    checkpoint = {
        "schema_version": provider.checkpoint_schema_version,
        "generation": 25,
        "checkpoint_binding": provider._checkpoint_binding(plan, source),
    }

    legacy = provider.prepare_fresh_owner_restore_checkpoint(
        plan, source, restore, checkpoint
    )
    adaptive_entry = provider.prepare_wrc_checkpoint_adaptive_rebinding_v1(
        plan, plan, source, restore, checkpoint
    )

    assert adaptive_entry == legacy

    terminal_digests = {cell_id: "d" * 64 for cell_id in CELL_IDS}
    legacy_checkpoint = {
        **legacy,
        "terminal_cell_digests": terminal_digests,
        "checkpoint_identity_contract": {
            "terminal_projection_digest_sha256s": terminal_digests,
        },
    }
    legacy_receipt = provider._fresh_owner_restore_receipt(
        request=restore,
        checkpoint=legacy_checkpoint,
        reload_receipts=[],
        terminal_cell_digests=terminal_digests,
    )
    assert "wrc_checkpoint_adaptive_rebinding" not in legacy_receipt


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (
            lambda values: values.update({"env_cfg.move_cost": (-0.1, 0.8)}),
            "PARAMETER_VALUE_OUT_OF_RANGE",
        ),
        (
            lambda values: values.update({"patch.capacity": (1024, 1024)}),
            "PARAMETER_NOT_IN_QUALIFIED_ENVELOPE",
        ),
        (
            lambda values: values.update(
                {
                    "env_cfg.patch_movement_cost_apply_enable": (False, False),
                    "env_cfg.patch_movement_cost_energy_scale": (2.0, 2.0),
                }
            ),
            "PARAMETER_CONTEXT_INACTIVE",
        ),
    ],
)
def test_rebinding_rejects_outside_envelope_qualification_and_inactive_context(
    mutator, error
):
    restored = _parameters(move_cost=(0.4, 0.8))
    mutator(restored)
    with pytest.raises(CampaignProviderError, match=error):
        _rebind(restore_parameters=restored)


def test_rebinding_rejects_control_change_that_deactivates_unchanged_parameter():
    restored = _parameters(move_cost=(0.2, 0.2), apply=False)

    with pytest.raises(CampaignProviderError, match="PARAMETER_CONTEXT_INACTIVE"):
        _rebind(
            restore_parameters=restored,
            envelope_names={
                "env_cfg.move_cost",
                "env_cfg.patch_movement_cost_apply_enable",
                "env_cfg.patch_movement_cost_energy_scale",
            },
        )


def test_rebinding_rejects_invalid_cross_parameter_wdl_context():
    source = {
        **_parameters(move_cost=(0.2, 0.2)),
        "wdl.delay_ticks": (8, 8),
        "wdl.cue_duration_ticks": (1, 1),
        "wdl.active_duration_ticks": (6, 6),
        "env_cfg.wdl_delayed_hazard_event_period_steps": (20, 20),
    }
    restored = {
        **source,
        "wdl.active_duration_ticks": (24, 24),
    }

    with pytest.raises(CampaignProviderError, match="PARAMETER_CONTEXT_INVALID"):
        _rebind(
            source_parameters=source,
            restore_parameters=restored,
            envelope_names={
                "env_cfg.move_cost",
                "wdl.active_duration_ticks",
                "env_cfg.wdl_delayed_hazard_event_period_steps",
            },
        )


def test_rebinding_rejects_mixed_fixed_mlp_wdl_batch_context():
    source = {
        **_parameters(move_cost=(0.2, 0.2)),
        "wdl.delay_ticks": (8, 8),
        "wdl.cue_duration_ticks": (1, 1),
        "wdl.active_duration_ticks": (6, 6),
        "env_cfg.wdl_delayed_hazard_event_period_steps": (40, 40),
    }
    restored = {
        **source,
        "wdl.active_duration_ticks": (6, 24),
    }

    with pytest.raises(CampaignProviderError, match="PARAMETER_CONTEXT_UNREPRESENTABLE"):
        _rebind(
            source_parameters=source,
            restore_parameters=restored,
            envelope_names={
                "env_cfg.move_cost",
                "wdl.active_duration_ticks",
                "env_cfg.wdl_delayed_hazard_event_period_steps",
            },
        )


def test_rebinding_rejects_seed_or_lineage_mapping_change():
    provider = ProductionEngineLifecycleV1()
    source_parameters = _parameters(move_cost=(0.2, 0.2))
    restore_parameters = _parameters(move_cost=(0.4, 0.8))
    source_plan = _plan(generations=25, parameters=source_parameters)
    restore_plan = _plan(generations=50, parameters=restore_parameters)
    restore_plan["cells"][0]["rng_lineage_id"] = "different-rng"
    unsigned = dict(restore_plan)
    unsigned.pop("plan_sha256")
    restore_plan["plan_sha256"] = canonical_sha256(unsigned)
    source_request = _request(batch_id="source", generations=25, parameters=source_parameters)
    restore_request = _request(batch_id="restore", generations=50, parameters=restore_parameters)
    checkpoint = {
        "schema_version": provider.checkpoint_schema_version,
        "generation": 25,
        "checkpoint_binding": provider._checkpoint_binding(source_plan, source_request),
    }

    with pytest.raises(CampaignProviderError, match="NON_ADAPTIVE_PLAN_CHANGE"):
        provider.prepare_wrc_checkpoint_adaptive_rebinding_v1(
            source_plan,
            restore_plan,
            source_request,
            restore_request,
            checkpoint,
        )


def test_restore_receipt_validates_adaptive_authority():
    provider, _source_plan, _restore_plan, _source_request, restore_request, checkpoint, rebound = _rebind()
    receipt = provider._fresh_owner_restore_receipt(
        request=restore_request,
        checkpoint=rebound,
        reload_receipts=[{"cell_id": cell_id} for cell_id in CELL_IDS],
        terminal_cell_digests={cell_id: "f" * 64 for cell_id in CELL_IDS},
    )

    assert receipt["wrc_checkpoint_adaptive_rebinding"]["receipt_sha256"] == rebound[
        WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1
    ]["receipt_sha256"]

    damaged = deepcopy(rebound)
    damaged[WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1]["restore_generation"] = 51
    with pytest.raises(CampaignProviderError, match="RECEIPT_INVALID"):
        provider._fresh_owner_restore_receipt(
            request=restore_request,
            checkpoint=damaged,
            reload_receipts=[],
            terminal_cell_digests={cell_id: "f" * 64 for cell_id in CELL_IDS},
        )


def test_runtime_readback_proves_each_rebound_cell_and_fails_closed(tmp_path):
    *_prefix, rebound = _rebind()
    adaptive = rebound[WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1]
    expected_by_cell = {
        row["cell_id"]: row["changes"]["env_cfg.move_cost"]["restore"]
        for row in adaptive["changed_cells"]
    }
    readback = {
        "schema_version": "eiel_production_control_readback_v1",
        "profile_id": "eiel-production-runtime-v1",
        "segment_ordinal": 0,
        "start_generation": 25,
        "end_generation": 50,
        "post_dispatch_abi_receipt_sha256": "9" * 64,
        "cells": [
            {
                "cell_id": cell_id,
                "controls": [
                    {
                        "control": "env_cfg.move_cost",
                        "requested": expected,
                        "compiled": expected,
                        "applied": expected,
                        "readback": expected,
                        "status": "READBACK_COMPLETE",
                    }
                ],
            }
            for cell_id, expected in expected_by_cell.items()
        ],
    }
    readback["receipt_sha256"] = canonical_sha256(readback)
    path = tmp_path / "readback.json"
    path.write_text(json.dumps(readback), encoding="utf-8")
    descriptors = [
        {
            "path": str(path),
            "receipt_sha256": readback["receipt_sha256"],
            "start_generation": 25,
            "end_generation": 50,
        }
    ]

    verified = validate_wrc_adaptive_runtime_readback_v1(
        adaptive_rebinding=adaptive,
        production_control_readbacks=descriptors,
    )
    assert verified["status"] == "READBACK_COMPLETE"
    assert [row["cell_id"] for row in verified["verified_cells"]] == list(CELL_IDS)

    readback["cells"][0]["controls"][0]["readback"] = 99.0
    readback["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in readback.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(readback), encoding="utf-8")
    descriptors[0]["receipt_sha256"] = readback["receipt_sha256"]
    with pytest.raises(
        WRCCheckpointAdaptiveRebindingError,
        match="ADAPTIVE_RUNTIME_READBACK_MISMATCH",
    ):
        validate_wrc_adaptive_runtime_readback_v1(
            adaptive_rebinding=adaptive,
            production_control_readbacks=descriptors,
        )
