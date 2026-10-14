from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_wrc_adaptive_rebinding_qualification import (
    AdaptiveRebindingQualificationError,
    CAPABILITY_ID,
    QUALIFICATION_ID,
    REBOUND_PARAMETERS,
    main,
    execute_qualprep_packet,
    execute_qualification_packet,
    materialize_plan_packet,
    materialize_qualification_plans,
)
from engine.campaigns.orchestration import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = ROOT / "_spec/campaigns/v1/wrc_b48_adaptive"


def _base_plan():
    return json.loads(
        (DESIGN_ROOT / "B48_EXPLORATORY_C1_PLAN.json").read_text(
            encoding="utf-8"
        )
    )


def test_materializes_only_bounded_g3_to_g6_b48_rebinding_target(tmp_path):
    base = _base_plan()

    source, restore, static_contract, profile = materialize_qualification_plans(
        base, evidence_root=tmp_path
    )

    assert source["execution"]["generations"] == 3
    assert source["execution"]["tick_schedule"]["total_ticks"] == 480
    assert restore["execution"]["generations"] == 6
    assert restore["execution"]["tick_schedule"]["total_ticks"] == 960
    assert source["physical_batches"] == restore["physical_batches"]
    assert len(source["cells"]) == 48
    assert source["qualification_key"] == restore["qualification_key"]
    assert source["artifacts"] == restore["artifacts"]
    assert source["output"]["chronicle_profile"] == "notable"
    assert restore["output"] == source["output"]
    assert len({canonical_sha256(cell["parameters"]) for cell in source["cells"]}) == 16
    assert all(
        any(
            before["parameters"][name] != after["parameters"][name]
            for name in REBOUND_PARAMETERS
        )
        for before, after in zip(source["cells"], restore["cells"], strict=True)
    )
    assert static_contract["changed_parameter_names"] == sorted(REBOUND_PARAMETERS)
    assert len(static_contract["changed_cells"]) == 48
    assert profile["static_contract_receipt_sha256"] == static_contract[
        "receipt_sha256"
    ]
    assert profile["capability_id"] == CAPABILITY_ID
    assert profile["qualification_id"] == QUALIFICATION_ID
    assert profile["route_scope"]["broad_route_requalification"] is False
    assert profile["route_scope"]["tick_native_horizon"] is False
    assert profile["route_scope"]["mid_generation_checkpoint_restore"] is False
    impact = profile["qualification_impact_map"]
    assert impact["broad_requalification_required"] is False
    assert impact["cartesian_matrix_planned"] is False
    assert impact["exact_changed_layer"] == CAPABILITY_ID
    assert {
        row["classification"] for row in impact["existing_qualified_groups"]
    } == {
        "PRESERVED_UNCHANGED",
        "PRESERVED_COMPATIBLE_INTERFACE",
        "DELTA_QUALIFICATION_REQUIRED",
    }
    reporting = next(
        row
        for row in impact["existing_qualified_groups"]
        if row["group_id"] == "PRODUCTION-REPORTING-OBSERVATIONAL-CONTRACT"
    )
    assert reporting["classification"] == "PRESERVED_COMPATIBLE_INTERFACE"
    assert canonical_sha256(
        {key: value for key, value in profile.items() if key != "receipt_sha256"}
    ) == profile["receipt_sha256"]


def test_plan_command_writes_nonexecuting_packet(tmp_path, capsys):
    output = tmp_path / "packet"
    allocation_receipt = tmp_path / "allocation.json"
    allocation_receipt.write_text(
        json.dumps({"schema_version": "test-allocation-v1", "value": 1}),
        encoding="utf-8",
    )

    assert main(
        [
            "plan",
            "--base-plan",
            str(DESIGN_ROOT / "B48_EXPLORATORY_C1_PLAN.json"),
            "--output-root",
            str(output),
            "--allocation-diagnostic-receipt",
            str(allocation_receipt),
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)

    assert receipt["ok"] is True
    assert receipt["qualification_executed"] is False
    assert receipt["gpu_used"] is False
    assert receipt["status"] == "QUALPREP_PLAN_GENERATED"
    assert (output / "source_plan.json").is_file()
    assert (output / "restore_plan.json").is_file()
    assert (output / "static_contract_receipt.json").is_file()
    assert (output / "readiness_profile.json").is_file()
    assert json.loads((output / "allocation_diagnostic_receipt.json").read_text()) == {
        "schema_version": "test-allocation-v1",
        "value": 1,
    }
    assert receipt["allocation_diagnostic_receipt_sha256"] == canonical_sha256(
        {"schema_version": "test-allocation-v1", "value": 1}
    )


def test_rejects_non_b48_or_non_t160_base(tmp_path):
    base = _base_plan()
    base["physical_batches"][0]["cell_ids"] = base["physical_batches"][0][
        "cell_ids"
    ][:-1]
    unsigned = dict(base)
    unsigned.pop("plan_sha256")
    base["plan_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        AdaptiveRebindingQualificationError,
        match="resident B48/T160",
    ):
        materialize_qualification_plans(base, evidence_root=tmp_path)


def test_qualification_refuses_to_start_without_readiness(tmp_path):
    packet = tmp_path / "packet"
    materialize_plan_packet(
        DESIGN_ROOT / "B48_EXPLORATORY_C1_PLAN.json", output_root=packet
    )

    with pytest.raises(
        AdaptiveRebindingQualificationError,
        match="readiness receipt",
    ):
        execute_qualification_packet(packet)


def test_rehearses_complete_qualification_contract_without_runtime(
    tmp_path, monkeypatch
):
    packet = tmp_path / "packet"
    allocation_receipt_path = tmp_path / "allocation.json"
    allocation_receipt = {"schema_version": "test-allocation-v1", "value": 1}
    allocation_receipt_path.write_text(
        json.dumps(allocation_receipt), encoding="utf-8"
    )
    materialize_plan_packet(
        DESIGN_ROOT / "B48_EXPLORATORY_C1_PLAN.json",
        output_root=packet,
        allocation_diagnostic_receipt_path=allocation_receipt_path,
    )
    source = json.loads((packet / "source_plan.json").read_text())
    restore = json.loads((packet / "restore_plan.json").read_text())
    cell_ids = list(source["physical_batches"][0]["cell_ids"])
    source_digests = {cell_id: f"digest-{index}" for index, cell_id in enumerate(cell_ids)}
    adaptive_body = {
        "schema_version": "wrc_checkpoint_adaptive_rebinding_v1",
        "capability_id": CAPABILITY_ID,
        "source_generation": 3,
        "restore_generation": 6,
        "cell_ids": cell_ids,
        "changed_parameter_names": sorted(REBOUND_PARAMETERS),
        "changed_cells": [
            {
                "cell_id": cell_id,
                "changes": {
                    name: {
                        "source": source["cells"][index]["parameters"][name],
                        "restore": restore["cells"][index]["parameters"][name],
                    }
                    for name in REBOUND_PARAMETERS
                    if source["cells"][index]["parameters"][name]
                    != restore["cells"][index]["parameters"][name]
                },
            }
            for index, cell_id in enumerate(cell_ids)
        ],
    }
    adaptive = {
        **adaptive_body,
        "receipt_sha256": canonical_sha256(adaptive_body),
    }
    runtime_body = {
        "schema_version": "wrc_checkpoint_adaptive_runtime_readback_v1",
        "adaptive_rebinding_receipt_sha256": adaptive["receipt_sha256"],
        "verified_cells": [
            {
                "cell_id": cell_id,
                "verified_controls": [
                    name
                    for name in REBOUND_PARAMETERS
                    if source["cells"][index]["parameters"][name]
                    != restore["cells"][index]["parameters"][name]
                ],
            }
            for index, cell_id in enumerate(cell_ids)
        ],
        "status": "READBACK_COMPLETE",
    }
    runtime = {
        **runtime_body,
        "receipt_sha256": canonical_sha256(runtime_body),
    }

    class Provider:
        def translate_batch(self, _plan, _batch_id):
            return object()

        def prepare_fresh_owner_restore_checkpoint(self, *_args):
            return {"legacy": "exact"}

        def prepare_wrc_checkpoint_adaptive_rebinding_v1(self, *_args):
            return {"legacy": "exact"}

    class Registry:
        def get(self, _profile):
            return provider

    provider = Provider()
    registry = Registry()

    monkeypatch.setattr(
        "tools.scripts.eiel_wrc_adaptive_rebinding_qualification._run_host_proofs",
        lambda _root: {"receipt_sha256": "h" * 64},
    )
    def fake_preflight(*_args, **kwargs):
        assert kwargs["allocation_diagnostic_resource_receipt"] == allocation_receipt
        return {
            "ready_for_execution": True,
            "reasons": [],
            "execution_authority_reasons": [],
        }

    monkeypatch.setattr(
        "tools.scripts.eiel_wrc_adaptive_rebinding_qualification.preflight_campaign",
        fake_preflight,
    )

    def fake_execute(plan, *, run_root, provider_registry):
        assert provider_registry is registry
        generation = plan["execution"]["generations"]
        batch_id = plan["physical_batches"][0]["batch_id"]
        if generation == 3:
            checkpoint = {
                "generation": 3,
                "terminal_cell_digests": source_digests,
                "checkpoint_identity_contract": {
                    "terminal_projection_digest_sha256s": source_digests
                },
            }
            result = {}
        else:
            checkpoint = {"generation": 6}
            result = {
                "fresh_owner_restore_receipt": {
                    "source_checkpoint_cell_digests": source_digests,
                    "parent_checkpoint_projection_digest_sha256s": source_digests,
                    "reload_receipts": [
                        {
                            "cell_id": cell_id,
                            "source_projection_sha256": source_digests[cell_id],
                            "loaded_projection_sha256": source_digests[cell_id],
                        }
                        for cell_id in cell_ids
                    ],
                    "wrc_checkpoint_adaptive_rebinding": adaptive,
                },
                "wrc_checkpoint_adaptive_runtime_readback": runtime,
            }
        atomic_write_json(
            run_root / "batches" / batch_id / "manifest.json",
            {"checkpoint": checkpoint, "result": result},
        )
        return {"status": "completed"}

    monkeypatch.setattr(
        "tools.scripts.eiel_wrc_adaptive_rebinding_qualification.execute_campaign",
        fake_execute,
    )
    monkeypatch.setattr(
        "tools.scripts.eiel_wrc_adaptive_rebinding_qualification.prepare_adaptive_campaign_continuation_v1",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "t" * 64,
            "qualification_receipt_action": "unchanged",
            "restore_batch_manifest_path": str(
                packet / "restore_run" / "batches" / "batch-0000" / "manifest.json"
            ),
        },
    )

    readiness = execute_qualprep_packet(packet, provider_registry=registry)
    receipt = execute_qualification_packet(packet, provider_registry=registry)

    assert readiness["status"] == "READY_FOR_QUALIFICATION"
    assert readiness["qualification_executed"] is False
    assert receipt["decision"] == "QUALIFIED"
    assert receipt["physical_batch_size"] == 48
    assert receipt["broad_route_requalification_performed"] is False
    assert receipt["tick_native_or_mid_generation_scope"] is False
    assert receipt["readiness_receipt_sha256"] == readiness["receipt_sha256"]
    assert (packet / "qualification_receipt.json").is_file()
