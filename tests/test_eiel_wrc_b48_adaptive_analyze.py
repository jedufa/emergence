from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.wrc_adaptive_rebinding import (
    materialize_wrc_adaptive_restore_plan_v1,
)
from tools.scripts.eiel_wrc_b48_adaptive_analyze import (
    AdaptiveAnalysisError,
    analyze_completed_stage,
    main,
    materialize_analysis_outputs,
)
from tools.scripts.eiel_wrc_b48_adaptive_stage import materialize_stage_plan


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = ROOT / "_spec/campaigns/v1/wrc_b48_adaptive"
ROUTE_SHA = "7" * 64


def _load(name: str):
    return json.loads((DESIGN_ROOT / name).read_text(encoding="utf-8"))


def _digest_map(cell_ids: list[str], marker: str) -> dict[str, str]:
    return {
        cell_id: canonical_sha256({"cell_id": cell_id, "marker": marker})
        for cell_id in cell_ids
    }


def _manifest(
    plan: dict,
    *,
    start_generation: int,
    end_generation: int,
    continued: bool = False,
    result: dict | None = None,
) -> dict:
    cell_ids = [cell["cell_id"] for cell in plan["cells"]]
    terminal = _digest_map(cell_ids, f"terminal-{end_generation}")
    canonical_terminal = _digest_map(cell_ids, f"canonical-{end_generation}")
    storage = _digest_map(cell_ids, f"storage-{end_generation}")
    parent = _digest_map(cell_ids, f"parent-{start_generation}") if continued else None
    rows = []
    for cell_index, cell_id in enumerate(cell_ids):
        for generation in range(start_generation, end_generation):
            rows.append(
                {
                    "schema_version": "eiel_architecture_b_compact_summary_v1",
                    "cell_id": cell_id,
                    "generation": generation,
                    "world": "patch",
                    "alive": 256 - cell_index % 16 - generation // 8,
                    "births": 2 + cell_index % 3,
                    "failed_births": cell_index % 2,
                    "deaths": 1 + generation % 2,
                    "moved": 100 + generation,
                    "gathered": 80 + generation,
                    "patch_food_stock_after_turnover": 1_000.0 - generation,
                    "energy_collected_exact_sum_after_turnover": 500.0
                    + generation,
                    "policy_energy_live_sum_after_turnover": 250.0 + generation,
                }
            )
    return {
        "schema_version": "eiel_campaign_batch_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": "eiel-production-engine-v1",
        "execution_route_sha256": ROUTE_SHA,
        "batch_id": "batch-0000",
        "cell_ids": cell_ids,
        "status": "completed",
        "attempts": 1,
        "checkpoint": {
            "schema_version": "eiel-production-checkpoint-v1",
            "generation": end_generation,
            "execution_route_sha256": ROUTE_SHA,
            "checkpoint_binding": {
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "qualification_key_sha256": plan["qualification_key_sha256"],
                "profile_id": plan["resolved_profile"],
                "provider_id": "eiel-production-engine-v1",
                "execution_route_sha256": ROUTE_SHA,
                "batch_id": "batch-0000",
                "cell_ids": cell_ids,
            },
            "cell_bundles": [
                {"cell_id": cell_id, "path": f"/{cell_id}.pt"}
                for cell_id in cell_ids
            ],
            "terminal_cell_digests": terminal,
            "checkpoint_identity_contract": {
                "terminal_projection_digest_sha256s": terminal,
                "parent_checkpoint_projection_digest_sha256s": parent,
                "canonical_terminal_state_digest_sha256s": canonical_terminal,
                "canonical_checkpoint_storage_digest_sha256s": storage,
            },
        },
        "metrics_rows": rows,
        "result": result or {},
        "error": None,
        "created_at": "2026-08-11T00:00:00Z",
        "updated_at": "2026-08-11T01:00:00Z",
    }


def _signed(value: dict) -> dict:
    return {**value, "receipt_sha256": canonical_sha256(value)}


def test_analyzes_exact_c1_with_existing_reducer_and_writes_outputs(tmp_path):
    plan = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    manifest = _manifest(plan, start_generation=0, end_generation=25)

    analysis = analyze_completed_stage(plan, binding, manifest, stage="C1")

    assert analysis["physical_cell_count"] == 48
    assert analysis["configuration_count"] == 16
    assert analysis["interval_observation_count"] == 1_200
    assert len(analysis["world_rows"]) == 48
    assert len(analysis["configuration_rows"]) == 16
    assert {row["replicate_count"] for row in analysis["configuration_rows"]} == {3}
    assert {row["observation_count"] for row in analysis["world_rows"]} == {25}
    assert analysis["checkpoint_row"]["continuity_status"] == "VERIFIED_EXACT_48"
    assert analysis["checkpoint_row"]["qualification_receipt_action"] == "unchanged"
    assert analysis["reducer_version"].startswith("eiel_wrc_reducer_v1")

    receipt = materialize_analysis_outputs(analysis, output_dir=tmp_path)
    assert receipt["qualification_receipt_action"] == "unchanged"
    assert set(receipt["outputs"]) == {
        "world_summary",
        "configuration_summary",
        "seed_divergence",
        "checkpoints",
        "world_reductions",
    }
    assert all(Path(row["path"]).is_file() for row in receipt["outputs"].values())
    assert len((tmp_path / "B48_EXPLORATORY_WORLD_SUMMARY.csv").read_text().splitlines()) == 49
    assert len(
        (tmp_path / "B48_EXPLORATORY_WORLD_REDUCTIONS.jsonl")
        .read_text()
        .splitlines()
    ) == 48

    restored, stage_manifest = materialize_stage_plan(
        plan,
        binding,
        [
            {
                "stage": "C2",
                "config_id": "C01",
                "tick_start": 4_000,
                "tick_end": 8_000,
                "parameter": "env_cfg.move_cost",
                "previous_value": 0.2,
                "new_value": 0.4,
                "source_WRC_ID": "P023",
                "reason": "bounded pressure shift",
                "evidence_used": receipt["receipt_sha256"],
                "envelope_status": "COVERED_APPROVED",
                "receipt_action": "unchanged",
            }
        ],
        stage="C2",
        analysis_receipt=receipt,
    )
    assert restored["execution"]["generations"] == 50
    assert stage_manifest["source_analysis_receipt_sha256"] == receipt[
        "receipt_sha256"
    ]


def test_rejects_incomplete_world_generation_coverage():
    plan = _load("B48_EXPLORATORY_C1_PLAN.json")
    manifest = _manifest(plan, start_generation=0, end_generation=25)
    manifest["metrics_rows"].pop()

    with pytest.raises(AdaptiveAnalysisError, match="48 x 25"):
        analyze_completed_stage(
            plan,
            _load("B48_EXPLORATORY_DESIGN_BINDING.json"),
            manifest,
            stage="C1",
        )


def test_analysis_cli_writes_bound_c1_outputs(tmp_path, capsys):
    plan = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    manifest = _manifest(plan, start_generation=0, end_generation=25)
    plan_path = tmp_path / "plan.json"
    binding_path = tmp_path / "binding.json"
    manifest_path = tmp_path / "manifest.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "analysis"

    assert main(
        [
            "--plan",
            str(plan_path),
            "--design-binding",
            str(binding_path),
            "--batch-manifest",
            str(manifest_path),
            "--stage",
            "C1",
            "--output-dir",
            str(output),
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["stage"] == "C1"
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert (output / "analysis_receipt.json").is_file()


def test_continued_stage_requires_matching_rebind_and_runtime_readback():
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    restored = materialize_wrc_adaptive_restore_plan_v1(
        source,
        restore_generation=50,
        parameter_arrays={"env_cfg.move_cost": [0.4] * 48},
    )
    adaptive = _signed(
        {
            "schema_version": "wrc_checkpoint_adaptive_rebinding_v1",
            "capability_id": "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1",
            "source_generation": 25,
            "restore_generation": 50,
        }
    )
    readback = _signed(
        {
            "schema_version": "wrc_checkpoint_adaptive_runtime_readback_v1",
            "adaptive_rebinding_receipt_sha256": adaptive["receipt_sha256"],
            "segment_control_readback_sha256": "8" * 64,
            "start_generation": 25,
            "end_generation": 30,
            "verified_cells": [],
            "status": "READBACK_COMPLETE",
        }
    )
    manifest = _manifest(
        restored,
        start_generation=25,
        end_generation=50,
        continued=True,
        result={
            "fresh_owner_restore_receipt": {
                "wrc_checkpoint_adaptive_rebinding": adaptive
            },
            "wrc_checkpoint_adaptive_runtime_readback": readback,
        },
    )
    transition = _signed(
        {
            "schema_version": "eiel_wrc_adaptive_campaign_transition_v1",
            "source_campaign_id": source["campaign_id"],
            "restore_campaign_id": restored["campaign_id"],
            "source_plan_sha256": source["plan_sha256"],
            "restore_plan_sha256": restored["plan_sha256"],
            "execution_route_sha256": ROUTE_SHA,
            "qualification_key_sha256": restored["qualification_key_sha256"],
            "batch_id": "batch-0000",
            "cell_ids": [cell["cell_id"] for cell in restored["cells"]],
            "source_generation": 25,
            "restore_generation": 50,
            "adaptive_rebinding_receipt_sha256": adaptive["receipt_sha256"],
            "source_batch_manifest_path": "/source/manifest.json",
            "restore_batch_manifest_path": "/restore/manifest.json",
            "qualification_receipt_action": "unchanged",
            "status": "READY_TO_RESUME",
        }
    )

    analysis = analyze_completed_stage(
        restored,
        _load("B48_EXPLORATORY_DESIGN_BINDING.json"),
        manifest,
        stage="C2",
        transition_receipt=transition,
    )
    assert analysis["checkpoint_row"]["adaptive_rebinding_receipt_sha256"] == adaptive[
        "receipt_sha256"
    ]
    assert analysis["checkpoint_row"][
        "adaptive_runtime_readback_receipt_sha256"
    ] == readback["receipt_sha256"]

    with pytest.raises(AdaptiveAnalysisError, match="transition receipt is absent"):
        analyze_completed_stage(
            restored,
            _load("B48_EXPLORATORY_DESIGN_BINDING.json"),
            deepcopy(manifest),
            stage="C2",
        )
