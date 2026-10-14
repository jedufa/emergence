from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_wrc_b48_adaptive_stage import (
    AdaptiveStageError,
    INTERVENTION_FIELDS,
    main,
    materialize_stage_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = ROOT / "_spec/campaigns/v1/wrc_b48_adaptive"
SOURCE_STAGE = {
    "C2": ("C1", 25, 4_000),
    "C3": ("C2", 50, 8_000),
    "C4": ("C3", 75, 12_000),
    "C5": ("C4", 100, 16_000),
}


def _load(name: str):
    return json.loads((DESIGN_ROOT / name).read_text(encoding="utf-8"))


def _row(**overrides):
    row = {
        "stage": "C2",
        "config_id": "C01",
        "tick_start": 4_000,
        "tick_end": 8_000,
        "parameter": "env_cfg.move_cost",
        "previous_value": 0.2,
        "new_value": 0.4,
        "source_WRC_ID": "P023",
        "reason": "bounded pressure shift",
        "evidence_used": "",
        "envelope_status": "COVERED_APPROVED",
        "receipt_action": "unchanged",
    }
    row.update(overrides)
    return row


def _analysis_receipt(source, binding, transition_stage):
    source_stage, generation, tick_end = SOURCE_STAGE[transition_stage]
    artifact_path = (DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv").resolve()
    artifact_raw = artifact_path.read_bytes()
    body = {
        "schema_version": "eiel_wrc_b48_adaptive_analysis_outputs_v1",
        "stage": source_stage,
        "plan_sha256": source["plan_sha256"],
        "design_binding_receipt_sha256": binding["receipt_sha256"],
        "terminal_generation": generation,
        "tick_end": tick_end,
        "checkpoint_sha256": "1" * 64,
        "reducer_version": "eiel_wrc_reducer_v1_1",
        "reducer_config_sha256": "2" * 64,
        "analysis_sha256": "3" * 64,
        "outputs": {
            name: {
                "path": str(artifact_path),
                "sha256": hashlib.sha256(artifact_raw).hexdigest(),
                "bytes": len(artifact_raw),
            }
            for name in (
                "world_summary",
                "configuration_summary",
                "seed_divergence",
                "checkpoints",
                "world_reductions",
            )
        },
        "qualification_receipt_action": "unchanged",
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def test_materializes_one_config_change_for_exactly_three_replicates():
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    analysis = _analysis_receipt(source, binding, "C2")

    restored, manifest = materialize_stage_plan(
        source,
        binding,
        [_row(evidence_used=analysis["receipt_sha256"])],
        stage="C2",
        analysis_receipt=analysis,
    )

    c01_ids = {
        row["cell_id"] for row in binding["rows"] if row["config_id"] == "C01"
    }
    changed = [
        cell
        for before, cell in zip(source["cells"], restored["cells"], strict=True)
        if before["parameters"] != cell["parameters"]
    ]
    assert {cell["cell_id"] for cell in changed} == c01_ids
    assert all(cell["parameters"]["env_cfg.move_cost"] == 0.4 for cell in changed)
    assert restored["execution"]["generations"] == 50
    assert restored["execution"]["tick_schedule"]["total_ticks"] == 8_000
    assert restored["qualification_key"] == source["qualification_key"]
    assert restored["artifacts"] == source["artifacts"]
    assert manifest["changed_configuration_count"] == 1
    assert manifest["changed_cell_count"] == 3
    assert {row["cell_id"] for row in manifest["cell_overrides"]} == c01_ids
    assert all(
        row["overrides"] == {"env_cfg.move_cost": 0.4}
        for row in manifest["cell_overrides"]
    )
    assert manifest["qualification_receipt_action"] == "unchanged"
    assert manifest["qualification_key_sha256"] == source[
        "qualification_key_sha256"
    ]
    assert manifest["qualification_receipt"] == source["artifacts"][
        "qualification_receipt"
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"new_value": -1.0},
        {"source_WRC_ID": "P999"},
        {"receipt_action": "requalified"},
        {"evidence_used": "unbound-analysis"},
        {"tick_end": 10_000},
        {"parameter": "patch.capacity", "source_WRC_ID": "P000", "new_value": 1024},
    ],
)
def test_rejects_out_of_envelope_or_stage_contract_changes(overrides):
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    analysis = _analysis_receipt(source, binding, "C2")
    row = _row(evidence_used=analysis["receipt_sha256"])
    row.update(overrides)
    with pytest.raises(AdaptiveStageError):
        materialize_stage_plan(
            source,
            binding,
            [row],
            stage="C2",
            analysis_receipt=analysis,
        )


def test_cli_writes_stage_plan_and_frozen_manifest(tmp_path, capsys):
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    analysis = _analysis_receipt(source, binding, "C2")
    analysis_path = tmp_path / "analysis-receipt.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    interventions = tmp_path / "interventions.csv"
    with interventions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERVENTION_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                **_row(evidence_used=analysis["receipt_sha256"]),
                "previous_value": json.dumps(0.2),
                "new_value": json.dumps(0.4),
            }
        )
    plan_output = tmp_path / "c2-plan.json"
    manifest_output = tmp_path / "c2-manifest.json"

    assert main(
        [
            "--source-plan",
            str(DESIGN_ROOT / "B48_EXPLORATORY_C1_PLAN.json"),
            "--design-binding",
            str(DESIGN_ROOT / "B48_EXPLORATORY_DESIGN_BINDING.json"),
            "--interventions",
            str(interventions),
            "--analysis-receipt",
            str(analysis_path),
            "--stage",
            "C2",
            "--plan-output",
            str(plan_output),
            "--manifest-output",
            str(manifest_output),
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)

    assert receipt["qualification_receipt_action"] == "unchanged"
    assert json.loads(plan_output.read_text())["execution"]["generations"] == 50
    assert json.loads(manifest_output.read_text())["stage"] == "C2"


def test_rejects_parameter_removed_from_source_envelope():
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    envelope = source["qualification_key"]["variability_envelope"]
    envelope["parameter_contracts"] = [
        row
        for row in envelope["parameter_contracts"]
        if row["name"] != "env_cfg.move_cost"
    ]
    source["qualification_key_sha256"] = canonical_sha256(
        source["qualification_key"]
    )
    source["plan_sha256"] = canonical_sha256(
        {key: value for key, value in source.items() if key != "plan_sha256"}
    )
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    analysis = _analysis_receipt(source, binding, "C2")

    with pytest.raises(AdaptiveStageError, match="envelope"):
        materialize_stage_plan(
            source,
            binding,
            [_row(evidence_used=analysis["receipt_sha256"])],
            stage="C2",
            analysis_receipt=analysis,
        )


def test_rejects_analysis_receipt_with_altered_artifact_identity():
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    analysis = _analysis_receipt(source, binding, "C2")
    analysis["outputs"]["world_summary"]["sha256"] = "0" * 64
    body = dict(analysis)
    body.pop("receipt_sha256")
    analysis["receipt_sha256"] = canonical_sha256(body)

    with pytest.raises(AdaptiveStageError, match="artifact identity differs"):
        materialize_stage_plan(
            source,
            binding,
            [_row(evidence_used=analysis["receipt_sha256"])],
            stage="C2",
            analysis_receipt=analysis,
        )


def test_materializes_complete_generation_boundary_stage_chain():
    source = _load("B48_EXPLORATORY_C1_PLAN.json")
    binding = _load("B48_EXPLORATORY_DESIGN_BINDING.json")
    transitions = (
        ("C2", 4_000, 8_000, 0.2, 0.4, 50),
        ("C3", 8_000, 12_000, 0.4, 0.6, 75),
        ("C4", 12_000, 16_000, 0.6, 0.8, 100),
        ("C5", 16_000, 20_000, 0.8, 1.0, 125),
    )

    for stage, tick_start, tick_end, previous, new, generation in transitions:
        analysis = _analysis_receipt(source, binding, stage)
        restored, manifest = materialize_stage_plan(
            source,
            binding,
            [
                _row(
                    stage=stage,
                    tick_start=tick_start,
                    tick_end=tick_end,
                    previous_value=previous,
                    new_value=new,
                    evidence_used=analysis["receipt_sha256"],
                )
            ],
            stage=stage,
            analysis_receipt=analysis,
        )
        assert restored["execution"]["generations"] == generation
        assert restored["execution"]["tick_schedule"]["total_ticks"] == tick_end
        assert restored["qualification_key"] == source["qualification_key"]
        assert restored["artifacts"] == source["artifacts"]
        assert manifest["source_plan_sha256"] == source["plan_sha256"]
        assert manifest["restore_plan_sha256"] == restored["plan_sha256"]
        for before, after in zip(source["cells"], restored["cells"], strict=True):
            for field in ("cell_id", "branch_id", "rng_lineage_id", "seeds"):
                assert after[field] == before[field]
        source = restored
