from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_wrc_b48_adaptive_analyze import (
    CHECKPOINT_FIELDS,
    CONFIG_FIELDS,
    DIVERGENCE_FIELDS,
    STAGE_WINDOWS,
    WORLD_FIELDS,
)
from tools.scripts.eiel_wrc_b48_adaptive_closeout import (
    AdaptiveCloseoutError,
    consolidate_adaptive_chain,
    materialize_closeout_outputs,
)
from tools.scripts.eiel_wrc_b48_adaptive_stage import STAGES


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = ROOT / "_spec/campaigns/v1/wrc_b48_adaptive"
STAGE_ORDER = ("C1", "C2", "C3", "C4", "C5")
TRANSITIONS = ("C2", "C3", "C4", "C5")


def _load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _csv_bytes(rows, fields):
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def _write(path: Path, raw: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _signed(body: dict):
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _resign(value: dict):
    body = dict(value)
    body.pop("receipt_sha256", None)
    return _signed(body)


def _analysis_receipt(tmp_path: Path, stage: str, binding: dict, plan_sha: str):
    stage_root = tmp_path / stage
    _start, generation, tick_start, tick_end = STAGE_WINDOWS[stage]
    world_rows = []
    reduction_rows = []
    for row in binding["rows"]:
        reduction = {
            "schema_version": "eiel_wrc_world_reduction_v1",
            "identity": {"world_id": row["cell_id"]},
        }
        signed_reduction = {
            **reduction,
            "result_sha256": canonical_sha256(reduction),
        }
        world = {field: "" for field in WORLD_FIELDS}
        config_index = int(row["config_id"][1:])
        seed_index = int(row["seed_tuple"][1:])
        alive = 300.0 - config_index * 5.0 - seed_index + generation / 10.0
        world.update(
            {
                "stage": stage,
                "tick_start": tick_start,
                "tick_end": tick_end,
                "config_id": row["config_id"],
                "seed_tuple": row["seed_tuple"],
                "cell_id": row["cell_id"],
                "branch_id": row["branch_id"],
                "rng_lineage_id": row["rng_lineage_id"],
                "active_bundle_sha256": "a" * 64,
                "observation_count": 25,
                "metric_count": 1,
                "exploratory_viability_band": "VIABLE",
                "alive_first": alive + 1,
                "alive_terminal": alive,
                "alive_minimum": alive,
                "alive_trajectory_mean": alive + 0.5,
                "births_total": 10,
                "failed_births_total": 1,
                "deaths_total": 5,
                "moved_total": 100,
                "gathered_total": 80,
                "food_stock_terminal": 1_000,
                "energy_collected_terminal": 500,
                "policy_energy_terminal": 250,
                "metric_summaries_json": "{}",
                "reducer_result_sha256": signed_reduction["result_sha256"],
            }
        )
        world_rows.append(world)
        reduction_rows.append(signed_reduction)
    config_rows = []
    divergence_rows = []
    for index in range(1, 17):
        config_id = f"C{index:02d}"
        alive = 300.0 - index * 5.0 + generation / 10.0
        config = {field: "" for field in CONFIG_FIELDS}
        config.update(
            {
                "stage": stage,
                "tick_start": tick_start,
                "tick_end": tick_end,
                "config_id": config_id,
                "replicate_count": 3,
                "viable_count": 3,
                "fragile_count": 0,
                "collapsed_count": 0,
                "extinct_count": 0,
                "alive_terminal_mean": alive,
                "alive_terminal_median": alive,
                "alive_terminal_minimum": alive - 1,
                "alive_terminal_maximum": alive + 1,
                "alive_terminal_range": 2,
                "births_total_mean": 10,
                "deaths_total_mean": 5,
                "metric_aggregate_json": "{}",
            }
        )
        config_rows.append(config)
        divergence = {field: "" for field in DIVERGENCE_FIELDS}
        divergence.update(
            {
                "stage": stage,
                "config_id": config_id,
                "metric": "alive",
                "replicate_count": 3,
                "terminal_mean": alive,
                "terminal_median": alive,
                "terminal_minimum": alive - 1,
                "terminal_maximum": alive + 1,
                "terminal_range": 2,
                "terminal_relative_range": 2 / alive,
                "terminal_population_stddev": 0.8,
                "agreement_band": "HIGH",
            }
        )
        divergence_rows.append(divergence)
    checkpoint = {field: "" for field in CHECKPOINT_FIELDS}
    checkpoint.update(
        {
            "stage": stage,
            "absolute_tick": tick_end,
            "generation": generation,
            "checkpoint_schema_version": "checkpoint-v1",
            "checkpoint_sha256": str(STAGE_ORDER.index(stage) + 1) * 64,
            "plan_sha256": plan_sha,
            "qualification_key_sha256": "6" * 64,
            "qualification_receipt_sha256": "7" * 64,
            "variability_envelope_id": "envelope-v1",
            "variability_envelope_sha256": "8" * 64,
            "execution_route_sha256": "9" * 64,
            "physical_cell_count": 48,
            "terminal_cell_digests_sha256": "a" * 64,
            "checkpoint_identity_contract_sha256": "b" * 64,
            "adaptive_rebinding_receipt_sha256": "" if stage == "C1" else "c" * 64,
            "adaptive_runtime_readback_receipt_sha256": ""
            if stage == "C1"
            else "d" * 64,
            "continuity_status": "VERIFIED_EXACT_48",
            "qualification_receipt_action": "unchanged",
        }
    )
    outputs = {
        "world_summary": _write(
            stage_root / "world.csv", _csv_bytes(world_rows, WORLD_FIELDS)
        ),
        "configuration_summary": _write(
            stage_root / "config.csv", _csv_bytes(config_rows, CONFIG_FIELDS)
        ),
        "seed_divergence": _write(
            stage_root / "divergence.csv",
            _csv_bytes(divergence_rows, DIVERGENCE_FIELDS),
        ),
        "checkpoints": _write(
            stage_root / "checkpoint.csv",
            _csv_bytes([checkpoint], CHECKPOINT_FIELDS),
        ),
        "world_reductions": _write(
            stage_root / "reductions.jsonl",
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in reduction_rows
            ).encode(),
        ),
    }
    return _signed(
        {
            "schema_version": "eiel_wrc_b48_adaptive_analysis_outputs_v1",
            "stage": stage,
            "plan_sha256": plan_sha,
            "design_binding_receipt_sha256": binding["receipt_sha256"],
            "terminal_generation": generation,
            "tick_end": tick_end,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "reducer_version": "eiel_wrc_reducer_v1_1",
            "reducer_config_sha256": "e" * 64,
            "analysis_sha256": "f" * 64,
            "outputs": outputs,
            "qualification_receipt_action": "unchanged",
        }
    )


def _stage_manifest(stage: str, source: dict, restore: dict, binding: dict):
    source_generation, restore_generation, tick_start, tick_end = STAGES[stage]
    row = {
        "stage": stage,
        "config_id": "C01",
        "tick_start": tick_start,
        "tick_end": tick_end,
        "parameter": "env_cfg.move_cost",
        "previous_value": (source_generation - 25) / 100 + 0.2,
        "new_value": source_generation / 100 + 0.2,
        "source_WRC_ID": "P023",
        "reason": "bounded synthetic transition",
        "evidence_used": source["receipt_sha256"],
        "envelope_status": "COVERED_APPROVED",
        "receipt_action": "unchanged",
    }
    c01 = [entry for entry in binding["rows"] if entry["config_id"] == "C01"]
    body = {
        "schema_version": "eiel_wrc_b48_adaptive_stage_manifest_v1",
        "stage": stage,
        "source_generation": source_generation,
        "restore_generation": restore_generation,
        "tick_start": tick_start,
        "tick_end": tick_end,
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "design_binding_receipt_sha256": binding["receipt_sha256"],
        "source_analysis_receipt_sha256": source["receipt_sha256"],
        "qualification_key_sha256": "6" * 64,
        "variability_envelope_id": "envelope-v1",
        "variability_envelope_sha256": "8" * 64,
        "qualification_receipt": {"path": "/qualification", "sha256": "7" * 64},
        "production_artifacts_sha256": "a" * 64,
        "configuration_count": 16,
        "seed_replicates_per_configuration": 3,
        "physical_cell_count": 48,
        "changed_configuration_count": 1,
        "changed_cell_count": 3,
        "rows": [row],
        "cell_overrides": [
            {
                "cell_id": entry["cell_id"],
                "config_id": "C01",
                "seed_tuple": entry["seed_tuple"],
                "overrides": {"env_cfg.move_cost": row["new_value"]},
            }
            for entry in c01
        ],
        "qualification_receipt_action": "unchanged",
    }
    return _signed(body)


def _chain_fixture(tmp_path: Path):
    binding = json.loads(
        (DESIGN_ROOT / "B48_EXPLORATORY_DESIGN_BINDING.json").read_text(
            encoding="utf-8"
        )
    )
    analyses = [
        _analysis_receipt(
            tmp_path,
            stage,
            binding,
            plan_sha=str(index + 1) * 64,
        )
        for index, stage in enumerate(STAGE_ORDER)
    ]
    manifests = [
        _stage_manifest(stage, analyses[index], analyses[index + 1], binding)
        for index, stage in enumerate(TRANSITIONS)
    ]
    return binding, analyses, manifests


def test_consolidates_exact_g125_chain_and_materializes_outputs(tmp_path):
    binding, analyses, manifests = _chain_fixture(tmp_path)

    consolidated = consolidate_adaptive_chain(
        design_binding=binding,
        initial_design_path=DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv",
        analysis_receipts=analyses,
        stage_manifests=manifests,
    )

    assert consolidated["chain"]["status"] == "CHAIN_VERIFIED_G125_T20000"
    assert len(consolidated["world_rows"]) == 240
    assert len(consolidated["configuration_rows"]) == 80
    assert len(consolidated["checkpoint_rows"]) == 5
    assert len(consolidated["intervention_rows"]) == 4
    assert consolidated["report_input"]["terminal_tick"] == 20_000

    output = tmp_path / "closeout"
    receipt = materialize_closeout_outputs(consolidated, output_dir=output)
    assert receipt["status"] == "CHAIN_VERIFIED_G125_T20000"
    assert len(receipt["outputs"]) == 10
    assert len((output / "B48_EXPLORATORY_WORLD_SUMMARY.csv").read_text().splitlines()) == 241
    assert len(
        (output / "B48_EXPLORATORY_WORLD_REDUCTIONS.jsonl")
        .read_text()
        .splitlines()
    ) == 240
    record = (
        output / "WRC_B48_20K_ADAPTIVE_EXPLORATORY_INTEGRATION_RECORD.md"
    ).read_text()
    assert "CHAIN_VERIFIED_G125_T20000" in record
    assert "not a new qualification action" in record
    report = (
        output / "WRC_B48_20K_ADAPTIVE_EXPLORATORY_REPORT.md"
    ).read_text()
    assert "## Required questions" in report
    assert "13. **Did the experiment expose" in report
    assert "does not establish isolated knob causality" in report
    assert "Terminal horizon: G125 / 20,000 ticks per world" in report


def test_rejects_broken_plan_chain_even_when_manifest_is_resigned(tmp_path):
    binding, analyses, manifests = _chain_fixture(tmp_path)
    damaged = deepcopy(manifests)
    damaged[1]["source_plan_sha256"] = "0" * 64
    damaged[1] = _resign(damaged[1])

    with pytest.raises(AdaptiveCloseoutError, match="intervention binding"):
        consolidate_adaptive_chain(
            design_binding=binding,
            initial_design_path=DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv",
            analysis_receipts=analyses,
            stage_manifests=damaged,
        )


def test_rejects_seed_specific_or_incomplete_override_expansion(tmp_path):
    binding, analyses, manifests = _chain_fixture(tmp_path)
    damaged = deepcopy(manifests)
    damaged[0]["cell_overrides"].pop()
    damaged[0] = _resign(damaged[0])

    with pytest.raises(AdaptiveCloseoutError, match="3-seed expansion"):
        consolidate_adaptive_chain(
            design_binding=binding,
            initial_design_path=DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv",
            analysis_receipts=analyses,
            stage_manifests=damaged,
        )


def test_rejects_route_change_hidden_in_resigned_analysis_receipt(tmp_path):
    binding, analyses, manifests = _chain_fixture(tmp_path)
    damaged = deepcopy(analyses)
    descriptor = damaged[3]["outputs"]["checkpoints"]
    checkpoint_path = Path(descriptor["path"])
    with checkpoint_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["execution_route_sha256"] = "0" * 64
    raw = _csv_bytes(rows, CHECKPOINT_FIELDS)
    checkpoint_path.write_bytes(raw)
    damaged[3]["outputs"]["checkpoints"] = {
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    damaged[3] = _resign(damaged[3])

    with pytest.raises(AdaptiveCloseoutError, match="route identity changed"):
        consolidate_adaptive_chain(
            design_binding=binding,
            initial_design_path=DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv",
            analysis_receipts=damaged,
            stage_manifests=manifests,
        )


def test_materializer_rejects_mutated_summary_after_chain_validation(tmp_path):
    binding, analyses, manifests = _chain_fixture(tmp_path)
    consolidated = consolidate_adaptive_chain(
        design_binding=binding,
        initial_design_path=DESIGN_ROOT / "B48_EXPLORATORY_INITIAL_DESIGN.csv",
        analysis_receipts=analyses,
        stage_manifests=manifests,
    )
    consolidated["configuration_rows"][0]["alive_terminal_mean"] = "99999"

    with pytest.raises(AdaptiveCloseoutError, match="consolidated closeout identity"):
        materialize_closeout_outputs(
            consolidated,
            output_dir=tmp_path / "tampered-closeout",
        )
