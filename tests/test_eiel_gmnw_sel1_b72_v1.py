from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tools.scripts import eiel_gmnw_sel1_b72_v1 as sel1


def test_matrix_has_exactly_72_unique_stable_cells() -> None:
    cells = sel1.cells_v1()

    assert len(cells) == sel1.BATCH_CELLS == 72
    assert tuple(cell.ordinal for cell in cells) == tuple(range(72))
    assert len({cell.cell_id for cell in cells}) == 72
    assert len({cell.pair_id for cell in cells}) == sel1.PAIR_COUNT == 36
    assert {cell.founder_count for cell in cells} == {0, 4, 8, 16}
    assert {cell.seed for cell in cells} == set(range(101, 910, 101))
    assert cells[0].cell_id == "sel1_f00_s101_d"
    assert cells[1].cell_id == "sel1_f00_s101_i"
    assert cells[-2].cell_id == "sel1_f16_s909_d"
    assert cells[-1].cell_id == "sel1_f16_s909_i"
    assert sel1.cells_v1() == cells


def test_every_pair_is_exact_delayed_immediate_identity() -> None:
    cells_by_id = {cell.cell_id: cell for cell in sel1.cells_v1()}
    pairs = sel1.pairs_v1()

    assert len(pairs) == 36
    for index, pair in enumerate(pairs):
        delayed = cells_by_id[pair.delayed_cell_id]
        immediate = cells_by_id[pair.immediate_cell_id]
        assert delayed.ordinal == index * 2
        assert immediate.ordinal == index * 2 + 1
        assert delayed.pair_id == immediate.pair_id == pair.pair_id
        assert delayed.founder_count == immediate.founder_count == pair.founder_count
        assert delayed.seed == immediate.seed == pair.seed
        assert (delayed.context, immediate.context) == ("delayed", "immediate")
        assert delayed.cue_visible_at_decision is False
        assert immediate.cue_visible_at_decision is True


def test_repeat_and_continuation_lists_match_the_preregistered_cells() -> None:
    by_id = {cell.cell_id: cell for cell in sel1.cells_v1()}
    repeat = sel1.repeat_cell_ids_v1()
    continuation = sel1.continuation_cell_ids_v1()

    assert repeat == sel1.REPEAT_CELL_IDS
    assert continuation == sel1.CONTINUATION_CELL_IDS
    assert len(repeat) == len(set(repeat)) == 6
    assert len(continuation) == len(set(continuation)) == 2
    assert all(cell_id in by_id for cell_id in (*repeat, *continuation))
    assert {by_id[cell_id].founder_count for cell_id in continuation} == {8}
    assert [by_id[cell_id].context for cell_id in continuation] == [
        "delayed",
        "immediate",
    ]


def test_thresholds_and_result_schema_are_frozen() -> None:
    assert sel1.THRESHOLDS == {
        "required_completed_cells": 72,
        "required_complete_pairs": 36,
        "maximum_failed_cells": 0,
        "required_completed_generations_per_cell": 12,
        "required_repeat_cells": 6,
        "required_continuation_cells": 2,
        "require_exact_repeat_state_digest": True,
        "require_exact_continuation_state_digest": True,
        "chance_accuracy": 0.25,
        "minimum_gru_delayed_accuracy": 0.90,
        "maximum_mlp_delayed_accuracy": 0.35,
        "minimum_gru_immediate_accuracy": 0.90,
        "minimum_mlp_immediate_accuracy": 0.90,
        "require_zero_gru_controls_exact": True,
        "require_positive_recurrent_cost": True,
        "maximum_zero_founder_final_gru_fraction": 0.0,
        "minimum_delayed_minus_immediate_gru_fraction": 0.25,
        "minimum_delayed_final_minus_initial_gru_fraction": 0.25,
        "minimum_supporting_seed_pairs_per_founder_count": 7,
        "minimum_parent_advantage_seeds_per_founder_count": 7,
        "minimum_offspring_advantage_seeds_per_founder_count": 7,
        "minimum_supporting_nonzero_founder_counts": 2,
        "maximum_immediate_gru_fraction_increase": 0.10,
        "maximum_immediate_advantage_seeds_per_founder_count": 5,
    }
    schema = sel1.result_schema_v1()
    assert schema["title"] == sel1.RESULT_SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert schema["properties"]["completed_generations"] == {"const": 12}
    assert schema["properties"]["policy_arm"] == {"const": "mixed_selection"}
    receipt = schema["properties"]["execution_receipt"]["properties"]
    assert receipt["native_dispatches"] == {"const": 1}
    assert receipt["physical_batch_cells"] == {"const": 72}
    assert receipt["logical_chunking"] == {"const": False}
    metrics = schema["properties"]["metrics"]
    assert metrics["additionalProperties"] is False
    generations = metrics["properties"]["selection_generations"]
    assert (generations["minItems"], generations["maxItems"]) == (12, 12)
    generation = generations["items"]
    assert generation["properties"]["native"]["required"] == list(
        sel1.NATIVE_METRIC_NAMES
    )
    assert set(generation["properties"]["derived"]["properties"]) == {"mlp", "gru"}


def test_configuration_rows_bind_only_preregistered_parent_fields() -> None:
    first, immediate = sel1.cells_v1()[:2]

    assert first.configuration_kwargs() == {
        "policy_arm": "mixed_selection",
        "gru_founder_count": 0,
        "cue_visible_at_decision": False,
        "task_seed": 101,
    }
    assert immediate.configuration_kwargs()["cue_visible_at_decision"] is True


def test_native_configurations_are_exact_mixed_selection_b72_schedule() -> None:
    configs = sel1.configurations_v1()

    assert len(configs) == 72
    assert [config.policy_arm for config in configs] == ["mixed_selection"] * 72
    assert [config.gru_founder_count for config in configs] == [
        cell.founder_count for cell in sel1.cells_v1()
    ]
    assert [config.task_seed for config in configs] == [
        cell.seed for cell in sel1.cells_v1()
    ]
    assert [config.cue_visible_at_decision for config in configs] == [
        cell.cue_visible_at_decision for cell in sel1.cells_v1()
    ]
    for config in configs:
        assert (config.cue_alphabet, config.delay_ticks) == (4, 12)
        assert config.distractors_enabled is True
        assert config.reset_hidden_each_tick is False
        assert config.terminal_low_population_no_reinit is True
        assert (
            config.grace_generation_count,
            config.severe_generation_count,
        ) == (4, 4)
        assert (
            config.grace_damage_fraction,
            config.severe_damage_fraction,
            config.lethal_damage_fraction,
        ) == (0.01, 0.01, 1.0)


def test_contract_is_canonical_hash_bound_and_rejects_drift() -> None:
    contract = sel1.contract_v1()

    assert sel1.validate_contract_v1(contract) == contract
    assert contract["run_label"] == "gmnw-sel1-b72g12t80"
    assert contract["physical_dispatch_count"] == 1
    assert contract["logical_chunking"] is False
    assert contract["native_window_count"] == 6
    assert (
        contract["contract_sha256"]
        == "bd4267418a9f2e75e0634e17468e643579b15bc434fceb6902bc80452e7eb20f"
    )
    assert len(contract["cells"]) == 72
    assert len(contract["pairs"]) == 36
    assert contract["selection_metrics"] == {
        "required": True,
        "family_axis": ["mlp", "gru"],
        "count_metrics": list(sel1.COUNT_METRIC_NAMES),
        "cost_metric": "recurrent_upkeep_charge_total",
        "generation_count": 12,
        "pending_non_native_measurements": list(sel1.PENDING_NATIVE_MEASUREMENTS),
    }

    contract["cells"][0]["seed"] = 202
    with pytest.raises(sel1.Sel1HarnessError, match="contract_sha256 mismatch"):
        sel1.validate_contract_v1(contract)


def test_execution_fails_closed_without_all_adapters() -> None:
    with pytest.raises(sel1.Sel1HarnessError, match="inputs are not wired"):
        sel1.execute_sel1_v1()


def test_source_projection_layout_requires_all_stable_ids(tmp_path: Path) -> None:
    for cell in sel1.cells_v1()[:-1]:
        (tmp_path / f"{cell.cell_id}.pt").touch()

    with pytest.raises(sel1.Sel1HarnessError, match="missing per-cell projections"):
        sel1.source_projection_paths_v1(tmp_path)

    last = sel1.cells_v1()[-1]
    (tmp_path / f"{last.cell_id}.pt").touch()
    paths = sel1.source_projection_paths_v1(tmp_path)
    assert len(paths) == 72
    assert [path.stem for path in paths] == [cell.cell_id for cell in sel1.cells_v1()]


def test_current_engine_readiness_reports_structured_blockers() -> None:
    readiness = sel1.engine_readiness_v1()

    assert set(readiness) == {
        "ok",
        "module",
        "batch_cells",
        "generations",
        "blockers",
    }
    assert isinstance(readiness["ok"], bool)
    assert isinstance(readiness["blockers"], list)
    if not readiness["ok"]:
        assert readiness["blockers"]


def test_result_validation_rejects_missing_and_duplicate_rows() -> None:
    with pytest.raises(sel1.Sel1HarnessError, match="exactly 72"):
        sel1.validate_result_rows_v1([])

    duplicate = [{"cell_id": "same"}] * 72
    with pytest.raises(sel1.Sel1HarnessError, match="mismatch"):
        sel1.validate_result_rows_v1(duplicate)


def _preparation_receipt() -> dict[str, Any]:
    return {
        "physical_batch_cells": 72,
        "logical_generation_count": 12,
        "ticks_per_generation": 80,
        "logical_chunking": False,
        "source_projection_count": 72,
        "source_projection_cell_ids": [cell.cell_id for cell in sel1.cells_v1()],
        "source_identity_count": 1,
        "per_cell_request_count": 1,
        "projection_builder_calls": 1,
        "allocator_calls": 1,
        "selection_metrics_enabled": True,
        "hidden_fallback": False,
    }


def _native_receipt() -> dict[str, Any]:
    return {
        "completed_generations": 12,
        "logical_generation_count": 12,
        "window_count": 6,
        "settled_window_count": 6,
        "native_dispatches": 1,
        "external_python_window_boundaries": 0,
        "native_window_loop_owner": "native_extension",
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "runtime_jit": False,
        "hidden_fallback": False,
        "gru_python_tick_or_generation_ownership": False,
        "gmnw_python_tick_or_generation_ownership": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("native_dispatches", 2),
        ("logical_generation_count", 8),
        ("completed_generations", 11),
        ("window_count", 5),
        ("hidden_fallback", True),
        ("logical_chunking", True),
        ("chunk_count", 2),
    ),
)
def test_native_dispatch_receipt_fails_closed_on_non_b72g12_contract(
    field: str,
    value: object,
) -> None:
    receipt = _native_receipt()
    receipt[field] = value

    with pytest.raises(sel1.Sel1HarnessError, match="dispatch receipt mismatch"):
        sel1.validate_native_dispatch_receipt_v1(receipt, dispatch_calls=1)


def test_native_dispatch_receipt_uses_observed_count_when_field_is_absent() -> None:
    raw = _native_receipt()
    del raw["native_dispatches"]

    receipt = sel1.validate_native_dispatch_receipt_v1(raw, dispatch_calls=1)

    assert receipt["native_dispatches"] == 1


def test_preparation_receipt_rejects_missing_per_cell_projection() -> None:
    receipt = _preparation_receipt()
    receipt["source_projection_cell_ids"] = receipt["source_projection_cell_ids"][:-1]

    with pytest.raises(sel1.Sel1HarnessError, match="preparation receipt mismatch"):
        sel1._validate_preparation_receipt_v1(receipt)


def _native_selection_metrics(
    *,
    amplified_delayed: bool = False,
) -> dict[str, torch.Tensor]:
    shape = (sel1.BATCH_CELLS, sel1.GENERATIONS, len(sel1.POLICY_FAMILIES))
    metrics = {
        name: torch.zeros(shape, dtype=torch.int64)
        for name in sel1.COUNT_METRIC_NAMES
    }
    metrics[sel1.COST_METRIC_NAME] = torch.zeros(shape, dtype=torch.float64)
    metrics["status"] = torch.zeros((sel1.BATCH_CELLS,), dtype=torch.int32)
    for index, cell in enumerate(sel1.cells_v1()):
        final_gru = (
            48
            if amplified_delayed and cell.context == "delayed" and cell.founder_count
            else cell.founder_count
        )
        final_mlp = sel1.INITIAL_POPULATION - final_gru
        metrics["alive_pre_turnover"][index, :, 0] = final_mlp
        metrics["alive_pre_turnover"][index, :, 1] = final_gru
        metrics["alive_post_turnover"][index, :, 0] = final_mlp
        metrics["alive_post_turnover"][index, :, 1] = final_gru
        metrics["decision_count"][index, :, 0] = 100
        metrics["correct_count"][index, :, 0] = (
            25 if cell.context == "delayed" else 95
        )
        if cell.founder_count:
            metrics["decision_count"][index, :, 1] = 100
            metrics["correct_count"][index, :, 1] = 95
            metrics[sel1.COST_METRIC_NAME][index, :, 1] = 2.0
        if amplified_delayed and cell.context == "delayed" and cell.founder_count:
            metrics["unique_selected_parent_pool_count"][index, :, :] = torch.tensor(
                [1, 7]
            )
            metrics["committed_turnover_offspring_count"][index, :, :] = (
                torch.tensor([1, 7])
            )
        else:
            total = 16
            gru = cell.founder_count // 4
            metrics["unique_selected_parent_pool_count"][index, :, 0] = total - gru
            metrics["unique_selected_parent_pool_count"][index, :, 1] = gru
            metrics["committed_turnover_offspring_count"][index, :, 0] = total - gru
            metrics["committed_turnover_offspring_count"][index, :, 1] = gru
    return metrics


def _prepared_with_metrics(
    *,
    amplified_delayed: bool = False,
) -> sel1.PreparedSel1B72V1:
    metrics = _native_selection_metrics(amplified_delayed=amplified_delayed)
    alive = torch.ones((72, sel1.INITIAL_POPULATION), dtype=torch.bool)
    family = torch.zeros_like(alive)
    for index, cell in enumerate(sel1.cells_v1()):
        final_gru = int(metrics["alive_post_turnover"][index, -1, 1].item())
        family[index, :final_gru] = True
    native = SimpleNamespace(
        patch=SimpleNamespace(
            source=SimpleNamespace(
                live_state={"alive_mask": alive, "policy_use_gru": family}
            )
        ),
        selection_metrics=metrics,
    )
    return sel1.PreparedSel1B72V1(
        native=native,
        preparation_receipt=_preparation_receipt(),
    )


def test_mocked_execution_is_one_physical_b72g12t80_dispatch() -> None:
    prepared = _prepared_with_metrics()
    native = prepared.native
    calls: list[tuple[object, object, dict[str, Any]]] = []

    def execute(module: object, actual: object, **kwargs: Any) -> dict[str, Any]:
        calls.append((module, actual, kwargs))
        return _native_receipt()

    report = sel1.execute_prepared_b72_v1(
        object(),
        prepared,
        run_id="sel1-b72-mock",
        execute_fn=execute,
        digest_fn=lambda _prepared: tuple(f"{index + 1:064x}" for index in range(72)),
    )

    assert len(calls) == 1
    assert calls[0][1] is native
    assert calls[0][2] == {
        "logical_generation_count": 12,
        "run_id": "sel1-b72-mock",
    }
    assert report["physical_batch_cells"] == 72
    assert report["logical_chunking"] is False
    assert report["chunk_count"] == report["native_dispatches"] == 1
    assert len(report["results"]) == 72
    assert report["results"][0]["metrics"]["final_gru_count"] == 0
    assert report["results"][-1]["metrics"]["final_gru_count"] == 16
    generations = report["results"][-1]["metrics"]["selection_generations"]
    assert len(generations) == 12
    assert generations[0]["native"]["alive_pre_turnover"] == {
        "mlp": 48,
        "gru": 16,
        "total": 64,
    }
    assert generations[0]["derived"]["gru"]["decision_accuracy"] == 0.95
    assert report["repeat_cell_ids"] == list(sel1.REPEAT_CELL_IDS)
    assert report["continuation_cell_ids"] == list(sel1.CONTINUATION_CELL_IDS)


def test_native_selection_metrics_fail_closed_on_shape_status_and_bounds() -> None:
    metrics = _native_selection_metrics()
    metrics["decision_count"] = torch.zeros((72, 12, 1), dtype=torch.int64)
    with pytest.raises(sel1.Sel1HarnessError, match=r"int64\[72,12,2\]"):
        sel1._validated_native_selection_metrics_v1(metrics)

    metrics = _native_selection_metrics()
    metrics["status"][3] = 9
    with pytest.raises(sel1.Sel1HarnessError, match="status is nonzero"):
        sel1._validated_native_selection_metrics_v1(metrics)

    metrics = _native_selection_metrics()
    metrics["correct_count"][20, 0, 0] = 101
    with pytest.raises(sel1.Sel1HarnessError, match="exceeds decision_count"):
        sel1._validated_native_selection_metrics_v1(metrics)

    metrics = _native_selection_metrics()
    metrics["task_death_count"][20, 0, 0] = -1
    with pytest.raises(sel1.Sel1HarnessError, match="is negative"):
        sel1._validated_native_selection_metrics_v1(metrics)

    metrics = _native_selection_metrics()
    metrics[sel1.COST_METRIC_NAME][20, 0, 1] = float("nan")
    with pytest.raises(sel1.Sel1HarnessError, match="finite and non-negative"):
        sel1._validated_native_selection_metrics_v1(metrics)

    metrics = _native_selection_metrics()
    metrics["alive_post_turnover"][0, 0, 1] = 1
    with pytest.raises(sel1.Sel1HarnessError, match="zero-GRU integrity"):
        sel1._validated_native_selection_metrics_v1(metrics)


def test_zero_denominators_are_null_and_final_alive_must_agree() -> None:
    prepared = _prepared_with_metrics()
    rows = sel1._result_rows_from_prepared_v1(
        prepared,
        state_digests=tuple(f"{index + 1:064x}" for index in range(72)),
        native_receipt=sel1.validate_native_dispatch_receipt_v1(
            _native_receipt(), dispatch_calls=1
        ),
    )
    zero_gru = rows[0]["metrics"]["selection_generations"][0]["derived"]["gru"]
    assert zero_gru["decision_accuracy"] is None
    assert zero_gru["recurrent_upkeep_charge_per_alive_pre_turnover"] is None

    changed = [dict(row) for row in rows]
    changed[-1] = {**changed[-1], "metrics": dict(changed[-1]["metrics"])}
    changed[-1]["metrics"]["final_gru_count"] = 15
    with pytest.raises(sel1.Sel1HarnessError, match="final_gru_fraction is invalid"):
        sel1.validate_result_rows_v1(changed)

    changed = [dict(row) for row in rows]
    changed[-1] = {**changed[-1], "metrics": dict(changed[-1]["metrics"])}
    changed_generations = list(changed[-1]["metrics"]["selection_generations"])
    changed[-1]["metrics"]["selection_generations"] = changed_generations
    changed_generation = dict(changed_generations[0])
    changed_generations[0] = changed_generation
    changed_native = dict(changed_generation["native"])
    changed_generation["native"] = changed_native
    changed_alive = dict(changed_native["alive_pre_turnover"])
    changed_native["alive_pre_turnover"] = changed_alive
    changed_alive["total"] += 1
    with pytest.raises(sel1.Sel1HarnessError, match="family sum mismatch"):
        sel1.validate_result_rows_v1(changed)


def test_analysis_evaluates_competence_advantage_cost_and_7_of_9_gates() -> None:
    prepared = _prepared_with_metrics(amplified_delayed=True)
    rows = sel1._result_rows_from_prepared_v1(
        prepared,
        state_digests=tuple(f"{index + 1:064x}" for index in range(72)),
        native_receipt=sel1.validate_native_dispatch_receipt_v1(
            _native_receipt(), dispatch_calls=1
        ),
    )

    analysis = sel1.analyze_result_rows_v1(rows)

    assert analysis["native_metric_gate_decision"] == "pass"
    assert analysis["preregistered_a_decision"] == "fail_closed"
    assert analysis["decision"]["letter"] is None
    assert analysis["decision"]["scientific_matrix_letter"] == "A"
    assert analysis["decision"]["a_eligible"] is False
    assert analysis["decision"]["a_gate_receipt"]["missing"] == list(
        sel1.A_GATE_RECEIPT_NAMES
    )
    assert analysis["competence"]["all_nonzero_founder_contexts_pass"] is True
    assert analysis["zero_gru_integrity"] == {
        "cell_count": 18,
        "exact_native_zero": True,
    }
    assert analysis["replication_and_specificity"] == {
        "required_supporting_nonzero_founder_counts": 2,
        "delayed_7_of_9_supporting_founder_count": 3,
        "immediate_specificity_supporting_founder_count": 3,
        "pass": True,
    }
    for gate in analysis["founder_fraction_gates"]:
        assert gate["positive_paired_seed_count"] == 9
        assert gate["delayed_parent_advantage_seed_count"] == 9
        assert (
            gate["delayed_committed_turnover_offspring_advantage_seed_count"] == 9
        )
        assert gate["delayed_positive_recurrent_cost_seed_count"] == 9
        assert gate["delayed_7_of_9_gate"] is True
        assert gate["immediate_specificity_gate"] is True
    assert {item["status"] for item in analysis["pending_non_native_measurements"]} == {
        "pending"
    }
    assert {item["measurement"] for item in analysis["measurement_inventory"]} - {
        item["measurement"] for item in analysis["pending_non_native_measurements"]
    } == {"lineage_extinction_generation"}
    json.dumps(analysis, allow_nan=False)


def test_extinct_population_keeps_null_fraction_but_contributes_zero() -> None:
    row = {
        "cell_id": "extinct",
        "metrics": {
            "final_gru_count": 0,
            "final_alive_count": 0,
            "final_gru_fraction": None,
        },
    }

    assert sel1._endpoint_population_contribution_v1(row) == 0.0
    assert row["metrics"]["final_gru_fraction"] is None


def test_policy_family_extinction_rejects_reappearance() -> None:
    extinct = {
        "cell_id": "extinct-gru",
        "founder_count": 4,
        "metrics": {
            "selection_generations": [
                {"native": {"alive_post_turnover": {"mlp": 60, "gru": count}}}
                for count in (4, 2, 0, 0)
            ]
        },
    }
    row = {
        "cell_id": "reappearance",
        "founder_count": 4,
        "metrics": {
            "selection_generations": [
                {"native": {"alive_post_turnover": {"mlp": 60, "gru": count}}}
                for count in (4, 0, 0, 1)
            ]
        },
    }

    assert sel1._family_extinction_generation_v1(extinct, "gru") == 3
    assert sel1._family_extinction_generation_v1(extinct, "mlp") is None
    with pytest.raises(sel1.Sel1HarnessError, match="reappeared after extinction"):
        sel1._family_extinction_generation_v1(row, "gru")


def test_decision_b_is_selected_for_directional_below_threshold_matrix() -> None:
    decision = sel1._select_decision_v1(
        native_a_gate_pass=False,
        directional_supporting_founder_count=2,
        required_supporting_founder_count=2,
        a_gate_receipt_status=sel1._a_gate_receipt_status_v1(None),
    )

    assert decision == {
        "letter": "B",
        "code": "GMNW-SEL1-B72-B/CAPABILITY-PROVED-SELECTION-LIMITED",
        "status": "provisional_scientific_matrix_decision",
        "scientific_matrix_letter": "B",
        "scientific_matrix_code": (
            "GMNW-SEL1-B72-B/CAPABILITY-PROVED-SELECTION-LIMITED"
        ),
        "directional_supporting_founder_count": 2,
        "a_eligible": False,
        "a_gate_receipt": sel1._a_gate_receipt_status_v1(None),
    }


def test_decision_a_requires_every_external_gate_receipt() -> None:
    incomplete = {name: True for name in sel1.A_GATE_RECEIPT_NAMES[:-1]}
    gated = sel1._select_decision_v1(
        native_a_gate_pass=True,
        directional_supporting_founder_count=3,
        required_supporting_founder_count=2,
        a_gate_receipt_status=sel1._a_gate_receipt_status_v1(incomplete),
    )
    complete = {name: True for name in sel1.A_GATE_RECEIPT_NAMES}
    accepted = sel1._select_decision_v1(
        native_a_gate_pass=True,
        directional_supporting_founder_count=3,
        required_supporting_founder_count=2,
        a_gate_receipt_status=sel1._a_gate_receipt_status_v1(complete),
    )

    assert gated["letter"] is None
    assert gated["status"] == "provisional_matrix_a_final_a_gated"
    assert gated["a_eligible"] is False
    assert accepted["letter"] == "A"
    assert accepted["a_eligible"] is True


def test_decision_c_accepts_prerequisite_but_not_integrity_failure() -> None:
    prerequisite = {name: True for name in sel1.A_GATE_RECEIPT_NAMES}
    prerequisite["b72_selection_envelope_admission"] = False
    prerequisite_decision = sel1._select_decision_v1(
        native_a_gate_pass=True,
        directional_supporting_founder_count=3,
        required_supporting_founder_count=2,
        a_gate_receipt_status=sel1._a_gate_receipt_status_v1(prerequisite),
    )
    integrity = {name: True for name in sel1.A_GATE_RECEIPT_NAMES}
    integrity["rng_integrity"] = False
    integrity_decision = sel1._select_decision_v1(
        native_a_gate_pass=True,
        directional_supporting_founder_count=3,
        required_supporting_founder_count=2,
        a_gate_receipt_status=sel1._a_gate_receipt_status_v1(integrity),
    )

    assert prerequisite_decision["letter"] == "C"
    assert prerequisite_decision["status"] == "final_one_precise_prerequisite"
    assert integrity_decision["letter"] is None
    assert integrity_decision["status"] == "provisional_matrix_a_final_a_gated"


def test_cli_validates_contract_and_execute_is_fail_closed(capsys) -> None:
    assert sel1.main(["--validate-contract"]) == 0
    valid = json.loads(capsys.readouterr().out)
    assert valid["ok"] is True
    assert (valid["cells"], valid["pairs"]) == (72, 36)

    assert sel1.main(["--execute"]) == 1
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["ok"] is False
    assert "inputs are not wired" in rejected["error"]["message"]
