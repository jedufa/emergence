from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tools.scripts import eiel_cuda_architecture_b_compact_summary_validator as validator


SHA = "0" * 64


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return validator.load_json(validator.DEFAULT_SCHEMA_PATH)


@pytest.fixture(scope="module")
def ownership() -> dict[str, Any]:
    return validator.load_json(validator.DEFAULT_OWNERSHIP_PATH)


def _flat_row() -> dict[str, Any]:
    return {
        "schema_version": "eiel_compact_generation_summary_v1",
        "summary_valid": True,
        "summary_status": "complete",
        "world": "flat",
        "generation_start": {"generation": 3, "tick": 0},
        "generation_end": {"generation": 3, "tick_exclusive": 80},
        "next_generation_start": {"generation": 4, "tick": 0},
        "alive": 6,
        "births": 2,
        "failed_births": 1,
        "deaths": 1,
        "deaths_by_class": {"energy_depleted": 1, "health_depleted": 0},
        "turnover": {"count": 1, "status": "committed"},
        "events": {
            "cursor_start": 10,
            "cursor_pre_turnover": 14,
            "cursor_end": 16,
            "authoritative_cursor": 16,
            "event_row_count": 6,
            "generation_event_count": 4,
            "turnover_event_count": 2,
            "generation_counts_by_type": {
                "birth_committed": 2,
                "mutation": 0,
                "birth_failed": 1,
                "death_committed": 1,
            },
            "turnover_counts_by_type": {
                "birth_committed": 0,
                "mutation": 2,
                "birth_failed": 0,
                "death_committed": 0,
            },
            "active_prefix_digest_sha256": SHA,
        },
        "identity": {
            "population_next_id": 10,
            "active_agent_count": 8,
            "active_agent_id_min": 1,
            "active_agent_id_max": 9,
            "active_agent_ids_digest_sha256": SHA,
            "free_slot_count": 2,
        },
        "rng": {
            "authority_digest_sha256": SHA,
            "policy_mutation_tape_cursor": 2,
            "policy_mutation_tape_digest_sha256": SHA,
            "authority_complete": True,
            "export_non_consuming": True,
        },
        "terminal": {
            "status": "not_terminal",
            "recoverable_error_status": "none",
            "recoverable_error_code": None,
            "recoverable_error_digest_sha256": None,
            "sticky_failure": False,
            "failure_generation": None,
            "failure_tick": None,
            "failure_stage": None,
        },
        "checkpoint_due": False,
        "state_digests": {
            "authoritative_state_digest_sha256": SHA,
            "canonical_checkpoint_storage_digest_sha256": SHA,
            "next_generation_start_state_digest_sha256": SHA,
        },
        "generation_metrics": {
            "capacity": 10,
            "free_slots": 4,
            "birth_attempts": 3,
            "native_failed_birth_event_rows": 1,
            "injury_blocked_births": 0,
            "moved": 4,
            "gathered": 2,
            "movement_attempts": 8,
            "movement_blocked_occupied": 3,
            "movement_blocked_bounds": 1,
            "peak_injury": 0.5,
            "policy_loop_tick_calls": 80,
            "policy_refill_calls": 2,
            "policy_refill_rows": 5,
            "policy_cache_hit_calls": 78,
            "policy_cache_rows_requested": 20,
            "policy_cache_rows_hit": 15,
            "policy_cache_rows_missed": 5,
            "policy_cache_rows_refilled": 5,
            "policy_cache_full_refill_ticks": 1,
            "policy_cache_partial_refill_ticks": 1,
        },
        "behavior_metrics": {
            "row_count": 6,
            "rows_digest_sha256": SHA,
            "cohort_size": 4,
            "cohort_survivors": 3,
            "action_total": 20,
            "move_total": 8,
            "move_n": 3,
            "move_s": 2,
            "move_e": 1,
            "move_w": 2,
            "gather": 6,
            "rest": 4,
            "attack": 2,
        },
        "lineage_metrics": {
            "record_count": 8,
            "records_digest_sha256": SHA,
            "active_lineages": 3,
            "dominant_lineage_id": "1",
            "dominant_lineage_population_share": 0.5,
            "top3_lineage_population_share": 1.0,
            "lineage_population_entropy": 0.9,
            "lineage_reproduction_entropy": 0.8,
            "dominant_lineage_reproduction_share": 0.5,
            "rare_persistent_lineages": 1,
            "lineage_collapse_warning": False,
            "lineage_collapse_streak": 0,
        },
        "patch_metrics": None,
        "undefined_fields": [
            "patch_metrics",
            "terminal.failure_generation",
            "terminal.failure_stage",
            "terminal.failure_tick",
            "terminal.recoverable_error_code",
            "terminal.recoverable_error_digest_sha256",
        ],
    }


def _patch_row() -> dict[str, Any]:
    row = _flat_row()
    row["world"] = "patch"
    row["patch_metrics"] = {
        "patch_movement_energy_event_count": 4,
        "patch_spawn_sync_rows": 2,
        "typed_death_event_rows": 1,
        "env_rng_padding_draws": 0,
        "patch_food_harvest_total": 5.0,
        "patch_scent_emit_total": 2.0,
        "patch_waste_emit_total": 1.0,
        "patch_movement_energy_total": 0.5,
        "energy_collected_delta": 5.0,
        "patch_scent_move_queued_total": 1.0,
        "patch_waste_move_queued_total": 0.5,
        "patch_scent_gather_queued_total": 1.0,
        "patch_waste_gather_queued_total": 0.5,
        "patch_movement_energy_extra_total": 0.5,
        "patch_scent_emitted_total": 2.0,
        "patch_waste_emitted_total": 1.0,
        "patch_food_depleted_frac": 0.25,
        "patch_food_full_frac": 0.5,
        "patch_density_hotspot_frac": 0.125,
    }
    row["undefined_fields"].remove("patch_metrics")
    return row


def test_contract_bundle_truthfully_separates_phase_a_from_canonical_v1() -> None:
    result = validator.validate_contract_bundle()

    assert result["ok"] is True
    assert result["read_only"] is True
    assert result["phase_a"]["phase_a_width"] == 28
    assert result["phase_a"]["phase_a_is_canonical_v1"] is False
    assert result["canonical_native_capability"] is False
    assert result["schedule"]["s8_runtime_qualified"] is False
    assert result["digest_preimages"] == {
        "schema_version": "eiel_compact_summary_digest_preimages_v1",
        "binding_count": 10,
        "digest_preimages_frozen": True,
        "native_digest_plane_producer_complete": False,
    }
    assert result["simulation_runs_started"] == 0
    assert result["cuda_builds_started"] == 0


def test_phase_a_column_order_and_capability_truth_are_exact() -> None:
    abi = validator.load_json(validator.DEFAULT_ABI_PATH)

    assert tuple(abi["implemented_phase_a"]["columns"]) == validator.PHASE_A_COLUMNS
    assert abi["implemented_phase_a"]["shape"] == ["generation_count", "cells", 28]
    assert abi["capability_truth"] == {
        "resident_generation_summaries_supported": False,
        "eiel_compact_generation_summary_v1_supported": False,
        "phase_a_generation_summary_scalar_supported": True,
        "phase_a_is_canonical_v1": False,
    }
    assert abi["precise_missing_native_producer"]


def test_target_plane_layout_is_lossless_bounded_and_exact() -> None:
    target = validator.load_json(validator.DEFAULT_ABI_PATH)["target_canonical_v1"]

    assert [plane["name"] for plane in target["planes"]] == [
        "validity",
        "status_codes",
        "integer_values",
        "float_values",
        "digest_values",
        "optionality_mask",
        "text_records",
    ]
    assert target["logical_bytes_per_generation_cell"] == 1336
    assert {name: len(fields) for name, fields in target["field_registry"].items()} == {
        "status_codes": 10,
        "integer_values": 76,
        "float_values": 21,
        "digest_values": 10,
        "text_records": 3,
    }
    assert len(target["optionality_bits"]) == 32
    assert target["optionality_bits"][17] == "patch_metrics"
    assert target["text_rules"]["record_width"] == 65


def test_abi_v2_uses_general_defined_mask_and_one_proven_text_record() -> None:
    receipt = validator.validate_native_abi_v2_contracts(
        validator.load_json(validator.DEFAULT_ABI_V2_PATH),
        validator.load_json(validator.DEFAULT_PLANE_LAYOUT_V2_PATH),
        validator.load_json(validator.DEFAULT_PROJECTION_V2_PATH),
        validator.load_json(validator.DEFAULT_INVALID_ROW_V2_PATH),
    )

    assert receipt == {
        "schema_version": "eiel_compact_summary_native_abi_v2",
        "wire_schema_version": "eiel_compact_generation_summary_v1",
        "logical_bytes_per_generation_cell": 1218,
        "defined_field_count": 32,
        "status_field_count": 12,
        "integer_field_count": 76,
        "float_field_count": 21,
        "digest_field_count": 10,
        "bounded_text_field_count": 1,
        "capability_promoted": False,
    }


def test_abi_v2_rejects_text_scope_growth_and_premature_capability() -> None:
    abi = validator.load_json(validator.DEFAULT_ABI_V2_PATH)
    layout = validator.load_json(validator.DEFAULT_PLANE_LAYOUT_V2_PATH)
    projection = validator.load_json(validator.DEFAULT_PROJECTION_V2_PATH)
    invalid = validator.load_json(validator.DEFAULT_INVALID_ROW_V2_PATH)

    widened = copy.deepcopy(abi)
    widened["representation_decisions"]["bounded_text_fields"].append(
        "terminal.failure_stage"
    )
    with pytest.raises(validator.ContractValidationError, match="only the proven"):
        validator.validate_native_abi_v2_contracts(
            widened, layout, projection, invalid
        )

    promoted = copy.deepcopy(abi)
    promoted["capability"]["eiel_compact_generation_summary_v1_supported"] = True
    with pytest.raises(validator.ContractValidationError, match="remain false"):
        validator.validate_native_abi_v2_contracts(
            promoted, layout, projection, invalid
        )


def test_wire_profile_fails_closed_on_plane_bounds_and_numeric_type(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    row = _flat_row()
    row["alive"] = 9223372036854775808
    with pytest.raises(validator.ContractValidationError, match="at most"):
        validator.validate_summary(row, schema, ownership)

    row = _flat_row()
    row["generation_metrics"]["peak_injury"] = 1
    with pytest.raises(validator.ContractValidationError, match="canonical Python float"):
        validator.validate_summary(row, schema, ownership)

    row = _flat_row()
    row["terminal"].update(
        recoverable_error_status="present",
        recoverable_error_code="x" * 65,
        recoverable_error_digest_sha256=SHA,
    )
    row["undefined_fields"].remove("terminal.recoverable_error_code")
    row["undefined_fields"].remove("terminal.recoverable_error_digest_sha256")
    with pytest.raises(validator.ContractValidationError, match="64 UTF-8 bytes"):
        validator.validate_summary(row, schema, ownership)


def test_digest_preimages_fail_closed_on_rng_split_or_tape_capacity() -> None:
    schema = validator.load_json(validator.DEFAULT_SCHEMA_PATH)
    contract = validator.load_json(validator.DEFAULT_DIGEST_PREIMAGES_PATH)

    split = copy.deepcopy(contract)
    split["bindings"]["rng.authority_digest_sha256"][
        "split_state_or_cursor_digests"
    ] = True
    with pytest.raises(validator.ContractValidationError, match="composite RNG"):
        validator.validate_digest_preimages(schema, split)

    capacity = copy.deepcopy(contract)
    capacity["bindings"]["rng.policy_mutation_tape_digest_sha256"][
        "unconsumed_capacity_included"
    ] = True
    with pytest.raises(validator.ContractValidationError, match="tape fields"):
        validator.validate_digest_preimages(schema, capacity)


@pytest.mark.parametrize("row_factory", [_flat_row, _patch_row])
def test_valid_flat_and_patch_rows_are_read_only_and_closed(
    schema: dict[str, Any],
    ownership: dict[str, Any],
    row_factory: Any,
) -> None:
    row = row_factory()
    before = copy.deepcopy(row)

    receipt = validator.validate_summary(row, schema, ownership)

    assert receipt["ok"] is True
    assert receipt["read_only"] is True
    assert row == before


def test_missing_and_extra_top_level_or_nested_fields_fail_closed(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    row = _flat_row()
    row.pop("rng")
    with pytest.raises(validator.ContractValidationError, match="missing required"):
        validator.validate_summary(row, schema, ownership)

    row = _flat_row()
    row["rng"]["state_payload"] = [1, 2, 3]
    with pytest.raises(validator.ContractValidationError, match="additional field"):
        validator.validate_summary(row, schema, ownership)


@pytest.mark.parametrize("mutation", ["missing", "extra", "misowned"])
def test_ownership_fails_closed_on_missing_extra_or_misowned_fields(
    schema: dict[str, Any], ownership: dict[str, Any], mutation: str
) -> None:
    candidate = copy.deepcopy(ownership)
    if mutation == "missing":
        candidate["field_owners"].pop("alive")
        message = "missing required"
    elif mutation == "extra":
        candidate["field_owners"]["invented"] = "wire_contract"
        message = "additional field"
    else:
        candidate["field_owners"]["alive"] = "cpu_behavior_reporting"
        message = "misowned field"

    with pytest.raises(validator.ContractValidationError, match=message):
        validator.validate_ownership(schema, candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["generation_end"].update(generation=7), "generation markers"),
        (lambda row: row["deaths_by_class"].update(health_depleted=2), "death-class"),
        (lambda row: row["events"].update(authoritative_cursor=15), "authoritative event cursor"),
        (lambda row: row["identity"].update(population_next_id=9), "exceed every active"),
        (lambda row: row["behavior_metrics"].update(action_total=21), "behavior action"),
        (lambda row: row["generation_metrics"].update(policy_cache_rows_hit=14), "policy-cache request"),
    ],
)
def test_cross_field_invariants_fail_closed(
    schema: dict[str, Any],
    ownership: dict[str, Any],
    mutation: Any,
    message: str,
) -> None:
    row = _flat_row()
    mutation(row)
    with pytest.raises(validator.ContractValidationError, match=message):
        validator.validate_summary(row, schema, ownership)


def test_patch_and_null_path_optionality_fail_closed(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    row = _flat_row()
    row["world"] = "patch"
    with pytest.raises(validator.ContractValidationError, match="Patch optionality"):
        validator.validate_summary(row, schema, ownership)

    row = _flat_row()
    row["undefined_fields"].remove("terminal.failure_tick")
    with pytest.raises(validator.ContractValidationError, match="every and only null"):
        validator.validate_summary(row, schema, ownership)


def test_terminal_failure_requires_sticky_coordinates(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    row = _flat_row()
    row["terminal"]["status"] = "failed"
    with pytest.raises(validator.ContractValidationError, match="sticky failure"):
        validator.validate_summary(row, schema, ownership)


def test_exact_row_comparison_reports_first_mismatch(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    expected = _flat_row()
    exact = validator.compare_exact_rows(expected, copy.deepcopy(expected), schema, ownership)
    assert exact["exact"] is True
    assert exact["first_mismatch"] is None
    assert exact["canonical_bytes_equal"] is True

    actual = copy.deepcopy(expected)
    actual["state_digests"]["authoritative_state_digest_sha256"] = "1" * 64
    mismatch = validator.compare_exact_rows(expected, actual, schema, ownership)
    assert mismatch["exact"] is False
    assert mismatch["first_mismatch"]["path"] == (
        "$.state_digests.authoritative_state_digest_sha256"
    )
    with pytest.raises(validator.ContractValidationError, match="exact row mismatch"):
        validator.assert_exact_rows(expected, actual, schema, ownership)


def test_exact_comparison_distinguishes_negative_zero(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    expected = _flat_row()
    actual = copy.deepcopy(expected)
    expected["generation_metrics"]["peak_injury"] = 0.0
    actual["generation_metrics"]["peak_injury"] = -0.0

    receipt = validator.compare_exact_rows(expected, actual, schema, ownership)

    assert receipt["exact"] is False
    assert receipt["first_mismatch"]["path"] == "$.generation_metrics.peak_injury"


def test_exact_comparison_uses_canonical_object_order(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    expected = _flat_row()
    actual = copy.deepcopy(expected)
    actual["generation_start"] = {
        "tick": actual["generation_start"]["tick"],
        "generation": actual["generation_start"]["generation"],
    }

    receipt = validator.compare_exact_rows(expected, actual, schema, ownership)

    assert receipt["exact"] is True
    assert receipt["canonical_bytes_equal"] is True


def test_bool_and_float_are_not_accepted_as_integers(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    for value in (False, 6.0):
        row = _flat_row()
        row["alive"] = value
        with pytest.raises(validator.ContractValidationError, match="type"):
            validator.validate_summary(row, schema, ownership)


def test_integer_one_is_not_accepted_for_boolean_const(
    schema: dict[str, Any], ownership: dict[str, Any]
) -> None:
    row = _flat_row()
    row["summary_valid"] = 1
    with pytest.raises(validator.ContractValidationError, match="must equal True"):
        validator.validate_summary(row, schema, ownership)


def test_strict_loader_rejects_duplicate_keys_and_nan(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(validator.ContractValidationError, match="duplicate"):
        validator.load_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(validator.ContractValidationError, match="non-finite"):
        validator.load_json(nonfinite)


def test_cli_bundle_and_exact_compare_receipts_are_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert validator.main([]) == 0
    bundle = json.loads(capsys.readouterr().out)
    assert bundle["ok"] is True
    assert bundle["canonical_native_capability"] is False

    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    expected.write_text(json.dumps(_flat_row()) + "\n", encoding="utf-8")
    actual.write_text(json.dumps(_flat_row()) + "\n", encoding="utf-8")
    assert validator.main(["--expected", str(expected), "--actual", str(actual)]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["exact"] is True
