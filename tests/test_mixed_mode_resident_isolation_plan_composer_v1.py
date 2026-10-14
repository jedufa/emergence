from __future__ import annotations

from copy import deepcopy
import json

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.inheritance_cross_path_mechanic_gate_adapter_v1 import (
    RUN_PAYLOAD_SCHEMA_VERSION,
)
from engine.campaigns.mixed_mode_resident_isolation_gate import (
    RUN_KIND_ALONE,
    RUN_KIND_CHECKPOINT_BASELINE,
    RUN_KIND_CHECKPOINT_FRESH_OWNER,
    RUN_KIND_DUPLICATE,
    RUN_KIND_MIXED,
    RUN_KIND_OVERFLOW,
    RUN_KIND_PERMUTED,
)
from engine.campaigns.mixed_mode_resident_isolation_plan_composer_v1 import (
    CONFIG_SCHEMA_VERSION,
    MAX_NATIVE_INT64,
    REQUIRED_GATE_CHECK_IDS,
    STORAGE_COMPACT_COMPLETE_REQUEST_ID,
    STORAGE_EXACT_FULL_REQUEST_ID,
    STORAGE_PARITY_FIELDS,
    ExactFullCapacityEnvelopeV1,
    MixedModeIsolationComposerConfigV1,
    MixedModeIsolationPlanComposerError,
    compose_mixed_mode_resident_isolation_plan_v1,
    main,
    publish_composed_plan_v1,
    verify_composed_plan_artifact_v1,
)


REFILL = "generation_boundary_refill"
LIVE = "ordinary_live_birth"


def _selection() -> dict[str, object]:
    return {
        "selection_mode": "uniform_eligible_parent_v1",
        "eligible_parent_source": "authoritative_eligible_parent_set_v1",
        "draw_distribution": "uniform_v1",
        "deterministic": True,
        "blinded_dimensions": [
            "genotype",
            "lineage",
            "policy",
            "strategy",
        ],
    }


def _cell(logical_id: str, path_id: str, mode_id: str, seed: int) -> dict:
    return {
        "config": {
            "logical_cell_id": logical_id,
            "equivalence_key": logical_id,
            "reproduction_path": path_id,
            "selection": _selection(),
            "parent_arity": 2,
            "donor_draws_per_successful_birth": 2,
            "authorized_rng_counter_ids": ["selection_transaction"],
        },
        "mode_id": mode_id,
        "seed": seed,
    }


def _payload(tmp_path) -> dict[str, object]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "execution_callback_id": "mixed-mode-isolation-callback-v1",
        "execution_authority_sha256": "a" * 64,
        "candidate_artifact_bindings": {
            "exact_full": {
                "candidate_authority_sha256": "b" * 64,
                "artifact_sha256": "c" * 64,
            },
            "compact_complete": {
                "candidate_authority_sha256": "d" * 64,
                "artifact_sha256": "e" * 64,
            },
        },
        "preregistered_at_utc": "2026-07-24T09:01:00+00:00",
        "runtime_roots": {
            "runs": str(tmp_path / "runs"),
            "owners": str(tmp_path / "owners"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
        "execution": {
            "generations": 2,
            "ticks_per_generation": 3,
            "checkpoint_split_generation": 1,
        },
        "exact_full_capacities": {
            "A": 24,
            "S": 48,
            "C": 4,
            "K": 300,
            "max_resident_population": 4,
            "current_agent_id_capacity": 4,
            "population_next_id": 4,
            "max_births_per_tick": 1,
            "retained_agent_id_capacity": 4,
        },
        "minimum_live_birth_exposure": {
            "threshold_id": "frozen-live-birth-minimum-v1",
            "metric": "successful_births",
            "minimum": 2,
            "frozen_at_utc": "2026-07-24T09:00:00+00:00",
        },
        "overflow": {
            "target_path_id": LIVE,
            "error_code": "attempt_capacity_exceeded",
            "capacity_kind": "attempt",
            "effective_bound": 0,
            "expected_status_code": 2,
        },
        "live_birth_path_ids": [LIVE],
        "cells": [
            _cell("refill-cell", REFILL, "refill-mode", 1101),
            _cell("live-cell", LIVE, "live-mode", 1102),
        ],
    }


def _artifact(tmp_path) -> dict[str, object]:
    config = MixedModeIsolationComposerConfigV1.from_payload(
        _payload(tmp_path)
    )
    return compose_mixed_mode_resident_isolation_plan_v1(config)


def _runs(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    runner = artifact["runner_plan"]
    assert isinstance(runner, dict)
    return {run["run_id"]: run for run in runner["runs"]}


def test_composer_materializes_complete_gate_and_runner_preregistration(
    tmp_path,
) -> None:
    artifact = _artifact(tmp_path)
    verified = verify_composed_plan_artifact_v1(artifact)
    runner = verified["runner_plan"]
    gate = runner["gate_plan"]
    runs = _runs(artifact)

    assert artifact["required_gate_check_ids"] == list(
        REQUIRED_GATE_CHECK_IDS
    )
    assert gate["mixed_reference_run_id"] == "mixed-reference"
    assert gate["permutation_run_ids"] == ["mixed-permutation-reverse"]
    assert gate["duplicate_run_ids"] == [
        "mixed-duplicate-refill-cell",
        "mixed-duplicate-live-cell",
    ]
    assert gate["duplicate_required_path_ids"] == [REFILL, LIVE]
    assert gate["live_birth_path_ids"] == [LIVE]
    assert gate["overflow_cases"] == [
        {
            "reference_run_id": "overflow-reference",
            "fault_run_id": "overflow-fault",
            "target_occurrence_id": "live-cell",
            "overflow_error_code": "attempt_capacity_exceeded",
        }
    ]
    assert gate["checkpoint_cases"] == [
        {
            "uninterrupted_run_id": "checkpoint-uninterrupted",
            "fresh_owner_run_id": "checkpoint-fresh-owner",
        }
    ]

    expected_kinds = {
        "mixed-reference": RUN_KIND_MIXED,
        "alone-refill-cell": RUN_KIND_ALONE,
        "alone-live-cell": RUN_KIND_ALONE,
        "mixed-permutation-reverse": RUN_KIND_PERMUTED,
        "mixed-duplicate-refill-cell": RUN_KIND_DUPLICATE,
        "mixed-duplicate-live-cell": RUN_KIND_DUPLICATE,
        "overflow-reference": RUN_KIND_MIXED,
        "overflow-fault": RUN_KIND_OVERFLOW,
        "checkpoint-uninterrupted": RUN_KIND_CHECKPOINT_BASELINE,
        "checkpoint-fresh-owner": RUN_KIND_CHECKPOINT_FRESH_OWNER,
        "reporting-enabled": RUN_KIND_MIXED,
        "reporting-disabled": RUN_KIND_MIXED,
    }
    assert {
        run_id: run["run_kind"] for run_id, run in runs.items()
    } == expected_kinds

    mixed_order = [
        cell["config"]["logical_cell_id"]
        for cell in runs["mixed-reference"]["cells"]
    ]
    permutation_order = [
        cell["config"]["logical_cell_id"]
        for cell in runs["mixed-permutation-reverse"]["cells"]
    ]
    assert mixed_order == ["refill-cell", "live-cell"]
    assert permutation_order == list(reversed(mixed_order))
    refill_duplicates = runs["mixed-duplicate-refill-cell"]["cells"]
    live_duplicates = runs["mixed-duplicate-live-cell"]["cells"]
    assert len(refill_duplicates) == len(live_duplicates) == 2
    assert [
        cell["config"]["reproduction_path"] for cell in refill_duplicates
    ] == [REFILL, REFILL]
    assert [
        cell["config"]["reproduction_path"] for cell in live_duplicates
    ] == [LIVE, LIVE]
    assert [
        cell["expected_neighbor_path_ids"] for cell in refill_duplicates[:2]
    ] == [[REFILL], [REFILL]]
    assert [
        cell["expected_neighbor_path_ids"] for cell in live_duplicates[:2]
    ] == [[LIVE], [LIVE]]


def test_run_payloads_bind_roots_controls_checkpoint_and_reporting(
    tmp_path,
) -> None:
    artifact = _artifact(tmp_path)
    runs = _runs(artifact)
    roots: set[tuple[str, str, str]] = set()

    for run in runs.values():
        payload = run["execution_payload"]
        segment = payload["segment_request"]
        assert payload["schema_version"] == RUN_PAYLOAD_SCHEMA_VERSION
        assert payload["candidate_authority_sha256"] == "b" * 64
        assert segment["candidate_artifact_binding_sha256"] == "c" * 64
        assert [row["batch_position"] for row in segment["physical_order"]] == (
            list(range(len(run["cells"])))
        )
        run_roots = (
            segment["run_root"],
            segment["owner_root"],
            segment["checkpoint_root"],
        )
        assert run_roots not in roots
        roots.add(run_roots)
        assert all(root.startswith(str(tmp_path)) for root in run_roots)

    fault_payload = runs["overflow-fault"]["execution_payload"]
    assert fault_payload["overflow_fault"] == {
        "target_occurrence_id": "live-cell",
        "error_code": "attempt_capacity_exceeded",
    }
    assert fault_payload["segment_request"]["overflow_control"] == {
        "target_occurrence_id": "live-cell",
        "error_code": "attempt_capacity_exceeded",
        "capacity_kind": "attempt",
        "effective_bound": 0,
        "expected_status_code": 2,
    }
    assert runs["checkpoint-fresh-owner"]["fresh_owner"] is True
    fresh = runs["checkpoint-fresh-owner"]["execution_payload"][
        "segment_request"
    ]
    assert fresh["checkpoint_role"] == "fresh_owner_continuation"
    assert fresh["checkpoint_source_run_id"] == "checkpoint-uninterrupted"
    assert fresh["checkpoint_source_root"].endswith(
        "/checkpoint-uninterrupted"
    )
    assert (
        runs["reporting-enabled"]["execution_payload"]["reporting_enabled"]
        is True
    )
    assert (
        runs["reporting-disabled"]["execution_payload"]["reporting_enabled"]
        is False
    )


def test_storage_mode_parity_is_honestly_preregistered(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    runs = _runs(artifact)
    parity = artifact["storage_mode_parity"]
    exact = parity["exact_full_request"]
    compact = parity["compact_complete_request"]
    comparison = parity["comparison_contract"]

    assert STORAGE_EXACT_FULL_REQUEST_ID not in runs
    assert STORAGE_COMPACT_COMPLETE_REQUEST_ID not in runs
    assert {
        run["execution_payload"]["exposure_storage_mode"]
        for run in runs.values()
    } == {"exact_full"}
    assert exact["isolation_callback_authorized"] is False
    assert compact["isolation_callback_authorized"] is False
    assert exact["candidate_authority_sha256"] == "b" * 64
    assert exact["candidate_artifact_sha256"] == "c" * 64
    assert compact["candidate_authority_sha256"] == "d" * 64
    assert compact["candidate_artifact_sha256"] == "e" * 64
    assert exact["physical_order"] == compact["physical_order"]
    assert exact["storage_mode"] == "exact_full"
    assert compact["storage_mode"] == "compact_complete"
    assert exact["exposure"]["S"] == 48
    assert compact["exposure"]["S"] == 1
    assert exact["exposure"][
        "per_attempt_eligible_parent_id_sets_available"
    ] is True
    assert compact["exposure"][
        "per_attempt_eligible_parent_id_sets_available"
    ] is False
    assert "eligible_parent_ids" not in json.dumps(compact, sort_keys=True)
    assert comparison["required_comparisons"] == list(STORAGE_PARITY_FIELDS)
    assert comparison[
        "compact_per_attempt_eligible_parent_id_sets_forbidden"
    ] is True
    assert comparison["promotion_authority"] is False


def test_resealed_compact_eligible_set_fabrication_is_rejected(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    tampered = deepcopy(artifact)
    parity = tampered["storage_mode_parity"]
    compact = parity["compact_complete_request"]
    compact["exposure"][
        "per_attempt_eligible_parent_id_sets_available"
    ] = True
    compact_unsigned = dict(compact)
    compact_unsigned.pop("request_sha256")
    compact["request_sha256"] = canonical_sha256(compact_unsigned)
    comparison = parity["comparison_contract"]
    comparison["compact_complete_request_sha256"] = compact["request_sha256"]
    comparison_unsigned = dict(comparison)
    comparison_unsigned.pop("comparison_contract_sha256")
    comparison["comparison_contract_sha256"] = canonical_sha256(
        comparison_unsigned
    )
    unsigned = dict(tampered)
    unsigned.pop("artifact_sha256")
    tampered["artifact_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="cannot claim per-attempt eligible sets",
    ):
        verify_composed_plan_artifact_v1(tampered)


def test_exact_full_capacity_proof_is_bounded_and_structural(tmp_path) -> None:
    artifact = _artifact(tmp_path)
    proof = artifact["capacity_proof"]

    assert proof["required_per_cell"] == {
        "A": 12,
        "S": 24,
        "C": 4,
        "K": 292,
    }
    assert proof["checkpoint_split_generation"] == 1
    assert proof["maximum_segment_generations"] == 1
    assert proof["declared_per_cell"] == {
        "A": 24,
        "S": 48,
        "C": 4,
        "K": 300,
    }
    assert proof["agent_id_allocation"] == {
        "schema_version": (
            "eiel_resident_exposure_compact_key_capacity_envelope_v2"
        ),
        "current_capacity": 4,
        "population_next_id": 4,
        "max_births_per_tick": 1,
        "slot_capacity": 4,
        "allocation_generation_count": 2,
        "retained_capacity": 4,
        "required_aggregate_capacity": 292,
    }
    assert proof["maximum_physical_batch"] == 2
    assert proof["ordinary_attempts_per_tick_bound"] == 4
    assert proof["turnover_attempts_per_generation_bound"] == 4
    assert proof["bounded"] is True
    high_watermark = ExactFullCapacityEnvelopeV1(
        attempt_capacity_a=24,
        snapshot_capacity_s=48,
        eligible_parent_capacity_c=4,
        aggregate_capacity_k=300,
        max_resident_population=4,
        current_agent_id_capacity=4,
        population_next_id=2_147_483_647,
        max_births_per_tick=1,
        retained_agent_id_capacity=4,
    )
    assert high_watermark.required(
        segment_generations=1,
        allocation_generation_count=2,
        ticks=3,
    )["K"] == 292

    tampered = deepcopy(artifact)
    tampered_proof = tampered["capacity_proof"]
    tampered_proof["agent_id_allocation"][
        "required_aggregate_capacity"
    ] += 1
    unsigned_proof = dict(tampered_proof)
    unsigned_proof.pop("capacity_proof_sha256")
    tampered_proof["capacity_proof_sha256"] = canonical_sha256(
        unsigned_proof
    )
    unsigned_artifact = dict(tampered)
    unsigned_artifact.pop("artifact_sha256")
    tampered["artifact_sha256"] = canonical_sha256(unsigned_artifact)
    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="does not match its derived bounds",
    ):
        verify_composed_plan_artifact_v1(tampered)

    bad = _payload(tmp_path)
    bad["exact_full_capacities"]["S"] = 47
    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match=r"at least 2 \* attempt",
    ):
        MixedModeIsolationComposerConfigV1.from_payload(bad)

    bad = _payload(tmp_path)
    bad["exact_full_capacities"]["K"] = 291
    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="do not bound the declared gate",
    ):
        MixedModeIsolationComposerConfigV1.from_payload(bad)

    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="exceeds native int64",
    ):
        ExactFullCapacityEnvelopeV1(
            attempt_capacity_a=MAX_NATIVE_INT64 // 2 + 1,
            snapshot_capacity_s=MAX_NATIVE_INT64,
            eligible_parent_capacity_c=1,
            aggregate_capacity_k=1,
            max_resident_population=1,
            current_agent_id_capacity=1,
            population_next_id=1,
            max_births_per_tick=1,
            retained_agent_id_capacity=1,
        )


def test_representative_equivalence_keys_must_be_path_unique(tmp_path) -> None:
    payload = _payload(tmp_path)
    payload["cells"][1]["config"]["equivalence_key"] = "refill-cell"

    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="equivalence keys must be unique",
    ):
        MixedModeIsolationComposerConfigV1.from_payload(payload)


def test_storage_candidates_must_have_distinct_authorities_and_artifacts(
    tmp_path,
) -> None:
    payload = _payload(tmp_path)
    payload["candidate_artifact_bindings"]["compact_complete"] = dict(
        payload["candidate_artifact_bindings"]["exact_full"]
    )

    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="must be distinct",
    ):
        MixedModeIsolationComposerConfigV1.from_payload(payload)


def test_freeze_ordering_and_artifact_tamper_fail_closed(tmp_path) -> None:
    late_freeze = _payload(tmp_path)
    late_freeze["minimum_live_birth_exposure"]["frozen_at_utc"] = (
        "2026-07-24T09:02:00+00:00"
    )
    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="must not predate",
    ):
        MixedModeIsolationComposerConfigV1.from_payload(late_freeze)

    artifact = _artifact(tmp_path)
    tampered = deepcopy(artifact)
    tampered["runner_plan"]["runs"][0]["execution_payload"][
        "reporting_enabled"
    ] = False
    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="artifact_sha256 does not bind",
    ):
        verify_composed_plan_artifact_v1(tampered)


def test_atomic_publication_and_cli_refuse_overwrite(tmp_path, capsys) -> None:
    artifact = _artifact(tmp_path)
    direct_output = tmp_path / "direct" / "plan.json"
    publish_composed_plan_v1(direct_output, artifact)
    assert verify_composed_plan_artifact_v1(
        json.loads(direct_output.read_text(encoding="utf-8"))
    ) == artifact
    with pytest.raises(
        MixedModeIsolationPlanComposerError,
        match="already exists",
    ):
        publish_composed_plan_v1(direct_output, artifact)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_payload(tmp_path), sort_keys=True),
        encoding="utf-8",
    )
    cli_output = tmp_path / "cli" / "plan.json"
    assert main(
        [
            "--config",
            str(config_path),
            "--output",
            str(cli_output),
        ]
    ) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["ok"] is True
    assert status["artifact_sha256"] == json.loads(
        cli_output.read_text(encoding="utf-8")
    )["artifact_sha256"]
    with pytest.raises(SystemExit, match="already exists"):
        main(
            [
                "--config",
                str(config_path),
                "--output",
                str(cli_output),
            ]
        )
