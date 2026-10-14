from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import build_phase_f_v2_matrix
from evolution.phase_f_v2.receipts import create_matrix_registry_receipt
from perf.cuda_architecture_b_phase_e_gpu_gate import PhaseEGpuAuthorization
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    build_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_forced_combination_oracle_v2 import (
    build_phase_f_cpu_expectation_v2,
)
from perf.eiel_phase_f_execution_source_v2 import (
    prepare_phase_f_execution_source_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    materialize_phase_f_source_v2,
)
import perf.eiel_phase_f_forced_combination_battery_v2 as battery


@pytest.fixture(scope="module")
def matrix():
    return build_phase_f_v2_matrix()


@pytest.fixture(scope="module")
def registry(matrix):
    return build_phase_f_fixture_registry_v2(matrix)


@pytest.fixture(scope="module")
def plan(matrix, registry):
    return battery.build_phase_f_battery_plan_v2(
        matrix=matrix, fixture_registry=registry
    )


def test_plan_freezes_exact_accounting_and_smoke_order(plan) -> None:
    assert plan["logical_case_count"] == 120
    assert plan["cpu_expectation_count"] == 134
    assert plan["uninterrupted_invocation_count"] == 240
    assert plan["resumed_invocation_count"] == 28
    assert plan["accelerated_invocation_count"] == 268
    assert len(plan["cpu_expectations"]) == 134
    assert len(plan["accelerated_invocations"]) == 268
    assert [row["execution_id"] for row in plan["accelerated_invocations"][:4]] == (
        plan["smoke_execution_ids"]
    )
    assert len({row["execution_id"] for row in plan["accelerated_invocations"]}) == 268
    assert plan["plan_sha256"] == canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def test_f1_disabled_and_enabled_empty_are_whole_invocation_recipes(registry) -> None:
    cases = {row["case_id"]: row for row in registry["cases"]}
    disabled = cases["pf2_f01_b12_r0"]["execution_recipe"]
    empty = cases["pf2_f01_b12_r1"]["execution_recipe"]
    assert disabled["scope"] == empty["scope"] == "whole_single_cell_invocation"
    assert disabled["modular_enabled"] is False
    assert empty["modular_enabled"] is True


def test_stage_sensitive_variants_bind_survival_steps_to_formal_stage_indices(
    matrix, registry
) -> None:
    cases = {row["case_id"]: row for row in matrix["cases"]}
    fixtures = {row["case_id"]: row for row in registry["cases"]}
    expected = {
        "pf2_f03_b12_r0": {6: (20, 1), 14: (20, 1)},
        "pf2_f07_b12_r0": {
            1: (20, 1),
            9: (20, 1),
            2: (40, 2),
            10: (40, 2),
        },
    }
    for case_id, slots in expected.items():
        case = cases[case_id]
        source = materialize_phase_f_source_v2(
            fixtures[case_id],
            source_seed=case["fixture_blueprint"]["construction_seed"],
        )
        for slot, (survival_steps, stage_index) in slots.items():
            assert (
                int(source.generation_tensors["survival_steps"][slot]) == survival_steps
            )
            assert (
                int(source.formal_oracle_inputs[slot].stage_indices.item())
                == stage_index
            )


def test_disabled_invocation_uses_zero_audit_and_unchanged_source_bank(
    matrix, registry
) -> None:
    case = next(row for row in matrix["cases"] if row["case_id"] == "pf2_f01_b12_r0")
    fixture = next(
        row for row in registry["cases"] if row["case_id"] == case["case_id"]
    )
    source = materialize_phase_f_source_v2(
        fixture,
        source_seed=case["fixture_blueprint"]["construction_seed"],
    )
    modular = {
        name: value.clone()
        for name, value in source.generation_tensors.items()
        if name.startswith("phase_e_modular_")
    }
    before = {name: value.clone() for name, value in modular.items()}
    audit, residuals = battery._disabled_modular_audit_v2(
        modular, action_count=source.generation_config.action_count
    )
    assert all(value.count_nonzero().item() == 0 for value in audit.values())
    assert residuals.count_nonzero().item() == 0
    assert audit["state_before"].equal(audit["state_after"])
    assert all(modular[name].equal(before[name]) for name in modular)

    expectation = build_phase_f_cpu_expectation_v2(
        matrix=matrix, fixture_case=fixture, source_state=source
    )["expectation"]
    assert expectation["profile_and_owner_ids"]["modular_enabled"] is False
    assert expectation["routed_action_residual"] == residuals[:16].tolist()
    assert expectation["module_output"] == audit["module_output"][:16].tolist()
    assert expectation["structural_cost_components_and_total"] == [
        {
            "components": audit["structural_cost_components"][row].tolist(),
            "total": audit["structural_cost_total"][row].tolist(),
        }
        for row in range(16)
    ]


def test_first_difference_accepts_the_real_comparison_receipt_shape(
    monkeypatch, tmp_path: Path, matrix, registry, plan
) -> None:
    calls, owner = _patch_execution_dependencies(monkeypatch, fail_at=0)
    cpu_manifest = _fake_cpu_manifest(plan)
    original_compare = battery.compare_phase_f_invocation_v2

    def real_shaped_compare(**kwargs):
        receipt = original_compare(**kwargs)
        nested = receipt["comparison_receipt"]["first_difference"]
        receipt["comparison_receipt"] = {
            **nested,
            "outcome": "first_difference",
        }
        receipt["outcome"] = "first_difference"
        return receipt

    monkeypatch.setattr(
        battery, "compare_phase_f_invocation_v2", real_shaped_compare
    )
    result = battery.execute_phase_f_battery_v2(
        matrix=matrix,
        fixture_registry=registry,
        plan=plan,
        cpu_manifest=cpu_manifest,
        entry_receipt=_entry(plan, cpu_manifest),
        output_directory=tmp_path,
        source_commit="1" * 40,
        matrix_receipt_sha256="2" * 64,
        owner_executor=owner,
    )
    assert result["status"] == "failed"
    assert result["first_difference"]["check_name"] == "exact:surface"
    assert len(calls) == 1


def test_pre_tick_resume_uses_detached_restored_generation_and_formal_banks(
    matrix, registry
) -> None:
    case = next(row for row in matrix["cases"] if row["case_id"] == "pf2_f12_b12_r0")
    fixture = next(
        row for row in registry["cases"] if row["case_id"] == case["case_id"]
    )
    prepared = prepare_phase_f_execution_source_v2(
        matrix=matrix,
        fixture_registry=registry,
        case_id=case["case_id"],
    )
    source = prepared.source
    restored, evidence = battery._fresh_checkpoint_source(
        matrix=matrix,
        case=case,
        fixture=fixture,
        fixture_registry=registry,
        source=source,
        source_preparation_receipt=prepared.preparation_receipt,
        prerequisite_receipt=None,
    )
    assert restored.source_sha256 == source.source_sha256
    assert evidence["checkpoint_scope"] == "fresh_modular_storage_owner_only"
    assert evidence["full_runtime_resume_claimed"] is False
    assert all(
        restored.generation_tensors[name].data_ptr()
        != source.generation_tensors[name].data_ptr()
        for name in restored.generation_tensors
    )
    assert all(
        restored.formal_oracle_inputs[index].planes[name].data_ptr()
        != source.formal_oracle_inputs[index].planes[name].data_ptr()
        for index in range(16)
        for name in restored.formal_oracle_inputs[index].planes
    )


def test_post_boundary_resume_fails_without_public_boundary_receipt(
    matrix, registry
) -> None:
    case = next(row for row in matrix["cases"] if row["case_id"] == "pf2_f09_b12_r4")
    fixture = next(
        row for row in registry["cases"] if row["case_id"] == case["case_id"]
    )
    prepared = prepare_phase_f_execution_source_v2(
        matrix=matrix,
        fixture_registry=registry,
        case_id=case["case_id"],
    )
    source = prepared.source
    with pytest.raises(
        battery.PhaseFV2BatteryError, match="requires a prepared_prune_reuse"
    ):
        battery._fresh_checkpoint_source(
            matrix=matrix,
            case=case,
            fixture=fixture,
            fixture_registry=registry,
            source=source,
            source_preparation_receipt=prepared.preparation_receipt,
            prerequisite_receipt=None,
        )


def test_entry_receipt_cross_binds_all_authorities(monkeypatch, plan) -> None:
    commit = "1" * 40
    cpu_manifest = {
        "schema_version": battery.CPU_MANIFEST_SCHEMA_VERSION,
        "expectation_count": 134,
        "source_manifest_sha256": "4" * 64,
        "records": [],
    }
    cpu_manifest["manifest_sha256"] = canonical_sha256(cpu_manifest)
    phase_e = {"receipt_kind": "final", "phase_f_authorized": True}
    phase_e["receipt_sha256"] = canonical_sha256(phase_e)
    matrix_receipt = create_matrix_registry_receipt(
        source_commit=commit,
        preregistration_sha256="2" * 64,
        phase_e_authorization_sha256=phase_e["receipt_sha256"],
        matrix_spec_sha256=plan["matrix_sha256"],
        fixture_manifest_sha256=plan["fixture_registry_sha256"],
        cell_registry_sha256=plan["plan_sha256"],
    )
    authorization = PhaseEGpuAuthorization(
        kind="final",
        receipt_sha256=phase_e["receipt_sha256"],
        source_commit=commit,
        generation_artifact_sha256="5" * 64,
        qualification_receipt_sha256="6" * 64,
        logical_profiles=("b2-12x32-v1",),
        bounded_stage1_owner_qualified=True,
        phase_f_authorized=True,
        _marker=object(),
    )
    monkeypatch.setattr(battery, "verify_clean_frozen_checkout", lambda **kwargs: None)
    monkeypatch.setattr(
        battery, "require_phase_e_gpu_authorization", lambda *args, **kwargs: None
    )
    receipt = battery.build_phase_f_entry_receipt_v2(
        plan=plan,
        matrix_receipt=matrix_receipt,
        source_commit=commit,
        phase_e_final_receipt=phase_e,
        phase_e_authorization=authorization,
        cpu_manifest=cpu_manifest,
        source_manifest_sha256="4" * 64,
        repo_root="/not-read-in-test",
    )
    assert receipt["phase_f_authorized"] is True
    assert receipt["matrix_sha256"] == plan["matrix_sha256"]
    assert receipt["fixture_registry_sha256"] == plan["fixture_registry_sha256"]
    assert receipt["cpu_manifest_sha256"] == cpu_manifest["manifest_sha256"]
    assert receipt["phase_e_final_receipt_sha256"] == phase_e["receipt_sha256"]
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def _fake_cpu_manifest(plan):
    records = [
        {
            "case_id": row["case_id"],
            "expectation_kind": row["expectation_kind"],
            "receipt_sha256": "a" * 64,
            "path": "/not/read/in-test",
            "file_sha256": "b" * 64,
        }
        for row in plan["cpu_expectations"]
    ]
    manifest = {
        "expectation_count": 134,
        "source_manifest_sha256": "3" * 64,
        "records": records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _entry(plan, cpu_manifest):
    value = {
        "schema_version": battery.ENTRY_SCHEMA_VERSION,
        "status": "execution_entry_authorized",
        "source_commit": "1" * 40,
        "matrix_sha256": plan["matrix_sha256"],
        "matrix_receipt_sha256": "2" * 64,
        "plan_sha256": plan["plan_sha256"],
        "fixture_registry_sha256": plan["fixture_registry_sha256"],
        "source_manifest_sha256": cpu_manifest["source_manifest_sha256"],
        "cpu_manifest_sha256": cpu_manifest["manifest_sha256"],
        "phase_e_final_receipt_sha256": "4" * 64,
        "phase_f_authorized": True,
        "smoke_required_before_full": True,
        "logical_case_count": 120,
        "cpu_expectation_count": 134,
        "accelerated_invocation_count": 268,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _patch_execution_dependencies(monkeypatch, *, fail_at: int | None = None):
    calls = []

    def fake_source(fixture, *, source_seed):
        return SimpleNamespace(
            source_sha256=f"{source_seed:064x}",
            profile_id=fixture["profile_id"],
        )

    monkeypatch.setattr(
        battery,
        "prepare_phase_f_execution_source_v2",
        lambda **kwargs: SimpleNamespace(
            source=fake_source(
                kwargs["fixture_registry"]["cases"][0],
                source_seed=next(
                    case["fixture_blueprint"]["construction_seed"]
                    for case in kwargs["matrix"]["cases"]
                    if case["case_id"] == kwargs["case_id"]
                ),
            ),
            checkpoint_prerequisite_receipt=None,
            preparation_receipt={"case_id": kwargs["case_id"]},
            boundary_receipts=(),
        ),
    )
    monkeypatch.setattr(
        battery,
        "_fresh_checkpoint_source",
        lambda **kwargs: (
            fake_source(
                kwargs["fixture"],
                source_seed=kwargs["case"]["fixture_blueprint"]["construction_seed"],
            ),
            {
                "checkpoint_scope": "fresh_modular_storage_owner_only",
                "full_runtime_resume_claimed": False,
            },
        ),
    )
    monkeypatch.setattr(
        battery,
        "_load_expectation",
        lambda record: {"expectation": {"surface": "expected"}},
    )

    def fake_compare(**kwargs):
        index = len(calls) - 1
        passed = fail_at is None or index != fail_at
        execution_id = kwargs["bindings"]["execution_id"]
        return {
            "receipt_sha256": f"{index + 1:064x}",
            "outcome": "no_mismatch" if passed else "first_difference",
            "comparison_receipt": {
                "first_difference": None if passed else {"check_name": "exact:surface"}
            },
            "execution_id": execution_id,
        }

    monkeypatch.setattr(battery, "compare_phase_f_invocation_v2", fake_compare)

    def owner(source, item, modular_enabled):
        calls.append((item["execution_id"], modular_enabled, source.source_sha256))
        return {"surface": "observed"}

    return calls, owner


def test_battery_invokes_exactly_one_fresh_owner_per_item_and_resumes_without_replay(
    monkeypatch, tmp_path: Path, matrix, registry, plan
) -> None:
    calls, owner = _patch_execution_dependencies(monkeypatch)
    cpu_manifest = _fake_cpu_manifest(plan)
    kwargs = dict(
        matrix=matrix,
        fixture_registry=registry,
        plan=plan,
        cpu_manifest=cpu_manifest,
        entry_receipt=_entry(plan, cpu_manifest),
        output_directory=tmp_path,
        source_commit="1" * 40,
        matrix_receipt_sha256="2" * 64,
        owner_executor=owner,
    )
    result = battery.execute_phase_f_battery_v2(**kwargs)
    assert result["status"] == "passed"
    assert result["smoke_status"] == "passed"
    assert result["completed_count"] == 268
    assert len(calls) == 268
    assert [row[0] for row in calls[:4]] == plan["smoke_execution_ids"]
    assert calls[0][1] is False
    assert calls[1][1] is True

    repeated = battery.execute_phase_f_battery_v2(**kwargs)
    assert repeated["completed_count"] == 268
    assert len(calls) == 268


def test_first_difference_stops_immediately_and_cannot_resume(
    monkeypatch, tmp_path: Path, matrix, registry, plan
) -> None:
    calls, owner = _patch_execution_dependencies(monkeypatch, fail_at=2)
    cpu_manifest = _fake_cpu_manifest(plan)
    kwargs = dict(
        matrix=matrix,
        fixture_registry=registry,
        plan=plan,
        cpu_manifest=cpu_manifest,
        entry_receipt=_entry(plan, cpu_manifest),
        output_directory=tmp_path,
        source_commit="1" * 40,
        matrix_receipt_sha256="2" * 64,
        owner_executor=owner,
    )
    result = battery.execute_phase_f_battery_v2(**kwargs)
    assert result["status"] == "failed"
    assert result["smoke_status"] == "failed"
    assert result["completed_count"] == 3
    assert result["first_difference"]["check_name"] == "exact:surface"
    assert len(calls) == 3
    with pytest.raises(battery.PhaseFV2BatteryError, match="first difference"):
        battery.execute_phase_f_battery_v2(**kwargs)
    assert len(calls) == 3


def test_phase_e_validation_requires_phase_f_flag_and_both_profile_envelope(
    monkeypatch,
) -> None:
    authorization = SimpleNamespace(
        source_commit="1" * 40,
        phase_f_authorized=True,
        logical_profiles=("b2-12x32-v1",),
    )
    required = []
    monkeypatch.setattr(
        battery,
        "validate_phase_e_gpu_final_receipt",
        lambda *args, **kwargs: authorization,
    )
    monkeypatch.setattr(
        battery,
        "require_phase_e_gpu_authorization",
        lambda auth, **kwargs: required.append((auth, kwargs)),
    )
    final = {
        "logical_profiles": ["b2-12x32-v1", "b2-16x48-v1"],
    }
    observed = battery.validate_phase_e_final_for_phase_f_v2(
        final_receipt=final,
        qualification_receipt={},
        bindings=SimpleNamespace(),
        source_commit="1" * 40,
    )
    assert observed is authorization
    assert required == [(authorization, {"final": True, "profile_id": "b2-12x32-v1"})]
    altered = deepcopy(final)
    altered["logical_profiles"] = ["b2-12x32-v1"]
    with pytest.raises(battery.PhaseFV2BatteryError, match="profile envelope"):
        battery.validate_phase_e_final_for_phase_f_v2(
            final_receipt=altered,
            qualification_receipt={},
            bindings=SimpleNamespace(),
            source_commit="1" * 40,
        )


def test_default_cuda_executor_is_not_called_by_cpu_tests(monkeypatch) -> None:
    monkeypatch.setattr(
        battery.torch.cuda,
        "is_available",
        lambda: False,
    )
    with pytest.raises(battery.PhaseFV2BatteryError, match="CUDA is unavailable"):
        battery.execute_phase_f_cuda_owner_observation_v2(SimpleNamespace(), {}, False)
