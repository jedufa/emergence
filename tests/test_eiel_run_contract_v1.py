from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import engine.campaigns.run_contract_v1 as run_contract_module
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.campaigns.orchestration import (
    _write_run_contract_artifacts_v1,
    atomic_write_json,
    execute_campaign,
)
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.route_qualification import derive_bounded_route_campaigns_v1
from engine.campaigns.providers import default_execution_provider_registry
from engine.campaigns.run_contract_v1 import (
    CAPABILITIES,
    CHECKPOINT_BINDING_SCHEMA_VERSION,
    PROFILE_ID,
    RunContractAdmissionError,
    build_activation_report_v1,
    checkpoint_contract_binding_v1,
    resolve_run_contract_v1,
    validate_resolved_run_contract_v1,
)
from engine.qualification_plans.v1 import generate_execution_route_canary_plan
from engine.qrr.tick_native_materialization_v1 import canonical_tick_native_plan_v1
from engine.campaigns.tick_native import TickNativeExecutionRequestV1

ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT / "_spec/campaigns/v1/"
    "wdl_jungle_eed_gv2_birth_w32p256c512_b48g200t160cuda_"
    "active_impact_allocation_diagnostic_v1.yaml"
)
SOURCE_IDENTITY = {
    "git_commit": "d7a9dd4f0ef48ca86095706359af109823f00109",
    "discovery": "test_fixture",
}


def _strict_plan() -> dict[str, object]:
    return compile_campaign(load_campaign_spec(SPEC)).payload()


def _unbound_plan() -> dict[str, object]:
    plan = _strict_plan()
    plan.pop("run_contract")
    plan.pop("plan_sha256")
    return plan


def _request(mode: str = "strict") -> dict[str, str]:
    return {"mode": mode, "profile_id": PROFILE_ID}


def _strip_lifecycle_parameters(plan: dict[str, object]) -> None:
    for cell in plan["cells"]:  # type: ignore[index]
        parameters = cell["parameters"]
        for name in tuple(parameters):
            if name.startswith("production_lifecycle."):
                del parameters[name]


def _set_overlays(plan: dict[str, object], overlays: list[str]) -> None:
    invariants = plan["qualification_key"]["fixed_invariants"]  # type: ignore[index]
    invariants["runtime.world_overlays"] = json.dumps(
        sorted(overlays), separators=(",", ":")
    )


def test_full_profile_resolves_all_capabilities_and_stable_hash() -> None:
    plan = _unbound_plan()
    left = resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)
    right = resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)

    assert left["admission_status"] == "passed"
    assert left["required_capabilities"] == list(CAPABILITIES)
    assert all(row["configured"] for row in left["capabilities"].values())
    assert left["contract_hash"] == right["contract_hash"]
    assert left["source_identity"]["git_commit"] == SOURCE_IDENTITY["git_commit"]


def test_strict_checkpoint_binding_is_compact_and_uses_canonical_contract() -> None:
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request(), source_identity=SOURCE_IDENTITY
    )

    binding = checkpoint_contract_binding_v1(plan)

    assert binding == {
        "schema_version": CHECKPOINT_BINDING_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "run_contract_schema_version": "eiel_resolved_run_contract_v1",
        "run_contract_hash": plan["run_contract"]["contract_hash"],
        "required_capabilities": list(CAPABILITIES),
        "capability_contracts": binding["capability_contracts"],
    }
    assert binding["capability_contracts"]["patch"]["contract_version"]
    assert binding["capability_contracts"]["genome_v2_mutation"]["mutation_contract"]
    assert "resolved_relevant_config" not in binding


def test_legacy_or_optional_contract_has_no_checkpoint_identity() -> None:
    assert checkpoint_contract_binding_v1(_unbound_plan()) is None
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request("optional"), source_identity=SOURCE_IDENTITY
    )
    assert checkpoint_contract_binding_v1(plan) is None


def test_checkpoint_binding_refuses_stale_strict_contract() -> None:
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request(), source_identity=SOURCE_IDENTITY
    )
    plan["run_contract"]["contract_hash"] = "sha256:" + "0" * 64

    with pytest.raises(RunContractAdmissionError, match="contract hash"):
        checkpoint_contract_binding_v1(plan)


def test_bounded_route_canary_refreshes_strict_contract_identity() -> None:
    plan = _strict_plan()
    provider = default_execution_provider_registry().get(plan["resolved_profile"])
    assert provider is not None
    route_contract = provider.execution_route_contract(plan).payload()
    route_contract["execution_route_sha256"] = canonical_sha256(route_contract)
    route_contract["canary_requirements"] = {
        "qualified_ticks_per_generation": 160,
        "admitted_physical_batch_sizes": [1, 2],
        "boundary_generations": 3,
        "resident_segment_generation_min": 3,
        "odd_tail_generations": 3,
    }
    canary = generate_execution_route_canary_plan(route_contract)

    derived = derive_bounded_route_campaigns_v1(plan, canary)

    assert derived
    assert all(
        validate_resolved_run_contract_v1(case)["contract_hash"]
        == case["run_contract"]["contract_hash"]
        for case in derived
    )


def test_tick_native_materialization_refreshes_strict_contract_identity() -> None:
    plan = _strict_plan()
    materialized = canonical_tick_native_plan_v1(
        plan,
        tick_request=TickNativeExecutionRequestV1(
            total_ticks=161,
            checkpoint_ticks=(17, 160, 161),
            deliberate_restore_tick=17,
        ),
    )

    validated = validate_resolved_run_contract_v1(materialized)

    assert validated is not None
    assert validated["contract_hash"] == materialized["run_contract"]["contract_hash"]
    assert validated["contract_hash"] == plan["run_contract"]["contract_hash"]


def test_tick_native_materialization_preserves_absolute_restore_gate() -> None:
    plan = _strict_plan()
    cell_id = plan["cells"][0]["cell_id"]
    materialized = canonical_tick_native_plan_v1(
        plan,
        tick_request=TickNativeExecutionRequestV1(
            total_ticks=1121,
            checkpoint_ticks=(500, 1121),
            deliberate_restore_tick=500,
        ),
        exercise_checkpoint_resume=True,
    )

    gate = materialized["execution"]["technical_gate"]
    assert gate["required"] is True
    assert gate["cell_ids"] == [cell_id]
    assert gate["canary"] == {
        "batch_id": "technical-gate-canary",
        "restore_batch_id": "technical-gate-fresh-owner-restore",
        "time_authority": "absolute_tick_v1",
        "total_ticks": 1121,
        "checkpoint_ticks": [500, 1121],
        "cell_ids": [cell_id],
    }
    assert not {"generations", "checkpoint_milestones"}.intersection(
        gate["canary"]
    )
    assert materialized["artifact_lifecycle"]["checkpoints"]["authority"] == (
        "absolute_tick"
    )


def test_eed_runner_receipt_is_runtime_activation_evidence() -> None:
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request(), source_identity=SOURCE_IDENTITY
    )
    batch = {
        "metrics_rows": [{"generation_metrics": [{"generation": 0}]}],
        "result": {
            "runner_receipts": [
                {
                    "eed": {
                        "component_id": "world.overlay.eed_resource_impact.v1",
                        "impact_applied": True,
                    }
                }
            ]
        },
    }

    report = build_activation_report_v1(plan, [batch], campaign_status="completed")

    assert report is not None
    assert report["capabilities"]["eed"]["status"] == "EXERCISED"
    assert report["capabilities"]["eed"]["evidence"] == {
        "completed_generation_rows": 1,
        "eed_metric_projection_observed": False,
        "eed_runner_receipt_observed": True,
    }


@pytest.mark.parametrize(
    ("capability", "overlays"),
    [
        ("wdl_delayed_hazard", ["eed_resource_impact", "jungle"]),
        ("jungle", ["eed_resource_impact", "wdl_delayed_hazard"]),
        ("eed", ["jungle", "wdl_delayed_hazard"]),
    ],
)
def test_missing_overlay_is_named_in_strict_admission_failure(
    capability: str, overlays: list[str]
) -> None:
    plan = _unbound_plan()
    _set_overlays(plan, overlays)
    if capability != "eed":
        _strip_lifecycle_parameters(plan)

    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)

    assert capability in {
        row["capability"] for row in caught.value.diagnostic["failures"]
    }


def test_disabled_ordinary_birth_is_rejected_and_named() -> None:
    plan = _unbound_plan()
    for cell in plan["cells"]:  # type: ignore[index]
        cell["parameters"]["production_lifecycle.ordinary_births_enabled"] = False

    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)

    assert "genome_v2_ordinary_birth" in {
        row["capability"] for row in caught.value.diagnostic["failures"]
    }


def test_disabled_mutation_is_rejected_and_named() -> None:
    plan = _unbound_plan()
    for cell in plan["cells"]:  # type: ignore[index]
        cell["parameters"]["genome_upgrade_v2.mutation_rate_q10"] = 0
        cell["parameters"]["genome_upgrade_v2.mutation_step_q10"] = 0

    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)

    assert "genome_v2_mutation" in {
        row["capability"] for row in caught.value.diagnostic["failures"]
    }


def test_patch_validator_rejects_non_patch_effective_base(monkeypatch) -> None:
    plan = _unbound_plan()
    actual_configuration = run_contract_module.production_engine_configuration_v1(plan)
    actual_binding = run_contract_module.production_engine_component_binding_v1(
        actual_configuration
    )
    monkeypatch.setattr(
        run_contract_module,
        "production_engine_configuration_v1",
        lambda _plan: SimpleNamespace(
            **{
                name: getattr(actual_configuration, name)
                for name in actual_configuration.__dataclass_fields__
            }
            | {
                "world_base": "not_patch",
                "payload": actual_configuration.payload,
                "configuration_sha256": actual_configuration.configuration_sha256,
            }
        ),
    )
    monkeypatch.setattr(
        run_contract_module,
        "production_engine_component_binding_v1",
        lambda _configuration: actual_binding,
    )

    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)

    failures = {
        row["capability"]: row["reason"]
        for row in caught.value.diagnostic["failures"]
    }
    assert "patch" in failures
    for capability in ("wdl_delayed_hazard", "jungle", "eed"):
        assert "requires the Patch world base" in failures[capability]


def test_recombination_validator_requires_exact_fixed_profile(monkeypatch) -> None:
    actual = run_contract_module.global_genome_profile_v1("genome_upgrade_v2")
    monkeypatch.setattr(
        run_contract_module,
        "global_genome_profile_v1",
        lambda _profile_id: SimpleNamespace(
            inheritance_contract="unexpected_inheritance_profile",
            mutation_contract=actual.mutation_contract,
        ),
    )

    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(
            _unbound_plan(), _request(), source_identity=SOURCE_IDENTITY
        )

    assert "recombination_profile" in {
        row["capability"] for row in caught.value.diagnostic["failures"]
    }


def test_eed_rejects_current_cpu_route_dependency_gap() -> None:
    plan = _unbound_plan()
    plan["qualification_key"]["backend"] = "cpu_native"  # type: ignore[index]

    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(plan, _request(), source_identity=SOURCE_IDENTITY)

    failures = {
        row["capability"]: row["reason"]
        for row in caught.value.diagnostic["failures"]
    }
    assert "respec0b_resource_specialization_enable" in failures["eed"]


def test_strict_full_profile_cpu_compile_fails_before_dispatch() -> None:
    spec = load_campaign_spec(SPEC)
    spec["execution"]["backend"] = "cpu_native"

    with pytest.raises(RunContractAdmissionError) as caught:
        compile_campaign(spec)

    assert caught.value.diagnostic["failures"] == [
        {
            "capability": "eed",
            "reason": (
                "EED is not initializable on the current production CPU route: "
                "that route disables respec0b_resource_specialization_enable, "
                "which eed_resource_impact_enable requires"
            ),
        }
    ]


def test_optional_mode_reports_but_does_not_reject_invalid_profile_config() -> None:
    plan = _unbound_plan()
    for cell in plan["cells"]:  # type: ignore[index]
        cell["parameters"]["genome_upgrade_v2.mutation_rate_q10"] = 0
        cell["parameters"]["genome_upgrade_v2.mutation_step_q10"] = 0

    contract = resolve_run_contract_v1(
        plan, _request("optional"), source_identity=SOURCE_IDENTITY
    )

    assert contract["admission_status"] == "advisory_failed"
    assert contract["capabilities"]["genome_v2_mutation"]["configured"] is False


def test_legacy_spec_without_request_retains_no_contract() -> None:
    spec = load_campaign_spec(SPEC)
    spec.pop("run_contract")

    plan = compile_campaign(spec).payload()

    assert "run_contract" not in plan


def test_relevant_mutation_regime_changes_contract_hash() -> None:
    left_plan = _unbound_plan()
    right_plan = copy.deepcopy(left_plan)
    for cell in right_plan["cells"]:  # type: ignore[index]
        cell["parameters"]["genome_upgrade_v2.mutation_step_q10"] = 32

    left = resolve_run_contract_v1(
        left_plan, _request(), source_identity=SOURCE_IDENTITY
    )
    right = resolve_run_contract_v1(
        right_plan, _request(), source_identity=SOURCE_IDENTITY
    )

    assert left["contract_hash"] != right["contract_hash"]


def test_source_discovery_method_is_not_part_of_contract_identity() -> None:
    plan = _unbound_plan()
    left = resolve_run_contract_v1(
        plan,
        _request(),
        source_identity={"git_commit": "a" * 40, "discovery": "checkout"},
    )
    right = resolve_run_contract_v1(
        plan,
        _request(),
        source_identity={"git_commit": "a" * 40, "discovery": "environment"},
    )

    assert left["contract_hash"] == right["contract_hash"]


def test_checkout_source_identity_cannot_be_overridden_by_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIEL_SOURCE_GIT_COMMIT", "f" * 40)

    identity = run_contract_module._source_identity()

    assert identity["discovery"] == "repository_checkout"
    assert identity["git_commit"] != "f" * 40


def test_source_commit_changes_contract_hash() -> None:
    plan = _unbound_plan()
    left = resolve_run_contract_v1(
        plan, _request(), source_identity={"git_commit": "a" * 40}
    )
    right = resolve_run_contract_v1(
        plan, _request(), source_identity={"git_commit": "b" * 40}
    )

    assert left["contract_hash"] != right["contract_hash"]


def test_output_root_is_not_part_of_contract_identity() -> None:
    left_plan = _unbound_plan()
    right_plan = copy.deepcopy(left_plan)
    left_plan["output"]["runtime_root"] = "/tmp/operator-a/run"  # type: ignore[index]
    right_plan["output"]["runtime_root"] = "/srv/operator-b/run"  # type: ignore[index]

    left = resolve_run_contract_v1(
        left_plan, _request(), source_identity=SOURCE_IDENTITY
    )
    right = resolve_run_contract_v1(
        right_plan, _request(), source_identity=SOURCE_IDENTITY
    )

    assert left["contract_hash"] == right["contract_hash"]


def test_strict_source_identity_must_be_a_full_lowercase_commit() -> None:
    with pytest.raises(RunContractAdmissionError) as caught:
        resolve_run_contract_v1(
            _unbound_plan(), _request(), source_identity={"git_commit": "ABC123"}
        )

    assert caught.value.diagnostic["failures"][0]["capability"] == "source_identity"


def test_execute_time_validation_accepts_unchanged_contract() -> None:
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request(), source_identity=SOURCE_IDENTITY
    )

    validated = validate_resolved_run_contract_v1(
        plan, source_identity=SOURCE_IDENTITY
    )

    assert validated == plan["run_contract"]


def test_execute_time_validation_rejects_source_change() -> None:
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request(), source_identity=SOURCE_IDENTITY
    )

    with pytest.raises(RunContractAdmissionError, match="current source identity"):
        validate_resolved_run_contract_v1(
            plan, source_identity={"git_commit": "f" * 40}
        )


def test_execute_time_validation_rejects_tampered_contract() -> None:
    plan = _unbound_plan()
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request(), source_identity=SOURCE_IDENTITY
    )
    plan["run_contract"]["capabilities"]["patch"]["configured"] = False  # type: ignore[index]

    with pytest.raises(RunContractAdmissionError, match="canonical inputs"):
        validate_resolved_run_contract_v1(
            plan, source_identity=SOURCE_IDENTITY
        )


def test_orchestration_rejects_tampered_contract_before_creating_run_root(
    tmp_path: Path,
) -> None:
    plan = _strict_plan()
    plan["run_contract"]["capabilities"]["patch"]["configured"] = False  # type: ignore[index]
    run_root = tmp_path / "must-not-exist"

    with pytest.raises(RunContractAdmissionError, match="canonical inputs"):
        execute_campaign(
            plan,
            run_root=run_root,
            provider_registry=default_execution_provider_registry(),
        )

    assert not run_root.exists()


def test_execute_time_validation_preserves_legacy_plan() -> None:
    assert validate_resolved_run_contract_v1(_unbound_plan()) is None


def test_zero_births_are_not_yet_exercised_not_missing() -> None:
    plan = _strict_plan()
    report = build_activation_report_v1(
        plan,
        [
            {
                "status": "completed",
                "metrics_rows": [
                    {
                        "metrics": {"population.successful_births_cumulative": 0},
                        "generation_metrics": [
                            {
                                "generation": 0,
                                "ordinary_birth_count": 0,
                                "wdl": {"lethal_exposure_count": 0},
                            }
                        ],
                    }
                ],
            }
        ],
        campaign_status="completed",
    )
    assert report is not None
    for capability in (
        "genome_v2_ordinary_birth",
        "genome_v2_mutation",
        "recombination_profile",
    ):
        assert report["capabilities"][capability]["status"] == "NOT_YET_EXERCISED"


def test_observable_opportunity_without_execution_is_expected_missing() -> None:
    report = build_activation_report_v1(
        _strict_plan(),
        [{"status": "running", "metrics_rows": []}],
        campaign_status="running",
        runtime_evidence={
            "genome_v2_mutation": {
                "eligible_events": 27,
                "execution_count": 0,
            }
        },
    )
    assert report is not None
    mutation = report["capabilities"]["genome_v2_mutation"]
    assert mutation["status"] == "EXPECTED_BUT_MISSING"
    assert mutation["evidence"]["eligible_events"] == 27


def test_optional_unconfigured_capability_is_not_applicable_at_runtime() -> None:
    plan = _unbound_plan()
    for cell in plan["cells"]:  # type: ignore[index]
        cell["parameters"]["genome_upgrade_v2.mutation_rate_q10"] = 0
        cell["parameters"]["genome_upgrade_v2.mutation_step_q10"] = 0
    plan["run_contract"] = resolve_run_contract_v1(
        plan, _request("optional"), source_identity=SOURCE_IDENTITY
    )

    report = build_activation_report_v1(
        plan, [], campaign_status="pending"
    )

    assert report is not None
    assert report["capabilities"]["genome_v2_mutation"]["status"] == (
        "NOT_APPLICABLE"
    )


def test_explicit_execution_counter_is_exercised() -> None:
    report = build_activation_report_v1(
        _strict_plan(),
        [{"status": "running", "metrics_rows": []}],
        campaign_status="running",
        runtime_evidence={
            "genome_v2_mutation": {
                "eligible_events": 1,
                "execution_count": 1,
            }
        },
    )
    assert report is not None
    mutation = report["capabilities"]["genome_v2_mutation"]
    assert mutation["status"] == "EXERCISED"
    assert mutation["initialized"] is True


def test_completed_runtime_requires_wdl_and_eed_projections() -> None:
    report = build_activation_report_v1(
        _strict_plan(),
        [
            {
                "status": "completed",
                "metrics_rows": [
                    {
                        "metrics": {"population.successful_births_cumulative": 0},
                        "generation_metrics": [
                            {"generation": 0, "ordinary_birth_count": 0}
                        ],
                    }
                ],
            }
        ],
        campaign_status="completed",
    )
    assert report is not None
    assert report["capabilities"]["wdl_delayed_hazard"]["status"] == (
        "EXPECTED_BUT_MISSING"
    )
    assert report["capabilities"]["eed"]["status"] == "EXPECTED_BUT_MISSING"


def test_contract_and_activation_sidecars_materialize(tmp_path: Path) -> None:
    plan = _strict_plan()
    batch = {"batch_id": "batch-0000", "cell_ids": ["cell-a"]}
    atomic_write_json(
        tmp_path / "batches/batch-0000/manifest.json",
        {"status": "pending", "metrics_rows": []},
    )

    _write_run_contract_artifacts_v1(tmp_path, plan, [batch], campaign_status="pending")

    resolved = json.loads((tmp_path / "resolved_run_contract.json").read_text())
    activation = json.loads((tmp_path / "activation_report.json").read_text())
    assert resolved["contract_hash"] == plan["run_contract"]["contract_hash"]
    assert activation["contract_hash"] == resolved["contract_hash"]


def test_eed_reporting_boundary_has_no_simulation_import_or_write_api() -> None:
    source = (ROOT / "engine/campaigns/run_contract_v1.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert not any(
        name == "simulation"
        or name.startswith("simulation.")
        or name == "engine.simulation"
        or name.startswith("engine.simulation.")
        for name in imports
    )
    assert "reporting_influence" not in source
    assert "selection" not in {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def test_activation_projection_is_deterministic_and_does_not_mutate_inputs() -> None:
    plan = _strict_plan()
    manifests = [
        {
            "status": "completed",
            "metrics_rows": [
                {
                    "metrics": {"eed.resource_pressure_mean": 0.25},
                    "generation_metrics": [
                        {"generation": 0, "wdl": {"lethal_exposure_count": 0}}
                    ],
                }
            ],
        }
    ]
    original_plan = copy.deepcopy(plan)
    original_manifests = copy.deepcopy(manifests)

    left = build_activation_report_v1(
        plan, manifests, campaign_status="completed"
    )
    right = build_activation_report_v1(
        plan, manifests, campaign_status="completed"
    )

    assert left == right
    assert plan == original_plan
    assert manifests == original_manifests
