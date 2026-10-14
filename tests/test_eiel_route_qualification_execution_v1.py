from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.tick_native_control import TickNativeControlV1
from engine.campaigns.qualification import (
    EXECUTION_ROUTE_OBLIGATIONS_V1,
    EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
    execution_route_qualification_sidecar_path_v1,
    execution_route_science_qualification_projection_path_v1,
    execution_route_qualification_status_v1,
    read_execution_route_qualification_sidecar_v1,
)
from engine.campaigns.long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    SCIENCE_ROUTE_REQUIRED_OBLIGATIONS_V1,
    SCIENCE_ROUTE_QUALIFIED,
    LongHorizonQualificationError,
    _canonical_production_wrc_envelope_v1,
    _validate_canonical_production_wrc_envelope_v1,
    _validate_canonical_wrc_parameter_contracts_v1,
    materialize_canonical_full_production_route_attestation_v1,
    validate_long_horizon_evidence_v1,
)
from engine.campaigns.route_qualification import (
    RouteQualificationError,
    _acs_persistence_route_v1,
    _batch_route_obligation_evidence_v1,
    _execute_acs_focused_route_probe_v1,
    _tick_native_evidence_row_v1,
    _validate_acs_async_route_obligations_v1,
    _validate_acs_focused_route_evidence_v1,
    _validate_b136_5000tick_qualification_pair_v1,
    _validate_compact_science_async_route_obligations_v1,
    _validate_semantic_route_obligations_v1,
    _validate_tick_native_route_obligations_v1,
    _validated_route_qualification_inheritance_v1,
    bounded_route_qualification_preflight_v1,
    derive_bounded_route_campaigns_v1,
    execute_route_qualification_v1,
    promote_execution_route_science_evidence_v1,
)
from engine.qualification_plans.v1 import (
    EXECUTION_ROUTE_OBLIGATIONS,
    generate_execution_route_canary_plan,
)


def test_q6_pair_revalidation_stays_strict_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.campaigns import long_horizon_evidence_materialization as materializer
    from engine.campaigns.long_horizon_evidence_materialization import (
        LongHorizonEvidenceMaterializationError,
    )

    bindings = {
        arm: {
            field: "x"
            for field in (
                "run_root",
                "request_sha256",
                "result_sha256",
                "plan_sha256",
                "run_passport_sha256",
                "lifecycle_contract_sha256",
                "disposition_manifest_sha256",
                "storage_disposition_receipt_sha256",
                "campaign_spec_sha256",
                "execution_route_sha256",
                "qualification_receipt_sha256",
                "invocation_identity_sha256",
            )
        }
        for arm in ("uninterrupted", "restored")
    }
    bindings["uninterrupted"]["run_root"] = "/tmp/gate27-uninterrupted"
    bindings["restored"]["run_root"] = "/tmp/gate27-restored"
    bindings["restored"]["process_instance_ids"] = ["producer", "restore"]
    evidence = {
        "materialization_profile": "eiel_b136_5000tick_qualification_pair_v1",
        "evidence_identity": {
            "route_scope": "long_horizon_runability_route_v1",
            "geometry": {
                "batch_cells": 136,
                "world_width": 32,
                "population_capacity": 256,
                "controller_capacity": 512,
            },
        },
        "ticks_requested": 5000,
        "ticks_committed": 5000,
        "checkpoint_restore": {
            "checkpoint_tick": 2500,
            "fresh_owner_restore_tick": 2500,
        },
        "paired_run_bindings": bindings,
        "paired_run_bindings_sha256": canonical_sha256(bindings),
    }

    def reject_canonical_revalidation(_evidence):
        raise LongHorizonEvidenceMaterializationError("old checkout receipt")

    monkeypatch.setattr(
        materializer,
        "validate_b136_5000tick_qualification_pair_evidence_v1",
        reject_canonical_revalidation,
    )
    with pytest.raises(RouteQualificationError, match="canonical revalidation failed"):
        _validate_b136_5000tick_qualification_pair_v1(
            evidence,
            qualification_dimensions=[LONG_HORIZON_RUNABILITY_QUALIFIED],
        )
    _validate_b136_5000tick_qualification_pair_v1(
        evidence,
        qualification_dimensions=[LONG_HORIZON_RUNABILITY_QUALIFIED],
        canonical_revalidation=False,
    )


def test_gate27_rejects_noncanonical_wrc_parameter_contracts() -> None:
    contract = {
        "name": "test.parameter",
        "type": "int",
        "per_cell": True,
        "qualification_affecting": False,
        "control_role": "independent_primitive",
        "variability_route_scopes": ["patch"],
        "variability_semantics": "experiment_variable",
        "activation_contract": {},
        "value_contract": {"kind": "int"},
    }
    assert _validate_canonical_wrc_parameter_contracts_v1(
        [contract], name="test"
    ) == [contract]
    missing = dict(contract)
    missing.pop("value_contract")
    with pytest.raises(LongHorizonQualificationError, match="field set"):
        _validate_canonical_wrc_parameter_contracts_v1([missing], name="test")
    duplicate = [dict(contract), dict(contract)]
    with pytest.raises(LongHorizonQualificationError, match="duplicate"):
        _validate_canonical_wrc_parameter_contracts_v1(duplicate, name="test")


def test_gate27_rejects_canonical_looking_but_wrong_wrc_envelope() -> None:
    canonical = _canonical_production_wrc_envelope_v1(
        [{"name": "env_cfg.eed_resource_a_capacity"}]
    )
    parameter_contracts = canonical["parameter_contracts"]
    wrong_name = deepcopy(canonical)
    wrong_name["parameter_contracts"][0]["name"] = (
        "env_cfg.eed_resource_b_capacity"
    )
    with pytest.raises(LongHorizonQualificationError, match="exact current"):
        _validate_canonical_production_wrc_envelope_v1(
            wrong_name,
            parameter_contracts=parameter_contracts,
            name="test.variability_envelope",
        )
    wrong_value = deepcopy(canonical)
    wrong_value["parameter_contracts"][0]["value_contract"]["minimum"] = -1.0
    with pytest.raises(LongHorizonQualificationError, match="exact current"):
        _validate_canonical_production_wrc_envelope_v1(
            wrong_value,
            parameter_contracts=parameter_contracts,
            name="test.variability_envelope",
        )


def _contract() -> ExecutionRouteContractV1:
    return ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {
                "component_id": "controller",
                "contract_version": "fixed-mlp-v1",
            },
        ),
        required_obligations=(
            "generation_boundary",
            "disk_checkpoint_reload",
            "reporting_parity",
            "multicell_execution",
            "alternate_batch_partition",
        ),
    )


def _turnover_contract() -> ExecutionRouteContractV1:
    return ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {
                "component_id": "controller",
                "contract_version": "fixed-mlp-v1",
            },
            {
                "component_id": "production_engine_lifecycle_control",
                "contract_version": "turnover-only-v1",
            },
        ),
        required_obligations=(
            "generation_boundary",
            "ordinary_birth_disabled",
            "turnover_refill",
        ),
    )


def _async_contract() -> ExecutionRouteContractV1:
    return ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {
                "component_id": "controller",
                "contract_version": "fixed-mlp-v1",
            },
            {
                "component_id": "acs_checkpoint_persistence",
                "contract_version": "eiel_acs_checkpoint_staging_v1",
                "immutable_snapshot_slots": 2,
            },
        ),
        required_obligations=(
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
            "durable_frontier_publication",
        ),
    )


def _compact_science_async_contract() -> ExecutionRouteContractV1:
    return ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
            {
                "component_id": (
                    "cuda_full_evolution_v2_compact_science_async_publication"
                ),
                "workers": 2,
                "pending_slots": 2,
                "worker_boundary": "immutable_tensor_free_host_capsule_v1",
                "worker_accepts_cuda_objects": False,
                "worker_accepts_runtime_owner": False,
                "fifo_publication": True,
                "checkpoint_frontier_requires_durable_prefix": True,
            },
        ),
        required_obligations=(
            "async_compact_science_backpressure",
            "async_compact_science_failure_latch",
            "compact_science_host_only_capture",
            "compact_science_checkpoint_prefix_drain",
        ),
    )


def test_compact_science_async_obligations_are_canonical_but_not_generic() -> None:
    obligations = set(_compact_science_async_contract().required_obligations)

    assert obligations <= EXECUTION_ROUTE_OBLIGATIONS
    assert obligations <= EXECUTION_ROUTE_OBLIGATIONS_V1
    plan = generate_execution_route_canary_plan(
        {
            **_compact_science_async_contract().payload(),
            "execution_route_sha256": "f" * 64,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1, 2],
            },
        }
    )
    assert plan["missing_obligations"] == sorted(obligations)


def _canary() -> dict[str, object]:
    contract = _contract()
    return generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1, 2],
            },
        }
    )


def _turnover_canary() -> dict[str, object]:
    contract = _turnover_contract()
    return generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1],
            },
        }
    )


def _campaign_plan() -> dict[str, object]:
    return {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "source-campaign",
        "resolved_profile": "fixed-profile",
        "qualification_key": {
            "artifact_hashes": {"cuda_extension": "a" * 64},
        },
        "execution": {
            "generations": 99,
            "ticks_per_generation": 80,
            "checkpoints": [99],
            "checkpoint_storage_mode": "disk",
            "technical_gate": {
                "required": True,
                "cell_ids": ["source-cell"],
                "batch_ids": ["source-batch"],
            },
        },
        "cells": [
            {
                "ordinal": 0,
                "cell_id": "source-cell",
                "branch_id": "source-branch",
                "rng_lineage_id": "source-rng",
                "dimensions": {"seed": 7},
                "parameters": {"patch.capacity": 256},
                "seeds": {"task": 7},
            }
        ],
        "physical_batches": [
            {
                "ordinal": 0,
                "batch_id": "source-batch",
                "cell_ids": ["source-cell"],
            }
        ],
        "comparisons": {"branches": []},
        "output": {"runtime_root": "/tmp/not-used"},
        "plan_sha256": "source-plan",
    }


def _turnover_campaign_plan() -> dict[str, object]:
    campaign = _campaign_plan()
    parameters = dict(campaign["cells"][0]["parameters"])  # type: ignore[index]
    parameters.update(
        {
            "patch.grid_size": 32,
            "patch.initial_population": 256,
            "patch.capacity": 256,
            "production_lifecycle.profile": (
                "patch_fixed_mlp_turnover_only_w32p256c256_v1"
            ),
            "production_lifecycle.turnover_enabled": True,
            "production_lifecycle.ordinary_births_enabled": False,
            "production_lifecycle.lifecycle_mode": (
                "legacy_synchronized_truncation_v1"
            ),
            "production_lifecycle.life_history_profile": "legacy_inert_v1",
            "production_lifecycle.turnover_cull_fraction": 0.5,
            "production_lifecycle.turnover_parent_fraction": 1.0,
        }
    )
    campaign["cells"][0]["parameters"] = parameters  # type: ignore[index]
    return campaign


def _preflight(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "eiel_campaign_preflight_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "ready_for_execution": True,
        "reasons": [],
        "execution_authority_reasons": [],
    }


class _BoundedPreflightProvider:
    profile_id = "fixed-profile"


class _CaseRouteProvider:
    profile_id = "fixed-profile"
    provider_id = "case-route-provider"

    def execution_route_contract(self, plan: dict[str, object]) -> ExecutionRouteContractV1:
        case_id = plan.get("route_qualification_case_id")
        components = _contract().components
        if isinstance(case_id, str):
            components = (
                *components,
                {"component_id": "bounded_case_extent", "case_id": case_id},
            )
        return ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id="fixed-provider",
            profile_id="fixed-profile",
            components=components,
            required_obligations=_contract().required_obligations,
        )


class _CaseRouteRegistry:
    def __init__(self) -> None:
        self.provider = _CaseRouteProvider()

    def get(self, profile_id: str) -> _CaseRouteProvider | None:
        return self.provider if profile_id == self.provider.profile_id else None


def _ordinary_preflight_receipt(
    *authority_reason_codes: str,
) -> dict[str, object]:
    return {
        "schema_version": "eiel_campaign_preflight_v1",
        "ok": True,
        "ready_for_execution": False,
        "reasons": [],
        "execution_authority_reasons": [
            {
                "code": code,
                "path": "test",
                "message": code,
            }
            for code in authority_reason_codes
        ],
        "execution_route": {"authority_state": "legacy_unbound"},
    }


def test_bounded_preflight_tolerates_only_unqualified_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = _ordinary_preflight_receipt(
        "profile_not_execution_qualified",
        "execution_route_qualification_missing",
    )
    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda plan, *, provider_registry, profile=None: {
            **ordinary,
            "execution_authority_reasons": [
                dict(row) for row in ordinary["execution_authority_reasons"]
            ],
            "execution_route": dict(ordinary["execution_route"]),
        },
    )

    receipt = bounded_route_qualification_preflight_v1(
        _BoundedPreflightProvider(), _campaign_plan()
    )

    assert ordinary["ready_for_execution"] is False
    assert "qualification_mode" not in ordinary
    assert receipt["qualification_mode"] is True
    assert receipt["ready_for_route_qualification"] is True
    assert receipt["ready_for_execution"] is False
    assert receipt["execution_route"]["authority_state"] == "qualification_only"


def test_bounded_preflight_refuses_unrelated_authority_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda plan, *, provider_registry, profile=None: _ordinary_preflight_receipt(
            "profile_not_execution_qualified",
            "execution_route_qualification_missing",
            "insufficient_free_vram",
        ),
    )

    receipt = bounded_route_qualification_preflight_v1(
        _BoundedPreflightProvider(), _campaign_plan()
    )

    assert receipt["ready_for_route_qualification"] is False
    assert receipt["ready_for_execution"] is False


def test_bounded_preflight_preserves_qualified_profile_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda plan, *, provider_registry, profile=None: _ordinary_preflight_receipt(
            "execution_route_qualification_missing"
        ),
    )

    receipt = bounded_route_qualification_preflight_v1(
        _BoundedPreflightProvider(), _campaign_plan()
    )

    assert receipt["ready_for_route_qualification"] is True
    assert receipt["ready_for_execution"] is False


def test_bounded_preflight_rebinds_a_nonexact_inherited_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = _ordinary_preflight_receipt("execution_route_qualification_missing")
    ordinary["execution_route"] = {
        "authority_state": "route_qualified",
        "route_hash_exact": False,
    }
    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda plan, *, provider_registry, profile=None: {
            **ordinary,
            "execution_authority_reasons": [
                dict(row) for row in ordinary["execution_authority_reasons"]
            ],
            "execution_route": dict(ordinary["execution_route"]),
        },
    )

    receipt = bounded_route_qualification_preflight_v1(
        _BoundedPreflightProvider(), _campaign_plan()
    )

    assert receipt["ready_for_route_qualification"] is True
    assert receipt["ready_for_execution"] is False
    assert receipt["execution_route"]["authority_state"] == "qualification_only"


def test_bounded_preflight_allows_vram_projection_only_with_live_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = _ordinary_preflight_receipt(
        "execution_route_qualification_missing",
        "provider_vram_bytes_upper_bound_unavailable",
    )
    ordinary["resources"] = {
        "cuda": {
            "free_covers_admission_bound_and_headroom": True,
            "live_admission_bound_source": "provider_conservative_projection",
            "live_admission_bound_bytes": 1024,
        }
    }
    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda plan, *, provider_registry, profile=None: ordinary,
    )

    receipt = bounded_route_qualification_preflight_v1(
        _BoundedPreflightProvider(), _campaign_plan()
    )

    assert receipt["ready_for_route_qualification"] is True
    assert receipt["ready_for_execution"] is False


def _async_route_case(*, blocked_time_ns: int = 11) -> dict[str, object]:
    case_id = "fresh-repeat-a"
    batch_id = "route-batch-0000"
    execution_route_sha256 = "f" * 64
    return {
        "case_id": case_id,
        "acs_async_evidence": [
            {
                "case_id": case_id,
                "batch_id": batch_id,
                "execution_route_sha256": execution_route_sha256,
                "telemetry": {
                    "schema_version": "eiel_acs_checkpoint_staging_v1",
                    "slots": 2,
                    "queue_occupancy_high_water": 2,
                    "producer_blocked_time_ns": blocked_time_ns,
                    "compute_frontier": 2,
                    "durable_checkpoint_frontier": 2,
                    "frontier_lag": 0,
                    "failure_latched": False,
                    "closed": True,
                    "writer_alive": False,
                },
            }
        ],
    }


def _detached_async_route_case() -> dict[str, object]:
    case = _async_route_case()
    telemetry = case["acs_async_evidence"][0]["telemetry"]  # type: ignore[index]
    telemetry.update(  # type: ignore[union-attr]
        {
            "schema_version": "eiel_acs_native_detached_slots_v1",
            "capture_mode": "detached_file_backed_native_v1",
            "writer_count": 1,
            "worker_accepts_cuda_objects": False,
            "worker_accepts_callbacks": False,
            "capture_pool_count": 1,
            "capture_pool_model": "one_shared_two_slab_pool_not_per_slot",
        }
    )
    return case


def test_async_route_leaves_behavioral_obligations_missing() -> None:
    covered = {
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
        "durable_frontier_publication",
    }
    case = _async_route_case()

    result = _validate_acs_async_route_obligations_v1(
        covered=covered,
        exact_cases=(case,),
        execution_route_sha256="f" * 64,
        expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
        immutable_snapshot_slots=2,
    )

    assert result[0]["exact"] is False
    assert result[0]["covered_obligations"] == ["durable_frontier_publication"]
    assert result[0]["missing_obligations"] == [
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    ]
    assert result[0]["behavioral_evidence_authority"] == (
        "ordinary_success_telemetry_is_not_fault_evidence_v1"
    )


def test_async_route_covers_host_only_capture_from_detached_telemetry() -> None:
    result = _validate_acs_async_route_obligations_v1(
        covered={"acs2_host_only_capture"},
        exact_cases=(_detached_async_route_case(),),
        execution_route_sha256="f" * 64,
        expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
        immutable_snapshot_slots=2,
    )

    assert result[0]["covered_obligations"] == ["acs2_host_only_capture"]
    assert result[0]["missing_obligations"] == []
    assert result[0]["batches"][0]["worker_accepts_cuda_objects"] is False


def _compact_science_async_route_case() -> dict[str, object]:
    case_id = "fresh-repeat-a"
    batch_id = "route-batch-0000"
    execution_route_sha256 = "f" * 64
    return {
        "case_id": case_id,
        "compact_science_async_evidence": [
            {
                "case_id": case_id,
                "batch_id": batch_id,
                "execution_route_sha256": execution_route_sha256,
                "terminal_next_absolute_tick": 128,
                "telemetry": {
                    "schema_version": "eiel_compact_science_async_publication_v1",
                    "workers": 2,
                    "pending_slots": 2,
                    "compute_frontier": 128,
                    "durable_frontier": 128,
                    "pending_jobs": 0,
                    "occupancy_high_water": 2,
                    "captured_host_bytes": 0,
                    "submitted_jobs": 2,
                    "completed_jobs": 2,
                    "failed_jobs": 0,
                    "queued_cuda_references": 0,
                    "fifo_publication": True,
                    "closed": True,
                    "aborted": False,
                },
            }
        ],
    }


def test_compact_science_async_route_covers_only_terminal_host_prefix_facts() -> None:
    contract = _compact_science_async_contract()
    result = _validate_compact_science_async_route_obligations_v1(
        covered=set(contract.required_obligations),
        exact_cases=(_compact_science_async_route_case(),),
        route_contract=contract,
        execution_route_sha256="f" * 64,
        expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
    )

    assert result[0]["covered_obligations"] == [
        "compact_science_checkpoint_prefix_drain",
        "compact_science_host_only_capture",
    ]
    assert result[0]["missing_obligations"] == [
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
    ]
    assert result[0]["exact"] is False
    assert result[0]["behavioral_evidence_authority"] == (
        "ordinary_success_telemetry_is_not_backpressure_or_failure_evidence_v1"
    )


def test_compact_science_async_route_rejects_nonterminal_fifo_or_tensor_reference() -> None:
    contract = _compact_science_async_contract()
    case = _compact_science_async_route_case()
    telemetry = case["compact_science_async_evidence"][0]["telemetry"]  # type: ignore[index]
    telemetry["queued_cuda_references"] = 1  # type: ignore[index]

    with pytest.raises(RouteQualificationError, match="tensor-free"):
        _validate_compact_science_async_route_obligations_v1(
            covered={"compact_science_host_only_capture"},
            exact_cases=(case,),
            route_contract=contract,
            execution_route_sha256="f" * 64,
            expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
        )


def test_compact_science_async_route_requires_exact_case_batch_coverage() -> None:
    contract = _compact_science_async_contract()
    with pytest.raises(RouteQualificationError, match="exactly cover"):
        _validate_compact_science_async_route_obligations_v1(
            covered={"compact_science_checkpoint_prefix_drain"},
            exact_cases=(_compact_science_async_route_case(),),
            route_contract=contract,
            execution_route_sha256="f" * 64,
            expected_case_batches={
                ("fresh-repeat-a", "route-batch-0000"),
                ("fresh-repeat-a", "route-batch-0001"),
            },
        )


def test_qualification_status_never_accepts_compact_fault_names_from_success_only() -> None:
    contract = _compact_science_async_contract()
    receipt = {
        "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
        "execution_route_sha256": contract.execution_route_sha256,
        "provider_id": contract.provider_id,
        "profile_id": contract.profile_id,
        "authority_state": "route_qualified",
        "covered_obligations": sorted(contract.required_obligations),
        "missing_obligations": [],
        "qualification_dimensions": [],
        "route_obligation_results": [],
    }

    status = execution_route_qualification_status_v1(contract, receipt)

    assert status["authority_state"] == "qualification_only"
    assert status["covered_obligations"] == [
        "compact_science_checkpoint_prefix_drain",
        "compact_science_host_only_capture",
    ]
    assert status["missing_obligations"] == [
        "async_compact_science_backpressure",
        "async_compact_science_failure_latch",
    ]
    assert any(
        row["code"] == "execution_route_focused_compact_science_evidence_missing"
        for row in status["reasons"]
    )


def test_async_route_rejects_legacy_capture_for_host_only_obligation() -> None:
    with pytest.raises(RouteQualificationError, match="host-only capture"):
        _validate_acs_async_route_obligations_v1(
            covered={"acs2_host_only_capture"},
            exact_cases=(_async_route_case(),),
            execution_route_sha256="f" * 64,
            expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
            immutable_snapshot_slots=2,
        )


def test_async_route_accepts_the_enclosing_case_route_hash() -> None:
    case = _async_route_case()
    case_route = "a" * 64
    case["execution_route_sha256"] = case_route
    case["acs_async_evidence"][0]["execution_route_sha256"] = case_route  # type: ignore[index]

    result = _validate_acs_async_route_obligations_v1(
        covered={"durable_frontier_publication"},
        exact_cases=(case,),
        execution_route_sha256="f" * 64,
        expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
        immutable_snapshot_slots=2,
    )

    assert result[0]["batches"][0]["execution_route_sha256"] == case_route  # type: ignore[index]


@pytest.mark.parametrize("writer_alive", [None, True])
def test_async_route_requires_writer_settled_at_close(
    writer_alive: bool | None,
) -> None:
    case = _async_route_case()
    telemetry = case["acs_async_evidence"][0]["telemetry"]  # type: ignore[index]
    telemetry["writer_alive"] = writer_alive  # type: ignore[index]

    with pytest.raises(RouteQualificationError, match="writer-settled"):
        _validate_acs_async_route_obligations_v1(
            covered={"durable_frontier_publication"},
            exact_cases=(case,),
            execution_route_sha256="f" * 64,
            expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
            immutable_snapshot_slots=2,
        )


def test_zero_blocked_time_does_not_qualify_async_backpressure() -> None:
    case = _async_route_case(blocked_time_ns=0)

    result = _validate_acs_async_route_obligations_v1(
        covered={
            "async_checkpoint_backpressure",
            "durable_frontier_publication",
        },
        exact_cases=(case,),
        execution_route_sha256="f" * 64,
        expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
        immutable_snapshot_slots=2,
    )

    assert result[0]["covered_obligations"] == ["durable_frontier_publication"]
    assert result[0]["missing_obligations"] == ["async_checkpoint_backpressure"]


def test_async_route_rejects_a_different_well_formed_route_hash() -> None:
    case = _async_route_case()

    with pytest.raises(RouteQualificationError, match="current execution route"):
        _validate_acs_async_route_obligations_v1(
            covered={"durable_frontier_publication"},
            exact_cases=(case,),
            execution_route_sha256="e" * 64,
            expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
            immutable_snapshot_slots=2,
        )


def test_async_route_rejects_duplicate_case_batch_rows() -> None:
    case = _async_route_case()
    case["acs_async_evidence"].append(  # type: ignore[union-attr]
        deepcopy(case["acs_async_evidence"][0])  # type: ignore[index]
    )

    with pytest.raises(RouteQualificationError, match="duplicate case/batch"):
        _validate_acs_async_route_obligations_v1(
            covered={"durable_frontier_publication"},
            exact_cases=(case,),
            execution_route_sha256="f" * 64,
            expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
            immutable_snapshot_slots=2,
        )


@pytest.mark.parametrize(
    "expected_case_batches",
    (
        {
            ("fresh-repeat-a", "route-batch-0000"),
            ("fresh-repeat-a", "route-batch-0001"),
        },
        {("fresh-repeat-a", "route-batch-unexpected")},
    ),
)
def test_async_route_requires_exact_expected_case_batch_coverage(
    expected_case_batches: set[tuple[str, str]],
) -> None:
    with pytest.raises(RouteQualificationError, match="exactly cover"):
        _validate_acs_async_route_obligations_v1(
            covered={"durable_frontier_publication"},
            exact_cases=(_async_route_case(),),
            execution_route_sha256="f" * 64,
            expected_case_batches=expected_case_batches,
            immutable_snapshot_slots=2,
        )


def test_async_route_rejects_telemetry_for_another_slot_contract() -> None:
    with pytest.raises(RouteQualificationError, match="writer-settled"):
        _validate_acs_async_route_obligations_v1(
            covered={"durable_frontier_publication"},
            exact_cases=(_async_route_case(),),
            execution_route_sha256="f" * 64,
            expected_case_batches={("fresh-repeat-a", "route-batch-0000")},
            immutable_snapshot_slots=1,
        )


@pytest.mark.parametrize("immutable_snapshot_slots", (1, 2))
def test_async_focused_probe_exercises_the_target_slot_contract(
    immutable_snapshot_slots: int,
) -> None:
    obligations = frozenset(
        {
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
        }
    )

    evidence = _execute_acs_focused_route_probe_v1(
        "f" * 64,
        obligations,
        immutable_snapshot_slots,
    )
    covered, validated = _validate_acs_focused_route_evidence_v1(
        execution_route_sha256="f" * 64,
        requested=set(obligations),
        evidence=evidence,
        immutable_snapshot_slots=immutable_snapshot_slots,
    )

    assert covered == set(obligations)
    assert validated["exact"] is True
    assert validated["target_immutable_snapshot_slots"] == immutable_snapshot_slots
    backpressure = validated["backpressure"]
    assert backpressure["all_slots_occupied_before_block"] is True
    assert backpressure["producer_thread_joined"] is True
    assert backpressure["blocked_submission_compute_frontier"] == (
        immutable_snapshot_slots + 1
    )
    assert backpressure["fifo_results"] == [
        "first",
        *[f"queued-{frontier}" for frontier in range(2, immutable_snapshot_slots + 1)],
        "blocked",
    ]
    assert backpressure["fifo_persist_order"] == backpressure["fifo_results"]
    assert backpressure["no_overwrite_or_drop"] is True
    assert backpressure["terminal_telemetry"]["queue_occupancy_high_water"] == (
        immutable_snapshot_slots
    )
    failure = validated["failure_latch"]
    assert failure["queued_token_count"] == immutable_snapshot_slots - 1
    assert failure["queued_callbacks_never_started"] is True
    assert failure["all_tokens_inaccessible_after_abort"] is True
    assert failure["terminal_telemetry"]["abandoned_snapshot_count"] == (
        immutable_snapshot_slots
    )
    assert failure["terminal_telemetry"]["slot_states"] == (
        ["FREE"] * immutable_snapshot_slots
    )


def test_async_focused_probe_evidence_is_exactly_route_bound() -> None:
    obligations = frozenset(
        {
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
        }
    )
    evidence = _execute_acs_focused_route_probe_v1("f" * 64, obligations, 2)

    covered, invalid = _validate_acs_focused_route_evidence_v1(
        execution_route_sha256="e" * 64,
        requested=set(obligations),
        evidence=evidence,
        immutable_snapshot_slots=2,
    )

    assert covered == set()
    assert invalid["exact"] is False
    assert invalid["missing_obligations"] == sorted(obligations)


def test_async_focused_probe_rejects_another_slot_contract() -> None:
    obligations = frozenset(
        {
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
        }
    )
    evidence = _execute_acs_focused_route_probe_v1("f" * 64, obligations, 1)

    covered, invalid = _validate_acs_focused_route_evidence_v1(
        execution_route_sha256="f" * 64,
        requested=set(obligations),
        evidence=evidence,
        immutable_snapshot_slots=2,
    )

    assert covered == set()
    assert invalid["exact"] is False
    assert invalid["missing_obligations"] == sorted(obligations)


def test_async_route_needs_the_focused_probe_to_publish_authority(
    tmp_path: Path,
) -> None:
    contract = _async_contract()
    canary = generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1, 2],
            },
        }
    )
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    sidecar = execution_route_qualification_sidecar_path_v1(
        mechanic_receipt,
        contract.execution_route_sha256,
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
                "authority_state": "route_qualified",
                "execution_route_sha256": contract.execution_route_sha256,
                "provider_id": contract.provider_id,
                "profile_id": contract.profile_id,
                "covered_obligations": list(contract.required_obligations),
            }
        ),
        encoding="utf-8",
    )
    stale = execution_route_qualification_status_v1(
        contract,
        json.loads(sidecar.read_text(encoding="utf-8")),
    )
    assert stale["authority_state"] == "qualification_only"
    assert stale["covered_obligations"] == ["durable_frontier_publication"]
    assert stale["missing_obligations"] == [
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    ]
    assert any(
        row["code"] == "execution_route_focused_acs_evidence_invalid"
        for row in stale["reasons"]
    )

    def ordinary_success(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        rows = []
        for batch in plan["physical_batches"]:  # type: ignore[union-attr]
            row = deepcopy(_async_route_case()["acs_async_evidence"][0])  # type: ignore[index]
            row["case_id"] = plan["route_qualification_case_id"]
            row["batch_id"] = batch["batch_id"]
            row["execution_route_sha256"] = plan["execution_route_sha256"]
            rows.append(row)
        result["acs_async_evidence"] = rows
        return result

    result = execute_route_qualification_v1(
        campaign_plan=_campaign_plan(),
        canary=canary,
        run_root=tmp_path / "run",
        route_contract=contract,
        qualification_receipt_path=mechanic_receipt,
        preflight_runner=_preflight,
        orchestration_runner=ordinary_success,
        acs_focused_probe_runner=lambda _route, _obligations, _slots: None,
    )

    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert result["qualification_succeeded"] is False
    assert published["authority_state"] == "qualification_only"
    assert published["covered_obligations"] == ["durable_frontier_publication"]
    assert published["missing_obligations"] == [
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    ]


    qualified = execute_route_qualification_v1(
        campaign_plan=_campaign_plan(),
        canary=canary,
        run_root=tmp_path / "probe-backed-run",
        route_contract=contract,
        qualification_receipt_path=mechanic_receipt,
        preflight_runner=_preflight,
        orchestration_runner=ordinary_success,
    )

    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert qualified["qualification_succeeded"] is True
    assert qualified["missing_obligations"] == []
    assert published["authority_state"] == "route_qualified"
    assert published["covered_obligations"] == sorted(contract.required_obligations)
    admitted = execution_route_qualification_status_v1(contract, published)
    assert admitted["authority_state"] == "route_qualified"
    assert admitted["missing_obligations"] == []
    focused = next(
        row
        for row in published["route_obligation_results"]
        if row.get("schema_version") == "eiel_acs_checkpoint_focused_route_evidence_v1"
    )
    assert focused["execution_route_sha256"] == contract.execution_route_sha256
    assert len(focused["implementation_source_sha256"]) == 64
    assert focused["backpressure"]["producer_blocked_time_ns"] > 0
    assert focused["backpressure"]["terminal_telemetry"]["writer_alive"] is False
    assert focused["failure_latch"]["post_latch_capture_rejected"] is True
    assert focused["failure_latch"]["abort_joined_writer"] is True
    assert focused["failure_latch"]["terminal_telemetry"]["writer_alive"] is False

    focused["target_immutable_snapshot_slots"] = 1
    wrong_slot = execution_route_qualification_status_v1(contract, published)
    assert wrong_slot["authority_state"] == "qualification_only"
    assert wrong_slot["covered_obligations"] == ["durable_frontier_publication"]
    focused["target_immutable_snapshot_slots"] = 2

    focused["backpressure"]["terminal_telemetry"]["slots"] = 1
    wrong_telemetry_slot = execution_route_qualification_status_v1(contract, published)
    assert wrong_telemetry_slot["authority_state"] == "qualification_only"
    assert wrong_telemetry_slot["covered_obligations"] == [
        "durable_frontier_publication"
    ]
    focused["backpressure"]["terminal_telemetry"]["slots"] = 2

    focused["backpressure"]["terminal_telemetry"]["slot_states"] = [
        "READY",
        "FREE",
    ]
    unsettled_slots = execution_route_qualification_status_v1(contract, published)
    assert unsettled_slots["authority_state"] == "qualification_only"
    assert unsettled_slots["covered_obligations"] == ["durable_frontier_publication"]
    focused["backpressure"]["terminal_telemetry"]["slot_states"] = [
        "FREE",
        "FREE",
    ]

    focused["implementation_source_sha256"] = "0" * 64
    source_stale = execution_route_qualification_status_v1(contract, published)
    assert source_stale["authority_state"] == "qualification_only"
    assert source_stale["covered_obligations"] == ["durable_frontier_publication"]
    assert source_stale["missing_obligations"] == [
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    ]


def test_acs_persistence_route_uses_native_contract_version() -> None:
    base = _async_contract()
    component = {
        **base.components[1],
        "contract_version": "acs_detached_file_backed_native_v1",
    }
    contract = ExecutionRouteContractV1(
        schema_version=base.schema_version,
        provider_id=base.provider_id,
        profile_id=base.profile_id,
        components=(base.components[0], component),
        required_obligations=base.required_obligations,
    )

    assert _acs_persistence_route_v1(contract) == (
        "acs_detached_file_backed_native_v1"
    )


def test_acs_persistence_route_rejects_conflicting_contract_names() -> None:
    base = _async_contract()
    conflicting_component = {
        **base.components[1],
        "contract_version": "acs_detached_file_backed_native_v1",
        "persistence_route": "eiel_acs_checkpoint_staging_v1",
    }
    contract = ExecutionRouteContractV1(
        schema_version=base.schema_version,
        provider_id=base.provider_id,
        profile_id=base.profile_id,
        components=(base.components[0], conflicting_component),
        required_obligations=base.required_obligations,
    )

    with pytest.raises(RouteQualificationError, match="disagree"):
        _acs_persistence_route_v1(contract)


def test_wait_storage_is_not_retried_or_parsed_as_batch_result(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def wait_storage(plan: dict[str, object], _run_root: Path) -> dict[str, object]:
        calls.append(str(plan["route_qualification_case_id"]))
        return {
            "status": "interrupted",
            "storage_wait": True,
            "storage_wait_receipt_path": str(tmp_path / "storage_wait_receipt.json"),
        }

    with pytest.raises(RouteQualificationError, match="WAIT_STORAGE"):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=tmp_path / "mechanic" / "qualification.json",
            preflight_runner=_preflight,
            orchestration_runner=wait_storage,
        )

    assert len(calls) == 1


def test_synchronous_route_ignores_irrelevant_stale_async_focused_claims() -> None:
    contract = _contract()
    stale_async_receipt = {
        "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
        "authority_state": "route_qualified",
        "execution_route_sha256": "f" * 64,
        "provider_id": contract.provider_id,
        "profile_id": contract.profile_id,
        "covered_obligations": [
            *contract.required_obligations,
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
        ],
        "route_obligation_results": [],
    }

    status = execution_route_qualification_status_v1(contract, stale_async_receipt)

    assert status["authority_state"] == "qualification_only"
    assert any(
        row["code"] == "execution_route_hash_mismatch" for row in status["reasons"]
    )
    assert not any(
        row["code"] == "execution_route_focused_acs_evidence_invalid"
        for row in status["reasons"]
    )


def test_bounded_preflight_refuses_vram_projection_without_live_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = _ordinary_preflight_receipt(
        "execution_route_qualification_missing",
        "provider_vram_bytes_upper_bound_unavailable",
    )
    ordinary["resources"] = {
        "cuda": {
            "free_covers_admission_bound_and_headroom": False,
            "live_admission_bound_source": "provider_conservative_projection",
            "live_admission_bound_bytes": 1024,
        }
    }
    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda plan, *, provider_registry, profile=None: ordinary,
    )

    receipt = bounded_route_qualification_preflight_v1(
        _BoundedPreflightProvider(), _campaign_plan()
    )

    assert receipt["ready_for_route_qualification"] is False
    assert receipt["ready_for_execution"] is False


def _exact_result(plan: dict[str, object], _run_root: Path) -> dict[str, object]:
    comparisons = plan["comparisons"]
    assert isinstance(comparisons, dict)
    return {
        "status": "completed",
        "exact": True,
        "execution_route_sha256": plan["execution_route_sha256"],
        "transition_sentinels_exact": True,
        "observed_transition_names": [
            "generation_boundary",
            "checkpoint_export_boundary",
        ],
        "terminal_cell_digests_by_ordinal": [
            f"terminal-{index}" for index, _cell in enumerate(plan["cells"])
        ],
        "terminal_cell_digests_by_cell_id": {
            str(cell["cell_id"]): f"terminal-{index}"
            for index, cell in enumerate(plan["cells"])
        },
        "exact_comparisons": [
            {
                "comparison_id": branch["branch_id"],
                "match": True,
                "first_difference": None,
                "execution_route_sha256": plan["execution_route_sha256"],
            }
            for branch in comparisons["branches"]
        ],
    }


def _turnover_exact_result(
    plan: dict[str, object], run_root: Path
) -> dict[str, object]:
    result = _exact_result(plan, run_root)
    result["observed_transition_names"] = ["generation_boundary"]
    result["route_obligation_evidence"] = {
        "schema_version": "eiel_route_obligation_evidence_v1",
        "cells": [
            {
                "schema_version": "eiel_route_obligation_cell_evidence_v1",
                "cell_id": cell["cell_id"],
                "source": "test",
                "profile_ordinary_births_enabled": False,
                "profile_turnover_enabled": True,
                "turnover_cull_fraction": 0.5,
                "turnover_parent_fraction": 1.0,
                "target_population": 256,
                "resident_capacity": 256,
                "generation_rows": [
                    {
                        "cell_id": cell["cell_id"],
                        "generation": 0,
                        "population_alive": 256,
                        "successful_births": 0,
                        "ordinary_birth_count": 0,
                        "refill_child_count": 0,
                    },
                    {
                        "cell_id": cell["cell_id"],
                        "generation": 1,
                        "population_alive": 256,
                        "successful_births": 128,
                        "ordinary_birth_count": 0,
                        "refill_child_count": 128,
                    },
                ],
            }
            for cell in plan["cells"]
        ],
    }
    return result


def test_derives_bounded_cases_and_does_not_reuse_campaign_extent() -> None:
    plans = derive_bounded_route_campaigns_v1(_campaign_plan(), _canary())

    assert {plan["execution"]["generations"] for plan in plans} == {2}
    assert {plan["execution"]["ticks_per_generation"] for plan in plans} == {80}
    assert all(len(plan["cells"]) == 2 for plan in plans)
    assert all("technical_gate" not in plan["execution"] for plan in plans)
    assert {plan["resource_estimate"]["physical_batch_cells"] for plan in plans} == {
        max(len(batch["cell_ids"]) for batch in plan["physical_batches"])
        for plan in plans
    }
    assert {plan["resource_estimate"]["logical_cells"] for plan in plans} == {2}
    assert all(
        plan["resource_estimate"]["logical_trajectories"]
        == len(plan["cells"]) + plan["resource_estimate"]["comparison_trajectories"]
        for plan in plans
    )
    reporting_by_case = {
        case["id"]: case["assignments"]["reporting_enabled"]
        for case in _canary()["qualification_plan"]["cases"]
    }
    assert {
        plan["route_qualification_case_id"]: plan["execution"]["reporting_enabled"]
        for plan in plans
    } == reporting_by_case
    alternate = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "partition-alternate"
    )
    default = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "partition-default"
    )
    assert [cell["cell_id"] for cell in alternate["cells"]] == [
        cell["cell_id"] for cell in default["cells"]
    ]
    fresh_a = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "fresh-repeat-a"
    )
    fresh_b = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "fresh-repeat-b"
    )
    for field in ("cell_id", "branch_id", "rng_lineage_id"):
        assert [cell[field] for cell in fresh_a["cells"]] == [
            cell[field] for cell in fresh_b["cells"]
        ]
    assert [len(batch["cell_ids"]) for batch in alternate["physical_batches"]] == [
        1,
        1,
    ]
    assert all(plan["plan_sha256"] != "source-plan" for plan in plans)


def test_partition_parity_preserves_distinct_source_cell_identity() -> None:
    campaign = deepcopy(_campaign_plan())
    first = deepcopy(campaign["cells"][0])  # type: ignore[index]
    first["parameters"] = {
        **first["parameters"],  # type: ignore[index]
        "genome_upgrade_v2.mutation_rate_q10": 20,
        "genome_upgrade_v2.mutation_step_q10": 64,
    }
    second = deepcopy(first)
    second.update(
        {
            "cell_id": "source-cell-2",
            "branch_id": "source-branch-2",
            "rng_lineage_id": "source-rng-2",
            "dimensions": {"seed": 8},
            "seeds": {"task": 8},
        }
    )
    campaign["cells"] = [first, second]  # type: ignore[index]

    plans = derive_bounded_route_campaigns_v1(campaign, _canary())
    partition_default = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "partition-default"
    )

    assert [cell["seeds"]["task"] for cell in partition_default["cells"]] == [
        7,
        8,
    ]


def test_route_qualification_rebinds_transformed_case_route_hashes(
    tmp_path: Path,
) -> None:
    registry = _CaseRouteRegistry()
    observed: dict[str, str] = {}

    def runner(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        case_id = str(plan["route_qualification_case_id"])
        observed[case_id] = str(plan["execution_route_sha256"])
        return _exact_result(plan, run_root)

    result = execute_route_qualification_v1(
        campaign_plan=_campaign_plan(),
        canary=_canary(),
        run_root=tmp_path / "run",
        route_contract=_contract(),
        qualification_receipt_path=tmp_path / "mechanic" / "qualification.json",
        preflight_runner=_preflight,
        provider_registry=registry,
        orchestration_runner=runner,
    )

    assert result["qualification_succeeded"] is True
    assert len(observed) == len(_canary()["qualification_plan"]["cases"])  # type: ignore[index]
    assert len(set(observed.values())) == len(observed)
    for plan in derive_bounded_route_campaigns_v1(_campaign_plan(), _canary()):
        case_id = str(plan["route_qualification_case_id"])
        expected = registry.provider.execution_route_contract(plan).execution_route_sha256
        assert observed[case_id] == expected


def test_async_canary_respects_the_native_minimum_segment_width() -> None:
    contract = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
            {
                "component_id": "acs_checkpoint_persistence",
                "contract_version": "eiel_acs_checkpoint_staging_v1",
                "immutable_snapshot_slots": 2,
            },
        ),
        required_obligations=(
            "generation_boundary",
            "disk_checkpoint_reload",
            "async_checkpoint_backpressure",
            "async_checkpoint_failure_latch",
            "durable_frontier_publication",
        ),
    )
    canary = generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 160,
                "admitted_physical_batch_sizes": [1],
                "resident_segment_generation_min": 3,
            },
        }
    )

    plans = derive_bounded_route_campaigns_v1(_campaign_plan(), canary)

    fresh = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "fresh-repeat-a"
    )
    continuation = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "disk-continuation"
    )
    assert fresh["execution"]["checkpoints"] == [3]
    assert continuation["execution"]["checkpoints"] == [3]


def test_derives_tick_native_partial_restore_case_without_generation_authority() -> (
    None
):
    obligations = (
        "non_generation_divisible_terminal_tick",
        "non_generation_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_generation_restore",
        "first_committed_tick_after_partial_restore",
    )
    contract = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
            {
                "component_id": "absolute_tick_time_contract",
                "time_authority": "absolute_tick_v1",
            },
        ),
        required_obligations=obligations,
    )
    canary = generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 160,
                "admitted_physical_batch_sizes": [1, 2],
            },
        },
        missing_obligations=list(obligations),
    )

    plans = derive_bounded_route_campaigns_v1(_campaign_plan(), canary)

    assert len(plans) == 1
    execution = plans[0]["execution"]
    assert execution["execution_route_identity"] == "tick_native_execution_route_v1"
    assert execution["total_ticks"] == 161
    assert execution["checkpoint_ticks"] == [17, 160, 161]
    assert execution["deliberate_restore_tick"] == 17
    assert "generations" not in execution
    assert "ticks_per_generation" not in execution
    assert "internal_lifecycle_schedule" not in execution
    assert execution["time_authority"] == "absolute_tick_v1"
    assert execution["tick_native_control"]["source_window_ticks"] == 160
    assert "refill_chunk_ticks" not in execution["tick_native_control"]


def test_tick_native_derived_plans_preserve_lossless_direct_control_aliases() -> None:
    contract = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
        ),
        required_obligations=("reporting_parity",),
    )
    canary = generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "time_authority": "absolute_tick_v1",
                "total_ticks": 161,
                "checkpoint_ticks": [17, 160, 161],
                "resident_owner_span_ticks": 1120,
                "native_dispatch_window_ticks": 1120,
                "event_window_capacity": 41804,
                "admitted_physical_batch_sizes": [1],
            },
        }
    )
    campaign = deepcopy(_campaign_plan())
    control = TickNativeControlV1(
        total_ticks=161,
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        checkpoint_ticks=(17, 160, 161),
        event_window_capacity=41804,
        source_window_ticks=160,
    ).payload()
    campaign["execution"] = {
        "execution_route_identity": "tick_native_execution_route_v1",
        "total_ticks": 161,
        "start_next_tick": 0,
        "checkpoint_ticks": [17, 160, 161],
        "checkpoint_storage_mode": "disk",
        "report_every_ticks": None,
        "tick_native_control": control,
        "acs_checkpoint_slots": 0,
    }

    plans = derive_bounded_route_campaigns_v1(campaign, canary)
    assert {
        plan["route_qualification_case_id"] for plan in plans
    } == {"reporting-disabled", "reporting-enabled"}

    direct_aliases = (
        "total_ticks",
        "resident_owner_span_ticks",
        "native_dispatch_window_ticks",
        "source_window_ticks",
        "event_window_capacity",
        "report_every_ticks",
        "science_sample_interval_ticks",
        "reducer_every_ticks",
        "archive_every_ticks",
        "checkpoint_every_ticks",
    )
    for plan in plans:
        execution = plan["execution"]
        derived_control = execution["tick_native_control"]
        assert isinstance(derived_control, dict)
        assert TickNativeControlV1.from_execution(execution).payload() == derived_control
        for name in direct_aliases:
            assert execution[name] == derived_control[name]
        assert execution["start_next_tick"] == derived_control["next_absolute_tick"]
        assert execution["checkpoint_ticks"] == derived_control["checkpoint_ticks"]
        assert derived_control["event_window_capacity"] == 41804

    reporting_enabled = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "reporting-enabled"
    )
    assert reporting_enabled["execution"]["report_every_ticks"] == 100
    assert (
        reporting_enabled["execution"]["tick_native_control"]["report_every_ticks"]
        == 100
    )


def test_tick_native_base_fresh_repeat_never_requires_generation_assignments() -> None:
    campaign = _campaign_plan()
    campaign["execution"] = {
        "execution_route_identity": "tick_native_execution_route_v1",
        "total_ticks": 5000,
        "start_next_tick": 0,
        "checkpoint_ticks": [5000],
        "internal_lifecycle_schedule": {
            "interval_ticks": 160,
            "authority": "internal_compatibility_only",
        },
        "digest_every_generation": False,
        "checkpoint_storage_mode": "disk",
        "acs_checkpoint_slots": 1,
        "technical_gate": {"required": False},
    }
    contract = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
        ),
        required_obligations=("cold_start", "first_generation", "first_tick"),
    )
    canary = generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "time_authority": "absolute_tick_v1",
                "total_ticks": 5_000,
                "checkpoint_ticks": [5_000],
                "resident_owner_span_ticks": 1_120,
                "native_dispatch_window_ticks": 1_120,
                "event_window_capacity": 65_536,
                "admitted_physical_batch_sizes": [1],
            },
        },
        missing_obligations=["cold_start", "first_generation", "first_tick"],
    )
    assert all(
        "generations" not in case["assignments"]
        and "ticks_per_generation" not in case["assignments"]
        for case in canary["qualification_plan"]["cases"]
    )
    plans = derive_bounded_route_campaigns_v1(campaign, canary)
    fresh = next(
        plan
        for plan in plans
        if plan["route_qualification_case_id"] == "fresh-repeat-a"
    )
    execution = fresh["execution"]
    assert execution["execution_route_identity"] == "tick_native_execution_route_v1"
    assert "generations" not in execution
    assert "ticks_per_generation" not in execution
    assert "internal_lifecycle_schedule" not in execution
    assert execution["total_ticks"] == 161
    assert execution["checkpoint_ticks"] == [17, 160, 161]


def test_tick_native_delta_evidence_requires_exact_cursor_and_schedule() -> None:
    obligations = {
        "non_lattice_divisible_terminal_tick",
        "non_lattice_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_tick_restore",
        "first_committed_tick_after_partial_restore",
        "first_post_restore_refill",
    }
    source_digest = "1" * 64
    terminal_digest = "2" * 64
    evidence = {
        "case_id": "tick-native-partial-restore",
        "observed_transition_names": [
            "first_policy_cache_refill_after_restore"
        ],
        "terminal_cell_digests_by_cell_id": {"cell-a": terminal_digest},
        "tick_native_evidence": [
            {
                "batch_id": "batch-0000",
                "requested_terminal_tick": 161,
                "requested_checkpoint_ticks": [17, 160, 161],
                "deliberate_restore_tick": 17,
                "actual_terminal_tick": 161,
                "actual_checkpoint_ticks": [17, 160, 161],
                "fresh_owner_restore_receipt": {
                    "schema_version": (
                        "eiel_patch_fixed_fresh_owner_restore_receipt_v1"
                    ),
                    "time_authority": "absolute_tick_v1",
                    "restore_next_tick": 17,
                    "first_executed_absolute_tick": 17,
                    "first_committed_next_tick": 18,
                    "off_by_one_check": "PASS",
                    "source_checkpoint_absolute_tick": 17,
                    "source_checkpoint_next_absolute_tick": 17,
                    "projection_bundles_reloaded": True,
                    "cell_ids": ["cell-a"],
                    "parent_checkpoint_projection_digest_sha256s": {
                        "cell-a": source_digest
                    },
                    "source_checkpoint_cell_digests": {
                        "cell-a": source_digest
                    },
                    "restored_terminal_projection_digest_sha256s": {
                        "cell-a": terminal_digest
                    },
                    "restored_terminal_cell_digests": {
                        "cell-a": terminal_digest
                    },
                    "reload_receipts": [
                        {
                            "cell_id": "cell-a",
                            "source_projection_sha256": source_digest,
                            "loaded_projection_sha256": source_digest,
                        }
                    ],
                },
            }
        ],
    }

    result = _validate_tick_native_route_obligations_v1(
        covered=obligations,
        exact_cases=(evidence,),
    )

    assert result[0]["exact"] is True
    broken = deepcopy(evidence)
    broken["tick_native_evidence"][0]["actual_terminal_tick"] = 160
    with pytest.raises(RouteQualificationError, match="is not exact"):
        _validate_tick_native_route_obligations_v1(
            covered=obligations,
            exact_cases=(broken,),
        )

    broken_restore = deepcopy(evidence)
    broken_restore["tick_native_evidence"][0]["fresh_owner_restore_receipt"][
        "reload_receipts"
    ][0]["loaded_projection_sha256"] = "3" * 64
    with pytest.raises(RouteQualificationError, match="policy-cache restore"):
        _validate_tick_native_route_obligations_v1(
            covered=obligations,
            exact_cases=(broken_restore,),
        )

    broken_terminal = deepcopy(evidence)
    broken_terminal["tick_native_evidence"][0]["fresh_owner_restore_receipt"][
        "restored_terminal_cell_digests"
    ]["cell-a"] = "3" * 64
    with pytest.raises(RouteQualificationError, match="policy-cache restore"):
        _validate_tick_native_route_obligations_v1(
            covered=obligations,
            exact_cases=(broken_terminal,),
        )


def test_tick_native_evidence_uses_the_absolute_terminal_cursor() -> None:
    row = _tick_native_evidence_row_v1(
        plan_execution={
            "total_ticks": 161,
            "checkpoint_ticks": [17, 160, 161],
            "deliberate_restore_tick": 17,
        },
        result={
            "terminal_next_absolute_tick": 161,
            "terminal_next_tick": 999,
            "checkpoint_catalog": {"17": {}, "160": {}, "161": {}},
            "fresh_owner_restore_receipt": {"restore_next_tick": 17},
        },
        batch_id="batch-0000",
    )

    assert row["actual_terminal_tick"] == 161
    assert row["actual_checkpoint_ticks"] == [17, 160, 161]


def test_tick_native_delta_inherits_only_same_artifact_base_route() -> None:
    tick_obligations = {
        "non_generation_divisible_terminal_tick",
        "non_generation_divisible_checkpoint_tick",
        "exact_absolute_tick_checkpoint_export",
        "fresh_owner_partial_generation_restore",
        "first_committed_tick_after_partial_restore",
    }
    base = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
        ),
        required_obligations=("checkpoint_export", "fresh_owner_restore"),
    )
    current = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id=base.provider_id,
        profile_id=base.profile_id,
        components=(
            *base.components,
            {
                "component_id": "absolute_tick_time_contract",
                "time_authority": "absolute_tick_v1",
            },
        ),
        required_obligations=tuple(
            sorted({*base.required_obligations, *tick_obligations})
        ),
    )
    prior = {
        "schema_version": "eiel_execution_route_qualification_v1",
        "authority_state": "route_qualified",
        "execution_route_sha256": base.execution_route_sha256,
        "provider_id": base.provider_id,
        "profile_id": base.profile_id,
        "covered_obligations": sorted(base.required_obligations),
        "artifact_sha256": "a" * 64,
        "variability_envelope": {"scope": "test"},
        "route_contract": base.payload(),
    }
    # Qualification inheritance crosses a durable JSON receipt boundary.  A
    # provider contract may contain tuples in memory, while the sealed receipt
    # necessarily reloads them as lists; canonical content, not Python
    # container identity, is the exactness authority.
    prior = json.loads(json.dumps(prior))
    plan = {
        "qualification_key": {
            "artifact_hashes": {"cuda_extension": "a" * 64},
            "variability_envelope": {"scope": "test"},
        },
        "route_qualification_inheritance": prior,
    }

    covered, result = _validated_route_qualification_inheritance_v1(
        campaign_plan=plan,
        route_contract=current,
    )

    assert covered == set(base.required_obligations)
    assert result is not None and result["component_contract_exact"] is True
    plan["qualification_key"]["artifact_hashes"]["cuda_extension"] = "b" * 64
    with pytest.raises(RouteQualificationError, match="does not exactly bind"):
        _validated_route_qualification_inheritance_v1(
            campaign_plan=plan,
            route_contract=current,
        )


def test_bounded_cases_preserve_route_affecting_mutation_regime_union() -> None:
    campaign = _campaign_plan()
    low = dict(campaign["cells"][0])  # type: ignore[index]
    low["parameters"] = {
        **low["parameters"],  # type: ignore[index]
        "genome_upgrade_v2.mutation_rate_q10": 16,
        "genome_upgrade_v2.mutation_step_q10": 16,
    }
    off = dict(low)
    off["cell_id"] = "source-cell-off"
    off["branch_id"] = "source-branch-off"
    off["rng_lineage_id"] = "source-rng-off"
    off["parameters"] = {
        **low["parameters"],  # type: ignore[index]
        "genome_upgrade_v2.mutation_rate_q10": 0,
        "genome_upgrade_v2.mutation_step_q10": 0,
    }
    campaign["cells"] = [low, off]

    canary = _canary()
    for case in canary["qualification_plan"]["cases"]:  # type: ignore[index]
        case["assignments"]["physical_batch_cells"] = 1

    plans = derive_bounded_route_campaigns_v1(campaign, canary)

    for plan in plans:
        regimes = {
            (
                cell["parameters"]["genome_upgrade_v2.mutation_rate_q10"],
                cell["parameters"]["genome_upgrade_v2.mutation_step_q10"],
            )
            for cell in plan["cells"]
        }
        assert regimes == {(0, 0), (16, 16)}
        assert all(len(batch["cell_ids"]) == 1 for batch in plan["physical_batches"])


def test_full_exact_canary_publishes_one_route_sidecar(tmp_path: Path) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    result = execute_route_qualification_v1(
        campaign_plan=_campaign_plan(),
        canary=_canary(),
        run_root=tmp_path / "run",
        route_contract=_contract(),
        qualification_receipt_path=mechanic_receipt,
        preflight_runner=_preflight,
        orchestration_runner=_exact_result,
    )

    sidecar = execution_route_qualification_sidecar_path_v1(mechanic_receipt)
    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert result["qualification_succeeded"] is True
    assert result["case_count"] == len(
        _canary()["qualification_plan"]["cases"]  # type: ignore[index]
    )
    assert published["authority_state"] == "route_qualified"
    assert published["execution_route_sha256"] == _contract().execution_route_sha256
    assert published["route_contract"] == _contract().payload()
    assert published["route_contract_sha256"] == _contract().execution_route_sha256
    assert published["covered_obligations"] == sorted(_contract().required_obligations)
    assert published["artifact_sha256"] == "a" * 64


def test_direct_route_rejects_generic_long_horizon_evidence_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engine.campaigns.route_qualification.validate_long_horizon_evidence_v1",
        lambda _evidence, *, expected_identity: {
            "qualification_dimensions": ["LONG_HORIZON_RUNABILITY_QUALIFIED"],
            "evidence_binding": {"expected_identity": expected_identity},
        },
    )

    def reject_generic_evidence(
        _evidence: dict[str, object], *, qualification_dimensions: object
    ) -> None:
        assert qualification_dimensions == ["LONG_HORIZON_RUNABILITY_QUALIFIED"]
        raise RouteQualificationError("dedicated qualification-pair required")

    monkeypatch.setattr(
        "engine.campaigns.route_qualification._validate_b136_5000tick_qualification_pair_v1",
        reject_generic_evidence,
    )
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    with pytest.raises(
        RouteQualificationError, match="dedicated qualification-pair required"
    ):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=_exact_result,
            long_horizon_evidence={"schema_version": "generic-long-horizon-v1"},
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_science_promotion_adds_attestation_without_relabeling_long_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="provider-v1",
        profile_id="profile-v1",
        components=(
            {
                "component_id": "acs_checkpoint_persistence",
                "contract_version": "eiel_acs_checkpoint_staging_v1",
                "immutable_snapshot_slots": 2,
            },
        ),
        required_obligations=tuple(sorted(SCIENCE_ROUTE_REQUIRED_OBLIGATIONS_V1)),
    )
    route = contract.execution_route_sha256
    terminal_digest = "a" * 64
    event_digest = "b" * 64
    identity = {
        "execution_route_sha256": route,
        "provider_id": contract.provider_id,
        "profile_id": contract.profile_id,
        "source_sha256": "d" * 64,
        "artifact_sha256s": {"cuda_extension": "e" * 64},
        "plan_sha256": "c" * 64,
        "run_id": "run-b136-5000-v1",
        "run_receipt_sha256": "f" * 64,
        "route_scope": "long_horizon_runability_route_v1",
        "geometry": {
            "batch_cells": 136,
            "world_width": 32,
            "population_capacity": 256,
            "controller_capacity": 512,
        },
    }
    windows = [
        {
            "start_tick": index * 1250,
            "end_tick": (index + 1) * 1250,
            "peak_gpu_allocated_bytes": 90,
            "peak_gpu_reserved_bytes": 100,
            "minimum_free_gpu_bytes": 500,
            "host_peak_rss_bytes": 100,
            "host_terminal_rss_bytes": 90,
            "resident_event_bytes": 64,
            "resident_source_bytes": 128,
            "event_ring_peak_occupancy": 10,
            "events_emitted": 100,
            "events_drained": 90 if index == 3 else 100,
        }
        for index in range(4)
    ]
    evidence_body = {
        "schema_version": "eiel_long_horizon_qualification_evidence_v1",
        "evidence_identity": identity,
        "resource_limits": {
            "gpu_budget_bytes": 1000,
            "host_budget_bytes": 1000,
            "minimum_free_gpu_bytes_required": 100,
            "maximum_gpu_memory_slope_bytes_per_window": 10,
            "maximum_host_memory_slope_bytes_per_window": 10,
            "maximum_event_ring_occupancy_slope_per_window": 10,
        },
        "ticks_requested": 5000,
        "ticks_committed": 5000,
        "owner_boundaries_crossed": 4,
        "peak_gpu_allocated_bytes": 90,
        "peak_gpu_reserved_bytes": 100,
        "minimum_free_gpu_bytes": 500,
        "host_terminal_rss_bytes": 90,
        "gpu_memory_slope_bytes_per_window": 0.0,
        "gpu_allocated_memory_slope_bytes_per_window": 0.0,
        "host_peak_rss_bytes": 100,
        "host_memory_slope_bytes_per_window": 0.0,
        "resident_event_bytes": 64,
        "resident_event_bytes_slope_per_window": 0.0,
        "resident_source_bytes": 128,
        "resident_source_bytes_slope_per_window": 0.0,
        "event_ring_capacity": 64,
        "event_ring_peak_occupancy": 10,
        "event_ring_occupancy_slope_per_window": 0.0,
        "events_emitted": 400,
        "events_drained": 390,
        "terminal_event_ring_occupancy": 10,
        "external_event_history_count": 390,
        "external_event_history_digest": event_digest,
        "checkpoint_frontier": {
            "next_absolute_tick": 5000,
            "terminal_digest": terminal_digest,
        },
        "checkpoint_restore": {
            "checkpoint_tick": 2501,
            "checkpoint_digest": "1" * 64,
            "restored_checkpoint_digest": "1" * 64,
            "fresh_owner_restore_tick": 3361,
            "fresh_owner_uninterrupted_terminal_digest": terminal_digest,
            "fresh_owner_restored_terminal_digest": terminal_digest,
            "fresh_owner_uninterrupted_event_digest": event_digest,
            "fresh_owner_restored_event_digest": event_digest,
        },
        "terminal_digest": terminal_digest,
        "event_digest": event_digest,
        "windows": windows,
        "errors": [],
        "ooms": [],
    }
    receipt_path = tmp_path / "mechanic" / "qualification.json"
    sidecar = execution_route_qualification_sidecar_path_v1(
        receipt_path, contract.execution_route_sha256
    )
    sidecar.parent.mkdir(parents=True)

    artifacts = {"cuda_extension": "e" * 64}
    fixed_invariants = {
        "checkpoint_schema": "authoritative_state_schema_bound",
        "device_owner": "cuda_resident_no_fallback",
        "parameter.cuda_architecture_b.cpu_request_staging_enable": False,
        "parameter.cuda_architecture_b.cpython_gauss_segment_capacity": 0,
        "parameter.cuda_architecture_b.policy_birth_segment_refill_enable": False,
        "parameter.cuda_architecture_b.turnover_mlp_shadow_elide_enable": False,
        "parameter.eed.ecology_frozen": False,
        "parameter.eed.phenotype_fixed": False,
        "parameter.eed.resource_impact_apply_enable": True,
        "parameter.genome_upgrade_v2.enable": True,
        "parameter.genome_upgrade_v2.mutation_rate_q10": 20,
        "parameter.genome_upgrade_v2.mutation_step_q10": 64,
        "parameter.genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
        "parameter.patch.capacity": 512,
        "parameter.patch.grid_size": 32,
        "parameter.patch.initial_population": 256,
        "parameter.production_lifecycle.life_history_profile": "stage1s_overlap_tick_life_history_v1",
        "parameter.production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
        "parameter.production_lifecycle.ordinary_births_enabled": True,
        "parameter.production_lifecycle.turnover_cull_fraction": 0.0,
        "parameter.production_lifecycle.turnover_enabled": True,
        "parameter.production_lifecycle.turnover_parent_fraction": 1.0,
        "runtime.policy_family": "fixed_mlp",
        "runtime.selection_mode": "genotype_blind_neutral",
        "runtime.world_base": "patch",
        "runtime.world_overlays": '["eed_resource_impact","jungle","wdl_delayed_hazard"]',
        "runtime_identity_contract": "campaign_requirements_bound_runtime_identity_v1",
        "semantic_registry": "eiel_global_genome_registry_v1",
    }
    envelope = {
        **_canonical_production_wrc_envelope_v1(
            [{"name": "env_cfg.eed_resource_a_capacity"}]
        )
    }
    fixed_invariants["wrc.parameter_variability_envelope_sha256"] = envelope[
        "envelope_sha256"
    ]
    qualification_key = {
        "schema_version": "eiel_runtime_qualification_key_v1",
        "profile_id": "eiel-production-runtime-v1",
        "profile_version": 2,
        "backend": "cuda_resident",
        "fixed_invariants": fixed_invariants,
        "artifact_hashes": artifacts,
        "variability_envelope": envelope,
    }
    qualification_key_sha256 = canonical_sha256(qualification_key)
    run_roots = {
        "uninterrupted": tmp_path / "gate26-uninterrupted",
        "restored": tmp_path / "gate26-restored",
    }
    cells = [
        {"cell_id": f"cell-{index:03d}"}
        for index in range(136)
    ]
    capabilities = {
        "patch": {
            "configured": True,
            "initializable": True,
            "supported": True,
            "evidence": {
                "component_id": "world.patch.v1",
                "contract_version": "patch_world_mechanics_v1",
            },
        },
        "eed": {
            "configured": True,
            "initializable": True,
            "supported": True,
            "evidence": {
                "component_id": "world.overlay.eed_resource_impact.v1",
                "contract_version": "eed_resource_impact_world_mechanics_v1",
            },
        },
        "jungle": {
            "configured": True,
            "initializable": True,
            "supported": True,
            "evidence": {
                "component_id": "world.overlay.jungle.v1",
                "contract_version": "jungle_world_mechanics_v1",
            },
        },
        "wdl_delayed_hazard": {
            "configured": True,
            "initializable": True,
            "supported": True,
            "evidence": {
                "component_id": "world.overlay.wdl_delayed_hazard.v1",
                "contract_version": "wdl_delayed_hazard_world_mechanics_v1",
            },
        },
    }
    for name in (
        "genome_v2_ordinary_birth",
        "genome_v2_mutation",
        "recombination_profile",
    ):
        capabilities[name] = {
            "configured": True,
            "initializable": True,
            "supported": True,
            "evidence": {"component_id": "genome.genome_upgrade_v2.v1"},
        }
    required_capabilities = sorted(capabilities)

    def _write_json(path: Path, payload: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _make_plan(name: str, *, restore: bool) -> dict:
        root = run_roots[name]
        execution = {
            "acs_capture_instrumentation": True,
            "acs_checkpoint_slots": 2,
            "checkpoint_storage_mode": "disk",
            "checkpoint_ticks": [2500, 5000],
            "deliberate_restore_tick": 2500 if restore else None,
            "deliberate_restore_ticks": [2500] if restore else [],
            "execution_route_identity": "tick_native_execution_route_v1",
            "native_streamed_checkpoint": True,
            "start_next_tick": 0,
            "time_authority": "absolute_tick_v1",
            "total_ticks": 5000,
            "tick_schedule": {
                "schema_version": "eiel-absolute-tick-schedule-v2",
                "science_sample_interval_ticks": None,
                "total_ticks": 5000,
            },
            "tick_native_control": {
                "schema_version": "eiel-tick-native-control-v2",
                "time_authority": "absolute_tick_v1",
                "total_ticks": 5000,
                "checkpoint_ticks": [2500, 5000],
                "source_window_ticks": 160,
                "native_dispatch_window_ticks": 1120,
                "resident_owner_span_ticks": 1120,
                "event_window_capacity": 41804,
                "schedules": {
                    "archive_ticks": [5000],
                    "checkpoint_ticks": [2500, 5000],
                    "reducer_ticks": [5000],
                    "report_ticks": [5000],
                    "science_sample_ticks": [5000],
                },
            },
        }
        plan = {
            "schema_version": "eiel_campaign_plan_v1",
            "resolved_profile": "eiel-production-runtime-v1",
            "campaign_id": identity["run_id"],
            "qualification_key": qualification_key,
            "qualification_key_sha256": qualification_key_sha256,
            "run_contract": {
                "schema_version": "eiel_resolved_run_contract_v1",
                "requested": {
                    "mode": "strict",
                    "profile_id": "eiel_full_evolution_v1",
                },
                "required_capabilities": required_capabilities,
                "capabilities": capabilities,
                "qualification_key_sha256": qualification_key_sha256,
            },
            "execution": execution,
            "output": {"runtime_root": str(root)},
            "cells": cells,
            "physical_batches": [{"batch_id": "batch-0000", "cell_ids": [cell["cell_id"] for cell in cells]}],
            "artifact_lifecycle": {
                "contract_sha256": "l" * 64,
            },
        }
        plan["plan_sha256"] = canonical_sha256(plan)
        _write_json(root / "execution_plan.json", plan)
        return plan

    plans = {
        "uninterrupted": _make_plan("uninterrupted", restore=False),
        "restored": _make_plan("restored", restore=True),
    }
    identity["plan_sha256"] = plans["uninterrupted"]["plan_sha256"]

    def _make_passport(name: str) -> dict:
        plan = plans[name]
        root = run_roots[name]
        passport = {
            "schema_version": "eiel_run_passport_v1",
            "run_id": root.name,
            "plan_sha256": plan["plan_sha256"],
            "qualification_key_sha256": qualification_key_sha256,
            "terminal_status": "completed",
            "lifecycle_contract_sha256": "l" * 64,
            "provider_identity": {
                "execution_route_sha256": route,
                "provider_id": contract.provider_id,
                "profile_id": contract.profile_id,
            },
            "source_identity": {"artifact_hashes": artifacts},
        }
        passport["passport_sha256"] = canonical_sha256(passport)
        _write_json(root / "run_passport.json", passport)
        return passport

    passports = {
        "uninterrupted": _make_passport("uninterrupted"),
        "restored": _make_passport("restored"),
    }
    evidence = {**evidence_body}
    evidence["paired_run_bindings"] = {
        name: {
            "plan_sha256": plans[name]["plan_sha256"],
            "run_passport_sha256": passports[name]["passport_sha256"],
            "run_root": str(run_roots[name]),
        }
        for name in ("uninterrupted", "restored")
    }
    evidence["paired_run_bindings_sha256"] = canonical_sha256(
        evidence["paired_run_bindings"]
    )
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    evidence_path = tmp_path / "gate26-evidence.json"
    _write_json(evidence_path, evidence)
    validated = validate_long_horizon_evidence_v1(evidence)
    long_scope = {
        "validated_evidence": evidence,
        "evidence_binding": validated["evidence_binding"],
    }
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
                "authority_state": "route_qualified",
                "execution_route_sha256": contract.execution_route_sha256,
                "route_contract_sha256": contract.execution_route_sha256,
                "route_contract": contract.payload(),
                "provider_id": contract.provider_id,
                "profile_id": contract.profile_id,
                    "covered_obligations": sorted(SCIENCE_ROUTE_REQUIRED_OBLIGATIONS_V1),
                    "missing_obligations": [],
                    "route_obligation_results": [],
                    "case_ids": [
                        "disk-continuation",
                        "fresh-repeat-a",
                        "fresh-repeat-b",
                        "partition-alternate",
                        "partition-default",
                        "reporting-disabled",
                        "reporting-enabled",
                        "tick-native-partial-restore",
                        "uninterrupted",
                    ],
                    "reporting_results": {"case_count": 9, "exact": True},
                "artifact_sha256": "e" * 64,
                "artifact_sha256s": {"cuda_extension": "e" * 64},
                "qualification_dimensions": [
                    BOUNDED_ROUTE_QUALIFIED,
                    LONG_HORIZON_RUNABILITY_QUALIFIED,
                ],
                "long_horizon_scope": long_scope,
            }
        ),
        encoding="utf-8",
    )
    for name in ("uninterrupted", "restored"):
        rows = [
            {
                "cell_id": cell["cell_id"],
                "metrics": {"runtime.terminal_absolute_tick": 5000},
            }
            for cell in cells
        ]
        _write_json(run_roots[name] / "cell_metrics.json", {
            "schema_version": "eiel_campaign_cell_metrics_v1",
            "campaign_id": identity["run_id"],
            "plan_sha256": plans[name]["plan_sha256"],
            "rows": rows,
        })
    gate24 = {
        "schema_version": "gate24-test-v1",
        "decision": "PASS",
        "criteria": [
            {
                "criterion": criterion,
                "status": "PASS",
            }
            for criterion in (
                "exact physical B136 W32/P256/C512 and 5000 absolute ticks",
                "exact event capacity and bytes",
                "all four expected in-place owner transitions with no overlap",
                "two-slot detached ACS high-water exactly two with zero queued CUDA references",
                "capture/release and successor work within ten seconds",
                "minimum VRAM, host RSS, pinned memory and temporary-file bounds",
                "no CPU staging/refill or future RNG/event tapes",
                "GPU duty, SM utilization, dispatch overhead and idle-gap thresholds",
                "terminal GPU release before separately measured ACS tail",
                "mechanical lifecycle, passport, plan and storage receipt bindings",
                "absolute-tick public runtime and checkpoint authority with compatibility metadata removed",
            )
        ],
        "execution_identity": {
            "qualification_key_sha256": qualification_key_sha256,
            "result_class": "SUCCESS",
            "baseline_profile": "EIEL_PRODUCTION_BASELINE_V1",
            "baseline_profile_version": 2,
            "baseline_status": "BASELINE_CONFORMANT",
            "baseline_deviations": [],
        },
        "acceptance_measurements": {
            "event_rows_per_cell": 41804,
            "event_row_bytes": 141,
            "total_event_bytes": 801633504,
            "minimum_free_vram_bytes": 1000,
            "minimum_free_vram_threshold_bytes": 500,
            "capture_barrier_tick2500_seconds": 1.0,
            "capture_barrier_tick5000_seconds": 1.0,
            "capture_release_threshold_seconds": 10.0,
            "successor_gpu_dispatch_delay_tick2500_seconds": 1.0,
            "successor_dispatch_threshold_seconds": 10.0,
            "active_gpu_duty_percent": 90.0,
            "active_gpu_duty_threshold_percent": 80.0,
            "mean_sm_percent": 90.0,
            "mean_sm_threshold_percent": 50.0,
            "dispatch_overhead_fraction": 0.01,
            "dispatch_overhead_threshold_fraction": 0.05,
            "longest_unexplained_idle_interval_seconds": 1.0,
            "idle_interval_threshold_seconds": 2.0,
            "queue_occupancy_high_water": 2,
            "queued_cuda_references": 0,
            "queued_cuda_callbacks": 0,
            "temporary_capture_file_high_water": 2,
            "temporary_capture_file_terminal_count": 0,
            "host_pinned_bytes": 0,
            "rss_high_water_bytes": 1000,
            "native_host_refill_calls": 0,
            "intermediate_host_to_device_copies": 0,
            "intermediate_device_to_host_copies": 0,
            "t_sim_seconds": 1.0,
            "t_detach_seconds": 1.0,
            "t_persist_seconds": 1.0,
            "t_acs_tail_seconds": 1.0,
            "t_restore_seconds": 1.0,
            "terminal_gpu_device_reset_return_code": 0,
            "post_release_target_gpu_resident_samples": 0,
            "post_release_target_max_framebuffer_bytes": 0,
            "wall_reconciliation_error_fraction": 0.0,
        },
        "physical_authority": {
            "batch_count": 1,
            "cell_count": 136,
            "world_width": 32,
            "population": 256,
            "capacity": 512,
            "time_authority": "absolute_tick_v1",
            "total_ticks": 5000,
            "checkpoint_ticks": [2500, 5000],
            "expected_owner_transition_ticks": [1120, 2240, 3360, 4480],
            "observed_owner_transition_ticks": [1120, 2240, 3360, 4480],
            "physical_owner_identity_count": 1,
            "resident_owner_max_simultaneous": 1,
            "resident_owner_live_count_terminal": 0,
        },
    }
    gate24_path = tmp_path / "gate24.json"
    _write_json(gate24_path, gate24)
    gate25_path = tmp_path / "gate25.json"
    _write_json(gate25_path, {
        "schema_version": "eiel_qualification_impact_map_v1",
        "gate": 25,
        "decision": "TARGETED_BOUNDED_ROUTE_REQUALIFICATION_COMPLETE_LONG_HORIZON_DELTA_PENDING",
        "target": {
            "qualification_key_sha256": qualification_key_sha256,
            "execution_route_sha256": route,
        },
        "uncertainty": {"impact_uncertain_count": 0, "unsafe_inheritance_count": 0},
        "impact_classifications": [{
            "group": "Gates 21-24 physical B136 mechanics",
            "classification": "PRESERVED_COMPATIBLE_INTERFACE",
        }],
    })
    gate27_path = tmp_path / "gate27.json"
    _write_json(gate27_path, {
        "schema_version": "eiel_qualification_impact_map_v1",
        "gate": 27,
        "decision": "TARGETED_SCIENCE_ROUTE_AUTHORITY_DELTA_ONLY",
        "target": {
            "execution_route_sha256": route,
            "gate26_evidence_sha256": evidence["evidence_sha256"],
        },
        "targeted_delta": {
            "native_rebuild": False,
            "gpu_execution": False,
            "broad_requalification": False,
        },
        "uncertainty": {"impact_uncertain_count": 0, "unsafe_inheritance_count": 0},
    })
    source_route_receipt_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    source_route_receipt_bytes = sidecar.read_bytes()
    def _ref(role: str, path: Path, *, source: str | None = None) -> dict:
        result = {
            "role": role,
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if source is not None:
            result["source_route_receipt_sha256"] = source
        return result

    refs = [
        _ref("gate26_evidence", evidence_path),
        _ref("gate26_uninterrupted_plan", run_roots["uninterrupted"] / "execution_plan.json"),
        _ref("gate26_restored_plan", run_roots["restored"] / "execution_plan.json"),
        _ref("gate26_uninterrupted_passport", run_roots["uninterrupted"] / "run_passport.json"),
        _ref("gate26_restored_passport", run_roots["restored"] / "run_passport.json"),
        _ref("gate26_uninterrupted_cell_metrics", run_roots["uninterrupted"] / "cell_metrics.json"),
        _ref("gate26_restored_cell_metrics", run_roots["restored"] / "cell_metrics.json"),
        _ref("promoted_route_receipt", sidecar, source=source_route_receipt_sha256),
        _ref("gate24_resource_decision", gate24_path),
        _ref("gate25_impact_map", gate25_path),
        _ref("gate27_impact_map", gate27_path),
    ]
    attestation = materialize_canonical_full_production_route_attestation_v1(
        evidence_refs=refs,
        expected_identity=validated["evidence_identity"],
        expected_evidence_sha256=evidence["evidence_sha256"],
        expected_evidence_binding_sha256=validated["evidence_binding"]["binding_sha256"],
        expected_source_route_receipt_sha256=source_route_receipt_sha256,
    )
    from engine.campaigns import route_qualification as route_qualification_module

    validation_markers: list[str] = []
    original_attestation_validator = (
        route_qualification_module.validate_canonical_full_production_route_attestation_v1
    )

    def observe_attestation_validation(*args, **kwargs):
        validation_markers.append("attestation")
        return original_attestation_validator(*args, **kwargs)

    monkeypatch.setattr(
        route_qualification_module,
        "validate_canonical_full_production_route_attestation_v1",
        observe_attestation_validation,
    )
    pair_revalidation_modes: list[object] = []

    def observe_gate27_pair_revalidation(*_args, **kwargs):
        validation_markers.append("pair")
        pair_revalidation_modes.append(kwargs.get("canonical_revalidation"))

    monkeypatch.setattr(
        "engine.campaigns.route_qualification._validate_b136_5000tick_qualification_pair_v1",
        observe_gate27_pair_revalidation,
    )
    result = promote_execution_route_science_evidence_v1(
        campaign_plan={
                "plan_sha256": plans["uninterrupted"]["plan_sha256"],
                "qualification_key": {
                    "artifact_hashes": {"cuda_extension": "e" * 64}
                },
        },
        route_contract=contract,
        qualification_receipt_path=receipt_path,
        long_horizon_evidence=evidence,
        canonical_full_production_route_attestation=attestation,
    )
    projection = execution_route_science_qualification_projection_path_v1(
        receipt_path, contract.execution_route_sha256
    )
    published = json.loads(projection.read_text(encoding="utf-8"))
    assert result["qualification_succeeded"] is True
    assert result["qualification_receipt_path"] == str(projection)
    assert sidecar.read_bytes() == source_route_receipt_bytes
    assert published["qualification_dimensions"] == [
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
        SCIENCE_ROUTE_QUALIFIED,
    ]
    assert published["long_horizon_scope"] == long_scope
    assert published["long_horizon_scope"]["validated_evidence"]["evidence_identity"][
        "route_scope"
    ] == "long_horizon_runability_route_v1"
    assert published["science_route_scope"]["route_scope"] == (
        "canonical_full_production_route_v1"
    )
    assert validation_markers == ["attestation", "pair"]
    read_back = read_execution_route_qualification_sidecar_v1(
        receipt_path, contract.execution_route_sha256
    )
    assert read_back == published
    monkeypatch.setattr(
        "engine.campaigns.qualification.validate_execution_route_acs_focused_evidence_v1",
        lambda **_kwargs: (
            {"async_checkpoint_backpressure", "async_checkpoint_failure_latch"},
            None,
        ),
    )
    status = execution_route_qualification_status_v1(contract, read_back)
    assert status["qualification_dimensions"] == [
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
        SCIENCE_ROUTE_QUALIFIED,
    ]
    idempotent = promote_execution_route_science_evidence_v1(
        campaign_plan={
            "plan_sha256": plans["uninterrupted"]["plan_sha256"],
            "qualification_key": {"artifact_hashes": {"cuda_extension": "e" * 64}},
        },
        route_contract=contract,
        qualification_receipt_path=receipt_path,
        long_horizon_evidence=evidence,
        canonical_full_production_route_attestation=attestation,
    )
    assert idempotent["idempotent"] is True
    assert pair_revalidation_modes == [False, False]
    assert validation_markers == [
        "attestation",
        "pair",
        "attestation",
        "attestation",
        "pair",
    ]
    assert sidecar.read_bytes() == source_route_receipt_bytes
    sidecar.write_bytes(source_route_receipt_bytes + b"\n")
    with pytest.raises(RouteQualificationError, match="attestation"):
        promote_execution_route_science_evidence_v1(
            campaign_plan={
                "plan_sha256": plans["uninterrupted"]["plan_sha256"],
                "qualification_key": {"artifact_hashes": {"cuda_extension": "e" * 64}},
            },
            route_contract=contract,
            qualification_receipt_path=receipt_path,
            long_horizon_evidence=evidence,
            canonical_full_production_route_attestation=attestation,
        )


def test_route_sidecars_are_partitioned_by_route_hash(tmp_path: Path) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    first = _contract()
    second = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id=first.provider_id,
        profile_id=first.profile_id,
        components=(
            {
                "component_id": "controller",
                "contract_version": "fixed-mlp-v2",
            },
        ),
        required_obligations=first.required_obligations,
    )

    for contract in (first, second):
        canary = generate_execution_route_canary_plan(
            {
                **contract.payload(),
                "execution_route_sha256": contract.execution_route_sha256,
                "canary_requirements": {
                    "qualified_ticks_per_generation": 80,
                    "admitted_physical_batch_sizes": [1, 2],
                },
            }
        )
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=canary,
            run_root=tmp_path / "run" / contract.execution_route_sha256,
            route_contract=contract,
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=_exact_result,
        )

    first_sidecar = execution_route_qualification_sidecar_path_v1(
        mechanic_receipt,
        first.execution_route_sha256,
    )
    second_sidecar = execution_route_qualification_sidecar_path_v1(
        mechanic_receipt,
        second.execution_route_sha256,
    )
    assert first_sidecar.is_file()
    assert second_sidecar.is_file()
    assert first_sidecar != second_sidecar
    assert (
        read_execution_route_qualification_sidecar_v1(
            mechanic_receipt,
            first.execution_route_sha256,
        )["execution_route_sha256"]
        == first.execution_route_sha256
    )
    assert (
        read_execution_route_qualification_sidecar_v1(
            mechanic_receipt,
            second.execution_route_sha256,
        )["execution_route_sha256"]
        == second.execution_route_sha256
    )


def test_reader_does_not_use_another_route_science_projection_as_source_fallback(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    unrelated_projection = execution_route_science_qualification_projection_path_v1(
        mechanic_receipt, "a" * 64
    )
    unrelated_projection.parent.mkdir(parents=True)
    unrelated_projection.write_text("{}", encoding="utf-8")
    assert (
        read_execution_route_qualification_sidecar_v1(
            mechanic_receipt, "b" * 64
        )
        is None
    )


def test_turnover_only_semantic_obligations_publish_with_exact_counts(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    result = execute_route_qualification_v1(
        campaign_plan=_turnover_campaign_plan(),
        canary=_turnover_canary(),
        run_root=tmp_path / "run",
        route_contract=_turnover_contract(),
        qualification_receipt_path=mechanic_receipt,
        preflight_runner=_preflight,
        orchestration_runner=_turnover_exact_result,
    )

    sidecar = execution_route_qualification_sidecar_path_v1(mechanic_receipt)
    published = json.loads(sidecar.read_text(encoding="utf-8"))
    assert result["qualification_succeeded"] is True
    assert published["covered_obligations"] == sorted(
        _turnover_contract().required_obligations
    )
    assert published["route_obligation_results"]
    assert {
        "ordinary_birth_disabled",
        "turnover_refill",
    } <= set(published["route_obligation_results"][0]["covered_obligations"])


def test_turnover_semantics_use_reporting_enabled_witness_when_available() -> None:
    disabled = _turnover_campaign_plan()
    disabled["route_qualification_case_id"] = "reporting-disabled"
    disabled["execution_route_sha256"] = "a" * 64
    disabled["execution"]["reporting_enabled"] = False  # type: ignore[index]
    disabled["comparisons"] = {"branches": []}
    enabled = deepcopy(disabled)
    enabled["route_qualification_case_id"] = "reporting-enabled"
    enabled["execution"]["reporting_enabled"] = True  # type: ignore[index]

    disabled_result = _turnover_exact_result(disabled, Path("/tmp/not-used"))
    disabled_result["case_id"] = "reporting-disabled"
    disabled_result["route_obligation_evidence"]["cells"][0][  # type: ignore[index]
        "generation_rows"
    ] = []
    enabled_result = _turnover_exact_result(enabled, Path("/tmp/not-used"))
    enabled_result["case_id"] = "reporting-enabled"

    rows = _validate_semantic_route_obligations_v1(
        covered={"ordinary_birth_disabled", "turnover_refill"},
        plans=(disabled, enabled),
        exact_cases=(disabled_result, enabled_result),
    )

    assert [row["case_id"] for row in rows] == ["reporting-enabled"]


def test_turnover_refill_accepts_one_exact_cell_and_one_extinction_cell() -> None:
    plan = _turnover_campaign_plan()
    plan["route_qualification_case_id"] = "turnover-case"
    plan["execution_route_sha256"] = "a" * 64
    cell = plan["cells"][0]
    extinct = {**cell, "cell_id": "extinct-cell"}
    plan["cells"] = [cell, extinct]
    qualifying = _turnover_exact_result(plan, Path("/tmp/not-used"))[
        "route_obligation_evidence"
    ]["cells"]
    qualifying[1]["cell_id"] = "extinct-cell"
    qualifying[1]["generation_rows"][1].update(
        {"population_alive": 0, "successful_births": 0, "refill_child_count": 0}
    )

    result = _validate_semantic_route_obligations_v1(
        covered={"ordinary_birth_disabled", "turnover_refill"},
        plans=[plan],
        exact_cases=[
            {
                "case_id": "turnover-case",
                "route_obligation_evidence": {
                    "schema_version": "eiel_route_obligation_evidence_v1",
                    "cells": qualifying,
                },
            }
        ],
    )

    assert {row["cell_id"] for row in result} == {"source-cell", "extinct-cell"}


def test_turnover_only_accepts_capacity_headroom_with_exact_target_refill(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    campaign = _turnover_campaign_plan()
    campaign["cells"][0]["parameters"]["patch.capacity"] = 512  # type: ignore[index]

    def capacity_headroom_result(
        plan: dict[str, object], run_root: Path
    ) -> dict[str, object]:
        result = _turnover_exact_result(plan, run_root)
        cells = result["route_obligation_evidence"]["cells"]  # type: ignore[index]
        cells[0]["resident_capacity"] = 512  # type: ignore[index]
        return result

    result = execute_route_qualification_v1(
        campaign_plan=campaign,
        canary=_turnover_canary(),
        run_root=tmp_path / "run",
        route_contract=_turnover_contract(),
        qualification_receipt_path=mechanic_receipt,
        preflight_runner=_preflight,
        orchestration_runner=capacity_headroom_result,
    )

    assert result["qualification_succeeded"] is True


def test_turnover_only_refuses_accidental_zero_birth_without_disabled_profile(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    campaign = _turnover_campaign_plan()
    campaign["cells"][0]["parameters"][  # type: ignore[index]
        "production_lifecycle.ordinary_births_enabled"
    ] = True

    with pytest.raises(
        RouteQualificationError,
        match="explicit disabled lifecycle control",
    ):
        execute_route_qualification_v1(
            campaign_plan=campaign,
            canary=_turnover_canary(),
            run_root=tmp_path / "run",
            route_contract=_turnover_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=_turnover_exact_result,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_turnover_only_refuses_missing_ordinary_birth_counter(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def missing_counter(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _turnover_exact_result(plan, run_root)
        cells = result["route_obligation_evidence"]["cells"]  # type: ignore[index]
        cells[0]["generation_rows"][1]["ordinary_birth_count"] = None  # type: ignore[index]
        return result

    with pytest.raises(
        RouteQualificationError,
        match="exact zero ordinary-birth counts",
    ):
        execute_route_qualification_v1(
            campaign_plan=_turnover_campaign_plan(),
            canary=_turnover_canary(),
            run_root=tmp_path / "run",
            route_contract=_turnover_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=missing_counter,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_turnover_obligation_evidence_reads_nested_generation_metrics() -> None:
    plan = _turnover_campaign_plan()
    batch = plan["physical_batches"][0]
    cell_id = batch["cell_ids"][0]
    evidence = _batch_route_obligation_evidence_v1(
        plan=plan,
        batch=batch,
        manifest={
            "metrics_rows": [
                {
                    "cell_id": cell_id,
                    "generation_metrics": [
                        {
                            "generation": 0,
                            "alive": 256,
                            "successful_births": 0,
                            "ordinary_birth_count": 0,
                        },
                        {
                            "generation": 1,
                            "alive": 256,
                            "successful_births": 0,
                            "ordinary_birth_count": 0,
                            "turnover_refill_child_count": 128,
                        },
                    ],
                }
            ]
        },
    )

    assert len(evidence) == 1
    row = evidence[0]["generation_rows"][1]
    assert row["population_alive"] == 256
    assert row["ordinary_birth_count"] == 0
    assert row["refill_child_count"] == 128
    assert row["successful_births"] == 128


def test_turnover_only_refuses_missing_refill_evidence(tmp_path: Path) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def missing_refill(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _turnover_exact_result(plan, run_root)
        cells = result["route_obligation_evidence"]["cells"]  # type: ignore[index]
        cells[0]["generation_rows"][1]["refill_child_count"] = 0  # type: ignore[index]
        return result

    with pytest.raises(
        RouteQualificationError,
        match="generation-boundary refill evidence",
    ):
        execute_route_qualification_v1(
            campaign_plan=_turnover_campaign_plan(),
            canary=_turnover_canary(),
            run_root=tmp_path / "run",
            route_contract=_turnover_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=missing_refill,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_alternate_partition_mismatch_publishes_nothing(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def mismatched_partition(
        plan: dict[str, object], run_root: Path
    ) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        if plan["route_qualification_case_id"] == "partition-alternate":
            result["terminal_cell_digests_by_ordinal"] = ["changed", "changed"]
            result["terminal_cell_digests_by_cell_id"] = {
                str(cell["cell_id"]): "changed" for cell in plan["cells"]
            }
        return result

    with pytest.raises(
        RouteQualificationError,
        match="alternate batch partition terminal state is not exact",
    ):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=mismatched_partition,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_alternate_partition_requires_logical_cell_digest_match(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def ordinal_only_match(
        plan: dict[str, object], run_root: Path
    ) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        if plan["route_qualification_case_id"] == "partition-alternate":
            result["terminal_cell_digests_by_ordinal"] = [
                "terminal-0",
                "terminal-1",
            ]
            result["terminal_cell_digests_by_cell_id"] = {
                str(cell["cell_id"]): f"changed-{index}"
                for index, cell in enumerate(plan["cells"])
            }
        return result

    with pytest.raises(
        RouteQualificationError,
        match="alternate batch partition terminal state is not exact",
    ):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=ordinal_only_match,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_missing_required_transition_sentinel_publishes_nothing(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def missing_boundary(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        result["observed_transition_names"] = ["checkpoint_export_boundary"]
        return result

    with pytest.raises(
        RouteQualificationError,
        match="generation_boundary",
    ):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=missing_boundary,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


@pytest.mark.parametrize(
    ("missing_obligation", "missing_transition"),
    [
        ("first_post_boundary_tick", "first_tick_after_boundary"),
        ("second_post_boundary_tick", "second_tick_after_boundary"),
        (
            "post_boundary_policy_cache_refill",
            "first_policy_cache_refill_after_boundary",
        ),
    ],
)
def test_tick_native_boundary_obligations_require_exact_sentinel_names(
    tmp_path: Path,
    missing_obligation: str,
    missing_transition: str,
) -> None:
    contract = ExecutionRouteContractV1(
        schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
        provider_id="fixed-provider",
        profile_id="fixed-profile",
        components=(
            {"component_id": "controller", "contract_version": "fixed-mlp-v1"},
        ),
        required_obligations=(
            "first_post_boundary_tick",
            "second_post_boundary_tick",
            "post_boundary_policy_cache_refill",
        ),
    )
    canary = generate_execution_route_canary_plan(
        {
            **contract.payload(),
            "execution_route_sha256": contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1, 2],
            },
        }
    )
    transition_by_obligation = {
        "first_post_boundary_tick": "first_tick_after_boundary",
        "second_post_boundary_tick": "second_tick_after_boundary",
        "post_boundary_policy_cache_refill": (
            "first_policy_cache_refill_after_boundary"
        ),
    }

    def missing_one(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        result["execution_route_sha256"] = contract.execution_route_sha256
        result["observed_transition_names"] = [
            transition
            for obligation, transition in transition_by_obligation.items()
            if obligation != missing_obligation
        ]
        return result

    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    with pytest.raises(RouteQualificationError, match=missing_transition):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=canary,
            run_root=tmp_path / "run",
            route_contract=contract,
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=missing_one,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


@pytest.mark.parametrize(
    "failure",
    ["case_non_exact", "sentinel_non_exact"],
)
def test_failed_or_partial_canary_publishes_nothing(
    tmp_path: Path, failure: str
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def failing_result(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        if plan["route_qualification_case_id"] == "fresh-repeat-b":
            if failure == "case_non_exact":
                result["exact"] = False
            elif failure == "sentinel_non_exact":
                result["transition_sentinels_exact"] = False
        return result

    with pytest.raises(RouteQualificationError):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=failing_result,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_fresh_repeat_terminal_digest_mismatch_publishes_nothing(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"

    def mismatched_result(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        result = _exact_result(plan, run_root)
        if plan["route_qualification_case_id"] == "fresh-repeat-b":
            result["terminal_cell_digests_by_cell_id"] = {
                str(cell["cell_id"]): "different-terminal" for cell in plan["cells"]
            }
        return result

    with pytest.raises(RouteQualificationError, match="fresh repeat"):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=mismatched_result,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_preflight_failure_stops_before_execution_and_publication(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    executions = 0

    def rejected(plan: dict[str, object]) -> dict[str, object]:
        receipt = _preflight(plan)
        receipt["ready_for_execution"] = False
        receipt["reasons"] = [{"code": "capacity"}]
        return receipt

    def should_not_run(plan: dict[str, object], run_root: Path) -> dict[str, object]:
        nonlocal executions
        executions += 1
        return _exact_result(plan, run_root)

    with pytest.raises(RouteQualificationError, match="bounded preflight"):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=_canary(),
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=rejected,
            orchestration_runner=should_not_run,
        )

    assert executions == 0
    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()


def test_partial_obligation_canary_cannot_publish_final_authority(
    tmp_path: Path,
) -> None:
    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    partial = generate_execution_route_canary_plan(
        {
            **_contract().payload(),
            "execution_route_sha256": _contract().execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1, 2],
            },
        },
        missing_obligations=["generation_boundary"],
    )

    with pytest.raises(RouteQualificationError, match="every required"):
        execute_route_qualification_v1(
            campaign_plan=_campaign_plan(),
            canary=partial,
            run_root=tmp_path / "run",
            route_contract=_contract(),
            qualification_receipt_path=mechanic_receipt,
            preflight_runner=_preflight,
            orchestration_runner=_exact_result,
        )

    assert not execution_route_qualification_sidecar_path_v1(mechanic_receipt).exists()
