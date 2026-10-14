from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.compiler import CampaignError
from engine.campaigns.providers import (
    CampaignProviderError,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
)
from engine.campaigns.qualification import (
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
    execution_route_qualification_sidecar_path_v1,
)
from engine.campaigns.resident_prepared_owner_resource_audit_v1 import (
    CudaMemorySampleV1,
    MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION,
    PREPARE_ONLY_STAGES,
    ResidentPreparedOwnerResourceAuditRequestV1,
    execute_prepared_owner_resource_audit_v1,
    prepared_owner_configuration_sha256_v1,
)
from tools.scripts.eiel_campaign import _resume, build_parser


_DIGEST = "a" * 64
GIB = 1024**3
_HARDWARE = {
    "device_index": 0,
    "device_name": "test-cuda",
    "device_uuid": "GPU-test",
    "compute_capability": [7, 5],
    "total_memory_bytes": 8 * GIB,
    "cuda_runtime_version": "test-cuda-runtime",
    "torch_version": "test-torch",
}


def _sample(stage: str) -> CudaMemorySampleV1:
    index = PREPARE_ONLY_STAGES.index(stage)
    active = stage not in {"process_baseline", "teardown"}
    return CudaMemorySampleV1(
        stage=stage,
        elapsed_ns=index,
        allocated_bytes=20 if active else 10,
        reserved_bytes=30 if active else 10,
        peak_allocated_bytes=20 if active else 10,
        peak_reserved_bytes=30 if active else 10,
        free_bytes=7 * GIB if active else 8 * GIB,
        total_bytes=8 * GIB,
    )


def _request_factory(
    projection_sha256: str,
) -> ResidentPreparedOwnerResourceAuditRequestV1:
    return ResidentPreparedOwnerResourceAuditRequestV1(
        audit_id="audit",
        provider_id="provider",
        profile_id="profile",
        campaign_id="campaign",
        plan_sha256=_DIGEST,
        qualification_key_sha256=_DIGEST,
        execution_route_sha256=_DIGEST,
        route_qualification={
            "authority_state": "route_qualified",
            "receipt_path": "/tmp/test-route-qualification.json",
            "receipt_sha256": _DIGEST,
            "artifact_conformance_exact": True,
        },
        candidate_authority_sha256=_DIGEST,
        artifact_sha256s={"artifact": _DIGEST},
        configuration_sha256=_DIGEST,
        resource_estimator_id="test-estimator-v1",
        conservative_vram_projection_bytes=100,
        runtime_assumptions={},
        measurement_authorization_sha256=_DIGEST,
        hardware_identity=_HARDWARE,
        source_projection_root_sha256=projection_sha256,
        ordered_cell_ids_sha256=_DIGEST,
        batch_request_sha256=_DIGEST,
        batch_cells=1,
        resident_segment_generations=1,
        physical_window_generations=2,
        ticks_per_generation=1,
        resident_capacity=2,
        initial_population=1,
        agent_id_allocation_generation_count=2,
        declared_vram_budget_bytes=100,
        minimum_physical_headroom_fraction=0.01,
    )


def test_generic_lifecycle_has_no_executor_surface_and_always_releases(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    receipt = execute_prepared_owner_resource_audit_v1(
        request_factory=_request_factory,
        module_loader=lambda: calls.append("module") or "module",
        owner_preparer=lambda module: (
            calls.append(f"prepare:{module}") or {"owner": True}
        ),
        projection_exporter=lambda owner: (
            calls.append(f"project:{owner['owner']}") or {"source": 1},
            canonical_sha256({"source": 1}),
        ),
        projection_destination=tmp_path / "projection.bin",
        projection_format_id="test-v1",
        projection_serializer=lambda _value, path: (
            calls.append("serialize"),
            path.write_bytes(b"projection"),
        )[-1],
        stage_sampler=_sample,
        resource_releaser=lambda owner, module: calls.append(
            f"release:{owner['owner']}:{module}"
        ),
    )

    assert calls == [
        "module",
        "prepare:module",
        "project:True",
        "serialize",
        "release:True:module",
    ]
    assert receipt.executor_call_count == 0
    assert receipt.scientific_tick_count == 0


def test_generic_lifecycle_preserves_aliased_projection_and_configuration(
    tmp_path: Path,
) -> None:
    projection = {"metadata": {"generation": 0}, "rows": [1, 2, 3]}
    configuration = {"policy": {"family": "audit-only"}}
    expected_projection = {
        "metadata": {"generation": 0},
        "rows": [1, 2, 3],
    }
    expected_configuration = {"policy": {"family": "audit-only"}}
    owner = SimpleNamespace(
        source_projections=(projection,),
        configuration=configuration,
    )

    execute_prepared_owner_resource_audit_v1(
        request_factory=_request_factory,
        module_loader=lambda: "module",
        owner_preparer=lambda _module: owner,
        projection_exporter=lambda prepared: (
            prepared.source_projections,
            canonical_sha256(prepared.source_projections),
        ),
        projection_destination=tmp_path / "projection.bin",
        projection_format_id="test-v1",
        projection_serializer=lambda _value, path: path.write_bytes(b"projection"),
        stage_sampler=_sample,
    )

    assert projection == expected_projection
    assert configuration == expected_configuration
    assert owner.source_projections == (projection,)
    assert owner.configuration is configuration


def test_throwing_releaser_still_runs_gc_and_post_release_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.campaigns import resident_prepared_owner_resource_audit_v1 as audit

    events: list[str] = []
    monkeypatch.setattr(audit.gc, "collect", lambda: events.append("gc"))

    def fail_release(_owner: object, _module: object) -> None:
        events.append("release")
        raise RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        execute_prepared_owner_resource_audit_v1(
            request_factory=_request_factory,
            module_loader=lambda: "module",
            owner_preparer=lambda _module: {"owner": True},
            projection_exporter=lambda _owner: (
                {"source": 1},
                canonical_sha256({"source": 1}),
            ),
            projection_destination=tmp_path / "projection.bin",
            projection_format_id="test-v1",
            projection_serializer=lambda _value, path: path.write_bytes(b"projection"),
            stage_sampler=_sample,
            resource_releaser=fail_release,
            post_release_finalizer=lambda: events.append("finalize"),
        )

    assert events == ["release", "gc", "finalize"]


def _plan(tmp_path: Path) -> dict:
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}", encoding="utf-8")
    return {
        "campaign_id": "campaign",
        "plan_sha256": "1" * 64,
        "qualification_key_sha256": "2" * 64,
        "resolved_profile": ProductionEngineLifecycleV1.profile_id,
        "artifacts": {
            "cuda_extension": {"path": "/artifact", "sha256": "3" * 64},
            "qualification_receipt": {
                "path": str(qualification),
                "sha256": "4" * 64,
            },
            "reporting_receipt": {"path": "/reporting", "sha256": "5" * 64},
        },
        "execution": {
            "generations": 3,
            "ticks_per_generation": 80,
            "checkpoints": [3],
            "budgets": {"vram_bytes": 6 * GIB},
        },
        "resource_estimate": {"logical_trajectories": 2},
        "cells": [
            {"cell_id": "a", "parameters": {}, "seeds": {}},
            {"cell_id": "b", "parameters": {}, "seeds": {}},
        ],
    }


def _measurement_authorization(
    provider: ProductionEngineLifecycleV1,
    plan: dict,
    request: ResidentBatchRequestV1,
) -> dict:
    estimate = provider.estimate_resources(plan, (request,))
    contract = provider.execution_route_contract(plan, request)
    route = contract.execution_route_sha256
    route_receipt = {
        "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
        "authority_state": "route_qualified",
        "provider_id": contract.provider_id,
        "profile_id": contract.profile_id,
        "execution_route_sha256": route,
        "covered_obligations": list(contract.required_obligations),
        "artifact_sha256": plan["artifacts"]["cuda_extension"]["sha256"],
    }
    qualification_path = Path(
        plan["artifacts"]["qualification_receipt"]["path"]
    )
    sidecar = execution_route_qualification_sidecar_path_v1(
        qualification_path,
        route,
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(route_receipt), encoding="utf-8")
    execution = plan["execution"]
    first_end = provider._segment_ends(request, start=0)[0]
    allocation_generations = provider._agent_id_allocation_generation_count(
        request, start_generation=0
    )
    measurement_bootstrap_floor = math.ceil(
        _HARDWARE["total_memory_bytes"] * 0.04
    )
    bindings = {
        "provider_id": provider.provider_id,
        "profile_id": provider.profile_id,
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "qualification_key_sha256": plan["qualification_key_sha256"],
        "execution_route_sha256": route,
        "route_qualification": {
            "authority_state": "route_qualified",
            "receipt_path": str(sidecar.resolve()),
            "receipt_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "artifact_conformance_exact": True,
        },
        "artifact_sha256s": {
            name: row["sha256"] for name, row in plan["artifacts"].items()
        },
        "configuration_sha256": prepared_owner_configuration_sha256_v1(
            plan, request.payload()
        ),
        "resource_estimator_id": estimate["resource_estimator_id"],
        "conservative_vram_projection_bytes": estimate["vram_bytes_projection"],
        "runtime_assumptions": {},
        "hardware_identity": _HARDWARE,
        "ordered_cell_ids_sha256": canonical_sha256(list(request.cell_ids)),
        "batch_request_sha256": canonical_sha256(request.payload()),
        "dimensions": {
            "batch_cells": len(request.cell_ids),
            "resident_segment_generations": first_end,
            "physical_window_generations": 2,
            "ticks_per_generation": request.ticks_per_generation,
            "resident_capacity": 256,
            "initial_population": 64,
            "agent_id_allocation_generation_count": allocation_generations,
        },
        "policy": {
            "declared_vram_budget_bytes": execution["budgets"]["vram_bytes"],
            "minimum_physical_headroom_fraction": 0.04,
            "minimum_physical_headroom_bytes": 0,
            "minimum_residual_headroom_bytes": 0,
            "measurement_live_free_policy": {
                "mode": "exact_allocation_only_measurement_bootstrap_v2",
                "minimum_free_bytes": measurement_bootstrap_floor,
                "requires_projection_within_total_memory": False,
                "allows_projection_over_declared_budget": True,
                "zero_dispatch_only": True,
                "production_execution_admission_unchanged": True,
            },
            "teardown_tolerance_bytes": 0,
        },
    }
    unsigned = {
        "schema_version": MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION,
        "request_bindings": bindings,
    }
    return {**unsigned, "authorization_sha256": canonical_sha256(unsigned)}


def _provider_request(
    provider: ProductionEngineLifecycleV1,
    plan: dict,
) -> ResidentBatchRequestV1:
    return ResidentBatchRequestV1(
        profile_id=provider.profile_id,
        batch_id="batch-000",
        cell_ids=("a", "b"),
        generations=3,
        ticks_per_generation=80,
        checkpoint_milestones=(3,),
        parameter_arrays=(
            ("patch.capacity", (256, 256)),
            ("patch.initial_population", (64, 64)),
        ),
        seed_arrays=(),
        agent_id_allocation_generation_count=3,
        execution_route_sha256=(
            provider.execution_route_contract(plan).execution_route_sha256
        ),
    )


def test_patch_provider_uses_production_preparer_and_never_calls_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    projections = (
        {"cell": "a", "metadata": {}},
        {"cell": "b", "metadata": {}},
    )

    def runner_builder(cell, *_args, **_kwargs):
        calls.append(f"runner:{cell['cell_id']}")
        return SimpleNamespace(runner=f"runner-{cell['cell_id']}")

    def request_builder(runner, **_kwargs):
        calls.append(f"request:{runner}")
        return runner

    def preparer(**kwargs):
        calls.append("prepare")
        return SimpleNamespace(source_projections=kwargs["resume_projections"])

    def source_preparer(_plan, _request, context, **_kwargs):
        assert context.run_root == tmp_path / "audit"
        assert context.batch_root == tmp_path / "audit"
        calls.append("sources")
        return projections

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: calls.append("empty-cache"),
    )

    provider = ProductionEngineLifecycleV1(
        artifact_loader=lambda *_args: (
            calls.append("load"),
            SimpleNamespace(
                module="module",
                candidate_authority_sha256="6" * 64,
            ),
        )[-1],
        runner_builder=runner_builder,
        request_builder=request_builder,
        preparer=preparer,
        executor=lambda *_args, **_kwargs: pytest.fail("executor was called"),
        checkpoint_digest=lambda value: canonical_sha256(value),
        source_preparer=source_preparer,
    )
    plan = _plan(tmp_path)
    route = provider.execution_route_contract(plan).execution_route_sha256
    request = ResidentBatchRequestV1(
        profile_id=provider.profile_id,
        batch_id="batch-000",
        cell_ids=("a", "b"),
        generations=3,
        ticks_per_generation=80,
        checkpoint_milestones=(3,),
        parameter_arrays=(
            ("patch.capacity", (256, 256)),
            ("patch.initial_population", (64, 64)),
        ),
        seed_arrays=(),
        agent_id_allocation_generation_count=3,
        execution_route_sha256=route,
    )

    result = provider.measure_prepared_owner_resources_v1(
        plan,
        request,
        audit_root=tmp_path / "audit",
        audit_id="audit",
        measurement_authorization=_measurement_authorization(provider, plan, request),
        cuda_memory_detector=lambda: (1 * GIB, 8 * GIB),
        cuda_hardware_detector=lambda: _HARDWARE,
        stage_sampler=_sample,
        projection_serializer=lambda _value, path: path.write_bytes(b"projection"),
        resource_releaser=lambda _owner, _module: calls.append("release"),
    )

    assert result["executor_call_count"] == 0
    assert result["scientific_tick_count"] == 0
    assert result["request"]["batch_request_sha256"] == canonical_sha256(
        request.payload()
    )
    assert result["request"]["candidate_authority_sha256"] == "6" * 64
    assert calls == [
        "load",
        "sources",
        "empty-cache",
        "runner:a",
        "runner:b",
        "request:runner-a",
        "request:runner-b",
        "prepare",
        "release",
        "empty-cache",
    ]


@pytest.mark.parametrize(
    ("binding_id", "expected_capabilities"),
    (
        (
            "wdl_fixed_mlp_native_binding_v1",
            {
                "flat": False,
                "patch": True,
                "churn": True,
                "wdl": True,
                "jungle": False,
                "gru": False,
                "cuda": False,
            },
        ),
        (
            "wdl_jungle_eed_fixed_mlp_native_binding_v1",
            {
                "flat": False,
                "patch": True,
                "churn": True,
                "wdl": True,
                "jungle": True,
                "gru": False,
                "cuda": False,
                "eed": True,
            },
        ),
    ),
)
def test_prepared_owner_measurement_uses_resolved_mechanics_adapters(
    tmp_path: Path,
    binding_id: str,
    expected_capabilities: dict[str, bool],
) -> None:
    calls: list[str] = []
    projections = ({"cell": "a", "metadata": {}}, {"cell": "b", "metadata": {}})
    provider = ProductionEngineLifecycleV1(
        artifact_loader=lambda *_args: pytest.fail("default loader was used"),
        runner_builder=lambda *_args, **_kwargs: pytest.fail(
            "default runner was used"
        ),
        request_builder=lambda *_args, **_kwargs: pytest.fail(
            "default request builder was used"
        ),
        preparer=lambda **_kwargs: pytest.fail("default preparer was used"),
        checkpoint_digest=lambda value: canonical_sha256(value),
        source_preparer=lambda *_args, **_kwargs: pytest.fail(
            "default source preparer was used"
        ),
    )
    plan = _plan(tmp_path)

    def runner_builder(cell, *_args, **_kwargs):
        calls.append(f"resolved-runner:{cell['cell_id']}")
        return SimpleNamespace(runner=f"resolved-{cell['cell_id']}")

    def request_builder(runner, **_kwargs):
        calls.append(f"resolved-request:{runner}")
        return runner

    def preparer(**kwargs):
        calls.append("resolved-prepare")
        return SimpleNamespace(source_projections=kwargs["resume_projections"])

    def source_preparer(_plan, _request, _context, **kwargs):
        calls.append(
            "resolved-sources:"
            + getattr(kwargs["runner_builder"], "__name__", "anonymous")
        )
        assert kwargs["source_capability_mask"] == expected_capabilities
        return projections

    provider._resolved_execution_adapters = lambda _plan: {
        "artifact_loader": lambda *_args: SimpleNamespace(
            module="module",
            candidate_authority_sha256="7" * 64,
        ),
        "runner_builder": runner_builder,
        "request_builder": request_builder,
        "preparer": preparer,
        "source_preparer": source_preparer,
        "mechanics_binding": SimpleNamespace(
            binding_id=binding_id
        ),
    }
    request = _provider_request(provider, plan)

    result = provider.measure_prepared_owner_resources_v1(
        plan,
        request,
        audit_root=tmp_path / "audit",
        audit_id="audit",
        measurement_authorization=_measurement_authorization(provider, plan, request),
        cuda_memory_detector=lambda: (7 * GIB, 8 * GIB),
        cuda_hardware_detector=lambda: _HARDWARE,
        stage_sampler=_sample,
        projection_serializer=lambda _value, path: path.write_bytes(b"projection"),
    )

    assert result["request"]["candidate_authority_sha256"] == "7" * 64
    assert calls == [
        "resolved-sources:runner_builder",
        "resolved-runner:a",
        "resolved-runner:b",
        "resolved-request:resolved-a",
        "resolved-request:resolved-b",
        "resolved-prepare",
    ]




def test_provider_rejects_artifact_without_audit_authority(tmp_path: Path) -> None:
    provider = ProductionEngineLifecycleV1(
        artifact_loader=lambda *_args: SimpleNamespace(module="module"),
        runner_builder=lambda *_args, **_kwargs: None,
        request_builder=lambda *_args, **_kwargs: None,
        preparer=lambda **_kwargs: None,
        checkpoint_digest=lambda value: canonical_sha256(value),
        source_preparer=lambda *_args, **_kwargs: (),
    )
    plan = _plan(tmp_path)
    request = ResidentBatchRequestV1(
        profile_id=provider.profile_id,
        batch_id="batch-000",
        cell_ids=("a", "b"),
        generations=3,
        ticks_per_generation=80,
        checkpoint_milestones=(3,),
        parameter_arrays=(
            ("patch.capacity", (256, 256)),
            ("patch.initial_population", (64, 64)),
        ),
        seed_arrays=(),
        execution_route_sha256=(
            provider.execution_route_contract(plan).execution_route_sha256
        ),
    )
    with pytest.raises(CampaignProviderError, match="does not expose"):
        provider.measure_prepared_owner_resources_v1(
            plan,
            request,
            audit_root=tmp_path / "audit",
            audit_id="audit",
            measurement_authorization=_measurement_authorization(
                provider, plan, request
            ),
            cuda_memory_detector=lambda: (7 * GIB, 8 * GIB),
            cuda_hardware_detector=lambda: _HARDWARE,
            stage_sampler=_sample,
            projection_serializer=lambda _value, path: path.write_bytes(b"x"),
        )


def test_provider_rejects_tampered_route_before_artifact_load(tmp_path: Path) -> None:
    loaded = False

    def loader(*_args):
        nonlocal loaded
        loaded = True
        return SimpleNamespace(module="module", candidate_authority_sha256="6" * 64)

    provider = ProductionEngineLifecycleV1(artifact_loader=loader)
    plan = _plan(tmp_path)
    request = ResidentBatchRequestV1(
        profile_id=provider.profile_id,
        batch_id="batch-000",
        cell_ids=("a", "b"),
        generations=3,
        ticks_per_generation=80,
        checkpoint_milestones=(3,),
        parameter_arrays=(
            ("patch.capacity", (256, 256)),
            ("patch.initial_population", (64, 64)),
        ),
        seed_arrays=(),
        execution_route_sha256=(
            provider.execution_route_contract(plan).execution_route_sha256
        ),
    )
    with pytest.raises(CampaignProviderError, match="execution-route binding"):
        provider.measure_prepared_owner_resources_v1(
            plan,
            replace(request, execution_route_sha256="f" * 64),
            audit_root=tmp_path / "audit",
            audit_id="audit",
            measurement_authorization={},
            stage_sampler=_sample,
        )
    assert loaded is False


def test_provider_rechecks_route_hardware_and_live_free_before_allocation(
    tmp_path: Path,
) -> None:
    provider = ProductionEngineLifecycleV1()
    plan = _plan(tmp_path)
    request = _provider_request(provider, plan)
    authorization = _measurement_authorization(provider, plan, request)
    sidecar = Path(
        authorization["request_bindings"]["route_qualification"][
            "receipt_path"
        ]
    )
    sidecar.unlink()

    with pytest.raises(CampaignProviderError, match="route qualification"):
        provider.measure_prepared_owner_resources_v1(
            plan,
            request,
            audit_root=tmp_path / "unqualified",
            audit_id="unqualified",
            measurement_authorization=authorization,
            cuda_memory_detector=lambda: (7 * GIB, 8 * GIB),
            cuda_hardware_detector=lambda: _HARDWARE,
            stage_sampler=_sample,
        )
    assert not (tmp_path / "unqualified").exists()

    authorization = _measurement_authorization(provider, plan, request)
    with pytest.raises(CampaignProviderError, match="hardware differs"):
        provider.measure_prepared_owner_resources_v1(
            plan,
            request,
            audit_root=tmp_path / "stale-hardware",
            audit_id="stale-hardware",
            measurement_authorization=authorization,
            cuda_memory_detector=lambda: (7 * GIB, 8 * GIB),
            cuda_hardware_detector=lambda: {
                **_HARDWARE,
                "device_uuid": "GPU-other",
            },
            stage_sampler=_sample,
        )
    assert not (tmp_path / "stale-hardware").exists()

    with pytest.raises(CampaignProviderError, match="bootstrap floor"):
        provider.measure_prepared_owner_resources_v1(
            plan,
            request,
            audit_root=tmp_path / "stale-free",
            audit_id="stale-free",
            measurement_authorization=authorization,
            cuda_memory_detector=lambda: (1, 8 * GIB),
            cuda_hardware_detector=lambda: _HARDWARE,
            stage_sampler=_sample,
        )
    assert not (tmp_path / "stale-free").exists()


def test_cli_exposes_typed_measurement_command() -> None:
    args = build_parser().parse_args(
        [
            "measure-prepared-owner",
            "campaign.yaml",
            "--batch-id",
            "batch-000",
            "--output-root",
            "/tmp/audit",
        ]
    )
    assert args.command == "measure-prepared-owner"
    assert args.batch_id == "batch-000"

    promoted = build_parser().parse_args(
        [
            "preflight",
            "campaign.yaml",
            "--prepared-owner-receipt",
            "audit-receipt.json",
        ]
    )
    assert promoted.command == "preflight"
    assert promoted.prepared_owner_receipt == [Path("audit-receipt.json")]

    resume = build_parser().parse_args(
        [
            "resume",
            "/tmp/run",
            "--prepared-owner-receipt",
            "audit-receipt.json",
        ]
    )
    assert resume.prepared_owner_receipt == Path("audit-receipt.json")


def test_promoted_resume_requires_explicit_receipt_and_fresh_preflight(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    (tmp_path / "execution_plan.json").write_text(
        json.dumps(plan),
        encoding="utf-8",
    )
    (tmp_path / "preflight_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_preflight_v1",
                "plan_sha256": plan["plan_sha256"],
                "ready_for_execution": True,
                "resources": {"vram_bytes_upper_bound": None},
                "prepared_owner_resource_audit": {
                    "receipt_promotion": {"receipt_sha256": _DIGEST}
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CampaignError, match="explicit prepared-owner"):
        _resume(tmp_path)
