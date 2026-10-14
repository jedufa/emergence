from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns import execution_front_door_v1 as front_door
from engine.campaigns import orchestration as campaign_orchestration
from engine.campaigns.artifact_lifecycle import (
    build_artifact_disposition_manifest_v1,
    lifecycle_profile_contract_v1,
)
from engine.campaigns.baseline_v1 import BASELINE_PROFILE_ID, BaselineDeclarationV1
from engine.campaigns.execution_front_door_v1 import (
    ExecutionFrontDoorError,
    ExecutionRequestV1,
    ExecutionRequestV2,
    ResourceWaitPolicyV1,
    evaluate_qualification_profile_v1,
    execute_bound_campaign_plan_v2,
    execute_request_v1,
    execute_request_v2,
    verify_mechanical_execution_v1,
)
from engine.campaigns.canonical import canonical_sha256
from engine.tensor_world.c7_qualification import (
    TensorWorldC7EvidenceReceiptV1,
    TensorWorldC7QualificationFreezeV1,
    c7_freeze_evidence_bindings_v1,
)
from engine.campaigns.providers import ExecutionProviderRegistryV1, ResidentBatchRequestV1


def _request(
    tmp_path: Path,
    *,
    wait: ResourceWaitPolicyV1 | None = None,
    restore_process_isolation: str = "inline",
) -> ExecutionRequestV1:
    spec = tmp_path / "campaign.yaml"
    spec.write_text("placeholder", encoding="utf-8")
    return ExecutionRequestV1(
        purpose="validation",
        campaign_spec=str(spec),
        output_root=str(tmp_path / "run"),
        resource_wait=wait or ResourceWaitPolicyV1(),
        restore_process_isolation=restore_process_isolation,
    )


def _v2_request(
    tmp_path: Path,
    *,
    wait: ResourceWaitPolicyV1 | None = None,
    restore_process_isolation: str = "inline",
) -> ExecutionRequestV2:
    spec = tmp_path / "campaign.yaml"
    spec.write_text("placeholder", encoding="utf-8")
    return ExecutionRequestV2(
        purpose="validation",
        campaign_spec=str(spec),
        output_root=str(tmp_path / "run"),
        resource_wait=wait or ResourceWaitPolicyV1(),
        restore_process_isolation=restore_process_isolation,
        baseline=BaselineDeclarationV1.from_value(
            {
                "profile_id": BASELINE_PROFILE_ID,
                "intent": "development",
                "deviations": [
                    {
                        "requirement_id": requirement_id,
                        "reason": "skeletal front-door unit fixture omits production authority",
                    }
                    for requirement_id in (
                        "evolution.full_evolution",
                        "runtime.production_cuda",
                        "time.absolute_native_tick",
                        "checkpoint.native_streamed",
                        "checkpoint.acs_two_slot",
                        "artifacts.lifecycle_contract",
                        "science.wrc_variability",
                        "authority.generation_runtime",
                        "authority.device_fallback",
                    )
                ],
            }
        ),
    )


def _v2_lifecycle_request(tmp_path: Path, *, purpose: str = "validation") -> ExecutionRequestV2:
    request = _v2_request(tmp_path)
    baseline = request.baseline.payload()
    baseline["deviations"] = [
        row
        for row in baseline["deviations"]
        if row["requirement_id"] != "artifacts.lifecycle_contract"
    ]
    return replace(
        request,
        purpose=purpose,
        baseline=BaselineDeclarationV1.from_value(baseline),
    )


def _fake_plan() -> dict[str, object]:
    return {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "front-door-test",
        "resolved_profile": "test-profile",
        "qualification_key": {},
        "qualification_key_sha256": "qualification-key",
        "execution": {"budgets": {}},
        "output": {"runtime_root": "/unused"},
        "artifacts": {},
        "resource_estimate": {},
        "cells": [],
        "physical_batches": [],
        "plan_sha256": "compiled-plan",
    }


def _fake_preflight(plan: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "eiel_campaign_preflight_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "ready_for_execution": True,
        "reasons": [],
        "execution_authority_reasons": [],
        "artifacts": [],
    }


def test_full_evolution_v2_engineering_scope_projection_matches_dispatch() -> None:
    plan = {
        **_fake_plan(),
        "resolved_profile": (
            "eiel_full_evolution_cuda_workhorse_v2__w64h64p1024c2048"
        ),
    }
    assert (
        front_door._execution_authority_scope_v1(
            purpose="engineering",
            plan=plan,
        )
        == "engineering_unqualified_real_use"
    )
    assert (
        front_door._execution_authority_scope_v1(
            purpose="benchmark",
            plan=plan,
        )
        == "engineering_unqualified_real_use"
    )
    assert (
        front_door._execution_authority_scope_v1(
            purpose="qualprep",
            plan=plan,
        )
        == "qualprep_development_only"
    )
    assert (
        front_door._execution_authority_scope_v1(
            purpose="qualification",
            plan=plan,
        )
        == "qualified_execution"
    )


def test_cpu_native_engineering_scope_is_unqualified_real_use() -> None:
    plan = {
        **_fake_plan(),
        "resolved_profile": "eiel-production-runtime-v1-cpu-native",
    }

    assert (
        front_door._execution_authority_scope_v1(
            purpose="engineering",
            plan=plan,
        )
        == "engineering_unqualified_real_use"
    )


def _fake_execution(
    plan: dict[str, object], *, run_root: Path, **_: object
) -> dict[str, object]:
    return {
        "schema_version": "eiel_campaign_execution_receipt_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": "test-provider",
        "status": "completed",
    }


def _fake_lifecycle_plan(plan: dict[str, object]) -> dict[str, object]:
    return {
        **plan,
        "artifact_lifecycle": lifecycle_profile_contract_v1(
            "QUALIFICATION",
            checkpoint_schedule=(),
            checkpoint_authority="absolute_tick",
        ),
    }


def _exact_plan(plan: dict[str, object]) -> dict[str, object]:
    core = dict(plan)
    core.pop("plan_sha256", None)
    return {**core, "plan_sha256": canonical_sha256(core)}


def _fake_execution_with_terminal_lifecycle(
    plan: dict[str, object], *, run_root: Path, **kwargs: object
) -> dict[str, object]:
    campaign_orchestration.atomic_write_json(
        run_root / "run_passport.json",
        {
            "schema_version": "eiel_test_run_passport_v1",
            "plan_sha256": plan["plan_sha256"],
        },
    )
    manifest = build_artifact_disposition_manifest_v1(
        plan,
        run_root=run_root,
        terminal_status="completed",
        live_reference_status="none",
    )
    campaign_orchestration.atomic_write_json(
        run_root / "artifact_disposition_manifest.json", manifest
    )
    return _fake_execution(plan, run_root=run_root, **kwargs)


def _passing_mechanical(**_: object) -> dict[str, object]:
    return {
        "schema_version": "eiel_execution_mechanical_verification_v1",
        "status": "PASS",
        "checks": [],
        "errors": [],
        "semantic_questions": [],
    }


def test_request_roundtrip_is_canonical_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    payload = request.payload()
    roundtrip = ExecutionRequestV1.from_mapping(payload)

    assert roundtrip.payload() == payload
    assert request.request_sha256 == payload["request_sha256"]
    assert "restore_process_isolation" not in payload
    with pytest.raises(ExecutionFrontDoorError, match="unsupported fields"):
        ExecutionRequestV1.from_mapping({**payload, "extra": True})
    with pytest.raises(ExecutionFrontDoorError, match="restore_process_isolation"):
        ExecutionRequestV1.from_mapping(
            {**payload, "restore_process_isolation": "same_thread"}
        )
    with pytest.raises(ExecutionFrontDoorError, match="restore_process_isolation"):
        _request(tmp_path, restore_process_isolation="same_thread")


def test_benchmark_purpose_is_a_canonical_execution_purpose(tmp_path: Path) -> None:
    spec = tmp_path / "campaign.yaml"
    spec.write_text("placeholder", encoding="utf-8")
    request = ExecutionRequestV1(
        purpose="benchmark",
        campaign_spec=str(spec),
        output_root=str(tmp_path / "run"),
    )
    payload = request.payload()
    assert payload["purpose"] == "benchmark"
    assert ExecutionRequestV1.from_mapping(payload).purpose == "benchmark"
    assert payload["purpose"] in front_door.EXECUTION_PURPOSES


def test_front_door_composes_preflight_and_execution_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_lifecycle_request(tmp_path)
    lifecycle_reseals: list[tuple[dict[str, object], Path]] = []
    plan = _fake_lifecycle_plan(_fake_plan())
    real_reseal = campaign_orchestration.reseal_terminal_lifecycle_inventory_v1
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )

    def reseal(
        reseal_plan: dict[str, object], *, run_root: Path, **kwargs: object
    ) -> dict[str, object]:
        lifecycle_reseals.append((reseal_plan, run_root))
        return real_reseal(reseal_plan, run_root=run_root, **kwargs)

    monkeypatch.setattr(
        front_door,
        "reseal_terminal_lifecycle_inventory_v1",
        reseal,
    )
    monkeypatch.setattr(
        campaign_orchestration,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda plan, **_: _fake_preflight(plan),
        execute_fn=_fake_execution_with_terminal_lifecycle,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert result["ok"] is True, json.dumps(result, indent=2, sort_keys=True)
    assert result["result_class"] == "SUCCESS"
    assert result["execution_receipt"]["status"] == "completed"
    assert (tmp_path / "run" / "execution_request.json").is_file()
    assert (tmp_path / "run" / "preflight_receipt.json").is_file()
    assert (tmp_path / "run" / "execution_result.json").is_file()
    assert len(lifecycle_reseals) == 1
    resealed_plan, resealed_root = lifecycle_reseals[0]
    assert resealed_root == (tmp_path / "run").resolve()
    assert resealed_plan["plan_sha256"] == result["plan_sha256"]
    assert (tmp_path / "run" / front_door.TERMINAL_LIFECYCLE_COMMIT_NAME).is_file()


def test_bound_campaign_plan_uses_canonical_front_door_and_seals_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = replace(_v2_lifecycle_request(tmp_path, purpose="engineering"))
    plan = _exact_plan(_fake_lifecycle_plan(_fake_plan()))
    source_lineage = {
        "campaign_spec_sha256": "source-spec-sha256",
        "authoring": {"schema_version": "eiel_science_campaign_v1"},
    }
    science_lineage = {
        "bundle_content_sha256": "bundle-content-sha256",
        "admission_receipt_sha256": "science-admission-sha256",
    }
    hook_observations: list[tuple[bool, bool]] = []

    def before_execute() -> None:
        hook_observations.append(
            (
                (tmp_path / "run" / "execution_request.json").is_file(),
                (tmp_path / "run" / "execution_plan.json").is_file(),
            )
        )

    monkeypatch.setattr(
        campaign_orchestration,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )

    result = execute_bound_campaign_plan_v2(
        request,
        plan,
        source_lineage=source_lineage,
        science_lineage=science_lineage,
        repo_root=tmp_path,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=_fake_execution_with_terminal_lifecycle,
        mechanical_verifier_fn=_passing_mechanical,
        canopy_start_fn=lambda **_: {"relays": []},
        before_execute_fn=before_execute,
    )

    assert result["ok"] is True, json.dumps(result, indent=2, sort_keys=True)
    identity = result["source_identity"]
    assert identity["source_lineage"] == source_lineage
    assert identity["science_lineage"] == science_lineage
    assert identity["compiled_plan_sha256"] == plan["plan_sha256"]
    assert identity["bound_plan_sha256"] == result["plan_sha256"]
    assert hook_observations == [(True, True)]
    request_receipt = json.loads(
        (tmp_path / "run" / "execution_request.json").read_text(encoding="utf-8")
    )
    assert request_receipt["source_identity"] == identity
    assert request_receipt["bound_plan_sha256"] == result["plan_sha256"]
    assert json.loads(
        (tmp_path / "run" / "execution_plan.json").read_text(encoding="utf-8")
    )["plan_sha256"] == result["plan_sha256"]


def test_bound_campaign_plan_rejects_unbound_science_and_tampered_plan(
    tmp_path: Path,
) -> None:
    request = _v2_lifecycle_request(tmp_path, purpose="engineering")
    plan = _exact_plan(_fake_lifecycle_plan(_fake_plan()))
    with pytest.raises(ExecutionFrontDoorError, match="scientific-execution lineage"):
        execute_bound_campaign_plan_v2(
            replace(request, purpose="simulation"),
            plan,
            source_lineage={},
            science_lineage={},
            repo_root=tmp_path,
        )
    with pytest.raises(ExecutionFrontDoorError, match="plan_sha256"):
        execute_bound_campaign_plan_v2(
            request,
            {**plan, "campaign_id": "tampered"},
            source_lineage={},
            science_lineage={},
            repo_root=tmp_path,
        )


def test_bound_campaign_plan_accepts_generic_science_instrument_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_lifecycle_request(tmp_path, purpose="simulation")
    plan = _fake_lifecycle_plan(_fake_plan())
    plan.update(
        {
            "science_use_class": "SCIENTIFIC_EXECUTION",
            "science_materialization": {
                "science_use_class": "SCIENTIFIC_EXECUTION",
                "observation_mode": "COMPACT_SIDECAR_V1",
            },
        }
    )
    plan = _exact_plan(plan)
    science_lineage = {
        "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
        "science_use_class": "SCIENTIFIC_EXECUTION",
        "bundle_content_sha256": "bundle-content-sha256",
    }
    monkeypatch.setattr(
        campaign_orchestration,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )

    result = execute_bound_campaign_plan_v2(
        request,
        plan,
        source_lineage={"campaign_spec_sha256": "source-spec-sha256"},
        science_lineage=science_lineage,
        repo_root=tmp_path,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=_fake_execution_with_terminal_lifecycle,
        mechanical_verifier_fn=_passing_mechanical,
        canopy_start_fn=lambda **_: {"relays": []},
    )

    assert result["ok"] is True
    assert result["source_identity"]["science_lineage"] == science_lineage


def test_front_door_finalizes_workhorse_row_after_scheduler_terminal_and_inventories_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_lifecycle_request(tmp_path, purpose="qualprep")
    profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"
    plan = _fake_lifecycle_plan(
        {**_fake_plan(), "resolved_profile": profile_id}
    )
    row_path = (
        (tmp_path / "run").resolve()
        / "batches/batch-0000/workhorse-technical-row-receipt.json"
    )
    finalized: list[dict[str, object]] = []

    class Provider:
        provider_id = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"

        def finalize_attempt_terminal_v2(self, **values: object) -> dict[str, str]:
            terminal = values["canonical_terminal_binding"]
            assert isinstance(terminal, dict)
            assert terminal["attempt_terminal_status"] == "COMPLETED"
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_text('{"terminalized":true}\n', encoding="utf-8")
            finalized.append(dict(values))
            return {
                "row_receipt_path": str(row_path),
                "row_receipt_file_sha256": sha256(row_path.read_bytes()).hexdigest(),
                "row_receipt_content_sha256": "a" * 64,
                "attempt_id": "attempt-workhorse-host",
            }

    provider = Provider()
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    shadow = SimpleNamespace(
        binding={"authority": "SHADOW_ONLY_TEST"},
        projection={"schema_version": "shadow-test"},
        bound_execution_request={"schema_version": "bound-test"},
    )
    monkeypatch.setattr(
        front_door,
        "_persist_cuda_full_evolution_v2_shadow_v1",
        lambda **_: shadow,
    )
    monkeypatch.setattr(
        front_door,
        "_bind_cuda_full_evolution_v2_attempt_authority_v1",
        lambda **_: {"execution_authority_granted": True},
    )
    terminal_binding = {
        "schema_version": "eiel_workhorse_canonical_parent_terminal_binding_v2",
        "attempt_terminal_status": "COMPLETED",
    }

    class Adapter:
        @staticmethod
        def terminalize_attempt_authority_v1(**_: object) -> dict[str, object]:
            return {
                "status": "TERMINAL",
                "workhorse_terminal_binding": terminal_binding,
            }

    lifecycle_calls: list[dict[str, object]] = []
    real_reseal = campaign_orchestration.reseal_terminal_lifecycle_inventory_v1

    def reseal_row(
        reseal_plan: dict[str, object], **kwargs: object
    ) -> dict[str, object]:
        lifecycle_calls.append({"plan": reseal_plan, **kwargs})
        return real_reseal(reseal_plan, **kwargs)

    monkeypatch.setattr(
        front_door, "reseal_terminal_lifecycle_inventory_v1", reseal_row
    )
    monkeypatch.setattr(
        campaign_orchestration,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )
    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        provider_registry=registry,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=_fake_execution_with_terminal_lifecycle,
        mechanical_verifier_fn=_passing_mechanical,
        cuda_full_evolution_v2_crca_adapter=Adapter(),
    )

    assert result["ok"] is True, json.dumps(result, indent=2, sort_keys=True)
    assert len(finalized) == 1
    assert result["workhorse_terminalized_row"]["row_receipt_path"] == str(
        row_path
    )
    additional = lifecycle_calls[0]["additional_artifacts"]
    assert additional == {
        "batches/batch-0000/workhorse-technical-row-receipt.json": {
            "artifact_family": "qualification.workhorse_technical_row_v2",
            "retention_class": "R3",
            "owner_writer": (
                "cuda_full_evolution_workhorse_v2_single_row_owner"
            ),
            "release_predicate": (
                "technical row retained for offline matrix aggregation"
            ),
        }
    }
    marker = json.loads(
        (tmp_path / "run" / front_door.TERMINAL_LIFECYCLE_COMMIT_NAME).read_text(
            encoding="utf-8"
        )
    )
    row_commit = next(
        item
        for item in marker["terminal_artifacts"]
        if item["path"]
        == "batches/batch-0000/workhorse-technical-row-receipt.json"
    )
    assert row_commit["retention_class"] == "R3"
    assert marker["storage_receipt"]["file_sha256"] == sha256(
        (tmp_path / "run" / "storage_disposition_receipt.json").read_bytes()
    ).hexdigest()


def test_failed_workhorse_execution_terminalizes_at_last_committed_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = replace(_v2_request(tmp_path), purpose="qualprep")
    profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"
    plan = {**_fake_plan(), "resolved_profile": profile_id}

    class Provider:
        provider_id = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"

    registry = ExecutionProviderRegistryV1()
    registry.register(Provider())
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    shadow = SimpleNamespace(
        binding={"authority": "SHADOW_ONLY_TEST"},
        projection={
            "layers": {
                "semantic": {
                    "payload": {"absolute_start_next_tick": 17}
                }
            }
        },
        bound_execution_request={"schema_version": "bound-test"},
    )
    monkeypatch.setattr(
        front_door,
        "_persist_cuda_full_evolution_v2_shadow_v1",
        lambda **_: shadow,
    )
    monkeypatch.setattr(
        front_door,
        "_bind_cuda_full_evolution_v2_attempt_authority_v1",
        lambda **_: {"execution_authority_granted": True},
    )
    terminal_calls: list[dict[str, object]] = []

    class Adapter:
        @staticmethod
        def terminalize_attempt_authority_v1(**values: object) -> dict[str, object]:
            terminal_calls.append(dict(values))
            return {"status": "TERMINAL"}

    def fail_execution(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("host provider failed before canonical commit")

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        provider_registry=registry,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=fail_execution,
        mechanical_verifier_fn=_passing_mechanical,
        canopy_start_fn=lambda **_: {"relays": []},
        cuda_full_evolution_v2_crca_adapter=Adapter(),
    )
    assert result["ok"] is False
    assert len(terminal_calls) == 1
    failed_receipt = terminal_calls[0]["execution_receipt"]
    assert isinstance(failed_receipt, dict)
    assert failed_receipt["frontier_semantics"] == (
        "last_canonically_committed_frontiers"
    )
    assert set(failed_receipt["crca_final_frontiers"].values()) == {17}


def test_attempt_admission_unavailable_retries_instead_of_replaying_sticky_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = replace(_v2_request(tmp_path), purpose="qualprep")
    profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"
    plan = {**_fake_plan(), "resolved_profile": profile_id}

    class Provider:
        provider_id = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"

    registry = ExecutionProviderRegistryV1()
    registry.register(Provider())
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    shadow = SimpleNamespace(
        binding={"authority": "SHADOW_ONLY_TEST"},
        projection={
            "layers": {
                "semantic": {"payload": {"absolute_start_next_tick": 29}}
            }
        },
        bound_execution_request={"schema_version": "bound-test"},
    )
    monkeypatch.setattr(
        front_door,
        "_persist_cuda_full_evolution_v2_shadow_v1",
        lambda **_: shadow,
    )
    admission_calls = 0

    def bind_attempt(**_: object) -> dict[str, object]:
        nonlocal admission_calls
        admission_calls += 1
        if admission_calls == 1:
            return {
                "execution_authority_granted": False,
                "reason_codes": ["TEST_RETRYABLE_ATTEMPT_ADMISSION"],
            }
        return {"execution_authority_granted": True}

    monkeypatch.setattr(
        front_door,
        "_bind_cuda_full_evolution_v2_attempt_authority_v1",
        bind_attempt,
    )
    terminal_calls: list[dict[str, object]] = []

    class Adapter:
        @staticmethod
        def terminalize_attempt_authority_v1(**values: object) -> dict[str, object]:
            terminal_calls.append(dict(values))
            return {"status": "TERMINAL"}

    execution_calls = 0

    def fail_execution(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal execution_calls
        execution_calls += 1
        raise RuntimeError("retry reached execution")

    kwargs = {
        "repo_root": tmp_path,
        "provider_registry": registry,
        "preflight_fn": lambda compiled, **_: _fake_preflight(compiled),
        "execute_fn": fail_execution,
        "mechanical_verifier_fn": _passing_mechanical,
        "canopy_start_fn": lambda **_: {"relays": []},
        "cuda_full_evolution_v2_crca_adapter": Adapter(),
    }
    first = execute_request_v2(request, **kwargs)
    assert first["result_class"] == "ATTEMPT_ADMISSION_UNAVAILABLE"
    assert execution_calls == 0

    second = execute_request_v2(request, **kwargs)
    assert second["result_class"] == "EXECUTION_FAILED"
    assert "retry reached execution" in second["warnings"][-1]
    assert admission_calls == 2
    assert execution_calls == 1
    assert len(terminal_calls) == 1


def test_runtime_headroom_failure_terminalizes_once_and_replay_requires_fresh_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = replace(_v2_request(tmp_path), purpose="qualprep")
    profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"
    plan = {**_fake_plan(), "resolved_profile": profile_id}

    class Provider:
        provider_id = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"

    registry = ExecutionProviderRegistryV1()
    registry.register(Provider())
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    shadow = SimpleNamespace(
        binding={"authority": "SHADOW_ONLY_TEST"},
        projection={
            "layers": {
                "semantic": {"payload": {"absolute_start_next_tick": 23}}
            }
        },
        bound_execution_request={"schema_version": "bound-test"},
    )
    monkeypatch.setattr(
        front_door,
        "_persist_cuda_full_evolution_v2_shadow_v1",
        lambda **_: shadow,
    )
    monkeypatch.setattr(
        front_door,
        "_bind_cuda_full_evolution_v2_attempt_authority_v1",
        lambda **_: {"execution_authority_granted": True},
    )
    terminal_calls: list[dict[str, object]] = []
    execution_calls = 0

    class Adapter:
        @staticmethod
        def terminalize_attempt_authority_v1(**values: object) -> dict[str, object]:
            terminal_calls.append(dict(values))
            return {"status": "TERMINAL"}

    def reject_headroom(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal execution_calls
        execution_calls += 1
        raise RuntimeError(
            "WORKHORSE_RUNTIME_HEADROOM_REJECTED: no dispatch was committed"
        )

    kwargs = {
        "repo_root": tmp_path,
        "provider_registry": registry,
        "preflight_fn": lambda compiled, **_: _fake_preflight(compiled),
        "execute_fn": reject_headroom,
        "mechanical_verifier_fn": _passing_mechanical,
        "cuda_full_evolution_v2_crca_adapter": Adapter(),
    }
    first = execute_request_v2(request, **kwargs)
    replay = execute_request_v2(request, **kwargs)
    assert first == replay
    assert first["result_class"] == "WORKHORSE_RUNTIME_HEADROOM_REJECTED"
    assert first["resource_boundary"] == "RUNTIME_VRAM_HEADROOM"
    assert first["retry_contract"] == "FRESH_EXECUTION_REQUEST_V2_REQUIRED"
    assert execution_calls == 1
    assert len(terminal_calls) == 1
    failed_frontiers = terminal_calls[0]["execution_receipt"]
    assert isinstance(failed_frontiers, dict)
    assert set(failed_frontiers["crca_final_frontiers"].values()) == {23}


def test_front_door_inventories_terminalized_calibration_as_r3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_lifecycle_request(tmp_path, purpose="qualprep")
    profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"
    plan = _fake_lifecycle_plan(
        {**_fake_plan(), "resolved_profile": profile_id}
    )
    receipt_path = (
        (tmp_path / "run").resolve()
        / "batches/batch-0000/workhorse-device-envelope-calibration-receipt.json"
    )

    class Provider:
        provider_id = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"

        def finalize_attempt_terminal_v2(self, **_: object) -> dict[str, object]:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text('{"terminalized":true}\n', encoding="utf-8")
            return {
                "schema_version": "eiel_workhorse_terminalized_operation_binding_v2",
                "operation": "DEVICE_ENVELOPE_CALIBRATION",
                "final_receipt_path": str(receipt_path),
                "final_receipt_file_sha256": sha256(
                    receipt_path.read_bytes()
                ).hexdigest(),
                "final_receipt_content_sha256": "a" * 64,
                "attempt_id": "attempt-calibration-host",
                "lifecycle_binding": {
                    "artifact_family": (
                        "qualification.workhorse_device_envelope_calibration_v2"
                    ),
                    "retention_class": "R3",
                    "owner_writer": (
                        "cuda_full_evolution_workhorse_v2_single_operation_owner"
                    ),
                    "release_predicate": (
                        "technical calibration retained for exact row admission"
                    ),
                },
            }

    registry = ExecutionProviderRegistryV1()
    registry.register(Provider())
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    monkeypatch.setattr(
        front_door,
        "_persist_cuda_full_evolution_v2_shadow_v1",
        lambda **_: SimpleNamespace(
            binding={"authority": "SHADOW_ONLY_TEST"},
            projection={"schema_version": "shadow-test"},
            bound_execution_request={"schema_version": "bound-test"},
        ),
    )
    monkeypatch.setattr(
        front_door,
        "_bind_cuda_full_evolution_v2_attempt_authority_v1",
        lambda **_: {"execution_authority_granted": True},
    )

    class Adapter:
        @staticmethod
        def terminalize_attempt_authority_v1(**_: object) -> dict[str, object]:
            return {
                "status": "TERMINAL",
                "workhorse_terminal_binding": {
                    "schema_version": (
                        "eiel_workhorse_canonical_parent_terminal_binding_v2"
                    ),
                    "attempt_terminal_status": "COMPLETED",
                },
            }

    lifecycle_calls: list[dict[str, object]] = []
    real_reseal = campaign_orchestration.reseal_terminal_lifecycle_inventory_v1

    def reseal_calibration(
        reseal_plan: dict[str, object], **kwargs: object
    ) -> dict[str, object]:
        lifecycle_calls.append({"plan": reseal_plan, **kwargs})
        return real_reseal(reseal_plan, **kwargs)

    monkeypatch.setattr(
        front_door,
        "reseal_terminal_lifecycle_inventory_v1",
        reseal_calibration,
    )
    monkeypatch.setattr(
        campaign_orchestration,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )
    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        provider_registry=registry,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=_fake_execution_with_terminal_lifecycle,
        mechanical_verifier_fn=_passing_mechanical,
        canopy_start_fn=lambda **_: {"relays": []},
        cuda_full_evolution_v2_crca_adapter=Adapter(),
    )
    assert result["ok"] is True
    assert result["workhorse_terminalized_calibration"]["operation"] == (
        "DEVICE_ENVELOPE_CALIBRATION"
    )
    relative = "batches/batch-0000/workhorse-device-envelope-calibration-receipt.json"
    assert lifecycle_calls[0]["additional_artifacts"] == {
        relative: {
            "artifact_family": (
                "qualification.workhorse_device_envelope_calibration_v2"
            ),
            "retention_class": "R3",
            "owner_writer": (
                "cuda_full_evolution_workhorse_v2_single_operation_owner"
            ),
            "release_predicate": (
                "technical calibration retained for exact row admission"
            ),
        }
    }


def test_front_door_accepts_terminalized_indexed_continuation_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ACS2 operation must be routed through the generic terminalizer."""

    request = _v2_request(tmp_path, restore_process_isolation="inline")
    profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"
    plan = {**_fake_plan(), "resolved_profile": profile_id}
    receipt_path = (
        (tmp_path / "run").resolve()
        / "batches/batch-0000/workhorse-terminalized-indexed-continuation-receipt.json"
    )

    class Provider:
        provider_id = "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        profile_id = "eiel_full_evolution_cuda_workhorse_v2__host-shape"

        def finalize_attempt_terminal_v2(self, **_: object) -> dict[str, object]:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text('{"terminalized":true}\n', encoding="utf-8")
            return {
                "schema_version": "eiel_workhorse_terminalized_operation_binding_v2",
                "operation": "INDEXED_ACS_CONTINUATION",
                "final_receipt_path": str(receipt_path),
                "final_receipt_file_sha256": sha256(
                    receipt_path.read_bytes()
                ).hexdigest(),
                "final_receipt_content_sha256": "a" * 64,
                "indexed_continuation_receipt_path": str(receipt_path),
                "attempt_id": "attempt-indexed-continuation-host",
                "lifecycle_binding": {
                    "artifact_family": "science.workhorse_indexed_acs_continuation_v2",
                    "retention_class": "R3",
                    "owner_writer": (
                        "cuda_full_evolution_workhorse_v2_single_operation_owner"
                    ),
                    "release_predicate": (
                        "technical continuation retained; no science authority"
                    ),
                },
            }

    registry = ExecutionProviderRegistryV1()
    registry.register(Provider())
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    monkeypatch.setattr(
        front_door,
        "_persist_cuda_full_evolution_v2_shadow_v1",
        lambda **_: SimpleNamespace(
            binding={"authority": "SHADOW_ONLY_TEST"},
            projection={"schema_version": "shadow-test"},
            bound_execution_request={"schema_version": "bound-test"},
        ),
    )
    monkeypatch.setattr(
        front_door,
        "_bind_cuda_full_evolution_v2_attempt_authority_v1",
        lambda **_: {"execution_authority_granted": True},
    )

    class Adapter:
        @staticmethod
        def terminalize_attempt_authority_v1(**_: object) -> dict[str, object]:
            return {
                "status": "TERMINAL",
                "workhorse_terminal_binding": {
                    "schema_version": (
                        "eiel_workhorse_canonical_parent_terminal_binding_v2"
                    ),
                    "attempt_terminal_status": "COMPLETED",
                },
            }

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        provider_registry=registry,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=_fake_execution,
        mechanical_verifier_fn=_passing_mechanical,
        cuda_full_evolution_v2_crca_adapter=Adapter(),
    )

    assert result["ok"] is True, json.dumps(result, indent=2, sort_keys=True)
    assert result["workhorse_terminalized_indexed_continuation"]["operation"] == (
        "INDEXED_ACS_CONTINUATION"
    )


def test_lifecycle_reseal_failure_rewrites_durable_success_to_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: _fake_plan()),
    )

    def fail_reseal(*_args: object, **_kwargs: object) -> None:
        raise ValueError("host terminal inventory failure")

    monkeypatch.setattr(
        front_door, "reseal_terminal_lifecycle_inventory_v1", fail_reseal
    )
    with pytest.raises(ExecutionFrontDoorError, match="terminal lifecycle reseal failed"):
        execute_request_v2(
            request,
            repo_root=tmp_path,
            preflight_fn=lambda plan, **_: _fake_preflight(plan),
            execute_fn=_fake_execution,
            mechanical_verifier_fn=_passing_mechanical,
            canopy_start_fn=lambda **_: {"relays": []},
        )
    durable = json.loads(
        (tmp_path / "run" / "execution_result.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (tmp_path / "run" / "execution_state.json").read_text(encoding="utf-8")
    )
    assert durable["ok"] is False
    assert durable["status"] == state["status"] == "failed"
    assert durable["result_class"] == "TERMINAL_LIFECYCLE_RESEAL_FAILED"


def test_success_replay_recovers_crash_after_pre_reseal_durable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_lifecycle_request(tmp_path)
    plan = _fake_lifecycle_plan(_fake_plan())
    execute_calls = 0
    reseal_calls = 0
    real_reseal = campaign_orchestration.reseal_terminal_lifecycle_inventory_v1
    monkeypatch.setattr(
        front_door,
        "load_campaign_spec",
        lambda _: {"artifact_lifecycle": {"schema_version": "explicit-test"}},
    )
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    monkeypatch.setattr(
        campaign_orchestration,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )

    def execute_once(
        compiled: dict[str, object], *, run_root: Path, **kwargs: object
    ) -> dict[str, object]:
        nonlocal execute_calls
        execute_calls += 1
        return _fake_execution_with_terminal_lifecycle(
            compiled, run_root=run_root, **kwargs
        )

    def crash_once(
        compiled: dict[str, object], *, run_root: Path, **kwargs: object
    ) -> dict[str, object]:
        nonlocal reseal_calls
        reseal_calls += 1
        if reseal_calls == 1:
            durable = json.loads(
                (run_root / "execution_result.json").read_text(encoding="utf-8")
            )
            assert durable["result_class"] == "SUCCESS"
            assert (run_root / "execution_state.json").is_file()
            assert not (
                run_root / front_door.TERMINAL_LIFECYCLE_COMMIT_NAME
            ).exists()
            raise KeyboardInterrupt("injected crash before terminal reseal")
        return real_reseal(compiled, run_root=run_root, **kwargs)

    monkeypatch.setattr(
        front_door, "reseal_terminal_lifecycle_inventory_v1", crash_once
    )
    with pytest.raises(KeyboardInterrupt, match="injected crash"):
        execute_request_v2(
            request,
            repo_root=tmp_path,
            preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
            execute_fn=execute_once,
            mechanical_verifier_fn=_passing_mechanical,
        )

    run_root = tmp_path / "run"
    precommit = json.loads(
        (run_root / "execution_result.json").read_text(encoding="utf-8")
    )
    assert precommit["result_class"] == "SUCCESS"
    assert not (run_root / front_door.TERMINAL_LIFECYCLE_COMMIT_NAME).exists()

    replay = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda compiled, **_: _fake_preflight(compiled),
        execute_fn=execute_once,
        mechanical_verifier_fn=_passing_mechanical,
    )
    assert replay["result_class"] == "SUCCESS"
    assert execute_calls == 1
    assert reseal_calls == 2
    marker = front_door._validate_terminal_lifecycle_commit_v1(
        run_root=run_root.resolve(),
        request_sha256=request.request_sha256,
        plan_sha256=str(replay["plan_sha256"]),
        source_identity=replay["source_identity"],
        result=replay,
    )
    assert marker["result"]["result_sha256"] == replay["result_sha256"]


def test_front_door_completes_declared_fresh_owner_restore_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    plan = _fake_plan()
    plan["execution"] = {
        "budgets": {},
        "deliberate_restore_tick": 2_500,
        "deliberate_restore_ticks": [2_500],
    }
    plan["physical_batches"] = ({"batch_id": "batch-0000", "cell_ids": ("cell-0000",)},)
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    statuses = iter(("interrupted", "completed"))

    def execute(plan: dict[str, object], **_: object) -> dict[str, object]:
        status = next(statuses)
        return {
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "status": status,
            **(
                {
                    "interrupted_batch_id": "batch-0000",
                    "interrupted_checkpoint_next_tick": 2_500,
                }
                if status == "interrupted"
                else {}
            ),
        }

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert result["ok"] is True
    assert result["execution_receipt"]["resume_invocations"] == 1
    assert result["execution_receipt"]["deliberate_restore_ticks_completed"] == [2_500]


class _AbsoluteTickInterruptionProvider:
    provider_id = "absolute-tick-interruption-test-provider"
    profile_id = "test-profile"

    def __init__(self) -> None:
        self.execute_calls = 0
        self.resume_calls = 0

    def translate_batch(self, plan, batch_id):
        del plan
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=("cell-0000",),
            generations=21,
            ticks_per_generation=160,
            checkpoint_milestones=(21,),
            parameter_arrays=(),
            seed_arrays=(),
            total_ticks=3_360,
            checkpoint_ticks=(3_359, 3_360),
        )

    def execute_batch(self, plan, request, context):
        del plan, context
        self.execute_calls += 1
        assert request.batch_id == "batch-0000"
        return {
            "status": "interrupted",
            "checkpoint": {
                "schema_version": "eiel_production_engine_checkpoint_set_v1",
                "time_authority": "absolute_tick_v1",
                "next_absolute_tick": 3_359,
                "terminal_cell_digests": {"cell-0000": "interrupted"},
            },
            "result": {"reason": "bounded physical interruption"},
        }

    def resume_batch(self, plan, request, checkpoint, context):
        del plan, context
        self.resume_calls += 1
        assert request.batch_id == "batch-0000"
        assert checkpoint["time_authority"] == "absolute_tick_v1"
        assert checkpoint["next_absolute_tick"] == 3_359
        assert "next_tick" not in checkpoint
        return {
            "status": "completed",
            "checkpoint": {
                "schema_version": "eiel_production_engine_checkpoint_set_v1",
                "time_authority": "absolute_tick_v1",
                "next_absolute_tick": 3_360,
                "terminal_cell_digests": {"cell-0000": "completed"},
            },
            "metrics_rows": [
                {"cell_id": "cell-0000", "metrics": {"population.alive": 1}}
            ],
            "result": {"terminal_next_absolute_tick": 3_360},
        }


def test_front_door_normalizes_absolute_tick_interruption_from_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    request = ExecutionRequestV2(
        purpose=request.purpose,
        campaign_spec=request.campaign_spec,
        output_root=request.output_root,
        resource_wait=request.resource_wait,
        restore_process_isolation=request.restore_process_isolation,
        baseline=BaselineDeclarationV1.from_value(
            {
                **request.baseline.payload(),
                "deviations": [
                    row
                    for row in request.baseline.payload()["deviations"]
                    if row["requirement_id"]
                    not in {"authority.generation_runtime", "time.absolute_native_tick"}
                ],
            }
        ),
    )
    plan = _fake_plan()
    plan["execution"] = {
        "budgets": {},
        "execution_route_identity": "tick_native_execution_route_v1",
        "time_authority": "absolute_tick_v1",
        "total_ticks": 3_360,
        "checkpoint_ticks": [3_359, 3_360],
        "tick_native_control": {"refill_chunk_ticks": 160},
        "deliberate_restore_ticks": [3_359],
    }
    plan["physical_batches"] = (
        {"batch_id": "batch-0000", "cell_ids": ("cell-0000",)},
    )
    plan["cells"] = [
        {"cell_id": "cell-0000", "parameters": {}, "seeds": {}},
    ]
    provider = _AbsoluteTickInterruptionProvider()
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    observed: list[dict[str, object]] = []

    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    monkeypatch.setattr(
        "engine.monitoring.slot_telemetry.start_slot_lease",
        lambda **_: SimpleNamespace(
            update=lambda **__: None,
            close=lambda **__: None,
        ),
    )

    def execute(plan_value, **kwargs):
        receipt = campaign_orchestration.execute_campaign(
            plan_value, **kwargs
        )
        manifest_path = tmp_path / "run" / "batches" / "batch-0000" / "manifest.json"
        observed.append(
            {
                "receipt": receipt,
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        )
        return receipt

    def canopy_unavailable(**_: object) -> dict[str, object]:
        raise front_door.CanopySupervisionError(
            "relay startup is outside this host-only orchestration fixture"
        )

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        provider_registry=registry,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
        canopy_start_fn=canopy_unavailable,
    )

    assert observed, json.dumps(result, indent=2, sort_keys=True)
    first = observed[0]
    checkpoint = first["manifest"]["checkpoint"]
    assert first["manifest"]["status"] == "interrupted"
    assert first["manifest"]["batch_id"] == "batch-0000"
    assert checkpoint["schema_version"] == "eiel_production_engine_checkpoint_set_v1"
    assert checkpoint["time_authority"] == "absolute_tick_v1"
    assert checkpoint["next_absolute_tick"] == 3_359
    assert "next_tick" not in checkpoint
    assert first["receipt"]["interrupted_batch_id"] == "batch-0000"
    assert first["receipt"]["interrupted_checkpoint_next_tick"] == 3_359
    assert result["ok"] is True, json.dumps(result, indent=2, sort_keys=True)
    assert result["execution_receipt"]["deliberate_restore_ticks_completed"] == [
        3_359
    ]
    assert provider.execute_calls == 1
    assert provider.resume_calls == 1


def test_interrupted_checkpoint_tick_normalization_is_authority_strict() -> None:
    normalize = campaign_orchestration._interrupted_checkpoint_next_tick_v1

    assert normalize(
        {
            "time_authority": "absolute_tick_v1",
            "next_absolute_tick": 3_359,
            "next_tick": 3_359,
        }
    ) == 3_359
    with pytest.raises(
        campaign_orchestration.CampaignExecutionError,
        match="coordinates conflict",
    ):
        normalize(
            {
                "time_authority": "absolute_tick_v1",
                "next_absolute_tick": 3_359,
                "next_tick": 3_358,
            }
        )
    assert normalize({"next_tick": 2_500}) == 2_500
    with pytest.raises(
        campaign_orchestration.CampaignExecutionError,
        match="lacks next_tick authority",
    ):
        normalize({"next_absolute_tick": 2_500})


def test_front_door_requires_a_fresh_invocation_for_isolated_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _v2_request(
        tmp_path,
        restore_process_isolation="fresh_invocation_per_restore",
    )
    execute_request = execute_request_v2
    plan = _fake_plan()
    plan["execution"] = {
        "budgets": {},
        "deliberate_restore_tick": 2_500,
        "deliberate_restore_ticks": [2_500],
    }
    plan["physical_batches"] = (
        {"batch_id": "batch-0000", "cell_ids": ("cell-0000",)},
    )
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    calls: list[int] = []

    def execute(value: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(1)
        status = "interrupted" if len(calls) == 1 else "completed"
        return {
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "campaign_id": value["campaign_id"],
            "plan_sha256": value["plan_sha256"],
            "status": status,
            **(
                {
                    "interrupted_batch_id": "batch-0000",
                    "interrupted_checkpoint_next_tick": 2_500,
                }
                if status == "interrupted"
                else {}
            ),
        }

    first = execute_request(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert calls == [1]
    assert first["result_class"] == "RESTORE_PENDING"
    assert first["pending_restore_event"] == {
        "batch_id": "batch-0000",
        "restore_tick": 2_500,
    }
    assert first["deliberate_restore_events_completed"] == []
    restore_state = json.loads(
        (Path(request.output_root) / "execution_state.json").read_text()
    )
    assert restore_state["preflight_poll_count"] == 1
    assert "attempt" not in restore_state
    assert request.payload()["restore_process_isolation"] == (
        "fresh_invocation_per_restore"
    )

    with pytest.raises(ExecutionFrontDoorError, match="fresh OS process"):
        execute_request(
            request,
            repo_root=tmp_path,
            preflight_fn=lambda value, **_: _fake_preflight(value),
            execute_fn=execute,
            mechanical_verifier_fn=_passing_mechanical,
        )
    assert calls == [1]

    monkeypatch.setattr(
        front_door,
        "_process_identity",
        lambda: {
            "schema_version": "eiel_execution_process_identity_v1",
            "pid": 0,
            "process_instance_id": "invalid-process-instance",
        },
    )
    with pytest.raises(ExecutionFrontDoorError, match="exact process identity"):
        execute_request(
            request,
            repo_root=tmp_path,
            preflight_fn=lambda value, **_: _fake_preflight(value),
            execute_fn=execute,
            mechanical_verifier_fn=_passing_mechanical,
        )
    assert calls == [1]

    monkeypatch.setattr(
        front_door,
        "_process_identity",
        lambda: {
            "schema_version": "eiel_execution_process_identity_v1",
            "pid": 222,
            "process_instance_id": "fresh-process-instance",
        },
    )

    second = execute_request(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert calls == [1, 1]
    assert second["ok"] is True
    assert second["result_class"] == "SUCCESS"
    assert second["execution_receipt"]["resume_invocations"] == 1
    assert second["execution_receipt"]["deliberate_restore_ticks_completed"] == [
        2_500
    ]
    assert second["execution_receipt"]["deliberate_restore_events_completed"] == [
        {"batch_id": "batch-0000", "restore_tick": 2_500}
    ]
    transition = second["execution_receipt"][
        "restore_process_transitions_completed"
    ][0]
    assert transition["event"] == {
        "batch_id": "batch-0000",
        "restore_tick": 2_500,
    }
    assert transition["producer_process"]["process_instance_id"] != (
        transition["restore_process"]["process_instance_id"]
    )


def test_fresh_restore_boundary_survives_execution_storage_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _v2_request(
        tmp_path,
        restore_process_isolation="fresh_invocation_per_restore",
    )
    plan = _fake_plan()
    plan["execution"] = {
        "budgets": {},
        "deliberate_restore_tick": 2_500,
        "deliberate_restore_ticks": [2_500],
    }
    plan["physical_batches"] = (
        {"batch_id": "batch-0000", "cell_ids": ("cell-0000",)},
    )
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    identities = {
        "current": {
            "schema_version": "eiel_execution_process_identity_v1",
            "pid": 101,
            "process_instance_id": "checkpoint-producer",
        }
    }
    monkeypatch.setattr(
        front_door,
        "_process_identity",
        lambda: dict(identities["current"]),
    )
    statuses = iter(("interrupted", "storage_wait", "completed"))
    calls: list[str] = []

    def execute(value: dict[str, object], **_: object) -> dict[str, object]:
        status = next(statuses)
        calls.append(status)
        return {
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "campaign_id": value["campaign_id"],
            "plan_sha256": value["plan_sha256"],
            **(
                {
                    "status": "interrupted",
                    "interrupted_batch_id": "batch-0000",
                    "interrupted_checkpoint_next_tick": 2_500,
                }
                if status == "interrupted"
                else (
                    {
                        "status": "interrupted",
                        "storage_wait": True,
                        "storage_wait_receipt_path": str(
                            Path(request.output_root) / "storage-wait.json"
                        ),
                    }
                    if status == "storage_wait"
                    else {"status": "completed"}
                )
            ),
        }

    first = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )
    boundary = first["restore_process_boundary"]
    assert first["result_class"] == "RESTORE_PENDING"

    identities["current"] = {
        "schema_version": "eiel_execution_process_identity_v1",
        "pid": 202,
        "process_instance_id": "storage-wait-restorer",
    }
    waiting = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )
    assert waiting["result_class"] == "STORAGE_WAIT"
    assert waiting["pending_restore_event"] == first["pending_restore_event"]
    assert waiting["restore_process_boundary"] == boundary
    wait_state = json.loads(
        (Path(request.output_root) / "execution_state.json").read_text()
    )
    assert wait_state["restore_process_boundary"] == boundary

    identities["current"] = dict(boundary["producer_process"])
    with pytest.raises(ExecutionFrontDoorError, match="fresh OS process"):
        execute_request_v2(
            request,
            repo_root=tmp_path,
            preflight_fn=lambda value, **_: _fake_preflight(value),
            execute_fn=execute,
            mechanical_verifier_fn=_passing_mechanical,
        )
    assert calls == ["interrupted", "storage_wait"]

    identities["current"] = {
        "schema_version": "eiel_execution_process_identity_v1",
        "pid": 303,
        "process_instance_id": "successful-restorer",
    }
    completed = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )
    transitions = completed["execution_receipt"][
        "restore_process_transitions_completed"
    ]
    assert completed["result_class"] == "SUCCESS"
    assert completed["execution_receipt"]["resume_invocations"] == 1
    assert len(transitions) == 1
    assert transitions[0]["producer_process"] == boundary["producer_process"]
    assert transitions[0]["restore_process"] == identities["current"]


def test_front_door_completes_restore_for_every_physical_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    plan = _fake_plan()
    plan["execution"] = {"budgets": {}, "deliberate_restore_ticks": [2_500]}
    plan["physical_batches"] = (
        {"batch_id": "batch-0000", "cell_ids": ("cell-0000",)},
        {"batch_id": "batch-0001", "cell_ids": ("cell-0001",)},
    )
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )
    receipts = iter(
        (
            {
                "status": "interrupted",
                "interrupted_batch_id": "batch-0000",
                "interrupted_checkpoint_next_tick": 2_500,
            },
            {
                "status": "interrupted",
                "interrupted_batch_id": "batch-0001",
                "interrupted_checkpoint_next_tick": 2_500,
            },
            {"status": "completed"},
        )
    )

    def execute(plan: dict[str, object], **_: object) -> dict[str, object]:
        return {
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            **next(receipts),
        }

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=execute,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert result["ok"] is True
    assert result["execution_receipt"]["resume_invocations"] == 2
    assert result["execution_receipt"]["deliberate_restore_events_completed"] == [
        {"batch_id": "batch-0000", "restore_tick": 2_500},
        {"batch_id": "batch-0001", "restore_tick": 2_500},
    ]


def test_front_door_rejects_restore_for_unbound_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    plan = _fake_plan()
    plan["execution"] = {"budgets": {}, "deliberate_restore_ticks": [2_500]}
    plan["physical_batches"] = ({"batch_id": "batch-0000", "cell_ids": ("cell-0000",)},)
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=lambda value, **_: {
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "campaign_id": value["campaign_id"],
            "plan_sha256": value["plan_sha256"],
            "status": "interrupted",
            "interrupted_batch_id": "batch-9999",
            "interrupted_checkpoint_next_tick": 2_500,
        },
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert result["ok"] is False
    assert result["result_class"] == "EXECUTION_FAILED"
    assert any(
        "outside its deliberate restore schedule" in warning
        for warning in result["warnings"]
    )


def test_front_door_records_operator_interrupt_as_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=_fake_plan),
    )

    def interrupt(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda value, **_: _fake_preflight(value),
        execute_fn=interrupt,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["result_class"] == "EXECUTION_FAILED"
    assert any("KeyboardInterrupt" in warning for warning in result["warnings"])
    state = json.loads((Path(request.output_root) / "execution_state.json").read_text())
    assert state["status"] == "failed"


def test_resource_wait_rechecks_only_transient_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(
        tmp_path,
        wait=ResourceWaitPolicyV1(enabled=True, poll_seconds=5, max_wait_seconds=30),
    )
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: _fake_plan()),
    )
    calls: list[int] = []

    def preflight(plan: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(1)
        if len(calls) == 1:
            receipt = _fake_preflight(plan)
            receipt["ready_for_execution"] = False
            receipt["reasons"] = [
                {
                    "code": "insufficient_free_vram",
                    "path": "execution.budgets.vram_bytes",
                    "message": "busy",
                }
            ]
            return receipt
        return _fake_preflight(plan)

    slept: list[float] = []
    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=preflight,
        execute_fn=_fake_execution,
        mechanical_verifier_fn=_passing_mechanical,
        sleep_fn=slept.append,
        monotonic_fn=lambda: 0.0,
    )

    assert result["ok"] is True
    assert len(calls) == 2
    assert slept == [5]
    assert result["preflight_poll_count"] == 2
    assert "preflight_attempts" not in result


def test_storage_wait_rechecks_same_attempt_with_distinct_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(
        tmp_path,
        wait=ResourceWaitPolicyV1(enabled=True, poll_seconds=5, max_wait_seconds=30),
    )
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: _fake_plan()),
    )
    calls: list[int] = []

    def preflight(plan: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(1)
        receipt = _fake_preflight(plan)
        if len(calls) == 1:
            receipt["ready_for_execution"] = False
            receipt["reasons"] = [
                {
                    "code": "lifecycle_storage_admission_failed",
                    "path": "artifact_lifecycle.storage_guard",
                    "message": "durable storage floor would be crossed",
                }
            ]
        return receipt

    observed_states: list[str] = []

    def sleep(_seconds: float) -> None:
        state = json.loads(
            (Path(request.output_root) / "execution_state.json").read_text()
        )
        observed_states.append(str(state["status"]))

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=preflight,
        execute_fn=_fake_execution,
        mechanical_verifier_fn=_passing_mechanical,
        sleep_fn=sleep,
        monotonic_fn=lambda: 0.0,
    )

    assert result["ok"] is True
    assert len(calls) == 2
    assert observed_states == ["WAIT_STORAGE"]


def test_storage_wait_timeout_is_a_resumable_storage_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(
        tmp_path,
        wait=ResourceWaitPolicyV1(enabled=True, poll_seconds=5, max_wait_seconds=1),
    )
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: _fake_plan()),
    )

    def preflight(plan: dict[str, object], **_: object) -> dict[str, object]:
        receipt = _fake_preflight(plan)
        receipt["ready_for_execution"] = False
        receipt["reasons"] = [
            {
                "code": "lifecycle_storage_admission_failed",
                "path": "artifact_lifecycle.storage_guard",
                "message": "durable storage floor would be crossed",
            }
        ]
        return receipt

    times = iter((0.0, 1.0))
    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=preflight,
        mechanical_verifier_fn=_passing_mechanical,
        sleep_fn=lambda _: None,
        monotonic_fn=lambda: next(times),
    )

    assert result["ok"] is False
    assert result["status"] == "wait_storage"
    assert result["result_class"] == "STORAGE_WAIT"
    state = json.loads(
        (Path(request.output_root) / "execution_state.json").read_text()
    )
    assert state["status"] == "WAIT_STORAGE"


def test_execution_storage_wait_is_not_parsed_as_restore_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _v2_request(tmp_path)
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: _fake_plan()),
    )

    def storage_wait_execution(
        plan: dict[str, object], *, run_root: Path, **_: object
    ) -> dict[str, object]:
        return {
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "profile_id": plan["resolved_profile"],
            "provider_id": "test-provider",
            "status": "interrupted",
            "storage_wait": True,
            "storage_wait_receipt_path": str(run_root / "storage_wait_receipt.json"),
        }

    result = execute_request_v2(
        request,
        repo_root=tmp_path,
        preflight_fn=lambda plan, **_: _fake_preflight(plan),
        execute_fn=storage_wait_execution,
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert result["ok"] is False
    assert result["status"] == "wait_storage"
    assert result["result_class"] == "STORAGE_WAIT"
    assert not any(
        "deliberate restore schedule" in warning for warning in result["warnings"]
    )


def test_validate_persists_standardized_preflight_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: _fake_plan()),
    )

    result = execute_request_v1(
        request,
        repo_root=tmp_path,
        mode="validate",
        preflight_fn=lambda plan, **_: _fake_preflight(plan),
        mechanical_verifier_fn=_passing_mechanical,
    )

    stored = json.loads(
        (tmp_path / "run" / "execution_result.json").read_text(encoding="utf-8")
    )
    assert result["result_class"] == "PREFLIGHT_PASS"
    assert stored == result
    assert stored["result_sha256"] == front_door.canonical_sha256(
        {key: value for key, value in stored.items() if key != "result_sha256"}
    )


def test_validate_reuses_json_roundtripped_plan_with_tuple_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    plan = _fake_plan()
    plan["physical_batches"] = ({"batch_id": "batch-0000", "cell_ids": ()},)
    monkeypatch.setattr(front_door, "load_campaign_spec", lambda _: {})
    monkeypatch.setattr(
        front_door,
        "compile_campaign",
        lambda _: SimpleNamespace(payload=lambda: plan),
    )

    first = execute_request_v1(
        request,
        repo_root=tmp_path,
        mode="validate",
        preflight_fn=lambda value, **_: _fake_preflight(value),
        mechanical_verifier_fn=_passing_mechanical,
    )
    second = execute_request_v1(
        request,
        repo_root=tmp_path,
        mode="validate",
        preflight_fn=lambda value, **_: _fake_preflight(value),
        mechanical_verifier_fn=_passing_mechanical,
    )

    assert first["result_class"] == "PREFLIGHT_PASS"
    assert second["validation_only"] is True
    assert second["mechanical_verification"]["status"] == "PASS"


def test_mechanical_verifier_rejects_changed_frozen_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"changed")
    expected = sha256(b"original").hexdigest()
    verification = verify_mechanical_execution_v1(
        run_root=tmp_path / "run",
        plan={"plan_sha256": "plan"},
        preflight={
            "schema_version": "eiel_campaign_preflight_v1",
            "plan_sha256": "plan",
            "artifacts": [
                {
                    "name": "fixture",
                    "path": str(artifact),
                    "expected_sha256": expected,
                }
            ],
        },
    )

    assert verification["status"] == "FAIL"
    assert any("frozen artifact" in error for error in verification["errors"])


def test_tensor_world_c7_profile_is_evaluated_without_formal_q6_claim(
    tmp_path: Path,
) -> None:
    freeze = TensorWorldC7QualificationFreezeV1(
        source_commit="source",
        oracle_commit="oracle",
        route_fingerprint="0" * 64,
        backend_id="cpu",
        provider_contract_id="provider",
        batch=1,
        grid_size=4,
        initial_population=2,
        slot_capacity=4,
        policy_family="policy",
        mechanics=("birth",),
        checkpoint_cadence=1,
        seeds=(7,),
        case_ids=("case-7",),
        evidence_root=str(tmp_path),
        working_tree_clean=True,
    )
    evidence = TensorWorldC7EvidenceReceiptV1(
        freeze_fingerprint=freeze.fingerprint(),
        semantic_parity=True,
        determinism_parity=True,
        checkpoint_replay_parity=True,
        external_contract=True,
        residency=True,
        resource_safe=True,
        material_performance=True,
        evidence_bindings=c7_freeze_evidence_bindings_v1(freeze),
    )
    freeze_path = tmp_path / "freeze.json"
    evidence_path = tmp_path / "evidence.json"
    freeze_path.write_text(json.dumps(freeze.to_dict()), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence.to_dict()), encoding="utf-8")

    evaluation = evaluate_qualification_profile_v1(
        {
            "type": "tensor_world_c7",
            "freeze": str(freeze_path),
            "evidence": str(evidence_path),
        },
        execution_result={"execution_status": "completed", "result_sha256": "result"},
    )

    assert evaluation is not None
    assert evaluation["status"] == "PASS"
    assert evaluation["decision"] == "TW-C7-QUALIFIED-FOR-ADMISSION-REVIEW"
    assert "formal Q6" in evaluation["semantic_questions"][0]


def test_evaluate_qualification_profile_none_is_not_requested() -> None:
    """FE Q6 uses purpose=qualification without a C7 profile; shipped evaluator stays None."""

    evaluation = evaluate_qualification_profile_v1(
        None,
        execution_result={
            "execution_status": "completed",
            "result_class": "SUCCESS",
            "purpose": "qualification",
            "result_sha256": "result",
        },
    )
    assert evaluation is None
