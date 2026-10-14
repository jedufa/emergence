from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from engine.campaigns.execution_front_door_v1 import (
    ExecutionRequestV1,
    ExecutionRequestV2,
)
from eiel_composition import execution_cli as canonical_execution_cli
from eiel_composition import cuda_full_evolution_v2_launchpad_adapter
from tools import eiel
from tools import execution_cli


def _result() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "preflight_only",
        "result_class": "PREFLIGHT_PASS",
        "campaign_id": "host-cli-fixture",
        "run_root": "/tmp/host-cli-fixture",
        "qualification_status": "not_requested",
    }


def test_cli_injects_one_canonical_adapter_at_both_v2_seams(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    request = ExecutionRequestV2.from_mapping(
        {
            "schema_version": "eiel_execution_request_v2",
            "purpose": "qualification",
            "campaign_spec": "candidate.yaml",
            "output_root": "run",
            "resource_wait": {
                "enabled": False,
                "poll_seconds": 300,
                "max_wait_seconds": None,
            },
            "qualification_profile": None,
            "baseline": {
                "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
                "intent": "qualification",
                "deviations": [],
            },
        },
        base_dir=tmp_path,
    )
    adapter = object()
    constructed: list[Path] = []
    called: dict[str, Any] = {}

    def adapter_factory(*, repo_root: Path) -> object:
        constructed.append(repo_root)
        return adapter

    def execute_v2(value: ExecutionRequestV2, **kwargs: Any) -> dict[str, Any]:
        called["request"] = value
        called.update(kwargs)
        return _result()

    monkeypatch.setattr(execution_cli, "load_execution_request_v1", lambda _: request)
    monkeypatch.setattr(
        canonical_execution_cli,
        "CanonicalCudaFullEvolutionV2LaunchpadAdapterV1",
        adapter_factory,
    )
    monkeypatch.setattr(execution_cli, "execute_request_v2", execute_v2)
    monkeypatch.setattr(
        execution_cli,
        "execute_request_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2 must not use the V1 front door")
        ),
    )

    assert (
        canonical_execution_cli.main(
            ["validate", str(tmp_path / "request.json"), "--repo-root", str(tmp_path)]
        )
        == 0
    )
    assert constructed == [tmp_path]
    assert called["request"] is request
    assert called["mode"] == "validate"
    assert called["cuda_full_evolution_v2_crca_adapter"] is adapter
    assert called["cuda_full_evolution_v2_checkpoint_occurrence_adapter"] is adapter


def test_cli_preserves_v1_without_constructing_the_v2_adapter(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    request = ExecutionRequestV1.from_mapping(
        {
            "schema_version": "eiel_execution_request_v1",
            "purpose": "validation",
            "campaign_spec": "historical.yaml",
            "output_root": "historical-run",
            "resource_wait": {
                "enabled": False,
                "poll_seconds": 300,
                "max_wait_seconds": None,
            },
            "qualification_profile": None,
        },
        base_dir=tmp_path,
    )
    called: dict[str, Any] = {}

    def execute_v1(value: ExecutionRequestV1, **kwargs: Any) -> dict[str, Any]:
        called["request"] = value
        called.update(kwargs)
        return _result()

    monkeypatch.setattr(execution_cli, "load_execution_request_v1", lambda _: request)
    monkeypatch.setattr(execution_cli, "execute_request_v1", execute_v1)
    monkeypatch.setattr(
        execution_cli,
        "execute_request_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V1 must not use the V2 front door")
        ),
    )
    monkeypatch.setattr(
        canonical_execution_cli,
        "CanonicalCudaFullEvolutionV2LaunchpadAdapterV1",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("V1 must not construct the V2 adapter")
        ),
    )

    assert (
        canonical_execution_cli.main(
            ["validate", str(tmp_path / "request.json"), "--repo-root", str(tmp_path)]
        )
        == 0
    )
    assert called == {
        "request": request,
        "repo_root": tmp_path,
        "mode": "validate",
    }


def test_unified_cli_routes_execution_through_canonical_composition() -> None:
    assert eiel.EXECUTION_TOOLS["execute"][1] == "eiel_composition.execution_cli"
    assert eiel.EXECUTION_TOOLS["validate"][1] == "eiel_composition.execution_cli"


def _resource_report(*, gpu_pid: int) -> dict[str, Any]:
    return {
        "schema_version": "eiel_resource_monitor_report_v1",
        "generated_at_utc": "2026-08-26T12:00:00Z",
        "memory": {"ram_available_bytes": 32 * 1024**3},
        "storage": {"roots": [{"free_bytes": 128 * 1024**3}]},
        "gpu": {
            "available": True,
            "gpus": [{"vram_free_bytes": 5 * 1024**3}],
            "processes": [{"pid": gpu_pid, "used_memory_bytes": 32 * 1024**2}],
        },
        "blockers": [],
        "recommendation": {
            "value": "SAFE",
            "confidence": "HIGH",
            "reason": "fixture",
        },
    }


def _resource_evidence(monkeypatch: Any, *, gpu_pid: int) -> dict[str, Any]:
    monkeypatch.setattr(
        cuda_full_evolution_v2_launchpad_adapter.resource_monitor,
        "collect_report",
        lambda _config: _resource_report(gpu_pid=gpu_pid),
    )
    return dict(
        cuda_full_evolution_v2_launchpad_adapter._default_resource_evidence_v1(
            repo_root=Path.cwd(),
            route={"device_id": "GPU-FIXTURE", "event_transport": "direct_sparse_v3"},
            campaign_plan={"execution": {"budgets": {"vram_bytes": 1024**3}}},
            at_utc="2026-08-26T12:00:00Z",
        )
    )


def test_resource_admission_excludes_only_the_observing_cuda_context(
    monkeypatch: Any,
) -> None:
    evidence = _resource_evidence(monkeypatch, gpu_pid=os.getpid())

    assert evidence["resource_recommendation"]["decision"] == "ADMIT"
    assert evidence["live_provider_check"]["available"] is True
    assert evidence["resource_observation"]["report"]["gpu"]["processes"] == [
        {"pid": os.getpid(), "used_memory_bytes": 32 * 1024**2}
    ]
    assert evidence["resource_observation"]["assessment_scope"] == {
        "schema_version": "eiel_launchpad_resource_assessment_scope_v1",
        "observer_pid": os.getpid(),
        "observer_gpu_process_count": 1,
        "external_gpu_process_count": 0,
        "observer_gpu_context_excluded_from_conflict": True,
    }


def test_resource_admission_still_rejects_an_external_cuda_process(
    monkeypatch: Any,
) -> None:
    evidence = _resource_evidence(monkeypatch, gpu_pid=os.getpid() + 100_000)

    assert evidence["resource_recommendation"]["decision"] == "WAIT"
    assert evidence["live_provider_check"]["available"] is False
    assert evidence["resource_observation"]["assessment_scope"] == {
        "schema_version": "eiel_launchpad_resource_assessment_scope_v1",
        "observer_pid": os.getpid(),
        "observer_gpu_process_count": 0,
        "external_gpu_process_count": 1,
        "observer_gpu_context_excluded_from_conflict": False,
    }
