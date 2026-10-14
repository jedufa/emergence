from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from engine.native_runtime import cuda_architecture_b_production_adapter as adapter_module
from engine.native_runtime import cuda_architecture_b_runner_owner_v1 as owner
from engine.native_runtime.cuda_architecture_b_routing import CUDA_BACKEND


def _selected_cuda_route() -> dict[str, Any]:
    return {
        "contract_status": "artifact_bound_adapter_resolved",
        "resolved_backend": CUDA_BACKEND,
        "dispatch_started": False,
        "fallback_performed": False,
        "fallback_phase": "none",
        "no_hidden_fallback": True,
    }


def _prepared() -> SimpleNamespace:
    return SimpleNamespace(
        physical_window_arguments=(),
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {"metadata": {}}
        },
    )


def _runner(tmp_path: Path) -> SimpleNamespace:
    artifact_paths = []
    for name in ("owner.so", "qualification.json", "reporting.json"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        artifact_paths.append(path)
    return SimpleNamespace(
        sim_cfg=SimpleNamespace(
            branch_id="fault-injection",
            campaign_id="phase1",
            seed=17,
            cuda_architecture_b_request_mode="explicit_cuda",
            cuda_architecture_b_allow_explicit_exactness_override=True,
            cuda_architecture_b_performance_decision="unmeasured",
            cuda_architecture_b_performance_decision_measured=False,
            cuda_architecture_b_artifact_path=str(artifact_paths[0]),
            cuda_architecture_b_qualification_receipt_path=str(artifact_paths[1]),
            cuda_architecture_b_reporting_receipt_path=str(artifact_paths[2]),
        ),
        run_dir=str(tmp_path / "run"),
        pop=SimpleNamespace(),
    )


def _adapter_with_selected_route(
    artifact: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> adapter_module.CudaArchitectureBProductionAdapter:
    adapter = adapter_module.CudaArchitectureBProductionAdapter(artifact)

    def resolve_once(**_kwargs: Any) -> dict[str, Any]:
        route = _selected_cuda_route()
        adapter._route_receipt = dict(route)
        return route

    monkeypatch.setattr(adapter, "resolve_once", resolve_once)
    return adapter


def _execute_owner(
    runner: SimpleNamespace,
    artifact: SimpleNamespace,
    adapter: adapter_module.CudaArchitectureBProductionAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = SimpleNamespace(routing_workload=lambda: {})
    monkeypatch.setattr(
        owner,
        "derive_cuda_architecture_b_production_runner_envelope",
        lambda _runner: envelope,
    )
    owner.execute_cuda_architecture_b_runner_run_v1(
        runner,
        start_gen=0,
        end_gen=4,
        skip_init=False,
        artifact_loader=lambda *_args: artifact,
        adapter_type=lambda _artifact: adapter,
        source_builder=lambda *_args, **_kwargs: _prepared(),
    )


def test_run_control_construction_failure_is_pre_dispatch_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = SimpleNamespace(module=object(), capability={})
    adapter = adapter_module.CudaArchitectureBProductionAdapter(artifact)
    adapter._route_receipt = _selected_cuda_route()

    def fail_control_construction(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected run-control construction failure")

    monkeypatch.setattr(
        adapter_module,
        "create_run_control_v2",
        fail_control_construction,
    )

    with pytest.raises(
        adapter_module.CudaArchitectureBProductionDispatchError
    ) as raised:
        adapter.execute_prepared(
            run_id="pre-dispatch-failure",
            physical_window_arguments=(),
            policy_mutation_enable=False,
            policy_mutation_mode="none",
            policy_mutation_rate=0.0,
            policy_mutation_std=0.0,
        )

    error = raised.value
    assert error.production_route_receipt["contract_status"] == (
        "native_engine_pre_dispatch_rejected"
    )
    assert error.production_route_receipt["resolved_backend"] == "none"
    assert error.production_route_receipt["dispatch_started"] is False
    assert error.native_failure_receipt is None
    assert error.run_control_manifest is None
    assert error.native_terminal_result is None
    assert error.failure_settlement_error is None


def test_owner_does_not_report_native_settlement_for_pre_dispatch_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    artifact = SimpleNamespace(module=object(), capability={})
    adapter = _adapter_with_selected_route(artifact, monkeypatch)

    def fail_control_construction(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected run-control construction failure")

    monkeypatch.setattr(
        adapter_module,
        "create_run_control_v2",
        fail_control_construction,
    )

    with pytest.raises(owner.CudaArchitectureBRunnerOwnerError) as raised:
        _execute_owner(runner, artifact, adapter, monkeypatch)

    receipt = runner._cuda_architecture_b_route_receipt
    violations = []
    if "pre-dispatch" not in str(raised.value):
        violations.append("owner exception omitted pre-dispatch classification")
    if receipt.get("contract_status") != "native_engine_pre_dispatch_rejected":
        violations.append("runner retained the selected-CUDA route")
    if receipt.get("resolved_backend") != "none":
        violations.append("runner still claimed a resolved CUDA backend")
    if receipt.get("dispatch_started") is not False:
        violations.append("runner claimed dispatch_started=true")
    if receipt.get("production_promotion_claimed") is not False:
        violations.append("runner did not explicitly prohibit production promotion")

    failure_path = Path(runner.run_dir) / "cuda_architecture_b_run_failed.json"
    if failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="ascii"))
        if failure.get("phase") != "pre_dispatch":
            violations.append("failure artifact labeled the phase as dispatch")
        if "native_failure_settlement_exact" in failure:
            violations.append(
                "failure artifact claimed native_failure_settlement_exact=false"
            )
        if failure.get("success_marker_allowed") is not False:
            violations.append("failure artifact allowed a success marker")
        if failure.get("promotion_result_allowed") is not False:
            violations.append("failure artifact allowed a promotion result")
    assert violations == []


def test_publish_failure_exception_is_structured_and_prohibits_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    artifact = SimpleNamespace(module=object(), capability={})
    adapter = _adapter_with_selected_route(artifact, monkeypatch)

    class Control:
        def manifest(self) -> dict[str, Any]:
            return {
                "status": "initialized",
                "sealed": False,
                "active_generation": -1,
                "last_boundary": {"trajectory_commit_frontier": 0},
            }

        def publish_failure(self, *_args: Any) -> Any:
            raise RuntimeError("injected native failure publication failure")

        def final_result(self) -> Any:
            raise AssertionError("final_result must not follow failed publication")

    control = Control()
    monkeypatch.setattr(
        adapter_module,
        "create_run_control_v2",
        lambda *_args, **_kwargs: control,
    )

    def fail_native_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected native dispatch failure")

    monkeypatch.setattr(
        adapter_module,
        "execute_direct_resident_windows_v2",
        fail_native_dispatch,
    )

    with pytest.raises(owner.CudaArchitectureBRunnerOwnerError) as raised:
        _execute_owner(runner, artifact, adapter, monkeypatch)

    dispatch_error = raised.value.__cause__
    assert isinstance(
        dispatch_error,
        adapter_module.CudaArchitectureBProductionDispatchError,
    )
    assert dispatch_error.production_route_receipt["dispatch_started"] is True
    assert dispatch_error.failure_settlement_error == {
        "error_type": "RuntimeError",
        "error": "injected native failure publication failure",
    }
    assert dispatch_error.native_failure_receipt is None
    assert dispatch_error.native_terminal_result is None

    root = Path(runner.run_dir)
    failure = json.loads(
        (root / "cuda_architecture_b_run_failed.json").read_text(encoding="ascii")
    )
    manifest = json.loads(
        (root / "cuda_architecture_b_output_manifest.json").read_text(
            encoding="ascii"
        )
    )
    assert failure["phase"] == "dispatch"
    assert failure["native_failure_settlement_exact"] is False
    assert failure["failure_settlement_error"] == (
        dispatch_error.failure_settlement_error
    )
    assert failure["success_marker_allowed"] is False
    assert failure["promotion_result_allowed"] is False
    assert manifest["status"] == "failed"
    assert manifest["success_marker_allowed"] is False
    assert manifest["promotion_result_allowed"] is False
