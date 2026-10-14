from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from engine.native_runtime import cuda_architecture_b_runner_owner_v1 as owner
from engine.native_runtime.cuda_architecture_b_production_adapter import (
    CudaArchitectureBProductionExecutionV2,
)
from engine.native_runtime.cuda_architecture_b_routing import CUDA_BACKEND
from engine.perf.cuda_architecture_b_checkpoint import CudaBoundaryCheckpoint
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    DirectResidentTerminalSettlementV1,
)
from env.env_config import EnvConfig
from evolution.population import PopConfig
from simulation.chronicle.recorder import ScientificChronicleRecorder
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg import runner_loop


class _Control:
    def __init__(self) -> None:
        self.expected_frontier = 3
        self.state = {
            "controller_id": "controller-1",
            "owner_epoch": 1,
            "active_generation": self.expected_frontier,
            "last_boundary": {"trajectory_commit_frontier": self.expected_frontier},
            "sealed": False,
            "awaiting_final_output_ack": True,
        }
        self.failures: list[tuple[Any, ...]] = []
        self.terminal: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        return dict(self.state)

    def acknowledge_run_complete(
        self, controller_id: str, owner_epoch: int, boundary: dict[str, Any]
    ) -> dict[str, Any]:
        assert controller_id == self.state["controller_id"]
        assert owner_epoch == self.state["owner_epoch"]
        assert boundary["snapshot_frontier"] == self.expected_frontier
        self.terminal = {
            "terminal_status": "run_complete",
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        self.state["sealed"] = True
        self.state["awaiting_final_output_ack"] = False
        return dict(self.terminal)

    def acknowledge_cancellation_checkpoint(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("normal g4 completion must not use cancellation ack")

    def publish_failure(self, *args: Any) -> dict[str, Any]:
        self.failures.append(args)
        self.terminal = {
            "terminal_status": "failed",
            "sealed": True,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        self.state.update(
            {
                "status": "failed",
                "sealed": True,
                "awaiting_final_output_ack": False,
            }
        )
        return {"disposition": "accepted", "terminal_status": "failed"}

    def final_result(self) -> dict[str, Any] | None:
        return self.terminal


def _runner(tmp_path: Path) -> SimpleNamespace:
    artifacts = []
    for name in ("owner.so", "qualification.json", "reporting.json"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        artifacts.append(path)
    sim_cfg = SimConfig(
        branch_id="branch",
        campaign_id="campaign",
        seed=101,
        generations=4,
        steps_per_generation=80,
        device="cpu",
        bank_device="cpu",
        campaign_output_profile="compact_default",
        allow_large_output_profile=False,
        allow_generation_checkpointing=False,
        native_runtime_allow_runtime_jit_build=False,
        native_runtime_fixed_policy_family="mlp",
        native_runtime_max_births_per_generation=64,
        cuda_architecture_b_max_births_per_tick=64,
        checkpoint_period_gen=0,
        checkpoint_bundle_v2_period_gen=0,
        policy_deterministic_actions=False,
        shock_period=0,
        shock_duration=0,
        shock_start_prob=0.0,
        pressure_autopilot_allow_shock_enable=False,
        cuda_architecture_b_request_mode="explicit_cuda",
        cuda_architecture_b_allow_explicit_exactness_override=True,
        cuda_architecture_b_performance_decision="unmeasured",
        cuda_architecture_b_performance_decision_measured=False,
        cuda_architecture_b_artifact_path=str(artifacts[0]),
        cuda_architecture_b_qualification_receipt_path=str(artifacts[1]),
        cuda_architecture_b_reporting_receipt_path=str(artifacts[2]),
    )
    env_cfg = EnvConfig(
        width=16,
        height=16,
        patch_world_enable=True,
        patch_world_schema_version=1,
        patch_world_observation_enable=True,
        patch_world_observation_schema_version=1,
        patch_world_observation_stacked_gather_enable=False,
        patch_world_observation_numba_gather_enable=False,
        patch_world_diagnostics_enable=True,
        patch_world_lineage_descriptors_enable=True,
        patch_world_descriptor_schema_version=1,
        patch_world_waste_consequence_enable=True,
        patch_movement_cost_apply_enable=True,
        reproduce_requires_mate=False,
        tier4_juvenile_vulnerability_enable=False,
        tier4_juvenile_steps=0,
    )
    return SimpleNamespace(
        sim_cfg=sim_cfg,
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=64),
        _policy_use_gru=[False] * 64,
        run_dir=str(tmp_path / "run"),
        pop=SimpleNamespace(),
    )


def _checkpoint(generation: int = 4) -> CudaBoundaryCheckpoint:
    return CudaBoundaryCheckpoint(
        projection={"metadata": {"generation": generation, "tick": 0}},
        projection_digest="a" * 64,
        device="cuda:0",
        device_to_host_operations=1,
        device_to_host_bytes=10,
        device_synchronizations=1,
    )


def _settlement(
    generation_count: int = 4,
    *,
    start_generation: int = 0,
) -> DirectResidentTerminalSettlementV1:
    rows = tuple(
        {
            "schema_version": "eiel_patch_p2_production_compact_summary_v1",
            "generation": generation,
            "world": "patch",
            "alive": 64,
            "births": 1,
            "failed_births": 0,
            "deaths": 1,
            "event_cursor": generation + 1,
            "moved": 10,
            "gathered": 2,
        }
        for generation in range(start_generation, start_generation + generation_count)
    )
    return DirectResidentTerminalSettlementV1(
        schema_version="settlement",
        compact_summary_rows=rows,
        terminal_checkpoint=_checkpoint(start_generation + generation_count),
        terminal_generation=start_generation + generation_count,
    )


def test_compact_payload_has_closed_jsonl_and_csv_schema() -> None:
    jsonl, csv_payload = owner._compact_payload(_settlement().compact_summary_rows)

    assert jsonl.count(b"\n") == 4
    assert csv_payload.splitlines()[0].decode("ascii").split(",") == list(
        owner.COMPACT_FIELD_ORDER
    )

    bad = dict(_settlement().compact_summary_rows[0])
    bad.pop("gathered")
    with pytest.raises(owner.CudaArchitectureBRunnerOwnerError, match="closed"):
        owner._compact_payload([bad])


def test_compact_payload_accepts_consistent_resource_summary_extension() -> None:
    rows = []
    for row in _settlement().compact_summary_rows:
        rows.append(
            {
                **row,
                "patch_food_stock_after_turnover": 12.5,
                "energy_collected_exact_sum_after_turnover": 3.25,
                "policy_energy_live_sum_after_turnover": 44.0,
            }
        )

    _jsonl, csv_payload = owner._compact_payload(rows)

    assert csv_payload.splitlines()[0].decode("ascii").split(",") == [
        *owner.COMPACT_FIELD_ORDER,
        "patch_food_stock_after_turnover",
        "energy_collected_exact_sum_after_turnover",
        "policy_energy_live_sum_after_turnover",
    ]


def test_compact_payload_rejects_inconsistent_resource_summary_extension() -> None:
    rows = [dict(row) for row in _settlement().compact_summary_rows]
    rows[0]["patch_food_stock_after_turnover"] = 12.5

    with pytest.raises(owner.CudaArchitectureBRunnerOwnerError, match="closed"):
        owner._compact_payload(rows)


def test_persist_outputs_adds_scientific_throughput_efficiency_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    checkpoint = tmp_path / "run" / "checkpoint"
    checkpoint.mkdir(parents=True)

    def write_events(
        jsonl: Path, csv_path: Path, *_args: Any, **_kwargs: Any
    ) -> dict[str, Any]:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        owner._atomic_write(jsonl, b"")
        owner._atomic_write(csv_path, b"")
        return {"end_cursor": 4}

    monkeypatch.setattr(owner, "typed_lifecycle_events_from_projection", lambda *_args: [])
    monkeypatch.setattr(owner, "validated_typed_lifecycle_event_cursor", lambda *_args: 4)
    monkeypatch.setattr(owner, "write_typed_lifecycle_public_outputs", write_events)
    monkeypatch.setattr(
        owner,
        "_persist_checkpoint",
        lambda *_args, **_kwargs: (
            checkpoint,
            {"schema_version": "fixture_checkpoint_manifest"},
        ),
    )
    monkeypatch.setattr(
        owner,
        "materialize_default_terminal_canopy",
        lambda **_kwargs: {"path": "lineage-playback-relay", "status": "generated"},
    )
    execution = CudaArchitectureBProductionExecutionV2(
        run_id="ste-run",
        owner_epoch=1,
        run_control=object(),
        direct_result={
            "grid_blocks": 48,
            "threads_per_block": 128,
            "persistent_hot_loop_launch_count": 8,
            "cuda_event_elapsed_ns": 1_000_000_000,
            "bounded_device_ticks": 3840,
        },
        run_control_manifest={"status": "generation_complete"},
        completed_generations=4,
        production_route_receipt={
            "resolved_backend": CUDA_BACKEND,
            "hardware": {
                "detector_status": "fixture",
                "name": "fixture gpu",
                "vram_bytes": 8_000,
                "sm_count": 24,
                "warp_size": 32,
                "max_resident_blocks_per_sm": 16,
                "max_resident_warps_per_sm": 64,
                "max_threads_per_sm": 2048,
            },
        },
    )

    manifest_path, _checkpoint_path, manifest = owner._persist_outputs(
        runner,
        settlement=_settlement(),
        execution=execution,
    )

    persisted = json.loads(manifest_path.read_text(encoding="ascii"))
    report = persisted["scientific_throughput_efficiency"]
    assert manifest["scientific_throughput_efficiency"] == report
    assert report["metric_name"] == "Scientific Throughput Efficiency"
    assert report["backend"] == "cuda"
    assert manifest["artifacts"]["default_canopy"]["status"] == "generated"
    gpu = report["backend_specific"]["gpu_resource_utilization"]
    assert gpu["sm_occupancy"] == 1.0
    assert gpu["worlds_per_sm"] == 2.0


def test_default_auto_mode_visibly_selects_legacy_owner_outside_native_envelope(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.cuda_architecture_b_request_mode = "auto"
    runner.sim_cfg.generations = 5

    result = runner_loop._try_execute_cuda_architecture_b_run(
        runner,
        start_gen=0,
        end_gen=5,
        skip_init=False,
    )

    assert result is None
    receipt = runner._cuda_architecture_b_route_receipt
    assert receipt["resolved_backend"] == "legacy_python"
    assert receipt["fallback_performed"] is True
    assert receipt["dispatch_started"] is False
    assert receipt["no_hidden_fallback"] is True
    assert receipt["reason"].startswith("outside_qualified_cuda_envelope:")


def test_explicit_cuda_mode_fails_closed_outside_native_envelope(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.cuda_architecture_b_request_mode = "explicit_cuda"
    runner.sim_cfg.generations = 5

    with pytest.raises(
        RuntimeError,
        match="EIEL-NATIVE-ENGINE-PRE-DISPATCH-REJECTED",
    ):
        runner_loop._try_execute_cuda_architecture_b_run(
            runner,
            start_gen=0,
            end_gen=5,
            skip_init=False,
        )


def test_owner_passes_configured_candidate_registry_paths_to_loader(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    registry_path = tmp_path / "candidate_registry.json"
    promotion_path = tmp_path / "candidate_promotion.json"
    runner.sim_cfg.cuda_architecture_b_qualification_registry_path = str(registry_path)
    runner.sim_cfg.cuda_architecture_b_promotion_registry_path = str(promotion_path)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def loader(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise RuntimeError("stop after loader")

    with pytest.raises(RuntimeError, match="stop after loader"):
        owner.execute_cuda_architecture_b_runner_run_v1(
            runner,
            start_gen=0,
            end_gen=4,
            skip_init=False,
            artifact_loader=loader,
        )

    assert calls == [
        (
            (
                Path(runner.sim_cfg.cuda_architecture_b_artifact_path),
                Path(runner.sim_cfg.cuda_architecture_b_qualification_receipt_path),
                Path(runner.sim_cfg.cuda_architecture_b_reporting_receipt_path),
            ),
            {
                "registry_path": str(registry_path),
                "promotion_registry_path": str(promotion_path),
            },
        )
    ]


def test_auto_mode_records_visible_cpp_selection_when_artifact_load_fails(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.cuda_architecture_b_request_mode = "auto"

    with pytest.raises(
        owner.CudaArchitectureBRunnerOwnerError,
        match="failed closed before dispatch",
    ):
        owner.execute_cuda_architecture_b_runner_run_v1(
            runner,
            start_gen=0,
            end_gen=4,
            skip_init=False,
            artifact_loader=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("artifact unavailable")
            ),
        )

    receipt = runner._cuda_architecture_b_route_receipt
    assert receipt["resolved_backend"] == "none"
    assert receipt["fallback_performed"] is False
    assert receipt["no_hidden_fallback"] is True
    assert receipt["reason"].startswith("qualified_cuda_artifact_unavailable:")


def test_live_cpp_cpu_route_fails_closed_before_python_can_own_generation(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.cuda_architecture_b_request_mode = "auto"

    class CppRouteAdapter:
        def __init__(self, _artifact: Any) -> None:
            pass

        def resolve_once(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "resolved_backend": "cpp_cpu",
                "reason": "test_cpp_cpu_selection",
            }

    with pytest.raises(
        owner.CudaArchitectureBRunnerOwnerError,
        match=r"live Phase 1 runner has no configured C\+\+ owner",
    ):
        owner.execute_cuda_architecture_b_runner_run_v1(
            runner,
            start_gen=0,
            end_gen=4,
            skip_init=False,
            artifact_loader=lambda *_args: SimpleNamespace(module=object()),
            adapter_type=CppRouteAdapter,
        )

    receipt = runner._cuda_architecture_b_route_receipt
    assert receipt["resolved_backend"] == "none"
    assert receipt["dispatch_started"] is False
    assert receipt["fallback_performed"] is False
    assert receipt["no_hidden_fallback"] is True
    assert receipt["reason"] == "cpp_cpu_owner_not_configured_for_live_phase1_route"


def test_exact_runner_owner_dispatches_once_and_seals_after_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.chronicle_enable = True
    runner.scientific_chronicle = ScientificChronicleRecorder(runner)
    control = _Control()
    calls: list[str] = []
    prepared = SimpleNamespace(
        physical_window_arguments=(1, 2),
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {
                "metadata": {
                    "policy_weight_mutation_enable": True,
                    "policy_weight_mutation_mode": "gaussian",
                    "policy_weight_mutation_rate": 0.1,
                    "policy_weight_mutation_std": 0.2,
                }
            }
        },
    )
    artifact = SimpleNamespace(module=object(), capability={})

    class Adapter:
        def __init__(self, loaded: Any) -> None:
            assert loaded is artifact

        def resolve_once(self, **kwargs: Any) -> dict[str, Any]:
            calls.append("resolve")
            assert kwargs["request_mode"] == "explicit_cuda"
            return {"resolved_backend": CUDA_BACKEND, "reason": "exact"}

        def execute_prepared(
            self, **kwargs: Any
        ) -> CudaArchitectureBProductionExecutionV2:
            calls.append("dispatch")
            assert kwargs["physical_window_arguments"] == (1, 2)
            assert kwargs["policy_mutation_mode"] == "gaussian"
            return CudaArchitectureBProductionExecutionV2(
                run_id=kwargs["run_id"],
                owner_epoch=1,
                run_control=control,
                direct_result={},
                run_control_manifest=control.manifest(),
                completed_generations=4,
                production_route_receipt={"resolved_backend": CUDA_BACKEND},
            )

    def source_builder(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("prepare")
        return prepared

    def settlement_fn(
        *_args: Any, **_kwargs: Any
    ) -> DirectResidentTerminalSettlementV1:
        calls.append("settle")
        return _settlement()

    def persist(*_args: Any, **_kwargs: Any) -> tuple[Path, Path, dict[str, Any]]:
        calls.append("persist")
        run_root = Path(runner.run_dir)
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = run_root / "manifest.json"
        checkpoint = run_root / "checkpoint"
        checkpoint.mkdir()
        owner._atomic_write(
            manifest, owner._canonical_json_bytes({"status": "waiting"})
        )
        return manifest, checkpoint, {"status": "waiting", "artifacts": {}}

    monkeypatch.setattr(owner, "_persist_outputs", persist)
    result = owner.execute_cuda_architecture_b_runner_run_v1(
        runner,
        start_gen=0,
        end_gen=4,
        skip_init=False,
        artifact_loader=lambda *_args: artifact,
        adapter_type=Adapter,
        source_builder=source_builder,
        settlement_fn=settlement_fn,
    )

    assert calls == ["resolve", "prepare", "dispatch", "settle", "persist"]
    assert result.terminal_status == "run_complete"
    assert result.completed_generations == 4
    assert result.python_generation_dispatches == 0
    assert control.state["sealed"] is True
    assert control.failures == []
    assert runner._current_gen == 3
    assert runner.pop._eiel_current_generation_index == 3
    assert runner.pop._current_generation == 3
    assert (
        Path(runner.run_dir) / "cuda_architecture_b_terminal_receipt.json"
    ).is_file()
    terminal_receipt = json.loads(
        (Path(runner.run_dir) / "cuda_architecture_b_terminal_receipt.json").read_text(
            encoding="ascii"
        )
    )
    assert terminal_receipt["success_marker_allowed"] is False
    assert terminal_receipt["promotion_result_allowed"] is False
    assert terminal_receipt["terminal_result"]["success_marker_allowed"] is False
    output_manifest = json.loads(
        Path(result.output_manifest_path).read_text(encoding="ascii")
    )
    assert output_manifest["chronicle_materialization"]["status"] == "complete"
    assert output_manifest["artifacts"]["scientific_chronicle"]["path"] == (
        "chronicle/chronicle_manifest.json"
    )
    assert (Path(runner.run_dir) / "chronicle" / "chronicle_manifest.json").is_file()
    assert not (Path(runner.run_dir) / "cuda_architecture_b_run_complete.json").exists()


def test_runner_owner_resumes_from_tick0_source_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    source_checkpoint = {"metadata": {"generation": 4, "tick": 0}}
    runner.cuda_architecture_b_source_checkpoint_projection = source_checkpoint
    control = _Control()
    control.expected_frontier = 7
    control.state["active_generation"] = 7
    control.state["last_boundary"] = {"trajectory_commit_frontier": 7}
    artifact = SimpleNamespace(module=object(), capability={})
    calls: list[str] = []

    class Adapter:
        def __init__(self, _artifact: Any) -> None: ...

        def resolve_once(self, **_kwargs: Any) -> dict[str, Any]:
            calls.append("resolve")
            return {"resolved_backend": CUDA_BACKEND}

        def execute_prepared(
            self, **kwargs: Any
        ) -> CudaArchitectureBProductionExecutionV2:
            calls.append("dispatch")
            return CudaArchitectureBProductionExecutionV2(
                run_id=kwargs["run_id"],
                owner_epoch=1,
                run_control=control,
                direct_result={},
                run_control_manifest=control.manifest(),
                completed_generations=4,
                production_route_receipt={"resolved_backend": CUDA_BACKEND},
            )

    def source_builder(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append("prepare")
        assert kwargs["source_checkpoint_projection"] is source_checkpoint
        return SimpleNamespace(
            physical_window_arguments=(),
            checkpoint_projection_inputs={
                "pre_generation_canonical_projection": {
                    "metadata": {
                        "policy_weight_mutation_enable": False,
                        "policy_weight_mutation_mode": "none",
                        "policy_weight_mutation_rate": 0.0,
                        "policy_weight_mutation_std": 0.0,
                    }
                }
            },
        )

    def persist(*_args: Any, **_kwargs: Any) -> tuple[Path, Path, dict[str, Any]]:
        calls.append("persist")
        run_root = Path(runner.run_dir)
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = run_root / "manifest.json"
        checkpoint = run_root / "checkpoint"
        checkpoint.mkdir()
        owner._atomic_write(
            manifest, owner._canonical_json_bytes({"status": "waiting"})
        )
        return manifest, checkpoint, {"status": "waiting", "artifacts": {}}

    monkeypatch.setattr(owner, "_persist_outputs", persist)
    result = owner.execute_cuda_architecture_b_runner_run_v1(
        runner,
        start_gen=4,
        end_gen=8,
        skip_init=True,
        artifact_loader=lambda *_args: artifact,
        adapter_type=Adapter,
        source_builder=source_builder,
        settlement_fn=lambda *_args, **_kwargs: _settlement(
            4,
            start_generation=4,
        ),
    )

    assert calls == ["resolve", "prepare", "dispatch", "persist"]
    assert result.terminal_status == "run_complete"
    assert result.completed_generations == 4
    assert runner._current_gen == 7


def test_cancelled_prefix_acknowledges_only_after_checkpoint_and_manifest_are_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    calls: list[str] = []

    class CancellationControl(_Control):
        def __init__(self) -> None:
            super().__init__()
            self.state.update(
                {
                    "active_generation": 1,
                    "last_boundary": {"trajectory_commit_frontier": 1},
                    "cancel_requested": True,
                    "awaiting_cancellation_checkpoint": True,
                    "awaiting_final_output_ack": False,
                }
            )

        def acknowledge_cancellation_checkpoint(
            self, controller_id: str, owner_epoch: int, boundary: dict[str, Any]
        ) -> dict[str, Any]:
            calls.append("ack_cancel")
            root = Path(runner.run_dir)
            assert (root / "checkpoint" / "manifest.json").is_file()
            waiting = json.loads((root / "manifest.json").read_text(encoding="ascii"))
            assert waiting["status"] == "durable_waiting_native_ack"
            assert controller_id == self.state["controller_id"]
            assert owner_epoch == self.state["owner_epoch"]
            assert boundary["snapshot_frontier"] == 1
            assert boundary["cancellation_checkpoint_requested"] is True
            assert boundary["cancellation_checkpoint_persisted"] is True
            self.terminal = {
                "terminal_status": "cancelled",
                "sealed": True,
                "success_marker_allowed": False,
                "promotion_result_allowed": False,
            }
            self.state.update(
                {
                    "status": "cancelled",
                    "sealed": True,
                    "awaiting_cancellation_checkpoint": False,
                }
            )
            return dict(self.terminal)

        def acknowledge_run_complete(self, *_args: Any) -> dict[str, Any]:
            raise AssertionError("cancelled prefix must not use completion ack")

    control = CancellationControl()
    artifact = SimpleNamespace(module=object(), capability={})
    prepared = SimpleNamespace(
        physical_window_arguments=(),
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {"metadata": {}}
        },
    )

    class Adapter:
        def __init__(self, _artifact: Any) -> None: ...

        def resolve_once(self, **_kwargs: Any) -> dict[str, Any]:
            return {"resolved_backend": CUDA_BACKEND}

        def execute_prepared(
            self, **kwargs: Any
        ) -> CudaArchitectureBProductionExecutionV2:
            calls.append("dispatch")
            return CudaArchitectureBProductionExecutionV2(
                run_id=kwargs["run_id"],
                owner_epoch=1,
                run_control=control,
                direct_result={},
                run_control_manifest=control.manifest(),
                completed_generations=2,
                production_route_receipt={"resolved_backend": CUDA_BACKEND},
            )

    def persist(*_args: Any, **_kwargs: Any) -> tuple[Path, Path, dict[str, Any]]:
        calls.append("persist_checkpoint_and_manifest")
        root = Path(runner.run_dir)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir(parents=True)
        owner._atomic_write(
            checkpoint / "manifest.json",
            owner._canonical_json_bytes({"status": "complete", "generation": 2}),
        )
        manifest = root / "manifest.json"
        payload = {"status": "durable_waiting_native_ack", "artifacts": {}}
        owner._atomic_write(manifest, owner._canonical_json_bytes(payload))
        return manifest, checkpoint, payload

    monkeypatch.setattr(owner, "_persist_outputs", persist)
    result = owner.execute_cuda_architecture_b_runner_run_v1(
        runner,
        start_gen=0,
        end_gen=4,
        skip_init=False,
        artifact_loader=lambda *_args: artifact,
        adapter_type=Adapter,
        source_builder=lambda *_args, **_kwargs: prepared,
        settlement_fn=lambda *_args, **_kwargs: _settlement(2),
    )

    assert calls == ["dispatch", "persist_checkpoint_and_manifest", "ack_cancel"]
    assert result.terminal_status == "cancelled"
    assert result.completed_generations == 2
    terminal_receipt = json.loads(
        (Path(runner.run_dir) / "cuda_architecture_b_terminal_receipt.json").read_text(
            encoding="ascii"
        )
    )
    output_manifest = json.loads(
        Path(result.output_manifest_path).read_text(encoding="ascii")
    )
    assert terminal_receipt["status"] == "cancelled"
    assert (
        terminal_receipt["acknowledged_boundary"]["cancellation_checkpoint_persisted"]
        is True
    )
    assert output_manifest["status"] == "complete"
    assert output_manifest["terminal_status"] == "cancelled"


def test_runner_cancellation_api_delegates_only_to_active_native_adapter() -> None:
    calls: list[dict[str, Any]] = []

    class Adapter:
        def request_cancel(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(dict(kwargs))
            return {"disposition": "accepted", "request_id": kwargs["request_id"]}

    runner = SimpleNamespace(_cuda_architecture_b_production_adapter=Adapter())
    receipt = owner.request_cuda_architecture_b_runner_cancellation_v1(
        runner,
        request_id="cancel-runner-1",
        reason="supervisor_requested",
        timeout_seconds=2.0,
    )

    assert receipt == {"disposition": "accepted", "request_id": "cancel-runner-1"}
    assert calls == [
        {
            "request_id": "cancel-runner-1",
            "reason": "supervisor_requested",
            "timeout_seconds": 2.0,
        }
    ]


def test_dispatch_failure_persists_native_terminal_receipt_and_failed_manifest(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    artifact = SimpleNamespace(module=object(), capability={})
    prepared = SimpleNamespace(
        physical_window_arguments=(),
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {"metadata": {}}
        },
    )
    failed_manifest = {"status": "failed", "sealed": True}
    failed_terminal = {"terminal_status": "failed", "sealed": True}

    class Adapter:
        def __init__(self, _artifact: Any) -> None: ...

        def resolve_once(self, **_kwargs: Any) -> dict[str, Any]:
            return {"resolved_backend": CUDA_BACKEND}

        def execute_prepared(self, **kwargs: Any) -> Any:
            raise owner.CudaArchitectureBProductionDispatchError(
                "injected dispatch failure",
                run_id=kwargs["run_id"],
                production_route_receipt={
                    "resolved_backend": CUDA_BACKEND,
                    "dispatch_started": True,
                    "fallback_performed": False,
                },
                native_failure_receipt={
                    "disposition": "accepted",
                    "terminal_status": "failed",
                },
                run_control_manifest=failed_manifest,
                native_terminal_result=failed_terminal,
                failure_settlement_error=None,
            )

    with pytest.raises(
        owner.CudaArchitectureBRunnerOwnerError,
        match="failure receipt and manifest were persisted",
    ):
        owner.execute_cuda_architecture_b_runner_run_v1(
            runner,
            start_gen=0,
            end_gen=4,
            skip_init=False,
            artifact_loader=lambda *_args: artifact,
            adapter_type=Adapter,
            source_builder=lambda *_args, **_kwargs: prepared,
        )

    root = Path(runner.run_dir)
    failure = json.loads(
        (root / "cuda_architecture_b_run_failed.json").read_text(encoding="ascii")
    )
    manifest = json.loads(
        (root / "cuda_architecture_b_output_manifest.json").read_text(encoding="ascii")
    )
    assert failure["phase"] == "dispatch"
    assert failure["native_failure_settlement_exact"] is True
    assert failure["run_control_manifest"] == failed_manifest
    assert failure["native_terminal_result"] == failed_terminal
    assert manifest["status"] == "failed"
    assert manifest["terminal_status"] == "failed"
    assert manifest["success_marker_allowed"] is False
    assert manifest["promotion_result_allowed"] is False
    assert manifest["no_hidden_fallback"] is True


def test_post_dispatch_settlement_failure_is_terminal_and_never_falls_back(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    control = _Control()
    artifact = SimpleNamespace(module=object(), capability={})
    prepared = SimpleNamespace(
        physical_window_arguments=(),
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {"metadata": {}}
        },
    )

    class Adapter:
        def __init__(self, _artifact: Any) -> None: ...

        def resolve_once(self, **_kwargs: Any) -> dict[str, Any]:
            return {"resolved_backend": CUDA_BACKEND}

        def execute_prepared(
            self, **kwargs: Any
        ) -> CudaArchitectureBProductionExecutionV2:
            return CudaArchitectureBProductionExecutionV2(
                run_id=kwargs["run_id"],
                owner_epoch=1,
                run_control=control,
                direct_result={},
                run_control_manifest=control.manifest(),
                completed_generations=4,
                production_route_receipt={"resolved_backend": CUDA_BACKEND},
            )

    with pytest.raises(
        owner.CudaArchitectureBRunnerOwnerError,
        match="fallback is prohibited",
    ):
        owner.execute_cuda_architecture_b_runner_run_v1(
            runner,
            start_gen=0,
            end_gen=4,
            skip_init=False,
            artifact_loader=lambda *_args: artifact,
            adapter_type=Adapter,
            source_builder=lambda *_args, **_kwargs: prepared,
            settlement_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected settlement failure")
            ),
        )

    assert len(control.failures) == 1
    assert control.final_result()["terminal_status"] == "failed"
    root = Path(runner.run_dir)
    failure = json.loads(
        (root / "cuda_architecture_b_run_failed.json").read_text(encoding="ascii")
    )
    manifest = json.loads(
        (root / "cuda_architecture_b_output_manifest.json").read_text(encoding="ascii")
    )
    assert failure["phase"] == "settlement"
    assert failure["native_failure_settlement_exact"] is True
    assert manifest["status"] == "failed"
    assert manifest["terminal_status"] == "failed"


def test_runner_loop_places_native_run_owner_before_python_generation_loop() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/simulation/runner_pkg/runner_loop.py"
    ).read_text(encoding="utf-8")

    hook = source.index("_cuda_architecture_b_result = _try_execute")
    legacy_loop = source.index("for gen in range(int(start_gen), int(end_gen))")
    assert hook < legacy_loop
    assert "return _cuda_architecture_b_result" in source[hook:legacy_loop]
