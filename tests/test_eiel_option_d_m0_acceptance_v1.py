from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from tools.scripts import eiel_option_d_m0_acceptance_v1 as subject


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )


def _write_mode(
    root: Path,
    *,
    mode: str,
    enabled: bool,
    source_commit: str,
    source_snapshot: str,
) -> tuple[Path, Path, Path | None]:
    build = root / "builds" / mode
    receipts = root / "receipts" / mode
    build.mkdir(parents=True)
    receipts.mkdir(parents=True)
    artifact = build / subject.EXTENSION_NAME
    artifact.write_bytes((mode + "-binary").encode("ascii"))
    manifest = Path(f"{artifact}.manifest.json")
    shared_manifest = {
        "schema_version": subject.MANIFEST_SCHEMA_VERSION,
        "module_name": subject.EXTENSION_NAME.removesuffix(".so"),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": _sha256(artifact),
        "cuda_arch": "sm_75",
        "compiler_binary_sha256": "1" * 64,
        "compiler_version": "g++ test",
        "nvcc_binary_sha256": "2" * 64,
        "nvcc_version": "nvcc test",
        "torch_version": "2.10.0+cu128",
        "torch_cuda_version": "12.8",
        "qualified_cache_width": {
            "schema_version": "eiel_cuda_architecture_b_qualified_cache_width_v1",
            "value": 2,
            "compile_definition": (
                "-DEIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID=(cache_width==2)"
            ),
        },
        "cxx_flags": [
            "-O2",
            f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={int(enabled)}",
        ],
        "nvcc_flags": [
            "-O2",
            f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={int(enabled)}",
        ],
        "source_hashes": {"source.cu": "3" * 64},
        "primary_sources": ["source.cu"],
        "relevant_linked_library_sha256": "4" * 64,
        "direct_link_library_identities": [{"link_name": "torch", "sha256": "5" * 64}],
        "committed_source_snapshot": {
            "build_source_commit": source_commit,
            "compiled_source_snapshot_sha256": source_snapshot,
        },
        "option_d_m0_telemetry_build": {
            "schema_version": "eiel_option_d_m0_telemetry_build_identity_v1",
            "compile_definition": (
                f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={int(enabled)}"
            ),
            "option_d_m0_telemetry_enabled": enabled,
        },
    }
    _write_json(manifest, shared_manifest)
    qualification = receipts / "qualification.json"
    _write_json(
        qualification,
        {
            "schema_version": "eiel_production_engine_mechanic_qualification_v1",
            "ok": True,
            "provider_id": subject.PROVIDER_ID,
            "profile_id": subject.PROFILE_ID,
            "artifact_abi": subject.ARTIFACT_ABI,
            "artifact": {
                "path": str(artifact.resolve()),
                "sha256": _sha256(artifact),
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": _sha256(manifest),
            },
        },
    )
    _write_json(
        receipts / "reporting.json",
        {
            "schema_version": "eiel_production_engine_reporting_qualification_v1",
            "ok": True,
            "provider_id": subject.PROVIDER_ID,
            "profile_id": subject.PROFILE_ID,
            "artifact_sha256": _sha256(artifact),
        },
    )
    source_environment: Path | None = None
    if enabled:
        source_environment = root / "source_environment.json"
        _write_json(
            source_environment,
            {
                "schema_version": subject.SOURCE_ENVIRONMENT_SCHEMA_VERSION,
                "production_interpreter": {"status": "PASS"},
                "route_scope_contract": {
                    "accepted_eiel_commit": source_commit,
                    "compiled_source_snapshot_sha256": source_snapshot,
                    "passed_for_measurement_identity": True,
                    "production_integration_authority": False,
                },
            },
        )
    return build, receipts, source_environment


def _options(tmp_path: Path) -> subject.HarnessOptions:
    source_commit = "a" * 40
    source_snapshot = "b" * 64
    off_build, off_receipts, _ = _write_mode(
        tmp_path,
        mode="telemetry-off",
        enabled=False,
        source_commit=source_commit,
        source_snapshot=source_snapshot,
    )
    on_build, on_receipts, source_environment = _write_mode(
        tmp_path,
        mode="telemetry-on",
        enabled=True,
        source_commit=source_commit,
        source_snapshot=source_snapshot,
    )
    assert source_environment is not None
    return subject.HarnessOptions(
        off_build_dir=off_build,
        on_build_dir=on_build,
        off_receipt_dir=off_receipts,
        on_receipt_dir=on_receipts,
        on_source_environment=source_environment,
        output_root=tmp_path / "output",
        python_executable=Path(subject.sys.executable),
    )


def _native_receipt(*, ordinary: int = 6, turnover: int = 15) -> dict[str, Any]:
    native = {name: 1 for name in subject._NATIVE_COMPONENT_FIELDS}
    native.update(
        {
            "option_d_m0_telemetry_enabled": True,
            "policy_birth_segment_refill_receipts": [{"ordinal": 0}],
            "gaussian_segment_refill_receipts": [{"ordinal": 0}],
        }
    )
    primary = {name: 1 for name in subject._PRIMARY_COMPONENT_FIELDS}
    primary.update(
        {
            "schema_version": subject.PRIMARY_RECEIPT_SCHEMA_VERSION,
            "option_d_m0_telemetry_enabled": True,
            "actual_ordinary_birth_count": ordinary,
            "actual_turnover_birth_count": turnover,
            "runtime_plan_binding": {
                "status": "PASS_EXACT_PROVIDER_RUNTIME_PLAN_BINDING",
                "physical_window_generations": 1,
                "requested_generations": 3,
                "ticks_per_generation": 160,
                "world_grid_size": 32,
                "production_integration_authority": False,
            },
            "cpu_policy_service_compact_summary": {
                "calls_inside_tick_loop": 480,
                "device_to_host_operations": 960,
                "host_to_device_operations": 960,
            },
            "resident_window_transfer_accounting": {
                "schema_version": (
                    "eiel_option_d_m0_resident_window_transfer_accounting_v1"
                ),
                "resident_window_outer_boundary_device_to_host_operations_lower_bound": 3,
            },
        }
    )
    native["option_d_rng_instrumentation_receipt"] = primary
    return native


def _fake_worker_runner(
    *,
    mutate: Callable[[str, dict[str, Any]], None] | None = None,
    ordinary: int = 6,
    turnover: int = 15,
    publish_route: bool = False,
) -> subject.WorkerRunner:
    def run(
        *,
        mode: str,
        plan_path: Path,
        run_root: Path,
        worker_result_path: Path,
        python_executable: Path,
        timeout_seconds: float,
    ) -> subject.WorkerInvocation:
        del python_executable, timeout_seconds
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        cell_id = plan["cells"][0]["cell_id"]
        digest_maps = {cell_id: "d" * 64}
        native_receipt = (
            {"option_d_m0_telemetry_enabled": False}
            if mode == "telemetry-off"
            else _native_receipt(ordinary=ordinary, turnover=turnover)
        )
        batch_manifest: dict[str, Any] = {
            "schema_version": "eiel_campaign_batch_manifest_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "profile_id": subject.PROFILE_ID,
            "provider_id": subject.PROVIDER_ID,
            "execution_route_sha256": ("0" if mode == "telemetry-off" else "1") * 64,
            "batch_id": "batch-0000",
            "cell_ids": [cell_id],
            "status": "completed",
            "attempts": 1,
            "checkpoint": {
                "generation": 3,
                **{
                    field: dict(digest_maps)
                    for field in subject._PARITY_CHECKPOINT_FIELDS
                },
            },
            "metrics_rows": [
                {
                    "cell_id": cell_id,
                    "generation": 2,
                    "metrics": {"population.alive": 512},
                }
            ],
            "result": {
                "terminal_generation": 3,
                "native_execution_receipts": [native_receipt],
            },
            "error": None,
        }
        if mutate is not None:
            mutate(mode, batch_manifest)
        _write_json(
            run_root / "preflight_receipt.json",
            {
                "schema_version": "eiel_campaign_preflight_v1",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "qualification_mode": True,
                "ready_for_route_qualification": True,
                "ready_for_execution": False,
            },
        )
        _write_json(run_root / "batches/batch-0000/manifest.json", batch_manifest)
        execution_receipt = {
            "ok": True,
            "status": "completed",
            "execution_route_sha256": batch_manifest["execution_route_sha256"],
        }
        _write_json(
            worker_result_path,
            {
                "schema_version": subject.WORKER_SCHEMA_VERSION,
                "mode": mode,
                "ok": True,
                "plan_sha256": plan["plan_sha256"],
                "execution_wall_time_ns": 100 if mode == "telemetry-off" else 110,
                "execution_receipt": execution_receipt,
            },
        )
        if publish_route:
            _write_json(run_root / "execution_route_qualification.json", {"ok": True})
        return subject.WorkerInvocation(
            mode=mode,
            command=("python", "-m", "worker"),
            subprocess_wall_time_ns=200 if mode == "telemetry-off" else 220,
            worker_execution_wall_time_ns=(100 if mode == "telemetry-off" else 110),
            stdout="",
            stderr="",
        )

    return run


def test_harness_builds_one_exact_logical_plan_and_emits_acceptance(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)

    receipt = subject.run_harness(options, worker_runner=_fake_worker_runner())

    assert receipt["passed"] is True
    assert receipt["production_integration_authority"] is False
    assert receipt["route_qualification_published"] is False
    assert receipt["telemetry_on_birth_evidence"]["actual_ordinary_birth_count"] == 6
    assert receipt["telemetry_on_birth_evidence"]["actual_turnover_birth_count"] == 15
    assert receipt["telemetry_on_birth_evidence"]["both_required_and_observed"] is True
    assert receipt["telemetry_on_birth_evidence"]["birth_surfaces"] == {
        "ordinary-surface": {
            "actual_ordinary_birth_count": 6,
            "actual_turnover_birth_count": 15,
        },
        "turnover-surface": {
            "actual_ordinary_birth_count": 6,
            "actual_turnover_birth_count": 15,
        },
    }
    off_plan = json.loads(
        (options.output_root / "telemetry-off/execution_plan.json").read_text()
    )
    on_plan = json.loads(
        (options.output_root / "telemetry-on/execution_plan.json").read_text()
    )
    assert off_plan["cells"] == on_plan["cells"]
    assert off_plan["cells"][0]["parameters"] == {
        "cuda_architecture_b.cpu_request_staging_enable": True,
        "cuda_architecture_b.cpython_gauss_segment_capacity": 65_536,
        "cuda_architecture_b.policy_birth_segment_capacity": -1,
        "cuda_architecture_b.policy_birth_segment_refill_enable": True,
        "cuda_architecture_b.turnover_mlp_shadow_elide_enable": True,
        "patch.capacity": 512,
        "patch.founder_placement": "legacy",
        "patch.grid_size": 32,
        "patch.initial_population": 256,
    }
    assert off_plan["execution"]["physical_window_generations"] == 1
    assert off_plan["execution"]["generations"] == 3
    assert off_plan["execution"]["ticks_per_generation"] == 160
    assert subject.SOURCE_ENVIRONMENT_ARTIFACT not in off_plan["artifacts"]
    assert subject.SOURCE_ENVIRONMENT_ARTIFACT in on_plan["artifacts"]
    turnover_plan = json.loads(
        (
            options.output_root
            / "turnover-surface/telemetry-off/execution_plan.json"
        ).read_text()
    )
    assert turnover_plan["campaign_id"] == f"{options.campaign_id}-turnover"
    assert turnover_plan["cells"][0]["parameters"][
        "production_lifecycle.ordinary_births_enabled"
    ] is False
    assert turnover_plan["cells"][0]["parameters"][
        "production_lifecycle.turnover_cull_fraction"
    ] == 0.5
    assert (options.output_root / "m0_component_profile.json").is_file()
    assert (options.output_root / "m0_instrumentation_overhead_receipt.json").is_file()


@pytest.mark.parametrize(
    ("ordinary", "turnover"),
    [(0, 15), (6, 0)],
)
def test_harness_requires_both_birth_classes_and_fails_closed(
    tmp_path: Path, ordinary: int, turnover: int
) -> None:
    options = _options(tmp_path)

    with pytest.raises(subject.M0AcceptanceError, match="required birth class"):
        subject.run_harness(
            options,
            worker_runner=_fake_worker_runner(
                ordinary=ordinary,
                turnover=turnover,
            ),
        )

    for name in (
        "m0_acceptance_receipt.json",
        "m0_instrumentation_overhead_receipt.json",
        "m0_component_profile.json",
    ):
        failed = json.loads((options.output_root / name).read_text())
        assert failed["passed"] is False
        assert failed["production_integration_authority"] is False


def test_harness_fails_on_terminal_or_metrics_difference(tmp_path: Path) -> None:
    options = _options(tmp_path)

    def mutate(mode: str, manifest: dict[str, Any]) -> None:
        if mode == "telemetry-on":
            manifest["metrics_rows"][0]["metrics"]["population.alive"] = 511

    with pytest.raises(subject.M0AcceptanceError, match="parity differs"):
        subject.run_harness(
            options,
            worker_runner=_fake_worker_runner(mutate=mutate),
        )


def test_harness_rejects_any_route_publication_artifact(tmp_path: Path) -> None:
    options = _options(tmp_path)

    with pytest.raises(subject.M0AcceptanceError, match="published route authority"):
        subject.run_harness(
            options,
            worker_runner=_fake_worker_runner(publish_route=True),
        )


def test_harness_refuses_existing_output_root(tmp_path: Path) -> None:
    options = _options(tmp_path)
    options.output_root.mkdir()

    with pytest.raises(subject.M0AcceptanceError, match="already exists"):
        subject.run_harness(options, worker_runner=_fake_worker_runner())


def test_validated_options_preserve_venv_python_symlink(tmp_path: Path) -> None:
    options = _options(tmp_path)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(Path(sys.executable).resolve())
    options = subject.HarnessOptions(
        off_build_dir=options.off_build_dir,
        on_build_dir=options.on_build_dir,
        off_receipt_dir=options.off_receipt_dir,
        on_receipt_dir=options.on_receipt_dir,
        on_source_environment=options.on_source_environment,
        output_root=options.output_root,
        python_executable=venv_python,
    )

    validated = subject._validated_options(options)

    assert validated.python_executable == venv_python.absolute()
    assert validated.python_executable != venv_python.resolve()


def test_build_twins_fail_if_nontelemetry_identity_differs(tmp_path: Path) -> None:
    options = _options(tmp_path)
    manifest_path = options.on_build_dir / f"{subject.EXTENSION_NAME}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cuda_arch"] = "sm_80"
    _write_json(manifest_path, manifest)
    qualification_path = options.on_receipt_dir / "qualification.json"
    qualification = json.loads(qualification_path.read_text())
    qualification["artifact"]["manifest_sha256"] = _sha256(manifest_path)
    _write_json(qualification_path, qualification)

    with pytest.raises(subject.M0AcceptanceError, match="build twins differ"):
        subject.run_harness(options, worker_runner=_fake_worker_runner())


@pytest.mark.parametrize(
    ("compute_row", "expected_idle"),
    (
        ("101, /usr/libexec/gnome-remote-desktop-daemon, 268", True),
        ("202, /usr/bin/python3.14, 1024", False),
    ),
)
def test_idle_gpu_probe_distinguishes_desktop_context_from_cuda_workload(
    monkeypatch: pytest.MonkeyPatch,
    compute_row: str,
    expected_idle: bool,
) -> None:
    def fake_run(command: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        stdout = (
            "2, 984\n"
            if "--query-gpu=utilization.gpu,memory.used" in command
            else f"{compute_row}\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)

    if expected_idle:
        receipt = subject._idle_gpu_receipt()
        assert receipt["idle"] is True
        assert receipt["blocking_compute_rows"] == []
    else:
        with pytest.raises(subject.M0AcceptanceError, match="not idle enough"):
            subject._idle_gpu_receipt()


def test_worker_uses_only_bounded_preflight_and_qualification_mode(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "worker"
    run_root.mkdir()
    plan = {
        "schema_version": "test-plan",
        "campaign_id": "m0-test",
        "resolved_profile": subject.PROFILE_ID,
    }
    plan["plan_sha256"] = subject._canonical_sha256(plan)
    plan_path = run_root / "execution_plan.json"
    _write_json(plan_path, plan)
    calls: list[tuple[str, Any]] = []

    class Registry:
        provider = object()

        def get(self, profile_id: str) -> object | None:
            calls.append(("get", profile_id))
            return self.provider if profile_id == subject.PROFILE_ID else None

    def preflight(provider: object, supplied: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(("preflight", provider))
        return {
            "schema_version": "eiel_campaign_preflight_v1",
            "campaign_id": supplied["campaign_id"],
            "plan_sha256": supplied["plan_sha256"],
            "qualification_mode": True,
            "ready_for_route_qualification": True,
            "ready_for_execution": False,
        }

    def execute(
        supplied: Mapping[str, Any],
        *,
        run_root: Path,
        provider_registry: Registry,
        qualification_mode: bool,
    ) -> Mapping[str, Any]:
        calls.append(
            (
                "execute",
                {
                    "plan": supplied,
                    "run_root": run_root,
                    "registry": provider_registry,
                    "qualification_mode": qualification_mode,
                },
            )
        )
        return {"ok": True, "status": "completed"}

    result = subject._execute_worker(
        mode="telemetry-on",
        plan_path=plan_path,
        run_root=run_root,
        worker_result_path=run_root / "worker_result.json",
        provider_registry_factory=Registry,
        preflight_runner=preflight,
        campaign_executor=execute,
    )

    assert result["qualification_mode"] is True
    execution = next(value for name, value in calls if name == "execute")
    assert execution["qualification_mode"] is True
    assert execution["run_root"] == run_root.resolve()
    source = Path(subject.__file__).read_text(encoding="utf-8")
    forbidden = "execute" + "_route_qualification_v1"
    assert forbidden not in source


def test_default_runner_invokes_fresh_module_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EIEL_NATIVE_RUNTIME_EXTENSION", raising=False)
    plan_path = tmp_path / "execution_plan.json"
    plan_path.write_text("{}", encoding="ascii")
    worker_result = tmp_path / "worker_result.json"
    observed: dict[str, Any] = {}

    def fake_run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = tuple(command)
        observed["kwargs"] = kwargs
        _write_json(worker_result, {"execution_wall_time_ns": 7})
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", fake_run)

    invocation = subject._default_worker_runner(
        mode="telemetry-off",
        plan_path=plan_path,
        run_root=tmp_path,
        worker_result_path=worker_result,
        python_executable=Path(subject.sys.executable),
        timeout_seconds=10,
    )

    assert observed["command"][1:3] == (
        "-m",
        "tools.scripts.eiel_option_d_m0_acceptance_v1",
    )
    assert observed["kwargs"]["cwd"] == subject.ROOT
    assert observed["kwargs"]["env"]["PYTHONHASHSEED"] == "0"
    assert invocation.worker_execution_wall_time_ns == 7
