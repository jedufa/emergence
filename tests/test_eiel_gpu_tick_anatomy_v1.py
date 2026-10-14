from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scripts.eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1 import (
    AllocationDiagnosticError,
)
from tools.scripts.eiel_gpu_tick_anatomy_analyze_v1 import (
    COMMIT,
    CORE,
    PREPARE,
    _stats,
    _tick_groups,
)
from tools.scripts.eiel_gpu_tick_anatomy_v1 import (
    PRODUCTION_HOST_FILES,
    _bind_artifacts,
    _comparison_identity,
    _competing_gpu_processes,
    _release_host_identity,
    _result_summary,
    _translation_compatible_plan,
    _validated_plan,
)
from engine.campaigns.canonical import canonical_sha256


def _plan() -> dict[str, object]:
    payload: dict[str, object] = {
        "resolved_profile": "eiel-production-runtime-v1",
        "physical_batches": [
            {
                "batch_id": "batch-0000",
                "cell_ids": [f"cell-{index:02d}" for index in range(48)],
            }
        ],
        "execution": {
            "execution_route_identity": "tick_native_execution_route_v1",
            "internal_lifecycle_schedule": {"interval_ticks": 160},
        },
        "artifacts": {
            "cuda_extension": {
                "abi": "architecture_b_wdl_jungle_eed_fixed_mlp_native_v1",
            },
            "extension_manifest": {
                "abi": "architecture_b_wdl_jungle_eed_fixed_mlp_native_v1",
            },
        },
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}


def _m7_build_identity(enabled: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "eiel_option_d_m7_batched_cpu_policy_refill_build_identity_v1",
        "compile_time_macro": "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED",
        "receipt_field": "option_d_m7_batched_cpu_policy_refill_enabled",
        "default_enabled": False,
        "option_d_m7_batched_cpu_policy_refill_enabled": enabled,
        "compile_definition": (
            "-DEIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED="
            f"{1 if enabled else 0}"
        ),
        "authority": (
            "candidate_batched_exact_cpu_lsb0_policy_refill" if enabled else "disabled"
        ),
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
        "runtime_resource_receipt_sha256": "a" * 64,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _m8_build_identity(enabled: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "eiel_option_d_m8_resident_policy_suffix_reuse_build_identity_v1",
        "compile_time_macro": "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED",
        "receipt_field": "option_d_m8_resident_policy_suffix_reuse_enabled",
        "default_enabled": False,
        "option_d_m8_resident_policy_suffix_reuse_enabled": enabled,
        "compile_definition": (
            "-DEIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED="
            f"{1 if enabled else 0}"
        ),
        "authority": "candidate_resident_policy_suffix_reuse"
        if enabled
        else "disabled",
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
        "runtime_resource_receipt_sha256": "b" * 64,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _m8_runtime_identity(*, enabled: bool, reused: bool = False) -> dict[str, object]:
    return {
        "option_d_m8_resident_policy_suffix_reuse_enabled": enabled,
        "option_d_m8_resident_policy_suffix_reuse_used": reused,
        "option_d_m8_resident_policy_suffix_reuse_authority": (
            "candidate_resident_policy_suffix_reuse"
            if reused
            else (
                "compiled_candidate_physical_refill_fallback" if enabled else "disabled"
            )
        ),
        "option_d_m8_resident_policy_suffix_reuse_schema_version": (
            "eiel_option_d_m8_resident_policy_suffix_reuse_v1" if enabled else None
        ),
        "resident_window_reused": reused,
        "physical_refill_performed": not reused,
    }


def test_validated_plan_accepts_exact_production_b48_route(tmp_path: Path) -> None:
    plan_path = tmp_path / "execution_plan.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")

    assert _validated_plan(plan_path)["resolved_profile"] == (
        "eiel-production-runtime-v1"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan.update(resolved_profile="diagnostic"), "production"),
        (
            lambda plan: plan["physical_batches"][0].update(cell_ids=["cell-00"]),
            "B48",
        ),
        (
            lambda plan: plan["execution"].update(
                execution_route_identity="legacy_generation_route_v1"
            ),
            "production",
        ),
        (
            lambda plan: plan["artifacts"]["cuda_extension"].update(abi="other"),
            "different native ABI",
        ),
    ],
)
def test_validated_plan_rejects_route_drift(
    tmp_path: Path, mutation: object, message: str
) -> None:
    plan = _plan()
    mutation(plan)
    plan_path = tmp_path / "execution_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(AllocationDiagnosticError, match=message):
        _validated_plan(plan_path)


def test_result_summary_requires_exact_native_tick_ownership() -> None:
    ticks = 24
    result = {
        "owner": "native_extension",
        "absolute_tick_span_native_owner": True,
        "native_tick_iterations": ticks,
        "tick_span_start": 0,
        "tick_span_end": ticks,
        "tick_span_count": ticks,
        "terminal_generation_boundary": False,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
        "turnover_shadow_plan_launches": 0,
        "turnover_cull_refill_plan_launches": 0,
        "turnover_commit_launches": 0,
        "turnover_publish_launches": 0,
        "phase_1_launches": ticks,
        "option_d_m7_batched_cpu_policy_refill_enabled": False,
        "option_d_m7_batched_cpu_policy_refill_used": False,
        "option_d_m7_batched_cpu_policy_refill_authority": "disabled",
        "option_d_m7_batched_cpu_policy_refill_schema_version": None,
        "policy_birth_segment_refill_receipts": [
            {
                "option_d_m7_batched_cpu_policy_refill_enabled": False,
                "option_d_m7_batched_cpu_policy_refill_used": False,
                "option_d_m7_batched_cpu_policy_refill_authority": "disabled",
                "option_d_m7_batched_cpu_policy_refill_schema_version": None,
                **_m8_runtime_identity(enabled=False),
            }
        ],
    }

    assert (
        _result_summary(result, ticks=ticks, expected_m7_enabled=False)[
            "phase_1_launches"
        ]
        == ticks
    )
    assert _result_summary(result, ticks=ticks, expected_m7_enabled=False)[
        "turnover_commit_launches"
    ] == 0

    result["native_tick_iterations"] = ticks - 1
    with pytest.raises(AllocationDiagnosticError, match="ownership contract"):
        _result_summary(result, ticks=ticks, expected_m7_enabled=False)


def test_release_host_identity_requires_exact_sealed_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    working_root = tmp_path / "working"
    release_root = tmp_path / "release"
    for relative in PRODUCTION_HOST_FILES:
        working_path = working_root / relative
        release_path = release_root / "payload" / "python" / relative
        working_path.parent.mkdir(parents=True, exist_ok=True)
        release_path.parent.mkdir(parents=True, exist_ok=True)
        working_path.write_text(relative, encoding="utf-8")
        release_path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(
        "tools.scripts.eiel_gpu_tick_anatomy_v1.RUNTIME_SOURCE_ROOT", working_root
    )

    assert len(_release_host_identity(release_root)) == len(PRODUCTION_HOST_FILES)

    (release_root / "payload" / "python" / PRODUCTION_HOST_FILES[0]).write_text(
        "drift", encoding="utf-8"
    )
    with pytest.raises(AllocationDiagnosticError, match="differs from sealed release"):
        _release_host_identity(release_root)


def test_translation_compatible_plan_preserves_tick_route() -> None:
    result = _translation_compatible_plan(_plan())

    assert result["execution"] == {
        "execution_route_identity": "tick_native_execution_route_v1",
        "internal_lifecycle_schedule": {"interval_ticks": 160},
        "generations": 1,
        "ticks_per_generation": 160,
        "checkpoints": [1],
    }
    assert isinstance(result["plan_sha256"], str)


def test_comparison_identity_requires_artifact_source_at_frozen_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen = "c" * 40
    monkeypatch.setattr(
        "tools.scripts.eiel_gpu_tick_anatomy_v1._git_head", lambda: frozen
    )
    hardware = {"identity_sha256": "d" * 64}
    identity = _comparison_identity(
        source_plan=_plan(),
        artifact_manifest={
            "committed_source_snapshot": {"build_source_commit": frozen}
        },
        release_host_identity=[],
        release_root=tmp_path,
        hardware_identity=hardware,
        cells=48,
        ticks=160,
    )
    assert len(identity["source_identity_sha256"]) == 64

    with pytest.raises(AllocationDiagnosticError, match="frozen repository HEAD"):
        _comparison_identity(
            source_plan=_plan(),
            artifact_manifest={
                "committed_source_snapshot": {"build_source_commit": "e" * 40}
            },
            release_host_identity=[],
            release_root=tmp_path,
            hardware_identity=hardware,
            cells=48,
            ticks=160,
        )


def test_diagnostic_artifact_rebind_requires_hash_bound_m0_manifest(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "eiel_cuda_architecture_b_wdl_jungle_eed_v1.so"
    artifact.write_bytes(b"diagnostic")
    import hashlib

    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / f"{artifact.name}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "module_name": "eiel_cuda_architecture_b_wdl_jungle_eed_v1",
                "artifact_sha256": artifact_sha256,
                "artifact_identity": {"sha256": artifact_sha256},
                "option_d_m0_telemetry_build": {"option_d_m0_telemetry_enabled": True},
                "option_d_m7_batched_cpu_policy_refill_build": (
                    _m7_build_identity(False)
                ),
                "option_d_m8_resident_policy_suffix_reuse_build": _m8_build_identity(
                    False
                ),
            }
        ),
        encoding="utf-8",
    )
    plan = _plan()
    plan["artifacts"]["cuda_extension"]["sha256"] = "0" * 64
    plan["artifacts"]["extension_manifest"]["sha256"] = "1" * 64

    rebound = _bind_artifacts(
        plan,
        artifact=artifact,
        manifest=manifest,
        allow_diagnostic_artifact_rebind=True,
        expected_m7_enabled=False,
    )

    assert rebound["artifacts"]["cuda_extension"]["sha256"] == artifact_sha256
    assert rebound["artifacts"]["extension_manifest"]["sha256"] != "1" * 64


def test_diagnostic_artifact_rebind_rejects_telemetry_off_manifest(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "eiel_cuda_architecture_b_wdl_jungle_eed_v1.so"
    artifact.write_bytes(b"diagnostic")
    import hashlib

    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / f"{artifact.name}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "module_name": "eiel_cuda_architecture_b_wdl_jungle_eed_v1",
                "artifact_sha256": artifact_sha256,
                "artifact_identity": {"sha256": artifact_sha256},
                "option_d_m0_telemetry_build": {"option_d_m0_telemetry_enabled": False},
                "option_d_m7_batched_cpu_policy_refill_build": (
                    _m7_build_identity(False)
                ),
                "option_d_m8_resident_policy_suffix_reuse_build": _m8_build_identity(
                    False
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AllocationDiagnosticError, match="not hash-bound"):
        _bind_artifacts(
            _plan(),
            artifact=artifact,
            manifest=manifest,
            allow_diagnostic_artifact_rebind=True,
            expected_m7_enabled=False,
        )

    rebound = _bind_artifacts(
        _plan(),
        artifact=artifact,
        manifest=manifest,
        allow_candidate_artifact_rebind=True,
        expected_m7_enabled=False,
    )
    assert rebound["artifacts"]["cuda_extension"]["sha256"] == artifact_sha256


def test_candidate_rebind_requires_candidate_m7_manifest_identity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "eiel_cuda_architecture_b_wdl_jungle_eed_v1.so"
    artifact.write_bytes(b"candidate")
    import hashlib

    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / f"{artifact.name}.manifest.json"
    payload = {
        "module_name": "eiel_cuda_architecture_b_wdl_jungle_eed_v1",
        "artifact_sha256": artifact_sha256,
        "artifact_identity": {"sha256": artifact_sha256},
        "option_d_m0_telemetry_build": {"option_d_m0_telemetry_enabled": False},
        "option_d_m7_batched_cpu_policy_refill_build": _m7_build_identity(True),
        "option_d_m8_resident_policy_suffix_reuse_build": _m8_build_identity(False),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    rebound = _bind_artifacts(
        _plan(),
        artifact=artifact,
        manifest=manifest,
        allow_candidate_artifact_rebind=True,
        expected_m7_enabled=True,
    )
    assert rebound["artifacts"]["cuda_extension"]["sha256"] == artifact_sha256

    with pytest.raises(AllocationDiagnosticError, match="M7 arm identity mismatch"):
        _bind_artifacts(
            _plan(),
            artifact=artifact,
            manifest=manifest,
            allow_candidate_artifact_rebind=True,
            expected_m7_enabled=False,
        )


def test_candidate_rebind_requires_exact_m8_manifest_and_m7_fallback(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "eiel_cuda_architecture_b_wdl_jungle_eed_v1.so"
    artifact.write_bytes(b"m8-candidate")
    import hashlib

    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / f"{artifact.name}.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "module_name": "eiel_cuda_architecture_b_wdl_jungle_eed_v1",
                "artifact_sha256": artifact_sha256,
                "artifact_identity": {"sha256": artifact_sha256},
                "option_d_m0_telemetry_build": {"option_d_m0_telemetry_enabled": False},
                "option_d_m7_batched_cpu_policy_refill_build": _m7_build_identity(True),
                "option_d_m8_resident_policy_suffix_reuse_build": _m8_build_identity(
                    True
                ),
            }
        ),
        encoding="utf-8",
    )

    rebound = _bind_artifacts(
        _plan(),
        artifact=artifact,
        manifest=manifest,
        allow_candidate_artifact_rebind=True,
        expected_m7_enabled=True,
        expected_m8_enabled=True,
    )
    assert rebound["artifacts"]["cuda_extension"]["sha256"] == artifact_sha256

    with pytest.raises(AllocationDiagnosticError, match="requires the M7 fallback"):
        _bind_artifacts(
            _plan(),
            artifact=artifact,
            manifest=manifest,
            allow_candidate_artifact_rebind=True,
            expected_m7_enabled=False,
            expected_m8_enabled=True,
        )


def test_result_summary_requires_candidate_m7_runtime_use() -> None:
    ticks = 160
    result = {
        "owner": "native_extension",
        "absolute_tick_span_native_owner": True,
        "native_tick_iterations": ticks,
        "tick_span_start": 0,
        "tick_span_end": ticks,
        "tick_span_count": ticks,
        "terminal_generation_boundary": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
        "turnover_shadow_plan_launches": 0,
        "turnover_cull_refill_plan_launches": 0,
        "turnover_commit_launches": 0,
        "turnover_publish_launches": 0,
        "option_d_m7_batched_cpu_policy_refill_enabled": True,
        "option_d_m7_batched_cpu_policy_refill_used": True,
        "option_d_m7_batched_cpu_policy_refill_authority": (
            "candidate_batched_exact_cpu_compact_refill"
        ),
        "option_d_m7_batched_cpu_policy_refill_schema_version": (
            "eiel_option_d_m7_batched_cpu_policy_refill_v1"
        ),
        "policy_birth_segment_refill_receipts": [
            {
                "option_d_m7_batched_cpu_policy_refill_enabled": True,
                "option_d_m7_batched_cpu_policy_refill_used": True,
                "option_d_m7_batched_cpu_policy_refill_authority": (
                    "candidate_batched_exact_cpu_compact_refill"
                ),
                "option_d_m7_batched_cpu_policy_refill_schema_version": (
                    "eiel_option_d_m7_batched_cpu_policy_refill_v1"
                ),
                **_m8_runtime_identity(enabled=False),
            }
        ],
    }

    assert (
        _result_summary(result, ticks=ticks, expected_m7_enabled=True)[
            "validated_policy_arm_identity"
        ]["m7_used_on_every_physical_refill"]
        is True
    )
    result["policy_birth_segment_refill_receipts"][0][
        "option_d_m7_batched_cpu_policy_refill_used"
    ] = False
    with pytest.raises(AllocationDiagnosticError, match="M7 arm identity mismatch"):
        _result_summary(result, ticks=ticks, expected_m7_enabled=True)


def test_result_summary_accepts_m8_reuse_and_requires_m7_on_fallback() -> None:
    ticks = 160
    base = {
        "owner": "native_extension",
        "absolute_tick_span_native_owner": True,
        "native_tick_iterations": ticks,
        "tick_span_start": 0,
        "tick_span_end": ticks,
        "tick_span_count": ticks,
        "terminal_generation_boundary": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
        "turnover_shadow_plan_launches": 0,
        "turnover_cull_refill_plan_launches": 0,
        "turnover_commit_launches": 0,
        "turnover_publish_launches": 0,
    }
    reuse = {
        "option_d_m7_batched_cpu_policy_refill_enabled": True,
        "option_d_m7_batched_cpu_policy_refill_used": False,
        "option_d_m7_batched_cpu_policy_refill_authority": (
            "compiled_candidate_not_used_for_surface"
        ),
        "option_d_m7_batched_cpu_policy_refill_schema_version": None,
        **_m8_runtime_identity(enabled=True, reused=True),
    }
    fallback = {
        "option_d_m7_batched_cpu_policy_refill_enabled": True,
        "option_d_m7_batched_cpu_policy_refill_used": True,
        "option_d_m7_batched_cpu_policy_refill_authority": (
            "candidate_batched_exact_cpu_compact_refill"
        ),
        "option_d_m7_batched_cpu_policy_refill_schema_version": (
            "eiel_option_d_m7_batched_cpu_policy_refill_v1"
        ),
        **_m8_runtime_identity(enabled=True, reused=False),
    }
    result = {**base, "policy_birth_segment_refill_receipts": [reuse, fallback]}

    summary = _result_summary(
        result,
        ticks=ticks,
        expected_m7_enabled=True,
        expected_m8_enabled=True,
    )["validated_policy_arm_identity"]
    assert summary["resident_suffix_reuse_receipt_count"] == 1
    assert summary["physical_refill_receipt_count"] == 1
    assert summary["m7_used_on_every_physical_refill"] is True
    assert summary["m7_unused_on_every_suffix_reuse"] is True

    fallback["option_d_m7_batched_cpu_policy_refill_used"] = False
    with pytest.raises(AllocationDiagnosticError, match="M7 arm identity mismatch"):
        _result_summary(
            result,
            ticks=ticks,
            expected_m7_enabled=True,
            expected_m8_enabled=True,
        )


def test_competing_gpu_processes_ignores_desktop_and_own_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.scripts.eiel_gpu_tick_anatomy_v1.os.getpid", lambda: 7)
    rows = [
        {"pid": 7, "process_name": "python", "used_mib": 10},
        {"pid": 8, "process_name": "Xorg", "used_mib": 100},
        {"pid": 9, "process_name": "python", "used_mib": 3000},
        {"pid": 10, "process_name": "nsys", "used_mib": 0},
    ]

    assert _competing_gpu_processes(rows) == [rows[2], rows[3]]


def test_analyzer_groups_three_kernel_tick_abi_and_computes_percentile() -> None:
    kernels = []
    for tick in range(3):
        for offset, name in enumerate((CORE, PREPARE, COMMIT)):
            start = tick * 100 + offset * 20
            kernels.append(
                {"name": f"void {name}(args)", "start": start, "end": start + 10}
            )

    groups = _tick_groups(kernels)

    assert len(groups) == 3
    assert set(groups[1]["stages"]) == {CORE, PREPARE, COMMIT}
    assert _stats([1.0, 2.0, 3.0, 4.0, 5.0]) == {
        "count": 5,
        "mean": 3.0,
        "median": 3.0,
        "p95": 4.8,
        "min": 1.0,
        "max": 5.0,
    }
