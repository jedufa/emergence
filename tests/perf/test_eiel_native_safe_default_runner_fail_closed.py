from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from env.env_config import EnvConfig
import native_runtime.runner_integration as integration
import native_runtime.safe_default as safe_default
from native_runtime.engine import _runtime_capability
from native_runtime.runner_integration import (
    execute_native_generation_from_runner,
    validate_native_generation_result,
    write_native_behavior_public_outputs,
)
from native_runtime.types import (
    NativeEngineCounters,
    NativeGenerationResult,
    NativeRuntimeCapability,
)
from simulation.runner_pkg import runner_loop
from simulation.runner_pkg.runner_config import SimConfig


def _cfg() -> SimConfig:
    cfg = SimConfig(
        seed=31,
        generations=1,
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
    )
    cfg.native_runtime_cxx26_scope_lock_enable = True
    safe_default.apply_native_runtime_scope_lock_defaults(cfg)
    cfg.native_runtime_fixed_policy_family = "mlp"
    cfg.native_runtime_diagnostic_enable = True
    cfg.force_prediction_disabled = True
    return cfg


def _env() -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.patch_world_enable = False
    cfg.patch_world_observation_enable = False
    cfg.reproduce_requires_mate = False
    return cfg


def _set_prebuilt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    extension = tmp_path / "eiel_native_runtime_ext.so"
    extension.write_bytes(b"runner fail-closed test extension")
    manifest = {
        "schema_version": "eiel_native_runtime_build_manifest_v1",
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "output_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    }
    extension.with_suffix(extension.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EIEL_NATIVE_RUNTIME_EXTENSION", str(extension))
    monkeypatch.setattr(safe_default, "load_prebuilt_extension", lambda path: SimpleNamespace())
    monkeypatch.setattr(safe_default, "NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED", False)
    monkeypatch.setattr(safe_default, "FLAT_NATIVE_PROMOTION_CERTIFICATE_PATH", "")
    monkeypatch.setattr(safe_default, "FLAT_NATIVE_PROMOTION_CERTIFICATE_SHA256", "")
    return extension


def _capability() -> NativeRuntimeCapability:
    return _runtime_capability(
        world="flat",
        prebuilt_available=True,
        extension_abi_version="eiel_native_runtime_abi_v1",
        extension_version="eiel_native_runtime_flat_v1",
        policy_family="mlp",
        device="cpu",
        lifecycle_branch="flat_generation_birth_commit",
        checkpoint_mode="off",
        reporting_supported=True,
        reporting_qualification="route_neutral_exact",
        benchmark_qualified=False,
        accepted_default_native_envelope=False,
    )


def _raw() -> dict[str, object]:
    return {
        "executed": True,
        "hidden_fallback": False,
        "native_owner": True,
        "single_native_generation_call": True,
        "native_tick_loop_count": 4,
        "alive_count": 2,
        "capacity": 4,
        "free_slots": 2,
        "typed_event_rows": 0,
        "behavior_rows": [],
    }


def _result(raw: dict[str, object] | None = None, *, executed: bool = True) -> NativeGenerationResult:
    payload = _raw() if raw is None else raw
    return NativeGenerationResult(
        schema_version="eiel_native_generation_result_v1",
        decision="EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION",
        executed=executed,
        generation_index=0,
        steps_per_generation=4,
        world="flat",
        native_tick_loop_count=4,
        committed_birth_rows=0,
        counters=NativeEngineCounters(),
        capability=_capability(),
        compact_result={},
        raw_result=payload,
    )


def _runner(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sim_cfg=_cfg(),
        env_cfg=_env(),
        run_dir=str(tmp_path),
        _policy_use_gru=[False],
    )


def _engine(*, tick: int = 0, phase: str = "generation_boundary") -> SimpleNamespace:
    return SimpleNamespace(
        export_checkpoint_projection=lambda: {
            "metadata": {
                "generation": 1,
                "tick": tick,
                "agent_grid_runner_phase": phase,
            }
        }
    )


@pytest.mark.parametrize(
    ("executed", "hidden", "message"),
    [
        (False, False, "executed=false"),
        (True, True, "hidden_fallback=true"),
    ],
)
def test_execute_native_generation_rejects_false_execution_and_hidden_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executed: bool,
    hidden: bool,
    message: str,
) -> None:
    runner = _runner(tmp_path)
    raw = _raw()
    raw["executed"] = executed
    raw["hidden_fallback"] = hidden
    result = _result(raw, executed=executed)
    engine = SimpleNamespace(execute_generation=lambda **kwargs: result)
    monkeypatch.setattr(integration, "validate_native_runtime_supported", lambda runner: None)
    monkeypatch.setattr(
        integration,
        "get_or_create_native_engine",
        lambda runner, generation_index, **kwargs: engine,
    )

    with pytest.raises(RuntimeError, match=message):
        execute_native_generation_from_runner(
            runner,
            generation_index=0,
            steps_per_generation=4,
        )

    failure = json.loads((tmp_path / "native_runtime_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["ok"] is False
    assert failure["promotion_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ok", False, "ok=false"),
        ("status", "failed", "status="),
        ("generation_complete", False, "generation_complete=false"),
        ("finalization_complete", False, "finalization_complete=false"),
        ("required_outputs_present", False, "required_outputs_present=false"),
        ("output_flush_complete", False, "output_flush_complete=false"),
        ("extension_hash_verified", False, "extension_hash_verified=false"),
        ("native_owner", False, "owner identity"),
        ("single_native_generation_call", False, "owner identity"),
        ("mixed_ownership", True, "mixed Python/native ownership"),
        ("runtime_jit", True, "runtime JIT detected"),
    ],
)
def test_native_result_rejects_every_failed_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    raw = _raw()
    raw[field] = value

    with pytest.raises(RuntimeError, match=message):
        validate_native_generation_result(
            _runner(tmp_path),
            _result(raw),
            _engine(),
            generation_index=0,
            steps_per_generation=4,
        )


def test_native_result_rejects_missing_output_partial_checkpoint_and_bad_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    runner = _runner(tmp_path)
    missing = _raw()
    del missing["behavior_rows"]
    with pytest.raises(RuntimeError, match="required native outputs missing"):
        validate_native_generation_result(
            runner,
            _result(missing),
            _engine(),
            generation_index=0,
            steps_per_generation=4,
        )

    with pytest.raises(RuntimeError, match="partial or unfinalized checkpoint"):
        validate_native_generation_result(
            runner,
            _result(),
            _engine(tick=2, phase="partial_generation"),
            generation_index=0,
            steps_per_generation=4,
        )

    malformed = _result()
    malformed.capability = replace(_capability(), supports_flat_generation=False)
    with pytest.raises(RuntimeError, match="malformed or contradictory native capability"):
        validate_native_generation_result(
            runner,
            malformed,
            _engine(),
            generation_index=0,
            steps_per_generation=4,
        )


def test_native_result_positive_validation_is_truthful_and_prepromotion_unbenchmarked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_prebuilt(monkeypatch, tmp_path)
    result = _result()
    validation = validate_native_generation_result(
        _runner(tmp_path),
        result,
        _engine(),
        generation_index=0,
        steps_per_generation=4,
    )

    assert validation["ok"] is True
    assert validation["status"] == "passed"
    assert validation["finalization_complete"] is True
    assert validation["extension_hash_verified"] is True
    assert result.capability.benchmark_qualification == "not_qualified"


def test_behavior_output_failure_rolls_back_and_writes_durable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    path = tmp_path / "behaviors.csv"
    path.write_bytes(b"existing\r\n")

    class BrokenWriter:
        def __init__(self, handle) -> None:
            self.handle = handle

        def writerow(self, row) -> None:
            self.handle.write("partial")
            raise OSError("disk full")

    monkeypatch.setattr(integration.csv, "writer", lambda handle, **kwargs: BrokenWriter(handle))
    with pytest.raises(OSError, match="disk full"):
        write_native_behavior_public_outputs(runner, {"behavior_rows": [[0] * 21]})

    assert path.read_bytes() == b"existing\r\n"
    failure = json.loads((tmp_path / "native_runtime_failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "behavior_public_output"
    assert failure["first_cause"] == "OSError('disk full')"


def test_native_checkpoint_failure_propagates_but_reference_failure_can_warn(
    tmp_path: Path,
) -> None:
    native = _runner(tmp_path)
    native._native_safe_default_decision = {"native_execution_active": True}
    with pytest.raises(RuntimeError, match="native-active checkpoint write failed"):
        runner_loop._raise_native_checkpoint_failure(
            native,
            PermissionError("denied"),
            stage="checkpoint_periodic_generation_boundary",
        )
    assert (tmp_path / "native_runtime_failure.json").is_file()

    reference = _runner(tmp_path / "reference")
    reference._native_safe_default_decision = {"native_execution_active": False}
    assert runner_loop._raise_native_checkpoint_failure(
        reference,
        PermissionError("denied"),
        stage="checkpoint_periodic_generation_boundary",
    ) is None


def test_source_freeze_removes_owned_eiel_trace_branches() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for relative in (
        "engine/native_runtime/runner_integration.py",
        "engine/simulation/runner_pkg/runner_ticks_loop.py",
        "engine/simulation/runnercore/tick_loop.py",
    ):
        assert "EIEL_TRACE_" not in (project_root / relative).read_text(encoding="utf-8")
