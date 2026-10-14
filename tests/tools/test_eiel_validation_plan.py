from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine.validation_plans import (
    ValidationPlanError,
    execute_plan,
    load_plan,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/scripts/eiel_validation_plan.py"


def _plan(argv: list[str] | None = None) -> dict[str, object]:
    return {
        "schema_version": "eiel_validation_plan_v1",
        "plan_id": "unit-cpu-v1",
        "cpu_only": True,
        "defaults": {"inherit_env": [], "env": {"PYTHONHASHSEED": "0"}},
        "steps": [
            {
                "id": "probe",
                "argv": argv
                or [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
            }
        ],
    }


def test_json_and_yaml_load_to_same_canonical_plan(tmp_path: Path) -> None:
    raw = _plan()
    json_path = tmp_path / "plan.json"
    yaml_path = tmp_path / "plan.yaml"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    import yaml

    yaml_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert load_plan(json_path) == load_plan(yaml_path) == validate_plan(raw)


@pytest.mark.parametrize(
    ("suffix", "payload"),
    (
        (
            ".json",
            '{"schema_version":"eiel_validation_plan_v1",'
            '"schema_version":"duplicate","plan_id":"x","cpu_only":true,"steps":[]}',
        ),
        (
            ".yaml",
            "schema_version: eiel_validation_plan_v1\n"
            "schema_version: duplicate\nplan_id: x\ncpu_only: true\nsteps: []\n",
        ),
    ),
)
def test_load_rejects_duplicate_fields(
    tmp_path: Path, suffix: str, payload: str
) -> None:
    path = tmp_path / f"duplicate{suffix}"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValidationPlanError, match="duplicate"):
        load_plan(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan.update(extra=True), "unsupported fields"),
        (lambda plan: plan.update(schema_version="v2"), "schema_version"),
        (lambda plan: plan.update(cpu_only=False), "cpu_only"),
        (lambda plan: plan["steps"][0].update(command="echo ok"), "unsupported fields"),
        (lambda plan: plan["steps"].append(dict(plan["steps"][0])), "duplicate step"),
        (lambda plan: plan["steps"][0].update(argv=["python3", "-V"]), "executable path"),
    ],
)
def test_validation_fails_closed_on_unknown_or_ambiguous_shapes(mutate, message: str) -> None:
    plan = _plan()
    mutate(plan)
    with pytest.raises(ValidationPlanError, match=message):
        validate_plan(plan)


@pytest.mark.parametrize(
    "argv",
    [
        ["nvidia-smi"],
        [sys.executable, "tool.py", "--device=cuda:0"],
        [sys.executable, "tool.py", "--accelerator", "gpu"],
        [sys.executable, "-c", "import torch; torch.cuda.init()"],
        [sys.executable, "tool.py", "--use-gpu"],
        ["/bin/sh", "-c", "nvidia-smi"],
    ],
)
def test_validation_rejects_gpu_command_requests(argv: list[str]) -> None:
    with pytest.raises(ValidationPlanError, match="GPU|shell"):
        validate_plan(_plan(argv))


@pytest.mark.parametrize("location", ["env", "inherit"])
def test_validation_rejects_gpu_environment_requests(location: str) -> None:
    plan = _plan()
    if location == "env":
        plan["defaults"]["env"]["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        plan["defaults"]["inherit_env"] = ["NVIDIA_VISIBLE_DEVICES"]
    with pytest.raises(ValidationPlanError, match="GPU|unsafe"):
        validate_plan(plan)


def test_validation_rejects_gpu_environment_value_with_neutral_key() -> None:
    plan = _plan()
    plan["defaults"]["env"]["EXECUTION_TARGET"] = "cuda:0"
    with pytest.raises(ValidationPlanError, match="GPU value"):
        validate_plan(plan)


def test_executor_uses_explicit_environment_masks_gpu_and_stops_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    marker = tmp_path / "should-not-run"
    plan = _plan(
        [
            sys.executable,
            "-c",
            "import os,sys; print(os.environ['NVIDIA_VISIBLE_DEVICES']); sys.exit(3)",
        ]
    )
    plan["steps"].append(
        {"id": "later", "argv": [sys.executable, "-c", f"open({str(marker)!r},'w').close()"]}
    )

    receipt = execute_plan(plan, root=tmp_path)

    assert receipt["status"] == "failed"
    assert receipt["steps"][0]["returncode"] == 3
    assert receipt["steps"][0]["stdout"] == "void\n"
    assert "CUDA_VISIBLE_DEVICES" in receipt["steps"][0]["environment_keys"]
    assert "CUDA_VISIBLE_DEVICES" in receipt["steps"][0]["cpu_masked_variables"]
    assert "PATH" not in receipt["steps"][0]["environment_keys"]
    assert len(receipt["steps"]) == 1
    assert not marker.exists()
    assert len(receipt["plan_sha256"]) == len(receipt["receipt_sha256"]) == 64


def test_receipt_never_persists_inherited_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "do-not-store-this-value"
    monkeypatch.setenv("VALIDATION_TOKEN", secret)
    plan = _plan([sys.executable, "-c", "print('ok')"])
    plan["defaults"]["inherit_env"] = ["VALIDATION_TOKEN"]

    receipt = execute_plan(plan, root=tmp_path)
    rendered = json.dumps(receipt, sort_keys=True)

    assert "VALIDATION_TOKEN" in receipt["steps"][0]["environment_keys"]
    assert secret not in rendered


def test_executor_rejects_cwd_escape_and_never_uses_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan([sys.executable, "-c", "print('ok')"])
    plan["steps"][0]["cwd"] = "linked"
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValidationPlanError, match="escapes"):
        execute_plan(plan, root=tmp_path)

    plan["steps"][0].pop("cwd")
    real_run = subprocess.run

    def checked_run(*args, **kwargs):
        assert kwargs["shell"] is False
        return real_run(*args, **kwargs)

    monkeypatch.setattr("engine.validation_plans.v1.subprocess.run", checked_run)
    assert execute_plan(plan, root=tmp_path)["status"] == "passed"


def test_executor_records_launch_failure_in_receipt(tmp_path: Path) -> None:
    plan = _plan([str(tmp_path / "missing-executable")])

    receipt = execute_plan(plan, root=tmp_path)

    assert receipt["status"] == "failed"
    assert receipt["steps"][0]["status"] == "launch_failed"
    assert receipt["steps"][0]["returncode"] is None


def test_executor_normalizes_timeout_bytes_for_json_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan([sys.executable, "-c", "print('unused')"])

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], kwargs["timeout"], output=b"partial\xff", stderr=b"late\xfe"
        )

    monkeypatch.setattr("engine.validation_plans.v1.subprocess.run", timed_out)
    receipt = execute_plan(plan, root=tmp_path)

    assert receipt["steps"][0]["status"] == "timed_out"
    assert receipt["steps"][0]["stdout"] == "partial\ufffd"
    assert receipt["steps"][0]["stderr"] == "late\ufffd"
    json.dumps(receipt)


def test_cli_validates_and_executes_with_json_receipt(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    plan_path.write_text(json.dumps(_plan([sys.executable, "-c", "print('ok')"])), encoding="utf-8")

    validated = subprocess.run(
        [sys.executable, str(SCRIPT), "validate", str(plan_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    executed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "execute",
            str(plan_path),
            "--root",
            str(tmp_path),
            "--receipt",
            str(receipt_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["ok"] is True
    assert executed.returncode == 0, executed.stderr
    assert json.loads(executed.stdout)["status"] == "passed"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "eiel_validation_plan_receipt_v1"
    assert receipt["steps"][0]["stdout"] == "ok\n"
