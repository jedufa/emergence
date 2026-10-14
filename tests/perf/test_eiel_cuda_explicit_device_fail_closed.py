from __future__ import annotations

from pathlib import Path

import pytest
import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_base import _resolve_torch_device
from simulation.runner_pkg.runner_config import SimConfig


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (False, torch.device("cpu")),
        (True, torch.device("cuda")),
    ],
)
def test_auto_device_preserves_availability_fallback(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    expected: torch.device,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)

    assert _resolve_torch_device("auto") == expected


def test_runner_retains_requested_and_effective_auto_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sim_cfg = SimConfig(
        seed=17,
        device="AUTO",
        bank_device="auto",
        behavior_log_enable=False,
    )

    runner = SimulationRunner(
        env_cfg=EnvConfig(width=6, height=6, view_radius=1),
        pop_cfg=PopConfig(population_size=2),
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(tmp_path)),
        run_dir=str(tmp_path),
    )

    assert runner.requested_device == "AUTO"
    assert runner.requested_bank_device == "auto"
    assert runner.device == torch.device("cpu")
    assert runner.bank_device == runner.device


def test_runner_explicit_bank_cuda_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sim_cfg = SimConfig(
        seed=17,
        device="cpu",
        bank_device="cuda:0",
        behavior_log_enable=False,
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        SimulationRunner(
            env_cfg=EnvConfig(width=6, height=6, view_radius=1),
            pop_cfg=PopConfig(population_size=2),
            sim_cfg=sim_cfg,
            logger=MetricsLogger(str(tmp_path)),
            run_dir=str(tmp_path),
        )


@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "CUDA:1", "gpu"])
def test_explicit_cuda_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_torch_device(requested)


@pytest.mark.parametrize(
    "requested",
    [
        "cuda:",
        "cuda:-1",
        "cuda:01",
        "cuda:x",
        "cuda:1:2",
        "cudafoo",
        "cuda-ish",
    ],
)
def test_malformed_explicit_cuda_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="Malformed explicit CUDA device request"):
        _resolve_torch_device(requested)


def test_explicit_cuda_index_must_be_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert _resolve_torch_device("cuda") == torch.device("cuda")
    assert _resolve_torch_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match=r"out of range: 2 CUDA device\(s\) are visible"):
        _resolve_torch_device("cuda:2")


def test_explicit_cuda_rejects_inconsistent_zero_device_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="no CUDA devices are visible"):
        _resolve_torch_device("cuda")


@pytest.mark.parametrize("requested", ["tpu", "gpu:0", "none"])
def test_unknown_device_requests_fail_closed(requested: str) -> None:
    with pytest.raises(ValueError, match="Unsupported device request"):
        _resolve_torch_device(requested)
