from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from simulation.rng_manager import RNGManager


def _idx(rm: RNGManager, key: str) -> int:
    return int(rm.indices_snapshot().get(key, 0))


def _strict_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["EIEL_STRICT_DETERMINISM"] = "1"
    env["PYTHONHASHSEED"] = "0"
    repo_root = Path(__file__).resolve().parents[3]
    evo_root = str((repo_root / "engine").resolve())
    old = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = evo_root if not old else (evo_root + os.pathsep + old)
    return env


def test_p0c_torch_replay_snapshot_restore_cpu() -> None:
    rm = RNGManager(global_seed=12345)
    tr = rm.torch_rng("policy", "cpu")
    assert tr is not None

    _ = tr.rand((2, 3), device="cpu", dtype=torch.float32)
    snap = rm.capture_snapshot(generation_index=7)
    b = tr.randn((4, 2), device="cpu", dtype=torch.float32)
    rm.apply_snapshot_dict(snap)
    c = tr.randn((4, 2), device="cpu", dtype=torch.float32)

    assert torch.equal(b, c)
    assert _idx(rm, "policy.torch_cpu") == 14


def test_p0c_torch_index_delta_matches_numel() -> None:
    rm = RNGManager(global_seed=77)
    tr = rm.torch_rng("mutation", "cpu")
    assert tr is not None

    k = "mutation.torch_cpu"
    before = _idx(rm, k)
    x = tr.rand((3, 5), device="cpu", dtype=torch.float32)
    after = _idx(rm, k)
    assert after - before == int(x.numel())


def test_p0c_torch_budget_scope_expected_and_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=9)
    tr = rm.torch_rng("policy", "cpu")
    assert tr is not None

    with rm.budget_scope("p0c_ok", expected_delta={"policy.torch_cpu": 6}):
        _ = tr.rand((2, 3), device="cpu", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        with rm.budget_scope("p0c_bad", expected_delta={"policy.torch_cpu": 5}):
            _ = tr.rand((2, 3), device="cpu", dtype=torch.float32)


def test_p0c_strict_violation_torch_rand_without_rngmanager_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import torch\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  torch.rand((2,3))\n"
        "except Exception as e:\n"
        "  raise SystemExit(0 if 'DET_FORBIDDEN_RNG_API' in str(e) else 2)\n"
        "raise SystemExit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0c_strict_violation_manual_seed_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import torch\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  torch.manual_seed(1)\n"
        "except Exception as e:\n"
        "  ok = ('DET_FORBIDDEN_SEED_CALL' in str(e)) or ('DET_FORBIDDEN_RNG_API' in str(e))\n"
        "  raise SystemExit(0 if ok else 2)\n"
        "raise SystemExit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0c_strict_violation_torch_generator_constructor_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import torch\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  torch.Generator(device='cpu')\n"
        "except Exception as e:\n"
        "  raise SystemExit(0 if 'DET_FORBIDDEN_RNG_API' in str(e) else 2)\n"
        "raise SystemExit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0c_strict_violation_generator_backed_uniform_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import torch\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  g = torch.Generator(device='cpu')\n"
        "  _ = torch.empty((2, 3)).uniform_(generator=g)\n"
        "except Exception as e:\n"
        "  raise SystemExit(0 if 'DET_FORBIDDEN_RNG_API' in str(e) else 2)\n"
        "raise SystemExit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0c_strict_bootstrap_guards_with_rngmanager_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "from simulation.rng_manager import RNGManager\n"
        "import torch\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  rm = RNGManager(global_seed=123)\n"
        "  tr = rm.torch_rng('policy', 'cpu')\n"
        "  assert tr is not None\n"
        "  x = tr.rand((2,3), device='cpu', dtype=torch.float32)\n"
        "  assert int(x.numel()) == 6\n"
        "except Exception as e:\n"
        "  raise SystemExit(2)\n"
        "raise SystemExit(0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0c_no_public_original_callable_bypass_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import torch\n"
        "install_runtime_rng_guards()\n"
        "raise SystemExit(0 if getattr(torch.rand, '__EIEL_DET_ORIGINAL__', None) is None else 2)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0c_strict_violation_direct_generator_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=11)
    tr = rm.torch_rng("policy", "cpu")
    assert tr is not None
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_RNG_API"):
        _ = tr.generator


def test_p0c_strict_violation_internal_generator_attr_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=12)
    tr = rm.torch_rng("policy", "cpu")
    assert tr is not None
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_RNG_API"):
        _ = tr._gen
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_RNG_API"):
        _ = tr._TrackedTorchRNG__gen


def test_p0c_strict_violation_unsupported_torch_rng_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=21)
    tr = rm.torch_rng("policy", "cpu")
    assert tr is not None
    with pytest.raises(RuntimeError, match="DET_TORCH_GENERATOR_UNSUPPORTED_OP"):
        getattr(tr, "multinomial")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable in this environment")
def test_p0c_torch_replay_snapshot_restore_cuda() -> None:
    rm = RNGManager(global_seed=12345)
    tr = rm.torch_rng("policy", "cuda:0")
    assert tr is not None

    _ = tr.rand((2, 2), device="cuda:0", dtype=torch.float32)
    snap = rm.capture_snapshot(generation_index=3)
    b = tr.rand((3, 3), device="cuda:0", dtype=torch.float32)
    rm.apply_snapshot_dict(snap)
    c = tr.rand((3, 3), device="cuda:0", dtype=torch.float32)
    assert torch.equal(b, c)
