from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents import spawn as agents_spawn
from evolution.population_pkg import spawn as population_spawn
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


def test_p0d_strict_violation_np_random_draw_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import numpy as np\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  np.random.random()\n"
        "except Exception as e:\n"
        "  raise SystemExit(0 if 'DET_FORBIDDEN_RNG_API' in str(e) else 2)\n"
        "raise SystemExit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0d_strict_violation_python_random_draw_subprocess() -> None:
    env = _strict_subprocess_env()
    code = (
        "from simulation.determinism_enforcer import install_runtime_rng_guards\n"
        "import random\n"
        "install_runtime_rng_guards()\n"
        "try:\n"
        "  random.random()\n"
        "except Exception as e:\n"
        "  raise SystemExit(0 if 'DET_FORBIDDEN_RNG_API' in str(e) else 2)\n"
        "raise SystemExit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"


def test_p0d_numpy_migrated_spawn_index_delta() -> None:
    rm = RNGManager(global_seed=101)
    host = SimpleNamespace(np_rng=rm.mutation_np)
    agent = SimpleNamespace(energy=5.0, max_energy_norm=10.0)

    before = _idx(rm, "mutation.np")
    agents_spawn._init_drives_for_agent(host, agent)
    after = _idx(rm, "mutation.np")

    assert after - before == 4
    assert isinstance(agent.drives, np.ndarray)
    assert agent.drives.shape == (4,)


def test_p0d_python_index_delta_and_budget_scope_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=202)

    with rm.budget_scope("p0d_py_ok", expected_delta={"mutation.py": 5}):
        _ = rm.mutation_py.choices([1, 2, 3], k=5)

    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        with rm.budget_scope("p0d_py_bad", expected_delta={"mutation.py": 4}):
            _ = rm.mutation_py.choices([1, 2, 3], k=5)


def test_p0d_numpy_snapshot_restore_future_draw_identity() -> None:
    rm = RNGManager(global_seed=303)
    _ = rm.policy_np.random()
    snap = rm.capture_snapshot(generation_index=3)
    b = rm.policy_np.normal(0.0, 1.0, size=(2, 4))
    rm.apply_snapshot_dict(snap)
    c = rm.policy_np.normal(0.0, 1.0, size=(2, 4))
    assert np.array_equal(b, c)


def test_p0d_python_snapshot_restore_future_draw_identity() -> None:
    rm = RNGManager(global_seed=404)
    _ = rm.policy_py.random()
    snap = rm.capture_snapshot(generation_index=4)
    b = [rm.policy_py.gauss(0.0, 1.0) for _ in range(5)]
    rm.apply_snapshot_dict(snap)
    c = [rm.policy_py.gauss(0.0, 1.0) for _ in range(5)]
    assert b == c


def test_p0d_no_silent_fallback_in_strict_agents_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    host = SimpleNamespace(np_rng=None)
    agent = SimpleNamespace(energy=5.0, max_energy_norm=10.0)
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_RNG_API"):
        agents_spawn._init_drives_for_agent(host, agent)


def test_p0d_no_silent_fallback_in_strict_population_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    host = SimpleNamespace(np_rng=None)
    stats = SimpleNamespace(
        exp_dE=np.zeros((7,), dtype=np.float32),
        exp_dInj=np.zeros((7,), dtype=np.float32),
        pred_queue=[],
        prev_drive_delta=np.zeros((4,), dtype=np.float32),
    )
    genome = SimpleNamespace()
    agent = SimpleNamespace(
        energy=5.0,
        max_energy_norm=10.0,
        genome=genome,
        stats=stats,
    )
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_RNG_API"):
        population_spawn._init_drives_for_agent(host, agent, None, None)
