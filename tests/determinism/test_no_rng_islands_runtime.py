from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

def _dummy_env_cfg():
    # Minimal object with attributes accessed early in StateMixin.__init__
    return SimpleNamespace(
        height=8,
        width=8,
        view_radius=1,
        population_size=16,
        max_agent_capacity_mult=4,
        tier4_juvenile_vulnerability_enable=False,
    )

def _dummy_pop_cfg():
    # Only population_size accessed before RNG checks should already have raised.
    return SimpleNamespace(
        population_size=8,
        slot_factor=2,
        max_slots_cap=0,
    )

def test_state_mixin_requires_rng_injection():
    from env.gridworld_pkg.state import StateMixin
    cfg = _dummy_env_cfg()
    with pytest.raises(RuntimeError, match="requires rng injection"):
        StateMixin(cfg=cfg, seed=0, run_dir=None, rng=None)

def test_population_requires_py_rng_injection():
    from evolution.population_pkg.core import Population
    env_cfg = _dummy_env_cfg()
    pop_cfg = _dummy_pop_cfg()
    with pytest.raises(RuntimeError, match="requires rng injection"):
        Population(env_cfg=env_cfg, pop_cfg=pop_cfg, seed=0, rng=None, np_rng=np.random.default_rng(0))

def test_population_requires_np_rng_injection():
    from evolution.population_pkg.core import Population
    env_cfg = _dummy_env_cfg()
    pop_cfg = _dummy_pop_cfg()
    import random
    with pytest.raises(RuntimeError, match="requires np_rng injection|requires np_rng"):
        Population(env_cfg=env_cfg, pop_cfg=pop_cfg, seed=0, rng=random.Random(0), np_rng=None)

def test_shock_mixin_requires_rng_manager():
    from simulation.runner_pkg.runner_shock import ShockMixin

    class Dummy(ShockMixin):
        def __init__(self):
            self.sim_cfg = SimpleNamespace(shock_pulsed=False, shock_start_prob=None, shock_period=0, shock_duration=0)

    d = Dummy()
    with pytest.raises(RuntimeError, match="requires rng_manager"):
        d._shock_active(gen=0)
