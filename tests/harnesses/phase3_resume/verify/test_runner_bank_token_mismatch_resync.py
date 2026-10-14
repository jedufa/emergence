"""tests/harnesses/phase3_resume/verify/test_runner_bank_token_mismatch_resync.py

Regression test for runner bank resync behavior.

Failure mode (historical):
- After resume, bank was not marked dirty, slot_token mismatched bank token,
  but a fast-path skipped the resync.

This test asserts two things:
1) A dirty bank forces at least one full sync, even if tokens "look" aligned.
2) A slot_token mismatch forces resync even when bank is not marked dirty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _DummyPopCfg:
    population_size: int


class _DummyPop:
    def __init__(self, *, max_slots: int) -> None:
        self.max_slots = int(max_slots)


class _DummyPolicy:
    def __init__(self, *, obs_size: int, hidden: int, out: int) -> None:
        # BankMixin expects `policy.net` with Linear layers at indices 0,2,4.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out, bias=True),
        )


def _set_policy_constants(pol: _DummyPolicy, v: float) -> None:
    with torch.no_grad():
        for p in pol.net.parameters():
            p.fill_(float(v))


@dataclass
class _DummyAgent:
    agent_id: int
    slot: int
    slot_token: int
    policy: _DummyPolicy

    # BankMixin uses `_bank_token` as the sync gate.
    _bank_token: int = -1


def test_runner_bank_token_mismatch_forces_resync_when_not_dirty() -> None:
    from simulation.runner_pkg.runner_bank import BankMixin

    class _DummyRunner(BankMixin):
        def __init__(self) -> None:
            # Minimal runner fields required by BankMixin.
            self.pop_cfg = _DummyPopCfg(population_size=4)
            self.pop = _DummyPop(max_slots=4)
            self._bank_device = torch.device("cpu")
            self._bank_dtype = torch.float32
            self._hidden = 4
            self._obs_size = 3
            self._out = 5

            # Banks are lazy.
            self._W1 = None
            self._b1 = None
            self._W2 = None
            self._b2 = None
            self._W3 = None
            self._b3 = None
            self._mlp_pack = None

            # Default: dirty until first sync.
            self._bank_dirty_mlp = True
            self._bank_dirty_gru = True

    r = _DummyRunner()

    # Two stable agents.
    p0 = _DummyPolicy(obs_size=r._obs_size, hidden=r._hidden, out=r._out)
    p1 = _DummyPolicy(obs_size=r._obs_size, hidden=r._hidden, out=r._out)
    _set_policy_constants(p0, 0.10)
    _set_policy_constants(p1, 0.20)

    # Simulate a "resume"-like state where tokens appear aligned already,
    # but the bank is still marked dirty (content unknown / uninitialized).
    a0 = _DummyAgent(agent_id=1, slot=0, slot_token=10, policy=p0, _bank_token=10)
    a1 = _DummyAgent(agent_id=2, slot=1, slot_token=11, policy=p1, _bank_token=11)

    # Initial sync must clear dirty even though `_bank_token == slot_token`.
    r._sync_bank_for_agents([a0, a1])
    assert bool(getattr(r, "_bank_dirty_mlp", True)) is False

    # Now make the bank "clean" but introduce a token mismatch on one agent.
    # This must force a resync regardless of dirty flag.
    old_row = r._mlp_pack[a0.slot].detach().clone()
    # Simulate the "new weights in the same slot" situation:
    # token changes AND weights differ, so resync must overwrite the row.
    _set_policy_constants(p0, 0.30)
    a0.slot_token = 99
    r._sync_bank_for_agents([a0, a1])

    assert int(getattr(a0, "_bank_token", -1)) == int(a0.slot_token)
    assert torch.any(r._mlp_pack[a0.slot] != old_row)
