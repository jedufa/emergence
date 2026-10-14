"""tests/control_plane/test_control_plane_timeline_freeze.py

Phase 2 exit-criteria tests:
1) deterministic command ordering
2) WAL written before apply
3) recovery restores pending commands
4) idempotent ack
5) plan freeze when command enters past
6) expectation_note required on plan version increment
7) sequence number persisted in checkpoint metadata + restore
8) ordering same after resume
"""

import os
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from simulation.control_plane_pkg import ControlPlane
from simulation.control_plane_pkg.commands import PressureUpdateCommand, CheckpointRequestCommand, PlanAmendCommand, PlanCancelCommand
from simulation.control_plane_pkg.wal import wal_path, WalCorruptionError
from simulation.control_plane_pkg.event_log import events_path, EventLogCorruptionError

import simulation.checkpoint as ckpt


@dataclass
class DummySimCfg:
    seed: int = 0
    branch_id: str = "test"
    campaign_id: str = "camp"
    checkpoint_rotate: int = 0
    checkpoint_subdir: str = "checkpoints"
    checkpoint_include_banks: bool = False


class DummySim:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.sim_cfg = DummySimCfg()
        self.env = type("Env", (), {})()
        self.env.cfg = type("Cfg", (), {})()
        # A known knob in env cfg (float) for PressureUpdate tests.
        setattr(self.env.cfg, "tier4_juvenile_hazard_multiplier", 1.0)
        self.pop = type("Pop", (), {"agents": {}})()


def _read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def test_command_order_deterministic(tmp_path: Path):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)

    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    # Same target generation, seq ordering decides, then uuid tie-break.
    # We submit in a different order to ensure sorting is applied at boundary.
    u1 = "00000000-0000-0000-0000-000000000001"
    u2 = "00000000-0000-0000-0000-000000000002"
    u3 = "00000000-0000-0000-0000-000000000003"

    c2 = PressureUpdateCommand("cmd_v1", u2, 5, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 2.0}])
    c1 = PressureUpdateCommand("cmd_v1", u1, 5, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 3.0}])
    c3 = PressureUpdateCommand("cmd_v1", u3, 5, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 4.0}])

    cp.submit_command(c2)
    cp.submit_command(c1)
    cp.submit_command(c3)

    applied = cp.process_pending(5, sb_ctx=None, sim=sim)
    # Deterministic order by submission_sequence_number first (assigned in submit order),
    # so we expect u2 then u1 then u3.
    assert [a.command_uuid for a in applied] == [u2, u1, u3]


def test_wal_written_before_apply(tmp_path: Path):
    run_dir = str(tmp_path)

    class WalAssertingSim(DummySim):
        def __init__(self, run_dir: str):
            super().__init__(run_dir)
            self._wal_seen_at_apply = False

    sim = WalAssertingSim(run_dir)
    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    # Command that, when applied via dotpath, triggers a property setter-like check.
    # We'll monkeypatch the env cfg attribute set by creating a custom object.
    class Cfg:
        def __init__(self):
            self.tier4_juvenile_hazard_multiplier = 1.0

        def __setattr__(self, k, v):
            if k == "tier4_juvenile_hazard_multiplier" and hasattr(self, "tier4_juvenile_hazard_multiplier"):
                # At apply time, WAL must already contain SCHEDULED->APPLYING.
                entries = _read_jsonl(wal_path(run_dir))
                sim._wal_seen_at_apply = any(e.get("transition") == "SCHEDULED->APPLYING" for e in entries)
            super().__setattr__(k, v)

    sim.env.cfg = Cfg()

    u = "00000000-0000-0000-0000-000000000010"
    cmd = PressureUpdateCommand("cmd_v1", u, 2, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 9.0}])
    cp.submit_command(cmd)
    cp.process_pending(2, sb_ctx=None, sim=sim)

    assert sim._wal_seen_at_apply is True


def test_recovery_restores_pending_commands(tmp_path: Path):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)

    cp1 = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    u = "00000000-0000-0000-0000-000000000020"
    cmd = PressureUpdateCommand("cmd_v1", u, 7, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 5.0}])
    cp1.submit_command(cmd)

    # New instance (simulating restart) should know about pending command from WAL.
    cp2 = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    applied = cp2.process_pending(7, sb_ctx=None, sim=sim)
    assert [a.command_uuid for a in applied] == [u]
    assert float(sim.env.cfg.tier4_juvenile_hazard_multiplier) == 5.0


def test_command_idempotent_ack(tmp_path: Path):
    run_dir = str(tmp_path)
    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    u = "00000000-0000-0000-0000-000000000030"
    cmd = PressureUpdateCommand("cmd_v1", u, 1, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 2.0}])

    a1 = cp.submit_command(cmd)
    a2 = cp.submit_command(cmd)
    assert a1.status == "ACCEPTED"
    assert a2.status == "IDEMPOTENT_ACK"


def test_plan_freezes_when_command_enters_past(tmp_path: Path):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)
    old = os.environ.get("EIEL_ENFORCE_PLAN_FREEZE", None)
    os.environ["EIEL_ENFORCE_PLAN_FREEZE"] = "1"
    try:
        cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

        u = "00000000-0000-0000-0000-000000000040"
        cmd = PressureUpdateCommand("cmd_v1", u, 3, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 2.0}])
        cp.submit_command(cmd)

        # When we process gen 3, plan should auto-freeze via new version.
        cp.process_pending(3, sb_ctx=None, sim=sim)

        # Submitting new command after freeze should be rejected.
        u2 = "00000000-0000-0000-0000-000000000041"
        cmd2 = PressureUpdateCommand("cmd_v1", u2, 10, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 3.0}])
        ack = cp.submit_command(cmd2)
        assert ack.status == "REJECTED"
        assert "frozen" in ack.reason.lower()
    finally:
        if old is None:
            os.environ.pop("EIEL_ENFORCE_PLAN_FREEZE", None)
        else:
            os.environ["EIEL_ENFORCE_PLAN_FREEZE"] = old


def test_expectation_note_required_on_version_increment(tmp_path: Path):
    run_dir = str(tmp_path)
    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    # We don't know plan_id/version here; we can read current pointer.
    ptr = json.load(open(os.path.join(run_dir, "plans", "current_plan.json"), "r", encoding="utf-8"))
    plan_id = ptr["plan_id"]
    version = int(ptr["version"])

    amend = PlanAmendCommand(
        "cmd_v1",
        "00000000-0000-0000-0000-000000000050",
        100,  # far future
        plan_id=plan_id,
        base_plan_version=version,
        expectation_note="",  # REQUIRED
        add_commands=[],
        supercede_command_uuids=[],
    )
    ack = cp.submit_command(amend)
    assert ack.status == "REJECTED"
    assert "expectation_note" in ack.reason


def test_sequence_number_persisted_in_checkpoint(tmp_path: Path, monkeypatch):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)
    sim.control_plane = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    sim.control_plane.set_submission_sequence_number(123)
    sim._control_plane_submission_sequence_number = 123

    meta = ckpt._build_meta(sim, include_rng=False, include_banks=False, label="x")
    assert meta["control_plane"]["submission_sequence_number"] == 123

    # Now test restore path by monkeypatching heavy apply functions to no-op.
    monkeypatch.setattr(ckpt, "_apply_world", lambda sim, world: None)
    monkeypatch.setattr(ckpt, "_apply_agents_and_genomes", lambda sim, agents, genomes: None)

    payload = {"meta": meta, "world": {}, "agents": {}, "genomes": {}, "rng": None, "banks": None, "rng_snapshot_v2": None}
    sim2 = DummySim(run_dir)
    sim2.control_plane = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    ckpt.checkpoint_apply(sim2, payload, restore_rng=False, restore_banks=False)

    assert int(sim2._control_plane_submission_sequence_number) == 123
    assert int(sim2.control_plane.get_submission_sequence_number()) == 123


def test_ordering_same_after_resume(tmp_path: Path):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)

    cp1 = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    # two commands for gen 9
    u1 = "00000000-0000-0000-0000-000000000060"
    u2 = "00000000-0000-0000-0000-000000000061"
    cp1.submit_command(PressureUpdateCommand("cmd_v1", u1, 9, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 2.0}]))
    cp1.submit_command(PressureUpdateCommand("cmd_v1", u2, 9, updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 3.0}]))

    # Restart
    cp2 = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    applied = cp2.process_pending(9, sb_ctx=None, sim=sim)
    assert [a.command_uuid for a in applied] == [u1, u2]

def test_rejected_command_not_executed_after_restart(tmp_path: Path):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)

    cp1 = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    # Invalid PressureUpdate path should be rejected deterministically and durably.
    u = "00000000-0000-0000-0000-00000000dead"
    bad = PressureUpdateCommand("cmd_v1", u, 3, updates=[{"path": "env.BAD.not_allowed", "value": 1.0}])
    ack = cp1.submit_command(bad)
    assert ack.status == "REJECTED"

    # Simulate restart: rejected command must NOT be applied even if WAL contains the submission record.
    cp2 = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)
    applied = cp2.process_pending(3, sb_ctx=None, sim=sim)
    assert [a.command_uuid for a in applied] == []
    assert float(sim.env.cfg.tier4_juvenile_hazard_multiplier) == 1.0


def test_wal_corruption_fails_fast(tmp_path: Path):
    run_dir = str(tmp_path)

    # Create a corrupted WAL line.
    os.makedirs(os.path.join(run_dir, "wal"), exist_ok=True)
    with open(wal_path(run_dir), "w", encoding="utf-8") as f:
        f.write("{ this is not json }\n")

    with pytest.raises(WalCorruptionError):
        ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)


def test_event_log_corruption_fails_fast(tmp_path: Path):
    run_dir = str(tmp_path)

    os.makedirs(os.path.join(run_dir, "events"), exist_ok=True)
    with open(events_path(run_dir), "w", encoding="utf-8") as f:
        f.write("{ this is not json }\n")

    with pytest.raises(EventLogCorruptionError):
        ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)


def test_wal_contains_transition_and_target_generation(tmp_path: Path):
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)
    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    u = "00000000-0000-0000-0000-0000000000A0"
    cmd = PressureUpdateCommand(
        "cmd_v1",
        u,
        2,
        updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 7.0}],
    )
    cp.submit_command(cmd)
    cp.process_pending(2, sb_ctx=None, sim=sim)

    entries = _read_jsonl(wal_path(run_dir))
    # Expect at least SCHEDULED->APPLYING and APPLYING->APPLIED for the command.
    applying = [e for e in entries if e.get("command_uuid") == u and e.get("transition") == "SCHEDULED->APPLYING"]
    assert len(applying) == 1
    e = applying[0]
    assert e.get("generation_index") == 2
    assert e.get("target_generation") == 2


def test_rng_violation_detected(tmp_path: Path):
    import random as _random
    from simulation.control_plane_pkg import DeterminismViolationError

    run_dir = str(tmp_path)
    sim = DummySim(run_dir)
    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    # Monkeypatch PressureUpdateCommand.apply_to_sim to (incorrectly) use global RNG.
    orig_apply = PressureUpdateCommand.apply_to_sim

    def bad_apply(self, sim_obj):
        _random.random()  # should be blocked by rng_guard
        return orig_apply(self, sim_obj)

    PressureUpdateCommand.apply_to_sim = bad_apply  # type: ignore

    try:
        u = "00000000-0000-0000-0000-0000000000B0"
        cmd = PressureUpdateCommand(
            "cmd_v1",
            u,
            1,
            updates=[{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 5.0}],
        )
        cp.submit_command(cmd)

        try:
            cp.process_pending(1, sb_ctx=None, sim=sim)
            assert False, "Expected DeterminismViolationError"
        except DeterminismViolationError:
            pass
    finally:
        PressureUpdateCommand.apply_to_sim = orig_apply  # type: ignore
