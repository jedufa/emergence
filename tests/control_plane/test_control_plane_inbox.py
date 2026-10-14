"""tests/control_plane/test_control_plane_inbox.py

Regression test: ControlPlane filesystem inbox handling.

Why this exists
---------------
The Web UI (Phase 5d) submits ControlPlane commands by writing JSON files into
<run_dir>/inbox/.

The simulator must:
- poll inbox
- submit parsed commands
- mark inbox files as processed (move to inbox_processed)

This test ensures inbox files do not crash the sim loop and are moved aside.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from simulation.control_plane_pkg import ControlPlane


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
        setattr(self.env.cfg, "tier4_juvenile_hazard_multiplier", 1.0)
        self.pop = type("Pop", (), {"agents": {}})()


def test_control_plane_inbox_file_processed(tmp_path: Path) -> None:
    run_dir = str(tmp_path)
    sim = DummySim(run_dir)

    cp = ControlPlane(run_dir=run_dir, branch_id="b1", campaign_id="c1", initial_generation=0)

    inbox_dir = Path(run_dir) / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)

    cmd = {
        "schema_version": "cmd_v1",
        "command_type": "PressureUpdateCommand",
        "command_uuid": "00000000-0000-0000-0000-00000000BEEF",
        "target_generation": 0,
        "updates": [{"path": "env.cfg.tier4_juvenile_hazard_multiplier", "value": 9.0}],
    }

    inbox_file = inbox_dir / "gen000000__PressureUpdateCommand__00000000-0000-0000-0000-00000000BEEF.json"
    inbox_file.write_text(json.dumps(cmd), encoding="utf-8")

    applied = cp.process_pending(0, sb_ctx=None, sim=sim)

    # Command applied.
    assert applied and applied[0].command_uuid == cmd["command_uuid"]
    assert float(getattr(sim.env.cfg, "tier4_juvenile_hazard_multiplier")) == 9.0

    # Inbox file should be moved out.
    assert not inbox_file.exists()
    processed = Path(run_dir) / "inbox_processed"
    assert processed.exists() and processed.is_dir()
    moved = list(processed.iterdir())
    assert any(p.name.startswith("gen000000__PressureUpdateCommand") for p in moved)

