from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import REPO_ROOT, run_eiel


BRIDGE_UNAVAILABLE = shutil.which("node") is None or not (REPO_ROOT / "bridge").is_dir()
BRIDGE_SKIP_REASON = "Node.js or optional repository-local Bridge unavailable"


@pytest.mark.skipif(BRIDGE_UNAVAILABLE, reason=BRIDGE_SKIP_REASON)
def test_optional_bridge_manager_init_and_list(tmp_path: Path) -> None:
    state = tmp_path / "actors"
    home = tmp_path / "bridge-home"
    env = {"CHATGPT_BRIDGE_MANAGER_HOME": str(home)}
    initialized = run_eiel(["bridgeManager", "init"], state_root=state, extra_env=env)
    assert initialized["ok"] is True
    assert Path(initialized["home"]) == home.resolve()
    listed = run_eiel(["bridgeManager", "list", "--json"], state_root=state, extra_env=env)
    assert listed == {"ok": True, "records": []}


@pytest.mark.skipif(BRIDGE_UNAVAILABLE, reason=BRIDGE_SKIP_REASON)
def test_underlying_alias_status_remains_reachable_through_run(tmp_path: Path) -> None:
    state = tmp_path / "actors"
    home = tmp_path / "bridge-home"
    env = {"CHATGPT_BRIDGE_MANAGER_HOME": str(home)}
    run_eiel(["bridgeManager", "init"], state_root=state, extra_env=env)
    url = "https://chatgpt.com/c/00000000-0000-0000-0000-000000000000"
    added = run_eiel(
        ["bridgeManager", "add", "actor-managers", url, "--project", "emergence", "--work-item", "WORK-ACTOR-MANAGERS"],
        state_root=state,
        extra_env=env,
    )
    assert added["record"]["alias"] == "actor-managers"
    shown = run_eiel(["bridgeManager", "run", "status", "actor-managers", "--json"], state_root=state, extra_env=env)
    assert shown["record"]["alias"] == "actor-managers"
    assert shown["record"]["current_work_item"] == "WORK-ACTOR-MANAGERS"
