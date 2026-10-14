"""Portable output-root selection at the real science planning consumer."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("configured", [None, "task-root"])
def test_environment_root_reaches_planning_consumer(tmp_path, configured):
    env = os.environ.copy()
    env.pop("EIEL_SCIENCE_RUNTIME_ROOT", None)
    env.pop("EIEL_RUNTIME_ROOT", None)
    expected = str(Path.home() / ".local" / "share" / "emergence" / "science-request-v1")
    if configured:
        expected = str(tmp_path / configured)
        env["EIEL_SCIENCE_RUNTIME_ROOT"] = expected
    code = '''
import json
from pathlib import Path
import eiel_composition.science_request_v1_front_door as f
def capture(loaded, compilation, **kwargs):
    assert compilation.world_bundle.world_count == 4
    return ({"parent": str(kwargs["materialization_parent"]), "effects": False}, None, None)
f._plan_loaded_request = capture
print(json.dumps(f.plan_science_request_v1("examples/release/paired_evolution.json")))
'''
    run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         text=True, capture_output=True, check=True)
    assert json.loads(run.stdout) == {"parent": expected + "/materializations", "effects": False}
    assert not (tmp_path / "task-root").exists()


@pytest.mark.parametrize("value", ["", "relative-output"])
def test_invalid_environment_root_fails_before_runtime_effects(value):
    env = {**os.environ, "EIEL_SCIENCE_RUNTIME_ROOT": value}
    run = subprocess.run([sys.executable, "-c",
                          "import eiel_composition.science_request_v1_front_door"],
                         cwd=ROOT, env=env, text=True, capture_output=True)
    assert run.returncode != 0
    assert "EIEL_SCIENCE_RUNTIME_ROOT must be a nonempty absolute path" in run.stderr
