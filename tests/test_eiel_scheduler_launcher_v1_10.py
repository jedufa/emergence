from __future__ import annotations

import json
import os
from pathlib import Path

from tests.eiel_scheduler_v110_helpers import make_scheduler_fixture
from tools.scheduler.dispatch import claim_assignment, request_assignment
from tools.scheduler.launcher import CodexExecLauncher, probe_codex_cli, process_identity_is_live, process_start_identity


def _fake_codex(path: Path) -> Path:
    script = path / "codex"
    script.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
FLAGS = '--json --color --sandbox --ask-for-approval --cd --output-last-message --output-schema --ephemeral'
if '--help' in sys.argv:
    print(FLAGS)
    raise SystemExit(0)
args = sys.argv[1:]
Path(os.environ['EIEL_FAKE_CODEX_ARGS']).write_text(json.dumps(args), encoding='utf-8')
out = args[args.index('--output-last-message') + 1]
Path(out).write_text('{}\\n', encoding='utf-8')
print(json.dumps({'type': 'thread.started', 'thread_id': 'thread-fixture'}))
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_codex_capability_probe_is_optional_when_executable_is_missing() -> None:
    result = probe_codex_cli("definitely-not-an-eiel-codex-binary")
    assert result["ok"] is False
    assert result["status"] == "OPTIONAL_UNAVAILABLE"
    assert result["optionality_audit"] == "NOT_MEASURED"


def test_codex_exec_launcher_uses_bounded_argv_and_fresh_exec_contract(tmp_path: Path, monkeypatch) -> None:
    fixture = make_scheduler_fixture(tmp_path)
    assignment = request_assignment(state_root=fixture.state)
    assert assignment is not None
    assignment, lease = claim_assignment(assignment["record_id"], scheduler_id="SCH-LAUNCHER-TEST", state_root=fixture.state)

    executable = _fake_codex(tmp_path)
    args_capture = tmp_path / "codex-args.json"
    monkeypatch.setenv("EIEL_FAKE_CODEX_ARGS", str(args_capture))
    monkeypatch.setenv("EIEL_SCHEDULER_TEST_MODE", "1")
    launcher = CodexExecLauncher(executable=str(executable))
    assert launcher.probe()["ok"] is True

    result_path = tmp_path / "result.json"
    launched = launcher.launch(
        assignment=assignment,
        lease=lease,
        state_root=fixture.state,
        result_path=result_path,
        log_dir=tmp_path / "logs",
    )
    assert launched.returncode == 0
    assert launched.worker_identity["thread_id"] == "thread-fixture"
    assert launched.worker_identity["worker_id"] == f"codex-{assignment['record_id'].lower()}"
    assert launched.command[0] == str(executable)
    assert launched.command[-1] == "<bounded-worker-prompt>"
    args = json.loads(args_capture.read_text(encoding="utf-8"))
    assert args[:3] == ["--ask-for-approval", "never", "--cd"]
    assert "exec" in args
    assert "--sandbox" in args
    assert "--output-schema" in args
    assert "--output-last-message" in args
    assert "--ephemeral" in args
    assert result_path.is_file()
    prompt = launcher._prompt(assignment, lease, state_root=fixture.state, result_path=result_path)
    assert "mechanically activated" in prompt
    assert "attach-bootstrap" in prompt
    assert "pause or interrupt" in prompt


def test_process_identity_rejects_pid_reuse_or_missing_identity() -> None:
    pid = os.getpid()
    identity = process_start_identity(pid)
    assert identity is not None
    assert process_identity_is_live(pid, identity) is True
    assert process_identity_is_live(pid, identity + "-wrong") is False
    assert process_identity_is_live(None, identity) is False
