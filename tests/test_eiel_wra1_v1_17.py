from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tests.eiel_actor_v11_helpers import run_eiel
from tests.eiel_actor_v14_helpers import make_bootstrap_repo, run_bootstrap
from tools.actors.resource_accounting import capture_start, finalize_session, query_ccusage_snapshot

FIXED = "2026-08-06T20:00:00Z"


def write_fake_ccusage(tmp_path: Path, *, supports_thread_id: bool = True) -> tuple[Path, Path, Path]:
    data = tmp_path / "ccusage-data.json"
    log = tmp_path / "ccusage-calls.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    executable = tmp_path / "ccusage"
    help_text = "USAGE: ccusage codex session --id <thread-id>" if supports_thread_id else "USAGE: ccusage codex session"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

log = Path(os.environ['FAKE_CCUSAGE_LOG'])
with log.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if '--version' in sys.argv:
    print('ccusage 99.17.0')
    raise SystemExit(0)
if '--help' in sys.argv:
    print(__CCUSAGE_HELP__)
    raise SystemExit(0)
rows = json.loads(Path(os.environ['FAKE_CCUSAGE_DATA']).read_text(encoding='utf-8'))
if '--id' in sys.argv:
    thread = sys.argv[sys.argv.index('--id') + 1]
    row = dict(rows.get(thread) or {})
    row['sessionId'] = thread
    candidates = [row]
else:
    candidates = []
    for thread, raw in rows.items():
        row = dict(raw)
        row['sessionId'] = thread
        candidates.append(row)
print(json.dumps({'sessions': candidates}))
""".replace("__CCUSAGE_HELP__", repr(help_text)),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, data, log


def ccusage_env(executable: Path, data: Path, log: Path) -> dict[str, str]:
    return {
        "EIEL_CCUSAGE_COMMAND": str(executable),
        "FAKE_CCUSAGE_DATA": str(data),
        "FAKE_CCUSAGE_LOG": str(log),
    }


def set_counters(
    data: Path,
    thread: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    cache_creation_tokens: int,
    output_tokens: int,
    model: str = "gpt-test",
) -> None:
    rows = json.loads(data.read_text(encoding="utf-8"))
    rows[thread] = {
        "inputTokens": input_tokens,
        "cacheReadTokens": cached_input_tokens,
        "cacheCreationTokens": cache_creation_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + cached_input_tokens + cache_creation_tokens + output_tokens,
        "modelsUsed": [model],
        "firstActivity": "2026-08-06T20:00:00Z",
        "lastActivity": "2026-08-06T20:01:00Z",
        "costUSD": 999.0,
        "credits": 123,
        "rawTelemetry": {"must": "not escape"},
    }
    data.write_text(json.dumps(rows), encoding="utf-8")


def create_work(
    state: Path,
    *,
    work_id: str = "WORK-WRA1",
    expected: int | None = None,
    status: str = "ACTIVE",
) -> dict[str, Any]:
    args = [
        "work",
        "create",
        "--work-id",
        work_id,
        "--title",
        "WRA-1 fixture",
        "--summary",
        "Measure Work Item resources",
        "--work-type",
        "ENGINEERING",
        "--subsystem",
        "operator-tooling",
        "--status",
        status,
        "--next-action",
        "Complete the bounded implementation.",
        "--state-root",
        str(state),
        "--generated-at",
        "2026-08-06T20:00:01Z",
    ]
    if expected is not None:
        args.extend(
            [
                "--estimated-tokens-low",
                str(expected - 50),
                "--estimated-tokens-expected",
                str(expected),
                "--estimated-tokens-high",
                str(expected + 50),
                "--estimate-confidence",
                "low",
            ]
        )
    return run_eiel(args, state_root=state)["work"]


def start_session(
    *,
    repo: Path,
    state: Path,
    bootstrap_id: str,
    thread: str,
    env: dict[str, str],
    work_id: str = "WORK-WRA1",
    at: str = "2026-08-06T20:00:02Z",
) -> dict[str, Any]:
    return run_eiel(
        [
            "session",
            "start",
            work_id,
            "--bootstrap-record",
            bootstrap_id,
            "--codex-id",
            thread,
            "--repo-root",
            str(repo),
            "--model",
            "GPT-5.3 Spark High",
            "--reasoning-level",
            "HIGH",
            "--stage",
            "implementation",
            "--current-step",
            "bounded edit",
            "--next-action",
            "Complete the bounded edit.",
            "--state-root",
            str(state),
            "--generated-at",
            at,
        ],
        state_root=state,
        extra_env=env,
    )["session"]


def bootstrap(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = make_bootstrap_repo(tmp_path)
    state = tmp_path / "state"
    bootstrap_id = run_bootstrap(
        repo=repo,
        state_root=state,
        generated_at=FIXED,
    )["bootstrap"]["bootstrap_id"]
    return repo, state, bootstrap_id


def read_ref(reference: dict[str, Any]) -> dict[str, Any]:
    path = Path(reference["path"])
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact_schema(name: str, value: dict[str, Any]) -> None:
    schema_path = Path(__file__).resolve().parents[1] / "tools" / "actors" / "schemas" / name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(value)


def assert_no_cost_or_raw_telemetry(value: Any) -> None:
    forbidden = {"cost", "costusd", "credits", "credit", "rawtelemetry", "raw_telemetry"}
    if isinstance(value, dict):
        for key, item in value.items():
            assert key.lower() not in forbidden
            assert_no_cost_or_raw_telemetry(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_cost_or_raw_telemetry(item)




def test_malformed_present_ccusage_counter_is_unavailable_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    data.write_text(
        json.dumps(
            {
                "thread-malformed": {
                    "inputTokens": "not-an-integer",
                    "cacheReadTokens": 10,
                    "cacheCreationTokens": 2,
                    "outputTokens": 3,
                    "totalTokens": 15,
                }
            }
        ),
        encoding="utf-8",
    )
    result = query_ccusage_snapshot(
        thread_id="thread-malformed",
        ccusage_command=str(executable),
    )
    assert result["status"] == "UNAVAILABLE"
    assert result.get("cumulative") is None
    assert any("inputTokens" in problem for problem in result["problems"])





def test_official_ccusage_session_json_shape_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path)
    executable.write_text(
        executable.read_text(encoding="utf-8").replace(
            "row['sessionId'] = thread\nprint(json.dumps({'sessions': [row]}))",
            "row['session'] = thread\ndecoy = {'session': 'other-thread', 'inputTokens': 999, 'outputTokens': 1, 'cacheCreationTokens': 0, 'cacheReadTokens': 0, 'totalTokens': 1000, 'models': ['decoy']}\nprint(json.dumps({'type': 'session', 'data': [decoy, row]}))",
        ),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    set_counters(
        data,
        "thread-official-shape",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    result = query_ccusage_snapshot(
        thread_id="thread-official-shape",
        ccusage_command=str(executable),
    )
    assert result["status"] == "MEASURED"
    assert result["cumulative"]["total_tokens"] == 16
    assert result["cumulative"]["models"] == ["gpt-test"]


def test_ccusage_session_collection_without_id_still_requires_exact_thread_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path, supports_thread_id=False)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    set_counters(
        data,
        "thread-v20",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    set_counters(
        data,
        "other-thread",
        input_tokens=999,
        cached_input_tokens=0,
        cache_creation_tokens=0,
        output_tokens=1,
    )

    result = query_ccusage_snapshot(thread_id="thread-v20", ccusage_command=str(executable))

    assert result["status"] == "MEASURED"
    assert result["cumulative"]["total_tokens"] == 16
    assert "--id" not in result["source"]["command"]
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert ["codex", "session", "--help"] in calls
    assert ["codex", "session", "--json", "--no-cost", "--offline"] in calls

    data.write_text(
        json.dumps(
            {
                "other-thread": {
                    "inputTokens": 999,
                    "cacheReadTokens": 0,
                    "cacheCreationTokens": 0,
                    "outputTokens": 1,
                    "totalTokens": 1000,
                }
            }
        ),
        encoding="utf-8",
    )
    unmatched = query_ccusage_snapshot(thread_id="thread-v20", ccusage_command=str(executable))
    assert unmatched["status"] == "UNAVAILABLE"
    assert any("did not return session thread-v20" in problem for problem in unmatched["problems"])


def test_ccusage_version_warning_does_not_invalidate_valid_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path)
    text = executable.read_text(encoding="utf-8")
    text = text.replace(
        "print('ccusage 99.17.0')\n    raise SystemExit(0)",
        "print('version unavailable', file=sys.stderr)\n    raise SystemExit(3)",
    )
    executable.write_text(text, encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    set_counters(
        data,
        "thread-versionless",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    result = query_ccusage_snapshot(
        thread_id="thread-versionless",
        ccusage_command=str(executable),
    )
    assert result["status"] == "MEASURED"
    assert result["cumulative"]["total_tokens"] == 16
    assert result["problems"] == []
    assert result["source"]["version"] is None
    assert result["source"]["version_warnings"]

def test_legacy_session_without_start_snapshot_is_unavailable_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    set_counters(
        data,
        "legacy-thread",
        input_tokens=1000,
        cached_input_tokens=500,
        cache_creation_tokens=100,
        output_tokens=200,
    )
    legacy_record = {
        "record_id": "SES-LEGACY-WRA",
        "created_at_utc": "2026-08-06T19:00:00Z",
        "payload": {
            "work_id": "WORK-LEGACY-WRA",
            "current_codex_id": "legacy-thread",
            "epoch": {"epoch_id": "EPOCH-LEGACY-WRA"},
            "model_assignment": {"model": "gpt-test", "reasoning_level": "HIGH"},
        },
    }
    accounting, refs = finalize_session(
        legacy_record,
        state_root=tmp_path / "state",
        generated_at="2026-08-06T20:00:00Z",
        ccusage_command=str(executable),
    )
    assert len(refs) == 1
    artifact = read_ref(refs[0])
    assert artifact["status"] == "UNAVAILABLE"
    assert artifact["total_tokens"] is None
    assert any("START snapshot is missing" in problem for problem in artifact["problems"])
    assert accounting["captures"][0]["status"] == "UNAVAILABLE"



def test_finalize_session_retry_adopts_durable_close_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    set_counters(
        data,
        "retry-thread",
        input_tokens=100,
        cached_input_tokens=20,
        cache_creation_tokens=10,
        output_tokens=30,
    )
    record = {
        "record_id": "SES-RETRY-WRA",
        "created_at_utc": "2026-08-06T19:00:00Z",
        "payload": {
            "work_id": "WORK-RETRY-WRA",
            "current_codex_id": "retry-thread",
            "epoch": {"epoch_id": "EPOCH-RETRY-WRA"},
            "model_assignment": {"model": "gpt-test", "reasoning_level": "HIGH"},
        },
    }
    accounting, _capture = capture_start(
        record,
        state_root=tmp_path / "state",
        generated_at="2026-08-06T19:01:00Z",
        ccusage_command=str(executable),
    )
    record["payload"]["resource_accounting"] = accounting

    first, first_refs = finalize_session(
        record,
        state_root=tmp_path / "state",
        generated_at="2026-08-06T20:00:00Z",
        ccusage_command=str(executable),
    )
    retried, retried_refs = finalize_session(
        record,
        state_root=tmp_path / "state",
        generated_at="2026-08-06T20:00:00Z",
        ccusage_command=str(executable),
    )

    assert retried == first
    assert retried_refs == first_refs


def test_corrected_start_replaces_current_capture_instead_of_double_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, data, log = write_fake_ccusage(tmp_path)
    monkeypatch.setenv("FAKE_CCUSAGE_DATA", str(data))
    monkeypatch.setenv("FAKE_CCUSAGE_LOG", str(log))
    set_counters(
        data,
        "corrected-thread",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    record = {
        "record_id": "SES-CORRECTED-WRA",
        "created_at_utc": "2026-08-06T19:00:00Z",
        "payload": {
            "work_id": "WORK-CORRECTED-WRA",
            "current_codex_id": "corrected-thread",
            "epoch": {"epoch_id": "EPOCH-CORRECTED-WRA"},
            "model_assignment": {"model": "gpt-test", "reasoning_level": "HIGH"},
        },
    }
    first, first_capture = capture_start(
        record,
        state_root=tmp_path / "state",
        generated_at="2026-08-06T19:01:00Z",
        ccusage_command=str(executable),
    )
    record["payload"]["resource_accounting"] = first
    set_counters(
        data,
        "corrected-thread",
        input_tokens=20,
        cached_input_tokens=4,
        cache_creation_tokens=2,
        output_tokens=6,
    )
    corrected, corrected_capture = capture_start(
        record,
        state_root=tmp_path / "state",
        generated_at="2026-08-06T19:02:00Z",
        ccusage_command=str(executable),
        correction_note="replace wrong START attribution",
    )
    assert len(corrected["captures"]) == 1
    assert corrected_capture["capture_id"] != first_capture["capture_id"]
    assert corrected_capture["supersedes_capture_ids"] == [first_capture["capture_id"]]

def test_work_creation_always_adds_static_estimate_and_accepts_explicit_range(tmp_path: Path) -> None:
    state = tmp_path / "state"
    default = create_work(state, work_id="WORK-WRA-DEFAULT")
    estimate = default["resource_estimate"]
    assert estimate["schema_version"] == "eiel_resource_estimate_v1"
    assert estimate["estimated_tokens"] == {"low": None, "expected": None, "high": None}
    assert estimate["confidence"] == "none"
    assert estimate["method"] == "static"
    assert default["resource_accounting"]["artifact"] is None

    explicit = create_work(state, work_id="WORK-WRA-EXPLICIT", expected=200)
    assert explicit["subsystem"] == "operator-tooling"
    assert explicit["resource_estimate"]["estimated_tokens"] == {"low": 150, "expected": 200, "high": 250}
    assert explicit["resource_estimate"]["confidence"] == "low"

    invalid = run_eiel(
        [
            "work",
            "create",
            "--work-id",
            "WORK-WRA-BAD",
            "--title",
            "Bad estimate",
            "--estimated-tokens-low",
            "300",
            "--estimated-tokens-expected",
            "200",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert invalid["reason_codes"] == ["resource_estimate_invalid"]


def test_creation_time_resource_estimate_cannot_be_rewritten_afterward(tmp_path: Path) -> None:
    _repo, state, _bootstrap_id = bootstrap(tmp_path)
    work = create_work(state, work_id="WORK-WRA-ESTIMATE-IMMUTABLE", expected=250)
    original = dict(work["resource_estimate"])
    rejected = run_eiel(
        [
            "work",
            "update",
            "WORK-WRA-ESTIMATE-IMMUTABLE",
            "--set",
            "resource_estimate.estimated_tokens.expected=999",
            "--state-root",
            str(state),
        ],
        state_root=state,
        expected=2,
    )
    assert rejected["reason_codes"] == ["work_managed_field_patch_forbidden"]
    inspected = run_eiel(
        ["work", "inspect", "WORK-WRA-ESTIMATE-IMMUTABLE", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert inspected["resource_estimate"] == original


def test_automatic_session_and_work_accounting_uses_ccusage_delta(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, expected=300)
    set_counters(data, "thread-a", input_tokens=100, cached_input_tokens=20, cache_creation_tokens=5, output_tokens=10)
    session = start_session(repo=repo, state=state, bootstrap_id=bootstrap_id, thread="thread-a", env=env)
    assert session["revision"] == 1
    accounting = session["resource_accounting"]
    assert len(accounting["captures"]) == 1
    capture = accounting["captures"][0]
    assert capture["work_id"] == "WORK-WRA1"
    assert capture["session_id"] == session["session_id"]
    assert capture["context_epoch_id"] == session["epoch"]["epoch_id"]
    assert capture["codex_thread_id"] == "thread-a"
    start_snapshot = read_ref(capture["start_snapshot"])
    assert start_snapshot["cumulative"]["total_tokens"] == 135

    run_eiel(
        [
            "session",
            "checkpoint",
            session["session_id"],
            "--tool-call-count",
            "7",
            "--compaction-count",
            "1",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:05:00Z",
        ],
        state_root=state,
        extra_env=env,
    )
    set_counters(data, "thread-a", input_tokens=180, cached_input_tokens=50, cache_creation_tokens=9, output_tokens=30)
    completed = run_eiel(
        [
            "session",
            "complete",
            session["session_id"],
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:12:00Z",
        ],
        state_root=state,
        extra_env=env,
    )["session"]
    session_usage = read_ref(completed["resource_accounting"]["latest_usage_artifact"])
    assert session_usage["status"] == "MEASURED"
    assert session_usage["input_tokens"] == 80
    assert session_usage["cached_input_tokens"] == 30
    assert session_usage["cache_creation_tokens"] == 4
    assert session_usage["output_tokens"] == 20
    assert session_usage["total_tokens"] == 134
    assert session_usage["duration_minutes"] == 11.967

    work = run_eiel(
        [
            "work",
            "complete",
            "WORK-WRA1",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:13:00Z",
        ],
        state_root=state,
        extra_env=env,
    )["work"]
    reference = work["resource_accounting"]["artifact"]
    usage = read_ref(reference)
    assert usage["status"] == "MEASURED"
    assert usage["model"] == "GPT-5.3 Spark High"
    assert usage["reasoning_level"] == "HIGH"
    assert usage["work_type"] == "ENGINEERING"
    assert usage["subsystem"] == "operator-tooling"
    assert usage["tool_calls"] == 7
    assert usage["compactions"] == 1
    assert usage["actual_tokens"] == 134
    assert usage["estimation_error"]["expected_tokens"] == 300
    assert usage["estimation_error"]["token_delta"] == -166
    assert usage["estimation_error"]["within_low_high"] is False
    assert set(work["resource_accounting"]) == {
        "schema_version",
        "estimate",
        "artifact",
        "estimation_error",
        "artifact_history",
    }
    assert "input_tokens" not in json.dumps(work["resource_accounting"])
    assert_no_cost_or_raw_telemetry(start_snapshot)
    assert_no_cost_or_raw_telemetry(session_usage)
    assert_no_cost_or_raw_telemetry(usage)
    validate_artifact_schema("eiel_token_snapshot_v1.schema.json", start_snapshot)
    validate_artifact_schema("eiel_session_token_usage_v1.schema.json", session_usage)
    validate_artifact_schema("eiel_token_usage_v1.schema.json", usage)

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    session_calls = [row for row in calls if "codex" in row and "--help" not in row]
    assert session_calls
    assert all("--json" in row and "--offline" in row and "--no-cost" in row for row in session_calls)


def test_missing_ccusage_and_decreasing_counters_never_block_or_invent_usage(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    create_work(state, work_id="WORK-WRA-MISSING")
    missing_env = {"EIEL_CCUSAGE_COMMAND": str(tmp_path / "does-not-exist")}
    session = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="missing-thread",
        env=missing_env,
        work_id="WORK-WRA-MISSING",
    )
    assert session["resource_accounting"]["captures"][0]["status"] == "UNAVAILABLE"
    run_eiel(
        ["session", "complete", session["session_id"], "--state-root", str(state)],
        state_root=state,
        extra_env=missing_env,
    )
    work = run_eiel(
        ["work", "complete", "WORK-WRA-MISSING", "--state-root", str(state)],
        state_root=state,
        extra_env=missing_env,
    )["work"]
    unavailable = read_ref(work["resource_accounting"]["artifact"])
    assert unavailable["status"] == "UNAVAILABLE"
    assert unavailable["actual_tokens"] is None

    # A cumulative counter reset is invalid evidence, not negative resource use.
    repo2, state2, bootstrap2 = bootstrap(tmp_path / "decrease")
    executable, data, log = write_fake_ccusage(tmp_path / "decrease")
    env = ccusage_env(executable, data, log)
    create_work(state2, work_id="WORK-WRA-DECREASE")
    set_counters(data, "thread-down", input_tokens=100, cached_input_tokens=40, cache_creation_tokens=10, output_tokens=20)
    decreasing = start_session(
        repo=repo2,
        state=state2,
        bootstrap_id=bootstrap2,
        thread="thread-down",
        env=env,
        work_id="WORK-WRA-DECREASE",
    )
    set_counters(data, "thread-down", input_tokens=90, cached_input_tokens=30, cache_creation_tokens=8, output_tokens=15)
    result = run_eiel(
        ["session", "complete", decreasing["session_id"], "--state-root", str(state2)],
        state_root=state2,
        extra_env=env,
    )["session"]
    artifact = read_ref(result["resource_accounting"]["latest_usage_artifact"])
    assert artifact["status"] == "UNAVAILABLE"
    assert artifact["total_tokens"] is None
    assert any("decreased" in problem for problem in artifact["problems"])




def test_terminal_session_outcome_closes_resource_segment(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, work_id="WORK-WRA-TERMINAL")
    set_counters(
        data,
        "thread-terminal",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    session = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="thread-terminal",
        env=env,
        work_id="WORK-WRA-TERMINAL",
    )
    set_counters(
        data,
        "thread-terminal",
        input_tokens=110,
        cached_input_tokens=12,
        cache_creation_tokens=6,
        output_tokens=23,
    )
    terminal = run_eiel(
        [
            "session",
            "discontinue",
            session["session_id"],
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:10:00Z",
        ],
        state_root=state,
        extra_env=env,
    )["session"]
    usage = read_ref(terminal["resource_accounting"]["latest_usage_artifact"])
    assert usage["status"] == "MEASURED"
    assert usage["total_tokens"] == 135
    rejected = run_eiel(
        ["token", "capture-start", session["session_id"], "--state-root", str(state)],
        state_root=state,
        extra_env=env,
        expected=2,
    )
    assert rejected["reason_codes"] == ["token_capture_terminal_session"]
    work = run_eiel(
        ["work", "complete", "WORK-WRA-TERMINAL", "--state-root", str(state)],
        state_root=state,
        extra_env=env,
    )["work"]
    assert read_ref(work["resource_accounting"]["artifact"])["total_tokens"] == 135

def test_multiple_sessions_aggregate_and_correction_preserves_old_artifact(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, expected=500)
    totals = []
    for index, thread in enumerate(("thread-one", "thread-two"), start=1):
        set_counters(data, thread, input_tokens=10 * index, cached_input_tokens=2, cache_creation_tokens=1, output_tokens=3)
        session = start_session(
            repo=repo,
            state=state,
            bootstrap_id=bootstrap_id,
            thread=thread,
            env=env,
            at=f"2026-08-06T20:0{index}:00Z",
        )
        set_counters(data, thread, input_tokens=10 * index + 100, cached_input_tokens=12, cache_creation_tokens=6, output_tokens=23)
        completed = run_eiel(
            [
                "session",
                "complete",
                session["session_id"],
                "--state-root",
                str(state),
                "--generated-at",
                f"2026-08-06T20:1{index}:00Z",
            ],
            state_root=state,
            extra_env=env,
        )["session"]
        usage = read_ref(completed["resource_accounting"]["latest_usage_artifact"])
        totals.append(usage["total_tokens"])

    work = run_eiel(
        ["work", "complete", "WORK-WRA1", "--state-root", str(state), "--generated-at", "2026-08-06T20:30:00Z"],
        state_root=state,
        extra_env=env,
    )["work"]
    original_ref = dict(work["resource_accounting"]["artifact"])
    original_bytes = Path(original_ref["path"]).read_bytes()
    original = read_ref(original_ref)
    assert original["total_tokens"] == sum(totals)
    assert original["session_count"] == 2

    corrected = run_eiel(
        [
            "token",
            "finalize-work",
            "WORK-WRA1",
            "--correction-note",
            "operator confirmed attribution",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:31:00Z",
        ],
        state_root=state,
        extra_env=env,
    )
    corrected_ref = corrected["artifact"]
    assert corrected_ref["artifact_id"] != original_ref["artifact_id"]
    assert Path(original_ref["path"]).read_bytes() == original_bytes
    inspected = run_eiel(["work", "inspect", "WORK-WRA1", "--state-root", str(state)], state_root=state)["work"]
    assert original_ref in inspected["resource_accounting"]["artifact_history"]


def test_token_report_exports_only_attached_completed_work_artifacts(tmp_path: Path) -> None:
    state = tmp_path / "state"
    create_work(state, work_id="WORK-WRA-REPORT")
    run_eiel(["work", "complete", "WORK-WRA-REPORT", "--state-root", str(state)], state_root=state)
    export = tmp_path / "export"
    report = run_eiel(
        ["token-report", "--output-dir", str(export), "--state-root", str(state), "--generated-at", FIXED],
        state_root=state,
    )["report"]
    assert report["work_item_count"] == 1
    assert (export / "WORK-WRA-REPORT" / "token_usage.json").is_file()
    assert (export / "token_report_index.json").is_file()


def test_wra1_cli_aliases_and_artifact_schemas(tmp_path: Path) -> None:
    state = tmp_path / "state"
    created = run_eiel(
        ["work", "create", "--work-id", "WORK-WRA-ALIAS", "--title", "Alias", "--state-root", str(state)],
        state_root=state,
    )["work"]
    assert created["work_id"] == "WORK-WRA-ALIAS"
    premature = run_eiel(
        ["token", "finalize-work", "WORK-WRA-ALIAS", "--state-root", str(state)],
        state_root=state,
        expected=2,
    )
    assert premature["reason_codes"] == ["token_finalize_work_not_terminal"]
    help_payload = run_eiel(["token", "capture-start", "SES-NOT-THERE", "--state-root", str(state)], state_root=state, expected=2)
    assert help_payload["reason_codes"]

    schema_root = Path(__file__).resolve().parents[1] / "tools" / "actors" / "schemas"
    store: dict[str, Any] = {}
    for path in schema_root.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        store[schema.get("$id", path.name)] = schema
        store[path.name] = schema
    resolver = jsonschema.RefResolver.from_schema(store["eiel_work_item_v1.schema.json"], store=store)
    jsonschema.Draft202012Validator(store["eiel_work_item_v1.schema.json"], resolver=resolver).validate(
        run_eiel(["work", "inspect", "WORK-WRA-ALIAS", "--state-root", str(state)], state_root=state)["record"]["payload"]
    )
    for name in (
        "eiel_resource_estimate_v1.schema.json",
        "eiel_token_snapshot_v1.schema.json",
        "eiel_session_token_usage_v1.schema.json",
        "eiel_token_usage_v1.schema.json",
        "eiel_token_report_index_v1.schema.json",
    ):
        jsonschema.Draft202012Validator.check_schema(store[name])


def test_context_epoch_boundary_closes_non_overlapping_resource_segments(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, work_id="WORK-WRA-EPOCHS", expected=250)
    set_counters(
        data,
        "thread-epochs",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    session = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="thread-epochs",
        env=env,
        work_id="WORK-WRA-EPOCHS",
        at="2026-08-06T20:00:02Z",
    )
    first_epoch = session["epoch"]["epoch_id"]

    set_counters(
        data,
        "thread-epochs",
        input_tokens=110,
        cached_input_tokens=12,
        cache_creation_tokens=6,
        output_tokens=23,
    )
    boundary = run_eiel(
        [
            "session",
            "checkpoint",
            session["session_id"],
            "--epoch-boundary-reason",
            "first compaction",
            "--tool-call-count",
            "4",
            "--compaction-count",
            "1",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:10:00Z",
        ],
        state_root=state,
        extra_env=env,
    )
    first_usage_ref = boundary["token_usage_artifacts"][0]
    first_usage = read_ref(first_usage_ref)
    assert first_usage["context_epoch_id"] == first_epoch
    assert first_usage["status"] == "MEASURED"
    assert first_usage["total_tokens"] == 135

    resumed = run_eiel(
        [
            "session",
            "resume",
            session["session_id"],
            "--codex-id",
            "thread-epochs",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:11:00Z",
        ],
        state_root=state,
        extra_env=env,
    )["session"]
    second_epoch = resumed["epoch"]["epoch_id"]
    assert second_epoch != first_epoch
    assert resumed["resource_accounting"]["active_capture_id"]

    run_eiel(
        [
            "session",
            "checkpoint",
            session["session_id"],
            "--tool-call-count",
            "3",
            "--compaction-count",
            "0",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:19:00Z",
        ],
        state_root=state,
        extra_env=env,
    )
    set_counters(
        data,
        "thread-epochs",
        input_tokens=160,
        cached_input_tokens=22,
        cache_creation_tokens=11,
        output_tokens=33,
    )
    completed = run_eiel(
        [
            "session",
            "complete",
            session["session_id"],
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:20:00Z",
        ],
        state_root=state,
        extra_env=env,
    )["session"]
    captures = completed["resource_accounting"]["captures"]
    assert len(captures) == 2
    usages = [read_ref(dict(capture["usage_artifact"])) for capture in captures]
    assert [row["context_epoch_id"] for row in usages] == [first_epoch, second_epoch]
    assert [row["total_tokens"] for row in usages] == [135, 75]
    assert [row["tool_calls"] for row in usages] == [4, 3]
    assert [row["compactions"] for row in usages] == [1, 0]

    work = run_eiel(
        [
            "work",
            "complete",
            "WORK-WRA-EPOCHS",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:21:00Z",
        ],
        state_root=state,
        extra_env=env,
    )["work"]
    work_usage = read_ref(work["resource_accounting"]["artifact"])
    assert work_usage["measured_segment_count"] == 2
    assert work_usage["total_tokens"] == 210
    assert work_usage["tool_calls"] == 7
    assert work_usage["compactions"] == 1


def test_codex_thread_cannot_overlap_resource_intervals_across_work_items(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, work_id="WORK-WRA-THREAD-A")
    create_work(state, work_id="WORK-WRA-THREAD-B")
    set_counters(
        data,
        "thread-exclusive",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    first = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="thread-exclusive",
        env=env,
        work_id="WORK-WRA-THREAD-A",
    )
    conflict = run_eiel(
        [
            "session",
            "start",
            "WORK-WRA-THREAD-B",
            "--bootstrap-record",
            bootstrap_id,
            "--codex-id",
            "thread-exclusive",
            "--repo-root",
            str(repo),
            "--stage",
            "implementation",
            "--current-step",
            "bounded edit",
            "--next-action",
            "Complete the bounded edit.",
            "--state-root",
            str(state),
        ],
        state_root=state,
        extra_env=env,
        expected=2,
    )
    assert conflict["reason_codes"] == ["resource_thread_attribution_conflict"]

    set_counters(
        data,
        "thread-exclusive",
        input_tokens=20,
        cached_input_tokens=4,
        cache_creation_tokens=2,
        output_tokens=6,
    )
    run_eiel(
        [
            "session",
            "checkpoint",
            first["session_id"],
            "--epoch-boundary-reason",
            "handoff to another Work Item",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:05:00Z",
        ],
        state_root=state,
        extra_env=env,
    )
    second = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="thread-exclusive",
        env=env,
        work_id="WORK-WRA-THREAD-B",
        at="2026-08-06T20:06:00Z",
    )
    assert second["resource_accounting"]["active_capture_id"]


def test_capture_end_retry_is_idempotent_and_returns_current_artifact(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, work_id="WORK-WRA-END-RETRY")
    set_counters(
        data,
        "thread-end-retry",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    session = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="thread-end-retry",
        env=env,
        work_id="WORK-WRA-END-RETRY",
    )
    set_counters(
        data,
        "thread-end-retry",
        input_tokens=20,
        cached_input_tokens=4,
        cache_creation_tokens=2,
        output_tokens=6,
    )
    completed = run_eiel(
        ["session", "complete", session["session_id"], "--state-root", str(state)],
        state_root=state,
        extra_env=env,
    )["session"]
    original_ref = dict(completed["resource_accounting"]["latest_usage_artifact"])
    original_bytes = Path(original_ref["path"]).read_bytes()
    retry = run_eiel(
        ["token", "capture-end", session["session_id"], "--state-root", str(state)],
        state_root=state,
        extra_env=env,
    )
    assert retry["status"] == "MEASURED"
    assert retry["token_usage_artifacts"] == [original_ref]
    assert Path(original_ref["path"]).read_bytes() == original_bytes


def test_session_usage_correction_creates_new_immutable_artifact(tmp_path: Path) -> None:
    repo, state, bootstrap_id = bootstrap(tmp_path)
    executable, data, log = write_fake_ccusage(tmp_path)
    env = ccusage_env(executable, data, log)
    create_work(state, work_id="WORK-WRA-SESSION-CORRECTION")
    set_counters(
        data,
        "thread-session-correction",
        input_tokens=10,
        cached_input_tokens=2,
        cache_creation_tokens=1,
        output_tokens=3,
    )
    session = start_session(
        repo=repo,
        state=state,
        bootstrap_id=bootstrap_id,
        thread="thread-session-correction",
        env=env,
        work_id="WORK-WRA-SESSION-CORRECTION",
    )
    set_counters(
        data,
        "thread-session-correction",
        input_tokens=110,
        cached_input_tokens=12,
        cache_creation_tokens=6,
        output_tokens=23,
    )
    completed = run_eiel(
        ["session", "complete", session["session_id"], "--state-root", str(state)],
        state_root=state,
        extra_env=env,
    )["session"]
    original_ref = dict(completed["resource_accounting"]["latest_usage_artifact"])
    original_bytes = Path(original_ref["path"]).read_bytes()
    assert read_ref(original_ref)["total_tokens"] == 135

    # Simulate corrected ccusage telemetry while the operator explicitly
    # confirms that the current cumulative values represent the intended END.
    set_counters(
        data,
        "thread-session-correction",
        input_tokens=120,
        cached_input_tokens=14,
        cache_creation_tokens=7,
        output_tokens=25,
    )
    corrected = run_eiel(
        [
            "token",
            "capture-end",
            session["session_id"],
            "--correction-note",
            "ccusage source correction confirmed at the original boundary",
            "--state-root",
            str(state),
            "--generated-at",
            "2026-08-06T20:30:00Z",
        ],
        state_root=state,
        extra_env=env,
    )
    corrected_ref = dict(corrected["token_usage_artifacts"][0])
    assert corrected_ref["artifact_id"] != original_ref["artifact_id"]
    assert Path(original_ref["path"]).read_bytes() == original_bytes
    corrected_usage = read_ref(corrected_ref)
    assert corrected_usage["total_tokens"] == 150
    inspected = run_eiel(
        ["session", "inspect", session["session_id"], "--state-root", str(state)],
        state_root=state,
    )["session"]
    assert original_ref in inspected["resource_accounting"]["usage_artifacts"]
    assert corrected_ref in inspected["resource_accounting"]["usage_artifacts"]
    assert inspected["resource_accounting"]["latest_usage_artifact"] == corrected_ref
