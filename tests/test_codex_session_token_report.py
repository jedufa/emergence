from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pytest

from tools.analysis import codex_session_token_report as report


def test_recent_sessions_uses_exact_utc_cutoff_and_sorts_newest_first() -> None:
    cutoff = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    rows = [
        {"sessionId": "old", "lastActivity": "2026-08-06T11:59:59Z"},
        {"sessionId": "boundary", "lastActivity": "2026-08-06T12:00:00Z"},
        {"sessionId": "new", "lastActivity": "2026-08-06T13:00:00+00:00"},
        {"sessionId": "missing"},
    ]

    selected = report.recent_sessions(rows, cutoff=cutoff)

    assert [row["sessionId"] for row in selected] == ["new", "boundary"]


def test_query_ccusage_requests_redacted_offline_json_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    cutoff = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    observed: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.extend(command)
        assert kwargs["timeout"] == 30
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "sessions": [
                        {"sessionId": "recent", "lastActivity": "2026-08-06T12:01:00Z"},
                        {"sessionId": "old", "lastActivity": "2026-08-06T11:00:00Z"},
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(report.subprocess, "run", fake_run)

    selected = report.query_ccusage(cutoff=cutoff, executable="fake-ccusage")

    assert [row["sessionId"] for row in selected] == ["recent"]
    assert observed == [
        "fake-ccusage", "codex", "session", "--since", "2026-08-06", "--timezone", "UTC", "--json", "--no-cost", "--offline"
    ]


def test_render_report_contains_only_requested_counters() -> None:
    generated_at = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    rendered = report.render_report(
        [
            {
                "sessionId": "session-a",
                "lastActivity": "2026-08-08T11:59:00Z",
                "inputTokens": 1000,
                "cacheReadTokens": 2000,
                "outputTokens": 300,
                "reasoningOutputTokens": 200,
                "totalTokens": 3300,
                "sessionFile": "must-not-appear.jsonl",
                "directory": "must-not-appear",
            }
        ],
        generated_at=generated_at,
        cutoff=generated_at,
    )

    assert "- Input: 1,000" in rendered
    assert "- Cache-read: 2,000" in rendered
    assert "- Output: 300" in rendered
    assert "- Reasoning output: 200" in rendered
    assert "- Total: 3,300" in rendered
    assert "must-not-appear" not in rendered


def test_render_report_rejects_invalid_counter() -> None:
    with pytest.raises(report.CodexSessionReportError, match="inputTokens"):
        report.render_report(
            [{"sessionId": "bad", "lastActivity": "2026-08-08T12:00:00Z", "inputTokens": -1}],
            generated_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            cutoff=datetime(2026, 8, 8, 11, 0, tzinfo=timezone.utc),
        )


def test_render_live_limits_display_to_twenty_session_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report.shutil, "get_terminal_size", lambda fallback: report.os.terminal_size((100, 30)))
    generated_at = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
    rows = [
        {
            "sessionId": f"session-{index}",
            "lastActivity": "2026-08-08T11:59:00Z",
            "totalTokens": 100,
            "_profile": report.RolloutProfile(thread_name=f"subject-{index}"),
        }
        for index in range(21)
    ]

    rendered = report.render_live(rows, generated_at=generated_at, cutoff=generated_at)

    assert "sessions 21 | showing 20" in rendered
    assert "subject-19" in rendered
    assert "subject-20" not in rendered


def test_profile_rollout_uses_latest_thread_settings_model(tmp_path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    events = [
        {
            "timestamp": "2026-08-08T11:00:00Z",
            "type": "turn_context",
            "payload": {"model": "gpt-5.6-luna"},
        },
        {
            "timestamp": "2026-08-08T11:01:00Z",
            "type": "event_msg",
            "payload": {
                "type": "thread_settings_applied",
                "thread_settings": {"model": "gpt-5.6-terra"},
            },
        },
    ]
    rollout.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    profile = report.profile_rollout(rollout)

    assert profile.model == "gpt-5.6-terra"
