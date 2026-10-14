#!/usr/bin/env python3
"""Write a redacted per-session Codex token report from ccusage JSON.

The report combines ccusage token counters with best-effort, locally derived
session metadata from Codex rollout JSONL files when available. Raw transcript
content is never emitted.

Use --live for an in-place terminal dashboard refreshed every two seconds.\nNormal-screen redraw is the default so mouse-wheel input remains harmless.\nPersisted /rename titles are read from ~/.codex/session_index.jsonl.\nUse --forensics to inspect per-model-call context growth from Codex token_count events.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


class CodexSessionReportError(RuntimeError):
    """The external session report could not be used safely."""


WORK_RE = re.compile(r"\bWORK-[A-Za-z0-9][A-Za-z0-9_-]*\b")
WP_RE = re.compile(r"\b(?:WP\d+|WORK-[A-Za-z0-9][A-Za-z0-9_-]*-WP\d+)\b", re.IGNORECASE)
COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
TOKENISH_KEYS = {
    "inputtokens",
    "cachereadtokens",
    "cachecreationtokens",
    "outputtokens",
    "reasoningoutputtokens",
    "totaltokens",
}

DISPLAY_TZ = ZoneInfo("Europe/Vienna")
DISPLAY_TZ_NAME = "Europe/Vienna"
LIVE_MAX_SESSION_ROWS = 40


@dataclass
class RolloutProfile:
    path: str | None = None
    start_time: str | None = None
    last_entry_time: str | None = None
    duration_seconds: float | None = None
    model: str | None = None
    thread_name: str | None = None
    thread_name_source: str | None = None
    subject: str | None = None
    subject_source: str | None = None
    purpose: str | None = None
    work_ids: list[str] = field(default_factory=list)
    work_package_ids: list[str] = field(default_factory=list)
    tool_calls: int = 0
    exec_calls: int = 0
    wait_calls: int = 0
    compactions: int = 0
    files_changed: int | None = None
    commits: list[str] = field(default_factory=list)
    outcome: str | None = None
    workflow_generation: str | None = None
    context_mode: str | None = None
    metadata_source: str = "ccusage_only"


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _as_display_tz(value: str | datetime) -> datetime:
    """Convert a timezone-aware value to the configured human display timezone."""

    parsed = _as_utc(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(DISPLAY_TZ)


def _display_timestamp(value: str | datetime | None, *, seconds: bool = True) -> str:
    if value is None:
        return "unknown"
    try:
        parsed = _as_display_tz(value)
    except (TypeError, ValueError):
        return str(value)
    fmt = "%Y-%m-%d %H:%M:%S %Z" if seconds else "%Y-%m-%d %H:%M %Z"
    return parsed.strftime(fmt)


def _iso_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _as_utc(value).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _counter(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if isinstance(value, bool):
        raise CodexSessionReportError(f"{key} must be a non-negative integer")
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise CodexSessionReportError(f"{key} must be a non-negative integer") from exc
    if parsed < 0:
        raise CodexSessionReportError(f"{key} must be a non-negative integer")
    return parsed


def recent_sessions(rows: Sequence[object], *, cutoff: datetime) -> list[dict[str, Any]]:
    """Return rows with an exact, timezone-aware last-activity cutoff."""

    selected: list[tuple[datetime, dict[str, Any]]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        value = raw.get("lastActivity")
        if not isinstance(value, str):
            continue
        try:
            observed = _as_utc(value)
        except ValueError:
            continue
        if observed >= cutoff:
            selected.append((observed, raw))
    return [row for _, row in sorted(selected, key=lambda pair: pair[0], reverse=True)]


def query_ccusage(*, cutoff: datetime, executable: str = "ccusage") -> list[dict[str, Any]]:
    """Retrieve structured Codex session rows without costs or raw transcripts."""

    command = [
        executable,
        "codex",
        "session",
        "--since",
        cutoff.date().isoformat(),
        "--timezone",
        "UTC",
        "--json",
        "--no-cost",
        "--offline",
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodexSessionReportError(f"ccusage query failed: {exc}") from exc
    if result.returncode != 0:
        raise CodexSessionReportError(result.stderr.strip() or f"ccusage exited {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CodexSessionReportError(f"ccusage returned invalid JSON: {exc}") from exc
    rows = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise CodexSessionReportError("ccusage did not return a sessions list")
    return recent_sessions(rows, cutoff=cutoff)


def _session_rollout_name(session_id: str) -> str | None:
    leaf = session_id.rsplit("/", 1)[-1]
    return leaf if leaf.startswith("rollout-") else None


def _candidate_rollout_paths(session_id: str, codex_home: Path) -> Iterable[Path]:
    leaf = _session_rollout_name(session_id)
    if not leaf:
        return
    relative = Path(session_id + ("" if session_id.endswith(".jsonl") else ".jsonl"))
    yield codex_home / "sessions" / relative
    yield codex_home / relative
    # Compatibility fallbacks. Keep bounded to the date directory/name rather
    # than recursively walking the full ~/.codex tree.
    parts = session_id.split("/")
    if len(parts) >= 4:
        date_dir = Path(*parts[:3])
        yield codex_home / "sessions" / date_dir / f"{leaf}.jsonl"
        yield codex_home / "rollouts" / date_dir / f"{leaf}.jsonl"


def find_rollout(session_id: str, codex_home: Path) -> Path | None:
    for path in _candidate_rollout_paths(session_id, codex_home):
        if path.is_file():
            return path
    leaf = _session_rollout_name(session_id)
    if not leaf:
        return None
    # Last-resort bounded glob only in likely session roots.
    for root in (codex_home / "sessions", codex_home / "rollouts"):
        if root.is_dir():
            matches = list(root.glob(f"**/{leaf}.jsonl"))
            if matches:
                return matches[0]
    return None


def _walk_strings(value: Any, *, max_strings: int = 250) -> list[str]:
    """Collect a bounded number of strings from one JSON event."""

    found: list[str] = []
    stack = [value]
    while stack and len(found) < max_strings:
        item = stack.pop()
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, dict):
            # Prefer semantically useful keys before arbitrary values.
            preferred = []
            rest = []
            for key, val in item.items():
                if str(key).lower() in {
                    "content", "text", "message", "prompt", "title", "name",
                    "model", "type", "event", "event_type", "timestamp",
                    "created_at", "time", "role", "command", "cmd",
                }:
                    preferred.append(val)
                else:
                    rest.append(val)
            stack.extend(reversed(rest))
            stack.extend(reversed(preferred))
        elif isinstance(item, list):
            stack.extend(reversed(item))
    return found


def _event_timestamp(event: Mapping[str, Any]) -> str | None:
    for key in ("timestamp", "created_at", "createdAt", "time", "ts"):
        parsed = _iso_or_none(event.get(key))
        if parsed:
            return parsed
    return None


def _event_kind(event: Mapping[str, Any]) -> str:
    pieces = []
    for key in ("type", "event", "event_type", "kind", "name"):
        val = event.get(key)
        if isinstance(val, str):
            pieces.append(val.lower())
    payload = event.get("payload")
    if isinstance(payload, dict):
        for key in ("type", "event", "event_type", "kind", "name"):
            val = payload.get(key)
            if isinstance(val, str):
                pieces.append(val.lower())
    return " ".join(pieces)


def _model_from_event(event: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [event]
    payload = event.get("payload")
    if isinstance(payload, dict):
        candidates.append(payload)
        # A mid-session model change is emitted as an event message whose
        # selected model lives in ``thread_settings``, rather than directly in
        # the event payload.  Include it in the same ordered event scan so the
        # last observed setting becomes the displayed current model.
        thread_settings = payload.get("thread_settings")
        if isinstance(thread_settings, dict):
            candidates.append(thread_settings)
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for key in ("model", "model_name", "modelName"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _input_text_values(value: Any) -> list[str]:
    """Extract Codex `input_text` strings from a structured message payload."""

    found: list[str] = []

    if isinstance(value, dict):
        item_type = str(value.get("type") or "").lower()
        if item_type == "input_text":
            for key in ("text", "content", "value"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    found.append(candidate.strip())
                    break
        for child in value.values():
            found.extend(_input_text_values(child))

    elif isinstance(value, list):
        for child in value:
            found.extend(_input_text_values(child))

    return found


def _is_injected_user_context(text: str) -> bool:
    """Return True for Codex bootstrap/control text masquerading as user input."""

    low = re.sub(r"\s+", " ", text).strip().lower()
    if not low:
        return True

    injected_prefixes = (
        "# agents.md instructions",
        "<environment_context",
        "# collaboration mode:",
        "collaboration mode:",
        "# model set context",
        "<instructions>",
        "<developer",
        "<system",
    )
    if low.startswith(injected_prefixes):
        return True

    injected_fragments = (
        "you are now in default mode. any previous instructions",
        "<cwd>",
        "<shell>",
        "<current_date>",
        "<timezone>",
        "personal codex output preferences",
    )
    # These fragments are sufficient to reject a bootstrap blob, but do not
    # reject a genuine user prompt merely for mentioning AGENTS.md in passing.
    if sum(fragment in low for fragment in injected_fragments) >= 2:
        return True

    return False


def _structural_user_texts(event: Mapping[str, Any]) -> list[str]:
    """Extract real operator text from known Codex rollout event shapes.

    Current Codex rollouts represent the actual operator prompt as:

      type=response_item
      payload.type=message
      payload.role=user
      payload.content=[{type: input_text, text: ...}]

    The same role also carries bootstrap injections earlier in the rollout, so
    those are filtered explicitly. Older event-message variants are accepted as
    a fallback.
    """

    event_type = str(event.get("type") or "").lower()
    payload = event.get("payload")
    texts: list[str] = []

    if isinstance(payload, dict):
        payload_type = str(payload.get("type") or "").lower()
        payload_role = str(payload.get("role") or "").lower()

        if event_type == "response_item" and payload_type == "message" and payload_role == "user":
            texts.extend(_input_text_values(payload.get("content")))

        elif payload_type == "user_message":
            # Older Codex event shape.
            for key in ("message", "text", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    texts.append(value.strip())
                else:
                    texts.extend(_input_text_values(value))

    # Very old/simple event shape fallback.
    if event_type in {"user_message", "user"}:
        for key in ("message", "text", "content"):
            value = event.get(key)
            if isinstance(value, str):
                texts.append(value.strip())
            else:
                texts.extend(_input_text_values(value))

    result: list[str] = []
    for value in texts:
        value = value.strip()
        if value and not _is_injected_user_context(value):
            result.append(value)
    return result


_GENERIC_PROMPTS = {
    "continue",
    "continue.",
    "resume",
    "resume.",
    "fix it",
    "fix it.",
    "do it",
    "do it.",
    "accept it",
    "accept it.",
    "commit it",
    "commit it.",
    "commit it all",
    "commit it all.",
    "go",
    "go.",
}


def _subject_from_user_prompt(text: str, *, limit: int = 100) -> str:
    """Turn a genuine operator prompt into a compact descriptive thread subject."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip("`# \t\r\n")
    if not cleaned:
        return "Unrenamed thread"

    # Long implementation prompts very often begin with a descriptive title
    # followed by OBJECTIVE. Prefer that title over the entire prompt.
    objective = re.match(r"(?is)^(.{8,120}?)\s+(?:OBJECTIVE|PURPOSE)\b", cleaned)
    if objective:
        candidate = objective.group(1).strip(" :-")
        if candidate:
            return _sanitize_subject(candidate, limit=limit)

    # Prefer a concise first sentence/question when present.
    sentence = re.match(r"^(.{8,160}?[.!?])(?:\s|$)", cleaned)
    if sentence:
        return _sanitize_subject(sentence.group(1), limit=limit)

    return _sanitize_subject(cleaned, limit=limit)


def _user_text_from_event(event: Mapping[str, Any]) -> str | None:
    """Compatibility wrapper returning the first genuine operator text."""

    texts = _structural_user_texts(event)
    return texts[0] if texts else None

def _extract_explicit_thread_name(event: Mapping[str, Any]) -> str | None:
    """Extract an explicit Codex thread name, preferring /rename commands.

    Codex versions may encode a rename either as ordinary user text or as
    structured title/name metadata.  This function is deliberately permissive
    on input shape but conservative about what it accepts as a title.
    """

    strings = _walk_strings(event, max_strings=160)

    # Most robust cross-version signal: the user's explicit /rename command.
    for value in strings:
        match = re.search(r"(?im)(?:^|\n)\s*/rename(?:\s+|=)([^\r\n]+)", value)
        if match:
            title = match.group(1).strip().strip('"').strip("'")
            if title and len(title) <= 200:
                return _sanitize_subject(title, limit=120)

    # Best-effort structured metadata used by some Codex builds.
    candidates: list[Mapping[str, Any]] = [event]
    payload = event.get("payload")
    if isinstance(payload, dict):
        candidates.append(payload)
    for obj in candidates:
        for key in (
            "thread_name", "threadName", "thread_title", "threadTitle",
            "conversation_name", "conversationName", "conversation_title",
            "conversationTitle", "session_title", "sessionTitle",
        ):
            value = obj.get(key)
            if isinstance(value, str):
                title = value.strip()
                if title and len(title) <= 200 and "://" not in title:
                    return _sanitize_subject(title, limit=120)
    return None


def _sanitize_subject(text: str, *, limit: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{10,}|password\s*[:=]\s*\S+|token\s*[:=]\s*\S+)\b", "[redacted]", text)
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _infer_purpose(subject: str | None, work_ids: Sequence[str], wp_ids: Sequence[str]) -> str | None:
    text = (subject or "").lower()
    rules = (
        ("review", ("review", "audit", "assess", "compare")),
        ("qualification", ("qualif", "parity", "benchmark")),
        ("repair", ("fix", "repair", "bug", "regression", "adapt")),
        ("implementation", ("implement", "wire", "add ", "build", "create")),
        ("planning", ("plan", "decompos", "work package", "work item")),
        ("integration", ("integrat", "merge", "reconcile")),
        ("science", ("campaign", "science", "wrc", "experiment")),
        ("inspection", ("inspect", "status", "what is", "find out", "analy")),
    )
    for purpose, needles in rules:
        if any(needle in text for needle in needles):
            return purpose
    if wp_ids or work_ids:
        return "implementation"
    return None


def _infer_outcome(texts: Sequence[str]) -> str | None:
    joined = "\n".join(texts[-120:]).lower()
    if re.search(r"\b(blocked|hard stop|cannot proceed|not ready)\b", joined):
        return "blocked"
    if re.search(r"\b(changes_required|rejected|reject)\b", joined):
        return "review_negative"
    if re.search(r"\b(accept|accepted)\b", joined) and "review" in joined:
        return "review_accepted"
    if re.search(r"\b(commit(?:ted)?|new commit|implementation complete|work complete|completed)\b", joined):
        return "delivered"
    return None


def _infer_context_mode(event_count: int, first_user_text: str | None) -> str:
    # A rollout is normally one fresh Codex thread. "resumed" is only claimed
    # when the visible first user instruction explicitly says so.
    if first_user_text and re.search(r"\b(resume|continue|pick up|handoff)\b", first_user_text, re.IGNORECASE):
        return "resumed_or_continuation"
    return "fresh"


def _infer_workflow_generation(texts: Sequence[str]) -> str | None:
    joined = "\n".join(texts[:180]).lower()
    # These are intentionally descriptive heuristics, not authority.
    if "active_goal.md" in joined or "eiel autonomous work protocol" in joined:
        return "legacy_or_transition"
    if "python -m tools.eiel start" in joined and (
        "work package" in joined or "sessionmanager" in joined or "workspace" in joined
    ):
        return "new_repository_workflow"
    return None


def profile_rollout(path: Path) -> RolloutProfile:
    profile = RolloutProfile(path=str(path), metadata_source="ccusage+rollout_jsonl")
    timestamps: list[str] = []
    texts: list[str] = []
    user_prompts: list[str] = []
    explicit_thread_name: str | None = None
    work_ids: set[str] = set()
    wp_ids: set[str] = set()
    commits: set[str] = set()
    changed_paths: set[str] = set()
    event_count = 0

    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return profile

    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_count += 1

            timestamp = _event_timestamp(event)
            if timestamp:
                timestamps.append(timestamp)

            model = _model_from_event(event)
            if model:
                profile.model = model

            kind = _event_kind(event)
            strings = _walk_strings(event, max_strings=120)
            combined = "\n".join(strings)
            if combined:
                texts.append(combined[:6000])
                work_ids.update(WORK_RE.findall(combined))
                wp_ids.update(match.group(0).upper() for match in WP_RE.finditer(combined))

            for prompt in _structural_user_texts(event):
                if prompt not in user_prompts:
                    user_prompts.append(prompt)

            renamed = _extract_explicit_thread_name(event)
            if renamed:
                explicit_thread_name = renamed

            if any(token in kind for token in ("tool_call", "function_call", "toolcall", "functioncall")):
                profile.tool_calls += 1
            elif any(
                isinstance(v, dict) and (
                    v.get("type") in {"function_call", "tool_call"}
                    or v.get("role") == "tool"
                )
                for v in event.values()
            ):
                profile.tool_calls += 1

            lower = combined.lower()
            if "exec_command" in lower or "shell" in kind or "command_execution" in kind:
                profile.exec_calls += 1
            if "wait" in kind or "write_stdin" in lower or "yield_time" in lower:
                profile.wait_calls += 1
            if "compact" in kind or "context_compact" in lower or "context compacted" in lower:
                profile.compactions += 1

            # Best-effort durable-output signals only. These are heuristic and
            # deliberately not used as proof of completion.
            if "git commit" in lower or "new commit" in lower or "committed" in lower:
                for match in COMMIT_RE.findall(combined):
                    commits.add(match.lower())
            for match in re.findall(r"(?:modified:|create mode \d+|delete mode \d+)\s+([^\s]+)", combined):
                changed_paths.add(match)

    if timestamps:
        parsed = []
        for value in timestamps:
            try:
                parsed.append(_as_utc(value))
            except ValueError:
                pass
        if parsed:
            start = min(parsed)
            end = max(parsed)
            profile.start_time = start.isoformat().replace("+00:00", "Z")
            profile.last_entry_time = end.isoformat().replace("+00:00", "Z")
            profile.duration_seconds = max(0.0, (end - start).total_seconds())

    profile.work_ids = sorted(work_ids)
    profile.work_package_ids = sorted(wp_ids)
    profile.commits = sorted(commits)
    profile.files_changed = len(changed_paths) if changed_paths else None
    chosen_prompt: str | None = None
    for prompt in user_prompts:
        if prompt.strip().lower() not in _GENERIC_PROMPTS and len(prompt.strip()) >= 8:
            chosen_prompt = prompt
            break
    if chosen_prompt is None and user_prompts:
        chosen_prompt = user_prompts[0]

    if chosen_prompt:
        profile.subject = _subject_from_user_prompt(chosen_prompt)
        profile.subject_source = "first_substantive_operator_prompt"
    else:
        profile.subject = None
        profile.subject_source = None

    if explicit_thread_name:
        profile.thread_name = explicit_thread_name
        profile.thread_name_source = "explicit_/rename"
    else:
        profile.thread_name = profile.subject
        profile.thread_name_source = profile.subject_source or "heuristic"

    profile.purpose = _infer_purpose(
        profile.thread_name or profile.subject,
        profile.work_ids,
        profile.work_package_ids,
    )
    profile.outcome = _infer_outcome(texts)
    profile.context_mode = _infer_context_mode(event_count, chosen_prompt)
    profile.workflow_generation = _infer_workflow_generation(texts)
    return profile


def load_session_index_names(codex_home: Path) -> dict[str, str]:
    """Load the latest persisted Codex `/rename` name for each thread ID."""

    path = codex_home / "session_index.jsonl"
    names: dict[str, str] = {}
    if not path.is_file():
        return names

    id_keys = (
        "thread_id", "threadId", "session_id", "sessionId", "id",
        "conversation_id", "conversationId",
    )
    name_keys = (
        "thread_name", "threadName", "name", "title",
        "session_name", "sessionName", "conversation_name", "conversationName",
    )

    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return names

    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            thread_id = next(
                (str(row[k]).strip() for k in id_keys if row.get(k) is not None and str(row[k]).strip()),
                None,
            )
            name = next(
                (str(row[k]).strip() for k in name_keys if isinstance(row.get(k), str) and str(row[k]).strip()),
                None,
            )
            if not thread_id or not name:
                continue

            normalized = _thread_uid(thread_id)
            if re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                normalized,
                re.IGNORECASE,
            ):
                # session_index.jsonl is append-only, so later entries win.
                names[normalized.lower()] = _sanitize_subject(name, limit=120)

    return names


def _intish(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _usage_mapping(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "cached_input_tokens": ("cached_input_tokens", "cachedInputTokens", "cache_read_input_tokens", "cacheReadInputTokens"),
        "cache_write_input_tokens": ("cache_write_input_tokens", "cacheWriteInputTokens", "cache_creation_tokens", "cacheCreationTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "reasoning_output_tokens": ("reasoning_output_tokens", "reasoningOutputTokens"),
        "total_tokens": ("total_tokens", "totalTokens"),
    }
    result: dict[str, int] = {}
    for canonical, keys in aliases.items():
        for key in keys:
            if key in value:
                parsed = _intish(value.get(key))
                if parsed is not None:
                    result[canonical] = parsed
                    break
    if not result:
        return None
    result.setdefault("cached_input_tokens", 0)
    result.setdefault("cache_write_input_tokens", 0)
    result.setdefault("output_tokens", 0)
    result.setdefault("reasoning_output_tokens", 0)
    if "input_tokens" not in result and "total_tokens" in result:
        result["input_tokens"] = max(0, result["total_tokens"] - result.get("output_tokens", 0))
    if "total_tokens" not in result and "input_tokens" in result:
        result["total_tokens"] = result["input_tokens"] + result.get("output_tokens", 0)
    return result


def _find_usage_object(value: Any, keys: Sequence[str]) -> dict[str, int] | None:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                normalized = _usage_mapping(value.get(key))
                if normalized:
                    return normalized
        for child in value.values():
            found = _find_usage_object(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_usage_object(child, keys)
            if found:
                return found
    return None


def _find_int_key(value: Any, keys: Sequence[str]) -> int | None:
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                parsed = _intish(value.get(key))
                if parsed is not None:
                    return parsed
        for child in value.values():
            found = _find_int_key(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_int_key(child, keys)
            if found is not None:
                return found
    return None


def _token_count_payload(event: Mapping[str, Any]) -> tuple[dict[str, int] | None, dict[str, int] | None, int | None]:
    payload = event.get("payload")
    payload_type = str(payload.get("type") or "").lower() if isinstance(payload, Mapping) else ""
    if payload_type != "token_count" and "token_count" not in _event_kind(event):
        return None, None, None
    root: Any = payload if isinstance(payload, Mapping) else event
    last_usage = _find_usage_object(root, ("last_token_usage", "lastTokenUsage", "last_usage", "lastUsage"))
    cumulative = _find_usage_object(root, ("total_token_usage", "totalTokenUsage", "total_usage", "totalUsage"))
    context_window = _find_int_key(root, ("model_context_window", "modelContextWindow", "context_window", "contextWindow"))
    return last_usage, cumulative, context_window


def _estimate_text_tokens_from_chars(chars: int) -> int:
    return 0 if chars <= 0 else max(1, int(round(chars / 4.0)))


def _redact_command(command: str, *, limit: int = 100) -> str:
    value = re.sub(r"\s+", " ", command).strip()
    value = re.sub(r"/home/[^/\s]+", "~", value)
    value = re.sub(r"(?i)\b(sk-[A-Za-z0-9_-]{8,}|(?:api[_-]?key|password|token)\s*[:=]\s*\S+)", "[redacted]", value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _command_from_event(event: Mapping[str, Any]) -> tuple[str | None, str | None]:
    payload = event.get("payload")
    candidates: list[Mapping[str, Any]] = [event]
    if isinstance(payload, Mapping):
        candidates.append(payload)
    function_name = None
    arguments: Any = None
    for obj in candidates:
        for key in ("name", "function_name", "functionName"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                function_name = value.strip()
                break
        for key in ("arguments", "args", "input"):
            if key in obj:
                arguments = obj.get(key)
                break
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping):
            arguments = decoded
    command = None
    if isinstance(arguments, Mapping):
        for key in ("cmd", "command", "query", "path", "pattern"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                command = value.strip()
                break
    elif isinstance(arguments, str) and arguments.strip():
        command = arguments.strip()
    return function_name, command


def _command_category(function_name: str | None, command: str | None) -> str:
    fn = (function_name or "").lower()
    cmd = (command or "").strip().lower()
    if fn in {"apply_patch", "patch"}: return "patch"
    if "search" in fn or fn in {"grep", "rg"}: return "search"
    if fn and fn not in {"exec_command", "shell", "command"}: return f"tool:{fn}"
    if not cmd: return "other"
    if re.search(r"(^|\s)(pytest|python\s+-m\s+pytest|nox|tox)\b", cmd): return "test"
    if re.search(r"(^|\s)git\b", cmd): return "git"
    if re.search(r"(^|\s)(rg|grep|find)\b", cmd): return "search"
    if re.search(r"(^|\s)(sed|cat|head|tail|less|awk)\b", cmd): return "read"
    if "tools.eiel" in cmd or re.search(r"\beiel\b", cmd): return "eiel"
    if re.search(r"\b(py_compile|compileall|ruff|mypy|flake8)\b", cmd): return "compile/lint"
    return "shell"


def _tool_output_chars(event: Mapping[str, Any]) -> int:
    kind = _event_kind(event)
    if not any(token in kind for token in ("function_call_output", "tool_call_output", "custom_tool_call_output", "command_execution_output", "tool_output")):
        return 0
    payload = event.get("payload")
    best = ""
    for root in (payload, event):
        if isinstance(root, Mapping):
            for key in ("output", "content", "text", "result"):
                value = root.get(key)
                if isinstance(value, str) and len(value) > len(best):
                    best = value
    return len(best)


def _activity_label(categories: Sequence[str], tool_calls: int, output_chars: int, max_output_chars: int, last_category: str | None, last_command: str | None, compactions: int, users: int) -> str:
    parts: list[str] = []
    if compactions: parts.append(f"compaction×{compactions}")
    if users: parts.append(f"user×{users}")
    if tool_calls:
        unique = list(dict.fromkeys(categories))[:3]
        parts.append(f"{','.join(unique) or 'tool'}×{tool_calls}")
    if output_chars: parts.append(f"tool-out≈{_human_tokens(_estimate_text_tokens_from_chars(output_chars))}tok")
    if max_output_chars: parts.append(f"max-out≈{_human_tokens(_estimate_text_tokens_from_chars(max_output_chars))}tok")
    if last_category and last_command: parts.append(f"last={last_category}:{_redact_command(last_command, limit=60)}")
    return "; ".join(parts) if parts else "model/assistant continuation"


def _build_forensic_summary(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not calls: return {}
    contexts = [int(c.get("context_tokens") or 0) for c in calls]
    fresh = [int(c.get("fresh_input_tokens") or 0) for c in calls]
    cached = [int(c.get("cached_input_tokens") or 0) for c in calls]
    growth = [int(c.get("context_growth_tokens") or 0) for c in calls]
    jumps = sorted(({
        "call": int(c.get("call") or 0), "timestamp": c.get("timestamp"),
        "jump_tokens": int(c.get("context_growth_tokens") or 0),
        "context_tokens": int(c.get("context_tokens") or 0),
        "preceding_activity": c.get("preceding_activity"),
    } for c in calls if int(c.get("context_growth_tokens") or 0) > 0), key=lambda x: x["jump_tokens"], reverse=True)[:8]
    return {
        "model_calls": len(calls), "starting_context_tokens": contexts[0], "final_context_tokens": contexts[-1],
        "max_context_tokens": max(contexts), "median_context_tokens": _median_int(contexts),
        "mean_context_tokens": int(round(sum(contexts)/len(contexts))),
        "mean_fresh_input_tokens_per_call": int(round(sum(fresh)/len(fresh))),
        "mean_cached_input_tokens_per_call": int(round(sum(cached)/len(cached))),
        "context_growth_from_first_to_final": contexts[-1]-contexts[0],
        "max_single_context_jump": max(growth), "context_shrink_events": sum(1 for g in growth if g < 0),
        "top_context_jumps": jumps,
    }


def _attach_forensics(profile: RolloutProfile, path: Path) -> None:
    calls: list[dict[str, Any]] = []
    prev_context: int | None = None
    prev_cumulative: dict[str, int] | None = None
    context_window: int | None = None
    categories: list[str] = []
    tool_calls = output_chars = max_output_chars = compactions = users = 0
    last_category = last_command = None
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        profile.model_calls = []
        profile.forensic_summary = {}
        return
    with handle:
        for line in handle:
            if not line.strip(): continue
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if not isinstance(event, dict): continue
            timestamp = _event_timestamp(event)
            kind = _event_kind(event)
            fn, command = _command_from_event(event)
            if ("function_call" in kind or "tool_call" in kind or "custom_tool_call" in kind) and "output" not in kind:
                cat = _command_category(fn, command)
                categories.append(cat); tool_calls += 1; last_category = cat
                if command: last_command = command
            chars = _tool_output_chars(event)
            if chars:
                output_chars += chars; max_output_chars = max(max_output_chars, chars)
            if "context_compact" in kind or "context compacted" in kind: compactions += 1
            if _structural_user_texts(event): users += 1
            last_usage, cumulative, window = _token_count_payload(event)
            if window is not None: context_window = window
            if last_usage is None and cumulative is not None and prev_cumulative is not None:
                last_usage = {k: max(0, int(cumulative.get(k,0))-int(prev_cumulative.get(k,0))) for k in ("input_tokens","cached_input_tokens","cache_write_input_tokens","output_tokens","reasoning_output_tokens","total_tokens")}
            if cumulative is not None: prev_cumulative = dict(cumulative)
            if last_usage is None: continue
            context = int(last_usage.get("input_tokens",0)); cached = min(context,int(last_usage.get("cached_input_tokens",0)))
            growth = 0 if prev_context is None else context-prev_context
            calls.append({
                "call": len(calls)+1, "timestamp": timestamp, "context_tokens": context,
                "context_growth_tokens": growth, "cached_input_tokens": cached,
                "fresh_input_tokens": max(0,context-cached), "output_tokens": int(last_usage.get("output_tokens",0)),
                "reasoning_output_tokens": int(last_usage.get("reasoning_output_tokens",0)),
                "context_window": window or context_window,
                "preceding_activity": _activity_label(categories,tool_calls,output_chars,max_output_chars,last_category,last_command,compactions,users),
            })
            prev_context = context
            categories=[]; tool_calls=output_chars=max_output_chars=compactions=users=0; last_category=last_command=None
    profile.model_calls = calls
    profile.forensic_summary = _build_forensic_summary(calls)
    profile.model_context_window = context_window


def _render_forensic_summary(profile: RolloutProfile) -> list[str]:
    s = getattr(profile, "forensic_summary", {}) or {}
    if not s: return []
    lines = ["### Context-efficiency diagnosis", "",
        f"- Model calls observed: {s.get('model_calls',0)}",
        f"- Starting context: {int(s.get('starting_context_tokens') or 0):,}",
        f"- Median context/call: {int(s.get('median_context_tokens') or 0):,}",
        f"- Mean context/call: {int(s.get('mean_context_tokens') or 0):,}",
        f"- Maximum context: {int(s.get('max_context_tokens') or 0):,}",
        f"- Final context: {int(s.get('final_context_tokens') or 0):,}",
        f"- First→final context growth: {int(s.get('context_growth_from_first_to_final') or 0):+,}",
        f"- Mean fresh input/call: {int(s.get('mean_fresh_input_tokens_per_call') or 0):,}",
        f"- Mean cached input/call: {int(s.get('mean_cached_input_tokens_per_call') or 0):,}",
        f"- Maximum single context jump: {int(s.get('max_single_context_jump') or 0):+,}",
        f"- Context shrink events: {int(s.get('context_shrink_events') or 0):,}", ""]
    jumps=s.get("top_context_jumps") or []
    if jumps:
        lines += ["Largest context jumps:"]
        for j in jumps[:5]: lines.append(f"- call {j.get('call')}: {int(j.get('jump_tokens') or 0):+,} → {int(j.get('context_tokens') or 0):,}; {j.get('preceding_activity') or 'unknown'}")
        lines.append("")
    return lines


def _render_forensic_call_table(profile: RolloutProfile, maximum: int) -> list[str]:
    calls=list(getattr(profile,"model_calls",[]) or [])
    if not calls: return []
    sampled=_evenly_sample(calls,maximum)
    lines=["### Model-call context forensics","","`Context` is the actual model input for that call; `Fresh = Context - cached`. Tool-output token sizes are rough 4-char/token diagnostic estimates, not accounting truth.","",
        "| Call | Vienna | Context | ΔContext | Cache | Cache% | Fresh | Output | Window% | Activity before call |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in sampled:
        context=int(c.get("context_tokens") or 0); cached=int(c.get("cached_input_tokens") or 0); window=int(c.get("context_window") or 0)
        lines.append("| "+" | ".join([str(int(c.get("call") or 0)),_markdown_cell(_display_timestamp(c.get("timestamp"))),f"{context:,}",f"{int(c.get('context_growth_tokens') or 0):+,}",f"{cached:,}",f"{_pct(cached,context):.1f}%",f"{int(c.get('fresh_input_tokens') or 0):,}",f"{int(c.get('output_tokens') or 0):,}",(f"{_pct(context,window):.1f}%" if window else "unknown"),_markdown_cell(c.get("preceding_activity") or "unknown")])+" |")
    lines.append(""); return lines


def render_live_forensic_panel(rows: Sequence[dict[str, Any]], maximum_sessions: int = 5) -> list[str]:
    candidates=[]
    for row in rows:
        p=row.get("_profile")
        if isinstance(p,RolloutProfile) and getattr(p,"forensic_summary",None): candidates.append((str(row.get("lastActivity") or ""),row,p))
    if not candidates: return []
    lines=["","CONTEXT FORENSICS (most recent sessions)",f"{'UID':8} {'CALLS':>5} {'START':>9} {'MEDIAN':>9} {'MAX':>9} {'FINAL':>9} {'GROWTH':>10} {'FRESH/C':>9} {'CACHE/C':>9}","-"*94]
    for _,row,p in sorted(candidates,key=lambda x:x[0],reverse=True)[:maximum_sessions]:
        s=p.forensic_summary; sid=str(row.get("sessionId") or row.get("session") or "unknown-session")
        lines.append(f"{_thread_uid(sid,short=True):8} {int(s.get('model_calls') or 0):5d} {_human_tokens(int(s.get('starting_context_tokens') or 0)):>9} {_human_tokens(int(s.get('median_context_tokens') or 0)):>9} {_human_tokens(int(s.get('max_context_tokens') or 0)):>9} {_human_tokens(int(s.get('final_context_tokens') or 0)):>9} {int(s.get('context_growth_from_first_to_final') or 0):+10,d} {_human_tokens(int(s.get('mean_fresh_input_tokens_per_call') or 0)):>9} {_human_tokens(int(s.get('mean_cached_input_tokens_per_call') or 0)):>9}")
    return lines


def enrich_rows(
    rows: Sequence[dict[str, Any]],
    *,
    codex_home: Path,
    include_rollout_metadata: bool = True,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    persisted_names = load_session_index_names(codex_home)

    for raw in rows:
        row = dict(raw)
        session_id = str(row.get("sessionId") or row.get("session") or "")
        profile = RolloutProfile()

        if include_rollout_metadata and session_id:
            rollout = find_rollout(session_id, codex_home)
            if rollout:
                profile = profile_rollout(rollout)
                _attach_forensics(profile, rollout)

        # Codex /rename persistence wins over every heuristic.
        full_uid = _thread_uid(session_id).lower() if session_id else ""
        persisted_name = persisted_names.get(full_uid)
        if persisted_name:
            profile.thread_name = persisted_name
            profile.thread_name_source = "session_index_/rename"
        elif not profile.thread_name and profile.subject:
            profile.thread_name = profile.subject
            profile.thread_name_source = profile.subject_source or "heuristic"

        row["_profile"] = profile
        enriched.append(row)

    return enriched


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _pct(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else (100.0 * numerator / denominator)


def _human_tokens(value: int) -> str:
    """Format a token count for quick human scanning."""

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,}"


def _median_int(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return int(round((ordered[middle - 1] + ordered[middle]) / 2))


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_executive_summary(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Render the same human-facing columns as the live dashboard."""

    total = sum(_counter(row, "totalTokens") for row in rows)
    cache = sum(_counter(row, "cacheReadTokens") for row in rows)
    output = sum(_counter(row, "outputTokens") for row in rows)

    lines = [
        "## Human view",
        "",
        (
            f"**Vienna | sessions {len(rows)} | total {_human_tokens(total)} | "
            f"cache {_human_tokens(cache)} ({_pct(cache, total):.1f}%) | "
            f"output {_human_tokens(output)}**"
        ),
        "",
        "| UID | Last Vienna | Model | Total | Cache% | In | Out | Calls | Duration | Thread / Subject |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        session_id = str(row.get("sessionId") or row.get("session") or "unknown-session")
        profile = row.get("_profile")
        if not isinstance(profile, RolloutProfile):
            profile = RolloutProfile()

        session_total = _counter(row, "totalTokens")
        cache_read = _counter(row, "cacheReadTokens")
        model = str(profile.model or row.get("model") or "unknown")
        last = _display_timestamp(str(row.get("lastActivity") or ""), seconds=True)
        thread = profile.thread_name or profile.subject or "unknown"

        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(_thread_uid(session_id, short=True)),
                    _markdown_cell(last),
                    _markdown_cell(model),
                    f"{session_total:,}",
                    f"{_pct(cache_read, session_total):.1f}%",
                    f"{_counter(row, 'inputTokens'):,}",
                    f"{_counter(row, 'outputTokens'):,}",
                    f"{profile.tool_calls:,}",
                    _markdown_cell(_fmt_duration(profile.duration_seconds)),
                    _markdown_cell(thread),
                ]
            )
            + " |"
        )

    lines.append("")
    return lines

def _history_record(row: Mapping[str, Any], *, sampled_at: datetime) -> dict[str, Any]:
    """Return one compact, redacted cumulative snapshot for progression analysis."""

    session_id = str(row.get("sessionId") or row.get("session") or "unknown-session")
    profile = row.get("_profile")
    if not isinstance(profile, RolloutProfile):
        profile = RolloutProfile()

    return {
        "sampled_at": sampled_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
        "uid": _thread_uid(session_id),
        "short_uid": _thread_uid(session_id, short=True),
        "last_activity": str(row.get("lastActivity") or ""),
        "model": str(profile.model or row.get("model") or "unknown"),
        "input_tokens": _counter(row, "inputTokens"),
        "cache_read_tokens": _counter(row, "cacheReadTokens"),
        "cache_creation_tokens": _counter(row, "cacheCreationTokens"),
        "output_tokens": _counter(row, "outputTokens"),
        "reasoning_output_tokens": _counter(row, "reasoningOutputTokens"),
        "total_tokens": _counter(row, "totalTokens"),
        "tool_calls": int(profile.tool_calls or 0),
        "compactions": int(profile.compactions or 0),
        "duration_seconds": profile.duration_seconds,
        "thread_name": profile.thread_name or profile.subject or "Unrenamed thread",
    }


def _history_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Fields whose change makes a new progression point scientifically useful."""

    return (
        record.get("last_activity"),
        record.get("model"),
        record.get("input_tokens"),
        record.get("cache_read_tokens"),
        record.get("cache_creation_tokens"),
        record.get("output_tokens"),
        record.get("reasoning_output_tokens"),
        record.get("total_tokens"),
        record.get("tool_calls"),
        record.get("compactions"),
        record.get("thread_name"),
    )


def load_history(path: Path | None, *, cutoff: datetime | None = None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []

    records: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return records

    with handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if cutoff is not None:
                try:
                    observed = _as_utc(str(row.get("sampled_at") or ""))
                except ValueError:
                    continue
                if observed < cutoff:
                    continue
            records.append(row)
    return records


def update_history(
    history: list[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    sampled_at: datetime,
    path: Path | None,
) -> int:
    """Append change-only session snapshots.

    A 2-second live refresh does *not* create a row when cumulative counters and
    identifying metadata are unchanged. This preserves the useful progression
    signal without recording thousands of idle duplicates.
    """

    latest_by_uid: dict[str, Mapping[str, Any]] = {}
    for record in history:
        uid = str(record.get("uid") or "")
        if uid:
            latest_by_uid[uid] = record

    additions: list[dict[str, Any]] = []
    for row in rows:
        record = _history_record(row, sampled_at=sampled_at)
        previous = latest_by_uid.get(record["uid"])
        if previous is not None and _history_signature(previous) == _history_signature(record):
            continue
        additions.append(record)
        latest_by_uid[record["uid"]] = record

    if not additions:
        return 0

    history.extend(additions)

    if path is not None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                for record in additions:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as exc:
            raise CodexSessionReportError(f"could not append history file {path}: {exc}") from exc

    return len(additions)


def _evenly_sample(records: Sequence[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    """Keep endpoints and evenly spaced interior points for a compact report."""

    values = list(records)
    if maximum <= 0 or len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[-1]]

    indices = {
        round(i * (len(values) - 1) / (maximum - 1))
        for i in range(maximum)
    }
    return [values[index] for index in sorted(indices)]


def render_progression_history(
    history: Sequence[dict[str, Any]],
    *,
    cutoff: datetime,
    max_points_per_session: int,
) -> list[str]:
    """Render cumulative thread progression with useful deltas."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in history:
        try:
            sampled = _as_utc(str(record.get("sampled_at") or ""))
        except ValueError:
            continue
        if sampled < cutoff:
            continue
        uid = str(record.get("uid") or "")
        if uid:
            grouped.setdefault(uid, []).append(record)

    if not grouped:
        return []

    for records in grouped.values():
        records.sort(key=lambda r: str(r.get("sampled_at") or ""))

    lines = [
        "## Thread progression",
        "",
        (
            "Change-only cumulative snapshots captured by live mode. "
            "The sidecar history retains every changed snapshot; this table "
            f"shows up to {max_points_per_session} evenly spaced points per thread."
        ),
        "",
        "| UID | Snapshot Vienna | Total | ΔTotal | Cache | ΔCache | Cache% | Input | ΔInput | Output | ΔOutput | Calls | ΔCalls | Duration | Thread |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    # Most recently active thread first.
    ordered_groups = sorted(
        grouped.items(),
        key=lambda pair: str(pair[1][-1].get("sampled_at") or ""),
        reverse=True,
    )

    for uid, all_records in ordered_groups:
        sampled_records = _evenly_sample(all_records, max_points_per_session)

        for record in sampled_records:
            # Deltas are against the immediately preceding *stored* point, not
            # merely the previous displayed/evenly-sampled point.
            position = all_records.index(record)
            previous = all_records[position - 1] if position > 0 else None

            def delta(key: str) -> str:
                if previous is None:
                    return "—"
                return f"{int(record.get(key) or 0) - int(previous.get(key) or 0):+,}"

            total = int(record.get("total_tokens") or 0)
            cache = int(record.get("cache_read_tokens") or 0)
            duration = record.get("duration_seconds")
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(str(record.get("short_uid") or _thread_uid(uid, short=True))),
                        _markdown_cell(_display_timestamp(str(record.get("sampled_at") or ""))),
                        f"{total:,}",
                        delta("total_tokens"),
                        f"{cache:,}",
                        delta("cache_read_tokens"),
                        f"{_pct(cache, total):.1f}%",
                        f"{int(record.get('input_tokens') or 0):,}",
                        delta("input_tokens"),
                        f"{int(record.get('output_tokens') or 0):,}",
                        delta("output_tokens"),
                        f"{int(record.get('tool_calls') or 0):,}",
                        delta("tool_calls"),
                        _markdown_cell(_fmt_duration(float(duration)) if duration is not None else "unknown"),
                        _markdown_cell(str(record.get("thread_name") or "Unrenamed thread")),
                    ]
                )
                + " |"
            )

    lines.append("")
    return lines


def render_report(
    rows: Sequence[dict[str, Any]],
    *,
    generated_at: datetime,
    cutoff: datetime,
    history: Sequence[dict[str, Any]] = (),
    history_points_per_session: int = 20,
    include_forensics: bool = False,
    forensic_points_per_session: int = 24,
) -> str:
    """Render aggregate counters plus redacted/best-effort metadata to Markdown."""

    total = sum(_counter(row, "totalTokens") for row in rows)
    cache = sum(_counter(row, "cacheReadTokens") for row in rows)
    input_tokens = sum(_counter(row, "inputTokens") for row in rows)
    output_tokens = sum(_counter(row, "outputTokens") for row in rows)
    lines = [
        f"# Codex sessions: last {(generated_at - cutoff).total_seconds() / 3600:g} hours",
        "",
    ]
    lines.extend(render_executive_summary(rows))
    progression = render_progression_history(
        history,
        cutoff=cutoff,
        max_points_per_session=history_points_per_session,
    )
    if progression:
        lines.extend(progression)
    lines.extend(
        [
            "## Report details",
            "",
            f"- Generated ({DISPLAY_TZ_NAME}): {_display_timestamp(generated_at)}",
            f"- Cutoff ({DISPLAY_TZ_NAME}): {_display_timestamp(cutoff)}",
            f"- Sessions: {len(rows)}",
            f"- Aggregate total: {total:,}",
            f"- Aggregate input: {input_tokens:,}",
            f"- Aggregate cache-read: {cache:,} ({_pct(cache, total):.2f}% of total)",
            f"- Aggregate output: {output_tokens:,}",
            "",
            "> Metadata derived from Codex rollout JSONL is heuristic and descriptive only. "
            "Raw transcript content is not included.",
            "",
        ]
    )
    for row in rows:
        session_id = str(row.get("sessionId") or row.get("session") or "unknown-session")
        last_activity = str(row.get("lastActivity") or "unknown")
        profile = row.get("_profile")
        if not isinstance(profile, RolloutProfile):
            profile = RolloutProfile()
        session_total = _counter(row, "totalTokens")
        cache_read = _counter(row, "cacheReadTokens")
        lines.extend(
            [
                f"## `{session_id}`",
                "",
                f"- Thread UID: `{_thread_uid(session_id)}`",
                f"- Short UID: `{_thread_uid(session_id, short=True)}`",
                f"- Thread name: {profile.thread_name or profile.subject or 'unknown'}",
                f"- Thread name source: {profile.thread_name_source or 'unavailable'}",
                f"- Heuristic subject: {profile.subject or 'unknown'}",
                f"- Subject source: {profile.subject_source or 'unavailable'}",
                f"- Purpose: {profile.purpose or 'unknown'}",
                f"- Model: {profile.model or row.get('model') or row.get('models') or 'unknown'}",
                f"- Start time ({DISPLAY_TZ_NAME}): {_display_timestamp(profile.start_time)}",
                f"- Last rollout entry ({DISPLAY_TZ_NAME}): {_display_timestamp(profile.last_entry_time)}",
                f"- Last ccusage activity ({DISPLAY_TZ_NAME}): {_display_timestamp(last_activity)}",
                f"- Duration: {_fmt_duration(profile.duration_seconds)}",
                f"- Work IDs: {', '.join(profile.work_ids) if profile.work_ids else 'none detected'}",
                f"- Work Package IDs: {', '.join(profile.work_package_ids) if profile.work_package_ids else 'none detected'}",
                f"- Purpose/outcome: {profile.purpose or 'unknown'} / {profile.outcome or 'unknown'}",
                f"- Workflow generation: {profile.workflow_generation or 'unknown'}",
                f"- Context mode: {profile.context_mode or 'unknown'}",
                f"- Tool calls (heuristic): {profile.tool_calls}",
                f"- Exec calls (heuristic): {profile.exec_calls}",
                f"- Wait calls (heuristic): {profile.wait_calls}",
                f"- Context compactions (heuristic): {profile.compactions}",
                f"- Files changed (heuristic): {profile.files_changed if profile.files_changed is not None else 'unknown'}",
                f"- Commits detected: {', '.join(profile.commits[:8]) if profile.commits else 'none detected'}",
                f"- Input: {_counter(row, 'inputTokens'):,}",
                f"- Cache-read: {cache_read:,} ({_pct(cache_read, session_total):.2f}%)",
                f"- Cache-creation: {_counter(row, 'cacheCreationTokens'):,}",
                f"- Output: {_counter(row, 'outputTokens'):,}",
                f"- Reasoning output: {_counter(row, 'reasoningOutputTokens'):,}",
                f"- Total: {session_total:,}",
                (
                    f"- Tokens/minute: {session_total / (profile.duration_seconds / 60):,.0f}"
                    if profile.duration_seconds and profile.duration_seconds > 0
                    else "- Tokens/minute: unknown"
                ),
                (
                    f"- Tokens/tool call: {session_total / profile.tool_calls:,.0f}"
                    if profile.tool_calls > 0
                    else "- Tokens/tool call: unknown"
                ),
                f"- Metadata source: {profile.metadata_source}",
                "",
            ]
        )
        if include_forensics:
            lines.extend(_render_forensic_summary(profile))
            lines.extend(_render_forensic_call_table(profile, forensic_points_per_session))
    return "\n".join(lines)


def _thread_uid(session_id: str, *, short: bool = False) -> str:
    """Return the stable rollout UUID portion of a Codex session identifier.

    Example:
      2026/08/08/rollout-...-019fe05a-8604-7503-bda5-d493a50a9729
      -> 019fe05a-8604-7503-bda5-d493a50a9729

    The short form uses the final eight characters for compact live display.
    """

    leaf = session_id.rsplit("/", 1)[-1]
    match = re.search(
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$",
        leaf,
    )
    uid = match.group(1).lower() if match else leaf
    if short and len(uid) > 8:
        return uid[-8:]
    return uid


def _short_subject(profile: RolloutProfile, width: int) -> str:
    value = profile.thread_name or profile.subject or "Unrenamed thread"
    if len(value) <= width:
        return value
    return value[: max(1, width - 1)] + "…"


def render_live(rows: Sequence[dict[str, Any]], *, generated_at: datetime, cutoff: datetime, include_forensics: bool = False) -> str:
    """Render a compact dashboard suitable for in-place terminal refresh."""

    width = shutil.get_terminal_size((140, 30)).columns
    displayed_rows = rows[:LIVE_MAX_SESSION_ROWS]
    total = sum(_counter(row, "totalTokens") for row in rows)
    cache = sum(_counter(row, "cacheReadTokens") for row in rows)
    output = sum(_counter(row, "outputTokens") for row in rows)
    lines = [
        "Codex session token monitor",
        f"Vienna {_as_display_tz(generated_at).strftime('%Y-%m-%d %H:%M:%S %Z')} | "
        f"window {(generated_at-cutoff).total_seconds()/3600:g}h | "
        f"sessions {len(rows)} | showing {len(displayed_rows)} | total {total:,} | cache {cache:,} ({_pct(cache,total):.1f}%) | output {output:,}",
        "",
    ]
    if width >= 120:
        lines.append(
            f"{'UID':8} {'LAST VIENNA':19} {'MODEL':18} {'TOTAL':>12} {'CACHE%':>7} {'IN':>10} {'OUT':>9} "
            f"{'CALLS':>6} {'DURATION':>10}  THREAD / SUBJECT"
        )
        lines.append("-" * min(width, 180))
        subject_width = max(18, width - 118)
        for row in displayed_rows:
            profile = row.get("_profile")
            if not isinstance(profile, RolloutProfile):
                profile = RolloutProfile()
            total_tokens = _counter(row, "totalTokens")
            cache_tokens = _counter(row, "cacheReadTokens")
            raw_last = row.get("lastActivity")
            try:
                last = _as_display_tz(str(raw_last)).strftime("%Y-%m-%d %H:%M:%S") if raw_last else "unknown"
            except ValueError:
                last = str(raw_last or "unknown")[:19]
            model = str(profile.model or row.get("model") or "unknown")[:18]
            session_id = str(row.get("sessionId") or row.get("session") or "unknown-session")
            uid = _thread_uid(session_id, short=True)
            lines.append(
                f"{uid:8} {last:19} {model:18} {total_tokens:12,} {_pct(cache_tokens,total_tokens):6.1f}% "
                f"{_counter(row,'inputTokens'):10,} {_counter(row,'outputTokens'):9,} "
                f"{profile.tool_calls:6d} {_fmt_duration(profile.duration_seconds):>10}  "
                f"{_short_subject(profile, subject_width)}"
            )
    else:
        for row in displayed_rows:
            profile = row.get("_profile")
            if not isinstance(profile, RolloutProfile):
                profile = RolloutProfile()
            total_tokens = _counter(row, "totalTokens")
            cache_tokens = _counter(row, "cacheReadTokens")
            session_id = str(row.get("sessionId") or row.get("session") or "unknown-session")
            uid = _thread_uid(session_id, short=True)
            lines.append(
                f"{uid} {_display_timestamp(str(row.get('lastActivity') or ''), seconds=True)[:19]}  "
                f"{total_tokens:,}  cache {_pct(cache_tokens,total_tokens):.1f}%  "
                f"calls {profile.tool_calls}  {_short_subject(profile, max(20, width-64))}"
            )
    if include_forensics:
        lines.extend(render_live_forensic_panel(rows))
    lines.extend(
        [
            "",
            "Ctrl-C to exit. Live view is read-only; rollout metadata is best-effort and redacted.",
        ]
    )
    return "\n".join(lines)


def _write_output(path: Path | None, report: str) -> None:
    if path is None:
        return
    path.write_text(report, encoding="utf-8")


class _LiveScreen:
    """In-place live display.

    Normal-screen mode is the default because many terminal emulators translate
    mouse-wheel activity into cursor escape sequences while an application owns
    the alternate screen.  Those bytes can visibly corrupt a dashboard that
    intentionally does not consume stdin.

    --alt-screen remains available for terminals where that behaviour is known
    to be safe.
    """

    _MOUSE_OFF = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l"

    def __init__(self, *, alternate: bool):
        self.alternate = alternate and sys.stdout.isatty()
        self.tty = sys.stdout.isatty()

    def __enter__(self) -> "_LiveScreen":
        if self.tty:
            # Explicitly disable common terminal mouse-reporting modes.  The
            # dashboard is non-interactive and never needs mouse events.
            sys.stdout.write(self._MOUSE_OFF)
            if self.alternate:
                sys.stdout.write("\x1b[?1049h")
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
        return self

    def draw(self, text: str) -> None:
        # Home + clear gives a stable, non-scrolling dashboard without requiring
        # the alternate screen. Re-assert mouse-off because parent TUIs and some
        # terminals can change private modes asynchronously.
        if self.tty:
            sys.stdout.write(self._MOUSE_OFF)
        sys.stdout.write("\x1b[H\x1b[2J")
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.tty:
            sys.stdout.write(self._MOUSE_OFF)
            sys.stdout.write("\x1b[?25h")
            if self.alternate:
                sys.stdout.write("\x1b[?1049l")
            sys.stdout.flush()


def _collect(
    *,
    hours: float,
    ccusage_command: str,
    codex_home: Path,
    include_rollout_metadata: bool,
) -> tuple[datetime, datetime, list[dict[str, Any]]]:
    generated_at = datetime.now(timezone.utc)
    cutoff = generated_at - timedelta(hours=hours)
    rows = query_ccusage(cutoff=cutoff, executable=ccusage_command)
    return (
        generated_at,
        cutoff,
        enrich_rows(rows, codex_home=codex_home, include_rollout_metadata=include_rollout_metadata),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown report path. Required unless --live is used; parent must already exist.",
    )
    parser.add_argument("--hours", type=float, default=48.0, help="Exact lookback window in hours (default: 48); human timestamps display in Europe/Vienna.")
    parser.add_argument("--ccusage-command", default="ccusage", help="ccusage executable or wrapper (default: ccusage).")
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser(),
        help="Codex state root used for optional rollout metadata (default: $CODEX_HOME or ~/.codex).",
    )
    parser.add_argument(
        "--no-rollout-metadata",
        action="store_true",
        help="Use only ccusage counters; do not inspect local Codex rollout JSONL files.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refresh an in-place terminal dashboard continuously (default interval: 2s).",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        help=(
            "Append change-only live snapshots to this JSONL file. "
            "Default with --output: <output-stem>.history.jsonl."
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable progression-history capture and rendering.",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=20,
        help="Maximum evenly spaced progression points shown per thread in Markdown (default: 20).",
    )
    parser.add_argument(
        "--forensics", action="store_true",
        help="Enable per-model-call context forensics from Codex token_count events.",
    )
    parser.add_argument(
        "--forensic-points", type=int, default=24,
        help="Maximum forensic model-call rows shown per session in Markdown (default: 24).",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=2.0,
        help="Refresh interval for --live (default: 2.0 seconds; minimum: 0.5).",
    )
    parser.add_argument(
        "--alt-screen",
        action="store_true",
        help=(
            "With --live, use the terminal alternate screen. Normal-screen redraw "
            "is the default because it behaves better with mouse-wheel input."
        ),
    )
    parser.add_argument(
        "--no-alt-screen",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.refresh_seconds < 0.5:
        parser.error("--refresh-seconds must be at least 0.5")
    if args.history_points <= 0:
        parser.error("--history-points must be positive")
    if args.forensic_points <= 0:
        parser.error("--forensic-points must be positive")
    if not args.live and args.output is None:
        parser.error("--output is required unless --live is used")
    if args.output is not None and not args.output.parent.is_dir():
        parser.error("--output parent directory must already exist")

    include_rollout_metadata = not args.no_rollout_metadata

    history_path: Path | None = None
    if not args.no_history:
        if args.history_file is not None:
            history_path = args.history_file
        elif args.output is not None:
            history_path = args.output.with_name(args.output.stem + ".history.jsonl")

    if history_path is not None and not history_path.parent.is_dir():
        parser.error("--history-file parent directory must already exist")

    if not args.live:
        try:
            generated_at, cutoff, rows = _collect(
                hours=args.hours,
                ccusage_command=args.ccusage_command,
                codex_home=args.codex_home,
                include_rollout_metadata=include_rollout_metadata,
            )
            history = [] if args.no_history else load_history(history_path, cutoff=cutoff)
            report = render_report(
                rows,
                generated_at=generated_at,
                cutoff=cutoff,
                history=history,
                history_points_per_session=args.history_points,
                include_forensics=args.forensics,
                forensic_points_per_session=args.forensic_points,
            )
        except CodexSessionReportError as exc:
            parser.error(str(exc))
        _write_output(args.output, report)
        print(f"Wrote {args.output} ({len(rows)} sessions)")
        return 0

    # Live mode intentionally re-queries ccusage and rollout metadata. It never
    # writes actor/repository state. Token progression is stored only in the
    # optional sidecar JSONL and is append-only/change-only.
    live_history = [] if args.no_history else load_history(history_path)

    with _LiveScreen(alternate=(args.alt_screen and not args.no_alt_screen)) as screen:
        try:
            while True:
                try:
                    generated_at, cutoff, rows = _collect(
                        hours=args.hours,
                        ccusage_command=args.ccusage_command,
                        codex_home=args.codex_home,
                        include_rollout_metadata=include_rollout_metadata,
                    )
                    if not args.no_history:
                        update_history(
                            live_history,
                            rows,
                            sampled_at=generated_at,
                            path=history_path,
                        )

                    dashboard = render_live(rows, generated_at=generated_at, cutoff=cutoff, include_forensics=args.forensics)
                    if args.output is not None:
                        report = render_report(
                            rows,
                            generated_at=generated_at,
                            cutoff=cutoff,
                            history=live_history,
                            history_points_per_session=args.history_points,
                        )
                        tmp = args.output.with_name(args.output.name + ".tmp")
                        tmp.write_text(report, encoding="utf-8")
                        os.replace(tmp, args.output)
                except CodexSessionReportError as exc:
                    dashboard = f"Codex session token monitor\n\nERROR: {exc}\n\nRetrying in {args.refresh_seconds:g}s. Ctrl-C to exit."
                screen.draw(dashboard)
                time.sleep(args.refresh_seconds)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
