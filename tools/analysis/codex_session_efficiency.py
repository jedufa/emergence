"""Read-only Codex JSONL session efficiency analyzer.

The analyzer is intentionally narrow. It measures the context-growth signals
needed to evaluate EIEL Context Epochs and bounded-output policies. It does not
interpret engineering correctness, archive transcripts, or create authority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from engine.campaigns.canonical import pretty_json

SCHEMA_VERSION = "eiel_codex_session_efficiency_v1"
ORIGINAL_TOKEN_RE = re.compile(r"Original token count:\s*([0-9]+)")
AUTHORITY_NAMES = ("AGENTS.md", "ACTIVE_GOAL.md", "MEMORY.md", "CODEX_OPERATOR_GUIDE.md")


def _timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _command(payload: Mapping[str, Any]) -> str | None:
    if payload.get("type") != "function_call":
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, Mapping):
        value = parsed.get("cmd") or parsed.get("command")
        return str(value).strip() if value else None
    return None


def _category(command: str) -> str:
    text = command.strip()
    first = text.split(maxsplit=1)[0] if text else ""
    if re.search(r"(?:^|\s)(?:rg|grep|find)(?:\s|$)", text):
        return "search"
    if re.search(r"(?:^|\s)(?:cat|sed|head|tail|less)(?:\s|$)", text):
        return "read"
    if re.search(r"(?:^|\s)git(?:\s|$)", text):
        return "git"
    if re.search(r"(?:^|\s)(?:pytest|python\s+-m\s+pytest)(?:\s|$)", text):
        return "test"
    if re.search(r"(?:compileall|py_compile|ruff|mypy|diff --check)", text):
        return "compile_or_lint"
    if first in {"apply_patch", "patch"} or "apply_patch" in text:
        return "patch"
    return "other"


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def analyze_session(path: str | Path, *, label: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    event_types: Counter[str] = Counter()
    payload_types: Counter[str] = Counter()
    function_names: Counter[str] = Counter()
    command_counts: Counter[str] = Counter()
    command_categories: Counter[str] = Counter()
    authority_reads: Counter[str] = Counter()
    timestamps: list[dt.datetime] = []
    task_starts = 0
    compactions = 0
    patch_events = 0
    tool_calls = 0
    shell_output_tokens = 0
    shell_output_events = 0
    broad_output_events = 0
    cumulative = Counter()
    final_usage: dict[str, Any] = {}
    context_window = None
    first_edit_tool_index: int | None = None
    searches_before_first_edit = 0
    tool_index = 0
    metadata: dict[str, Any] = {}

    for row in _iter_rows(source):
        if row.get("timestamp"):
            timestamps.append(_timestamp(str(row["timestamp"])))
        event_type = str(row.get("type") or "unknown")
        event_types[event_type] += 1
        payload = dict(row.get("payload") or {})
        payload_type = str(payload.get("type") or "")
        if payload_type:
            payload_types[payload_type] += 1
        if event_type == "session_meta":
            metadata = {
                key: payload.get(key)
                for key in ("session_id", "timestamp", "cwd", "originator", "cli_version", "model_provider")
            }
        if payload_type == "task_started":
            task_starts += 1
            context_window = payload.get("model_context_window") or context_window
        # Codex emits both a top-level ``compacted`` row and an adjacent
        # ``context_compacted`` payload for one logical compaction. Count the
        # payload only so the metric is not doubled.
        if payload_type == "context_compacted":
            compactions += 1
        if payload_type == "patch_apply_end":
            patch_events += 1
            if first_edit_tool_index is None:
                first_edit_tool_index = tool_index
        if event_type == "response_item" and payload_type in {"function_call", "custom_tool_call"}:
            tool_calls += 1
            tool_index += 1
            name = str(payload.get("name") or payload.get("tool_name") or payload_type)
            function_names[name] += 1
            command = _command(payload)
            if command:
                command_counts[command] += 1
                category = _category(command)
                command_categories[category] += 1
                if first_edit_tool_index is None and category == "search":
                    searches_before_first_edit += 1
                if category == "patch" and first_edit_tool_index is None:
                    first_edit_tool_index = tool_index
                if category == "read":
                    for authority in AUTHORITY_NAMES:
                        if authority in command:
                            authority_reads[authority] += 1
        if event_type == "response_item" and payload_type in {"function_call_output", "custom_tool_call_output"}:
            shell_output_events += 1
            output = str(payload.get("output") or payload.get("content") or "")
            match = ORIGINAL_TOKEN_RE.search(output)
            if match:
                count = int(match.group(1))
                shell_output_tokens += count
                if count > 2000:
                    broad_output_events += 1
            elif output.count("\n") > 120:
                broad_output_events += 1
        if payload_type == "token_count":
            info = dict(payload.get("info") or {})
            last = dict(info.get("last_token_usage") or {})
            total = dict(info.get("total_token_usage") or {})
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "total_tokens",
            ):
                cumulative[key] += int(last.get(key) or 0)
            if total:
                final_usage = total
            context_window = info.get("model_context_window") or context_window

    duration_seconds = (
        max(0.0, (max(timestamps) - min(timestamps)).total_seconds()) if timestamps else 0.0
    )
    repeated_invocations = sum(count - 1 for count in command_counts.values() if count > 1)
    top_repeated = [
        {"command": command, "count": count}
        for command, count in command_counts.most_common(25)
        if count > 1
    ]
    denominator = task_starts or 1
    reported_input = int(final_usage.get("input_tokens") or cumulative["input_tokens"])
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label or source.stem,
        "source": str(source),
        "metadata": metadata,
        "duration_seconds": round(duration_seconds, 3),
        "duration_hours": round(duration_seconds / 3600.0, 4),
        "jsonl_events": sum(event_types.values()),
        "event_types": dict(event_types),
        "payload_types": dict(payload_types),
        "task_starts": task_starts,
        "tool_calls": tool_calls,
        "function_names": dict(function_names),
        "command_categories": dict(command_categories),
        "patch_events": patch_events,
        "context_compactions": compactions,
        "model_context_window": context_window,
        "summed_turn_processing": dict(cumulative),
        "cumulative_model_processing": dict(final_usage or cumulative),
        "reported_final_session_usage": final_usage,
        "shell_output_tokens": shell_output_tokens,
        "shell_output_events": shell_output_events,
        "broad_output_events": broad_output_events,
        "authority_reads": dict(authority_reads),
        "repeated_exact_command_invocations": repeated_invocations,
        "unique_commands": len(command_counts),
        "top_repeated_commands": top_repeated,
        "searches_before_first_edit": searches_before_first_edit,
        "per_task": {
            "tool_calls": round(tool_calls / denominator, 3),
            "shell_output_tokens": round(shell_output_tokens / denominator, 3),
            "cumulative_input_tokens": round(reported_input / denominator, 3),
        },
    }


def comparison_report(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "eiel_codex_session_efficiency_comparison_v1",
        "sessions": [dict(row) for row in sessions],
        "interpretation_boundary": (
            "Metrics describe context growth and command/output behavior only; they do not prove engineering quality or causality."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = list(report.get("sessions") or [])
    lines = [
        "# EIEL Codex Context-Efficiency Baseline",
        "",
        "This report is generated from supplied Codex JSONL evidence. It measures context-growth signals only.",
        "",
        "| Session | Hours | Tasks | Tool calls | Compactions | Cumulative input | Cached input | Shell output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        usage = dict(row.get("cumulative_model_processing") or {})
        lines.append(
            f"| {row.get('label')} | {row.get('duration_hours')} | {row.get('task_starts')} | "
            f"{row.get('tool_calls')} | {row.get('context_compactions')} | "
            f"{usage.get('input_tokens', 0)} | {usage.get('cached_input_tokens', 0)} | "
            f"{row.get('shell_output_tokens', 0)} |"
        )
    lines.extend(["", "## Session details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('label')}",
                "",
                f"- Source: `{row.get('source')}`",
                f"- Repeated exact command invocations: {row.get('repeated_exact_command_invocations')}",
                f"- Searches before first edit: {row.get('searches_before_first_edit')}",
                f"- Authority reads: `{json.dumps(row.get('authority_reads') or {}, sort_keys=True)}`",
                f"- Broad output events: {row.get('broad_output_events')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            str(report.get("interpretation_boundary") or ""),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True, help="Codex rollout JSONL path; repeatable")
    parser.add_argument("--label", action="append", default=[], help="Label corresponding to each --session")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = list(args.label)
    sessions = [
        analyze_session(path, label=labels[index] if index < len(labels) else None)
        for index, path in enumerate(args.session)
    ]
    report = comparison_report(sessions)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(pretty_json(report), encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
