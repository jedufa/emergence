"""Project discovery, source indexing, and bounded context preparation."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.runtime_roots import resolve_runtime_roots


CONFIG_VERSION = "emergence_dev_config_v1"
CONTEXT_VERSION = "emergence_dev_context_v1"
INDEX_VERSION = "emergence_dev_source_index_v1"
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{1,}")
_PYTHON_DEFINITION_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_OBJECTIVE_STOPWORDS = frozenset(
    {"add", "and", "change", "fix", "implement", "its", "repair", "test", "tests", "the", "update"}
)


class DevAutomationError(RuntimeError):
    """The requested development operation could not be performed safely."""


@dataclass(frozen=True, slots=True)
class DevConfig:
    root: Path
    path: Path
    raw: Mapping[str, Any]
    config_sha256: str

    @property
    def index(self) -> Mapping[str, Any]:
        return self.raw["index"]

    @property
    def limits(self) -> Mapping[str, int]:
        return self.raw["limits"]

    @property
    def relationships(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("relationships", ()))

    @property
    def critical_rules(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("critical_rules", ()))

    @property
    def validation_rules(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.raw.get("validation_rules", ()))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def find_project_root(start: str | Path | None = None) -> Path:
    candidate = Path.cwd() if start is None else Path(start)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if payload.get("project", {}).get("name") == "emergence":
            return directory
    raise DevAutomationError(f"cannot find the Emergence project root from {candidate}")


def _string_list(value: Any, *, field: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise DevAutomationError(f"{field} must be an array of non-empty strings")
    if not allow_empty and not value:
        raise DevAutomationError(f"{field} must not be empty")
    return list(value)


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise DevAutomationError(f"{field} must be an integer from {minimum} through {maximum}")
    return value


def load_config(root: str | Path, path: str | Path | None = None) -> DevConfig:
    project_root = Path(root).resolve()
    config_path = (
        project_root / "emergence-dev.toml"
        if path is None
        else Path(path).expanduser().resolve()
    )
    try:
        raw_bytes = config_path.read_bytes()
        payload = tomllib.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise DevAutomationError(f"cannot load configuration: {config_path}") from exc
    if payload.get("schema_version") != CONFIG_VERSION:
        raise DevAutomationError(f"configuration schema_version must be {CONFIG_VERSION!r}")
    index = payload.get("index")
    limits = payload.get("limits")
    if not isinstance(index, dict) or not isinstance(limits, dict):
        raise DevAutomationError("configuration requires [index] and [limits]")
    _string_list(index.get("roots"), field="index.roots")
    _string_list(index.get("config_roots", []), field="index.config_roots", allow_empty=True)
    _string_list(index.get("extensions"), field="index.extensions")
    _string_list(index.get("exclude", []), field="index.exclude", allow_empty=True)
    normalized_limits = dict(limits)
    normalized_limits["context_tokens"] = _integer(
        limits.get("context_tokens"), field="limits.context_tokens", minimum=500, maximum=50_000
    )
    normalized_limits["internal_context_tokens"] = _integer(
        limits.get("internal_context_tokens"),
        field="limits.internal_context_tokens",
        minimum=500,
        maximum=50_000,
    )
    if normalized_limits["internal_context_tokens"] > normalized_limits["context_tokens"]:
        raise DevAutomationError("internal_context_tokens must not exceed context_tokens")
    normalized_limits["candidate_files"] = _integer(
        limits.get("candidate_files"), field="limits.candidate_files", minimum=1, maximum=50
    )
    normalized_limits["default_owned_files"] = _integer(
        limits.get("default_owned_files"), field="limits.default_owned_files", minimum=1, maximum=50
    )
    normalized_limits["extended_owned_files"] = _integer(
        limits.get("extended_owned_files"), field="limits.extended_owned_files", minimum=1, maximum=100
    )
    normalized_limits["max_validation_commands"] = _integer(
        limits.get("max_validation_commands"),
        field="limits.max_validation_commands",
        minimum=1,
        maximum=100,
    )
    normalized_limits["log_tail_lines"] = _integer(
        limits.get("log_tail_lines"), field="limits.log_tail_lines", minimum=5, maximum=500
    )
    if normalized_limits["extended_owned_files"] < normalized_limits["default_owned_files"]:
        raise DevAutomationError("extended_owned_files must be at least default_owned_files")
    normalized = dict(payload)
    normalized["limits"] = normalized_limits
    return DevConfig(
        root=project_root,
        path=config_path,
        raw=normalized,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def resolve_state_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise DevAutomationError("state root must be absolute")
        return path.resolve()
    return resolve_runtime_roots().runtime_root / "emergence_dev"


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _candidate_paths(config: DevConfig) -> list[Path]:
    extensions = frozenset(config.index["extensions"])
    excludes = tuple(config.index["exclude"])
    paths: set[Path] = set()
    for configured_root in (*config.index["roots"], *config.index["config_roots"]):
        base = (config.root / configured_root).resolve()
        try:
            base.relative_to(config.root)
        except ValueError as exc:
            raise DevAutomationError(f"index root escapes project: {configured_root}") from exc
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else base.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            relative = path.relative_to(config.root).as_posix()
            if not _matches(relative, excludes):
                paths.add(path)
    return sorted(paths, key=lambda item: item.relative_to(config.root).as_posix())


def _module_name(relative: str) -> str | None:
    if not relative.endswith(".py"):
        return None
    parts = list(Path(relative).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _tokens(value: str) -> set[str]:
    result: set[str] = set()
    for match in _WORD_RE.finditer(value):
        raw = match.group(0).replace("/", " ").replace(".", " ").replace("-", " ")
        for part in re.split(r"(?<=[a-z0-9])(?=[A-Z])|_|\s+", raw):
            lowered = part.lower()
            if len(lowered) >= 2:
                result.add(lowered)
                if len(lowered) > 4 and lowered.endswith("s"):
                    result.add(lowered[:-1])
    return result


def _objective_tokens(value: str) -> set[str]:
    return _tokens(value) - _OBJECTIVE_STOPWORDS


def _python_metadata(text: str) -> tuple[list[dict[str, Any]], list[str], bool]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], [], True
    definitions: list[dict[str, Any]] = []
    imports: set[str] = set()
    dynamic = False
    for node in ast.walk(tree):
        if isinstance(node, _PYTHON_DEFINITION_TYPES):
            definitions.append(
                {
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                }
            )
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in {"import_module", "__import__", "getattr", "setattr"}:
                dynamic = True
    return sorted(definitions, key=lambda row: (row["line"], row["name"])), sorted(imports), dynamic


def build_source_index(config: DevConfig, *, state_root: Path) -> dict[str, Any]:
    candidate_paths = _candidate_paths(config)
    fingerprint = canonical_sha256(
        {
            "config_sha256": config.config_sha256,
            "files": [
                {
                    "path": path.relative_to(config.root).as_posix(),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in candidate_paths
            ],
        }
    )
    cache_path = state_root / "cache" / "source_index_v1.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if (
        isinstance(cached, dict)
        and cached.get("cache_fingerprint") == fingerprint
        and isinstance(cached.get("index"), dict)
        and cached["index"].get("schema_version") == INDEX_VERSION
    ):
        return cached["index"]

    records: list[dict[str, Any]] = []
    for path in candidate_paths:
        relative = path.relative_to(config.root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        definitions: list[dict[str, Any]] = []
        imports: list[str] = []
        dynamic = False
        if path.suffix.lower() == ".py":
            definitions, imports, dynamic = _python_metadata(text)
        records.append(
            {
                "path": relative,
                "module": _module_name(relative),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "line_count": text.count("\n") + 1,
                "definitions": definitions,
                "imports": imports,
                "dynamic_coupling": dynamic,
                "tokens": sorted(_tokens(relative) | _tokens(" ".join(row["name"] for row in definitions)) | _tokens(text)),
            }
        )
    manifest = {
        "schema_version": INDEX_VERSION,
        "project_root": str(config.root),
        "config_sha256": config.config_sha256,
        "records": records,
    }
    manifest["index_sha256"] = canonical_sha256(
        {
            "schema_version": INDEX_VERSION,
            "config_sha256": config.config_sha256,
            "records": records,
        }
    )
    atomic_write_json(
        cache_path,
        {"cache_fingerprint": fingerprint, "index": manifest},
    )
    return manifest


def _git_output(root: Path, *argv: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *argv],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _instruction_hashes(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for directory in (root, *root.parents):
        path = directory / "AGENTS.md"
        if path.is_file() and path not in seen:
            seen.add(path)
            rows.append({"path": str(path), "sha256": file_sha256(path)})
        if (directory / ".git").exists():
            break
    return rows


def _score_record(record: Mapping[str, Any], objective_tokens: set[str]) -> tuple[int, str]:
    path_tokens = _tokens(record["path"])
    symbol_tokens = _tokens(" ".join(item["name"] for item in record["definitions"]))
    all_tokens = set(record["tokens"])
    score = (
        12 * len(objective_tokens & symbol_tokens)
        + 22 * len(objective_tokens & path_tokens)
        + min(4, len(objective_tokens & all_tokens))
    )
    if record["path"].startswith("tests/"):
        score -= 20
    elif record["path"].startswith(("engine/", "tools/", "experiments/", "experiments_shared/")):
        score += 20
    if record["path"].startswith("docs/"):
        score -= 4
    return score, record["path"]


def _relevant_snippet(root: Path, record: Mapping[str, Any], terms: set[str], max_lines: int = 40) -> str:
    path = root / record["path"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    ranges: list[tuple[int, int]] = []
    for definition in record["definitions"]:
        if terms & _tokens(definition["name"]):
            ranges.append((max(1, definition["line"] - 2), min(len(lines), definition["end_line"] + 2)))
    if not ranges:
        for index, line in enumerate(lines, 1):
            if terms & _tokens(line):
                ranges.append((max(1, index - 3), min(len(lines), index + 3)))
            if len(ranges) >= 4:
                break
    if not ranges:
        ranges.append((1, min(len(lines), 35)))
    selected: list[int] = []
    for start, end in ranges:
        for line_number in range(start, end + 1):
            if line_number not in selected:
                selected.append(line_number)
            if len(selected) >= max_lines:
                break
        if len(selected) >= max_lines:
            break
    chunks: list[str] = []
    previous = 0
    for line_number in selected:
        if previous and line_number != previous + 1:
            chunks.append("    …")
        chunks.append(f"{line_number:>5}: {lines[line_number - 1]}")
        previous = line_number
    return "\n".join(chunks)


def _related_tests(records: list[Mapping[str, Any]], selected: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    module_weights: dict[str, int] = {}
    for index, record in enumerate(selected):
        module = record["module"]
        if not module or record["path"].startswith("tests/"):
            continue
        weight = 20 if index == 0 else max(1, len(selected) - index)
        module_weights[module] = max(weight, module_weights.get(module, 0))
        if module.startswith("engine."):
            alias = module.removeprefix("engine.")
            module_weights[alias] = max(weight, module_weights.get(alias, 0))
    stems = [_tokens(record["path"]) for record in selected]
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record["path"].startswith("tests/") or not Path(record["path"]).name.startswith("test_"):
            continue
        import_scores: dict[str, int] = {}
        for module, module_weight in module_weights.items():
            score = 0
            for value in record["imports"]:
                if value == module:
                    score = max(score, 100 + 20 * module_weight)
                elif value.startswith(f"{module}."):
                    score = max(score, 80 + 20 * module_weight)
                elif module.startswith(f"{value}."):
                    score = max(score, 50 + 20 * module_weight)
            if score:
                import_scores[module] = score
        import_match = sorted(import_scores)
        import_score = max(import_scores.values(), default=0)
        lexical = max((len(tokens & set(record["tokens"])) for tokens in stems), default=0)
        selected_candidate = record in selected
        if import_match or lexical >= 3 or selected_candidate:
            rows.append(
                {
                    "path": record["path"],
                    "reason": (
                        "direct_import"
                        if import_match
                        else "ranked_candidate"
                        if selected_candidate
                        else "lexical_advisory"
                    ),
                    "imports": import_match,
                    "selection_score": import_score + lexical * 5,
                }
            )
    priority = {"direct_import": 0, "ranked_candidate": 1, "lexical_advisory": 2}
    return sorted(
        rows,
        key=lambda row: (priority[row["reason"]], -row["selection_score"], row["path"]),
    )[:12]


def _relationship_rows(config: DevConfig, candidate_paths: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relationship in config.relationships:
        source = relationship.get("source")
        counterparts = relationship.get("counterparts", [])
        related = bool(
            isinstance(source, str)
            and (
                source in candidate_paths
                or any(isinstance(item, str) and item in candidate_paths for item in counterparts)
            )
        )
        if related:
            rows.append(dict(relationship))
    return rows


def _critical_rows(config: DevConfig, objective: str, candidate_paths: set[str]) -> list[dict[str, Any]]:
    terms = _objective_tokens(objective)
    rows: list[dict[str, Any]] = []
    for rule in config.critical_rules:
        paths = [item for item in rule.get("paths", []) if isinstance(item, str)]
        keywords = _tokens(" ".join(item for item in rule.get("keywords", []) if isinstance(item, str)))
        matches = sorted(path for path in candidate_paths if _matches(path, paths))
        if matches or terms & keywords:
            rows.append(
                {
                    "id": rule.get("id", "unnamed"),
                    "matching_paths": matches,
                    "notes": list(rule.get("notes", [])),
                    "mandatory": bool(rule.get("mandatory", False)),
                    "required_validation": rule.get("required_validation"),
                }
            )
    return rows


def estimate_context_tokens(value: str) -> int:
    """Conservatively estimate tokens without a model-specific tokenizer."""
    return math.ceil(len(value.encode("utf-8")) / 3)


def _render_context(payload: Mapping[str, Any], token_budget: int) -> str:
    lines = [
        f"# Emergence development context: {payload['objective']}",
        "",
        f"- Baseline: `{payload['git']['head']}` on `{payload['git']['branch']}`",
        f"- Index: `{payload['index_sha256']}`",
        f"- Internal context ceiling: {token_budget} conservatively estimated tokens.",
        "- Candidate list is advisory; critical and dynamic relationships may expand it.",
        "",
        "## Obligations and uncertainty",
        "",
    ]
    for row in payload["critical_obligations"]:
        requirement = row.get("required_validation") or "explicit independent validation"
        lines.append(f"- {row['id']} ({requirement}): {' '.join(row['notes'])}")
    for row in payload["relationships"]:
        lines.append(f"- Generated relationship `{row.get('id')}` must remain synchronized.")
    if payload["dynamic_candidates"]:
        lines.append(
            "- Dynamic coupling detected in: "
            + ", ".join(f"`{path}`" for path in payload["dynamic_candidates"])
        )
    if not payload["critical_obligations"] and not payload["relationships"] and not payload["dynamic_candidates"]:
        lines.append("- No configured critical obligation was selected; candidate selection remains advisory.")
    lines.extend(["", "## Suggested tests", ""])
    lines.extend(f"- `{row['path']}`: {row['reason']}" for row in payload["suggested_tests"])
    if not payload["suggested_tests"]:
        lines.append("- No direct test was found; an explicit selector is required before validation.")
    lines.extend(["", "## Ranked candidates", ""])
    lines.extend(
        f"- `{candidate['path']}` (score {candidate['score']})"
        for candidate in payload["candidates"]
    )
    lines.extend(["", "## Candidate excerpts", ""])
    omitted: list[str] = []
    for candidate in payload["candidates"]:
        block = [f"### {candidate['path']} (score {candidate['score']})", ""]
        if candidate["dynamic_coupling"]:
            block.append("- Warning: dynamic import/attribute behavior detected.")
        if candidate["definitions"]:
            block.append("- Symbols: " + ", ".join(candidate["definitions"][:12]))
        block.extend(["", "```text", candidate["snippet"], "```", ""])
        projected = "\n".join([*lines, *block]).rstrip() + "\n"
        if estimate_context_tokens(projected) <= token_budget:
            lines.extend(block)
        else:
            omitted.append(candidate["path"])
    if omitted:
        suffix = ["Context ceiling omitted excerpts for:", *[f"- `{path}`" for path in omitted], ""]
        projected = "\n".join([*lines, *suffix]).rstrip() + "\n"
        if estimate_context_tokens(projected) <= token_budget:
            lines.extend(suffix)
    return "\n".join(lines).rstrip() + "\n"


def build_context(
    objective: str,
    *,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    objective = objective.strip()
    if not objective:
        raise DevAutomationError("objective must not be empty")
    root = find_project_root(project_root)
    config = load_config(root, config_path)
    resolved_state = resolve_state_root(state_root)
    index = build_source_index(config, state_root=resolved_state)
    records = index["records"]
    terms = _objective_tokens(objective)
    ranked = sorted(
        ((_score_record(record, terms), record) for record in records),
        key=lambda item: (-item[0][0], item[0][1]),
    )
    selected = [record for (score, _), record in ranked if score > 0][: config.limits["candidate_files"]]
    if not selected:
        selected = [record for _, record in ranked[: config.limits["candidate_files"]]]
    candidates = [
        {
            "path": record["path"],
            "score": _score_record(record, terms)[0],
            "sha256": record["sha256"],
            "definitions": [row["name"] for row in record["definitions"]],
            "imports": record["imports"],
            "dynamic_coupling": record["dynamic_coupling"],
            "snippet": _relevant_snippet(root, record, terms),
        }
        for record in selected
    ]
    candidate_paths = {row["path"] for row in candidates}
    payload: dict[str, Any] = {
        "schema_version": CONTEXT_VERSION,
        "objective": objective,
        "project_root": str(root),
        "config_path": str(config.path),
        "config_sha256": config.config_sha256,
        "index_sha256": index["index_sha256"],
        "git": {
            "head": _git_output(root, "rev-parse", "HEAD"),
            "branch": _git_output(root, "branch", "--show-current"),
            "status": _git_output(root, "status", "--short"),
        },
        "instruction_files": _instruction_hashes(root),
        "candidates": candidates,
        "suggested_tests": _related_tests(records, selected),
        "relationships": _relationship_rows(config, candidate_paths),
        "critical_obligations": _critical_rows(config, objective, candidate_paths),
        "dynamic_candidates": sorted(row["path"] for row in candidates if row["dynamic_coupling"]),
        "advisory": True,
        "state_root": str(resolved_state),
    }
    internal_budget = config.limits["internal_context_tokens"]
    markdown = _render_context(payload, internal_budget)
    payload["context_markdown"] = markdown
    payload["context_budget"] = {
        "external_tokens": config.limits["context_tokens"],
        "internal_tokens": internal_budget,
        "estimator": "utf8_bytes_ceil_div_3",
    }
    payload["estimated_context_tokens"] = estimate_context_tokens(markdown)
    if payload["estimated_context_tokens"] > internal_budget:
        raise DevAutomationError(
            "mandatory context exceeds internal ceiling: "
            f"{payload['estimated_context_tokens']} > {internal_budget}"
        )
    payload["context_sha256"] = canonical_sha256(payload)
    return payload


def write_context(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() == ".json":
        atomic_write_json(path, payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload["context_markdown"], encoding="utf-8")


def interpreter() -> str:
    return str(Path(sys.executable).resolve())
