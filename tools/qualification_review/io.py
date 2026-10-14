"""Read-only input handling and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.actors.canonical import canonical_bytes


class ReviewInputError(ValueError):
    """Raised for unsafe, missing, or malformed review inputs."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_sha256(receipt: dict[str, Any]) -> str:
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ReviewInputError(f"JSON pointer must start with '/': {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                current = current[int(token)]
            else:
                current = current[token]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ReviewInputError(f"JSON pointer not found: {pointer}") from exc
    return current


class InputStore:
    """Resolve inputs beneath a fixed root and record every file hash."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def resolve(self, value: str, *, require_file: bool = True) -> Path:
        raw = Path(value)
        if raw.is_absolute():
            raise ReviewInputError(f"absolute input paths are forbidden: {value}")
        lexical = self.root / raw
        probe = self.root
        for part in raw.parts:
            if part in ("", "."):
                continue
            probe /= part
            if probe.is_symlink():
                raise ReviewInputError(f"symlink inputs are forbidden: {value}")
        candidate = lexical.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ReviewInputError(f"input escapes review root: {value}") from exc
        if require_file and not candidate.is_file():
            raise ReviewInputError(f"input file does not exist: {value}")
        return candidate

    def key(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def hash_file(self, path: Path) -> str:
        digest = file_sha256(path)
        self.hashes[self.key(path)] = digest
        return digest

    def read_json(self, value: str) -> Any:
        path = self.resolve(value)
        self.hash_file(path)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReviewInputError(f"invalid JSON input {value}: {exc}") from exc


def assertion_matches(actual: Any, operator: str, expected: Any = None) -> bool:
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "contains":
        return expected in actual
    raise ReviewInputError(f"unsupported assertion operator: {operator}")
