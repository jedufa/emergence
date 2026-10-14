"""Content binding for executable human/operator instructions."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping

from tools.actors.canonical import CANONICAL_SHA256_KIND
from tools.actors.common import ActorError, actor_state_root, file_sha256

INSTRUCTION_BINDING_SCHEMA_VERSION = "eiel_operator_instruction_binding_v1"


def bind_operator_instruction(
    raw: str | Path | Mapping[str, Any] | None,
    *,
    state_root: str | Path | None = None,
    base_dir: str | Path | None = None,
    allow_inline_authority: bool = False,
) -> dict[str, Any] | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, Mapping):
        value = dict(raw)
        if value.get("schema_version") != INSTRUCTION_BINDING_SCHEMA_VERSION:
            raise ActorError("unsupported operator instruction binding", code="schema_version_unsupported")
        return value
    text = str(raw).strip()
    candidate = Path(text).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = Path(base_dir).expanduser().resolve() / candidate
    if candidate.is_file():
        candidate = candidate.resolve()
        digest = file_sha256(candidate)
        retained = actor_state_root(state_root) / "operator-instructions" / f"{digest}{candidate.suffix or '.txt'}"
        retained.parent.mkdir(parents=True, exist_ok=True)
        if retained.exists():
            if file_sha256(retained) != digest:
                raise ActorError("retained operator instruction conflicts with digest", code="operator_instruction_conflict")
        else:
            shutil.copyfile(candidate, retained)
        base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
        try:
            source_locator = candidate.relative_to(base).as_posix() if base is not None else candidate.name
        except ValueError:
            # Exact authority comes from the digest and retained locator. Keep only
            # a stable advisory filename for external material so actor records do
            # not depend on machine- or fixture-specific absolute paths.
            source_locator = f"external:{candidate.name}"
        return {
            "schema_version": INSTRUCTION_BINDING_SCHEMA_VERSION,
            "binding_kind": "CONTENT_ADDRESSED_FILE",
            "source_path": source_locator,
            "state_root_locator": retained.relative_to(actor_state_root(state_root)).as_posix(),
            "sha256": digest,
            "digest_kind": "raw_file_sha256_v1",
            "execution_authority": True,
        }
    if allow_inline_authority:
        payload = text.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        retained = actor_state_root(state_root) / "operator-instructions" / f"{digest}.txt"
        retained.parent.mkdir(parents=True, exist_ok=True)
        if retained.exists():
            if hashlib.sha256(retained.read_bytes()).hexdigest() != digest:
                raise ActorError(
                    "retained inline operator instruction conflicts with digest",
                    code="operator_instruction_conflict",
                )
        else:
            retained.write_bytes(payload)
        return {
            "schema_version": INSTRUCTION_BINDING_SCHEMA_VERSION,
            "binding_kind": "CONTENT_ADDRESSED_FILE",
            "source_path": f"inline:{digest}",
            "state_root_locator": retained.relative_to(actor_state_root(state_root)).as_posix(),
            "sha256": digest,
            "digest_kind": "raw_utf8_sha256_v1",
            "execution_authority": True,
        }
    return {
        "schema_version": INSTRUCTION_BINDING_SCHEMA_VERSION,
        "binding_kind": "PROVENANCE_ONLY_LABEL",
        "label": text,
        "digest_kind": CANONICAL_SHA256_KIND,
        "execution_authority": False,
    }


def instruction_is_authoritative(value: Mapping[str, Any] | None) -> bool:
    return bool(value and value.get("execution_authority") and value.get("sha256"))

__all__ = ["INSTRUCTION_BINDING_SCHEMA_VERSION", "bind_operator_instruction", "instruction_is_authoritative"]
