"""Prevent optional, disposable checkpoint copies becoming run dependencies.

The interesting-checkpoint directory is an operator convenience, never an
authoritative input store. Copies require their existing format-specific
checkpoint verifier; this module deliberately does not invent a copy format.
"""
from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT = Path(public_runtime_path("interesting-checkpoints"))
_PROSE_FIELDS = frozenset({
    "description", "question", "notes", "note", "rationale", "reproduction",
    "limitations", "title", "summary", "comment", "comments", "hypothesis",
})


class DisposableCheckpointDependencyError(ValueError):
    """A run attempts to depend on explicitly disposable state."""


def reject_disposable_checkpoint_dependencies(
    value: object, *, base_dir: Path | None = None,
) -> None:
    """Reject standalone path references into the disposable stash.

    Inspect nested mappings/sequences, including file URLs and resolved symlink
    aliases. Descriptive prose fields are not input bindings, so mentioning the
    directory in a question or report does not reject an otherwise valid run.
    Relative paths resolve against the request file's directory when supplied,
    otherwise the current working directory. No directory is created and no
    checkpoint bytes are opened.
    """
    protected = DEFAULT.resolve(strict=False)
    reference_base = Path.cwd() if base_dir is None else Path(base_dir)

    def visit(node: object, location: str) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if str(key).lower() not in _PROSE_FIELDS:
                    visit(item, f"{location}.{key}")
            return
        if isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                visit(item, f"{location}[{index}]")
            return
        if isinstance(node, Path):
            text = str(node)
        elif isinstance(node, str):
            text = node.strip()
        else:
            return
        if "\n" in text or "\r" in text:
            return
        if text.startswith("file:"):
            parsed = urlparse(text)
            if parsed.netloc not in {"", "localhost"}:
                return
            text = unquote(parsed.path)
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = reference_base / path
        resolved = path.resolve(strict=False)
        if resolved == protected or protected in resolved.parents:
            raise DisposableCheckpointDependencyError(
                f"{location} depends on disposable interesting checkpoints; "
                "regenerate the state from the run report into a fresh managed run"
            )

    visit(value, "request")


__all__ = ["DEFAULT", "DisposableCheckpointDependencyError", "reject_disposable_checkpoint_dependencies"]
