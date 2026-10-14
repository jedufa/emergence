"""Compatibility wrapper over shared manifest IO helpers."""

from __future__ import annotations

from typing import Any, Dict

from experiments_shared.job_manifests.io import (
    load_manifest,
    repo_root_from_manifest_file,
    resolve_output_root,
    utc_now_iso,
)


def _utc_now_iso() -> str:
    return utc_now_iso()


def _load_manifest(path: str) -> Dict[str, Any]:
    return load_manifest(path)


def _repo_root_from_this_file() -> str:
    return repo_root_from_manifest_file(__file__, levels_up=2)


def _resolve_output_root(cli_output_root: str, manifest_output_root: str, env_key: str, repo_root: str) -> str:
    return resolve_output_root(cli_output_root, manifest_output_root, env_key, repo_root)
