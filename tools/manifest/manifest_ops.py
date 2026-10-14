#!/usr/bin/env python3
"""Compatibility wrapper over shared manifest-document operations."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from experiments_shared.job_manifests.ops import (
    append_jobs,
    canonical_json,
    compute_job_fingerprint,
    dump_manifest_yaml,
    get_job_fingerprint,
    load_manifest_document,
    write_manifest,
)

Manifest = Dict[str, Any]
JobDef = Dict[str, Any]


def load_manifest(path: str) -> Manifest:
    return load_manifest_document(path)
