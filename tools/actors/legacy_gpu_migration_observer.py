#!/usr/bin/env python3
"""Resource-Manager observation subprocess for the one bounded GPU cutover.

The Scheduler invokes this module while holding its global execution lock.
It accepts a complete immutable request on stdin, performs the live procfs and
CUDA observations itself, and returns the Resource Manager proof on stdout.
It has no mutation or lease-release capability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from tools.actors.common import ActorError
from tools.actors.resource_manager import observe_legacy_gpu_migration_reclaim_v1


REQUEST_SCHEMA_VERSION = "eiel_legacy_gpu_migration_observation_request_v1"
_REQUEST_FIELDS = {
    "schema_version",
    "repo_root",
    "migration_manifest_sha256",
    "migration_row_sha256",
    "classification",
    "attempt_id",
    "resource_lease_digest",
    "fencing_token_digest",
    "continuation_lease_digest",
    "continuation_fence_code",
    "continuation_terminal_event_digest",
    "continuation_successor_event_digest",
    "device_uuid",
    "selected_resource_ids",
    "start_effect_digest",
    "resource_started_event_digest",
    "owner_pid",
    "owner_process_start_ticks",
}


def observe(request: Mapping[str, Any]) -> dict[str, Any]:
    """Collect a fresh proof without accepting caller-supplied observations."""

    value = dict(request)
    if set(value) != _REQUEST_FIELDS or value.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ActorError(
            "legacy GPU migration observation request is malformed",
            code="launchpad_legacy_gpu_migration_denied",
        )
    repo_root_value = value.get("repo_root")
    if not isinstance(repo_root_value, str) or not repo_root_value:
        raise ActorError(
            "legacy GPU migration observation lacks a repository root",
            code="launchpad_legacy_gpu_migration_denied",
        )
    repo_root = Path(repo_root_value).expanduser().resolve()
    if not repo_root.is_dir():
        raise ActorError(
            "legacy GPU migration repository root is unavailable",
            code="launchpad_legacy_gpu_migration_denied",
        )
    return observe_legacy_gpu_migration_reclaim_v1(
        repo_root=repo_root,
        migration_manifest_sha256=str(value["migration_manifest_sha256"]),
        migration_row_sha256=str(value["migration_row_sha256"]),
        classification=str(value["classification"]),
        attempt_id=str(value["attempt_id"]),
        resource_lease_digest=dict(value["resource_lease_digest"]),
        fencing_token_digest=dict(value["fencing_token_digest"]),
        continuation_lease_digest=dict(value["continuation_lease_digest"]),
        continuation_fence_code=str(value["continuation_fence_code"]),
        continuation_terminal_event_digest=value[
            "continuation_terminal_event_digest"
        ],
        continuation_successor_event_digest=value[
            "continuation_successor_event_digest"
        ],
        device_uuid=str(value["device_uuid"]),
        selected_resource_ids=list(value["selected_resource_ids"]),
        start_effect_digest=dict(value["start_effect_digest"]),
        resource_started_event_digest=dict(value["resource_started_event_digest"]),
        owner_pid=value.get("owner_pid"),
        owner_process_start_ticks=value.get("owner_process_start_ticks"),
    )


def main() -> int:
    try:
        raw = json.load(sys.stdin)
        if not isinstance(raw, Mapping):
            raise ActorError(
                "legacy GPU migration observation input is not an object",
                code="launchpad_legacy_gpu_migration_denied",
            )
        print(json.dumps(observe(raw), sort_keys=True, separators=(",", ":")))
        return 0
    except (ActorError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "eiel_legacy_gpu_migration_observation_error_v1",
                    "error": str(error),
                    "code": getattr(error, "code", "legacy_gpu_migration_observation_failed"),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
