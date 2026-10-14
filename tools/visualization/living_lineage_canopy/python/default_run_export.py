"""Default compact terminal Canopy output for native simulation runs.

The artifact is deliberately a terminal identity snapshot.  It gives every
completed native run a browseable Canopy bundle without retaining a dense
per-generation replay or adding work to the generation loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import SCHEMA_VERSION
from .provenance import (
    NO_CLAIMS_STATEMENT,
    exporter_git_commit,
    output_hash,
    source_file_manifest,
    write_canonical_json,
)
from .schema import validate_playback


RELAY_SCHEMA_VERSION = "llc_near_live_playback_relay_v0_manifest"
RELAY_STATUS_SCHEMA_VERSION = "llc_near_live_playback_relay_v0_status"
DEFAULT_DIRECTORY_NAME = "lineage-playback-relay"
DEFAULT_POLICY_ENV = "EIEL_CANOPY_DEFAULT_PLAYBACK"
DEFAULT_POLICY_FILE = "canopy_playback_policy.json"


class DefaultCanopyExportError(ValueError):
    """The terminal checkpoint cannot produce a safe compact Canopy snapshot."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_canopy_enabled() -> bool:
    """Operational output policy; never a simulation or campaign input."""

    return os.environ.get(DEFAULT_POLICY_ENV, "true").strip().lower() not in {"0", "false", "no", "off"}


def _values(projection: Mapping[str, Any], name: str) -> list[Any]:
    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping) or name not in tensors:
        raise DefaultCanopyExportError(f"terminal projection is missing {name}")
    value = tensors[name]
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").reshape(-1).tolist()
    elif hasattr(value, "reshape") and hasattr(value, "tolist"):
        value = value.reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise DefaultCanopyExportError(f"terminal projection {name} is not a vector")
    return value


def _terminal_agents(projection: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alive = _values(projection, "alive_mask")
    agent_ids = _values(projection, "slot_to_agent_id")
    founders = _values(projection, "lineage_founder_id")
    if not (len(alive) == len(agent_ids) == len(founders)):
        raise DefaultCanopyExportError("terminal identity vectors have inconsistent lengths")

    agents: list[dict[str, Any]] = []
    lineage_ids: set[int] = set()
    seen_agent_ids: set[int] = set()
    for slot, (is_alive, raw_agent_id, raw_founder) in enumerate(zip(alive, agent_ids, founders, strict=True)):
        agent_id = int(raw_agent_id)
        if bool(is_alive) != (agent_id >= 0):
            raise DefaultCanopyExportError(f"terminal slot {slot} has inconsistent alive identity")
        if not bool(is_alive):
            continue
        if agent_id in seen_agent_ids:
            raise DefaultCanopyExportError(f"terminal projection duplicates agent id {agent_id}")
        seen_agent_ids.add(agent_id)
        founder_id = int(raw_founder)
        if founder_id < 0:
            raise DefaultCanopyExportError(f"terminal agent {agent_id} has no lineage founder")
        lineage_ids.add(founder_id)
        lineage_id = f"L{founder_id}"
        agents.append(
            {
                "agent_id": f"A{agent_id}",
                "lineage_id": lineage_id,
                "branch_id": f"B{founder_id}",
                "birth_tick": 0,
                "death_tick": None,
                "death_tick_source": "not_recorded",
                "parent_agent_ids": [],
            }
        )
    lineages = [
        {
            "lineage_id": f"L{founder_id}",
            "founder_lineage_id": f"L{founder_id}",
            "parent_lineage_id": None,
            "creation_tick": 0,
            "extinction_tick": None,
        }
        for founder_id in sorted(lineage_ids)
    ]
    branches = [
        {
            "branch_id": f"B{founder_id}",
            "lineage_id": f"L{founder_id}",
            "parent_branch_id": None,
            "creation_tick": 0,
            "extinction_tick": None,
        }
        for founder_id in sorted(lineage_ids)
    ]
    return agents, lineages, branches


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_default_terminal_canopy(
    *,
    run_root: Path,
    run_id: str,
    terminal_generation: int,
    projection: Mapping[str, Any],
    source_files: tuple[Path, ...] = (),
    source_kind: str = "native_terminal_identity_snapshot",
    source_schema_version: str = "native_terminal_checkpoint_projection",
    source_state: str = "source_complete",
) -> dict[str, Any]:
    """Write one verified terminal Canopy bundle for a completed native run.

    Existing valid bundles are retained verbatim.  The visualization is an
    identity snapshot at tick zero, so it never claims a historical replay.
    """

    root = run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not default_canopy_enabled():
        write_canonical_json(
            root / DEFAULT_POLICY_FILE,
            {"schema_version": "eiel_canopy_playback_policy_v1", "enabled": False, "reason": "explicit operational opt-out", "recorded_at": _utc_now()},
            overwrite=True,
        )
        return {"path": None, "status": "disabled", "policy_path": DEFAULT_POLICY_FILE}
    target = root / DEFAULT_DIRECTORY_NAME
    manifest_path = target / "relay_manifest.json"
    replaced_interrupted: Path | None = None
    if manifest_path.is_file():
        required = ("playback_snapshot.json", "provenance.json", "relay_status.json")
        if all((target / name).is_file() for name in required):
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DefaultCanopyExportError("existing default Canopy manifest is invalid") from exc
            if existing_manifest.get("source_state") != "source_interrupted" or source_state == "source_interrupted":
                return {
                    "path": DEFAULT_DIRECTORY_NAME,
                    "manifest_path": str(manifest_path.relative_to(root)),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "status": "existing",
                }
            replaced_interrupted = root / f".{DEFAULT_DIRECTORY_NAME}.interrupted"
            if replaced_interrupted.exists():
                shutil.rmtree(replaced_interrupted)
            os.replace(target, replaced_interrupted)
        else:
            raise DefaultCanopyExportError("existing default Canopy directory is incomplete")

    agents, lineages, branches = _terminal_agents(projection)
    playback = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": str(run_id),
            "source_kind": str(source_kind),
            "source_classification": "EXP-B",
            "semantics_mode": "partial_agents",
            "time_unit": "terminal_snapshot",
            "time_semantics": "single terminal identity snapshot; no historical playback is retained",
            "first_tick": 0,
            "last_tick": 0,
            "generation_mapping": {"terminal_generation": int(terminal_generation)},
            "parent_reference_policy": "allow_missing_external",
            "branch_semantics": "founder_lineage_terminal_group",
            "visualization_seed": 0,
            "source_cell_id": None,
            "notes": "Default terminal Canopy snapshot generated at run finalization.",
            "warnings": [
                "Terminal identity snapshot only: it does not reconstruct historical births, deaths, or branches.",
                NO_CLAIMS_STATEMENT,
            ],
        },
        "lineages": lineages,
        "branches": branches,
        "agents": agents,
    }
    validate_playback(playback)
    playback_hash = output_hash(playback)
    provenance = {
        "schema_version": "eiel_lineage_playback_export_provenance_v1",
        "exporter_schema_version": "eiel_default_terminal_canopy_export_v1",
        "exporter_git_commit": exporter_git_commit(),
        "playback_schema_version": SCHEMA_VERSION,
        "dataset_id": str(run_id),
        "source_root": str(root),
        "source_root_supplied": str(run_root),
        "source_root_resolved": str(root),
        "source_files": source_file_manifest(source_files),
        "source_run_id": str(run_id),
        "source_cell_id": None,
        "source_classification": "EXP-B",
        "semantics_mode": "partial_agents",
        "source_code_commit": None,
        "production_provider_identity": "eiel-production-provider-v1",
        "production_runtime_identity": "eiel-production-runtime-v1",
        "source_schema_versions": [str(source_schema_version)],
        "campaign_manifest_identity": None,
        "unavailable_fields": ["historical birth and death timing", "historical branch transitions"],
        "missing_fields": ["historical lifecycle replay"],
        "transformations_applied": ["terminal live identities grouped by founder lineage"],
        "projections_applied": ["terminal identity snapshot"],
        "output_sha256": playback_hash,
        "command": "native terminal default Canopy export",
        "generated_at": _utc_now(),
        "warnings": playback["metadata"]["warnings"],
        "no_claims_statement": NO_CLAIMS_STATEMENT,
    }
    provenance_hash = output_hash(provenance)
    generated_at = _utc_now()
    status = {
        "schema_version": RELAY_STATUS_SCHEMA_VERSION,
        "state": str(source_state),
        "source_boundary": 0,
        "playback_boundary": 0,
        "last_successful_update": generated_at,
        "incomplete_artifact_warnings": playback["metadata"]["warnings"],
    }
    manifest = {
        "schema_version": RELAY_SCHEMA_VERSION,
        "sequence_number": 1,
        "source_run_id": str(run_id),
        "source_state": str(source_state),
        "latest_completed_source_boundary": 0,
        "latest_exported_playback_boundary": 0,
        "source_lag": 0,
        "semantics_mode": "partial_agents",
        "playback_file_name": "playback_snapshot.json",
        "playback_sha256": playback_hash,
        "provenance_file_name": "provenance.json",
        "provenance_sha256": provenance_hash,
        "status_file_name": "relay_status.json",
        "poll_seconds": 60,
        "warnings": playback["metadata"]["warnings"],
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{DEFAULT_DIRECTORY_NAME}.", dir=root))
    try:
        write_canonical_json(temporary / "playback_snapshot.json", playback)
        write_canonical_json(temporary / "provenance.json", provenance)
        write_canonical_json(temporary / "relay_status.json", status)
        write_canonical_json(temporary / "relay_manifest.json", manifest)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if replaced_interrupted is not None and not target.exists():
            os.replace(replaced_interrupted, target)
        raise
    if replaced_interrupted is not None:
        shutil.rmtree(replaced_interrupted)
    return {
        "path": DEFAULT_DIRECTORY_NAME,
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": _sha256_file(manifest_path),
        "status": "generated",
        "agent_count": len(agents),
        "lineage_count": len(lineages),
    }


def materialize_default_terminal_canopy_from_agents(
    *,
    run_root: Path,
    run_id: str,
    terminal_generation: int,
    agents: Any,
    source_state: str = "source_complete",
) -> dict[str, Any]:
    """Write the same compact terminal Canopy bundle for the Python runner.

    Only live identities are read.  This is intentionally independent of the
    optional branch-observability recorder, so enabling the default artifact
    neither retains a full event stream nor alters simulation mechanics.
    """

    identity_rows: list[tuple[int, int]] = []
    for agent in agents:
        try:
            agent_id = int(getattr(agent, "agent_id"))
        except (TypeError, ValueError):
            raise DefaultCanopyExportError("live agent is missing a valid agent_id") from None
        stats = getattr(agent, "stats", None)
        try:
            founder_id = int(getattr(stats, "founder_id", agent_id))
        except (TypeError, ValueError):
            founder_id = agent_id
        if agent_id < 0 or founder_id < 0:
            raise DefaultCanopyExportError("live agent has an invalid lineage identity")
        identity_rows.append((agent_id, founder_id))
    identity_rows.sort()
    agent_ids = [agent_id for agent_id, _founder_id in identity_rows]
    founders = [founder_id for _agent_id, founder_id in identity_rows]
    return materialize_default_terminal_canopy(
        run_root=run_root,
        run_id=run_id,
        terminal_generation=terminal_generation,
        projection={
            "tensors": {
                "alive_mask": [True] * len(agent_ids),
                "slot_to_agent_id": agent_ids,
                "lineage_founder_id": founders,
            }
        },
        source_kind="python_terminal_identity_snapshot",
        source_schema_version="python_runner_live_population",
        source_state=source_state,
    )
