"""Completed-campaign Chronicle materialization and permanent-history closeout."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from simulation.chronicle.recorder import ScientificChronicleRecorder

from .provider_native_chronicle_v1 import (
    RECEIPT_SCHEMA_VERSION as PROVIDER_NATIVE_RECEIPT_SCHEMA_VERSION,
    SOURCE_COMPLETENESS as PROVIDER_NATIVE_SOURCE_COMPLETENESS,
)
from .production_engine_identity import PRODUCTION_PROVIDER_ID


class CampaignChronicleCloseoutError(RuntimeError):
    """A completed campaign cannot be closed into permanent Chronicle history."""


def _integer(value: Any, default: int = 0) -> int:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _metric(row: Mapping[str, Any], *names: str, default: int = 0) -> int:
    nested = row.get("metrics")
    sources = (row, nested if isinstance(nested, Mapping) else {})
    for source in sources:
        for name in names:
            if name in source:
                return _integer(source[name], default)
    return default


def _summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    terminal_generation: int,
) -> list[dict[str, Any]]:
    by_generation: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        generation = _integer(row.get("generation"), terminal_generation)
        if generation < 0:
            continue
        by_generation[generation] = {
            "generation": generation,
            "alive": _metric(row, "alive", "population.alive"),
            "births": _metric(row, "births", "population.births"),
            "failed_births": _metric(
                row, "failed_births", "population.failed_births"
            ),
            "deaths": _metric(row, "deaths", "population.deaths"),
            "chronicle_source_component": "engine.campaigns.provider_metrics",
            "chronicle_source_record_ref": (
                f"batches/*/manifest.json:cell={row.get('cell_id')}:"
                f"generation={generation}"
            ),
        }
    return [by_generation[key] for key in sorted(by_generation)]


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignChronicleCloseoutError(f"{name} is missing or malformed")
    return value


def _read_json_object(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignChronicleCloseoutError(f"{name} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignChronicleCloseoutError(f"cannot read {name}: {path}: {exc}") from exc
    return _object(value, name=name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl_objects(path: Path, *, name: str) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise CampaignChronicleCloseoutError(f"{name} is missing or unsafe: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                raise CampaignChronicleCloseoutError(
                    f"{name} contains a blank row at line {line_number}"
                )
            rows.append(
                _object(
                    json.loads(line),
                    name=f"{name} line {line_number}",
                )
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignChronicleCloseoutError(f"cannot read {name}: {path}: {exc}") from exc
    return rows


def _provider_native_chronicles(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    batch_manifests: Sequence[Mapping[str, Any]],
    cell_ids: Sequence[str],
) -> dict[str, Path]:
    """Validate generic provider-native receipts without metrics fallback."""

    campaign_id = str(plan.get("campaign_id") or "")
    expected = set(cell_ids)
    roots: dict[str, Path] = {}
    resolved_run_root = run_root.resolve()
    for raw_manifest in batch_manifests:
        manifest = _object(raw_manifest, name="completed provider-native batch")
        batch_id = manifest.get("batch_id")
        manifest_cells = manifest.get("cell_ids")
        result = _object(manifest.get("result"), name="provider-native batch result")
        if (
            result.get("provider_native_chronicle_required") is not True
            or not isinstance(batch_id, str)
            or not batch_id
            or Path(batch_id).name != batch_id
            or not isinstance(manifest_cells, list)
            or not all(
                isinstance(value, str) and Path(value).name == value
                for value in manifest_cells
            )
        ):
            raise CampaignChronicleCloseoutError(
                "provider/profile requires a valid provider-native Chronicle receipt"
            )
        receipt = _object(
            result.get("provider_native_chronicle"),
            name=f"batch {batch_id!r} provider-native Chronicle receipt",
        )
        cell_receipts = _object(
            receipt.get("cell_receipts"),
            name=f"batch {batch_id!r} provider-native cell receipts",
        )
        if (
            receipt.get("schema_version") != PROVIDER_NATIVE_RECEIPT_SCHEMA_VERSION
            or receipt.get("campaign_id") != campaign_id
            or receipt.get("batch_id") != batch_id
            or receipt.get("profile") != "scientific_full"
            or receipt.get("source_completeness")
            != PROVIDER_NATIVE_SOURCE_COMPLETENESS
            or receipt.get("finalized") is not True
            or set(cell_receipts) != set(manifest_cells)
        ):
            raise CampaignChronicleCloseoutError(
                f"batch {batch_id!r} provider-native Chronicle receipt is invalid"
            )
        for cell_id in manifest_cells:
            if cell_id not in expected or cell_id in roots:
                raise CampaignChronicleCloseoutError(
                    f"provider-native Chronicle cell is foreign or duplicated: {cell_id!r}"
                )
            cell_receipt = _object(
                cell_receipts[cell_id],
                name=f"world {cell_id!r} provider-native Chronicle receipt",
            )
            raw_root = cell_receipt.get("chronicle_root")
            if not isinstance(raw_root, str) or not raw_root:
                raise CampaignChronicleCloseoutError(
                    f"world {cell_id!r} provider-native Chronicle root is missing"
                )
            root = Path(raw_root).expanduser()
            cell_root = resolved_run_root / "batches" / batch_id / "cells" / cell_id
            resolved_cell_root = cell_root.resolve()
            resolved_root = root.resolve()
            if (
                not root.is_absolute()
                or root.is_symlink()
                or not resolved_root.is_dir()
                or cell_root.is_symlink()
                or not resolved_cell_root.is_dir()
                or resolved_root == resolved_cell_root
                or resolved_cell_root not in resolved_root.parents
            ):
                raise CampaignChronicleCloseoutError(
                    f"world {cell_id!r} provider-native Chronicle root is unsafe"
                )
            manifest_path = resolved_root / "chronicle_manifest.json"
            summary_path = resolved_root / "scientific_chronicle_summary.json"
            chronicle_manifest = _read_json_object(
                manifest_path,
                name=f"world {cell_id!r} provider-native Chronicle manifest",
            )
            summary = _read_json_object(
                summary_path,
                name=f"world {cell_id!r} provider-native Chronicle summary",
            )
            segments = cell_receipt.get("segment_bindings")
            if (
                cell_receipt.get("cell_id") != cell_id
                or cell_receipt.get("finalized") is not True
                or not isinstance(segments, list)
                or not segments
                or cell_receipt.get("segment_count") != len(segments)
                or cell_receipt.get("chronicle_manifest_sha256")
                != _sha256_file(manifest_path)
                or cell_receipt.get("chronicle_summary_sha256")
                != _sha256_file(summary_path)
            ):
                raise CampaignChronicleCloseoutError(
                    f"world {cell_id!r} provider-native cell receipt is invalid"
                )
            expected_segment = 0
            previous_end = 0
            for binding in segments:
                if (
                    not isinstance(binding, Mapping)
                    or binding.get("segment_index") != expected_segment
                    or binding.get("start_generation") != previous_end
                    or type(binding.get("end_generation")) is not int
                    or binding["end_generation"] <= previous_end
                    or not isinstance(binding.get("source_segment_sha256"), str)
                    or len(binding["source_segment_sha256"]) != 64
                    or not isinstance(binding.get("cell_segment_sha256"), str)
                    or len(binding["cell_segment_sha256"]) != 64
                ):
                    raise CampaignChronicleCloseoutError(
                        f"world {cell_id!r} provider-native segment chain is invalid"
                    )
                previous_end = binding["end_generation"]
                expected_segment += 1
            required = {
                "run_id": cell_id,
                "campaign_id": campaign_id,
                "profile": "scientific_full",
                "source_completeness": PROVIDER_NATIVE_SOURCE_COMPLETENESS,
                "finalized": True,
                "segment_count": len(segments),
            }
            if any(
                chronicle_manifest.get(name) != value
                for name, value in required.items()
            ) or any(summary.get(name) != value for name, value in required.items()):
                raise CampaignChronicleCloseoutError(
                    f"world {cell_id!r} provider-native Chronicle is incomplete"
                )
            events = _read_jsonl_objects(
                resolved_root / "formal_events.jsonl",
                name=f"world {cell_id!r} provider-native formal events",
            )
            if (
                cell_receipt.get("event_cursor") != len(events)
                or summary.get("event_cursor") != len(events)
                or any(
                    event.get("event_type")
                    != "provider_native_reproduction_attempt"
                    or event.get("run_id") != cell_id
                    or event.get("campaign_id") != campaign_id
                    or not isinstance(event.get("measurements"), Mapping)
                    or not isinstance(
                        event["measurements"].get("source_segment_sha256"), str
                    )
                    or event["measurements"]["source_segment_sha256"]
                    not in {
                        binding["source_segment_sha256"] for binding in segments
                    }
                    for event in events
                )
            ):
                raise CampaignChronicleCloseoutError(
                    f"world {cell_id!r} provider-native events are not source-bound"
                )
            roots[cell_id] = resolved_root
    if set(roots) != expected:
        raise CampaignChronicleCloseoutError(
            f"provider-native closeout is missing worlds: {sorted(expected - set(roots))!r}"
        )
    return roots


def materialize_completed_campaign_chronicles(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    batch_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Return finalized per-world Chronicles owned by the production engine."""

    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise CampaignChronicleCloseoutError("compiled plan has no output contract")
    profile = str(output.get("chronicle_profile", "notable"))
    if profile not in {"summary", "notable", "scientific_full"}:
        raise CampaignChronicleCloseoutError(
            f"unsupported completed-run Chronicle profile: {profile!r}"
        )
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise CampaignChronicleCloseoutError("compiled plan has no primary worlds")
    cell_ids = []
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
            raise CampaignChronicleCloseoutError("compiled plan has an invalid world")
        cell_ids.append(str(cell["cell_id"]))

    provider_ids = {manifest.get("provider_id") for manifest in batch_manifests}
    if not batch_manifests or provider_ids != {PRODUCTION_PROVIDER_ID}:
        observed = sorted(repr(provider_id) for provider_id in provider_ids)
        raise CampaignChronicleCloseoutError(
            "production Chronicle closeout requires only provider "
            f"{PRODUCTION_PROVIDER_ID!r}; observed {observed!r}"
        )
    provider_native_required = any(
        isinstance(manifest.get("result"), Mapping)
        and manifest["result"].get("provider_native_chronicle_required") is True
        for manifest in batch_manifests
    )
    if provider_native_required:
        return _provider_native_chronicles(
            plan,
            run_root=run_root,
            batch_manifests=batch_manifests,
            cell_ids=cell_ids,
        )
    rows_by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for manifest in batch_manifests:
        raw_rows = manifest.get("metrics_rows")
        if not isinstance(raw_rows, list):
            raise CampaignChronicleCloseoutError(
                "completed batch has no provider metrics rows"
            )
        for row in raw_rows:
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str):
                rows_by_cell[str(row["cell_id"])].append(row)

    execution = plan.get("execution")
    terminal_generation = (
        _integer(execution.get("generations"), 0)
        if isinstance(execution, Mapping)
        else 0
    )
    world_roots: dict[str, Path] = {}
    campaign_id = str(plan["campaign_id"])
    for cell_id in cell_ids:
        summaries = _summary_rows(
            rows_by_cell.get(cell_id, ()),
            terminal_generation=terminal_generation,
        )
        if not summaries:
            raise CampaignChronicleCloseoutError(
                f"completed world {cell_id!r} has no Chronicle-capable metrics"
            )
        world_root = run_root / "chronicle_worlds" / cell_id
        sim_cfg = SimpleNamespace(
            chronicle_enable=True,
            chronicle_profile=profile,
            chronicle_allow_large_output=profile == "scientific_full",
            generations=terminal_generation,
            campaign_id=campaign_id,
            chronicle_run_id=cell_id,
        )
        recorder = ScientificChronicleRecorder(
            SimpleNamespace(sim_cfg=sim_cfg, run_dir=world_root)
        )
        recorder.observe_native_terminal_summaries(summaries)
        recorder.finalize(status="complete")
        manifest_path = world_root / "chronicle" / "chronicle_manifest.json"
        if not recorder.finalized or not manifest_path.is_file():
            raise CampaignChronicleCloseoutError(
                f"world {cell_id!r} Chronicle did not finalize"
            )
        world_roots[cell_id] = world_root / "chronicle"
    return world_roots


__all__ = [
    "CampaignChronicleCloseoutError",
    "materialize_completed_campaign_chronicles",
]
