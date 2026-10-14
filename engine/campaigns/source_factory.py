"""Canonical campaign source preparation backed by the exact source cache."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any, Final

from .canonical import canonical_sha256
from .contracts import CampaignPlanV1
from .source_cache import (
    SourceCacheEntry,
    cache_source_projection,
    load_cached_source_projection,
    materialize_cached_source,
)


SOURCE_FACTORY_SCHEMA_VERSION: Final[str] = "eiel_campaign_source_factory_v2"
DISTINCT_PER_CELL: Final[str] = "distinct_per_cell"
SHARED_BY_EXACT_SOURCE_IDENTITY: Final[str] = "shared_by_exact_source_identity"
EXACT_INPUT_SOURCE_IDENTITY: Final[str] = "exact_inputs"
PER_CELL_SOURCE_IDENTITY: Final[str] = "per_cell"
EXECUTION_ONLY_PARAMETER_PREFIXES_V1: Final[tuple[str, ...]] = ("hapb_frag.",)


def source_identity_parameters_v1(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Drop execution-only controls from generation-zero source identity.

    ``hapb_frag.*`` force/reuse flags are applied at execute time. They must
    not fork otherwise-identical founder sources.
    """

    return {
        str(name): value
        for name, value in dict(parameters).items()
        if not str(name).startswith(EXECUTION_ONLY_PARAMETER_PREFIXES_V1)
    }


class SourceFactoryError(RuntimeError):
    """Campaign seed ownership or source generation is unsafe."""


@dataclass(frozen=True, slots=True)
class SourceGenerationRequestV2:
    """All inputs allowed to influence one generation-zero projection."""

    source_identity_sha256: str
    source_inputs: Mapping[str, Any]


SourceGeneratorV2 = Callable[[SourceGenerationRequestV2, Path], Path | None]


@dataclass(frozen=True, slots=True)
class PreparedSourceV2:
    cell_id: str
    source_identity_sha256: str
    source_inputs: Mapping[str, Any]
    cache_key: str
    projection_sha256: str
    projection_path: str
    cache_reused: bool
    sharing_mode: str


def _payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise SourceFactoryError("plan must be a CampaignPlanV1 or mapping")
    return dict(plan)


def prepare_campaign_sources(
    plan: CampaignPlanV1 | Mapping[str, Any],
    *,
    cache_root: str | Path,
    materialization_root: str | Path,
    source_contract: Mapping[str, Any],
    generator: SourceGeneratorV2,
    sharing_mode: str = DISTINCT_PER_CELL,
    source_identity_scope: str = EXACT_INPUT_SOURCE_IDENTITY,
) -> dict[str, Any]:
    """Prepare one cache entry per exact full source identity and bind every cell."""

    if sharing_mode not in {DISTINCT_PER_CELL, SHARED_BY_EXACT_SOURCE_IDENTITY}:
        raise SourceFactoryError("unsupported source sharing mode")
    if source_identity_scope not in {
        EXACT_INPUT_SOURCE_IDENTITY,
        PER_CELL_SOURCE_IDENTITY,
    }:
        raise SourceFactoryError("unsupported source identity scope")
    if (
        source_identity_scope == PER_CELL_SOURCE_IDENTITY
        and sharing_mode != DISTINCT_PER_CELL
    ):
        raise SourceFactoryError(
            "per-cell source identity requires distinct-per-cell materialization"
        )
    if not callable(generator):
        raise SourceFactoryError("generator must be callable")
    payload = _payload(plan)
    cells = payload.get("cells")
    artifacts = payload.get("artifacts")
    if not isinstance(cells, list) or not isinstance(artifacts, Mapping):
        raise SourceFactoryError("compiled plan lacks cells or artifacts")
    relevant_hashes = {
        str(name): str(row.get("sha256"))
        for name, row in artifacts.items()
        if isinstance(row, Mapping)
    }
    if not relevant_hashes:
        raise SourceFactoryError("compiled plan has no artifact hashes")
    cache = Path(cache_root)
    target_root = Path(materialization_root)
    if not cache.is_absolute() or not target_root.is_absolute():
        raise SourceFactoryError("source cache and materialization roots must be absolute")

    entries: dict[str, SourceCacheEntry] = {}
    generated: set[str] = set()
    source_inputs_by_identity: dict[str, dict[str, Any]] = {}
    cell_sources: list[tuple[str, str, dict[str, Any]]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
            raise SourceFactoryError(f"compiled cell {index} is malformed")
        seeds = cell.get("seeds")
        parameters = cell.get("parameters")
        if not isinstance(seeds, Mapping) or not seeds:
            raise SourceFactoryError(f"compiled cell {cell['cell_id']} has no named seeds")
        if not isinstance(parameters, Mapping):
            raise SourceFactoryError(f"compiled cell {cell['cell_id']} has no parameters")
        source_inputs = {
            "seeds": {str(name): value for name, value in sorted(seeds.items())},
            "parameters": {
                str(name): value
                for name, value in sorted(
                    source_identity_parameters_v1(parameters).items()
                )
            },
        }
        if source_identity_scope == PER_CELL_SOURCE_IDENTITY:
            source_inputs["logical_cell_id"] = str(cell["cell_id"])
        try:
            source_identity = canonical_sha256(source_inputs)
        except ValueError as exc:
            raise SourceFactoryError(
                f"compiled cell {cell['cell_id']} source inputs are not canonical JSON data"
            ) from exc
        prior = source_inputs_by_identity.setdefault(source_identity, source_inputs)
        if prior != source_inputs:
            raise SourceFactoryError("source identity collision")
        cell_sources.append((str(cell["cell_id"]), source_identity, source_inputs))

    for source_identity in sorted(source_inputs_by_identity):
        source_inputs = source_inputs_by_identity[source_identity]
        entry = load_cached_source_projection(
            cache,
            source_contract=source_contract,
            source_inputs=source_inputs,
            relevant_hashes=relevant_hashes,
        )
        if entry is None:
            with tempfile.TemporaryDirectory(prefix="eiel-source-") as directory:
                destination = Path(directory) / "projection.bin"
                request = SourceGenerationRequestV2(
                    source_identity_sha256=source_identity,
                    source_inputs=source_inputs,
                )
                returned = generator(request, destination)
                source = destination if returned is None else Path(returned)
                if not source.is_file():
                    raise SourceFactoryError(
                        "source generator did not produce a regular file for exact "
                        f"source identity {source_identity}"
                    )
                entry = cache_source_projection(
                    source,
                    cache,
                    source_contract=source_contract,
                    source_inputs=source_inputs,
                    relevant_hashes=relevant_hashes,
                )
            generated.add(source_identity)
        entries[source_identity] = entry

    rows: list[PreparedSourceV2] = []
    for cell_id, source_identity, source_inputs in cell_sources:
        entry = entries[source_identity]
        if sharing_mode == DISTINCT_PER_CELL:
            destination = target_root / cell_id / "projection.bin"
            projection_path = materialize_cached_source(
                entry, destination, reuse_existing=True
            )
        else:
            projection_path = entry.payload_path
        rows.append(
            PreparedSourceV2(
                cell_id=cell_id,
                source_identity_sha256=source_identity,
                source_inputs=source_inputs,
                cache_key=entry.key,
                projection_sha256=entry.sha256,
                projection_path=str(projection_path),
                cache_reused=source_identity not in generated,
                sharing_mode=sharing_mode,
            )
        )
    receipt_payload = {
        "schema_version": SOURCE_FACTORY_SCHEMA_VERSION,
        "campaign_id": payload.get("campaign_id"),
        "plan_sha256": payload.get("plan_sha256"),
        "source_contract_sha256": canonical_sha256(source_contract),
        "sharing_mode": sharing_mode,
        "source_identity_scope": source_identity_scope,
        "unique_source_identities": len(entries),
        "rows": [asdict(row) for row in rows],
    }
    receipt_payload["receipt_sha256"] = canonical_sha256(receipt_payload)
    return receipt_payload


__all__ = [
    "DISTINCT_PER_CELL",
    "EXACT_INPUT_SOURCE_IDENTITY",
    "PER_CELL_SOURCE_IDENTITY",
    "SHARED_BY_EXACT_SOURCE_IDENTITY",
    "SOURCE_FACTORY_SCHEMA_VERSION",
    "PreparedSourceV2",
    "SourceGenerationRequestV2",
    "SourceFactoryError",
    "prepare_campaign_sources",
]
