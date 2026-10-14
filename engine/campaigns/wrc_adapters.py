"""Typed, non-executing planning adapters for the authoritative WRC Atlas.

This module deliberately distinguishes the three launcher classes recorded in
the tracked Atlas.  It creates deterministic planning identities only: a
later execution package must select a production route and establish its own
qualification evidence.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_sha256
from .wrc_atlas_v3 import ATLAS_CANONICAL_FILENAME, ATLAS_MANIFEST_FILENAME, ATLAS_SOURCE_FILENAME


class WRCAtlasAdapterError(ValueError):
    """The authoritative Atlas cannot be converted into a typed plan."""


GENERIC_RUNTIME = "GENERIC_RUNTIME"
SPECIALIZED_CAMPAIGN = "SPECIALIZED_CAMPAIGN"
CONSTRUCTION_CONTEXT = "CONSTRUCTION_CONTEXT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _string_tuple(value: str, *, field: str, row_id: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has malformed {field}") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) or (isinstance(item, int) and not isinstance(item, bool))
        for item in parsed
    ):
        raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has malformed {field}")
    return tuple(str(item) for item in parsed)


def _json_mapping(value: str, *, field: str, row_id: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has malformed {field}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has malformed {field}")
    return parsed


def _assertion_tuple(value: str, *, row_id: str) -> tuple[dict[str, Any], ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has malformed preflight assertions") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) and all(isinstance(key, str) for key in item)
        for item in parsed
    ):
        raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has malformed preflight assertions")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class AtlasRowV1:
    row_id: str
    parameter_name: str
    launcher_profile: str
    world_base: str
    world_overlays: tuple[str, ...]
    levels: tuple[str, ...]
    seeds: tuple[str, ...]
    expected_metrics: str
    preflight_mode: str
    required_gates: str
    required_overrides: Mapping[str, Any]
    activation_preconditions: str
    activation_overrides: Mapping[str, Any]
    preflight_assertions: tuple[Mapping[str, Any], ...]
    execution_status: str
    notes: str
    launcher_schema: str


@dataclass(frozen=True, slots=True)
class NormalizedAtlasCellV1:
    row_id: str
    adapter_kind: str
    parameter_name: str
    level: str | None
    seed: str | None
    world_base: str
    world_overlays: tuple[str, ...]
    expected_metrics: str
    preflight_mode: str
    required_gates: str
    required_overrides: Mapping[str, Any]
    activation_preconditions: str
    activation_overrides: Mapping[str, Any]
    preflight_assertions: tuple[Mapping[str, Any], ...]
    execution_status: str
    notes: str
    launcher_schema: str

    def configuration_identity_payload(self) -> dict[str, Any]:
        """Return the deterministic planned configuration identity.

        This is deliberately a planning identity, rather than a claim that
        the configuration has been admitted or executed.
        """

        return {
            "schema_version": "eiel_wrc_atlas_configuration_identity_v1",
            "row_id": self.row_id,
            "adapter_kind": self.adapter_kind,
            "parameter_name": self.parameter_name,
            "level": self.level,
            "seed": self.seed,
            "world": {"base": self.world_base, "overlays": list(self.world_overlays)},
            "preflight_mode": self.preflight_mode,
            "required_gates": self.required_gates,
            "required_overrides": dict(self.required_overrides),
            "activation_preconditions": self.activation_preconditions,
            "activation_overrides": dict(self.activation_overrides),
            "preflight_assertions": [dict(item) for item in self.preflight_assertions],
            "launcher_schema": self.launcher_schema,
        }

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(self.configuration_identity_payload())

    def output_identity_payload(self) -> dict[str, Any]:
        """Return the planned output contract without inferring results."""

        return {
            "schema_version": "eiel_wrc_atlas_output_identity_v1",
            "configuration_sha256": self.configuration_sha256,
            "expected_metrics": self.expected_metrics,
            "execution_status": self.execution_status,
            "notes": self.notes,
        }

    @property
    def output_sha256(self) -> str:
        return canonical_sha256(self.output_identity_payload())

    def reducer_identity_payload(self) -> dict[str, Any]:
        """Bind the planned reducer input to the exact normalized output.

        The value names the WRC reducer protocol only; it intentionally does
        not manufacture a reducer configuration or assert a completed result.
        """

        return {
            "schema_version": "eiel_wrc_atlas_reducer_identity_v1",
            "output_sha256": self.output_sha256,
            "reducer_protocol": "eiel_wrc_reducer_v1",
        }

    @property
    def reducer_sha256(self) -> str:
        return canonical_sha256(self.reducer_identity_payload())

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_wrc_normalized_atlas_cell_v1",
            "row_id": self.row_id,
            "adapter_kind": self.adapter_kind,
            "parameter_name": self.parameter_name,
            "level": self.level,
            "seed": self.seed,
            "world": {"base": self.world_base, "overlays": list(self.world_overlays)},
            "expected_metrics": self.expected_metrics,
            "preflight_mode": self.preflight_mode,
            "configuration_sha256": self.configuration_sha256,
            "output_sha256": self.output_sha256,
            "reducer_sha256": self.reducer_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())


class _AtlasAdapter:
    kind: str
    launcher_profile: str

    def plan_cells(self, row: AtlasRowV1) -> tuple[NormalizedAtlasCellV1, ...]:
        if row.launcher_profile != self.launcher_profile:
            raise WRCAtlasAdapterError(
                f"Atlas row {row.row_id!r} cannot use {self.kind}: "
                f"expected {self.launcher_profile}, got {row.launcher_profile}"
            )
        if not row.levels:
            if self.kind != CONSTRUCTION_CONTEXT:
                raise WRCAtlasAdapterError(f"Atlas row {row.row_id!r} lacks scalar planning levels")
            # Context rows are explicit whole-world bundle requests, not scalar
            # injections.  Retain one non-runnable planning identity rather
            # than fabricating a level or silently omitting the row.
            return (NormalizedAtlasCellV1(
                row_id=row.row_id,
                adapter_kind=self.kind,
                parameter_name=row.parameter_name,
                level=None,
                seed=None,
                world_base=row.world_base,
                world_overlays=row.world_overlays,
                expected_metrics=row.expected_metrics,
                preflight_mode=row.preflight_mode,
                required_gates=row.required_gates,
                required_overrides=row.required_overrides,
                activation_preconditions=row.activation_preconditions,
                activation_overrides=row.activation_overrides,
                preflight_assertions=row.preflight_assertions,
                execution_status=row.execution_status,
                notes=row.notes,
                launcher_schema=row.launcher_schema,
            ),)
        return tuple(
            NormalizedAtlasCellV1(
                row_id=row.row_id,
                adapter_kind=self.kind,
                parameter_name=row.parameter_name,
                level=level,
                seed=seed,
                world_base=row.world_base,
                world_overlays=row.world_overlays,
                expected_metrics=row.expected_metrics,
                preflight_mode=row.preflight_mode,
                required_gates=row.required_gates,
                required_overrides=row.required_overrides,
                activation_preconditions=row.activation_preconditions,
                activation_overrides=row.activation_overrides,
                preflight_assertions=row.preflight_assertions,
                execution_status=row.execution_status,
                notes=row.notes,
                launcher_schema=row.launcher_schema,
            )
            for level in row.levels
            for seed in row.seeds
        )


class GenericRuntimeParameterAdapter(_AtlasAdapter):
    kind = GENERIC_RUNTIME
    launcher_profile = "STANDARD_CUDA"


class SpecializedCampaignAdapter(_AtlasAdapter):
    kind = SPECIALIZED_CAMPAIGN
    launcher_profile = "SPECIALIZED_MANUAL"


class ConstructionContextAdapter(_AtlasAdapter):
    kind = CONSTRUCTION_CONTEXT
    launcher_profile = "CONTEXT_MANUAL"


_ADAPTERS: Mapping[str, _AtlasAdapter] = {
    "STANDARD_CUDA": GenericRuntimeParameterAdapter(),
    "SPECIALIZED_MANUAL": SpecializedCampaignAdapter(),
    "CONTEXT_MANUAL": ConstructionContextAdapter(),
}


def adapter_for_row_v1(row: AtlasRowV1) -> _AtlasAdapter:
    """Select exactly one typed planner; manual rows never use generic injection."""

    try:
        return _ADAPTERS[row.launcher_profile]
    except KeyError as exc:
        raise WRCAtlasAdapterError(
            f"Atlas row {row.row_id!r} has unsupported launcher profile {row.launcher_profile!r}"
        ) from exc


def load_authoritative_atlas_rows_v1(registry_dir: str | Path) -> tuple[AtlasRowV1, ...]:
    """Load every canonical Atlas row after verifying the source/manifest pair."""

    root = Path(registry_dir)
    source = root / ATLAS_SOURCE_FILENAME
    canonical = root / ATLAS_CANONICAL_FILENAME
    manifest_path = root / ATLAS_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WRCAtlasAdapterError("WRC Atlas normalization manifest is unavailable") from exc
    if not isinstance(manifest, Mapping):
        raise WRCAtlasAdapterError("WRC Atlas normalization manifest is malformed")
    source_record = manifest.get("source")
    canonical_record = manifest.get("canonical")
    if not isinstance(source_record, Mapping) or not isinstance(canonical_record, Mapping):
        raise WRCAtlasAdapterError("WRC Atlas normalization manifest is malformed")
    if source_record.get("sha256") != _sha256(source) or canonical_record.get("sha256") != _sha256(canonical):
        raise WRCAtlasAdapterError("WRC Atlas source or canonical digest does not match its manifest")

    with canonical.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[AtlasRowV1] = []
    for raw in raw_rows:
        row_id = str(raw.get("ID") or "")
        parameter_name = str(raw.get("Parameter") or "")
        launcher_profile = str(raw.get("Launcher profile") or "")
        if not row_id or not parameter_name or launcher_profile not in _ADAPTERS:
            raise WRCAtlasAdapterError(f"Atlas row {row_id!r} has unsupported identity or launcher profile")
        row = AtlasRowV1(
            row_id=row_id,
            parameter_name=parameter_name,
            launcher_profile=launcher_profile,
            world_base=str(raw.get("World base") or ""),
            world_overlays=_string_tuple(str(raw.get("World overlays JSON") or ""), field="world overlays", row_id=row_id),
            levels=_string_tuple(str(raw.get("Levels JSON") or ""), field="levels", row_id=row_id),
            seeds=_string_tuple(str(raw.get("Seeds JSON") or ""), field="seeds", row_id=row_id),
            expected_metrics=str(raw.get("Primary response metrics") or ""),
            preflight_mode=str(raw.get("Preflight mode") or ""),
            required_gates=str(raw.get("Required gates") or ""),
            required_overrides=_json_mapping(
                str(raw.get("Required overrides JSON") or ""),
                field="required overrides",
                row_id=row_id,
            ),
            activation_preconditions=str(raw.get("Activation preconditions") or ""),
            activation_overrides=_json_mapping(
                str(raw.get("Activation overrides JSON") or ""),
                field="activation overrides",
                row_id=row_id,
            ),
            preflight_assertions=_assertion_tuple(
                str(raw.get("Preflight assertions JSON") or ""), row_id=row_id
            ),
            execution_status=str(raw.get("Execution status") or ""),
            notes=str(raw.get("Notes") or ""),
            launcher_schema=str(raw.get("Launcher schema") or ""),
        )
        if not row.world_base or not row.seeds or (
            not row.levels and launcher_profile != "CONTEXT_MANUAL"
        ):
            raise WRCAtlasAdapterError(f"Atlas row {row_id!r} lacks planning values")
        rows.append(row)
    if len(rows) != 95 or len({row.row_id for row in rows}) != len(rows):
        raise WRCAtlasAdapterError("authoritative WRC Atlas must contain exactly 95 unique rows")
    return tuple(rows)


def plan_authoritative_atlas_v1(registry_dir: str | Path) -> tuple[NormalizedAtlasCellV1, ...]:
    """Return stable, non-executable cell plans for all authoritative rows."""

    return tuple(
        cell
        for row in load_authoritative_atlas_rows_v1(registry_dir)
        for cell in adapter_for_row_v1(row).plan_cells(row)
    )


__all__ = [
    "AtlasRowV1",
    "CONSTRUCTION_CONTEXT",
    "ConstructionContextAdapter",
    "GENERIC_RUNTIME",
    "GenericRuntimeParameterAdapter",
    "NormalizedAtlasCellV1",
    "SPECIALIZED_CAMPAIGN",
    "SpecializedCampaignAdapter",
    "WRCAtlasAdapterError",
    "adapter_for_row_v1",
    "load_authoritative_atlas_rows_v1",
    "plan_authoritative_atlas_v1",
]
