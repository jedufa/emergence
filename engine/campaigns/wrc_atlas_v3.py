"""Read-only WRC Atlas V3 campaign-request metadata.

The Atlas is not qualification authority.  These helpers only validate that a
canonical Atlas row requests a capability already declared by the production
parameter contracts and route-capability declarations.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import RuntimeProfileV1
from .wrc_variability import route_scope_ids_v1


ATLAS_SOURCE_FILENAME = "EIEL_WRC_Campaign_Matrix_v3_source.csv"
ATLAS_CANONICAL_FILENAME = "EIEL_WRC_Campaign_Matrix_v3_canonical.csv"
ATLAS_MANIFEST_FILENAME = "EIEL_WRC_Campaign_Matrix_v3_normalization_manifest.json"
V1_CANDIDATE_FIRST_ID = "P011"
V1_CANDIDATE_LAST_ID = "P083"


class WRCAtlasError(ValueError):
    """The tracked Atlas is malformed or requests an undeclared capability."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class WRCAtlasRouteRequestV3:
    row_id: str
    parameter_name: str
    world_base: str
    world_overlays: tuple[str, ...]

    def campaign_request_metadata(self) -> dict[str, Any]:
        """Return descriptive request metadata; never a qualification key."""

        return {
            "atlas_row_id": self.row_id,
            "parameter_name": self.parameter_name,
            "world": {
                "base": self.world_base,
                "overlays": list(self.world_overlays),
            },
        }


@dataclass(frozen=True, slots=True)
class WRCAtlasRowV3:
    """One manifest-bound Atlas row without granting execution authority."""

    row_id: str
    parameter_name: str
    launcher_profile: str
    world_base: str
    world_overlays: tuple[str, ...]
    fields: Mapping[str, str]


def load_wrc_atlas_rows_v3(registry_dir: str | Path) -> tuple[WRCAtlasRowV3, ...]:
    """Load every canonical Atlas row after validating source/manifest identity."""

    root = Path(registry_dir)
    source = root / ATLAS_SOURCE_FILENAME
    canonical = root / ATLAS_CANONICAL_FILENAME
    manifest_path = root / ATLAS_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WRCAtlasError("WRC Atlas normalization manifest is unavailable") from exc
    if not isinstance(manifest, Mapping):
        raise WRCAtlasError("WRC Atlas normalization manifest is malformed")
    source_row = manifest.get("source")
    canonical_row = manifest.get("canonical")
    if not isinstance(source_row, Mapping) or not isinstance(canonical_row, Mapping):
        raise WRCAtlasError("WRC Atlas normalization manifest is malformed")
    if source_row.get("sha256") != _sha256(source):
        raise WRCAtlasError("WRC Atlas source digest does not match its manifest")
    if canonical_row.get("sha256") != _sha256(canonical):
        raise WRCAtlasError("WRC Atlas canonical digest does not match its manifest")
    with canonical.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[WRCAtlasRowV3] = []
    for raw in raw_rows:
        row_id = raw.get("ID", "")
        parameter_name = raw.get("Parameter", "")
        launcher_profile = raw.get("Launcher profile", "")
        world_base = raw.get("World base", "")
        try:
            overlays = tuple(json.loads(raw.get("World overlays JSON", "")))
        except json.JSONDecodeError as exc:
            raise WRCAtlasError(f"Atlas row {row_id!r} has invalid overlays") from exc
        if (
            not row_id
            or not parameter_name
            or not launcher_profile
            or not all(isinstance(item, str) for item in overlays)
        ):
            raise WRCAtlasError(f"Atlas row {row_id!r} is malformed")
        rows.append(
            WRCAtlasRowV3(
                row_id=row_id,
                parameter_name=parameter_name,
                launcher_profile=launcher_profile,
                world_base=world_base,
                world_overlays=overlays,
                fields=dict(raw),
            )
        )
    if len(rows) != 95:
        raise WRCAtlasError("WRC Atlas V3 must contain exactly 95 rows")
    return tuple(rows)


def load_wrc_v1_candidate_requests_v3(
    registry_dir: str | Path,
    *,
    profile: RuntimeProfileV1,
) -> tuple[WRCAtlasRouteRequestV3, ...]:
    """Load P011-P083 as capability candidates, not scientific completion.

    Current WRC evidence-completion status is deliberately absent here: that
    determination belongs to its evidence package, not the capability Atlas.
    """

    requests: list[WRCAtlasRouteRequestV3] = []
    for row in load_wrc_atlas_rows_v3(registry_dir):
        row_id = row.row_id
        if not V1_CANDIDATE_FIRST_ID <= row_id <= V1_CANDIDATE_LAST_ID:
            continue
        if row.launcher_profile != "STANDARD_CUDA":
            raise WRCAtlasError(f"Atlas candidate {row_id!r} is not STANDARD_CUDA")
        parameter_name = row.parameter_name
        contract = profile.parameter_contracts.get(parameter_name)
        if contract is None or contract.variability_semantics != "experiment_variable":
            raise WRCAtlasError(
                f"Atlas candidate {row_id!r} requests undeclared capability {parameter_name!r}"
            )
        try:
            overlays = row.world_overlays
        except (TypeError, ValueError) as exc:
            raise WRCAtlasError(f"Atlas candidate {row_id!r} has invalid overlays") from exc
        world_base = row.world_base
        scopes = route_scope_ids_v1(world_base, overlays)
        if not scopes or not set(scopes) & set(contract.variability_route_scopes):
            raise WRCAtlasError(
                f"Atlas candidate {row_id!r} requests a route not declared for {parameter_name!r}"
            )
        requests.append(
            WRCAtlasRouteRequestV3(
                row_id=row_id,
                parameter_name=parameter_name,
                world_base=world_base,
                world_overlays=overlays,
            )
        )
    if len(requests) != 73:
        raise WRCAtlasError("WRC Atlas V1 candidate cohort must contain P011-P083")
    return tuple(requests)


__all__ = [
    "ATLAS_CANONICAL_FILENAME",
    "ATLAS_MANIFEST_FILENAME",
    "ATLAS_SOURCE_FILENAME",
    "WRCAtlasError",
    "WRCAtlasRowV3",
    "WRCAtlasRouteRequestV3",
    "load_wrc_atlas_rows_v3",
    "load_wrc_v1_candidate_requests_v3",
]
