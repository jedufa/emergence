#!/usr/bin/env python3
"""Create the tracked canonical WRC Atlas and its correction provenance.

The source Atlas is preserved byte-for-byte.  This normalizer changes only
route-selection metadata that is necessary to request an already-declared
production route capability; it never makes the Atlas qualification authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_wrc_atlas_normalization_manifest_v1"
SOURCE_DIGEST = "46b65cdd2ff744bc0f2cda2acd05a032a0ec3f24bad478e9691e6092e984838f"
JUNGLE_PREFIX = "env_cfg.jungle_"
WDL_SOURCE_OVERLAYS = "[\"wdl\"]"
WDL_CANONICAL_OVERLAYS = "[\"wdl_delayed_hazard\"]"
EED_PARAMETER_NAMES = frozenset(
    {
        "env_cfg.food_spawn_per_step",
        "env_cfg.max_food",
        "env_cfg.patch_scent_gather_emit",
        "env_cfg.patch_waste_consume_emit",
    }
)
PROVING_TEST = (
    "tests/test_eiel_wrc_atlas_v3.py::"
    "test_canonical_route_metadata_requests_declared_capabilities_only"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_rows(rows: list[dict[str, str]], *, source_digest: str) -> list[dict[str, Any]]:
    """Normalize only the two route-context aliases proved by route contracts."""

    corrections: list[dict[str, Any]] = []
    for row in rows:
        parameter = row["Parameter"]
        old = row["World overlays JSON"]
        new = old
        reason = ""
        if parameter.startswith(JUNGLE_PREFIX) and old == "[]":
            new = "[\"jungle\"]"
            reason = (
                "Jungle primitive requires the declared Jungle route capability; "
                "the source Atlas omitted its route overlay."
            )
        elif parameter in EED_PARAMETER_NAMES and old == "[]":
            new = "[\"eed_resource_impact\"]"
            reason = (
                "This primitive is admitted by the EED resource-impact route; "
                "the source Atlas omitted its required route overlay."
            )
        elif parameter.startswith("env_cfg.wdl_") and old == WDL_SOURCE_OVERLAYS:
            new = WDL_CANONICAL_OVERLAYS
            reason = (
                "The source Atlas uses the WDL shorthand; the production route "
                "identifier is wdl_delayed_hazard."
            )
        if new == old:
            continue
        row["World overlays JSON"] = new
        corrections.append(
            {
                "row_id": row["ID"],
                "field": "World overlays JSON",
                "old_value": old,
                "new_value": new,
                "reason": reason,
                "proving_test": PROVING_TEST,
                "evidence": {
                    "kind": "authoritative_route_capability_declaration",
                    "reference": "engine/campaigns/production_control_path_v1.py",
                },
                "source_atlas_sha256": source_digest,
            }
        )
    return corrections


def normalize(source: Path, canonical: Path, manifest: Path) -> None:
    source_digest = _sha256(source)
    if source_digest != SOURCE_DIGEST:
        raise ValueError(
            "unexpected Matrix v3 source digest; do not normalize an unbound Atlas"
        )
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Matrix v3 source has no header")
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    if len(rows) != 95:
        raise ValueError("Matrix v3 source row count is not the frozen 95 rows")
    corrections = normalize_rows(rows, source_digest=source_digest)
    with canonical.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    canonical_digest = _sha256(canonical)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": source.name,
            "sha256": source_digest,
            "rows": len(rows),
        },
        "canonical": {
            "path": canonical.name,
            "sha256": canonical_digest,
            "rows": len(rows),
        },
        "normalization_authority": (
            "Authoritative production parameter contracts and route-capability "
            "declarations only; Atlas metadata requests those capabilities and "
            "cannot expand them."
        ),
        "corrections": corrections,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    normalize(args.source, args.canonical, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
