#!/usr/bin/env python3
"""Extend factory 200 to a larger physical B without reminting the leading 48.

CPU-only planning writes the merged receipt skeleton and the extra cell-id
list.  Minting the extra G0 projections is a separate QualPrep launch.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.perf.cuda_architecture_b_hapb_v1 import hapb_physical_cell_count_v1
from engine.perf.hapb_physical_extent_v1 import (
    CANONICAL_B48_V1,
    expand_campaign_spec_physical_cells_v1,
    flatten_plan_cell_ids_v1,
    leading_source_extent_cell_ids_v1,
)


SCHEMA_VERSION = "eiel_hapb_extended_source_factory_v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def plan_extended_source_factory_v1(
    *,
    spec_path: Path,
    base_source_root: Path,
    cells: int,
) -> dict[str, Any]:
    count = hapb_physical_cell_count_v1(cells)
    receipt = _load_json(base_source_root / "source_factory_receipt.json")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != CANONICAL_B48_V1:
        raise RuntimeError("base source factory is not exact B48")
    canonical_ids = [str(row.get("cell_id")) for row in rows if isinstance(row, Mapping)]
    if len(canonical_ids) != CANONICAL_B48_V1 or any(not cell_id for cell_id in canonical_ids):
        raise RuntimeError("base source factory cell identities are incomplete")
    spec = load_campaign_spec(spec_path)
    spec["campaign_id"] = f"{spec['campaign_id']}-hapb-ordinary-birth"
    spec["parameters"]["production_lifecycle.ordinary_births_enabled"] = True
    # Match factory 200 / H4 --generation-turnover so extra cell IDs stay
    # inside the same parameter envelope as the leading B48 extent.
    spec["parameters"]["production_lifecycle.lifecycle_mode"] = (
        "legacy_synchronized_truncation_v1"
    )
    spec["parameters"]["production_lifecycle.life_history_profile"] = (
        "legacy_inert_v1"
    )
    spec["parameters"]["production_lifecycle.turnover_cull_fraction"] = 0.5
    spec["parameters"]["production_lifecycle.turnover_parent_fraction"] = 1.0
    expanded = expand_campaign_spec_physical_cells_v1(spec, count)
    plan = compile_campaign(expanded).payload()
    extent = leading_source_extent_cell_ids_v1(
        canonical_b48_cell_ids=canonical_ids,
        expanded_cell_ids=flatten_plan_cell_ids_v1(plan),
        cells=count,
    )
    extras = extent[CANONICAL_B48_V1:]
    return {
        "schema_version": SCHEMA_VERSION,
        "physical_cells": count,
        "canonical_b48_cell_ids": list(canonical_ids),
        "extended_cell_ids": list(extent),
        "extra_cell_ids": list(extras),
        "extra_count": len(extras),
        "expanded_plan_sha256": plan.get("plan_sha256"),
        "base_source_root": str(base_source_root),
        "base_receipt_sha256": receipt.get("receipt_sha256"),
        "qualification_claim": False,
        "production_execution_admission": False,
    }


def materialize_leading_extent_v1(
    *,
    plan: Mapping[str, Any],
    base_source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Copy factory-200 projections into the new root; extras stay unminted."""

    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite immutable source root: {output_root}")
    output_root.mkdir(parents=True)
    dest_proj = output_root / "source_projections_v2"
    dest_proj.mkdir()
    base_receipt = _load_json(base_source_root / "source_factory_receipt.json")
    by_id = {
        str(row["cell_id"]): row
        for row in base_receipt["rows"]
        if isinstance(row, Mapping)
    }
    copied_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell_id in plan["extended_cell_ids"]:
        row = by_id.get(cell_id)
        if row is None:
            missing.append(cell_id)
            continue
        source = Path(str(row["projection_path"]))
        cell_dir = dest_proj / cell_id
        cell_dir.mkdir()
        dest = cell_dir / "projection.bin"
        shutil.copy2(source, dest)
        copied = dict(row)
        copied["projection_path"] = str(dest)
        copied_rows.append(copied)
    payload = {
        **dict(plan),
        "copied_canonical_rows": len(copied_rows),
        "unminted_extra_cell_ids": missing,
        "copied_rows": copied_rows,
    }
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    (output_root / "extended_source_factory_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def finalize_extended_source_factory_v1(
    *,
    plan_path: Path,
    extra_source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Merge copied B48 projections with minted extras into one factory receipt."""

    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite immutable source root: {output_root}")
    plan = _load_json(plan_path)
    extra_receipt = _load_json(extra_source_root / "source_factory_receipt.json")
    extra_rows = extra_receipt.get("rows")
    if not isinstance(extra_rows, list):
        raise RuntimeError("extra source factory receipt has no rows")
    extra_by_id = {
        str(row["cell_id"]): row
        for row in extra_rows
        if isinstance(row, Mapping)
    }
    output_root.mkdir(parents=True)
    dest_proj = output_root / "source_projections_v2"
    dest_proj.mkdir()
    merged: list[dict[str, Any]] = []
    for cell_id in plan["extended_cell_ids"]:
        row = None
        for copied in plan.get("copied_rows") or []:
            if isinstance(copied, Mapping) and copied.get("cell_id") == cell_id:
                row = copied
                break
        if row is None:
            row = extra_by_id.get(cell_id)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"missing source projection for {cell_id}")
        source = Path(str(row["projection_path"]))
        if not source.is_file():
            raise RuntimeError(f"source projection missing: {source}")
        cell_dir = dest_proj / cell_id
        cell_dir.mkdir()
        dest = cell_dir / "projection.bin"
        shutil.copy2(source, dest)
        merged_row = dict(row)
        merged_row["projection_path"] = str(dest)
        merged.append(merged_row)
    receipt = {
        "schema_version": "eiel_campaign_source_factory_v2",
        "campaign_id": "hapb-extended-physical-b-source-factory-v1",
        "rows": merged,
        "plan_sha256": plan.get("expanded_plan_sha256"),
        "source_contract_sha256": extra_receipt.get("source_contract_sha256"),
        "qualification_claim": False,
        "production_execution_admission": False,
        "physical_cells": plan["physical_cells"],
        "canonical_leading_extent": CANONICAL_B48_V1,
        "extra_count": plan["extra_count"],
    }
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    (output_root / "source_factory_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--base-source-root", type=Path)
    parser.add_argument("--cells", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--finalize-plan", type=Path)
    parser.add_argument("--extra-source-root", type=Path)
    parser.add_argument("--finalize-output-root", type=Path)
    args = parser.parse_args()
    if args.finalize_plan is not None:
        if args.extra_source_root is None or args.finalize_output_root is None:
            raise RuntimeError("finalize requires --extra-source-root and --finalize-output-root")
        receipt = finalize_extended_source_factory_v1(
            plan_path=args.finalize_plan.expanduser().resolve(),
            extra_source_root=args.extra_source_root.expanduser().resolve(),
            output_root=args.finalize_output_root.expanduser().resolve(),
        )
        print(json.dumps({
            "schema_version": receipt["schema_version"],
            "physical_cells": receipt["physical_cells"],
            "extra_count": receipt["extra_count"],
            "output_root": str(args.finalize_output_root.expanduser().resolve()),
            "receipt_sha256": receipt["receipt_sha256"],
            "qualification_claim": False,
            "production_execution_admission": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.spec is None or args.base_source_root is None or args.cells is None or args.output_root is None:
        raise RuntimeError("planning requires --spec --base-source-root --cells --output-root")
    plan = plan_extended_source_factory_v1(
        spec_path=args.spec.expanduser().resolve(),
        base_source_root=args.base_source_root.expanduser().resolve(),
        cells=args.cells,
    )
    materialize_leading_extent_v1(
        plan=plan,
        base_source_root=args.base_source_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "physical_cells": plan["physical_cells"],
        "extra_count": plan["extra_count"],
        "output_root": str(args.output_root.expanduser().resolve()),
        "qualification_claim": False,
        "production_execution_admission": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
