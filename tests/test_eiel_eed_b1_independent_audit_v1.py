from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.scripts.eiel_eed_b1_independent_audit_v1 import audit


TREATMENTS = (
    "impact_on",
    "impact_off",
    "ecology_frozen",
    "phenotype_fixed",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_independent_audit_matches_positive_admissions_to_reduction(
    tmp_path: Path,
) -> None:
    items: list[dict[str, object]] = []
    reduced: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        for ordinal in range(18):
            cell_id = f"cell-{treatment}-{ordinal:02d}"
            batch_id = f"batch-{ordinal:04d}"
            batch_path = (
                tmp_path / "runs" / treatment / "batches" / batch_id / "manifest.json"
            )
            _write_json(
                batch_path,
                {
                    "batch_id": batch_id,
                    "cell_ids": [cell_id],
                    "attempts": 1,
                    "status": "completed",
                    "execution_route_sha256": f"route-{treatment}",
                },
            )
            digest = hashlib.sha256(batch_path.read_bytes()).hexdigest()
            item = {
                "logical_cell_id": cell_id,
                "execution_attempt_id": f"{batch_id}:attempt-1",
                "treatment": treatment,
                "founder_profile": "a_high",
                "seed": {"ecology": ordinal, "task": 29741},
                "source_commit": "source",
                "artifact_sha256": "artifact",
                "package_id": "package",
                "route_sha256": f"route-{treatment}",
                "qualification_key_sha256": f"qualification-{treatment}",
                "path": str(batch_path.resolve()),
                "file_size": batch_path.stat().st_size,
                "sha256": digest,
                "scientific_role": "primary_terminal_science_cell",
                "admission_status": "admitted",
            }
            items.append(item)
            reduced.append(
                {
                    "treatment": treatment,
                    "cell_id": cell_id,
                    "execution_attempt_id": item["execution_attempt_id"],
                    "admission_status": "admitted",
                    "admitted_manifest_path": item["path"],
                    "admitted_manifest_sha256": digest,
                }
            )
    admission_path = tmp_path / "admission.json"
    reduction_one_path = tmp_path / "reduction-one.json"
    reduction_two_path = tmp_path / "reduction-two.json"
    _write_json(
        admission_path,
        {
            "schema_version": (
                "eiel_eed_b1_positive_scientific_admission_manifest_v1"
            ),
            "status": "frozen",
            "scientific_cell_count": 72,
            "admitted_items": items,
        },
    )
    reduction = {
        "provenance_audit": {
            "status": "pass",
            "audited_primary_cells": 72,
            "duplicate_scientific_contributions": 0,
            "cells": reduced,
        }
    }
    _write_json(reduction_one_path, reduction)
    _write_json(reduction_two_path, reduction)

    result = audit(admission_path, reduction_one_path, reduction_two_path)

    assert result["status"] == "pass"
    assert result["admitted_scientific_cells"] == 72
    assert result["unadmitted_scientific_contributions"] == 0
