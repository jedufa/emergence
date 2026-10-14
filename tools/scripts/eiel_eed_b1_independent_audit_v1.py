from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class EedB1IndependentAuditError(ValueError):
    pass


TREATMENTS = (
    "impact_on",
    "impact_off",
    "ecology_frozen",
    "phenotype_fixed",
)
EXCLUDED_PATH_COMPONENTS = {
    "comparison_recovery_archive",
    "batch_attempt_recovery_archive",
}


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EedB1IndependentAuditError(
            f"cannot read {name}: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise EedB1IndependentAuditError(f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    admission_path: Path,
    reduction_one_path: Path,
    reduction_two_path: Path,
) -> dict[str, Any]:
    admission = _read_object(admission_path, name="positive admission manifest")
    reduction_one = _read_object(reduction_one_path, name="first reduction")
    _read_object(reduction_two_path, name="second reduction")
    if reduction_one_path.read_bytes() != reduction_two_path.read_bytes():
        raise EedB1IndependentAuditError("reducer outputs are not byte-identical")
    items = admission.get("admitted_items")
    if (
        admission.get("schema_version")
        != "eiel_eed_b1_positive_scientific_admission_manifest_v1"
        or admission.get("status") != "frozen"
        or admission.get("scientific_cell_count") != 72
        or not isinstance(items, list)
        or len(items) != 72
    ):
        raise EedB1IndependentAuditError("positive admission manifest is invalid")
    admitted: dict[tuple[str, str], Mapping[str, Any]] = {}
    treatment_counts = {treatment: 0 for treatment in TREATMENTS}
    for item in items:
        if not isinstance(item, Mapping):
            raise EedB1IndependentAuditError("admission item is not an object")
        treatment = item.get("treatment")
        cell_id = item.get("logical_cell_id")
        if treatment not in treatment_counts or not isinstance(cell_id, str):
            raise EedB1IndependentAuditError("admission identity is invalid")
        identity = (str(treatment), cell_id)
        if identity in admitted:
            raise EedB1IndependentAuditError("duplicate admitted logical cell")
        path = Path(str(item.get("path"))).resolve()
        if any(part in EXCLUDED_PATH_COMPONENTS for part in path.parts):
            raise EedB1IndependentAuditError("recovery path was positively admitted")
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise EedB1IndependentAuditError(
                f"admitted path is unavailable: {path}"
            ) from exc
        if (
            item.get("admission_status") != "admitted"
            or item.get("scientific_role") != "primary_terminal_science_cell"
            or item.get("file_size") != file_size
            or item.get("sha256") != _sha256(path)
        ):
            raise EedB1IndependentAuditError("admitted item identity mismatch")
        batch = _read_object(path, name="admitted batch manifest")
        if (
            batch.get("status") != "completed"
            or batch.get("cell_ids") != [cell_id]
            or item.get("execution_attempt_id")
            != f"{batch.get('batch_id')}:attempt-{batch.get('attempts')}"
            or batch.get("execution_route_sha256") != item.get("route_sha256")
        ):
            raise EedB1IndependentAuditError("admitted batch binding mismatch")
        admitted[identity] = item
        treatment_counts[str(treatment)] += 1
    if treatment_counts != {treatment: 18 for treatment in TREATMENTS}:
        raise EedB1IndependentAuditError("treatment admission counts are not 18")
    provenance = reduction_one.get("provenance_audit")
    cells = provenance.get("cells") if isinstance(provenance, Mapping) else None
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("status") != "pass"
        or provenance.get("audited_primary_cells") != 72
        or provenance.get("duplicate_scientific_contributions") != 0
        or not isinstance(cells, list)
        or len(cells) != 72
    ):
        raise EedB1IndependentAuditError("reduction provenance audit is invalid")
    reduced: set[tuple[str, str]] = set()
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise EedB1IndependentAuditError("reduced cell audit is invalid")
        identity = (str(cell.get("treatment")), str(cell.get("cell_id")))
        admitted_item = admitted.get(identity)
        if admitted_item is None or identity in reduced:
            raise EedB1IndependentAuditError(
                "reduced cell is unadmitted or duplicated"
            )
        if (
            cell.get("admission_status") != "admitted"
            or cell.get("execution_attempt_id")
            != admitted_item.get("execution_attempt_id")
            or cell.get("admitted_manifest_path") != admitted_item.get("path")
            or cell.get("admitted_manifest_sha256") != admitted_item.get("sha256")
        ):
            raise EedB1IndependentAuditError(
                "reduced cell does not match positive admission"
            )
        reduced.add(identity)
    if reduced != set(admitted):
        raise EedB1IndependentAuditError(
            "positive admissions and reduced cells differ"
        )
    return {
        "schema_version": "eiel_eed_b1_independent_integrity_audit_v1",
        "status": "pass",
        "positive_admission_manifest": {
            "path": str(admission_path),
            "sha256": _sha256(admission_path),
        },
        "reducer_outputs": [
            {"path": str(reduction_one_path), "sha256": _sha256(reduction_one_path)},
            {"path": str(reduction_two_path), "sha256": _sha256(reduction_two_path)},
        ],
        "byte_identical_reducer_outputs": True,
        "admitted_scientific_cells": 72,
        "audited_reduced_cells": 72,
        "duplicate_logical_cells": 0,
        "unadmitted_scientific_contributions": 0,
        "recovery_or_superseded_paths_admitted": 0,
        "treatment_counts": treatment_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently audit frozen EED B1 admissions and reduction"
    )
    parser.add_argument("--admission-manifest", required=True, type=Path)
    parser.add_argument("--reduction-one", required=True, type=Path)
    parser.add_argument("--reduction-two", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.admission_manifest.resolve(),
        args.reduction_one.resolve(),
        args.reduction_two.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output": str(args.output), "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
