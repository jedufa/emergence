from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

import yaml

from tools.scripts.eiel_eed_b1_reducer_v1 import (
    ARTIFACT_SHA256,
    CAMPAIGN_IDS,
    CONTRACT_SHA256,
    PACKAGE_ID,
    ROUTES,
    SOURCE_COMMIT,
    TREATMENTS,
    reduce_evidence,
)


class EedB1IntegrityError(ValueError):
    pass


SEMANTIC_PAIRS = {
    "impact_on__impact_off": ("impact_on", "impact_off"),
    "impact_on__ecology_frozen": ("impact_on", "ecology_frozen"),
    "impact_on__phenotype_fixed": ("impact_on", "phenotype_fixed"),
    "impact_off__ecology_frozen": ("impact_off", "ecology_frozen"),
}
INTENDED_DIFFERENCES = {
    "impact_on__impact_off": {
        "parameters.eed.resource_impact_apply_enable",
    },
    "impact_on__ecology_frozen": {
        "parameters.eed.ecology_frozen",
    },
    "impact_on__phenotype_fixed": {
        "parameters.eed.phenotype_fixed",
    },
    "impact_off__ecology_frozen": {
        "parameters.eed.ecology_frozen",
        "parameters.eed.resource_impact_apply_enable",
    },
}
IDENTITY_DIFFERENCES = {
    "campaign_id",
    "description",
    "output.runtime_root",
    "artifacts.publication_receipt.path",
    "artifacts.qualification_receipt.path",
    "artifacts.reporting_receipt.path",
}
IMPACT_ARCHIVE_RELATIVE = Path(
    "10_b1_science_runs"
    "/impact_on"
    "/eed-b1-science-impact_on-g100"
    "/comparison_recovery_archive"
    "/resume-d7a167b3d364c8bb2b52-attempt1-zero-checkpoint-20260729T210148Z"
)
IMPACT_ARCHIVE_HASHES = {
    "manifest.json": "327f44a602852c4ea17316d350d0c473a615ec2ea03bfc0108e376ec4379909e",
    "uninterrupted_baseline_manifest.json": (
        "2bada4e64b0923dcdf6b9dc5ed6b84fa1c97755c2f693292d4d0612656b1a33a"
    ),
    "recovery_receipt.json": (
        "397bd1bcd6daebab4e3df2cf14801d7f0c68f9275bea035962f3c3fe4053ace6"
    ),
}
PHENOTYPE_RUN_RELATIVE = (
    Path("10_b1_science_runs")
    / "phenotype_fixed"
    / "eed-b1-science-phenotype_fixed-g100"
)
PHENOTYPE_ATTEMPT_ARCHIVE_NAME = (
    "batch-0001-attempt1-unpublished-g50-20260730T045813Z"
)
PHENOTYPE_SOURCE_PROJECTION_SHA256 = (
    "b3b57b47aa96b3fa349ae27e1fe5a46060dd24e037ba0e8dd89188d0da4f3371"
)
EXCLUDED_DISCOVERY_COMPONENTS = {
    "comparison_recovery_archive",
    "batch_attempt_recovery_archive",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EedB1IntegrityError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EedB1IntegrityError(f"{name} must be an object: {path}")
    return value


def _read_yaml(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise EedB1IntegrityError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EedB1IntegrityError(f"{name} must be an object: {path}")
    return value


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, prefix=path))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, prefix=f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def _csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or not any(value.strip() for value in rows[0]):
        raise EedB1IntegrityError(f"CSV lacks one non-empty header row: {path}")
    return sum(any(value.strip() for value in row) for row in rows[1:])


def _jsonl_nonempty_records(path: Path) -> int:
    return sum(
        bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
    )


FileClassifier = Callable[[Path, Path], dict[str, Any]]


def recursive_inventory(
    root: Path,
    *,
    classifier: FileClassifier,
) -> dict[str, Any]:
    if not root.is_dir():
        raise EedB1IntegrityError(f"inventory root is not a directory: {root}")
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        files.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                **classifier(relative, path),
            }
        )
    return {
        "schema_version": "eiel_eed_b1_recursive_inventory_v1",
        "root": str(root.resolve()),
        "file_count": len(files),
        "file_bytes": sum(row["size_bytes"] for row in files),
        "canonical_inventory_sha256": hashlib.sha256(
            _canonical_bytes(files)
        ).hexdigest(),
        "files": files,
    }


def _classify_common_setup(relative: Path, path: Path) -> dict[str, Any] | None:
    name = relative.as_posix()
    if name.endswith("/metrics.csv"):
        rows = _csv_data_rows(path)
        if rows:
            raise EedB1IntegrityError(f"metrics CSV has data rows: {path}")
        return {"classification": "setup_header_only_metrics_csv", "csv_data_rows": 0}
    if name.endswith("/metrics/lineage_summary.csv"):
        rows = _csv_data_rows(path)
        if rows:
            raise EedB1IntegrityError(f"lineage CSV has data rows: {path}")
        return {"classification": "setup_header_only_lineage_csv", "csv_data_rows": 0}
    if path.suffix == ".jsonl":
        records = _jsonl_nonempty_records(path)
        if records:
            raise EedB1IntegrityError(f"JSONL has decision records: {path}")
        return {
            "classification": "setup_empty_jsonl_surface",
            "jsonl_nonempty_records": 0,
        }
    if "/plans/" in f"/{name}":
        _read_json(path, name="setup plan")
        return {"classification": "setup_route_plan"}
    if name.endswith("/maps/tier3_regions_stats.json"):
        _read_json(path, name="setup map statistics")
        return {"classification": "setup_map_statistics"}
    if name.endswith("/maps/tier3_region_id.npy"):
        return {"classification": "setup_map_array"}
    if name.endswith("/chronicle/gen1_f1/gen1_f1_manifest.json"):
        _read_json(path, name="setup Chronicle manifest")
        return {"classification": "setup_chronicle_manifest"}
    return None


def classify_impact_archive_file(relative: Path, path: Path) -> dict[str, Any]:
    name = relative.as_posix()
    if name in IMPACT_ARCHIVE_HASHES:
        return {"classification": "preserved_original_manifest_or_receipt"}
    common = _classify_common_setup(relative, path)
    if common is not None:
        return common
    if path.stat().st_size:
        raise EedB1IntegrityError(f"unknown non-empty archive file: {path}")
    return {"classification": "unknown_empty_setup_file"}


def classify_phenotype_attempt_file(relative: Path, path: Path) -> dict[str, Any]:
    name = relative.as_posix()
    if name == "manifest.json":
        return {"classification": "preserved_original_batch_manifest"}
    common = _classify_common_setup(relative, path)
    if common is not None:
        return common
    if "/.bundle_gen000050.tmp-" in f"/{name}":
        if path.name in {
            "metadata.json",
            "world.json",
            "population.json",
            "rng_snapshot_v2.json",
            "control_plane.json",
            "runner_state.json",
            "resident_runtime_state_v2.json",
            "metrics_state.json",
            "native_checkpoint_bundle_coherence_report.json",
            "bundle_complete.json",
            "events_hash.txt",
            "bank_state.json",
        }:
            return {"classification": "unpublished_checkpoint_staging_payload"}
        if path.name in {
            "native_runtime_checkpoint_projection.json",
            "native_runtime_checkpoint_projection.json.gz",
        }:
            return {"classification": "unpublished_native_runtime_projection"}
        if "/weights/" in f"/{name}":
            return {"classification": "unpublished_checkpoint_weight_payload"}
        if path.stat().st_size:
            raise EedB1IntegrityError(
                f"unknown non-empty checkpoint staging file: {path}"
            )
        return {"classification": "unknown_empty_checkpoint_staging_file"}
    if path.stat().st_size:
        raise EedB1IntegrityError(f"unknown non-empty interrupted file: {path}")
    return {"classification": "unknown_empty_setup_file"}


def treatment_semantic_isolation(spec_root: Path) -> dict[str, Any]:
    specs = {
        treatment: _read_yaml(
            spec_root / f"eed-b1-science-{treatment}-g100.yaml",
            name=f"{treatment} spec",
        )
        for treatment in TREATMENTS
    }
    rows: dict[str, Any] = {}
    failure_count = 0
    for name, (left, right) in SEMANTIC_PAIRS.items():
        left_values = _flatten(specs[left])
        right_values = _flatten(specs[right])
        differences = []
        observed_intended: set[str] = set()
        for path in sorted(set(left_values) | set(right_values)):
            left_value = left_values.get(path)
            right_value = right_values.get(path)
            if left_value == right_value:
                continue
            if path in INTENDED_DIFFERENCES[name]:
                classification = "intended_causal_difference"
                observed_intended.add(path)
            elif path in IDENTITY_DIFFERENCES:
                classification = "required_identity_or_label_difference"
            else:
                classification = "forbidden_extra_difference"
                failure_count += 1
            differences.append(
                {
                    "path": path,
                    "left": left_value,
                    "right": right_value,
                    "classification": classification,
                }
            )
        missing = sorted(INTENDED_DIFFERENCES[name] - observed_intended)
        failure_count += len(missing)
        rows[name] = {
            "left_treatment": left,
            "right_treatment": right,
            "differences": differences,
            "missing_intended_differences": missing,
            "status": (
                "pass"
                if not missing
                and all(
                    row["classification"] != "forbidden_extra_difference"
                    for row in differences
                )
                else "fail"
            ),
            "interpretation": (
                "descriptive_control_contrast"
                if name == "impact_off__ecology_frozen"
                else "single_mechanic_isolation"
            ),
        }
    result = {
        "schema_version": "eiel_eed_b1_treatment_semantic_isolation_v1",
        "status": "pass" if failure_count == 0 else "fail",
        "forbidden_or_missing_difference_count": failure_count,
        "pairs": rows,
    }
    if result["status"] != "pass":
        raise EedB1IntegrityError("treatment semantic isolation failed")
    return result


def attempt_identity(owner_id: str, attempt: int) -> str:
    if not owner_id or type(attempt) is not int or attempt < 1:
        raise EedB1IntegrityError("attempt identity is invalid")
    return f"{owner_id}:attempt-{attempt}"


def audit_impact_archive(evidence_root: Path) -> dict[str, Any]:
    archive = evidence_root / IMPACT_ARCHIVE_RELATIVE
    for relative, expected in IMPACT_ARCHIVE_HASHES.items():
        if _sha256(archive / relative) != expected:
            raise EedB1IntegrityError(f"original impact archive changed: {relative}")
    inventory = recursive_inventory(
        archive,
        classifier=classify_impact_archive_file,
    )
    if inventory["file_count"] != 17 or inventory["file_bytes"] != 46711:
        raise EedB1IntegrityError("impact archive recursive inventory changed")

    parent_path = archive / "manifest.json"
    baseline_path = archive / "uninterrupted_baseline_manifest.json"
    receipt_path = archive / "recovery_receipt.json"
    parent = _read_json(parent_path, name="archived comparison manifest")
    baseline = _read_json(baseline_path, name="archived baseline manifest")
    receipt = _read_json(receipt_path, name="original recovery receipt")
    if receipt.get("parent_manifest_sha256") != _sha256(parent_path):
        raise EedB1IntegrityError("archived comparison manifest hash mismatch")
    if receipt.get("uninterrupted_baseline_manifest_sha256") != _sha256(
        baseline_path
    ):
        raise EedB1IntegrityError("archived baseline manifest hash mismatch")
    for label, row in (("parent", parent), ("baseline", baseline)):
        if (
            row.get("checkpoint") is not None
            or row.get("metrics_rows") != []
            or row.get("result") is not None
        ):
            raise EedB1IntegrityError(
                f"archived {label} contains admitted decision output"
            )
    if (
        receipt.get("persisted_non_manifest_file_count") != 0
        or receipt.get("scientific_contribution_count") != 0
        or receipt.get("live_process_present_at_recovery") is not False
        or receipt.get("attempt") != 1
    ):
        raise EedB1IntegrityError("original recovery receipt boundary changed")

    active_path = (
        evidence_root
        / "10_b1_science_runs"
        / "impact_on"
        / CAMPAIGN_IDS["impact_on"]
        / "comparisons"
        / str(parent.get("branch_id"))
        / "manifest.json"
    )
    active = _read_json(active_path, name="redispatched comparison manifest")
    active_attempt = int(active.get("attempts", 0))
    if (
        active.get("status") != "completed"
        or active_attempt <= int(receipt["attempt"])
        or active.get("checkpoint") is None
        or active.get("result") is None
    ):
        raise EedB1IntegrityError("redispatched comparison is not distinctly complete")
    archived_identity = attempt_identity(
        str(receipt.get("branch_id")), int(receipt["attempt"])
    )
    active_identity = attempt_identity(str(active.get("branch_id")), active_attempt)
    if archived_identity == active_identity:
        raise EedB1IntegrityError("archived and admitted attempt identities collide")
    return {
        "schema_version": "eiel_eed_b1_archive_zero_decision_output_audit_v1",
        "status": "pass",
        "procedural_classification": (
            "REPAIRABLE_EVIDENCE_ACCOUNTING_DEFECT_NO_SCIENTIFIC_CONTAMINATION"
        ),
        "disposition": "superseded_zero_decision_output_scaffolding_only_attempt",
        "archive_path": str(archive.resolve()),
        "original_receipt_preserved": True,
        "original_receipt_file_count_statement_accurate": False,
        "original_receipt_file_count_correction": {
            "stated_persisted_non_manifest_file_count": 0,
            "recursive_file_count": inventory["file_count"],
            "recursive_file_bytes": inventory["file_bytes"],
        },
        "zero_decision_output": True,
        "scientific_contribution_count": 0,
        "archived_attempt_identity": archived_identity,
        "redispatched_attempt_identity": active_identity,
        "archive_excluded_from_all_scientific_discovery": True,
        "inventory": inventory,
    }


def is_reducer_input_path(evidence_root: Path, path: Path) -> bool:
    del evidence_root
    candidate = path.resolve()
    if any(part in EXCLUDED_DISCOVERY_COMPONENTS for part in candidate.parts):
        return False
    return not any(part.startswith(".bundle_") and ".tmp-" in part for part in candidate.parts)


def public_checkpoint_admitted(
    manifest: Mapping[str, Any],
    public_bundle_path: Path,
) -> bool:
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return False
    rows = checkpoint.get("cell_bundles")
    if not isinstance(rows, list):
        return False
    expected = str(public_bundle_path.resolve())
    return (
        public_bundle_path.is_dir()
        and ".tmp-" not in public_bundle_path.name
        and any(
            isinstance(row, Mapping) and row.get("path") == expected for row in rows
        )
    )


def _strict_load_staged_bundle(path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    engine_root = project_root / "engine"
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    from simulation.checkpoint_bundle_v2 import (  # noqa: PLC0415
        ARCHITECTURE_B_BUNDLE_KIND,
        checkpoint_bundle_v2_load,
    )

    loaded = checkpoint_bundle_v2_load(
        str(path),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    metadata = loaded.get("metadata")
    if not isinstance(metadata, Mapping):
        raise EedB1IntegrityError("strict staged-bundle load lacks metadata")
    return {
        "schema_version": "eiel_eed_b1_staged_bundle_strict_load_v1",
        "status": "pass",
        "strict": True,
        "required_bundle_kind": ARCHITECTURE_B_BUNDLE_KIND,
        "generation_index": metadata.get("generation_index"),
        "checkpoint_authority_schema_version": metadata.get(
            "checkpoint_authority_schema_version"
        ),
        "source_projection_sha256": metadata.get("source_projection_sha256"),
    }


def audit_phenotype_attempt_one(evidence_root: Path) -> dict[str, Any]:
    run_root = evidence_root / PHENOTYPE_RUN_RELATIVE
    active_root = run_root / "batches" / "batch-0001"
    archive_root = (
        run_root
        / "batch_attempt_recovery_archive"
        / PHENOTYPE_ATTEMPT_ARCHIVE_NAME
    )
    attempt_root = archive_root if archive_root.is_dir() else active_root
    manifest_path = attempt_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_json(manifest_path, name="phenotype batch 0001 attempt 1")
    if (
        manifest.get("batch_id") != "batch-0001"
        or manifest.get("status") != "running"
        or manifest.get("attempts") != 1
        or manifest.get("checkpoint") is not None
        or manifest.get("result") is not None
        or manifest.get("metrics_rows") != []
    ):
        raise EedB1IntegrityError("phenotype attempt-1 public manifest is not exact")
    staged = sorted(
        path
        for path in attempt_root.rglob(".bundle_gen000050.tmp-*")
        if path.is_dir()
    )
    if len(staged) != 1:
        raise EedB1IntegrityError("phenotype attempt-1 staged G50 bundle is not unique")
    public_path = staged[0].parent / "bundle_gen000050"
    if public_path.exists() or public_checkpoint_admitted(manifest, public_path):
        raise EedB1IntegrityError("unpublished G50 bundle entered public authority")
    inventory = recursive_inventory(
        attempt_root,
        classifier=classify_phenotype_attempt_file,
    )
    strict_load = _strict_load_staged_bundle(staged[0])
    if (
        strict_load.get("generation_index") != 50
        or strict_load.get("source_projection_sha256")
        != PHENOTYPE_SOURCE_PROJECTION_SHA256
    ):
        raise EedB1IntegrityError("staged G50 checkpoint identity mismatch")
    return {
        "schema_version": "eiel_eed_b1_phenotype_attempt_one_audit_v1",
        "status": "pass",
        "procedural_classification": (
            "RECOVERABLE_UNPUBLISHED_CHECKPOINT_STAGING_INTERRUPTION"
        ),
        "disposition": "superseded_unpublished_coherent_g50_transaction_staging_attempt",
        "attempt_identity": attempt_identity("batch-0001", 1),
        "attempt_root": str(attempt_root.resolve()),
        "original_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "public_checkpoint_path": str(public_path.resolve()),
        "public_checkpoint_path_absent": True,
        "public_checkpoint_admitted": False,
        "strict_load_success_does_not_imply_public_admission": True,
        "source_projection_sha256": PHENOTYPE_SOURCE_PROJECTION_SHA256,
        "zero_populated_public_metric_rows": True,
        "zero_jsonl_decision_records": True,
        "reducer_contribution_count": 0,
        "strict_load": strict_load,
        "inventory": inventory,
    }


def _classification_csv(inventory: Mapping[str, Any]) -> bytes:
    fields = [
        "relative_path",
        "size_bytes",
        "sha256",
        "classification",
        "csv_data_rows",
        "jsonl_nonempty_records",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in inventory.get("files", []):
        if isinstance(row, Mapping):
            writer.writerow(dict(row))
    return stream.getvalue().encode("utf-8")


def write_impact_archive_audit(
    evidence_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = audit_impact_archive(evidence_root)
    inventory = audit["inventory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "archive_recursive_inventory.json"
    classification_path = output_dir / "archive_file_classification.csv"
    audit_path = output_dir / "archive_zero_decision_output_audit.json"
    exclusion_path = output_dir / "archive_discovery_exclusion_receipt.json"
    reclassification_path = output_dir / "archive_reclassification_receipt.json"
    _atomic_write(inventory_path, _pretty_bytes(inventory))
    _atomic_write(classification_path, _classification_csv(inventory))
    _atomic_write(audit_path, _pretty_bytes(audit))
    exclusion = {
        "schema_version": "eiel_eed_b1_archive_discovery_exclusion_v1",
        "status": "pass",
        "excluded_root": audit["archive_path"],
        "excluded_surfaces": [
            "primary_batch_discovery",
            "comparison_discovery",
            "reporting_parity_discovery",
            "provenance_audit",
            "reducer_input_manifest",
            "evidence_decision_layer",
            "terminal_report_aggregation",
        ],
        "archive_paths_in_reducer_inputs": 0,
    }
    _atomic_write(exclusion_path, _pretty_bytes(exclusion))
    reclassification = {
        "schema_version": "eiel_eed_b1_archive_reclassification_v1",
        "status": "pass",
        "original_receipt_preserved": True,
        "original_receipt_hash": IMPACT_ARCHIVE_HASHES["recovery_receipt.json"],
        "original_file_count_statement_accurate": False,
        "supplemental_correction": True,
        "supersedes_original_file_count_claim_only": True,
        "corrected_recursive_file_count": 17,
        "corrected_total_bytes": 46711,
        "no_admitted_checkpoint_result_metric_or_scientific_contribution": True,
        "every_file_recursively_inventoried_hashed_and_classified": True,
        "scientific_admission": False,
        "archive_excluded_from_discovery_and_reducer_surfaces": True,
        "corrected_disposition": audit["disposition"],
        "canonical_inventory_sha256": inventory["canonical_inventory_sha256"],
        "audit_sha256": _sha256(audit_path),
        "exclusion_receipt_sha256": _sha256(exclusion_path),
    }
    _atomic_write(reclassification_path, _pretty_bytes(reclassification))
    return {
        "status": "pass",
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "classification_path": str(classification_path),
        "classification_sha256": _sha256(classification_path),
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "exclusion_path": str(exclusion_path),
        "exclusion_sha256": _sha256(exclusion_path),
        "reclassification_path": str(reclassification_path),
        "reclassification_sha256": _sha256(reclassification_path),
    }


def write_phenotype_attempt_audit(
    evidence_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = audit_phenotype_attempt_one(evidence_root)
    inventory = audit["inventory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "phenotype_batch0001_attempt1_inventory.json"
    classification_path = (
        output_dir / "phenotype_batch0001_attempt1_file_classification.csv"
    )
    strict_load_path = (
        output_dir / "phenotype_batch0001_attempt1_strict_load_receipt.json"
    )
    audit_path = output_dir / "phenotype_batch0001_attempt1_audit.json"
    exclusion_path = (
        output_dir / "phenotype_batch0001_attempt1_exclusion_receipt.json"
    )
    _atomic_write(inventory_path, _pretty_bytes(inventory))
    _atomic_write(classification_path, _classification_csv(inventory))
    _atomic_write(strict_load_path, _pretty_bytes(audit["strict_load"]))
    _atomic_write(audit_path, _pretty_bytes(audit))
    exclusion = {
        "schema_version": "eiel_eed_b1_phenotype_attempt_exclusion_v1",
        "status": "pass",
        "attempt_identity": audit["attempt_identity"],
        "excluded_root": audit["attempt_root"],
        "excluded_from": [
            "primary_batch_discovery",
            "checkpoint_catalog_discovery",
            "campaign_resume_discovery",
            "comparison_discovery",
            "reporting_parity_discovery",
            "provenance_audit",
            "reducer_input_manifest",
            "evidence_decision_layer",
            "terminal_report_aggregation",
        ],
        "scientific_contribution_count": 0,
    }
    _atomic_write(exclusion_path, _pretty_bytes(exclusion))
    return {
        "status": "pass",
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "classification_path": str(classification_path),
        "classification_sha256": _sha256(classification_path),
        "strict_load_path": str(strict_load_path),
        "strict_load_sha256": _sha256(strict_load_path),
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "exclusion_path": str(exclusion_path),
        "exclusion_sha256": _sha256(exclusion_path),
    }


def preserve_phenotype_attempt_one(
    evidence_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    run_root = evidence_root / PHENOTYPE_RUN_RELATIVE
    active_root = run_root / "batches" / "batch-0001"
    archive_root = (
        run_root
        / "batch_attempt_recovery_archive"
        / PHENOTYPE_ATTEMPT_ARCHIVE_NAME
    )
    if archive_root.exists():
        raise EedB1IntegrityError("phenotype attempt-1 archive already exists")
    audit = audit_phenotype_attempt_one(evidence_root)
    original_manifest = _read_json(
        active_root / "manifest.json",
        name="phenotype batch 0001 attempt 1",
    )
    original_inventory_hash = audit["inventory"]["canonical_inventory_sha256"]
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    active_root.rename(archive_root)
    archived_inventory = recursive_inventory(
        archive_root,
        classifier=classify_phenotype_attempt_file,
    )
    if archived_inventory["canonical_inventory_sha256"] != original_inventory_hash:
        raise EedB1IntegrityError("phenotype attempt changed during preservation")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "phenotype_batch0001_attempt1_inventory.json"
    classification_path = (
        output_dir / "phenotype_batch0001_attempt1_file_classification.csv"
    )
    strict_load_path = (
        output_dir / "phenotype_batch0001_attempt1_strict_load_receipt.json"
    )
    exclusion_path = (
        output_dir / "phenotype_batch0001_attempt1_exclusion_receipt.json"
    )
    supersession_path = (
        output_dir / "phenotype_batch0001_attempt1_supersession_receipt.json"
    )
    audit["attempt_root"] = str(archive_root.resolve())
    audit["inventory"] = archived_inventory
    _atomic_write(inventory_path, _pretty_bytes(archived_inventory))
    _atomic_write(classification_path, _classification_csv(archived_inventory))
    _atomic_write(strict_load_path, _pretty_bytes(audit["strict_load"]))
    exclusion = {
        "schema_version": "eiel_eed_b1_phenotype_attempt_exclusion_v1",
        "status": "pass",
        "attempt_identity": attempt_identity("batch-0001", 1),
        "excluded_root": str(archive_root.resolve()),
        "excluded_from": [
            "primary_batch_discovery",
            "checkpoint_catalog_discovery",
            "campaign_resume_discovery",
            "comparison_discovery",
            "reporting_parity_discovery",
            "provenance_audit",
            "reducer_input_manifest",
            "evidence_decision_layer",
            "terminal_report_aggregation",
        ],
        "scientific_contribution_count": 0,
    }
    _atomic_write(exclusion_path, _pretty_bytes(exclusion))
    supersession = {
        **audit,
        "archive_inventory_sha256": _sha256(inventory_path),
        "classification_sha256": _sha256(classification_path),
        "strict_load_receipt_sha256": _sha256(strict_load_path),
        "exclusion_receipt_sha256": _sha256(exclusion_path),
        "next_attempt_identity": attempt_identity("batch-0001", 2),
        "fresh_redispatch_start_generation": 0,
        "reuse_attempt_one_rng_or_checkpoint": False,
    }
    _atomic_write(supersession_path, _pretty_bytes(supersession))

    replacement = {
        **original_manifest,
        "status": "pending",
        "attempts": 1,
        "checkpoint": None,
        "metrics_rows": [],
        "result": None,
        "error": None,
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "superseded_attempts": [
            {
                "attempt_identity": attempt_identity("batch-0001", 1),
                "archive_path": str(archive_root.resolve()),
                "receipt_path": str(supersession_path.resolve()),
            }
        ],
        "next_attempt_identity": attempt_identity("batch-0001", 2),
    }
    _atomic_write(active_root / "manifest.json", _pretty_bytes(replacement))
    return {
        "status": "pass",
        "archive_root": str(archive_root),
        "archive_inventory_sha256": original_inventory_hash,
        "supersession_receipt": str(supersession_path),
        "supersession_receipt_sha256": _sha256(supersession_path),
        "fresh_attempt_manifest": str(active_root / "manifest.json"),
        "fresh_attempt_manifest_sha256": _sha256(active_root / "manifest.json"),
        "next_attempt_identity": attempt_identity("batch-0001", 2),
    }


def discover_reducer_inputs(evidence_root: Path, spec_root: Path) -> list[Path]:
    paths = {
        evidence_root
        / "08_b1_gate_and_contract"
        / "EIEL_EED_B1_Reducer_Contract_V1_2026_07_29.md"
    }
    for treatment in TREATMENTS:
        paths.add(spec_root / f"eed-b1-science-{treatment}-g100.yaml")
        run_root = (
            evidence_root
            / "10_b1_science_runs"
            / treatment
            / CAMPAIGN_IDS[treatment]
        )
        plan_path = run_root / "execution_plan.json"
        campaign_path = run_root / "campaign_manifest.json"
        comparisons_path = run_root / "comparisons_manifest.json"
        reporting_path = run_root / "reporting_parity_receipt.json"
        paths.update((plan_path, campaign_path, comparisons_path, reporting_path))
        plan = _read_json(plan_path, name=f"{treatment} execution plan")
        batches = plan.get("physical_batches")
        if not isinstance(batches, list):
            raise EedB1IntegrityError(f"{treatment} plan has no physical batches")
        for batch in batches:
            if not isinstance(batch, Mapping) or not isinstance(
                batch.get("batch_id"), str
            ):
                raise EedB1IntegrityError(f"{treatment} physical batch is invalid")
            paths.add(
                run_root / "batches" / str(batch["batch_id"]) / "manifest.json"
            )
        artifacts = plan.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise EedB1IntegrityError(f"{treatment} plan has no artifacts")
        for artifact in artifacts.values():
            if isinstance(artifact, Mapping) and isinstance(artifact.get("path"), str):
                paths.add(Path(str(artifact["path"])))
        comparisons = _read_json(
            comparisons_path,
            name=f"{treatment} comparisons manifest",
        )
        branches = comparisons.get("branches")
        if not isinstance(branches, list):
            raise EedB1IntegrityError(f"{treatment} comparisons have no branches")
        for branch in branches:
            if not isinstance(branch, Mapping) or not isinstance(
                branch.get("manifest_path"), str
            ):
                raise EedB1IntegrityError(f"{treatment} comparison branch is invalid")
            paths.add(Path(str(branch["manifest_path"])))
    resolved = sorted(path.resolve() for path in paths)
    excluded = [
        path for path in resolved if not is_reducer_input_path(evidence_root, path)
    ]
    if excluded:
        raise EedB1IntegrityError(f"excluded attempt entered reducer inputs: {excluded}")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise EedB1IntegrityError(f"reducer input files are missing: {missing}")
    return resolved


def freeze_input_manifest(
    evidence_root: Path,
    spec_root: Path,
    *,
    reducer_source: Path,
) -> dict[str, Any]:
    contract_path = (
        evidence_root
        / "08_b1_gate_and_contract"
        / "EIEL_EED_B1_Reducer_Contract_V1_2026_07_29.md"
    )
    if _sha256(contract_path) != CONTRACT_SHA256:
        raise EedB1IntegrityError("frozen reducer contract hash mismatch")
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(reducer_source)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    inputs = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "included_in_reduction": True,
        }
        for path in discover_reducer_inputs(evidence_root, spec_root)
    ]
    return {
        "schema_version": "eiel_eed_b1_frozen_input_manifest_v1",
        "reducer_source": {
            "path": str(reducer_source.resolve()),
            "sha256": _sha256(reducer_source),
            "implementation_commit": source_commit,
        },
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": CONTRACT_SHA256,
        },
        "included_file_count": len(inputs),
        "inputs": inputs,
    }


def freeze_positive_admission_manifest(evidence_root: Path) -> dict[str, Any]:
    admitted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for treatment in TREATMENTS:
        run_root = (
            evidence_root
            / "10_b1_science_runs"
            / treatment
            / CAMPAIGN_IDS[treatment]
        )
        plan = _read_json(
            run_root / "execution_plan.json",
            name=f"{treatment} execution plan",
        )
        qualification_key = plan.get("qualification_key_sha256")
        if not isinstance(qualification_key, str):
            raise EedB1IntegrityError(
                f"{treatment} qualification key is unavailable"
            )
        cells = plan.get("cells")
        batches = plan.get("physical_batches")
        if not isinstance(cells, list) or not isinstance(batches, list):
            raise EedB1IntegrityError(f"{treatment} plan matrix is invalid")
        cell_by_id = {
            row.get("cell_id"): row for row in cells if isinstance(row, Mapping)
        }
        if len(cell_by_id) != 18 or len(batches) != 18:
            raise EedB1IntegrityError(f"{treatment} matrix is not 18 physical B1 cells")
        for batch_row in batches:
            if not isinstance(batch_row, Mapping):
                raise EedB1IntegrityError(f"{treatment} batch row is invalid")
            cell_ids = batch_row.get("cell_ids")
            batch_id = batch_row.get("batch_id")
            if (
                not isinstance(cell_ids, list)
                or len(cell_ids) != 1
                or not isinstance(cell_ids[0], str)
                or not isinstance(batch_id, str)
            ):
                raise EedB1IntegrityError(f"{treatment} batch identity is invalid")
            cell_id = cell_ids[0]
            cell = cell_by_id.get(cell_id)
            if not isinstance(cell, Mapping):
                raise EedB1IntegrityError(
                    f"{treatment}/{cell_id} is absent from the plan"
                )
            dimensions = cell.get("dimensions")
            seeds = cell.get("seeds")
            if not isinstance(dimensions, Mapping) or not isinstance(seeds, Mapping):
                raise EedB1IntegrityError(
                    f"{treatment}/{cell_id} dimensions or seeds are invalid"
                )
            batch_path = (
                run_root / "batches" / batch_id / "manifest.json"
            ).resolve()
            if not is_reducer_input_path(evidence_root, batch_path):
                raise EedB1IntegrityError(
                    f"{treatment}/{cell_id} batch path is excluded"
                )
            batch = _read_json(
                batch_path,
                name=f"{treatment}/{cell_id} batch manifest",
            )
            attempts = batch.get("attempts")
            if (
                batch.get("status") != "completed"
                or batch.get("batch_id") != batch_id
                or batch.get("cell_ids") != [cell_id]
                or not isinstance(attempts, int)
                or attempts < 1
                or batch.get("execution_route_sha256") != ROUTES[treatment]
            ):
                raise EedB1IntegrityError(
                    f"{treatment}/{cell_id} is not an admissible terminal batch"
                )
            identity = (treatment, cell_id)
            if identity in seen:
                raise EedB1IntegrityError("duplicate logical cell admission")
            seen.add(identity)
            admitted.append(
                {
                    "logical_cell_id": cell_id,
                    "execution_attempt_id": attempt_identity(batch_id, attempts),
                    "treatment": treatment,
                    "founder_profile": dimensions.get("founder_profile"),
                    "seed": dict(seeds),
                    "source_commit": SOURCE_COMMIT,
                    "artifact_sha256": ARTIFACT_SHA256,
                    "package_id": PACKAGE_ID,
                    "route_sha256": ROUTES[treatment],
                    "qualification_key_sha256": qualification_key,
                    "path": str(batch_path),
                    "file_size": batch_path.stat().st_size,
                    "sha256": _sha256(batch_path),
                    "scientific_role": "primary_terminal_science_cell",
                    "admission_status": "admitted",
                }
            )
    if len(admitted) != 72:
        raise EedB1IntegrityError("positive scientific admission count is not 72")
    return {
        "schema_version": "eiel_eed_b1_positive_scientific_admission_manifest_v1",
        "status": "frozen",
        "scientific_cell_count": 72,
        "one_authoritative_attempt_per_logical_cell": True,
        "duplicate_logical_cells": 0,
        "recovery_or_superseded_paths_admitted": 0,
        "denominator_rule": (
            "paired contrasts use only complete explicitly admitted cells; "
            "excluded or censored cells contribute zero science and may not "
            "silently alter denominators"
        ),
        "censoring_and_interruption_rules": {
            "incomplete_cells": "excluded_zero_science",
            "interrupted_attempts": "excluded_zero_science",
            "missing_terminal_generations": "excluded_zero_science",
            "superseded_attempts": "excluded_zero_science",
            "unpublished_checkpoints": "excluded_zero_science",
            "runtime_complete_reporting_incomplete": "excluded_zero_science",
            "integrity_excluded_cells": "excluded_zero_science",
        },
        "admitted_items": sorted(
            admitted,
            key=lambda row: (
                TREATMENTS.index(str(row["treatment"])),
                str(row["founder_profile"]),
                int(dict(row["seed"])["ecology"]),
            ),
        ),
    }


def verify_input_manifest(
    manifest: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
) -> None:
    root = evidence_root or Path("/")
    rows = manifest.get("inputs")
    if not isinstance(rows, list) or len(rows) != manifest.get("included_file_count"):
        raise EedB1IntegrityError("frozen input manifest count mismatch")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("included_in_reduction") is not True
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
        ):
            raise EedB1IntegrityError("frozen input manifest row is invalid")
        path = Path(str(row["path"]))
        if not is_reducer_input_path(root, path):
            raise EedB1IntegrityError(f"excluded reducer input path: {path}")
        if _sha256(path) != row["sha256"]:
            raise EedB1IntegrityError(f"frozen reducer input changed: {path}")


def run_deterministic_reduction(
    evidence_root: Path,
    spec_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    semantic = treatment_semantic_isolation(spec_root)
    archive = write_impact_archive_audit(
        evidence_root,
        output_dir / "impact_archive_accounting",
    )
    phenotype = audit_phenotype_attempt_one(evidence_root)
    if "batch_attempt_recovery_archive" not in Path(
        str(phenotype["attempt_root"])
    ).parts:
        raise EedB1IntegrityError("phenotype attempt 1 is not preserved outside batches")
    reducer_source = Path(__file__).with_name("eiel_eed_b1_reducer_v1.py")
    input_manifest = freeze_input_manifest(
        evidence_root,
        spec_root,
        reducer_source=reducer_source,
    )
    manifest_path = output_dir / "eed_b1_frozen_input_manifest.json"
    _atomic_write(manifest_path, _pretty_bytes(input_manifest))
    positive_admission = freeze_positive_admission_manifest(evidence_root)
    positive_admission_path = (
        output_dir / "eed_b1_positive_scientific_admission_manifest.json"
    )
    _atomic_write(positive_admission_path, _pretty_bytes(positive_admission))

    verify_input_manifest(input_manifest, evidence_root=evidence_root)
    first = _pretty_bytes(reduce_evidence(evidence_root, positive_admission))
    verify_input_manifest(input_manifest, evidence_root=evidence_root)
    second = _pretty_bytes(reduce_evidence(evidence_root, positive_admission))
    if first != second:
        raise EedB1IntegrityError("frozen reducer output is not byte-identical")

    first_path = output_dir / "eed_b1_reduction_run1.json"
    second_path = output_dir / "eed_b1_reduction_run2.json"
    semantic_path = output_dir / "eed_b1_treatment_semantic_isolation.json"
    _atomic_write(first_path, first)
    _atomic_write(second_path, second)
    _atomic_write(semantic_path, _pretty_bytes(semantic))
    receipt = {
        "schema_version": "eiel_eed_b1_integrity_and_determinism_receipt_v1",
        "status": "pass",
        "impact_archive_accounting": archive,
        "phenotype_attempt_one_excluded": True,
        "treatment_semantic_isolation": {
            "path": str(semantic_path),
            "sha256": _sha256(semantic_path),
        },
        "frozen_input_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "positive_scientific_admission_manifest": {
            "path": str(positive_admission_path),
            "sha256": _sha256(positive_admission_path),
            "admitted_scientific_cells": 72,
            "recovery_or_superseded_paths_admitted": 0,
        },
        "reducer_runs": [
            {"path": str(first_path), "sha256": _sha256(first_path)},
            {"path": str(second_path), "sha256": _sha256(second_path)},
        ],
        "byte_identical_reducer_outputs": True,
    }
    receipt_path = output_dir / "eed_b1_integrity_and_determinism_receipt.json"
    _atomic_write(receipt_path, _pretty_bytes(receipt))
    return {
        "status": "pass",
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and deterministically reduce the frozen EED B1 matrix"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "audit-impact-archive",
        "audit-phenotype-attempt-one",
        "preserve-phenotype-attempt-one",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--evidence-root", required=True, type=Path)
        child.add_argument("--output-dir", required=True, type=Path)
    reduce_parser = subparsers.add_parser("reduce")
    reduce_parser.add_argument("--evidence-root", required=True, type=Path)
    reduce_parser.add_argument("--spec-root", required=True, type=Path)
    reduce_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "audit-impact-archive":
        result = write_impact_archive_audit(evidence_root, output_dir)
    elif args.command == "audit-phenotype-attempt-one":
        result = write_phenotype_attempt_audit(evidence_root, output_dir)
    elif args.command == "preserve-phenotype-attempt-one":
        result = preserve_phenotype_attempt_one(evidence_root, output_dir)
    else:
        result = run_deterministic_reduction(
            evidence_root,
            args.spec_root.resolve(),
            output_dir,
        )
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
