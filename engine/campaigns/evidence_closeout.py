"""Provider-neutral, deterministic campaign evidence closeout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import zipfile


SCHEMA_VERSION = "eiel_campaign_evidence_closeout_v1"
PROJECTION_SCHEMA_VERSION = "eiel_campaign_evidence_projection_v1"
VERIFICATION_SCHEMA_VERSION = "eiel_campaign_evidence_zip_verification_v1"
CELL_CORE_FILES = (
    "cell_manifest.json",
    "config_snapshot.json",
    "completion.json",
)
CELL_EVIDENCE_FILES = (
    "execution_record.json",
    "generation_metrics_manifest.json",
    "reproduction_summary.json",
    "inheritance_summary.json",
    "lineage_summary.json",
    "final_state_digest.json",
    "checkpoint_identities.json",
    "provenance.json",
    "storage_summary.json",
)
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class CampaignEvidenceCloseoutError(RuntimeError):
    """Completed campaign data cannot support a verified evidence closeout."""


EvidenceProjection = Callable[
    [str, Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CampaignEvidenceCloseoutError(
            "evidence payload is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, *, name: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise CampaignEvidenceCloseoutError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return result


def _component(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_COMPONENT.fullmatch(value)
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise CampaignEvidenceCloseoutError(f"{name} is not a safe identifier")
    return value


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignEvidenceCloseoutError(f"{name} must be an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    payload = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def default_evidence_projection(
    cell_id: str,
    cell_result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read a provider-neutral strict projection from ``cell_result``."""

    del cell_id, plan
    return _object(
        cell_result.get("evidence_projection"),
        name="cell_result.evidence_projection",
    )


def _expected_plan_contract(
    plan: Mapping[str, Any],
) -> tuple[
    str,
    str,
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    dict[str, dict[str, str]],
]:
    campaign_id = _component(plan.get("campaign_id"), name="campaign_id")
    plan_sha256 = _digest(plan.get("plan_sha256"), name="plan.plan_sha256")
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise CampaignEvidenceCloseoutError("compiled plan has no cells")
    cell_ids: list[str] = []
    for index, row in enumerate(raw_cells):
        row = _object(row, name=f"plan.cells[{index}]")
        cell_ids.append(_component(row.get("cell_id"), name=f"plan.cells[{index}].cell_id"))
    if len(set(cell_ids)) != len(cell_ids):
        raise CampaignEvidenceCloseoutError("compiled plan contains duplicate cells")
    execution = _object(plan.get("execution"), name="plan.execution")
    generations = execution.get("generations")
    if type(generations) is not int or generations < 0:
        raise CampaignEvidenceCloseoutError("plan generations are invalid")
    expected_generations = tuple(range(generations + 1))
    checkpoints = plan.get("checkpoints")
    raw_milestones = (
        checkpoints.get("milestones")
        if isinstance(checkpoints, Mapping)
        else execution.get("checkpoints")
    )
    if not isinstance(raw_milestones, list) or any(
        type(value) is not int or value < 0 or value > generations
        for value in raw_milestones
    ):
        raise CampaignEvidenceCloseoutError("plan checkpoint milestones are invalid")
    required_checkpoints = tuple(sorted({0, generations, *raw_milestones}))
    raw_artifacts = _object(plan.get("artifacts"), name="plan.artifacts")
    if not raw_artifacts:
        raise CampaignEvidenceCloseoutError("compiled plan has no bound artifacts")
    artifacts: dict[str, dict[str, str]] = {}
    for artifact_id, raw in raw_artifacts.items():
        safe_id = _component(artifact_id, name="plan artifact ID")
        row = _object(raw, name=f"plan.artifacts.{safe_id}")
        path = Path(str(row.get("path") or "")).expanduser()
        abi = row.get("abi")
        if not path.is_absolute() or not isinstance(abi, str) or not abi:
            raise CampaignEvidenceCloseoutError(
                f"plan artifact {safe_id!r} has an invalid path or ABI"
            )
        artifacts[safe_id] = {
            "path": str(path.resolve()),
            "sha256": _digest(
                row.get("sha256"),
                name=f"plan.artifacts.{safe_id}.sha256",
            ),
            "abi": abi,
        }
    return (
        campaign_id,
        plan_sha256,
        tuple(cell_ids),
        expected_generations,
        required_checkpoints,
        artifacts,
    )


def _cell_result_index(
    values: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(values, Mapping):
        rows = [
            (_component(cell_id, name="provider cell ID"), _object(value, name=f"result {cell_id}"))
            for cell_id, value in values.items()
        ]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for index, value in enumerate(values):
            row = _object(value, name=f"provider_cell_results[{index}]")
            cell_id = _component(
                row.get("cell_id"),
                name=f"provider_cell_results[{index}].cell_id",
            )
            rows.append((cell_id, row))
    else:
        raise CampaignEvidenceCloseoutError("provider cell results are malformed")
    if len({cell_id for cell_id, _row in rows}) != len(rows):
        raise CampaignEvidenceCloseoutError("provider cell results contain duplicates")
    return dict(rows)


def _discover_cell_roots(
    run_root: Path,
    cell_ids: Sequence[str],
) -> dict[str, Path]:
    cells_root = run_root / "cells"
    if cells_root.is_symlink() or not cells_root.is_dir():
        raise CampaignEvidenceCloseoutError("completed run lacks a safe cells root")
    discovered: list[tuple[str, Path]] = []
    for path in cells_root.iterdir():
        if not path.is_dir() and not path.is_symlink():
            continue
        cell_id = _component(path.name, name="discovered cell ID")
        discovered.append((cell_id, path))
    if (
        len(discovered) != len(cell_ids)
        or {cell_id for cell_id, _path in discovered} != set(cell_ids)
    ):
        raise CampaignEvidenceCloseoutError(
            "completed run cell roots do not exactly match compiled plan cells"
        )
    roots: dict[str, Path] = {}
    for cell_id in cell_ids:
        candidate = cells_root / cell_id
        if candidate.is_symlink() or not candidate.is_dir():
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} lacks a safe completed cell root"
            )
        root = candidate.resolve()
        try:
            root.relative_to(run_root)
        except ValueError as exc:
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} root escapes the completed run"
            ) from exc
        for filename in CELL_CORE_FILES:
            path = root / filename
            if path.is_symlink() or not path.is_file():
                raise CampaignEvidenceCloseoutError(
                    f"cell {cell_id!r} lacks safe {filename}"
                )
        roots[cell_id] = root
    return roots


def _verify_artifacts(
    provenance: Mapping[str, Any],
    *,
    cell_id: str,
    plan_sha256: str,
    expected_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    required_text = ("provider_id", "profile_id", "mode")
    if any(
        not isinstance(provenance.get(name), str) or not provenance.get(name)
        for name in required_text
    ):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} provenance identity is incomplete"
        )
    if _digest(
        provenance.get("plan_sha256"), name=f"{cell_id}.provenance.plan_sha256"
    ) != plan_sha256:
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} provenance plan hash mismatch"
        )
    for name in ("provider_sha256", "mode_sha256", "source_sha256"):
        _digest(provenance.get(name), name=f"{cell_id}.provenance.{name}")
    source_commit = provenance.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", source_commit) is None
    ):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} provenance source commit is invalid"
        )
    raw_bindings = provenance.get("artifact_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} provenance lacks artifact bindings"
        )
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_bindings):
        row = _object(raw, name=f"{cell_id}.artifact_bindings[{index}]")
        artifact_id = _component(
            row.get("artifact_id"),
            name=f"{cell_id}.artifact_bindings[{index}].artifact_id",
        )
        if artifact_id in seen:
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} has duplicate artifact bindings"
            )
        seen.add(artifact_id)
        expected_binding = expected_artifacts.get(artifact_id)
        if expected_binding is None:
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} has an unexpected artifact binding: {artifact_id}"
            )
        declared_path = Path(str(row.get("path") or "")).expanduser()
        if not declared_path.is_absolute() or declared_path.is_symlink():
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} artifact path is stale: {artifact_id}"
            )
        path = declared_path.resolve()
        expected = _digest(
            row.get("sha256"),
            name=f"{cell_id}.artifact_bindings[{index}].sha256",
        )
        abi = row.get("abi")
        if (
            str(path) != expected_binding["path"]
            or expected != expected_binding["sha256"]
            or abi != expected_binding["abi"]
        ):
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} artifact binding disagrees with the plan: "
                f"{artifact_id}"
            )
        if not path.is_file():
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} artifact path is stale: {artifact_id}"
            )
        if _sha256_file(path) != expected:
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} artifact hash mismatch: {artifact_id}"
            )
        bindings.append(
            {
                "artifact_id": artifact_id,
                "path": str(path),
                "sha256": expected,
                "abi": abi,
                "size_bytes": path.stat().st_size,
            }
        )
    if seen != set(expected_artifacts):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} artifact bindings do not exactly match the plan"
        )
    return {
        **dict(provenance),
        "artifact_bindings": sorted(bindings, key=lambda row: row["artifact_id"]),
    }


def _verify_checkpoints(
    raw: Any,
    *,
    cell_id: str,
    required_generations: Sequence[int],
) -> dict[str, Any]:
    catalog = _object(raw, name=f"{cell_id}.checkpoint_identities")
    expected_keys = {str(value) for value in required_generations}
    if set(catalog) != expected_keys:
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} checkpoint coverage mismatch"
        )
    verified: dict[str, Any] = {}
    for generation in required_generations:
        row = _object(catalog[str(generation)], name=f"{cell_id}.checkpoint.G{generation}")
        if row.get("generation") != generation:
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} checkpoint generation mismatch"
            )
        declared_path = Path(str(row.get("checkpoint_path") or "")).expanduser()
        if not declared_path.is_absolute() or declared_path.is_symlink():
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} checkpoint path is stale at G{generation}"
            )
        path = declared_path.resolve()
        file_sha256 = _digest(
            row.get("checkpoint_file_sha256"),
            name=f"{cell_id}.checkpoint.G{generation}.checkpoint_file_sha256",
        )
        state_sha256 = _digest(
            row.get("state_sha256"),
            name=f"{cell_id}.checkpoint.G{generation}.state_sha256",
        )
        if not path.is_file():
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} checkpoint path is stale at G{generation}"
            )
        if _sha256_file(path) != file_sha256:
            raise CampaignEvidenceCloseoutError(
                f"cell {cell_id!r} checkpoint hash mismatch at G{generation}"
            )
        verified[str(generation)] = {
            "generation": generation,
            "checkpoint_path": str(path),
            "checkpoint_file_sha256": file_sha256,
            "state_sha256": state_sha256,
            "size_bytes": path.stat().st_size,
        }
    return verified


def _validate_projection(
    projection: Mapping[str, Any],
    *,
    cell_id: str,
    plan_sha256: str,
    expected_generations: Sequence[int],
    required_checkpoints: Sequence[int],
    expected_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    if projection.get("schema_version") != PROJECTION_SCHEMA_VERSION:
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} evidence projection schema mismatch"
        )
    raw_rows = projection.get("generation_rows")
    if not isinstance(raw_rows, list):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} generation series is missing"
        )
    rows = [
        dict(_object(row, name=f"{cell_id}.generation_rows[{index}]"))
        for index, row in enumerate(raw_rows)
    ]
    generations = [row.get("generation") for row in rows]
    if generations != list(expected_generations):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} generation coverage mismatch"
        )
    if any(
        row.get("cell_id", cell_id) != cell_id
        or type(row.get("generation")) is not int
        for row in rows
    ):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} generation identity mismatch"
        )
    row_sha256 = [_sha256_bytes(_canonical_bytes(row)) for row in rows]
    summaries = {
        name: dict(
            _object(
                projection.get(name),
                name=f"{cell_id}.{name}",
            )
        )
        for name in (
            "reproduction_summary",
            "inheritance_summary",
            "lineage_summary",
        )
    }
    final_state = dict(
        _object(
            projection.get("final_state_digest"),
            name=f"{cell_id}.final_state_digest",
        )
    )
    if (
        final_state.get("generation") != expected_generations[-1]
        or _digest(
            final_state.get("state_sha256"),
            name=f"{cell_id}.final_state_digest.state_sha256",
        )
        != final_state.get("state_sha256")
    ):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} final state digest is invalid"
        )
    checkpoints = _verify_checkpoints(
        projection.get("checkpoint_identities"),
        cell_id=cell_id,
        required_generations=required_checkpoints,
    )
    if (
        checkpoints[str(expected_generations[-1])]["state_sha256"]
        != final_state["state_sha256"]
    ):
        raise CampaignEvidenceCloseoutError(
            f"cell {cell_id!r} final state does not match terminal checkpoint"
        )
    provenance = _verify_artifacts(
        _object(projection.get("provenance"), name=f"{cell_id}.provenance"),
        cell_id=cell_id,
        plan_sha256=plan_sha256,
        expected_artifacts=expected_artifacts,
    )
    generation_manifest = {
        "schema_version": "eiel_generation_metrics_manifest_v1",
        "cell_id": cell_id,
        "source": "provider_projection",
        "reference": "completed provider cell result generation_rows",
        "generations": list(expected_generations),
        "row_count": len(rows),
        "row_sha256": row_sha256,
        "series_sha256": _sha256_bytes(_canonical_bytes(rows)),
    }
    return {
        "generation_manifest": generation_manifest,
        "summaries": summaries,
        "final_state": final_state,
        "checkpoints": checkpoints,
        "provenance": provenance,
    }


def _file_record(path: Path, *, run_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_verified_zip(
    zip_path: Path,
    *,
    inventory: Sequence[Mapping[str, Any]],
    run_root: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    members = [
        (str(row["path"]), run_root / str(row["path"]))
        for row in inventory
    ]
    members.append(("evidence_closeout/artifact_inventory.json", inventory_path))
    members.sort(key=lambda row: row[0])
    temporary = zip_path.with_name(f".{zip_path.name}.tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for member_name, source in members:
            with archive.open(_zip_info(member_name), "w") as target:
                target.write(source.read_bytes())
    temporary.replace(zip_path)
    zip_sha256 = _sha256_file(zip_path)
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sidecar.write_text(f"{zip_sha256}  {zip_path.name}\n", encoding="ascii")

    expected = {name: _sha256_file(source) for name, source in members}
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(expected):
            raise CampaignEvidenceCloseoutError("evidence ZIP member verification failed")
        for name, digest in expected.items():
            if _sha256_bytes(archive.read(name)) != digest:
                raise CampaignEvidenceCloseoutError(
                    f"evidence ZIP rehash failed: {name}"
                )
    sidecar_value = sidecar.read_text(encoding="ascii").split()[0]
    if sidecar_value != _sha256_file(zip_path):
        raise CampaignEvidenceCloseoutError("evidence ZIP sidecar verification failed")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha256,
        "sha256_sidecar_path": str(sidecar),
        "member_count": len(members),
        "members": [name for name, _source in members],
        "bulky_payloads_included": False,
        "unzip_test_passed": True,
        "member_rehash_passed": True,
        "sidecar_verified": True,
        "verified": True,
    }


def closeout_campaign_evidence(
    plan: Mapping[str, Any],
    *,
    run_root: str | Path,
    provider_cell_results: (
        Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]
    ),
    projection: EvidenceProjection = default_evidence_projection,
) -> dict[str, Any]:
    """Validate, materialize, package, and verify one completed campaign record."""

    (
        campaign_id,
        plan_sha256,
        cell_ids,
        expected_generations,
        required_checkpoints,
        expected_artifacts,
    ) = _expected_plan_contract(plan)
    root = Path(run_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CampaignEvidenceCloseoutError("completed run root is missing or unsafe")
    indexed = _cell_result_index(provider_cell_results)
    if set(indexed) != set(cell_ids) or len(indexed) != len(cell_ids):
        raise CampaignEvidenceCloseoutError(
            "provider cell results do not exactly match compiled plan cells"
        )
    cell_roots = _discover_cell_roots(root, cell_ids)

    validated: dict[str, dict[str, Any]] = {}
    for cell_id in cell_ids:
        raw_projection = _object(
            projection(cell_id, indexed[cell_id], plan),
            name=f"projection for {cell_id}",
        )
        validated[cell_id] = _validate_projection(
            raw_projection,
            cell_id=cell_id,
            plan_sha256=plan_sha256,
            expected_generations=expected_generations,
            required_checkpoints=required_checkpoints,
            expected_artifacts=expected_artifacts,
        )

    cell_records: dict[str, Any] = {}
    inventory_paths: list[Path] = []
    for cell_id in cell_ids:
        cell_root = cell_roots[cell_id]
        value = validated[cell_id]
        payloads = {
            "generation_metrics_manifest.json": value["generation_manifest"],
            "reproduction_summary.json": value["summaries"]["reproduction_summary"],
            "inheritance_summary.json": value["summaries"]["inheritance_summary"],
            "lineage_summary.json": value["summaries"]["lineage_summary"],
            "final_state_digest.json": value["final_state"],
            "checkpoint_identities.json": {
                "schema_version": "eiel_checkpoint_identities_v1",
                "cell_id": cell_id,
                "generations": list(required_checkpoints),
                "checkpoints": value["checkpoints"],
            },
            "provenance.json": value["provenance"],
        }
        for filename, payload in payloads.items():
            _write_json(cell_root / filename, payload)
        execution_record = {
            "schema_version": "eiel_cell_execution_record_v1",
            "campaign_id": campaign_id,
            "cell_id": cell_id,
            "plan_sha256": plan_sha256,
            "provider_id": value["provenance"]["provider_id"],
            "profile_id": value["provenance"]["profile_id"],
            "mode": value["provenance"]["mode"],
            "generation_coverage": list(expected_generations),
            "checkpoint_coverage": list(required_checkpoints),
            "references": {
                filename: {
                    "sha256": _sha256_file(cell_root / filename),
                }
                for filename in payloads
            },
        }
        _write_json(cell_root / "execution_record.json", execution_record)
        stored_paths = [
            *(cell_root / filename for filename in CELL_CORE_FILES),
            *(cell_root / filename for filename in CELL_EVIDENCE_FILES if filename != "storage_summary.json"),
        ]
        storage_summary = {
            "schema_version": "eiel_cell_evidence_storage_summary_v1",
            "cell_id": cell_id,
            "files": [
                _file_record(path, run_root=root)
                for path in sorted(stored_paths, key=lambda item: item.name)
            ],
            "excluded_payload_classes": [
                "checkpoints",
                "raw_tensors",
                "native_binaries",
                "runtime_logs",
            ],
        }
        storage_summary["file_count"] = len(storage_summary["files"])
        storage_summary["total_bytes"] = sum(
            row["bytes"] for row in storage_summary["files"]
        )
        _write_json(cell_root / "storage_summary.json", storage_summary)
        included = [
            *(cell_root / filename for filename in CELL_CORE_FILES),
            *(cell_root / filename for filename in CELL_EVIDENCE_FILES),
        ]
        inventory_paths.extend(included)
        cell_records[cell_id] = {
            "cell_root": cell_root.relative_to(root).as_posix(),
            "execution_record_sha256": _sha256_file(
                cell_root / "execution_record.json"
            ),
            "final_state_sha256": value["final_state"]["state_sha256"],
            "generation_series_sha256": value["generation_manifest"][
                "series_sha256"
            ],
        }

    closeout_root = root / "evidence_closeout"
    campaign_manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "cell_ids": list(cell_ids),
        "expected_generations": list(expected_generations),
        "required_checkpoint_generations": list(required_checkpoints),
        "cells": cell_records,
        "provider_neutral_projection": True,
        "bulky_payload_policy": "references_only_excluded_from_zip",
    }
    campaign_manifest["manifest_sha256"] = _sha256_bytes(
        _canonical_bytes(campaign_manifest)
    )
    campaign_manifest_path = closeout_root / "campaign_manifest.json"
    _write_json(campaign_manifest_path, campaign_manifest)
    inventory_paths.append(campaign_manifest_path)
    inventory = [
        _file_record(path, run_root=root)
        for path in sorted(inventory_paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    artifact_inventory = {
        "schema_version": "eiel_campaign_artifact_inventory_v1",
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory),
    }
    artifact_inventory["inventory_sha256"] = _sha256_bytes(
        _canonical_bytes(artifact_inventory)
    )
    inventory_path = closeout_root / "artifact_inventory.json"
    _write_json(inventory_path, artifact_inventory)
    zip_path = closeout_root / f"{campaign_id}-evidence.zip"
    verification = _write_verified_zip(
        zip_path,
        inventory=inventory,
        run_root=root,
        inventory_path=inventory_path,
    )
    verification_path = closeout_root / "verification_receipt.json"
    _write_json(verification_path, verification)
    replacement_artifact = {
        "artifact_family": "campaign_cell_evidence",
        "replacement_path": str(zip_path),
        "replacement_sha256": verification["zip_sha256"],
        "verification_receipt_path": str(verification_path),
        "verified": True,
        "replaces": [
            "cell_manifests",
            "generation_metrics",
            "reproduction_summary",
            "inheritance_summary",
            "lineage_summary",
            "final_state_digest",
            "checkpoint_identities",
            "provenance",
            "storage_summary",
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_manifest_path": str(campaign_manifest_path),
        "artifact_inventory_path": str(inventory_path),
        "zip_path": str(zip_path),
        "sha256_sidecar_path": verification["sha256_sidecar_path"],
        "verification_receipt_path": str(verification_path),
        "zip_sha256": verification["zip_sha256"],
        "verified": True,
        "replacement_artifacts": [replacement_artifact],
    }


__all__ = [
    "CELL_CORE_FILES",
    "CELL_EVIDENCE_FILES",
    "CampaignEvidenceCloseoutError",
    "EvidenceProjection",
    "PROJECTION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "closeout_campaign_evidence",
    "default_evidence_projection",
]
