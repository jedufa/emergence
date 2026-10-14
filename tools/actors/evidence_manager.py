"""EIEL Evidence Decision Manager over the existing read-only EDL."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    actor_state_root,
    atomic_write_json,
    deterministic_zip,
    emit_actor_error,
    emit_payload,
    file_sha256,
    forbidden_operation_error,
    new_record,
    read_json,
    read_structured,
    stable_id,
    utc_now,
    with_guidance,
)
from tools.actors.evidence_inventory import evidence_attachment_inventory
from tools.actors.reference_integrity import actor_source_ref
from tools.actors.store import ActorRecordStore
from tools.actors.release import operator_metadata
from tools.actors.authority import evidence_preflight_markdown, validate_authority_context
from tools.actors.evidence_archive import (
    inspect_published_evidence_archive,
    is_evidence_archive as is_published_evidence_archive,
)
from tools.evidence.librarian.core import inspect_zip_safety
from tools.evidence import SHARED_EVIDENCE_ROOT
from tools.evidence.contracts import check_campaign_schema
from tools.evidence.manifest import REQUIRED_FIELDS, validate_manifest_data
from tools.evidence.profiles import get_profile
from tools.evidence.reports import generate_campaign_decision_report

ACTOR_NAME = "evidenceManager"
STORE_NAMESPACE = "evidence-decisions"
ASSESSMENT_RECORD_TYPE = "evidence_assessment"
ASSESSMENT_SCHEMA_VERSION = "eiel_evidence_assessment_v1"
PREPARATION_SCHEMA_VERSION = "eiel_evidence_manifest_preparation_v1"


def _configured_evidence_root(explicit: str | None) -> Path:
    """Resolve an explicit root or an absolute operator-configured environment root."""
    value = explicit or os.environ.get("EIEL_EVIDENCE_ROOT") or SHARED_EVIDENCE_ROOT
    if not value or not value.strip():
        raise ActorError(
            "configure --root / --evidence-root or EIEL_EVIDENCE_ROOT; assess also accepts --out-dir",
            code="evidence_root_not_configured",
        )
    root = Path(value).expanduser()
    if not explicit and not root.is_absolute():
        raise ActorError("EIEL_EVIDENCE_ROOT must be absolute", code="evidence_root_invalid")
    if root.exists() and not root.is_dir():
        raise ActorError("evidence root must be a directory", code="evidence_root_invalid")
    return root.resolve()


def _store(args: argparse.Namespace) -> ActorRecordStore:
    return ActorRecordStore(actor_state_root(args.state_root), STORE_NAMESPACE)


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def inspect_evidence_archive(path: Path) -> dict[str, Any]:
    """Inspect an archive without extraction or evidence reinterpretation.

    Registered ``EIEL-EV-xxxxxx--*.evidence.zip`` files are durable
    Knowledge Librarian archive objects.  They are verified through the
    published-archive contract and are never treated as fresh raw campaign
    trees.  Other ZIP files receive only generic safety and integrity
    inspection and remain unregistered until the Librarian says otherwise.
    """

    archive = path.expanduser().resolve()
    if not archive.is_file():
        raise ActorError(
            f"evidence archive does not exist: {archive}",
            code="evidence_archive_missing",
        )

    if is_published_evidence_archive(archive):
        published = inspect_published_evidence_archive(archive)
        return {
            **published,
            "readiness": published["status"],
            "safety": published["zip_safety"],
            "automatic_extraction": False,
            "procedural_decision": None,
            "historical_archive_only": True,
        }

    safety = inspect_zip_safety(archive)
    safe = safety.get("safety_state") == "safe"
    status = "UNREGISTERED_ARCHIVE_INSPECTED" if safe else "INVALID_ARCHIVE"
    return {
        "schema_version": "eiel_unregistered_archive_inspection_v1",
        "archive_path": str(archive),
        "archive_name": archive.name,
        "archive_sha256": file_sha256(archive),
        "status": status,
        "readiness": status,
        "registered_filename": False,
        "evidence_id": None,
        "safety": safety,
        "zip_safety": safety,
        "problems": [] if safe else [
            "ZIP safety/integrity inspection reported one or more issues"
        ],
        "warnings": [
            "archive has no registered EIEL-EV identity and carries no authority effect"
        ] if safe else [],
        "automatic_extraction": False,
        "historical_archive_only": True,
        "procedural_decision": None,
        "authority_effect": "none",
        "scientific_claim_effect": "none",
        "assessment_eligibility": "not_raw_campaign_input",
        "interpretation": (
            "Only ZIP safety and file integrity were inspected.  The archive was "
            "not extracted, assessed, qualified, or published."
        ),
    }


def _is_evidence_archive(target: str | Path) -> bool:
    path = Path(target).expanduser()
    return path.is_file() and path.suffix.lower() == ".zip"


def _resolve_manifest(target: str | Path) -> Path:
    path = Path(target).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise ActorError(
            f"campaign or manifest does not exist: {path}",
            code="campaign_or_manifest_missing",
            next_actions=[{"description": "provide the real campaign output or evidence manifest path", "command": "python -m tools.eiel evidenceManager inspect PATH"}],
        )
    candidates = [
        path / "campaign_evidence_manifest.json",
        path / "campaign_evidence_manifest.draft.json",
    ]
    candidates.extend(sorted(path.glob("**/campaign_evidence_manifest.json")))
    candidates.extend(sorted(path.glob("**/campaign_evidence_manifest.draft.json")))
    unique: list[Path] = []
    for candidate in candidates:
        if candidate.is_file() and candidate not in unique:
            unique.append(candidate)
    if not unique:
        raise ActorError(
            f"no campaign evidence manifest found under: {path}",
            code="campaign_evidence_manifest_missing",
            problems=["Evidence Manager will not guess campaign identity, matrix, source, or profile."],
            next_actions=[{"description": "prepare a draft manifest from available outputs", "command": f"python -m tools.eiel evidenceManager prepare {path}"}],
        )
    if len(unique) > 1:
        exact = [candidate for candidate in unique if candidate.parent == path]
        if len(exact) == 1:
            return exact[0]
        raise ActorError(
            "multiple campaign evidence manifests found; provide one explicitly: "
            + ", ".join(str(candidate) for candidate in unique[:10])
        )
    return unique[0]


def _raw_manifest(path: Path) -> dict[str, Any]:
    return read_json(path, name="campaign evidence manifest")


def _profile_state(name: str) -> dict[str, Any]:
    try:
        profile = get_profile(name)
    except KeyError as exc:
        return {"known": False, "implemented": False, "error": str(exc)}
    return {
        "known": True,
        "implemented": bool(profile.implemented_decision_logic),
        "required_dimensions": list(profile.required_dimensions),
        "forbidden_claims": list(profile.forbidden_claims),
    }


def inspect_manifest(path: Path) -> dict[str, Any]:
    raw = _raw_manifest(path)
    structure_errors = validate_manifest_data(raw)
    profile_name = str(raw.get("profile") or "")
    profile = _profile_state(profile_name) if profile_name else {"known": False, "implemented": False, "error": "profile missing"}
    schema_status: dict[str, Any] | None = None
    if not structure_errors:
        schema_status = check_campaign_schema(path)
    if structure_errors:
        readiness = "DRAFT"
    elif not profile.get("implemented"):
        readiness = "PROFILE_NOT_IMPLEMENTED"
    elif schema_status and schema_status.get("ok") is True:
        readiness = "READY_FOR_ASSESSMENT"
    else:
        readiness = "INVALID_OR_INCOMPLETE"
    return {
        "schema_version": "eiel_evidence_manager_inspection_v1",
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "campaign_id": raw.get("campaign_id"),
        "profile": profile_name,
        "profile_state": profile,
        "readiness": readiness,
        "structure_errors": structure_errors,
        "schema_status": schema_status,
        "planned_run_count": raw.get("planned_run_count"),
        "completed_run_count": raw.get("completed_run_count"),
        "failed_run_count": raw.get("failed_run_count"),
    }


def _discover(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    manifests = sorted(
        {
            *root.rglob("campaign_evidence_manifest.json"),
            *root.rglob("campaign_evidence_manifest.draft.json"),
        }
    )
    rows: list[dict[str, Any]] = []
    manifest_parents = {path.parent.resolve() for path in manifests}
    for path in manifests:
        try:
            rows.append(inspect_manifest(path))
        except ActorError as exc:
            rows.append({"manifest_path": str(path), "readiness": "INVALID", "error": str(exc)})
    metrics_parents = sorted({path.parent.resolve() for path in root.rglob("metrics.csv")})
    for parent in metrics_parents:
        if any(parent == known or known in parent.parents for known in manifest_parents):
            continue
        rows.append(
            {
                "manifest_path": None,
                "campaign_root": str(parent),
                "readiness": "MANIFEST_MISSING",
                "metrics_path": str(parent / "metrics.csv"),
            }
        )
    rows.sort(key=lambda row: (str(row.get("campaign_id") or ""), str(row.get("manifest_path") or row.get("campaign_root") or "")))
    return rows


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _nested_value(data: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    for section in ("sim_cfg", "env_cfg", "run", "settings", "config"):
        nested = data.get(section)
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested:
                    return nested[name]
    return None


def _consistent_config_value(configs: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Any:
    values = [_nested_value(config, names) for config in configs]
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    return first if all(value == first for value in present) else None


def _summary_candidates(root: Path) -> list[Path]:
    names = ("summary.json", "batch_summary.json", "harness_summary.json", "run_summary.json")
    return sorted({path for name in names for path in root.rglob(name) if path.is_file()})


def _summary_counts(paths: Sequence[Path]) -> tuple[int | None, int | None, int | None]:
    for path in paths:
        try:
            data = read_json(path, name="harness summary")
        except ActorError:
            continue
        planned = data.get("planned_runs", data.get("planned_run_count"))
        completed = data.get("completed_runs", data.get("completed_run_count"))
        failed = data.get("failed_runs", data.get("failed_run_count"))
        if isinstance(data.get("runs"), list):
            runs = data["runs"]
            completed = completed if isinstance(completed, int) else len(runs)
            failed = failed if isinstance(failed, int) else sum(
                1 for row in runs if isinstance(row, Mapping) and str(row.get("status") or "").lower() in {"failed", "error"}
            )
        if any(isinstance(value, int) for value in (planned, completed, failed)):
            return (
                int(planned) if isinstance(planned, int) else None,
                int(completed) if isinstance(completed, int) else None,
                int(failed) if isinstance(failed, int) else None,
            )
    return None, None, None


def _matrix_from_summaries(paths: Sequence[Path]) -> tuple[dict[str, Any], list[int]]:
    axes: dict[str, set[Any]] = {}
    seeds: set[int] = set()
    for path in paths:
        try:
            data = read_json(path, name="harness summary")
        except ActorError:
            continue
        for row in data.get("runs") or []:
            if not isinstance(row, Mapping):
                continue
            seed = row.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                seeds.add(seed)
            row_axes = row.get("axes")
            if isinstance(row_axes, Mapping):
                for name, value in row_axes.items():
                    if isinstance(value, (str, int)) and not isinstance(value, bool):
                        axes.setdefault(str(name), set()).add(value)
    matrix = {
        "dimensions": [
            {"name": name, "values": sorted(values, key=lambda value: (str(type(value)), str(value)))}
            for name, values in sorted(axes.items())
        ]
    }
    return matrix, sorted(seeds)


def _matrix_from_csv(root: Path) -> tuple[dict[str, Any], list[int]]:
    for path in sorted(root.rglob("matrix.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error):
            continue
        if not rows:
            continue
        ignored = {"run_id", "run_dir", "status", "command", "cmd", "output", "metrics_path", "config_path"}
        dimensions: list[dict[str, Any]] = []
        seeds: set[int] = set()
        for name in rows[0].keys():
            if not name or name in ignored:
                continue
            values = sorted({str(row.get(name) or "") for row in rows if str(row.get(name) or "")})
            if name == "seed":
                for value in values:
                    try:
                        seeds.add(int(value))
                    except ValueError:
                        pass
                continue
            if 0 < len(values) <= 100:
                dimensions.append({"name": name, "values": values})
        return {"dimensions": dimensions}, sorted(seeds)
    return {"dimensions": []}, []


def prepare_manifest(root: Path, metadata: Mapping[str, Any], *, profile: str | None, generated_at: str) -> dict[str, Any]:
    campaign_root = root.resolve()
    config_paths = sorted(campaign_root.rglob("config.json"))
    metrics_paths = sorted(campaign_root.rglob("metrics.csv"))
    summaries = _summary_candidates(campaign_root)
    configs: list[dict[str, Any]] = []
    for path in config_paths[:5000]:
        try:
            configs.append(read_json(path, name="run config"))
        except ActorError:
            pass
    matrix, seeds = _matrix_from_summaries(summaries)
    if not matrix.get("dimensions"):
        matrix, csv_seeds = _matrix_from_csv(campaign_root)
        seeds = sorted({*seeds, *csv_seeds})
    planned, completed, failed = _summary_counts(summaries)
    registry_candidates = sorted(
        [
            *campaign_root.glob("*.yaml"),
            *campaign_root.glob("*.yml"),
            *campaign_root.glob("*registry*.json"),
        ]
    )
    repo = Path(str(metadata.get("worktree") or campaign_root)).resolve()
    if not (repo / ".git").exists():
        probe = campaign_root
        while probe.parent != probe:
            if (probe / ".git").exists():
                repo = probe
                break
            probe = probe.parent
    inferred: dict[str, Any] = {
        "campaign_id": metadata.get("campaign_id") or campaign_root.name,
        "profile": profile or metadata.get("profile") or "",
        "git_commit": metadata.get("git_commit") or _git_value(repo, "rev-parse", "HEAD"),
        "branch": metadata.get("branch") or _git_value(repo, "branch", "--show-current"),
        "worktree": str(metadata.get("worktree") or repo),
        "run_root": str(metadata.get("run_root") or campaign_root),
        "registry_path": str(metadata.get("registry_path") or (registry_candidates[0] if len(registry_candidates) == 1 else "")),
        "scenario_matrix": metadata.get("scenario_matrix") or matrix,
        "seed_list": metadata.get("seed_list") or seeds,
        "generations": metadata.get("generations", _consistent_config_value(configs, ("generations", "generation_count", "gen_total")) or 0),
        "steps_per_generation": metadata.get("steps_per_generation", _consistent_config_value(configs, ("steps_per_generation", "ticks_per_generation", "steps")) or 0),
        "population_size": metadata.get("population_size", _consistent_config_value(configs, ("population_size", "population", "initial_population")) or 0),
        "grid_size": metadata.get("grid_size", _consistent_config_value(configs, ("grid_size", "grid_width", "world_size")) or 0),
        "device": metadata.get("device") or _consistent_config_value(configs, ("device",)) or "",
        "planned_run_count": metadata.get("planned_run_count", planned if planned is not None else len(metrics_paths)),
        "completed_run_count": metadata.get("completed_run_count", completed if completed is not None else len(metrics_paths)),
        "failed_run_count": metadata.get("failed_run_count", failed if failed is not None else 0),
        "config_files": metadata.get("config_files") or [str(path) for path in config_paths],
        "output_paths": metadata.get("output_paths") or sorted({str(path.parent) for path in metrics_paths}),
        "metrics_glob": metadata.get("metrics_glob") or "**/metrics.csv",
        "config_glob": metadata.get("config_glob") or "**/config.json",
        "harness_manifest_paths": metadata.get("harness_manifest_paths") or [str(path) for path in sorted(campaign_root.rglob("*manifest*.json")) if "campaign_evidence_manifest" not in path.name],
        "harness_summary_paths": metadata.get("harness_summary_paths") or [str(path) for path in summaries],
        "timestamp_utc": metadata.get("timestamp_utc") or generated_at,
    }
    for key, value in metadata.items():
        if key not in inferred:
            inferred[key] = value
    errors = validate_manifest_data(inferred)
    for field in ("generations", "steps_per_generation", "population_size", "grid_size", "planned_run_count"):
        value = inferred.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{field} must be a positive integer for manager readiness")
    planned_value = inferred.get("planned_run_count")
    completed_value = inferred.get("completed_run_count")
    failed_value = inferred.get("failed_run_count")
    if all(isinstance(value, int) and not isinstance(value, bool) for value in (planned_value, completed_value, failed_value)):
        if int(completed_value) + int(failed_value) > int(planned_value):
            errors.append("completed_run_count + failed_run_count exceeds planned_run_count")
    errors = list(dict.fromkeys(errors))
    unresolved_fields = sorted(
        field
        for field in REQUIRED_FIELDS
        if field not in inferred
        or inferred.get(field) in (None, "", [])
        or field in {"generations", "steps_per_generation", "population_size", "grid_size", "planned_run_count"}
        and inferred.get(field) == 0
    )
    inferred["manager_preparation"] = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "READY" if not errors else "DRAFT",
        "unresolved_fields": unresolved_fields,
        "validation_errors": errors,
        "inferred_config_count": len(configs),
        "metrics_file_count": len(metrics_paths),
        "no_silent_guessing": True,
    }
    return inferred



def _technical_qualification_requirement(raw: Mapping[str, Any]) -> tuple[bool, list[str]]:
    manager = raw.get("manager_requirements") if isinstance(raw.get("manager_requirements"), Mapping) else {}
    required = bool(manager.get("technical_qualification_required") or raw.get("technical_qualification_required"))
    refs = [str(item) for item in (
        manager.get("technical_qualification_refs")
        or raw.get("technical_qualification_refs")
        or raw.get("qualification_refs")
        or []
    ) if str(item).strip()]
    return required, refs


def _mission_context(args: argparse.Namespace) -> dict[str, Any]:
    mission_id = getattr(args, "mission", None)
    if not mission_id:
        return {
            "mission_id": None,
            "qualification_refs": [],
            "expected_source_commit": None,
            "source_ref": None,
            "authority_context": None,
            "authority_validation": {"valid": False, "ready_for_work": False, "errors": ["managed evidence assessment requires a mission authority context"]},
            "evidence_preflight_markdown": evidence_preflight_markdown(None),
        }
    mission = ActorRecordStore(actor_state_root(args.state_root), "missions").get(mission_id)
    payload = dict(mission.get("payload") or {})
    references = dict(payload.get("references") or {})
    source_identity = dict(payload.get("source_identity") or {})
    authority_context = payload.get("authority_context") if isinstance(payload.get("authority_context"), Mapping) else None
    authority_validation = validate_authority_context(
        authority_context,
        mission_type=str(payload.get("mission_type") or "IMPLEMENTATION_VALIDATION"),
        resource_demand=payload.get("resource_demand") if isinstance(payload.get("resource_demand"), Mapping) else {},
    )
    preflight = dict(authority_context or {}).get("preflight") if isinstance(authority_context, Mapping) else None
    return {
        "mission_id": mission_id,
        "qualification_refs": [str(item) for item in references.get("qualification_record_ids") or []],
        "expected_source_commit": source_identity.get("source_commit"),
        "source_ref": actor_source_ref("mission", mission),
        "authority_context": dict(authority_context) if isinstance(authority_context, Mapping) else None,
        "authority_validation": authority_validation,
        "evidence_preflight_markdown": evidence_preflight_markdown(preflight),
    }


def _validate_technical_qualification_refs(
    refs: Sequence[str],
    *,
    state_root: str | None,
) -> tuple[list[str], list[str]]:
    """Validate bound technical authority without converting evidence into qualification."""

    if not refs:
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    manager_store = ActorRecordStore(actor_state_root(state_root), "qualifications")
    from tools.actors import qualification_manager as qm

    register = None
    if qm.DEFAULT_AUTHORITY_REGISTER.is_file():
        try:
            register = qm.load_authority_register(qm.DEFAULT_AUTHORITY_REGISTER)
        except ActorError as exc:
            warnings.append(f"qualification authority register could not be read: {exc}")
    for ref in refs:
        ref_text = str(ref)
        if manager_store.exists(ref_text):
            record = manager_store.get(ref_text)
            record_type = str(record.get("record_type") or "")
            payload = dict(record.get("payload") or {})
            if record_type == qm.CHECK_RECORD_TYPE:
                result = dict(payload.get("result") or {})
                if not qm._positive(result):
                    errors.append(f"technical qualification check {ref_text} is not admitted: {result.get('decision')}")
            elif record_type == qm.DECISION_RECORD_TYPE:
                decision = dict(payload.get("decision") or {})
                if not decision.get("accepted_for_envelope_expansion"):
                    errors.append(f"technical qualification decision {ref_text} is not accepted")
            else:
                errors.append(f"technical qualification reference {ref_text} is a {record_type}, not admitted authority")
            continue
        if register is not None:
            resolution = qm.resolve_authority_register({"track_label": ref_text}, register)
            if qm._positive(resolution):
                continue
            if resolution.get("record_id"):
                errors.append(f"qualification authority track {ref_text} is not admitted: {resolution.get('decision')}")
                continue
        errors.append(f"unknown technical qualification reference: {ref_text}")
    return errors, warnings


def _inspection_guidance(inspection: Mapping[str, Any], *, target: str) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    readiness = str(inspection.get("readiness") or "UNKNOWN")
    reasons: list[str] = []
    problems: list[str] = []
    actions: list[dict[str, Any]] = []
    if readiness == "READY_FOR_ASSESSMENT":
        reasons.append("campaign_ready_for_assessment")
        actions.append({"description": "run the implemented procedural decision profile", "command": f"python -m tools.eiel evidenceManager assess {target} --deterministic"})
    elif readiness == "DRAFT":
        reasons.append("manifest_draft")
        problems.extend(inspection.get("structure_errors") or [])
        actions.append({"description": "prepare or repair the evidence manifest", "command": f"python -m tools.eiel evidenceManager prepare {Path(target).parent if Path(target).is_file() else target}"})
    elif readiness == "PROFILE_NOT_IMPLEMENTED":
        reasons.append("decision_profile_not_implemented")
        problems.append(f"profile decision logic is not implemented: {inspection.get('profile')}")
    elif readiness == "TECHNICAL_QUALIFICATION_MISSING":
        reasons.append("technical_qualification_missing")
        problems.append("the manifest requires technical qualification but no technical qualification reference is bound")
        actions.append({"description": "check and bind technical qualification", "command": "python -m tools.eiel qualificationManager check --mission MIS-... --store"})
    else:
        reasons.append("campaign_invalid_or_incomplete")
        status = inspection.get("schema_status") if isinstance(inspection.get("schema_status"), Mapping) else {}
        problems.extend(status.get("errors") or inspection.get("structure_errors") or ["campaign evidence is incomplete or invalid"])
        actions.append({"description": "inspect missing runs, controls, pairs, or schema fields", "command": f"python -m tools.eiel evidenceManager inspect {target}"})
    return reasons, problems, actions


def command_discover(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = _configured_evidence_root(args.root)
    rows = _discover(root)
    counts = Counter(str(row.get("readiness") or "UNKNOWN") for row in rows)
    ready = [row for row in rows if row.get("readiness") == "READY_FOR_ASSESSMENT"]
    return with_guidance(
        {"ok": True, "root": str(root), "count": len(rows), "readiness_counts": dict(sorted(counts.items())), "campaigns": rows},
        status="CAMPAIGNS_DISCOVERED",
        warnings=[] if rows else ["no evidence manifests or unmanifested metrics roots were found"],
        next_actions=[{"description": "inspect a ready campaign", "command": f"python -m tools.eiel evidenceManager inspect {ready[0].get('manifest_path')}"}] if ready else [],
    ), 0


def command_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if _is_evidence_archive(args.target):
        archive = inspect_evidence_archive(Path(args.target))
        status = str(archive.get("status") or archive.get("readiness"))
        safe = status not in {"INVALID_ARCHIVE", "INVALID_PUBLISHED_EVIDENCE_ARCHIVE"}
        evidence_id = archive.get("evidence_id")
        if status == "PUBLISHED_EVIDENCE_ARCHIVE_VERIFIED":
            actions = [{
                "action": "resolve_historical_evidence_authority",
                "description": (
                    "Use the Knowledge Librarian to resolve this permanent evidence ID. "
                    "Do not feed the archive back into Evidence Manager as raw campaign input."
                ),
                "command": (
                    "python -m tools.evidence.librarian preflight "
                    f"--topic '{evidence_id}' --collection evidence --json"
                ),
            }]
        elif status == "UNREGISTERED_ARCHIVE_INSPECTED":
            actions = [{
                "action": "register_or_stage_archive",
                "description": (
                    "Use the Knowledge Librarian ingress workflow if this archive should become "
                    "institutional evidence.  Otherwise provide an explicit raw campaign directory "
                    "and manifest; Evidence Manager will not infer one from a ZIP."
                ),
                "command": f"python -m tools.evidence.librarian inspect-zip {archive['archive_path']} --json",
            }]
        else:
            actions = []
        problems = list(archive.get("problems") or [])
        if not safe and not problems:
            problems = [str(row) for row in dict(archive.get("safety") or {}).get("issues") or []]
        return with_guidance(
            {"ok": safe, "archive_inspection": archive},
            status=status,
            reason_codes=[status.lower()],
            problems=problems,
            warnings=list(archive.get("warnings") or []),
            next_actions=actions,
            related_records=[str(evidence_id)] if evidence_id else [],
        ), 0 if safe else 1
    manifest = _resolve_manifest(args.target)
    inspection = inspect_manifest(manifest)
    raw = _raw_manifest(manifest)
    required, refs = _technical_qualification_requirement(raw)
    qualification_errors, qualification_warnings = _validate_technical_qualification_refs(refs, state_root=args.state_root)
    if required and not refs and inspection["readiness"] == "READY_FOR_ASSESSMENT":
        inspection["readiness"] = "TECHNICAL_QUALIFICATION_MISSING"
    elif qualification_errors and inspection["readiness"] == "READY_FOR_ASSESSMENT":
        inspection["readiness"] = "TECHNICAL_QUALIFICATION_INVALID"
    inspection["technical_qualification_required"] = required
    inspection["technical_qualification_refs"] = refs
    inspection["technical_qualification_errors"] = qualification_errors
    reasons, problems, actions = _inspection_guidance(inspection, target=str(manifest))
    problems.extend(qualification_errors)
    if qualification_errors:
        reasons.append("technical_qualification_invalid")
        actions.append({"description": "bind a current admitted technical qualification record", "command": "python -m tools.eiel qualificationManager check --mission MIS-... --store"})
    positive = inspection["readiness"] == "READY_FOR_ASSESSMENT"
    return with_guidance(
        {"ok": positive, "inspection": inspection},
        status=str(inspection["readiness"]), reason_codes=reasons, problems=problems,
        warnings=[*list(dict(inspection.get("schema_status") or {}).get("warnings") or []), *qualification_warnings], next_actions=actions,
        related_records=refs,
    ), 0 if positive else 1


def command_prepare(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = Path(args.campaign_root).expanduser().resolve()
    if not root.is_dir():
        raise ActorError(
            f"campaign root does not exist: {root}", code="campaign_root_missing",
            next_actions=[{"description": "provide the actual campaign output directory", "command": "python -m tools.eiel evidenceManager prepare PATH_TO_CAMPAIGN"}],
        )
    metadata = read_structured(args.metadata, name="manifest metadata") if args.metadata else {}
    generated_at = utc_now(args.generated_at)
    manifest = prepare_manifest(root, metadata, profile=args.profile, generated_at=generated_at)
    errors = list(dict(manifest.get("manager_preparation") or {}).get("validation_errors") or [])
    target = Path(args.manifest_output).expanduser().resolve() if args.manifest_output else root / ("campaign_evidence_manifest.json" if not errors else "campaign_evidence_manifest.draft.json")
    atomic_write_json(target, manifest)
    unresolved = dict(manifest.get("manager_preparation") or {}).get("unresolved_fields") or []
    actions = [] if not errors else [{"description": f"provide unresolved manifest field: {field}", "command": f"edit {target}"} for field in unresolved[:8]]
    return with_guidance(
        {"ok": not errors, "manifest_path": str(target), "status": "READY" if not errors else "DRAFT", "unresolved_fields": unresolved, "validation_errors": errors},
        status="READY" if not errors else "DRAFT", reason_codes=[] if not errors else ["manifest_incomplete"],
        problems=errors, next_actions=actions,
    ), 0 if not errors else 1


def _validate_expected_source(raw: Mapping[str, Any], expected: str | None) -> list[str]:
    if not expected:
        return []
    actual = str(raw.get("git_commit") or "")
    if actual != expected:
        return [f"manifest git_commit {actual!r} does not match expected source commit {expected!r}"]
    return []


def command_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = _resolve_manifest(args.target)
    inspection = inspect_manifest(manifest)
    raw = _raw_manifest(manifest)
    mission = _mission_context(args)
    authority_errors = list(dict(mission.get("authority_validation") or {}).get("errors") or []) if args.mission else []
    expected_source = args.expected_source_commit or mission.get("expected_source_commit")
    source_errors = _validate_expected_source(raw, expected_source)
    required, manifest_refs = _technical_qualification_requirement(raw)
    refs = list(dict.fromkeys([*manifest_refs, *mission.get("qualification_refs", [])]))
    qualification_errors, qualification_warnings = _validate_technical_qualification_refs(refs, state_root=args.state_root)
    if required and not refs:
        inspection["readiness"] = "TECHNICAL_QUALIFICATION_MISSING"
    elif qualification_errors and inspection["readiness"] == "READY_FOR_ASSESSMENT":
        inspection["readiness"] = "TECHNICAL_QUALIFICATION_INVALID"
    valid = inspection["readiness"] == "READY_FOR_ASSESSMENT" and not source_errors and not qualification_errors and not authority_errors
    reasons, problems, actions = _inspection_guidance(inspection, target=str(manifest))
    problems.extend(source_errors)
    problems.extend(qualification_errors)
    problems.extend(authority_errors)
    if source_errors:
        reasons.append("source_identity_mismatch")
        actions.append({"description": "use outputs from the expected source or correct the manifest binding"})
    return with_guidance(
        {
            "ok": valid,
            "inspection": inspection,
            "mission_id": mission.get("mission_id"),
            "expected_source_commit": expected_source,
            "source_identity_errors": source_errors,
            "technical_qualification_refs": refs,
            "technical_qualification_errors": qualification_errors,
            "authority_errors": authority_errors,
            "evidence_preflight_markdown": mission.get("evidence_preflight_markdown"),
        },
        status="READY_FOR_ASSESSMENT" if valid else "BLOCKED", reason_codes=reasons, problems=problems,
        warnings=[*list(dict(inspection.get("schema_status") or {}).get("warnings") or []), *qualification_warnings], next_actions=actions, related_records=refs,
    ), 0 if valid else 1


def command_assess(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = _resolve_manifest(args.target)
    inspection = inspect_manifest(manifest)
    raw = _raw_manifest(manifest)
    mission = _mission_context(args)
    if args.mission:
        authority_errors = list(dict(mission.get("authority_validation") or {}).get("errors") or [])
    else:
        authority_errors = []
    if authority_errors:
        raise ActorError(
            "mission authority context is not ready for evidence assessment",
            code="mission_authority_context_invalid",
            problems=authority_errors,
            next_actions=[{"description": "repair the mission authority binding", "command": f"python -m tools.eiel missionPlanner next {args.mission}"}],
            related_records=[args.mission],
        )
    expected_source = args.expected_source_commit or mission.get("expected_source_commit")
    source_errors = _validate_expected_source(raw, expected_source)
    required, manifest_refs = _technical_qualification_requirement(raw)
    qualification_refs = list(dict.fromkeys([*manifest_refs, *mission.get("qualification_refs", [])]))
    qualification_errors, qualification_warnings = _validate_technical_qualification_refs(qualification_refs, state_root=args.state_root)
    if required and not qualification_refs:
        raise ActorError(
            "technical qualification is required but no technical qualification reference is bound",
            code="technical_qualification_missing",
            next_actions=[{"description": "check technical qualification", "command": "python -m tools.eiel qualificationManager check --mission MIS-... --store"}],
        )
    if qualification_errors:
        raise ActorError(
            "technical qualification references are not current admitted authority",
            code="technical_qualification_invalid", problems=qualification_errors, warnings=qualification_warnings,
            next_actions=[{"description": "check and bind current technical authority", "command": "python -m tools.eiel qualificationManager check --mission MIS-... --store"}],
            related_records=qualification_refs,
        )
    if source_errors:
        raise ActorError("campaign source identity does not match", code="source_identity_mismatch", problems=source_errors)
    if inspection["readiness"] != "READY_FOR_ASSESSMENT":
        reasons, problems, actions = _inspection_guidance(inspection, target=str(manifest))
        raise ActorError("campaign cannot be assessed", code=reasons[0] if reasons else "campaign_not_ready", problems=problems, next_actions=actions)
    profile_name = str(args.profile or inspection.get("profile") or "")
    profile = _profile_state(profile_name)
    if not profile.get("implemented"):
        raise ActorError(
            f"evidence profile decision logic is not implemented: {profile_name}",
            code="decision_profile_not_implemented",
            problems=["reserved or unknown profiles cannot be assessed by prose inference"],
        )
    assessment_id = stable_id("EDM", {"manifest_sha256": inspection["manifest_sha256"], "profile": profile_name})
    generated_at = utc_now(args.generated_at)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else _configured_evidence_root(args.evidence_root) / "managed" / assessment_id
    )
    result = generate_campaign_decision_report(manifest, profile_name, out_dir, generated_at=generated_at, deterministic=args.deterministic)
    decision = dict(result["decision"])
    payload = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "assessment_id": assessment_id,
        "campaign_id": inspection.get("campaign_id"),
        "profile": profile_name,
        "procedural_decision": decision.get("decision"),
        "decision_wording": decision.get("decision_wording"),
        "recommended_next_action": decision.get("recommended_next_action"),
        "claims_allowed": decision.get("claims_allowed") or [],
        "claims_not_allowed": decision.get("claims_not_allowed") or [],
        "rationale": decision.get("rationale") or [],
        "manifest_path": str(manifest),
        "manifest_sha256": inspection["manifest_sha256"],
        "source_commit": raw.get("git_commit"),
        "technical_qualification_refs": qualification_refs,
        "mission_id": mission.get("mission_id"),
        "authority_context": mission.get("authority_context"),
        "evidence_preflight_markdown": mission.get("evidence_preflight_markdown"),
        "unbound_tool_validation": not bool(args.mission),
        "schema_ok": bool(dict(result.get("schema_status") or {}).get("ok")),
        "artifacts": dict(result.get("artifacts") or {}),
        "procedural_only": True,
        "scientific_qualification_effect": "none",
        "mechanics_write_effect": "none",
    }
    source_refs: list[dict[str, Any]] = [
        {"kind": "campaign_evidence_manifest", "uri": str(manifest), "sha256": inspection["manifest_sha256"]}
    ]
    if mission.get("source_ref"):
        source_refs.append(dict(mission["source_ref"]))
    qualification_store = ActorRecordStore(actor_state_root(args.state_root), "qualifications")
    source_refs.extend(
        actor_source_ref("technical_qualification", qualification_store.get_read_only(ref))
        for ref in qualification_refs
    )
    record = new_record(
        actor=ACTOR_NAME, record_type=ASSESSMENT_RECORD_TYPE, record_id=assessment_id,
        status=f"DECIDED_{decision.get('decision')}", payload=payload, generated_at=generated_at,
        source_refs=tuple(source_refs),
        action="assess",
    )
    record_path = _store(args).put(record)
    next_actions = [{"description": "attach this assessment to the mission", "command": f"python -m tools.eiel missionPlanner update {args.mission} --evidence-ref {assessment_id}"}] if args.mission else []
    next_actions.append({"description": "export a compact evidence handoff", "command": f"python -m tools.eiel evidenceManager handoff {assessment_id}"})
    return with_guidance(
        {"ok": True, "assessment_id": assessment_id, "record_path": str(record_path), "assessment": payload},
        status=f"DECIDED_{decision.get('decision')}",
        reason_codes=[f"procedural_decision_{str(decision.get('decision')).lower()}"],
        warnings=payload["claims_not_allowed"] + (["assessment is not bound to a Mission Planner authority context; use this only for synthetic/tool-validation fixtures"] if not args.mission else []), next_actions=next_actions,
        related_records=[assessment_id, *qualification_refs],
    ), 0


def _assessment_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    return {
        "assessment_id": record.get("record_id"), "campaign_id": payload.get("campaign_id"),
        "profile": payload.get("profile"), "decision": payload.get("procedural_decision"),
        "status": record.get("status"), "revision": record.get("revision"),
        "updated_at_utc": record.get("updated_at_utc"), "manifest_sha256": payload.get("manifest_sha256"),
        "recommended_next_action": payload.get("recommended_next_action"),
    }


def command_status(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    store = _store(args)
    if args.assessment_id:
        record = store.get(args.assessment_id)
        summary = _assessment_summary(record)
        return with_guidance(
            {"ok": True, "assessment": summary}, status=str(summary["status"]),
            next_actions=[{"description": "show the complete assessment", "command": f"python -m tools.eiel evidenceManager show {args.assessment_id}"}],
            related_records=[args.assessment_id],
        ), 0
    rows = [_assessment_summary(store.get(str(row["record_id"]))) for row in store.list() if row.get("status") != "INVALID"]
    counts = Counter(str(row.get("decision") or "UNKNOWN") for row in rows)
    return with_guidance({"ok": True, "count": len(rows), "decision_counts": dict(sorted(counts.items())), "assessments": rows}, status="ASSESSMENT_HISTORY"), 0


def command_list(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = _store(args).list()
    invalid = any(row.get("status") == "INVALID" for row in rows)
    return with_guidance(
        {"ok": not invalid, "count": len(rows), "records": rows}, status="ASSESSMENT_HISTORY",
        warnings=["invalid stored assessment exists"] if invalid else [],
    ), 0


def command_show(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.assessment_id)
    payload = dict(record.get("payload") or {})
    return with_guidance(
        {"ok": True, "record": record}, status=str(record.get("status")),
        warnings=payload.get("claims_not_allowed") or [],
        next_actions=[{"description": "export the assessment", "command": f"python -m tools.eiel evidenceManager export {args.assessment_id} --export-output assessment.zip"}],
        related_records=[args.assessment_id, *list(payload.get("technical_qualification_refs") or [])],
    ), 0


def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.assessment_id)
    payload = dict(record.get("payload") or {})
    handoff = {
        "schema_version": "eiel_codex_evidence_handoff_v1",
        **operator_metadata(),
        "assessment_id": args.assessment_id,
        "campaign_id": payload.get("campaign_id"),
        "profile": payload.get("profile"),
        "procedural_decision": payload.get("procedural_decision"),
        "decision_wording": payload.get("decision_wording"),
        "supporting_rationale": payload.get("rationale") or [],
        "claims_allowed": payload.get("claims_allowed") or [],
        "claims_not_allowed": payload.get("claims_not_allowed") or [],
        "recommended_next_action": payload.get("recommended_next_action"),
        "source_commit": payload.get("source_commit"),
        "technical_qualification_refs": payload.get("technical_qualification_refs") or [],
        "manifest_path": payload.get("manifest_path"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "artifacts": payload.get("artifacts") or {},
        "mission_id": payload.get("mission_id"),
        "authority_context": payload.get("authority_context"),
        "evidence_preflight_markdown": payload.get("evidence_preflight_markdown") or evidence_preflight_markdown(None),
        "record_sha256": record.get("record_sha256"),
        "scientific_qualification_effect": "none",
    }
    return with_guidance(
        {"ok": True, "handoff": handoff}, status=str(record.get("status")),
        warnings=handoff["claims_not_allowed"], related_records=[args.assessment_id, *handoff["technical_qualification_refs"]],
    ), 0


def command_next(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if _is_evidence_archive(args.target):
        return command_inspect(args)
    if args.target.upper().startswith("EDM-"):
        record = _store(args).get(args.target)
        summary = _assessment_summary(record)
        actions = [{"description": "export or attach this assessment", "command": f"python -m tools.eiel evidenceManager handoff {args.target}"}]
        return with_guidance({"ok": True, "assessment": summary}, status=str(summary["status"]), next_actions=actions, related_records=[args.target]), 0
    manifest = _resolve_manifest(args.target)
    inspection = inspect_manifest(manifest)
    reasons, problems, actions = _inspection_guidance(inspection, target=str(manifest))
    positive = inspection["readiness"] == "READY_FOR_ASSESSMENT"
    return with_guidance({"ok": positive, "inspection": inspection}, status=str(inspection["readiness"]), reason_codes=reasons, problems=problems, next_actions=actions), 0 if positive else 1


def command_export(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    store = _store(args)
    record = store.get(args.assessment_id)
    files = list(store.files_for_record(args.assessment_id))
    missing: list[str] = []
    artifacts = dict(dict(record.get("payload") or {}).get("artifacts") or {})
    for name, raw_path in sorted(artifacts.items()):
        path = Path(str(raw_path))
        if path.is_file():
            files.append((f"artifacts/{path.name}", path))
        else:
            missing.append(f"{name}: {path}")
    if missing and not args.allow_missing:
        raise ActorError(
            "cannot export incomplete assessment", code="assessment_artifact_missing", problems=missing,
            next_actions=[{"description": "regenerate or locate the missing decision artifacts"}], related_records=[args.assessment_id],
        )
    target = Path(args.export_output).expanduser().resolve()
    deterministic_zip(target, files)
    return with_guidance(
        {"ok": not missing, "assessment_id": args.assessment_id, "export_path": str(target), "export_sha256": file_sha256(target), "file_count": len(files), "missing_artifacts": missing},
        status="EXPORTED" if not missing else "EXPORTED_WITH_MISSING", problems=missing,
        related_records=[args.assessment_id],
    ), 0 if not missing else 1


def command_inventory(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    inventory = evidence_attachment_inventory(args.state_root)
    return with_guidance(
        {"ok": True, "inventory": inventory},
        status="EVIDENCE_ATTACHMENT_INVENTORY",
        reason_codes=["orphan_evidence_visible"],
        problems=[f"orphan evidence assessments: {inventory['counts']['orphan']}"] if inventory["counts"]["orphan"] else [],
        next_actions=[{
            "action": "attach_or_import_evidence",
            "description": "Attach each orphan assessment to the intended Work Item or mark it as a tool-validation exception.",
        }] if inventory["counts"]["orphan"] else [],
        related_records=[row["assessment_id"] for row in inventory["records"] if row["attachment_status"] == "ORPHAN"],
    ), 0


def _human(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("assessment"), Mapping):
        row = payload["assessment"]
        return f"Evidence assessment {row.get('assessment_id')}\nDecision: {row.get('procedural_decision', row.get('decision'))}\nCampaign: {row.get('campaign_id')}\nProfile: {row.get('profile')}"
    if isinstance(payload.get("inspection"), Mapping):
        row = payload["inspection"]
        return f"Campaign: {row.get('campaign_id')}\nProfile: {row.get('profile')}\nReadiness: {row.get('readiness')}"
    if "campaigns" in payload:
        return f"Campaign candidates: {payload.get('count', 0)}\n" + "\n".join(f"{row.get('readiness')} {row.get('campaign_id') or row.get('campaign_root')}" for row in payload.get("campaigns") or [])
    if "records" in payload or "assessments" in payload:
        return f"Evidence assessments: {payload.get('count', 0)}"
    if "handoff" in payload:
        return f"Evidence handoff: {payload['handoff'].get('assessment_id')}\nDecision: {payload['handoff'].get('procedural_decision')}"
    return str(dict(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog="python -m tools.eiel evidenceManager", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    discover = sub.add_parser("discover", help="Find evidence manifests and unmanifested metrics roots.")
    _common_options(discover)
    discover.add_argument("--root")

    inspect = sub.add_parser("inspect", help="Inspect campaign readiness without writing evidence.")
    _common_options(inspect)
    inspect.add_argument("target")

    prepare = sub.add_parser("prepare", help="Prepare a draft or valid evidence manifest without guessing missing facts.")
    _common_options(prepare)
    prepare.add_argument("campaign_root")
    prepare.add_argument("--profile")
    prepare.add_argument("--metadata")
    prepare.add_argument("--manifest-output")

    validate = sub.add_parser("validate", help="Validate a campaign evidence manifest, source identity, and qualification binding.")
    _common_options(validate)
    validate.add_argument("target")
    validate.add_argument("--expected-source-commit")
    validate.add_argument("--mission", help="Use mission source and qualification bindings as additional validation context.")

    assess = sub.add_parser("assess", help="Apply an implemented EDL decision profile and retain the assessment.")
    _common_options(assess)
    assess.add_argument("target")
    assess.add_argument("--profile")
    assess.add_argument("--evidence-root")
    assess.add_argument("--out-dir")
    assess.add_argument("--deterministic", action="store_true")
    assess.add_argument("--expected-source-commit")
    assess.add_argument("--mission")

    status = sub.add_parser("status", help="Show one assessment or decision counts.")
    _common_options(status)
    status.add_argument("assessment_id", nargs="?")

    list_parser = sub.add_parser("list", help="List retained evidence assessment records.")
    _common_options(list_parser)

    show = sub.add_parser("show", help="Show the complete retained assessment record.")
    _common_options(show)
    show.add_argument("assessment_id")

    next_parser = sub.add_parser("next", help="Show the next safe evidence action for a campaign or assessment.")
    _common_options(next_parser)
    next_parser.add_argument("target")

    handoff = sub.add_parser("handoff", help="Produce a concise Codex evidence handoff without widening claims.")
    _common_options(handoff)
    handoff.add_argument("assessment_id")

    export = sub.add_parser("export", help="Export an assessment and generated artifacts as a deterministic ZIP.")
    _common_options(export)
    export.add_argument("assessment_id")
    export.add_argument("--export-output", required=True)
    export.add_argument("--allow-missing", action="store_true")

    inventory = sub.add_parser("inventory", help="List attached, orphaned, and tool-validation Evidence Decision records.")
    _common_options(inventory)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    forbidden = {
        "run": "Evidence Manager only reads completed or partial campaign outputs.",
        "execute": "Use Codex and the existing campaign tooling to execute work.",
        "tune": "Evidence Manager never changes mechanics or campaign parameters.",
        "claim": "Scientific qualification is outside the procedural evidence boundary.",
        "publish": "Use the existing Knowledge Librarian after authority review.",
    }
    if raw_args and raw_args[0] in forbidden:
        exc = forbidden_operation_error(f"eiel {ACTOR_NAME}", raw_args[0], forbidden[raw_args[0]])
        emit_actor_error(exc, actor=ACTOR_NAME, command=raw_args[0], raw_args=raw_args)
        return 2
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw_args)
        handlers = {
            "discover": command_discover, "inspect": command_inspect, "prepare": command_prepare,
            "validate": command_validate, "assess": command_assess, "status": command_status,
            "list": command_list, "show": command_show, "next": command_next,
            "handoff": command_handoff, "export": command_export, "inventory": command_inventory,
        }
        payload, code = handlers[args.command](args)
        emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        command = raw_args[0] if raw_args else "unknown"
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw_args,
            parsed_args=args,
        )
        return 2
    except NotImplementedError as exc:
        command = raw_args[0] if raw_args else "unknown"
        wrapped = ActorError(str(exc), code="decision_profile_not_implemented")
        emit_actor_error(
            wrapped,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw_args,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
