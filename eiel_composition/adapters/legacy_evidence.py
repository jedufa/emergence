"""Read-only compatibility adapters for legacy EIEL evidence surfaces.

The existing ``tools.evidence`` implementation remains authoritative.  These
adapters validate immutable artifact references, delegate to its public
functions, and translate only their results into frozen C-130/C-140 DTOs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from emergence_contracts import (
    ArtifactId,
    ArtifactIntegrityStatusV1,
    AssemblyId,
    CellId,
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    ContractErrorCategory,
    ContractIssueV1,
    EvidenceEvaluationV1,
    EvidenceProfileV1,
    NamedDigestV1,
    RunArtifactManifestV1,
    RunArtifactRefV1,
    RunCompletionStatusV1,
    RunId,
    SchemaVersion,
    ValidationResultV1,
    canonical_json_bytes,
)
from tools.evidence.contracts import check_campaign_schema
from tools.evidence.decisions import analyze_profile
from tools.evidence.manifest import ManifestError, load_manifest
from tools.evidence.reports import generate_campaign_decision_report


LEGACY_EVIDENCE_MANIFEST_SCHEMA = SchemaVersion("legacy_campaign_evidence_manifest_v1")
LEGACY_EVIDENCE_INPUT_SCHEMA = SchemaVersion("legacy_evidence_input_v1")
LEGACY_EVIDENCE_COMPARISON_SCHEMA = SchemaVersion("legacy_evidence_comparison_v1")
LEGACY_EVIDENCE_DECISION_SCHEMA = SchemaVersion("legacy_evidence_decision_v1")
LEGACY_REPORT_REQUEST_SCHEMA = SchemaVersion("legacy_evidence_report_request_v1")
LEGACY_REPORT_BUNDLE_SCHEMA = SchemaVersion("legacy_evidence_report_bundle_v1")

_REPORT_REQUEST_KIND = "legacy.evidence.report_request"
_REPORT_BUNDLE_KIND = "legacy.evidence.report_bundle"
_MANIFEST_KIND = "legacy.campaign_evidence_manifest"


def _evidence_error(
    message: str, *, details: Mapping[str, object] | None = None
) -> ContractError:
    return ContractError(
        ContractErrorCategory.EVIDENCE_INVALID,
        owner="eiel_composition.adapters.legacy_evidence",
        message=message,
        details=details,
    )


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise _evidence_error(
            f"{field} must be a non-empty, unpadded string", details={"field": field}
        )
    return value


def _sha256_file(path: Path) -> ContentDigest:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise _evidence_error(
            "cannot read referenced legacy artifact",
            details={"path": str(path), "error": type(error).__name__},
        ) from error
    return ContentDigest("sha256", digest.hexdigest())


def _regular_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise _evidence_error(
            "referenced legacy artifact is not a regular file",
            details={"field": field, "path": str(candidate)},
        )
    return candidate


def _artifact_id(kind: str, path: Path, digest: ContentDigest) -> ArtifactId:
    identity = hashlib.sha256(
        canonical_json_bytes(
            {"artifact_kind": kind, "locator": str(path), "content_digest": digest.to_dict()}
        )
    ).hexdigest()
    return ArtifactId(f"legacy-artifact-{identity}")


def _file_ref(
    path: str | Path, *, artifact_kind: str, artifact_schema: SchemaVersion
) -> RunArtifactRefV1:
    resolved = _regular_file(path, field="artifact_path")
    digest = _sha256_file(resolved)
    return RunArtifactRefV1(
        artifact_id=_artifact_id(artifact_kind, resolved, digest),
        artifact_kind=artifact_kind,
        artifact_schema_version=artifact_schema,
        locator=str(resolved),
        content_digest=digest,
        size_bytes=resolved.stat().st_size,
    )


def _verify_file_ref(reference: RunArtifactRefV1, *, expected_kind: str) -> Path:
    if reference.artifact_kind != expected_kind:
        raise _evidence_error(
            "artifact kind is not supported at this adapter boundary",
            details={"expected": expected_kind, "actual": reference.artifact_kind},
        )
    path = _regular_file(reference.locator, field="locator")
    actual_size = path.stat().st_size
    actual_digest = _sha256_file(path)
    if actual_size != reference.size_bytes or actual_digest != reference.content_digest:
        raise _evidence_error(
            "artifact reference does not match immutable file content",
            details={"path": str(path)},
        )
    return path


def _expand_legacy_paths(
    values: object, *, field: str, allow_missing: bool = False
) -> tuple[Path, ...]:
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise _evidence_error(
            "legacy manifest path field must contain strings", details={"field": field}
        )
    files: list[Path] = []
    for item in values:
        path = Path(item).expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in sorted(path.rglob("*")) if candidate.is_file())
        elif allow_missing:
            continue
        else:
            raise _evidence_error(
                "legacy manifest references a missing artifact",
                details={"field": field, "path": str(path)},
            )
    unique = {str(path): path for path in files}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True)
class LegacyRunArtifactBindingV1:
    """Contract identity absent from the legacy campaign-evidence manifest."""

    run_id: RunId
    cell_id: CellId
    assembly_id: AssemblyId
    assembly_digest: ContentDigest
    runtime_route_id: str
    source_versions: tuple[str, ...]
    output_schema_versions: tuple[SchemaVersion, ...]

    def __post_init__(self) -> None:
        for field_name, expected in (
            ("run_id", RunId),
            ("cell_id", CellId),
            ("assembly_id", AssemblyId),
            ("assembly_digest", ContentDigest),
        ):
            if not isinstance(getattr(self, field_name), expected):
                raise _evidence_error(
                    f"{field_name} has an invalid type", details={"field": field_name}
                )
        _require_text(self.runtime_route_id, field="runtime_route_id")
        if (
            not isinstance(self.source_versions, tuple)
            or not self.source_versions
            or any(not isinstance(item, str) or not item for item in self.source_versions)
            or tuple(sorted(set(self.source_versions))) != self.source_versions
        ):
            raise _evidence_error(
                "source_versions must be a non-empty sorted unique tuple",
                details={"field": "source_versions"},
            )
        if (
            not isinstance(self.output_schema_versions, tuple)
            or not self.output_schema_versions
            or any(not isinstance(item, SchemaVersion) for item in self.output_schema_versions)
        ):
            raise _evidence_error(
                "output_schema_versions must contain explicit schema versions",
                details={"field": "output_schema_versions"},
            )


class LegacyRunArtifactManifestAdapter:
    """Project a validated legacy campaign manifest into immutable C-130 refs."""

    def reference_evidence_manifest(self, manifest_path: str | Path) -> RunArtifactRefV1:
        path = _regular_file(manifest_path, field="manifest_path")
        try:
            load_manifest(path)
        except (ManifestError, OSError) as error:
            raise _evidence_error(
                "legacy evidence manifest is invalid", details={"path": str(path)}
            ) from error
        return _file_ref(
            path,
            artifact_kind=_MANIFEST_KIND,
            artifact_schema=LEGACY_EVIDENCE_MANIFEST_SCHEMA,
        )

    def adapt(
        self, manifest_path: str | Path, binding: LegacyRunArtifactBindingV1
    ) -> RunArtifactManifestV1:
        path = _regular_file(manifest_path, field="manifest_path")
        try:
            legacy = load_manifest(path).data
        except (ManifestError, OSError) as error:
            raise _evidence_error(
                "legacy evidence manifest is invalid", details={"path": str(path)}
            ) from error

        try:
            status = check_campaign_schema(path)
        except (OSError, TypeError, ValueError) as error:
            raise _evidence_error(
                "legacy evidence checker failed while projecting artifacts",
                details={"path": str(path), "error": type(error).__name__},
            ) from error
        run_status = status.get("run_status", [])
        if not isinstance(run_status, list):
            raise _evidence_error("legacy evidence checker returned invalid run status")

        config_candidates = list(
            _expand_legacy_paths(
                legacy.get("config_files"), field="config_files", allow_missing=True
            )
        )
        for row in run_status:
            if isinstance(row, Mapping) and isinstance(row.get("config_path"), str):
                config_path = Path(row["config_path"]).expanduser().resolve()
                if config_path.is_file():
                    config_candidates.append(config_path)
        config_paths = tuple(
            {str(item): item for item in config_candidates}[key]
            for key in sorted({str(item) for item in config_candidates})
        )
        config_digests = tuple(
            NamedDigestV1(
                name=(
                    "legacy.config."
                    + hashlib.sha256(str(config_path).encode("utf-8")).hexdigest()[:20]
                ),
                digest=_sha256_file(config_path),
            )
            for config_path in config_paths
        )

        metric_candidates: dict[str, tuple[Path, str, SchemaVersion]] = {}
        for field, kind in (
            ("output_paths", "legacy.evidence.output"),
            ("harness_manifest_paths", "legacy.harness.manifest"),
            ("harness_summary_paths", "legacy.harness.summary"),
        ):
            for item in _expand_legacy_paths(
                legacy.get(field), field=field, allow_missing=True
            ):
                metric_candidates[str(item)] = (item, kind, LEGACY_EVIDENCE_INPUT_SCHEMA)
        for checked in status.get("checked_files", []):
            if isinstance(checked, str):
                checked_path = Path(checked).expanduser().resolve()
                if checked_path.is_file() and checked_path not in config_paths:
                    metric_candidates.setdefault(
                        str(checked_path),
                        (checked_path, "legacy.evidence.input", LEGACY_EVIDENCE_INPUT_SCHEMA),
                    )
        metric_candidates.pop(str(path), None)
        ordered_metric_paths = [(path, _MANIFEST_KIND, LEGACY_EVIDENCE_MANIFEST_SCHEMA)]
        ordered_metric_paths.extend(
            metric_candidates[key] for key in sorted(metric_candidates)
        )
        metric_refs = tuple(
            _file_ref(item, artifact_kind=kind, artifact_schema=schema)
            for item, kind, schema in ordered_metric_paths
        )

        planned = legacy.get("planned_run_count")
        completed = legacy.get("completed_run_count")
        failed = legacy.get("failed_run_count")
        if failed:
            completion = RunCompletionStatusV1.FAILED
        elif isinstance(planned, int) and completed == planned:
            completion = RunCompletionStatusV1.COMPLETE
        else:
            completion = RunCompletionStatusV1.INCOMPLETE

        return RunArtifactManifestV1(
            run_id=binding.run_id,
            cell_id=binding.cell_id,
            assembly_id=binding.assembly_id,
            assembly_digest=binding.assembly_digest,
            runtime_route_id=binding.runtime_route_id,
            source_versions=binding.source_versions,
            config_digests=config_digests,
            checkpoint_refs=(),
            metric_refs=metric_refs,
            event_refs=(),
            completion_status=completion,
            integrity_status=(
                ArtifactIntegrityStatusV1.UNVERIFIED
                if status.get("missing_files")
                else ArtifactIntegrityStatusV1.VERIFIED
            ),
            output_schema_versions=binding.output_schema_versions,
        )


class LegacyEvidenceEvaluatorAdapter:
    """C-140 facade over current deterministic ``tools.evidence`` functions."""

    def __init__(self) -> None:
        self._manifest_ref: RunArtifactRefV1 | None = None
        self._manifest_path: Path | None = None
        self._last_profile: EvidenceProfileV1 | None = None

    def ingest(self, manifest_or_result_set_ref: RunArtifactRefV1) -> None:
        if not isinstance(manifest_or_result_set_ref, RunArtifactRefV1):
            raise _evidence_error("ingest requires a C-130 artifact reference")
        path = _verify_file_ref(manifest_or_result_set_ref, expected_kind=_MANIFEST_KIND)
        try:
            load_manifest(path)
        except (ManifestError, OSError) as error:
            raise _evidence_error(
                "legacy evidence manifest is invalid", details={"path": str(path)}
            ) from error
        self._manifest_ref = manifest_or_result_set_ref
        self._manifest_path = path
        self._last_profile = None

    def _require_ingested(self) -> tuple[RunArtifactRefV1, Path]:
        if self._manifest_ref is None or self._manifest_path is None:
            raise _evidence_error("no evidence manifest has been ingested")
        _verify_file_ref(self._manifest_ref, expected_kind=_MANIFEST_KIND)
        return self._manifest_ref, self._manifest_path

    def _schema_status(self) -> dict[str, object]:
        _reference, path = self._require_ingested()
        try:
            return dict(check_campaign_schema(path))
        except (OSError, TypeError, ValueError) as error:
            raise _evidence_error(
                "legacy evidence validation failed",
                details={"path": str(path), "error": type(error).__name__},
            ) from error

    def validate(self) -> ValidationResultV1:
        status = self._schema_status()
        issues = tuple(
            ContractIssueV1(
                category=ContractErrorCategory.EVIDENCE_INVALID,
                owner="tools.evidence",
                message=str(message),
            )
            for message in status.get("errors", [])
        )
        return ValidationResultV1(valid=bool(status.get("ok")) and not issues, issues=issues)

    def compare(self, profile: EvidenceProfileV1) -> CanonicalPayloadV1:
        if not isinstance(profile, EvidenceProfileV1):
            raise _evidence_error("compare requires an EvidenceProfileV1")
        try:
            result = analyze_profile(self._schema_status(), profile.profile_id)
        except (KeyError, NotImplementedError) as error:
            raise _evidence_error(
                "legacy evidence profile is unavailable",
                details={"profile_id": profile.profile_id},
            ) from error
        self._last_profile = profile
        return CanonicalPayloadV1.from_value(LEGACY_EVIDENCE_COMPARISON_SCHEMA, result)

    def decide(self, profile: EvidenceProfileV1) -> EvidenceEvaluationV1:
        comparison = self.compare(profile)
        result = comparison.to_value()
        decision = {
            key: result.get(key)
            for key in (
                "profile",
                "decision",
                "rationale",
                "claims_allowed",
                "claims_not_allowed",
                "recommended_next_action",
            )
        }
        return EvidenceEvaluationV1(
            profile=profile,
            validation=self.validate(),
            comparison=comparison,
            decision=CanonicalPayloadV1.from_value(LEGACY_EVIDENCE_DECISION_SCHEMA, decision),
        )

    def report_output_request(self, output_locator: str | Path) -> RunArtifactRefV1:
        reference, _path = self._require_ingested()
        if self._last_profile is None:
            raise _evidence_error("decide must be called before requesting report output")
        output = Path(output_locator).expanduser().resolve()
        request = {
            "manifest_digest": reference.content_digest.to_dict(),
            "profile_id": self._last_profile.profile_id,
            "output_locator": str(output),
            "deterministic": True,
        }
        digest = ContentDigest("sha256", hashlib.sha256(canonical_json_bytes(request)).hexdigest())
        return RunArtifactRefV1(
            artifact_id=ArtifactId(f"legacy-report-request-{digest.hex}"),
            artifact_kind=_REPORT_REQUEST_KIND,
            artifact_schema_version=LEGACY_REPORT_REQUEST_SCHEMA,
            locator=str(output),
            content_digest=digest,
            size_bytes=0,
        )

    def materialize_report(self, output_ref: RunArtifactRefV1) -> RunArtifactRefV1:
        _reference, manifest_path = self._require_ingested()
        profile = self._last_profile
        if profile is None:
            raise _evidence_error("decide must be called before report materialization")
        expected = self.report_output_request(output_ref.locator)
        if output_ref != expected:
            raise _evidence_error("report output request is stale or invalid")
        output = Path(output_ref.locator)
        if output.exists():
            raise _evidence_error(
                "report output locator already exists", details={"path": str(output)}
            )
        try:
            generate_campaign_decision_report(
                manifest_path, profile.profile_id, output, deterministic=True
            )
        except (KeyError, NotImplementedError, OSError, TypeError, ValueError) as error:
            raise _evidence_error(
                "legacy deterministic report generation failed",
                details={"profile_id": profile.profile_id, "error": type(error).__name__},
            ) from error
        entries: list[dict[str, object]] = []
        total_size = 0
        for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
            size = path.stat().st_size
            total_size += size
            entries.append(
                {
                    "path": str(path.relative_to(output)),
                    "size_bytes": size,
                    "sha256": _sha256_file(path).hex,
                }
            )
        bundle_digest = ContentDigest(
            "sha256", hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
        )
        return RunArtifactRefV1(
            artifact_id=ArtifactId(f"legacy-report-bundle-{bundle_digest.hex}"),
            artifact_kind=_REPORT_BUNDLE_KIND,
            artifact_schema_version=LEGACY_REPORT_BUNDLE_SCHEMA,
            locator=str(output),
            content_digest=bundle_digest,
            size_bytes=total_size,
        )
