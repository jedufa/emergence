"""Post-qualification Runability Closure gate and failure taxonomy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Final

from engine.campaigns.canonical import canonical_sha256
from engine.qrr.qualified_runtime_release_v1 import (
    QualifiedRuntimeReleaseError,
    load_qualified_runtime_release_v1,
)


RUNABILITY_CLOSURE_SCHEMA_VERSION: Final[str] = "eiel_runability_closure_v1"
RUNABILITY_FAILURE_CLASSES: Final[tuple[str, ...]] = (
    "QRR_BUILD_FAILURE",
    "QRR_INTEGRITY_FAILURE",
    "RUNABILITY_CONTRACT_FAILURE",
    "QRR_REPRESENTABILITY_BLOCKED",
    "RESOURCE_BLOCKED",
    "CAPABILITY_BLOCKED",
    "ENVIRONMENT_BLOCKED",
    "CHECKPOINT_RESUME_BLOCKED",
    "POST_QUALIFICATION_RUNABILITY_ESCAPE",
    "SCIENCE_DESIGN_BLOCKED",
)


class RunabilityClosureError(ValueError):
    """The run cannot support a truthful RUNABILITY_CLOSED decision."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_runability_failure_v1(
    exc: BaseException | str,
    *,
    stage: str,
) -> str:
    """Classify a failure without converting resource/science limits to bugs."""

    message = str(exc).lower()
    if "representab" in message:
        return "QRR_REPRESENTABILITY_BLOCKED"
    if stage == "build":
        return "QRR_BUILD_FAILURE"
    if "integrity" in message or "sha-256" in message or "sha256" in message:
        return "QRR_INTEGRITY_FAILURE"
    if "environment" in message or "pythonpath" in message:
        return "ENVIRONMENT_BLOCKED"
    if "resource" in message or "vram" in message or "cuda memory" in message:
        return "RESOURCE_BLOCKED"
    if "checkpoint" in message or "resume" in message or "restore" in message:
        return "CHECKPOINT_RESUME_BLOCKED"
    if "capability" in message or "envelope" in message or "route" in message:
        return "CAPABILITY_BLOCKED"
    if "science_design" in message or "scientific design" in message:
        return "SCIENCE_DESIGN_BLOCKED"
    if "source edit" in message or "rebuild" in message or "reconstruct" in message:
        return "POST_QUALIFICATION_RUNABILITY_ESCAPE"
    return "RUNABILITY_CONTRACT_FAILURE"


@dataclass(frozen=True, slots=True)
class RunabilityClosureReceiptV1:
    release_id: str
    run_root: str
    decision: str
    checks: Mapping[str, bool]
    failure_class: str | None
    reasons: tuple[str, ...]
    created_at_utc: str

    def payload(self) -> dict[str, Any]:
        core = {
            "schema_version": RUNABILITY_CLOSURE_SCHEMA_VERSION,
            "release_id": self.release_id,
            "run_root": self.run_root,
            "decision": self.decision,
            "checks": dict(sorted(self.checks.items())),
            "failure_class": self.failure_class,
            "reasons": list(self.reasons),
            "created_at_utc": self.created_at_utc,
        }
        return {**core, "receipt_sha256": canonical_sha256(core)}


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunabilityClosureError(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise RunabilityClosureError(f"{name} is malformed: {path}")
    return dict(value)


def assess_runability_closure_v1(
    *,
    release_root: Path,
    run_root: Path,
    historical_suite_receipt: Mapping[str, Any],
) -> RunabilityClosureReceiptV1:
    """Assess a completed release-only run; never grant Qualification."""

    reasons: list[str] = []
    checks: dict[str, bool] = {}
    failure_class: str | None = None
    historical_suite_id = "unavailable"
    historical_inventory_sha256 = "unavailable"
    release_envelope_sha256 = "unavailable"
    try:
        release = load_qualified_runtime_release_v1(release_root)
    except QualifiedRuntimeReleaseError as exc:
        failure_class = classify_runability_failure_v1(exc, stage="verify")
        reasons.append(str(exc))
        release_id = "unknown"
    else:
        release_id = release.release_id
        historical_suite_id = release.historical_escape_suite_id
        historical_inventory_sha256 = release.historical_escape_suite_inventory_sha256
        release_envelope_sha256 = str(
            release.qualified_execution_envelope.get("envelope_sha256", "")
        )
        checks["qrr_integrity_and_environment_exact"] = True
    root = run_root.expanduser().resolve()
    try:
        binding = _read_mapping(root / "qrr_run_binding.json", name="QRR run binding")
        run_receipt = _read_mapping(
            root / "qrr_run_receipt.json", name="QRR run receipt"
        )
        representability_receipt = _read_mapping(
            root / "qrr_representability_receipt.json",
            name="QRR representability receipt",
        )
    except RunabilityClosureError as exc:
        reasons.append(str(exc))
        failure_class = failure_class or "RUNABILITY_CONTRACT_FAILURE"
        binding = {}
        run_receipt = {}
        representability_receipt = {}
    checks["release_identity_attributed"] = (
        binding.get("release_id") == release_id
        and run_receipt.get("release_id") == release_id
    )
    checks["canonical_launcher_only"] = (
        binding.get("launcher") == "qrr_v1"
        and binding.get("route_reconstruction_steps") == 0
    )
    checks["no_source_edits_or_rebuilds"] = (
        binding.get("source_edits_after_qrr_creation") == 0
        and binding.get("native_rebuilds") == 0
        and binding.get("runtime_jit_builds") == 0
        and binding.get("qualification_actions") == 0
    )
    representation_digest = binding.get("representability_receipt_sha256")
    representation_core = {
        name: value
        for name, value in representability_receipt.items()
        if name != "receipt_sha256"
    }
    checks["experiment_representability_passed"] = (
        isinstance(representation_digest, str)
        and len(representation_digest) == 64
        and run_receipt.get("representability_receipt_sha256")
        == representation_digest
        and representability_receipt.get("schema_version")
        == "eiel_qrr_experiment_representability_receipt_v1"
        and representability_receipt.get("status") == "PASS"
        and representability_receipt.get("failure_class") is None
        and representability_receipt.get("envelope_sha256")
        == release_envelope_sha256
        and representability_receipt.get("receipt_sha256")
        == representation_digest
        and canonical_sha256(representation_core) == representation_digest
    )
    checks["bounded_run_completed"] = run_receipt.get("status") == "completed"
    checks["checkpoint_resume_exact"] = (
        run_receipt.get("checkpoint_resume_exercised") is True
        and run_receipt.get("checkpoint_resume_exact") is True
    )
    checks["historical_escape_suite_green"] = (
        historical_suite_receipt.get("suite_id") == historical_suite_id
        and historical_suite_receipt.get("inventory_sha256")
        == historical_inventory_sha256
        and historical_suite_receipt.get("passed") is True
    )
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        reasons.append("failed runability checks: " + ", ".join(sorted(failed)))
        if failure_class is None:
            if "checkpoint_resume_exact" in failed:
                failure_class = "CHECKPOINT_RESUME_BLOCKED"
            elif "experiment_representability_passed" in failed:
                failure_class = "QRR_REPRESENTABILITY_BLOCKED"
            elif "no_source_edits_or_rebuilds" in failed:
                failure_class = "POST_QUALIFICATION_RUNABILITY_ESCAPE"
            else:
                failure_class = "RUNABILITY_CONTRACT_FAILURE"
    decision = "RUNABILITY_CLOSED" if not reasons else "RUNABILITY_NOT_CLOSED"
    return RunabilityClosureReceiptV1(
        release_id=release_id,
        run_root=str(root),
        decision=decision,
        checks=checks,
        failure_class=failure_class,
        reasons=tuple(reasons),
        created_at_utc=_now(),
    )


__all__ = [
    "RUNABILITY_CLOSURE_SCHEMA_VERSION",
    "RUNABILITY_FAILURE_CLASSES",
    "RunabilityClosureError",
    "RunabilityClosureReceiptV1",
    "assess_runability_closure_v1",
    "classify_runability_failure_v1",
]
