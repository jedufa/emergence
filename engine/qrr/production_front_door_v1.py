"""Science-shaped active-QRR resolution, admission, launch, and resume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Final

from engine.campaigns.canonical import canonical_sha256, pretty_json
from engine.campaigns.long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    SCIENCE_ROUTE_QUALIFIED,
)
from engine.campaigns.compiler import load_campaign_spec
from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    validate_run_artifact_lifecycle_contract_v1,
)
from engine.campaigns.baseline_v1 import (
    BASELINE_ADMISSION_SCHEMA_VERSION,
    BASELINE_CONFORMANCE_SCHEMA_VERSION,
    BaselineDeclarationV1,
    BaselineError,
    baseline_admission_shadow_v1,
    baseline_result_binding_v1,
    build_baseline_admission_receipt_v1,
    build_baseline_conformance_receipt_v1,
    validate_baseline_purpose_v1,
    validate_baseline_receipt_binding_v1,
    write_baseline_receipt_v1,
)
from engine.campaigns.canopy_supervision_v1 import (
    CANOPY_RELAY_POLL_SECONDS,
    CANOPY_SUPERVISION_SCHEMA_VERSION,
    DEFAULT_CANOPY_RELAY_ROOT,
    CanopySupervisionError,
    canopy_relay_process_matches_v1,
    mark_canopy_source_interrupted_v1,
    record_canopy_observability_degraded_v1,
    start_canopy_relays_v1,
)
from engine.qrr.qualified_execution_envelope_v1 import (
    load_qualified_execution_envelope_v1,
)
from engine.qrr.qualified_runtime_release_v1 import (
    DEFAULT_QRR_ROOT,
    DEFAULT_QRR_RUN_ROOT,
    load_qualified_runtime_release_v1,
)
ACTIVE_QRR_REGISTRY_SCHEMA_VERSION: Final[str] = "eiel_active_qrr_registry_v1"
PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_production_science_request_v1"
)
PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION: Final[str] = (
    "eiel_production_science_request_v2"
)
PRODUCTION_SCIENCE_RESULT_V2_SCHEMA_VERSION: Final[str] = (
    "eiel_production_science_front_door_result_v2"
)
PRODUCTION_SCIENCE_REQUEST_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_production_science_request_receipt_v1"
)


def _qrr_baseline_command_allowed_v1(
    *,
    admission_receipt: Mapping[str, Any] | None,
    command: str,
    request_sha256: str,
    plan_sha256: str,
) -> bool:
    """Shadow QRR's retained Baseline policy without cutting it over."""

    if admission_receipt is None:
        return True
    execution_required = command in {"preflight", "run"}
    legacy_allowed = (
        admission_receipt.get("admitted") is True
        if execution_required
        else admission_receipt.get("configuration_eligible") is True
    )
    baseline_admission_shadow_v1(
        front_door_id=PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        intent="execution" if execution_required else "configuration",
        request_sha256=request_sha256,
        plan_sha256=plan_sha256,
        admission_receipt=admission_receipt,
        execution_required=execution_required,
    )
    return legacy_allowed


class ProductionScienceFrontDoorError(ValueError):
    """Science intent cannot safely resolve to one executable QRR."""


class _SealedQrrInvocationError(ProductionScienceFrontDoorError):
    """The sealed child did not return its exact machine-readable contract."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionScienceFrontDoorError(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ProductionScienceFrontDoorError(f"{name} must be an object")
    return dict(value)


def _read_bound_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProductionScienceFrontDoorError(
            f"{name} is not a regular bound file: {path}"
        )
    return _read_mapping(path, name=name)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(path)


def _write_or_verify_bound_json_v1(
    path: Path, value: Mapping[str, Any], *, name: str
) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ProductionScienceFrontDoorError(
                f"{name} is not a regular bound file: {path}"
            )
        if canonical_sha256(_read_mapping(path, name=name)) != canonical_sha256(
            value
        ):
            raise ProductionScienceFrontDoorError(
                f"run root is bound to another {name}"
            )
        return
    _atomic_json(path, value)


def _bind_science_run_root_v1(
    *,
    destination: Path,
    normalized_request: Mapping[str, Any],
    experiment: Mapping[str, Any],
    resolution: Mapping[str, Any],
    release_root: Path,
) -> dict[str, Any]:
    """Bind a run root before any sealed QRR command can mutate it."""

    destination.mkdir(parents=True, exist_ok=True)
    request_sha256 = normalized_request.get("request_sha256")
    if not isinstance(request_sha256, str):
        request_sha256 = canonical_sha256(normalized_request)
    expected = {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_RECEIPT_SCHEMA_VERSION,
        "request_schema_version": normalized_request.get("schema_version"),
        "request_sha256": request_sha256,
        "release_id": release_root.name,
        "release_root": str(release_root.resolve()),
        "resolution_manifest_sha256": resolution.get("manifest_sha256"),
        "resolution_receipt_sha256": resolution.get("receipt_sha256"),
        "resolution_sha256": canonical_sha256(resolution),
        "experiment_sha256": canonical_sha256(experiment),
        "run_root": str(destination),
    }
    receipt_path = destination / "production_science_request_receipt.json"
    if receipt_path.exists():
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise ProductionScienceFrontDoorError(
                "production science request receipt is not a regular file"
            )
        receipt = _read_mapping(
            receipt_path, name="production science request receipt"
        )
        supplied = receipt.get("receipt_sha256")
        core = dict(receipt)
        core.pop("receipt_sha256", None)
        if supplied != canonical_sha256(core):
            raise ProductionScienceFrontDoorError(
                "production science request receipt SHA-256 mismatch"
            )
        mismatches = sorted(
            key for key, value in expected.items() if receipt.get(key) != value
        )
        if mismatches:
            raise ProductionScienceFrontDoorError(
                "run root is bound to another production science request: "
                + ", ".join(mismatches)
            )
    else:
        if any(destination.iterdir()):
            raise ProductionScienceFrontDoorError(
                "run root contains files but no canonical production science "
                "request receipt"
            )
        core = {**expected, "created_at_utc": _now()}
        receipt = {**core, "receipt_sha256": canonical_sha256(core)}
        _atomic_json(receipt_path, receipt)
    _write_or_verify_bound_json_v1(
        destination / "production_science_qrr_experiment.json",
        experiment,
        name="production science QRR experiment",
    )
    _write_or_verify_bound_json_v1(
        destination / "active_qrr_resolution_receipt.json",
        resolution,
        name="active QRR resolution receipt",
    )
    return receipt


def _existing_science_result_v1(
    *,
    destination: Path,
    request_receipt: Mapping[str, Any],
    normalized_request: Mapping[str, Any],
    command: str,
) -> dict[str, Any] | None:
    path = destination / "production_science_front_door_result.json"
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() or path.is_symlink():
        raise ProductionScienceFrontDoorError(
            "production science result is not a regular file"
        )
    result = _read_mapping(path, name="production science front-door result")
    supplied = result.get("result_sha256")
    core = dict(result)
    core.pop("result_sha256", None)
    if supplied != canonical_sha256(core):
        raise ProductionScienceFrontDoorError(
            "production science result SHA-256 mismatch"
        )
    expected_schema = (
        PRODUCTION_SCIENCE_RESULT_V2_SCHEMA_VERSION
        if request_receipt.get("request_schema_version")
        == PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION
        else "eiel_production_science_front_door_result_v1"
    )
    if result.get("schema_version") != expected_schema:
        raise ProductionScienceFrontDoorError(
            "production science result schema does not match its bound request"
        )
    if result.get("request_receipt_sha256") != request_receipt.get(
        "receipt_sha256"
    ):
        raise ProductionScienceFrontDoorError(
            "production science result request binding mismatch"
        )
    qrr_result = result.get("qrr_result")
    if not isinstance(qrr_result, Mapping):
        raise ProductionScienceFrontDoorError(
            "production science result lacks its nested QRR result"
        )
    expected_release_id = request_receipt.get("release_id")
    expected_run_root = str(destination)
    expected_resolution_sha256 = request_receipt.get(
        "resolution_receipt_sha256"
    )
    if (
        result.get("release_id") != expected_release_id
        or result.get("run_root") != expected_run_root
        or result.get("resolution_receipt_sha256")
        != expected_resolution_sha256
        or (
            qrr_result.get("release_id") is not None
            and qrr_result.get("release_id") != expected_release_id
        )
        or (
            qrr_result.get("run_root") is not None
            and qrr_result.get("run_root") != expected_run_root
        )
    ):
        raise ProductionScienceFrontDoorError(
            "production science retained result identity mismatch"
        )
    baseline_permits_result = True
    if (
        normalized_request.get("schema_version")
        == PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION
    ):
        plan = _read_bound_mapping(
            destination / "execution_plan.json",
            name="bound V2 production science plan",
        )
        preflight_path = destination / "preflight_receipt.json"
        if preflight_path.exists() or preflight_path.is_symlink():
            preflight = _read_bound_mapping(
                preflight_path, name="bound V2 production preflight"
            )
        else:
            preflight = None
        declaration = BaselineDeclarationV1.from_value(
            normalized_request["baseline"]
        )
        admission = validate_baseline_receipt_binding_v1(
            _read_bound_mapping(
                destination / "baseline_admission_receipt.json",
                name="bound V2 Baseline admission receipt",
            ),
            schema_version=BASELINE_ADMISSION_SCHEMA_VERSION,
            declaration=declaration,
            plan=plan,
            request_sha256=str(normalized_request["request_sha256"]),
            front_door_id=PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
            preflight=preflight,
            stage="admission",
            artifact_lifecycle_explicit="artifact_lifecycle"
            in normalized_request,
            canopy_enabled=normalized_request["observability"][
                "living_lineage_canopy"
            ],
        )
        conformance: dict[str, Any] | None = None
        if result.get("command") == "run":
            qrr_receipt_path = destination / "qrr_run_receipt.json"
            if not qrr_receipt_path.is_file() or qrr_receipt_path.is_symlink():
                raise ProductionScienceFrontDoorError(
                    "existing V2 run lacks its durable QRR run receipt"
                )
            qrr_receipt = _read_mapping(
                qrr_receipt_path, name="bound V2 QRR run receipt"
            )
            nested_qrr_receipt = qrr_result.get("run_receipt")
            execution_receipt = qrr_receipt.get("execution_receipt")
            if (
                not isinstance(nested_qrr_receipt, Mapping)
                or dict(nested_qrr_receipt) != qrr_receipt
                or qrr_receipt.get("schema_version")
                != "eiel_qrr_run_receipt_v1"
                or qrr_receipt.get("release_id") != expected_release_id
                or qrr_receipt.get("plan_sha256") != plan.get("plan_sha256")
                or not isinstance(execution_receipt, Mapping)
                or execution_receipt.get("plan_sha256")
                != plan.get("plan_sha256")
                or execution_receipt.get("status") != qrr_receipt.get("status")
                or execution_receipt.get("ok") is not (qrr_result.get("ok") is True)
                or (
                    qrr_result.get("ok") is True
                    and qrr_receipt.get("status") != "completed"
                )
                or (
                    qrr_result.get("ok") is False
                    and qrr_receipt.get("status") == "completed"
                )
            ):
                raise ProductionScienceFrontDoorError(
                    "production science retained QRR receipt binding mismatch"
                )
            execution_status = (
                "completed"
                if qrr_result.get("ok") is True
                else "failed"
            )
            canopy: Mapping[str, Any] | None = None
            if normalized_request["observability"]["living_lineage_canopy"]:
                canopy_path = destination / "canopy_relay_supervision.json"
                if not canopy_path.is_file() or canopy_path.is_symlink():
                    raise ProductionScienceFrontDoorError(
                        "existing V2 run lacks its durable Canopy supervision receipt"
                    )
                canopy = _read_mapping(
                    canopy_path, name="bound V2 Canopy supervision receipt"
                )
                if result.get("canopy_relay_supervision") != canopy:
                    raise ProductionScienceFrontDoorError(
                        "production science retained Canopy receipt binding mismatch"
                    )
            else:
                embedded_canopy = result.get("canopy_relay_supervision")
                if (
                    not isinstance(embedded_canopy, Mapping)
                    or embedded_canopy.get("schema_version")
                    != CANOPY_SUPERVISION_SCHEMA_VERSION
                    or embedded_canopy.get("state") != "disabled_by_request"
                    or embedded_canopy.get("authority")
                    != "read_only_external_projection_v1"
                    or embedded_canopy.get("simulation_state_mutation") is not False
                    or embedded_canopy.get("run_root") != expected_run_root
                    or embedded_canopy.get("relays") != []
                ):
                    raise ProductionScienceFrontDoorError(
                        "production science retained disabled Canopy binding mismatch"
                    )
                canopy = embedded_canopy
            conformance = validate_baseline_receipt_binding_v1(
                _read_bound_mapping(
                    destination / "baseline_conformance_receipt.json",
                    name="bound V2 Baseline conformance receipt",
                ),
                schema_version=BASELINE_CONFORMANCE_SCHEMA_VERSION,
                declaration=declaration,
                plan=plan,
                request_sha256=str(normalized_request["request_sha256"]),
                front_door_id=PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
                preflight=preflight,
                stage="terminal",
                artifact_lifecycle_explicit="artifact_lifecycle"
                in normalized_request,
                canopy_enabled=normalized_request["observability"][
                    "living_lineage_canopy"
                ],
                canopy_supervision=(
                    canopy if isinstance(canopy, Mapping) else None
                ),
                admission_receipt_sha256=str(admission["receipt_sha256"]),
                execution_status=execution_status,
            )
        if result.get("baseline") != baseline_result_binding_v1(
            admission_receipt=admission,
            conformance_receipt=conformance,
        ):
            raise ProductionScienceFrontDoorError(
                "production science result Baseline receipt binding mismatch"
            )
        baseline_permits_result = _qrr_baseline_command_allowed_v1(
            admission_receipt=admission,
            command=str(result.get("command")),
            request_sha256=str(normalized_request["request_sha256"]),
            plan_sha256=str(plan["plan_sha256"]),
        )
    expected_ok = qrr_result.get("ok") is True and baseline_permits_result
    if result.get("ok") is not expected_ok:
        raise ProductionScienceFrontDoorError(
            "production science retained result outcome mismatch"
        )
    prior_command = result.get("command")
    if prior_command == command and (command != "run" or result.get("ok") is True):
        return result
    if prior_command == "run" and result.get("ok") is True:
        raise ProductionScienceFrontDoorError(
            "completed production science run root cannot be reused for another command"
        )
    return None


def _load_bound_science_resolution_v1(
    *,
    destination: Path,
    normalized_request: Mapping[str, Any],
    qrr_root: Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve an exact continuation from its immutable request binding."""

    receipt_path = destination / "production_science_request_receipt.json"
    if not receipt_path.exists():
        return None
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ProductionScienceFrontDoorError(
            "production science request receipt is not a regular file"
        )
    receipt = _read_mapping(receipt_path, name="production science request receipt")
    supplied = receipt.get("receipt_sha256")
    core = dict(receipt)
    core.pop("receipt_sha256", None)
    if supplied != canonical_sha256(core):
        raise ProductionScienceFrontDoorError(
            "production science request receipt SHA-256 mismatch"
        )
    request_sha256 = normalized_request.get("request_sha256")
    if not isinstance(request_sha256, str):
        request_sha256 = canonical_sha256(normalized_request)
    if (
        receipt.get("request_schema_version")
        != normalized_request.get("schema_version")
        or receipt.get("request_sha256") != request_sha256
        or receipt.get("run_root") != str(destination)
    ):
        raise ProductionScienceFrontDoorError(
            "run root is bound to another production science request"
        )
    release_value = receipt.get("release_root")
    if not isinstance(release_value, str):
        raise ProductionScienceFrontDoorError(
            "bound production science release root is absent"
        )
    release_root = Path(release_value).expanduser().resolve()
    if release_root.parent != qrr_root.expanduser().resolve():
        raise ProductionScienceFrontDoorError(
            "bound production science release is outside the requested QRR root"
        )
    release = load_qualified_runtime_release_v1(release_root)
    envelope = _require_science_release_authority_v1(release)
    if (
        release.release_id != receipt.get("release_id")
        or release.payload()["manifest_sha256"]
        != receipt.get("resolution_manifest_sha256")
    ):
        raise ProductionScienceFrontDoorError(
            "bound production science release identity changed"
        )
    resolution_path = destination / "active_qrr_resolution_receipt.json"
    if not resolution_path.is_file() or resolution_path.is_symlink():
        raise ProductionScienceFrontDoorError(
            "bound active QRR resolution receipt is absent"
        )
    resolution = _read_mapping(
        resolution_path, name="bound active QRR resolution receipt"
    )
    resolution_supplied = resolution.get("receipt_sha256")
    resolution_core = dict(resolution)
    resolution_core.pop("receipt_sha256", None)
    if resolution_supplied != canonical_sha256(resolution_core):
        raise ProductionScienceFrontDoorError(
            "bound active QRR resolution receipt SHA-256 mismatch"
        )
    if (
        receipt.get("resolution_receipt_sha256") != resolution_supplied
        or receipt.get("resolution_sha256") != canonical_sha256(resolution)
        or resolution.get("schema_version")
        != "eiel_active_qrr_resolution_receipt_v1"
        or resolution.get("status") != "PASS"
        or resolution.get("manual_internal_decisions_required") != 0
        or resolution.get("reasons") != []
        or resolution.get("release_id") != release.release_id
        or resolution.get("manifest_sha256")
        != release.payload()["manifest_sha256"]
        or resolution.get("science_request_sha256") != request_sha256
        or resolution.get("qualification_evidence_binding_sha256")
        != envelope.qualification_evidence_binding["binding_sha256"]
    ):
        raise ProductionScienceFrontDoorError(
            "bound active QRR resolution no longer matches its request and release"
        )
    return release_root, resolution


def _v1_qrr_continuation_authorized_v1(
    *, destination: Path, release_root: Path
) -> bool:
    binding_path = destination / "qrr_run_binding.json"
    plan_path = destination / "execution_plan.json"
    if (
        not binding_path.is_file()
        or binding_path.is_symlink()
        or not plan_path.is_file()
        or plan_path.is_symlink()
    ):
        return False
    binding = _read_mapping(binding_path, name="V1 QRR continuation binding")
    plan = _read_mapping(plan_path, name="V1 QRR continuation plan")
    if (
        binding.get("schema_version") != "eiel_qrr_run_binding_v1"
        or binding.get("release_id") != release_root.name
        or not isinstance(binding.get("plan_sha256"), str)
        or binding.get("plan_sha256") != plan.get("plan_sha256")
    ):
        return False
    receipt_path = destination / "qrr_run_receipt.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    receipt = _read_mapping(receipt_path, name="V1 QRR continuation receipt")
    if receipt.get("status") == "completed":
        return False
    result_path = destination / "production_science_front_door_result.json"
    if not result_path.is_file() or result_path.is_symlink():
        return False
    result = _read_mapping(result_path, name="V1 continuation front-door result")
    supplied = result.get("result_sha256")
    core = dict(result)
    core.pop("result_sha256", None)
    if supplied != canonical_sha256(core):
        raise ProductionScienceFrontDoorError(
            "V1 continuation front-door result SHA-256 mismatch"
        )
    qrr_result = result.get("qrr_result")
    nested_receipt = (
        qrr_result.get("run_receipt")
        if isinstance(qrr_result, Mapping)
        else None
    )
    execution_receipt = (
        nested_receipt.get("execution_receipt")
        if isinstance(nested_receipt, Mapping)
        else None
    )
    return bool(
        result.get("schema_version")
        == "eiel_production_science_front_door_result_v1"
        and result.get("command") == "run"
        and result.get("release_id") == release_root.name
        and result.get("run_root") == str(destination)
        and result.get("ok") is False
        and isinstance(qrr_result, Mapping)
        and qrr_result.get("ok") is False
        and isinstance(nested_receipt, Mapping)
        and dict(nested_receipt) == receipt
        and nested_receipt.get("schema_version") == "eiel_qrr_run_receipt_v1"
        and nested_receipt.get("release_id") == release_root.name
        and nested_receipt.get("plan_sha256") == plan.get("plan_sha256")
        and nested_receipt.get("status") == "interrupted"
        and isinstance(execution_receipt, Mapping)
        and execution_receipt.get("ok") is False
        and execution_receipt.get("plan_sha256") == plan.get("plan_sha256")
        and execution_receipt.get("status") == "interrupted"
    )


def active_registry_path_v1(qrr_root: Path = DEFAULT_QRR_ROOT) -> Path:
    return qrr_root.expanduser().resolve() / "active_qrr_registry_v1.json"


def _require_science_release_authority_v1(release: Any) -> Any:
    """Reject bounded/legacy QRRs before they enter a science authority path."""

    raw_envelope = getattr(release, "qualified_execution_envelope", None)
    if not isinstance(raw_envelope, Mapping):
        raise ProductionScienceFrontDoorError(
            "SCIENCE_AUTHORITY_MISSING: QRR has no frozen qualification envelope"
        )
    try:
        envelope = load_qualified_execution_envelope_v1(raw_envelope)
    except (TypeError, ValueError) as exc:
        raise ProductionScienceFrontDoorError(
            f"SCIENCE_AUTHORITY_INVALID: {exc}"
        ) from exc
    required = (
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
        SCIENCE_ROUTE_QUALIFIED,
    )
    authority = getattr(release, "qualification_authority", None)
    if (
        envelope.qualification_dimensions != required
        or not envelope.qualification_evidence_binding
        or not isinstance(authority, Mapping)
        or authority.get("qualification_dimensions") != list(required)
        or authority.get("qualification_evidence_binding")
        != dict(envelope.qualification_evidence_binding)
        or authority.get("execution_lane") != "production_science"
        or authority.get("production_science_authorized") is not True
    ):
        raise ProductionScienceFrontDoorError(
            "SCIENCE_ROUTE_QUALIFIED is required for production-science authority"
        )
    return envelope


def activate_qrr_v1(
    release_root: Path, *, qrr_root: Path = DEFAULT_QRR_ROOT
) -> dict[str, Any]:
    """Atomically activate one verified release without mutating sealed QRRs."""

    root = qrr_root.expanduser().resolve()
    release_path = release_root.expanduser().resolve()
    release = load_qualified_runtime_release_v1(release_path)
    _require_science_release_authority_v1(release)
    if release_path.parent != root:
        raise ProductionScienceFrontDoorError(
            "release is outside the QRR registry root"
        )
    registry_path = active_registry_path_v1(root)
    previous: dict[str, Any] | None = None
    if registry_path.is_file():
        previous = _read_mapping(registry_path, name="active QRR registry")
        _validate_registry_v1(previous)
    rows = {
        str(row["release_id"]): dict(row)
        for row in (previous or {}).get("releases", ())
        if isinstance(row, Mapping) and isinstance(row.get("release_id"), str)
    }
    prior_active = (previous or {}).get("active_release_id")
    if (
        isinstance(prior_active, str)
        and prior_active in rows
        and prior_active != release.release_id
    ):
        rows[prior_active] = {
            **rows[prior_active],
            "status": "superseded",
            "superseded_by": release.release_id,
        }
    manifest_sha256 = release.payload()["manifest_sha256"]
    rows[release.release_id] = {
        "release_id": release.release_id,
        "manifest_sha256": manifest_sha256,
        "status": "active",
        "superseded_by": None,
        "activated_at_utc": _now(),
    }
    core = {
        "schema_version": ACTIVE_QRR_REGISTRY_SCHEMA_VERSION,
        "active_release_id": release.release_id,
        "releases": [rows[name] for name in sorted(rows)],
        "updated_at_utc": _now(),
    }
    registry = {**core, "registry_sha256": canonical_sha256(core)}
    _atomic_json(registry_path, registry)
    return registry


def _validate_registry_v1(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "registry_sha256"}
    rows = value.get("releases")
    active = value.get("active_release_id")
    if (
        value.get("schema_version") != ACTIVE_QRR_REGISTRY_SCHEMA_VERSION
        or value.get("registry_sha256") != canonical_sha256(core)
        or not isinstance(rows, list)
        or not isinstance(active, str)
        or sum(
            isinstance(row, Mapping)
            and row.get("release_id") == active
            and row.get("status") == "active"
            for row in rows
        )
        != 1
        or sum(
            isinstance(row, Mapping) and row.get("status") == "active" for row in rows
        )
        != 1
    ):
        raise ProductionScienceFrontDoorError("active QRR registry is malformed")


def _resolve_active_qrr_normalized_v1(
    request: Mapping[str, Any], *, qrr_root: Path = DEFAULT_QRR_ROOT
) -> tuple[Path, dict[str, Any]]:
    """Resolve exactly one active, intact, capability-matching release."""

    root = qrr_root.expanduser().resolve()
    registry = _read_mapping(active_registry_path_v1(root), name="active QRR registry")
    _validate_registry_v1(registry)
    release_id = str(registry["active_release_id"])
    row = next(
        item
        for item in registry["releases"]
        if isinstance(item, Mapping) and item.get("release_id") == release_id
    )
    release_root = root / release_id
    release = load_qualified_runtime_release_v1(release_root)
    envelope = _require_science_release_authority_v1(release)
    if release.payload()["manifest_sha256"] != row.get("manifest_sha256"):
        raise ProductionScienceFrontDoorError("active QRR manifest binding differs")
    controls = {
        str(name)
        for configuration in request["configurations"]
        for name in configuration["controls"]
    }
    admitted_controls = {
        str(contract["name"])
        for contract in envelope.experiment_variable_contracts
        if isinstance(contract, Mapping) and isinstance(contract.get("name"), str)
    }
    reasons: list[dict[str, Any]] = []
    evidence_geometry = envelope.qualification_evidence_binding["geometry"]
    evidence_target_ticks = envelope.qualification_evidence_binding["target_ticks"]
    if envelope.time_contract.get("authority") != "absolute_tick_v1":
        reasons.append({"code": "ABSOLUTE_TICK_CAPABILITY_MISSING"})
    if request["total_ticks"] > evidence_target_ticks:
        reasons.append({"code": "SCIENCE_TICK_HORIZON_NOT_QUALIFIED"})
    if (
        request["batch_cells"] != evidence_geometry["batch_cells"]
        or request["world_width"] != evidence_geometry["world_width"]
    ):
        reasons.append({"code": "SCIENCE_GEOMETRY_NOT_QUALIFIED"})
    if not (
        envelope.admitted_batch_range[0]
        <= request["batch_cells"]
        <= envelope.admitted_batch_range[1]
    ):
        reasons.append({"code": "BATCH_ENVELOPE_BLOCKED"})
    missing_controls = sorted(controls - admitted_controls)
    if missing_controls:
        reasons.append(
            {
                "code": "EXPERIMENT_NOT_REPRESENTABLE",
                "unsupported_controls": missing_controls,
            }
        )
    base = load_campaign_spec(release_root / "payload/spec/base_spec.yaml")
    if (
        base.get("world", {}).get("base") != "patch"
        or tuple(base.get("world", {}).get("overlays", ()))
        != ("eed_resource_impact", "jungle", "wdl_delayed_hazard")
        or base.get("parameters", {}).get("patch.grid_size") != request["world_width"]
    ):
        reasons.append({"code": "MECHANICS_OR_WORLD_ENVELOPE_BLOCKED"})
    if reasons:
        raise ProductionScienceFrontDoorError(
            "QRR_REPRESENTABILITY_BLOCKED: "
            + ", ".join(str(reason["code"]) for reason in reasons)
        )
    receipt_core = {
        "schema_version": "eiel_active_qrr_resolution_receipt_v1",
        "status": "PASS",
        "release_id": release.release_id,
        "manifest_sha256": release.payload()["manifest_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "science_request_sha256": request.get(
            "request_sha256", canonical_sha256(request)
        ),
        "qualification_evidence_binding_sha256": (
            envelope.qualification_evidence_binding["binding_sha256"]
        ),
        "manual_internal_decisions_required": 0,
        "reasons": [],
    }
    return release_root, {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core),
    }


def resolve_active_qrr_v1(
    science_request: Mapping[str, Any], *, qrr_root: Path = DEFAULT_QRR_ROOT
) -> tuple[Path, dict[str, Any]]:
    return _resolve_active_qrr_normalized_v1(
        validate_production_science_request_v1(science_request),
        qrr_root=qrr_root,
    )


def validate_production_science_request_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(value)
    allowed = {
        "schema_version",
        "request_id",
        "world_width",
        "batch_cells",
        "total_ticks",
        "checkpoint_ticks",
        "checkpoint_every_ticks",
        "acs_capture_instrumentation",
        "acs_checkpoint_slots",
        "seeds",
        "configurations",
        "deliberate_restore_ticks",
        "observability",
        "artifact_lifecycle",
    }
    unknown = sorted(set(request) - allowed)
    if (
        request.get("schema_version") != PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION
        or unknown
    ):
        raise ProductionScienceFrontDoorError(
            f"science request schema is unsupported; unknown={unknown}"
        )
    for name in ("world_width", "batch_cells", "total_ticks"):
        if type(request.get(name)) is not int or request[name] <= 0:
            raise ProductionScienceFrontDoorError(f"{name} must be a positive integer")
    seeds = request.get("seeds")
    configurations = request.get("configurations", [])
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or not isinstance(configurations, list)
    ):
        raise ProductionScienceFrontDoorError("seeds/configurations are malformed")
    normalized_configurations: list[dict[str, Any]] = []
    configuration_ids: set[str] = set()
    for index, row in enumerate(configurations):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("controls"), Mapping)
            or not row["controls"]
        ):
            raise ProductionScienceFrontDoorError(f"configuration {index} is malformed")
        if row["id"] in configuration_ids:
            raise ProductionScienceFrontDoorError("configuration ids must be unique")
        configuration_ids.add(row["id"])
        normalized_configurations.append(
            {"id": row["id"], "controls": dict(sorted(row["controls"].items()))}
        )
    configuration_count = max(1, len(normalized_configurations))
    if configuration_count * len(seeds) != request["batch_cells"]:
        raise ProductionScienceFrontDoorError(
            "batch_cells must equal configuration rows crossed with seeds"
        )
    checkpoint_ticks = request.get("checkpoint_ticks", [])
    checkpoint_every = request.get("checkpoint_every_ticks")
    if bool(checkpoint_ticks) == (checkpoint_every is not None):
        raise ProductionScienceFrontDoorError(
            "supply exactly one checkpoint schedule representation"
        )
    if checkpoint_ticks and not isinstance(checkpoint_ticks, list):
        raise ProductionScienceFrontDoorError("checkpoint_ticks are malformed")
    if checkpoint_ticks and (
        any(type(tick) is not int or tick <= 0 for tick in checkpoint_ticks)
        or checkpoint_ticks != sorted(set(checkpoint_ticks))
        or checkpoint_ticks[-1] != request["total_ticks"]
    ):
        raise ProductionScienceFrontDoorError(
            "checkpoint_ticks must be unique, increasing, positive, and end at total_ticks"
        )
    if checkpoint_every is not None and (
        type(checkpoint_every) is not int
        or checkpoint_every <= 0
        or checkpoint_every > request["total_ticks"]
    ):
        raise ProductionScienceFrontDoorError(
            "checkpoint_every_ticks must be positive and within total_ticks"
        )
    lifecycle_raw = request.get("artifact_lifecycle")
    lifecycle: dict[str, Any] | None = None
    if lifecycle_raw is not None:
        try:
            lifecycle = validate_run_artifact_lifecycle_contract_v1(lifecycle_raw)
        except ArtifactLifecycleError as exc:
            raise ProductionScienceFrontDoorError(
                f"artifact_lifecycle is invalid: {exc}"
            ) from exc
        expected_schedule = (
            list(checkpoint_ticks)
            if checkpoint_ticks
            else list(range(int(checkpoint_every), request["total_ticks"] + 1, int(checkpoint_every)))
        )
        if expected_schedule[-1] != request["total_ticks"]:
            expected_schedule.append(request["total_ticks"])
        lifecycle_checkpoints = lifecycle["checkpoints"]
        if lifecycle_checkpoints.get("authority") != "absolute_tick":
            raise ProductionScienceFrontDoorError(
                "artifact_lifecycle checkpoint authority must be absolute_tick"
            )
        if lifecycle_checkpoints["checkpoint_ticks"] != sorted(
            set(expected_schedule)
        ):
            raise ProductionScienceFrontDoorError(
                "artifact_lifecycle checkpoint schedule does not bind the production request"
            )
    raw_acs_capture_instrumentation = request.get(
        "acs_capture_instrumentation", False
    )
    raw_acs_checkpoint_slots = request.get("acs_checkpoint_slots", 0)
    if type(raw_acs_capture_instrumentation) is not bool:
        raise ProductionScienceFrontDoorError(
            "acs_capture_instrumentation must be an exact boolean"
        )
    if (
        type(raw_acs_checkpoint_slots) is not int
        or raw_acs_checkpoint_slots not in {0, 1, 2, 3}
    ):
        raise ProductionScienceFrontDoorError(
            "acs_checkpoint_slots must be zero, one, two, or three"
        )
    acs_capture_instrumentation = (
        raw_acs_capture_instrumentation or bool(raw_acs_checkpoint_slots)
    )
    restore_ticks = request.get("deliberate_restore_ticks", [])
    if not isinstance(restore_ticks, list) or any(
        type(tick) is not int for tick in restore_ticks
    ):
        raise ProductionScienceFrontDoorError("deliberate_restore_ticks are malformed")
    if (
        restore_ticks != sorted(set(restore_ticks))
        or any(tick <= 0 or tick >= request["total_ticks"] for tick in restore_ticks)
        or (
            checkpoint_ticks
            and any(tick not in checkpoint_ticks for tick in restore_ticks)
        )
        or (
            checkpoint_every is not None
            and any(tick % checkpoint_every != 0 for tick in restore_ticks)
        )
    ):
        raise ProductionScienceFrontDoorError(
            "restore ticks must be unique, increasing, nonterminal checkpoint ticks"
        )
    raw_observability = request.get("observability", {})
    if not isinstance(raw_observability, Mapping):
        raise ProductionScienceFrontDoorError("observability must be an object")
    unknown_observability = sorted(set(raw_observability) - {"living_lineage_canopy"})
    canopy_enabled = raw_observability.get("living_lineage_canopy", True)
    if unknown_observability or type(canopy_enabled) is not bool:
        raise ProductionScienceFrontDoorError(
            "observability is malformed; "
            f"unknown={unknown_observability}, living_lineage_canopy must be boolean"
        )
    return {
        "schema_version": PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION,
        "request_id": str(request.get("request_id") or "production-science-request"),
        "world_width": request["world_width"],
        "batch_cells": request["batch_cells"],
        "total_ticks": request["total_ticks"],
        "checkpoint_ticks": list(checkpoint_ticks),
        "checkpoint_every_ticks": checkpoint_every,
        **(
            {"acs_capture_instrumentation": True}
            if acs_capture_instrumentation
            else {}
        ),
        **(
            {"acs_checkpoint_slots": raw_acs_checkpoint_slots}
            if raw_acs_checkpoint_slots
            else {}
        ),
        "seeds": list(seeds),
        "configurations": normalized_configurations,
        "deliberate_restore_ticks": list(restore_ticks),
        "observability": {"living_lineage_canopy": canopy_enabled},
        **({"artifact_lifecycle": lifecycle} if lifecycle is not None else {}),
    }


def validate_production_science_request_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a Baseline-declared science request without erasing ACS0."""

    if not isinstance(value, Mapping):
        raise ProductionScienceFrontDoorError("science request must be an object")
    raw = dict(value)
    v1_fields = {
        "schema_version",
        "request_id",
        "world_width",
        "batch_cells",
        "total_ticks",
        "checkpoint_ticks",
        "checkpoint_every_ticks",
        "acs_capture_instrumentation",
        "acs_checkpoint_slots",
        "seeds",
        "configurations",
        "deliberate_restore_ticks",
        "observability",
        "artifact_lifecycle",
    }
    unknown = sorted(set(raw) - v1_fields - {"baseline", "request_sha256"})
    if (
        raw.get("schema_version") != PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION
        or unknown
    ):
        raise ProductionScienceFrontDoorError(
            f"science request schema is unsupported; unknown={unknown}"
        )
    legacy = {
        key: item
        for key, item in raw.items()
        if key not in {"baseline", "request_sha256"}
    }
    legacy["schema_version"] = PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION
    normalized = validate_production_science_request_v1(legacy)
    try:
        baseline = BaselineDeclarationV1.from_value(raw.get("baseline"))
        validate_baseline_purpose_v1(baseline, purpose="simulation")
    except BaselineError as exc:
        raise ProductionScienceFrontDoorError(str(exc)) from exc
    core = {
        **normalized,
        "schema_version": PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
        "baseline": baseline.payload(),
        **(
            {"acs_checkpoint_slots": raw["acs_checkpoint_slots"]}
            if "acs_checkpoint_slots" in raw
            else {}
        ),
    }
    request_sha256 = canonical_sha256(core)
    if (
        raw.get("request_sha256") is not None
        and raw.get("request_sha256") != request_sha256
    ):
        raise ProductionScienceFrontDoorError("science request SHA-256 mismatch")
    return {**core, "request_sha256": request_sha256}


def resolve_active_qrr_v2(
    science_request: Mapping[str, Any], *, qrr_root: Path = DEFAULT_QRR_ROOT
) -> tuple[Path, dict[str, Any]]:
    return _resolve_active_qrr_normalized_v1(
        validate_production_science_request_v2(science_request),
        qrr_root=qrr_root,
    )


def materialize_qrr_experiment_v1(
    science_request: Mapping[str, Any], *, release_root: Path
) -> dict[str, Any]:
    request = validate_production_science_request_v1(science_request)
    base = load_campaign_spec(release_root / "payload/spec/base_spec.yaml")
    configurations = list(request["configurations"])
    if not configurations:
        default_move_cost = base.get("parameters", {}).get("env_cfg.move_cost")
        if not isinstance(default_move_cost, (int, float)) or isinstance(
            default_move_cost, bool
        ):
            raise ProductionScienceFrontDoorError(
                "active QRR has no explicit sealed default science configuration"
            )
        configurations = [
            {
                "id": "default",
                "controls": {"env_cfg.move_cost": default_move_cost},
            }
        ]
    return {
        "schema_version": "eiel_qrr_experiment_v1",
        "axes": {},
        "configurations": configurations,
        "seeds": {"task": list(request["seeds"])},
        "fixed_context_axes": {},
        "total_ticks": request["total_ticks"],
        "checkpoint_ticks": list(request["checkpoint_ticks"]),
        "checkpoint_every_ticks": request["checkpoint_every_ticks"],
        **(
            {"acs_capture_instrumentation": True}
            if request.get("acs_capture_instrumentation") is True
            else {}
        ),
        **(
            {"acs_checkpoint_slots": request["acs_checkpoint_slots"]}
            if request.get("acs_checkpoint_slots")
            else {}
        ),
        "deliberate_restore_ticks": list(request["deliberate_restore_ticks"]),
        "exercise_checkpoint_resume": bool(request["deliberate_restore_ticks"]),
        "observability": dict(request["observability"]),
        **(
            {"artifact_lifecycle": dict(request["artifact_lifecycle"])}
            if isinstance(request.get("artifact_lifecycle"), Mapping)
            else (
                {"artifact_lifecycle": dict(base["artifact_lifecycle"])}
                if isinstance(base.get("artifact_lifecycle"), Mapping)
                else {}
            )
        ),
    }


def materialize_qrr_experiment_v2(
    science_request: Mapping[str, Any], *, release_root: Path
) -> dict[str, Any]:
    """Materialize V2 while preserving explicit zero as an authoritative choice."""

    request = validate_production_science_request_v2(science_request)
    legacy = {
        key: item
        for key, item in request.items()
        if key not in {"baseline", "request_sha256"}
    }
    legacy["schema_version"] = PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION
    experiment = materialize_qrr_experiment_v1(legacy, release_root=release_root)
    if "acs_checkpoint_slots" in request:
        experiment["acs_checkpoint_slots"] = request["acs_checkpoint_slots"]
    return experiment


def _invoke_sealed_qrr_v1(argv: Sequence[str]) -> dict[str, Any]:
    child_environment = dict(os.environ)
    child_environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise _SealedQrrInvocationError(
            "sealed QRR result is not exact JSON "
            f"(returncode={completed.returncode}, stdout_bytes={len(completed.stdout.encode())}, "
            f"stderr_bytes={len(completed.stderr.encode())})",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ) from exc
    if completed.returncode != 0 or not isinstance(result, Mapping):
        raise _SealedQrrInvocationError(
            f"sealed QRR command failed (returncode={completed.returncode}): {result}",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return dict(result)


def _recover_completed_qrr_result_v1(
    *,
    error: _SealedQrrInvocationError,
    release_id: str,
    run_root: Path,
    plan: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recover only an exit-zero launch with an exact durable completion chain."""

    if error.returncode != 0:
        return None
    try:
        receipt = _read_mapping(
            run_root / "qrr_run_receipt.json", name="QRR run receipt"
        )
        binding = _read_mapping(
            run_root / "qrr_run_binding.json", name="QRR run binding"
        )
        campaign = _read_mapping(
            run_root / "campaign_manifest.json", name="campaign manifest"
        )
    except ProductionScienceFrontDoorError:
        return None
    execution = receipt.get("execution_receipt")
    counts = campaign.get("counts")
    cells = plan.get("cells")
    plan_sha256 = plan.get("plan_sha256")
    restore_ticks = experiment.get("deliberate_restore_ticks", [])
    if not (
        isinstance(plan_sha256, str)
        and isinstance(cells, list)
        and receipt.get("schema_version") == "eiel_qrr_run_receipt_v1"
        and receipt.get("status") == "completed"
        and receipt.get("release_id") == release_id
        and receipt.get("plan_sha256") == plan_sha256
        and isinstance(execution, Mapping)
        and execution.get("ok") is True
        and execution.get("status") == "completed"
        and execution.get("plan_sha256") == plan_sha256
        and binding.get("schema_version") == "eiel_qrr_run_binding_v1"
        and binding.get("release_id") == release_id
        and binding.get("plan_sha256") == plan_sha256
        and campaign.get("schema_version") == "eiel_campaign_execution_manifest_v1"
        and campaign.get("status") == "completed"
        and isinstance(counts, Mapping)
        and counts.get("completed") == len(cells)
        and all(
            counts.get(name) == 0
            for name in ("failed", "interrupted", "pending", "running")
        )
        and isinstance(restore_ticks, list)
        and (
            not restore_ticks
            or (
                receipt.get("checkpoint_resume_exact") is True
                and binding.get("deliberate_fresh_owner_restore_ticks") == restore_ticks
                and binding.get("resume_invocations") == len(restore_ticks)
            )
        )
    ):
        return None
    return {
        "ok": True,
        "schema_version": "eiel_qrr_launch_result_v1",
        "release_id": release_id,
        "run_root": str(run_root),
        "run_receipt": receipt,
        "result_delivery": {
            "state": "recovered_from_durable_completed_receipt",
            "transport_returncode": error.returncode,
            "stdout_sha256": canonical_sha256({"stdout": error.stdout}),
            "stderr_sha256": canonical_sha256({"stderr": error.stderr}),
        },
    }


def _relay_process_matches_v1(pid: int, *, output_root: Path) -> bool:
    return canopy_relay_process_matches_v1(pid, output_root=output_root)


def _start_canopy_relays_v1(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    relay_root: Path = DEFAULT_CANOPY_RELAY_ROOT,
    repository_root: Path | None = None,
    python_executable: Path | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    try:
        return start_canopy_relays_v1(
            plan=plan,
            run_root=run_root,
            relay_root=relay_root,
            repository_root=repository_root,
            python_executable=python_executable,
            enabled=enabled,
            group_prefix="science",
            popen_fn=subprocess.Popen,
            process_matcher=_relay_process_matches_v1,
        )
    except CanopySupervisionError as exc:
        raise ProductionScienceFrontDoorError(
            str(exc)
        ) from exc


def _mark_canopy_source_interrupted_v1(run_root: Path, *, reason: str) -> None:
    mark_canopy_source_interrupted_v1(run_root, reason=reason)


def _record_canopy_observability_degraded_v1(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return record_canopy_observability_degraded_v1(
        run_root, plan=plan, reason=reason
    )


def _validate_acs_materialization_v1(
    plan: Mapping[str, Any], *, request: Mapping[str, Any]
) -> None:
    """Prevent older sealed QRR launchers from silently dropping ACS selection."""

    expected_slots = request.get("acs_checkpoint_slots", 0)
    expected_instrumentation = (
        request.get("acs_capture_instrumentation", False)
        or bool(expected_slots)
    )
    if (
        "acs_checkpoint_slots" not in request
        and not expected_instrumentation
        and expected_slots == 0
    ):
        return
    execution = plan.get("execution")
    actual_slots = (
        execution.get("acs_checkpoint_slots", 0)
        if isinstance(execution, Mapping)
        else 0
    )
    actual_instrumentation = (
        execution.get("acs_capture_instrumentation", False)
        if isinstance(execution, Mapping)
        else False
    )
    if (
        actual_slots != expected_slots
        or actual_instrumentation is not expected_instrumentation
    ):
        raise ProductionScienceFrontDoorError(
            "ACS_ROUTE_NOT_MATERIALIZED: active sealed QRR does not preserve "
            "the requested ACS checkpoint route"
        )


def _execute_science_request_common_v1(
    command: str,
    science_request_path: Path,
    *,
    request_version: int,
    qrr_root: Path = DEFAULT_QRR_ROOT,
    run_root: Path | None = None,
) -> dict[str, Any]:
    if command not in {"materialize", "check-representability", "preflight", "run"}:
        raise ProductionScienceFrontDoorError("front-door command is unsupported")
    request = _read_mapping(science_request_path, name="production science request")
    if request_version == 2:
        normalized = validate_production_science_request_v2(request)
    elif request_version == 1:
        normalized = validate_production_science_request_v1(request)
    else:
        raise ProductionScienceFrontDoorError("unsupported science request version")
    destination = run_root.expanduser().resolve() if run_root is not None else None
    bound = (
        _load_bound_science_resolution_v1(
            destination=destination,
            normalized_request=normalized,
            qrr_root=qrr_root,
        )
        if destination is not None
        else None
    )
    if bound is not None:
        release_root, resolution = bound
    elif request_version == 2:
        release_root, resolution = resolve_active_qrr_v2(
            normalized, qrr_root=qrr_root
        )
    else:
        release_root, resolution = resolve_active_qrr_v1(
            normalized, qrr_root=qrr_root
        )
    if destination is None:
        destination = (
            DEFAULT_QRR_RUN_ROOT
            / release_root.name
            / f"science-{canonical_sha256(normalized)[:20]}"
        )
    if request_version == 1 and command == "run" and (
        bound is None
        or not _v1_qrr_continuation_authorized_v1(
            destination=destination, release_root=release_root
        )
    ):
        raise ProductionScienceFrontDoorError(
            "PRODUCTION_SCIENCE_REQUEST_V2_REQUIRED: fresh run requires Baseline; "
            "V1 is limited to exact bound nonterminal continuation"
        )
    experiment = (
        materialize_qrr_experiment_v2(normalized, release_root=release_root)
        if request_version == 2
        else materialize_qrr_experiment_v1(normalized, release_root=release_root)
    )
    experiment_path = destination / "production_science_qrr_experiment.json"
    request_receipt = _bind_science_run_root_v1(
        destination=destination,
        normalized_request=normalized,
        experiment=experiment,
        resolution=resolution,
        release_root=release_root,
    )
    existing_result = _existing_science_result_v1(
        destination=destination,
        request_receipt=request_receipt,
        normalized_request=normalized,
        command=command,
    )
    if existing_result is not None:
        return existing_result
    argv = [
        str(release_root / "bin/qrr"),
        command,
        "--experiment",
        str(experiment_path),
        "--run-root",
        str(destination),
    ]
    plan: dict[str, Any] | None = None
    preflight: dict[str, Any] | None = None
    baseline_admission: dict[str, Any] | None = None
    baseline_declaration = (
        BaselineDeclarationV1.from_value(normalized["baseline"])
        if request_version == 2
        else None
    )
    canopy_enabled = normalized["observability"]["living_lineage_canopy"]
    canopy_supervision: dict[str, Any] | None = None

    def finalize_baseline(execution_status: str) -> dict[str, Any] | None:
        if (
            baseline_declaration is None
            or baseline_admission is None
            or plan is None
        ):
            return None
        conformance = build_baseline_conformance_receipt_v1(
            admission_receipt=baseline_admission,
            execution_status=execution_status,
            declaration=baseline_declaration,
            plan=plan,
            request_sha256=normalized["request_sha256"],
            front_door_id=PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
            preflight=preflight,
            artifact_lifecycle_explicit="artifact_lifecycle" in normalized,
            canopy_enabled=canopy_enabled,
            canopy_supervision=canopy_supervision,
        )
        write_baseline_receipt_v1(
            destination / "baseline_conformance_receipt.json", conformance
        )
        return conformance

    if command == "run":
        preflight_argv = list(argv)
        preflight_argv[1] = "preflight"
        preflight_result = _invoke_sealed_qrr_v1(preflight_argv)
        plan = _read_mapping(
            destination / "execution_plan.json", name="materialized execution plan"
        )
        preflight_value = preflight_result.get("preflight")
        if isinstance(preflight_value, Mapping):
            preflight = dict(preflight_value)
        elif request_version == 2:
            preflight = _read_mapping(
                destination / "preflight_receipt.json",
                name="materialized preflight receipt",
            )
        _validate_acs_materialization_v1(plan, request=normalized)
        if baseline_declaration is not None:
            baseline_admission = build_baseline_admission_receipt_v1(
                declaration=baseline_declaration,
                plan=plan,
                request_sha256=normalized["request_sha256"],
                front_door_id=PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
                preflight=preflight,
                artifact_lifecycle_explicit=(
                    "artifact_lifecycle" in normalized
                ),
                canopy_enabled=canopy_enabled,
            )
            write_baseline_receipt_v1(
                destination / "baseline_admission_receipt.json",
                baseline_admission,
            )
            if baseline_admission.get("admitted") is not True:
                reasons = ", ".join(
                    str(value)
                    for value in baseline_admission.get("reason_codes", [])
                )
                raise ProductionScienceFrontDoorError(
                    "BASELINE_ADMISSION_FAILED: " + (reasons or "not admitted")
                )
        try:
            canopy_supervision = _start_canopy_relays_v1(
                plan=plan,
                run_root=destination,
                enabled=canopy_enabled,
            )
        except ProductionScienceFrontDoorError as exc:
            canopy_supervision = _record_canopy_observability_degraded_v1(
                destination,
                plan=plan,
                reason=f"{type(exc).__name__}: {exc}",
            )
    try:
        if request_version == 2 and command == "check-representability":
            materialize_argv = list(argv)
            materialize_argv[1] = "materialize"
            _invoke_sealed_qrr_v1(materialize_argv)
        result = _invoke_sealed_qrr_v1(argv)
    except _SealedQrrInvocationError as exc:
        result = (
            _recover_completed_qrr_result_v1(
                error=exc,
                release_id=release_root.name,
                run_root=destination,
                plan=plan,
                experiment=experiment,
            )
            if command == "run"
            else None
        )
        if result is None:
            if canopy_supervision is not None and canopy_supervision.get("relays"):
                _mark_canopy_source_interrupted_v1(
                    destination,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            finalize_baseline("failed")
            raise
    except BaseException as exc:
        if canopy_supervision is not None and canopy_supervision.get("relays"):
            _mark_canopy_source_interrupted_v1(
                destination,
                reason=f"{type(exc).__name__}: {exc}",
            )
        finalize_baseline("failed")
        raise
    plan_path = destination / "execution_plan.json"
    if command != "run" and plan_path.is_file():
        plan = _read_mapping(plan_path, name="materialized execution plan")
        if (
            normalized.get("acs_capture_instrumentation") is True
            or "acs_checkpoint_slots" in normalized
        ):
            _validate_acs_materialization_v1(plan, request=normalized)
        preflight_path = destination / "preflight_receipt.json"
        if preflight_path.is_file():
            preflight = _read_mapping(
                preflight_path, name="materialized preflight receipt"
            )
        if baseline_declaration is not None:
            baseline_admission = build_baseline_admission_receipt_v1(
                declaration=baseline_declaration,
                plan=plan,
                request_sha256=normalized["request_sha256"],
                front_door_id=PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
                preflight=preflight,
                artifact_lifecycle_explicit="artifact_lifecycle" in normalized,
                canopy_enabled=canopy_enabled,
            )
            write_baseline_receipt_v1(
                destination / "baseline_admission_receipt.json",
                baseline_admission,
            )
    if request_version == 2 and baseline_admission is None:
        raise ProductionScienceFrontDoorError(
            "BASELINE_ADMISSION_MISSING: V2 command did not materialize a bound plan"
        )
    baseline_conformance = (
        finalize_baseline("completed" if result.get("ok") is True else "failed")
        if command == "run"
        else None
    )
    baseline_binding = (
        baseline_result_binding_v1(
            admission_receipt=baseline_admission,
            conformance_receipt=baseline_conformance,
        )
        if baseline_admission is not None
        else None
    )
    baseline_command_allowed = _qrr_baseline_command_allowed_v1(
        admission_receipt=baseline_admission,
        command=command,
        request_sha256=str(
            normalized.get("request_sha256") or canonical_sha256(normalized)
        ),
        plan_sha256=(
            str(plan["plan_sha256"]) if plan is not None else "not-materialized"
        ),
    )
    front_door = {
        "schema_version": (
            PRODUCTION_SCIENCE_RESULT_V2_SCHEMA_VERSION
            if request_version == 2
            else "eiel_production_science_front_door_result_v1"
        ),
        "ok": result.get("ok") is True and baseline_command_allowed,
        "command": command,
        "release_id": release_root.name,
        "run_root": str(destination),
        "manual_internal_decisions_required": 0,
        "request_receipt_sha256": request_receipt["receipt_sha256"],
        "resolution_receipt_sha256": resolution["receipt_sha256"],
        "qrr_result": dict(result),
        "canopy_relay_supervision": canopy_supervision,
        "baseline": (
            baseline_binding
            if baseline_binding is not None
            else {"overall_status": "LEGACY_UNPROFILED"}
        ),
    }
    front_door = {
        **front_door,
        "result_sha256": canonical_sha256(front_door),
    }
    _atomic_json(destination / "production_science_front_door_result.json", front_door)
    return front_door


def execute_science_request_v1(
    command: str,
    science_request_path: Path,
    *,
    qrr_root: Path = DEFAULT_QRR_ROOT,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Execute or inspect an exact historical V1 science request."""

    return _execute_science_request_common_v1(
        command,
        science_request_path,
        request_version=1,
        qrr_root=qrr_root,
        run_root=run_root,
    )


def execute_science_request_v2(
    command: str,
    science_request_path: Path,
    *,
    qrr_root: Path = DEFAULT_QRR_ROOT,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Execute or inspect a Baseline-declared V2 science request."""

    return _execute_science_request_common_v1(
        command,
        science_request_path,
        request_version=2,
        qrr_root=qrr_root,
        run_root=run_root,
    )


__all__ = [
    "ACTIVE_QRR_REGISTRY_SCHEMA_VERSION",
    "PRODUCTION_SCIENCE_REQUEST_SCHEMA_VERSION",
    "PRODUCTION_SCIENCE_REQUEST_RECEIPT_SCHEMA_VERSION",
    "PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION",
    "ProductionScienceFrontDoorError",
    "activate_qrr_v1",
    "execute_science_request_v1",
    "execute_science_request_v2",
    "materialize_qrr_experiment_v1",
    "materialize_qrr_experiment_v2",
    "resolve_active_qrr_v1",
    "resolve_active_qrr_v2",
    "validate_production_science_request_v1",
    "validate_production_science_request_v2",
]
