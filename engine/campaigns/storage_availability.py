"""Canonical storage-availability policy and evidence-backed reserve maths.

The policy is a checked-in admission authority.  Reclamation remains owned by
the lifecycle/prune actuator; this module only validates the policy and
derives a hash-bound reserve from the admitted run's measured envelope.  That
separation lets admission reclaim exact manifest-authorized transient material
before returning ``WAIT_STORAGE`` without turning this module into another
storage manager.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping
from fractions import Fraction
import json
import os
from pathlib import Path
import re
from typing import Any, Final

from .canonical import canonical_sha256


STORAGE_AVAILABILITY_POLICY_SCHEMA_VERSION: Final[str] = (
    "eiel_storage_availability_policy_v1"
)
STORAGE_AVAILABILITY_SNAPSHOT_SCHEMA_VERSION: Final[str] = (
    "eiel_storage_availability_snapshot_v1"
)
FILESYSTEM_IDENTITY_SCHEMA_VERSION: Final[str] = "eiel_filesystem_identity_v1"
DATA2_STORAGE_AVAILABILITY_POLICY_ID: Final[str] = "EIEL_DATA2_STORAGE_AVAILABILITY_V1"
DATA2_STORAGE_AVAILABILITY_POLICY_VERSION: Final[int] = 1
DATA2_FILESYSTEM_ROOT: Final[Path] = Path(public_runtime_path())
GIB_BYTES: Final[int] = 1024**3
# These legacy names remain exported for receipt/schema compatibility.  The
# canonical policy no longer uses a ritual 100 GiB or 20% floor; both values
# are zero in the reform policy.  A legacy policy carrying the old values is
# accepted conservatively by ``validate_*`` and never silently relaxed.
DATA2_MINIMUM_PROTECTED_FREE_BYTES: Final[int] = 0
DATA2_MINIMUM_FREE_FRACTION_NUMERATOR: Final[int] = 0
DATA2_MINIMUM_FREE_FRACTION_DENOMINATOR: Final[int] = 1
DATA2_LEGACY_MINIMUM_PROTECTED_FREE_BYTES: Final[int] = 100 * GIB_BYTES
DATA2_LEGACY_MINIMUM_FREE_FRACTION_NUMERATOR: Final[int] = 1
DATA2_LEGACY_MINIMUM_FREE_FRACTION_DENOMINATOR: Final[int] = 5
RESERVE_MODEL_SCHEMA_VERSION: Final[str] = "eiel_evidence_backed_reserve_model_v1"
RESERVE_MODEL_MODE: Final[str] = "measured_execution_envelope_v1"
DATA2_FILESYSTEM_RESERVE_BYTES: Final[int] = 4 * GIB_BYTES
DATA2_RESERVE_UNCERTAINTY_NUMERATOR: Final[int] = 1
DATA2_RESERVE_UNCERTAINTY_DENOMINATOR: Final[int] = 20
DATA2_RESERVE_MAX_UNCERTAINTY_BYTES: Final[int] = 8 * GIB_BYTES
DATA2_PERMANENT_EVIDENCE_ROOT: Final[Path] = Path(public_runtime_path(".permanent-evidence"))
DATA2_PERMANENT_EVIDENCE_MAX_TOTAL_BYTES: Final[int] = 512 * 1024**2
DATA2_PERMANENT_EVIDENCE_MAX_CAPSULE_BYTES: Final[int] = 256 * 1024
DATA2_COLD_PROFILE_MAX_OBJECT_BYTES: Final[int] = 512 * 1024**2
DATA2_COLD_PROFILE_MAX_AGGREGATE_BYTES: Final[int] = 20 * GIB_BYTES
DATA2_COLD_PROFILE_ADMISSION_POLICY: Final[str] = "explicit_manifest_selected_only"
DATA2_RUNTIME_ROOT: Final[Path] = Path(public_runtime_path(''))
DATA2_LIFECYCLE_MANAGEMENT_ROOT: Final[Path] = DATA2_RUNTIME_ROOT / "lifecycle"
CANONICAL_DATA2_STORAGE_AVAILABILITY_POLICY_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "_spec"
    / "storage"
    / "v1"
    / "eiel_data2_storage_availability_v1.json"
)


class StorageAvailabilityPolicyError(ValueError):
    """The canonical storage-availability authority is unavailable or invalid."""


def _exact(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    if set(value) != expected:
        raise StorageAvailabilityPolicyError(
            f"{path} fields differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def validate_data2_storage_availability_policy_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact V1 policy, including its canonical digest."""

    if not isinstance(value, Mapping):
        raise StorageAvailabilityPolicyError(
            "storage availability policy must be an object"
        )
    raw = dict(value)
    supplied_sha256 = raw.pop("policy_sha256", None)
    expected_core = {
        "cold_diagnostic_profile_assets",
        "schema_version",
        "policy_id",
        "policy_version",
        "filesystem_root",
        "permanent_evidence_capsules",
        "protected_free_floor",
    }
    legacy_policy = "reserve_model" not in raw
    _exact(
        raw,
        expected_core if legacy_policy else {*expected_core, "reserve_model"},
        path="storage availability policy",
    )
    if raw.get("schema_version") != STORAGE_AVAILABILITY_POLICY_SCHEMA_VERSION:
        raise StorageAvailabilityPolicyError(
            "storage availability policy schema_version is unsupported"
        )
    if raw.get("policy_id") != DATA2_STORAGE_AVAILABILITY_POLICY_ID:
        raise StorageAvailabilityPolicyError(
            "storage availability policy_id is unsupported"
        )
    if (
        type(raw.get("policy_version")) is not int
        or raw["policy_version"] != DATA2_STORAGE_AVAILABILITY_POLICY_VERSION
    ):
        raise StorageAvailabilityPolicyError(
            "storage availability policy_version is unsupported"
        )
    if raw.get("filesystem_root") != str(DATA2_FILESYSTEM_ROOT):
        raise StorageAvailabilityPolicyError(
            "storage availability policy must bind configured runtime root"
        )
    protected = raw.get("protected_free_floor")
    if not isinstance(protected, Mapping):
        raise StorageAvailabilityPolicyError(
            "storage availability protected_free_floor must be an object"
        )
    protected = dict(protected)
    _exact(
        protected,
        {
            "minimum_bytes",
            "minimum_fraction_numerator",
            "minimum_fraction_denominator",
        },
        path="storage availability protected_free_floor",
    )
    if any(
        type(protected.get(name)) is not int or int(protected[name]) < 0
        for name in ("minimum_bytes", "minimum_fraction_numerator", "minimum_fraction_denominator")
    ) or protected["minimum_fraction_denominator"] <= 0:
        raise StorageAvailabilityPolicyError(
            "storage availability protected-free floor values are invalid"
        )
    if legacy_policy:
        if protected != {
            "minimum_bytes": DATA2_LEGACY_MINIMUM_PROTECTED_FREE_BYTES,
            "minimum_fraction_numerator": DATA2_LEGACY_MINIMUM_FREE_FRACTION_NUMERATOR,
            "minimum_fraction_denominator": DATA2_LEGACY_MINIMUM_FREE_FRACTION_DENOMINATOR,
        }:
            raise StorageAvailabilityPolicyError(
                "legacy storage availability floor is not the recognized conservative policy"
            )
    elif protected != {
        "minimum_bytes": DATA2_MINIMUM_PROTECTED_FREE_BYTES,
        "minimum_fraction_numerator": DATA2_MINIMUM_FREE_FRACTION_NUMERATOR,
        "minimum_fraction_denominator": DATA2_MINIMUM_FREE_FRACTION_DENOMINATOR,
    }:
        raise StorageAvailabilityPolicyError(
            "reformed storage availability policy must disable the static percentage floor"
        )
    reserve_model = raw.get("reserve_model")
    if reserve_model is not None:
        if not isinstance(reserve_model, Mapping):
            raise StorageAvailabilityPolicyError(
                "storage availability reserve_model must be an object"
            )
        reserve_model = dict(reserve_model)
        _exact(
            reserve_model,
            {
                "schema_version",
                "mode",
                "filesystem_reserve_bytes",
                "uncertainty_margin_numerator",
                "uncertainty_margin_denominator",
                "max_uncertainty_bytes",
            },
            path="storage availability reserve_model",
        )
        if reserve_model.get("schema_version") != RESERVE_MODEL_SCHEMA_VERSION:
            raise StorageAvailabilityPolicyError(
                "storage availability reserve_model schema is unsupported"
            )
        if reserve_model.get("mode") != RESERVE_MODEL_MODE:
            raise StorageAvailabilityPolicyError(
                "storage availability reserve_model mode is unsupported"
            )
        if any(
            type(reserve_model.get(name)) is not int
            or int(reserve_model[name]) < 0
            for name in (
                "filesystem_reserve_bytes",
                "uncertainty_margin_numerator",
                "uncertainty_margin_denominator",
                "max_uncertainty_bytes",
            )
        ) or reserve_model["uncertainty_margin_denominator"] <= 0:
            raise StorageAvailabilityPolicyError(
                "storage availability reserve_model bounds are invalid"
            )
        if reserve_model["filesystem_reserve_bytes"] <= 0:
            raise StorageAvailabilityPolicyError(
                "storage availability reserve_model requires a positive filesystem reserve"
            )
    permanent = raw.get("permanent_evidence_capsules")
    if not isinstance(permanent, Mapping):
        raise StorageAvailabilityPolicyError(
            "storage availability permanent_evidence_capsules must be an object"
        )
    permanent = dict(permanent)
    _exact(
        permanent,
        {"root", "max_total_bytes", "max_capsule_bytes"},
        path="storage availability permanent_evidence_capsules",
    )
    if permanent != {
        "root": str(DATA2_PERMANENT_EVIDENCE_ROOT),
        "max_total_bytes": DATA2_PERMANENT_EVIDENCE_MAX_TOTAL_BYTES,
        "max_capsule_bytes": DATA2_PERMANENT_EVIDENCE_MAX_CAPSULE_BYTES,
    }:
        raise StorageAvailabilityPolicyError(
            "storage availability permanent evidence capsule budget is unsupported"
        )
    cold = raw.get("cold_diagnostic_profile_assets")
    if not isinstance(cold, Mapping):
        raise StorageAvailabilityPolicyError(
            "storage availability cold_diagnostic_profile_assets must be an object"
        )
    cold = dict(cold)
    _exact(
        cold,
        {"admission_policy", "max_object_bytes", "max_aggregate_bytes"},
        path="storage availability cold_diagnostic_profile_assets",
    )
    if cold != {
        "admission_policy": DATA2_COLD_PROFILE_ADMISSION_POLICY,
        "max_object_bytes": DATA2_COLD_PROFILE_MAX_OBJECT_BYTES,
        "max_aggregate_bytes": DATA2_COLD_PROFILE_MAX_AGGREGATE_BYTES,
    }:
        raise StorageAvailabilityPolicyError(
            "storage availability cold diagnostic/profile asset budget is unsupported"
        )
    normalized = {
        **raw,
        "protected_free_floor": protected,
        "permanent_evidence_capsules": permanent,
        "cold_diagnostic_profile_assets": cold,
    }
    if reserve_model is not None:
        normalized["reserve_model"] = reserve_model
    if not isinstance(supplied_sha256, str) or supplied_sha256 != canonical_sha256(
        normalized
    ):
        raise StorageAvailabilityPolicyError(
            "storage availability policy SHA-256 mismatch"
        )
    return {**normalized, "policy_sha256": supplied_sha256}


def load_data2_storage_availability_policy_v1(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the checked-in `configured runtime root` policy and fail closed on any defect."""

    requested_path = (
        CANONICAL_DATA2_STORAGE_AVAILABILITY_POLICY_PATH
        if path is None
        else path.expanduser()
    )
    if requested_path.is_symlink() or not requested_path.is_file():
        raise StorageAvailabilityPolicyError(
            f"canonical configured runtime root storage availability policy is missing: {requested_path}"
        )
    policy_path = requested_path.resolve()
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageAvailabilityPolicyError(
            f"canonical configured runtime root storage availability policy is unreadable: {policy_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise StorageAvailabilityPolicyError(
            "canonical configured runtime root storage availability policy must be an object"
        )
    # The portable checked-in template is authenticated before binding paths.
    # The resulting receipt binds the concrete configured roots and the same
    # reserve/budget constraints; arbitrary templates are never accepted.
    if payload.get("filesystem_root") == "${EIEL_RUNTIME_ROOT}":
        template = dict(payload)
        digest = template.pop("policy_sha256", None)
        if digest != canonical_sha256(template):
            raise StorageAvailabilityPolicyError("storage policy template SHA-256 mismatch")
        permanent = dict(template.get("permanent_evidence_capsules", {}))
        if permanent.get("root") != "${EIEL_RUNTIME_ROOT}/.permanent-evidence":
            raise StorageAvailabilityPolicyError("storage policy template evidence root is invalid")
        template["filesystem_root"] = str(DATA2_FILESYSTEM_ROOT)
        permanent["root"] = str(DATA2_PERMANENT_EVIDENCE_ROOT)
        template["permanent_evidence_capsules"] = permanent
        payload = {**template, "policy_sha256": canonical_sha256(template)}
    return validate_data2_storage_availability_policy_v1(payload)


def storage_availability_policy_binding_v1(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_data2_storage_availability_policy_v1(policy)
    return {
        "policy_id": validated["policy_id"],
        "policy_version": validated["policy_version"],
        "policy_sha256": validated["policy_sha256"],
    }


def validate_data2_run_root_v1(run_root: Path) -> Path:
    """Reject aggregate and lifecycle-management roots as run roots."""

    root = run_root.expanduser().resolve(strict=False)
    if root != DATA2_FILESYSTEM_ROOT and DATA2_FILESYSTEM_ROOT not in root.parents:
        raise StorageAvailabilityPolicyError(
            "canonical configured runtime root run root is outside configured runtime root"
        )
    if root in {DATA2_FILESYSTEM_ROOT, DATA2_RUNTIME_ROOT}:
        raise StorageAvailabilityPolicyError(
            f"aggregate storage root is not a valid run root: {root}"
        )
    for management_root in (
        DATA2_PERMANENT_EVIDENCE_ROOT,
        DATA2_LIFECYCLE_MANAGEMENT_ROOT,
    ):
        if root == management_root or management_root in root.parents:
            raise StorageAvailabilityPolicyError(
                f"storage management root is not a valid run root: {root}"
            )
    if ".eiel-storage-reservations" in root.parts:
        raise StorageAvailabilityPolicyError(
            f"storage reservation root is not a valid run root: {root}"
        )
    return root


def validate_filesystem_identity_v1(
    value: Mapping[str, Any],
    *,
    observed_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StorageAvailabilityPolicyError("filesystem identity must be an object")
    raw = dict(value)
    supplied_sha256 = raw.pop("identity_sha256", None)
    _exact(
        raw,
        {
            "schema_version",
            "filesystem_root",
            "device_id",
            "mount_id",
            "filesystem_id",
        },
        path="filesystem identity",
    )
    if raw.get("schema_version") != FILESYSTEM_IDENTITY_SCHEMA_VERSION:
        raise StorageAvailabilityPolicyError(
            "filesystem identity schema is unsupported"
        )
    filesystem_root = raw.get("filesystem_root")
    if not isinstance(filesystem_root, str) or not filesystem_root:
        raise StorageAvailabilityPolicyError(
            "filesystem identity filesystem_root must be an absolute path"
        )
    identity_root = Path(filesystem_root)
    if (
        not identity_root.is_absolute()
        or identity_root.resolve(strict=False) != identity_root
    ):
        raise StorageAvailabilityPolicyError(
            "filesystem identity filesystem_root must be normalized"
        )
    if type(raw.get("device_id")) is not int or raw["device_id"] < 0:
        raise StorageAvailabilityPolicyError(
            "filesystem identity device_id must be a non-negative integer"
        )
    if raw.get("mount_id") is not None and (
        type(raw["mount_id"]) is not int or raw["mount_id"] <= 0
    ):
        raise StorageAvailabilityPolicyError(
            "filesystem identity mount_id must be a positive integer or null"
        )
    if raw.get("filesystem_id") is not None and type(raw["filesystem_id"]) is not int:
        raise StorageAvailabilityPolicyError(
            "filesystem identity filesystem_id must be an integer or null"
        )
    if observed_path is not None:
        observed = observed_path.expanduser().resolve(strict=False)
        if identity_root != observed and identity_root not in observed.parents:
            raise StorageAvailabilityPolicyError(
                "filesystem identity root does not contain the observed path"
            )
    if not isinstance(supplied_sha256, str) or supplied_sha256 != canonical_sha256(raw):
        raise StorageAvailabilityPolicyError("filesystem identity SHA-256 mismatch")
    return {**raw, "identity_sha256": supplied_sha256}


def _mountinfo_unescape(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _mountinfo_identity(path: Path) -> tuple[int, Path] | None:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    matches: list[tuple[int, Path]] = []
    for line in lines:
        left, separator, _right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        if len(fields) < 6:
            continue
        try:
            mount_id = int(fields[0])
        except ValueError:
            continue
        mount_root = Path(_mountinfo_unescape(fields[4])).resolve(strict=False)
        if mount_root == path or mount_root in path.parents:
            matches.append((mount_id, mount_root))
    return max(matches, key=lambda item: len(item[1].parts)) if matches else None


def _device_mount_root(path: Path, device_id: int) -> Path:
    mount_root = path
    while mount_root != mount_root.parent:
        parent = mount_root.parent
        try:
            if int(os.stat(parent).st_dev) != device_id:
                break
        except OSError:
            break
        mount_root = parent
    return mount_root


def observe_filesystem_identity_v1(observed_path: Path) -> dict[str, Any]:
    requested = observed_path.expanduser().resolve(strict=False)
    existing = requested
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    try:
        stat = os.stat(existing)
        statvfs = os.statvfs(existing)
    except OSError as exc:
        raise StorageAvailabilityPolicyError(
            f"cannot observe filesystem identity for {requested}"
        ) from exc
    mountinfo = _mountinfo_identity(existing)
    if mountinfo is None:
        mount_id = None
        filesystem_root = _device_mount_root(existing, int(stat.st_dev))
    else:
        mount_id, filesystem_root = mountinfo
    filesystem_id = getattr(statvfs, "f_fsid", None)
    if type(filesystem_id) is not int:
        filesystem_id = None
    core = {
        "schema_version": FILESYSTEM_IDENTITY_SCHEMA_VERSION,
        "filesystem_root": str(filesystem_root),
        "device_id": int(stat.st_dev),
        "mount_id": mount_id,
        "filesystem_id": filesystem_id,
    }
    return validate_filesystem_identity_v1(
        {**core, "identity_sha256": canonical_sha256(core)},
        observed_path=requested,
    )


def build_storage_availability_snapshot_v1(
    *,
    policy: Mapping[str, Any],
    filesystem_identity: Mapping[str, Any],
    filesystem_total_bytes: int | None,
    filesystem_free_bytes: int | None,
    lifecycle_reserve_bytes: int,
    configured_minimum_free_fraction: float,
    execution_reserve_bytes: int | None = None,
) -> dict[str, Any]:
    """Derive the canonical evidence-backed reserve and usable headroom."""

    validated_policy = validate_data2_storage_availability_policy_v1(policy)
    validated_identity = validate_filesystem_identity_v1(filesystem_identity)
    if type(lifecycle_reserve_bytes) is not int or lifecycle_reserve_bytes < 0:
        raise StorageAvailabilityPolicyError(
            "lifecycle_reserve_bytes must be a non-negative integer"
        )
    if not 0.0 <= configured_minimum_free_fraction < 1.0:
        raise StorageAvailabilityPolicyError(
            "configured_minimum_free_fraction must be in [0, 1)"
        )
    if execution_reserve_bytes is not None and (
        type(execution_reserve_bytes) is not int or execution_reserve_bytes < 0
    ):
        raise StorageAvailabilityPolicyError(
            "execution_reserve_bytes must be a non-negative integer or null"
        )
    if execution_reserve_bytes is None:
        execution_reserve = evidence_backed_execution_reserve_v1(
            policy=validated_policy,
            remaining_run_bytes=0,
            reservation_window_bytes=0,
            lifecycle_reserve_bytes=lifecycle_reserve_bytes,
        )
        execution_reserve_bytes = int(execution_reserve["execution_reserve_bytes"])
    observed = (
        type(filesystem_total_bytes) is int
        and filesystem_total_bytes > 0
        and type(filesystem_free_bytes) is int
        and 0 <= filesystem_free_bytes <= filesystem_total_bytes
    )
    authority = {
        **storage_availability_policy_binding_v1(validated_policy),
        "filesystem_identity": validated_identity,
        "permanent_evidence_capsules": dict(
            validated_policy["permanent_evidence_capsules"]
        ),
        "cold_diagnostic_profile_assets": dict(
            validated_policy["cold_diagnostic_profile_assets"]
        ),
    }
    minimum = validated_policy["protected_free_floor"]
    known_minimum_floor = max(
        int(minimum["minimum_bytes"]),
        lifecycle_reserve_bytes,
        execution_reserve_bytes,
    )
    if not observed:
        return {
            "schema_version": STORAGE_AVAILABILITY_SNAPSHOT_SCHEMA_VERSION,
            **authority,
            "status": "UNOBSERVED",
            "filesystem_total_bytes": filesystem_total_bytes,
            "filesystem_free_bytes": filesystem_free_bytes,
            "policy_minimum_bytes": int(minimum["minimum_bytes"]),
            "policy_fraction_floor_bytes": None,
            "configured_fraction_floor_bytes": None,
            "lifecycle_reserve_bytes": lifecycle_reserve_bytes,
            "execution_reserve_bytes": execution_reserve_bytes,
            "reserve_model": dict(validated_policy.get("reserve_model") or {}),
            "known_minimum_floor_bytes": known_minimum_floor,
            "protected_free_floor_bytes": None,
            "usable_headroom_bytes": None,
            "protected_floor_deficit_bytes": None,
        }
    assert filesystem_total_bytes is not None
    assert filesystem_free_bytes is not None
    policy_numerator = int(minimum["minimum_fraction_numerator"])
    policy_denominator = int(minimum["minimum_fraction_denominator"])
    policy_fraction_floor = (
        filesystem_total_bytes * policy_numerator + policy_denominator - 1
    ) // policy_denominator
    if configured_minimum_free_fraction == policy_numerator / policy_denominator:
        configured_fraction_floor = policy_fraction_floor
    else:
        configured_numerator, configured_denominator = Fraction(
            str(configured_minimum_free_fraction)
        ).as_integer_ratio()
        configured_fraction_floor = (
            filesystem_total_bytes * configured_numerator + configured_denominator - 1
        ) // configured_denominator
    protected_floor = max(
        int(minimum["minimum_bytes"]),
        policy_fraction_floor,
        configured_fraction_floor,
        lifecycle_reserve_bytes,
        execution_reserve_bytes,
    )
    return {
        "schema_version": STORAGE_AVAILABILITY_SNAPSHOT_SCHEMA_VERSION,
        **authority,
        "status": "OBSERVED",
        "filesystem_total_bytes": filesystem_total_bytes,
        "filesystem_free_bytes": filesystem_free_bytes,
        "policy_minimum_bytes": int(minimum["minimum_bytes"]),
        "policy_fraction_floor_bytes": policy_fraction_floor,
        "configured_fraction_floor_bytes": configured_fraction_floor,
        "lifecycle_reserve_bytes": lifecycle_reserve_bytes,
        "execution_reserve_bytes": execution_reserve_bytes,
        "reserve_model": dict(validated_policy.get("reserve_model") or {}),
        "known_minimum_floor_bytes": known_minimum_floor,
        "protected_free_floor_bytes": protected_floor,
        "usable_headroom_bytes": max(0, filesystem_free_bytes - protected_floor),
        "protected_floor_deficit_bytes": max(
            0, protected_floor - filesystem_free_bytes
        ),
    }


def evidence_backed_execution_reserve_v1(
    *,
    policy: Mapping[str, Any],
    remaining_run_bytes: int,
    reservation_window_bytes: int,
    lifecycle_reserve_bytes: int,
    active_checkpoint_bytes: int = 0,
    acs_staging_bytes: int = 0,
    reducer_staging_bytes: int = 0,
    archive_staging_bytes: int = 0,
    quarantine_allowance_bytes: int = 0,
) -> dict[str, Any]:
    """Compute a bounded reserve from the admitted run's real envelope.

    ``remaining_run_bytes`` is the candidate reservation handled separately by
    ``RunStorageGuard``.  It contributes only to the uncertainty margin, so a
    large run is not silently double-counted while small runs are not exposed
    to a zero-byte filesystem reserve.  Legacy policies retain their old
    static floor and are intentionally conservative.
    """

    validated = validate_data2_storage_availability_policy_v1(policy)
    values = {
        "remaining_run_bytes": remaining_run_bytes,
        "reservation_window_bytes": reservation_window_bytes,
        "lifecycle_reserve_bytes": lifecycle_reserve_bytes,
        "active_checkpoint_bytes": active_checkpoint_bytes,
        "acs_staging_bytes": acs_staging_bytes,
        "reducer_staging_bytes": reducer_staging_bytes,
        "archive_staging_bytes": archive_staging_bytes,
        "quarantine_allowance_bytes": quarantine_allowance_bytes,
    }
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise StorageAvailabilityPolicyError(
            "execution reserve inputs must be non-negative integers"
        )
    model = validated.get("reserve_model")
    if not isinstance(model, Mapping):
        # Historical policy input is accepted, but it cannot receive a more
        # permissive interpretation than the floor it explicitly declared.
        minimum = dict(validated["protected_free_floor"])
        static_fraction = {
            "minimum_bytes": int(minimum["minimum_bytes"]),
            "minimum_fraction_numerator": int(minimum["minimum_fraction_numerator"]),
            "minimum_fraction_denominator": int(minimum["minimum_fraction_denominator"]),
        }
        return {
            "schema_version": "eiel_legacy_static_reserve_model_v1",
            "mode": "legacy_static_floor_v1",
            "remaining_run_bytes": remaining_run_bytes,
            "reservation_window_bytes": reservation_window_bytes,
            "lifecycle_reserve_bytes": lifecycle_reserve_bytes,
            "measured_transient_bytes": max(
                lifecycle_reserve_bytes,
                reservation_window_bytes,
                active_checkpoint_bytes,
            ),
            "filesystem_reserve_bytes": int(static_fraction["minimum_bytes"]),
            "uncertainty_margin_bytes": 0,
            "execution_reserve_bytes": max(
                int(static_fraction["minimum_bytes"]),
                lifecycle_reserve_bytes,
            ),
            "legacy_static_floor": static_fraction,
        }
    model = dict(model)
    window_peak = max(reservation_window_bytes, active_checkpoint_bytes)
    staging_peak = (
        acs_staging_bytes
        + reducer_staging_bytes
        + archive_staging_bytes
    )
    measured_transient = max(
        lifecycle_reserve_bytes,
        window_peak + staging_peak + quarantine_allowance_bytes,
    )
    uncertainty_base = max(remaining_run_bytes, measured_transient)
    uncertainty = (
        uncertainty_base * int(model["uncertainty_margin_numerator"])
        + int(model["uncertainty_margin_denominator"])
        - 1
    ) // int(model["uncertainty_margin_denominator"])
    uncertainty = min(uncertainty, int(model["max_uncertainty_bytes"]))
    execution_reserve = max(
        int(model["filesystem_reserve_bytes"]),
        measured_transient + uncertainty,
    )
    return {
        "schema_version": RESERVE_MODEL_SCHEMA_VERSION,
        "mode": RESERVE_MODEL_MODE,
        "remaining_run_bytes": remaining_run_bytes,
        "reservation_window_bytes": reservation_window_bytes,
        "active_checkpoint_bytes": active_checkpoint_bytes,
        "acs_staging_bytes": acs_staging_bytes,
        "reducer_staging_bytes": reducer_staging_bytes,
        "archive_staging_bytes": archive_staging_bytes,
        "quarantine_allowance_bytes": quarantine_allowance_bytes,
        "window_peak_bytes": window_peak,
        "staging_peak_bytes": staging_peak,
        "lifecycle_reserve_bytes": lifecycle_reserve_bytes,
        "measured_transient_bytes": measured_transient,
        "uncertainty_base_bytes": uncertainty_base,
        "uncertainty_margin_bytes": uncertainty,
        "filesystem_reserve_bytes": int(model["filesystem_reserve_bytes"]),
        "execution_reserve_bytes": execution_reserve,
    }


__all__ = [
    "CANONICAL_DATA2_STORAGE_AVAILABILITY_POLICY_PATH",
    "DATA2_COLD_PROFILE_ADMISSION_POLICY",
    "DATA2_FILESYSTEM_ROOT",
    "DATA2_COLD_PROFILE_MAX_AGGREGATE_BYTES",
    "DATA2_COLD_PROFILE_MAX_OBJECT_BYTES",
    "DATA2_LIFECYCLE_MANAGEMENT_ROOT",
    "DATA2_MINIMUM_FREE_FRACTION_DENOMINATOR",
    "DATA2_MINIMUM_FREE_FRACTION_NUMERATOR",
    "DATA2_MINIMUM_PROTECTED_FREE_BYTES",
    "DATA2_LEGACY_MINIMUM_FREE_FRACTION_DENOMINATOR",
    "DATA2_LEGACY_MINIMUM_FREE_FRACTION_NUMERATOR",
    "DATA2_LEGACY_MINIMUM_PROTECTED_FREE_BYTES",
    "DATA2_FILESYSTEM_RESERVE_BYTES",
    "DATA2_RESERVE_MAX_UNCERTAINTY_BYTES",
    "DATA2_RESERVE_UNCERTAINTY_DENOMINATOR",
    "DATA2_RESERVE_UNCERTAINTY_NUMERATOR",
    "DATA2_PERMANENT_EVIDENCE_MAX_CAPSULE_BYTES",
    "DATA2_PERMANENT_EVIDENCE_MAX_TOTAL_BYTES",
    "DATA2_PERMANENT_EVIDENCE_ROOT",
    "DATA2_RUNTIME_ROOT",
    "DATA2_STORAGE_AVAILABILITY_POLICY_ID",
    "DATA2_STORAGE_AVAILABILITY_POLICY_VERSION",
    "FILESYSTEM_IDENTITY_SCHEMA_VERSION",
    "STORAGE_AVAILABILITY_POLICY_SCHEMA_VERSION",
    "STORAGE_AVAILABILITY_SNAPSHOT_SCHEMA_VERSION",
    "StorageAvailabilityPolicyError",
    "RESERVE_MODEL_MODE",
    "RESERVE_MODEL_SCHEMA_VERSION",
    "build_storage_availability_snapshot_v1",
    "evidence_backed_execution_reserve_v1",
    "load_data2_storage_availability_policy_v1",
    "observe_filesystem_identity_v1",
    "storage_availability_policy_binding_v1",
    "validate_data2_storage_availability_policy_v1",
    "validate_data2_run_root_v1",
    "validate_filesystem_identity_v1",
]
