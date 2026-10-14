"""Canonical request/result execution front door for EIEL campaigns.

The front door deliberately composes the maintained campaign compiler,
preflight authority, provider registry, durable campaign runner, and artifact
lifecycle validators.  It owns request normalization and result aggregation;
it does not implement another simulation engine, scheduler, resource manager,
evidence store, or qualification database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time
from typing import Any, Callable, Final, Mapping, Protocol
import uuid

from .artifact_lifecycle import (
    ArtifactLifecycleError,
    ensure_lifecycle_contract_v1,
    validate_artifact_disposition_manifest_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from .admission_kernel_v1 import baseline_admission_shadow_v1
from .baseline_v1 import (
    BASELINE_ADMISSION_SCHEMA_VERSION,
    BASELINE_CONFORMANCE_SCHEMA_VERSION,
    BaselineDeclarationV1,
    BaselineError,
    baseline_result_binding_v1,
    build_baseline_admission_receipt_v1,
    build_baseline_conformance_receipt_v1,
    validate_baseline_receipt_binding_v1,
    validate_baseline_purpose_v1,
    write_baseline_receipt_v1,
)
from .canopy_supervision_v1 import (
    CanopySupervisionError,
    mark_canopy_source_interrupted_v1,
    record_canopy_observability_degraded_v1,
    start_canopy_relays_v1,
)
from .canonical import canonical_sha256
from .compiler import compile_campaign, load_campaign_spec
from .orchestration import (
    CampaignExecutionError,
    atomic_write_json,
    execute_campaign,
    reseal_terminal_lifecycle_inventory_v1,
)
from .preflight import preflight_campaign
from .production_engine_identity import PRODUCTION_CPU_PROFILE_ID
from .providers import (
    ExecutionProviderRegistryV1,
    default_execution_provider_registry,
)

EXECUTION_REQUEST_SCHEMA_VERSION: Final[str] = "eiel_execution_request_v1"
EXECUTION_REQUEST_V2_SCHEMA_VERSION: Final[str] = "eiel_execution_request_v2"
EXECUTION_REQUEST_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_request_receipt_v1"
)
EXECUTION_REQUEST_V2_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_request_receipt_v2"
)
EXECUTION_RESULT_SCHEMA_VERSION: Final[str] = "eiel_execution_result_v1"
EXECUTION_RESULT_V2_SCHEMA_VERSION: Final[str] = "eiel_execution_result_v2"
MECHANICAL_VERIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_mechanical_verification_v1"
)
QUALIFICATION_EVALUATION_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_qualification_evaluation_v1"
)
EXECUTION_PURPOSES: Final[tuple[str, ...]] = (
    "engineering",
    "simulation",
    "validation",
    "benchmark",
    "qualprep",
    "qualification",
)
ENGINEERING_EXECUTION_SCOPE_V1: Final[str] = "engineering_only"
RESTORE_PROCESS_ISOLATION_MODES: Final[tuple[str, ...]] = (
    "inline",
    "fresh_invocation_per_restore",
)
RESOURCE_WAITABLE_CODES: Final[frozenset[str]] = frozenset({"insufficient_free_vram"})
STORAGE_WAITABLE_CODES: Final[frozenset[str]] = frozenset(
    {"lifecycle_storage_admission_failed"}
)
_PROCESS_INSTANCE_ID: Final[str] = uuid.uuid4().hex
CRCA_CONFIGURATION_ADMISSION_CUTOVER_V1: Final[bool] = True
TERMINAL_LIFECYCLE_COMMIT_SCHEMA_VERSION: Final[str] = (
    "eiel_execution_terminal_lifecycle_commit_v1"
)
TERMINAL_LIFECYCLE_COMMIT_NAME: Final[str] = "terminal_lifecycle_commit.json"


class ExecutionFrontDoorError(ValueError):
    """A request or canonical execution binding is not safe to run."""


class CudaFullEvolutionV2LaunchpadCrcaAdapterV1(Protocol):
    """Composition/Scheduler capability injected at the Launchpad boundary."""

    def persist_shadow_projection_v1(
        self,
        *,
        target: Path,
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        baseline: Mapping[str, Any],
        source_identity: Mapping[str, Any],
        assurance: Mapping[str, Any],
        preflight: Mapping[str, Any],
        attempt_nonce: str,
        proposed_reason: str,
    ) -> Mapping[str, Any]: ...

    def bind_attempt_authority_v1(
        self,
        *,
        target: Path,
        request: Mapping[str, Any],
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        source_identity: Mapping[str, Any],
        preflight: Mapping[str, Any],
        shadow_projection: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def terminalize_attempt_authority_v1(
        self,
        *,
        attempt_authority_path: Path,
        campaign_plan: Mapping[str, Any],
        execution_status: str,
        execution_receipt: Mapping[str, Any] | None,
        result_binding: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def materialize_terminal_cuda_memory_release_v1(
        self,
        *,
        attempt_authority_path: Path,
        campaign_plan: Mapping[str, Any],
        bound_execution_request: Mapping[str, Any],
        observer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _CudaFullEvolutionV2ShadowStateV1:
    binding: Mapping[str, Any]
    projection: Mapping[str, Any]
    bound_execution_request: Mapping[str, Any]


def _observe_terminal_cuda_memory_release_v1(
    device_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Release process-local PyTorch allocations and report exact zero counters."""

    index = device_identity.get("index")
    if type(index) is not int or index < 0:
        raise ExecutionFrontDoorError("shared CUDA release lacks a device index")
    import torch

    device = f"cuda:{index}"
    started_at_utc = _now()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return {
        "schema_version": "eiel_cuda_allocator_release_observation_v1",
        "device_index": index,
        "garbage_collection_completed": True,
        "cuda_synchronize_completed": True,
        "empty_cache_completed": True,
        "memory_allocated_bytes": allocated,
        "memory_reserved_bytes": reserved,
        "started_at_utc": started_at_utc,
        "completed_at_utc": _now(),
    }


def _configuration_admission_allowed_v1(
    *,
    admission_receipt: Mapping[str, Any],
    request_sha256: str,
    plan_sha256: str,
) -> bool:
    """Cut over one low-risk intent with the direct policy retained as oracle."""

    legacy_allowed = admission_receipt.get("configuration_eligible") is True
    shadow = baseline_admission_shadow_v1(
        front_door_id=EXECUTION_REQUEST_V2_SCHEMA_VERSION,
        intent="configuration",
        request_sha256=request_sha256,
        plan_sha256=plan_sha256,
        admission_receipt=admission_receipt,
        execution_required=False,
    )
    if CRCA_CONFIGURATION_ADMISSION_CUTOVER_V1:
        return shadow["decision"]["allowed"] is True
    return legacy_allowed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _process_identity() -> dict[str, Any]:
    """Return an invocation-stable identity that changes across OS processes."""

    return {
        "schema_version": "eiel_execution_process_identity_v1",
        "pid": os.getpid(),
        "process_instance_id": _PROCESS_INSTANCE_ID,
    }


def _validated_restore_process_identity(value: Any, *, name: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "pid", "process_instance_id"}
        or value.get("schema_version") != "eiel_execution_process_identity_v1"
        or type(value.get("pid")) is not int
        or value["pid"] <= 0
        or not isinstance(value.get("process_instance_id"), str)
        or not value["process_instance_id"]
    ):
        raise ExecutionFrontDoorError(f"{name} is not an exact process identity")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionFrontDoorError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ExecutionFrontDoorError(f"{name} must be a JSON object: {path}")
    return dict(value)


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionFrontDoorError(f"{path} must be a non-empty string")
    return value.strip()


def _exact(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ExecutionFrontDoorError(f"{path} contains unsupported fields: {unknown}")


def _resolve_path(value: str, *, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


@dataclass(frozen=True, slots=True)
class ResourceWaitPolicyV1:
    """Bounded, sparse waiting for a known transient resource admission result."""

    enabled: bool = False
    poll_seconds: int = 300
    max_wait_seconds: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> "ResourceWaitPolicyV1":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ExecutionFrontDoorError("resource_wait must be an object")
        raw = dict(value)
        _exact(
            raw,
            {"enabled", "poll_seconds", "max_wait_seconds"},
            path="resource_wait",
        )
        enabled = raw.get("enabled", False)
        poll_seconds = raw.get("poll_seconds", 300)
        max_wait_seconds = raw.get("max_wait_seconds")
        if type(enabled) is not bool:
            raise ExecutionFrontDoorError("resource_wait.enabled must be boolean")
        if type(poll_seconds) is not int or poll_seconds < 5:
            raise ExecutionFrontDoorError(
                "resource_wait.poll_seconds must be an integer >= 5"
            )
        if max_wait_seconds is not None and (
            type(max_wait_seconds) is not int or max_wait_seconds <= 0
        ):
            raise ExecutionFrontDoorError(
                "resource_wait.max_wait_seconds must be a positive integer or null"
            )
        return cls(enabled, poll_seconds, max_wait_seconds)

    def payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "poll_seconds": self.poll_seconds,
            "max_wait_seconds": self.max_wait_seconds,
        }


@dataclass(frozen=True, slots=True)
class ExecutionRequestV1:
    """Machine-readable WHAT for one canonical campaign execution."""

    purpose: str
    campaign_spec: str
    output_root: str | None = None
    resource_wait: ResourceWaitPolicyV1 = ResourceWaitPolicyV1()
    qualification_profile: Mapping[str, Any] | None = None
    restore_process_isolation: str = "inline"
    schema_version: str = EXECUTION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.restore_process_isolation not in RESTORE_PROCESS_ISOLATION_MODES:
            raise ExecutionFrontDoorError(
                "restore_process_isolation must be one of "
                f"{RESTORE_PROCESS_ISOLATION_MODES!r}"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> "ExecutionRequestV1":
        if not isinstance(value, Mapping):
            raise ExecutionFrontDoorError("execution request must be an object")
        raw = dict(value)
        _exact(
            raw,
            {
                "schema_version",
                "purpose",
                "campaign_spec",
                "output_root",
                "resource_wait",
                "qualification_profile",
                "restore_process_isolation",
                "request_sha256",
            },
            path="execution_request",
        )
        if raw.get("schema_version", EXECUTION_REQUEST_SCHEMA_VERSION) != (
            EXECUTION_REQUEST_SCHEMA_VERSION
        ):
            raise ExecutionFrontDoorError("unsupported execution request schema")
        purpose = _text(raw.get("purpose"), path="purpose")
        if purpose not in EXECUTION_PURPOSES:
            raise ExecutionFrontDoorError(
                f"purpose must be one of {EXECUTION_PURPOSES!r}"
            )
        restore_process_isolation = raw.get("restore_process_isolation", "inline")
        if restore_process_isolation not in RESTORE_PROCESS_ISOLATION_MODES:
            raise ExecutionFrontDoorError(
                "restore_process_isolation must be one of "
                f"{RESTORE_PROCESS_ISOLATION_MODES!r}"
            )
        root = Path(base_dir or Path.cwd()).expanduser().resolve()
        campaign_spec = _resolve_path(
            _text(raw.get("campaign_spec"), path="campaign_spec"),
            base_dir=root,
        )
        output_value = raw.get("output_root")
        output_root = (
            None
            if output_value is None
            else _resolve_path(_text(output_value, path="output_root"), base_dir=root)
        )
        profile_value = raw.get("qualification_profile")
        qualification_profile: Mapping[str, Any] | None
        if profile_value is None:
            qualification_profile = None
        else:
            if not isinstance(profile_value, Mapping):
                raise ExecutionFrontDoorError("qualification_profile must be an object")
            profile = dict(profile_value)
            _exact(
                profile,
                {"type", "freeze", "evidence"},
                path="qualification_profile",
            )
            if profile.get("type") != "tensor_world_c7":
                raise ExecutionFrontDoorError(
                    "qualification_profile.type must be 'tensor_world_c7'"
                )
            for name in ("freeze", "evidence"):
                profile[name] = _resolve_path(
                    _text(profile.get(name), path=f"qualification_profile.{name}"),
                    base_dir=root,
                )
            qualification_profile = profile
        request = cls(
            purpose=purpose,
            campaign_spec=campaign_spec,
            output_root=output_root,
            resource_wait=ResourceWaitPolicyV1.from_value(raw.get("resource_wait")),
            qualification_profile=qualification_profile,
            restore_process_isolation=restore_process_isolation,
        )
        declared_hash = raw.get("request_sha256")
        if declared_hash is not None and declared_hash != request.request_sha256:
            raise ExecutionFrontDoorError("execution request SHA-256 mismatch")
        return request

    @classmethod
    def from_file(cls, path: str | Path) -> "ExecutionRequestV1":
        request_path = Path(path).expanduser().resolve()
        return cls.from_mapping(
            _read_object(request_path, name="execution request"),
            base_dir=request_path.parent,
        )

    def canonical_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "campaign_spec": self.campaign_spec,
            "output_root": self.output_root,
            "resource_wait": self.resource_wait.payload(),
            "qualification_profile": (
                None
                if self.qualification_profile is None
                else dict(self.qualification_profile)
            ),
        }
        # Preserve hashes for all existing requests.  Process isolation is an
        # orchestration-only opt-in and enters the request identity only when
        # it changes the default inline continuation behavior.
        if self.restore_process_isolation != "inline":
            payload["restore_process_isolation"] = self.restore_process_isolation
        return payload

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.canonical_payload())

    def payload(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class ExecutionRequestV2:
    """Canonical campaign execution request with mandatory Baseline intent."""

    purpose: str
    campaign_spec: str
    baseline: BaselineDeclarationV1
    output_root: str | None = None
    resource_wait: ResourceWaitPolicyV1 = ResourceWaitPolicyV1()
    qualification_profile: Mapping[str, Any] | None = None
    restore_process_isolation: str = "inline"
    schema_version: str = EXECUTION_REQUEST_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_REQUEST_V2_SCHEMA_VERSION:
            raise ExecutionFrontDoorError("unsupported execution request schema")
        if self.purpose not in EXECUTION_PURPOSES:
            raise ExecutionFrontDoorError(
                f"purpose must be one of {EXECUTION_PURPOSES!r}"
            )
        if not isinstance(self.baseline, BaselineDeclarationV1):
            raise ExecutionFrontDoorError(
                "ExecutionRequestV2 requires a Baseline declaration"
            )
        if self.restore_process_isolation not in RESTORE_PROCESS_ISOLATION_MODES:
            raise ExecutionFrontDoorError(
                "restore_process_isolation must be one of "
                f"{RESTORE_PROCESS_ISOLATION_MODES!r}"
            )
        try:
            validate_baseline_purpose_v1(self.baseline, purpose=self.purpose)
        except BaselineError as exc:
            raise ExecutionFrontDoorError(str(exc)) from exc

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> "ExecutionRequestV2":
        if not isinstance(value, Mapping):
            raise ExecutionFrontDoorError("execution request must be an object")
        raw = dict(value)
        _exact(
            raw,
            {
                "schema_version",
                "purpose",
                "campaign_spec",
                "output_root",
                "resource_wait",
                "qualification_profile",
                "baseline",
                "restore_process_isolation",
                "request_sha256",
            },
            path="execution_request",
        )
        if raw.get("schema_version") != EXECUTION_REQUEST_V2_SCHEMA_VERSION:
            raise ExecutionFrontDoorError("unsupported execution request schema")
        legacy_raw = {
            key: item
            for key, item in raw.items()
            if key not in {"baseline", "request_sha256"}
        }
        legacy_raw["schema_version"] = EXECUTION_REQUEST_SCHEMA_VERSION
        base = ExecutionRequestV1.from_mapping(legacy_raw, base_dir=base_dir)
        try:
            declaration = BaselineDeclarationV1.from_value(raw.get("baseline"))
            validate_baseline_purpose_v1(declaration, purpose=base.purpose)
        except BaselineError as exc:
            raise ExecutionFrontDoorError(str(exc)) from exc
        request = cls(
            purpose=base.purpose,
            campaign_spec=base.campaign_spec,
            output_root=base.output_root,
            resource_wait=base.resource_wait,
            qualification_profile=base.qualification_profile,
            restore_process_isolation=base.restore_process_isolation,
            baseline=declaration,
        )
        declared_hash = raw.get("request_sha256")
        if declared_hash is not None and declared_hash != request.request_sha256:
            raise ExecutionFrontDoorError("execution request SHA-256 mismatch")
        return request

    @classmethod
    def from_file(cls, path: str | Path) -> "ExecutionRequestV2":
        request_path = Path(path).expanduser().resolve()
        return cls.from_mapping(
            _read_object(request_path, name="execution request"),
            base_dir=request_path.parent,
        )

    def canonical_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "campaign_spec": self.campaign_spec,
            "output_root": self.output_root,
            "resource_wait": self.resource_wait.payload(),
            "qualification_profile": (
                None
                if self.qualification_profile is None
                else dict(self.qualification_profile)
            ),
            "baseline": self.baseline.payload(),
        }
        if self.restore_process_isolation != "inline":
            payload["restore_process_isolation"] = self.restore_process_isolation
        return payload

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.canonical_payload())

    def payload(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "request_sha256": self.request_sha256}


ExecutionRequest = ExecutionRequestV1 | ExecutionRequestV2


def load_execution_request_v1(path: str | Path) -> ExecutionRequest:
    """Load the exact request version without treating V1 as Baseline."""

    request_path = Path(path).expanduser().resolve()
    raw = _read_object(request_path, name="execution request")
    if raw.get("schema_version") == EXECUTION_REQUEST_V2_SCHEMA_VERSION:
        return ExecutionRequestV2.from_mapping(raw, base_dir=request_path.parent)
    return ExecutionRequestV1.from_mapping(raw, base_dir=request_path.parent)


def _git_identity(repo_root: Path) -> dict[str, Any]:
    """Capture source/tree identity without treating unrelated dirt as authority."""

    # The run-contract compiler already accepts this explicit identity for
    # installed distributions. Carry the same input through Launchpad into
    # CRCA; an absent Git worktree must never be represented as clean.
    override = os.environ.get("EIEL_SOURCE_GIT_COMMIT", "").strip()
    if override:
        from .run_contract_v1 import _normalized_source_identity

        try:
            _normalized_source_identity({"git_commit": override})
        except ValueError as exc:
            raise ExecutionFrontDoorError(str(exc)) from exc

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        if override and override != commit:
            raise ExecutionFrontDoorError(
                "EIEL_SOURCE_GIT_COMMIT conflicts with repository HEAD"
            )
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        return {
            "repository_root": str(repo_root),
            "commit": commit,
            "working_tree_clean": status == "",
            "working_tree_status_sha256": canonical_sha256(status.splitlines()),
            "working_tree_status_lines": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        if override:
            return {
                "repository_root": str(repo_root),
                "commit": override,
                "identity_available": True,
                "discovery": "EIEL_SOURCE_GIT_COMMIT",
                "working_tree_clean": None,
                "working_tree_status_sha256": None,
                "working_tree_status_lines": None,
                "working_tree_status_known": False,
            }
        return {
            "repository_root": str(repo_root),
            "identity_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _plan_payload(plan: Any) -> dict[str, Any]:
    if isinstance(plan, Mapping):
        return dict(plan)
    payload = getattr(plan, "payload", None)
    if not callable(payload):
        raise ExecutionFrontDoorError("compiled campaign does not expose a payload")
    value = payload()
    if not isinstance(value, Mapping):
        raise ExecutionFrontDoorError("compiled campaign payload is not an object")
    return dict(value)


def _bind_runtime_root(plan: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    bound = dict(plan)
    output = bound.get("output")
    if not isinstance(output, Mapping):
        raise ExecutionFrontDoorError("compiled campaign lacks output mapping")
    bound["output"] = {**dict(output), "runtime_root": str(runtime_root)}
    bound.pop("plan_sha256", None)
    bound["plan_sha256"] = canonical_sha256(bound)
    return bound


def _validate_bound_campaign_plan_v1(
    campaign_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an already-materialized campaign plan before dispatch.

    The ordinary request path validates the plan indirectly while compiling a
    campaign specification.  Science-instrument lowering already has a
    materialized plan, so this seam must perform the non-negotiable plan
    identity checks without re-reading or re-interpreting the authoring
    specification.  Runtime, provider, and lifecycle validation remains owned
    by the canonical preflight/execution path.
    """

    if not isinstance(campaign_plan, Mapping):
        raise ExecutionFrontDoorError("bound campaign plan must be an object")
    plan = dict(campaign_plan)
    if plan.get("schema_version") != "eiel_campaign_plan_v1":
        raise ExecutionFrontDoorError(
            "bound campaign plan schema must be eiel_campaign_plan_v1"
        )
    declared = plan.get("plan_sha256")
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise ExecutionFrontDoorError(
            "bound campaign plan lacks an exact plan_sha256"
        )
    core = dict(plan)
    core.pop("plan_sha256", None)
    try:
        expected = canonical_sha256(core)
    except (TypeError, ValueError) as exc:
        raise ExecutionFrontDoorError(
            "bound campaign plan is not canonical JSON"
        ) from exc
    if declared != expected:
        raise ExecutionFrontDoorError(
            "bound campaign plan plan_sha256 does not match canonical content"
        )
    for field in (
        "campaign_id",
        "resolved_profile",
        "execution",
        "cells",
        "physical_batches",
        "artifacts",
        "output",
        "resource_estimate",
    ):
        if field not in plan:
            raise ExecutionFrontDoorError(
                f"bound campaign plan lacks required field: {field}"
            )
    if not isinstance(plan["campaign_id"], str) or not plan["campaign_id"]:
        raise ExecutionFrontDoorError("bound campaign plan campaign_id is invalid")
    if not isinstance(plan["resolved_profile"], str) or not plan["resolved_profile"]:
        raise ExecutionFrontDoorError(
            "bound campaign plan resolved_profile is invalid"
        )
    if not isinstance(plan["execution"], Mapping):
        raise ExecutionFrontDoorError("bound campaign plan execution is not an object")
    if not isinstance(plan["cells"], list):
        raise ExecutionFrontDoorError("bound campaign plan cells is not an array")
    if not isinstance(plan["physical_batches"], list):
        raise ExecutionFrontDoorError(
            "bound campaign plan physical_batches is not an array"
        )
    for field in ("artifacts", "output", "resource_estimate"):
        if not isinstance(plan[field], Mapping):
            raise ExecutionFrontDoorError(
                f"bound campaign plan {field} is not an object"
            )
    if not isinstance(plan.get("artifact_lifecycle"), Mapping):
        raise ExecutionFrontDoorError(
            "bound campaign plan requires an explicit artifact_lifecycle contract"
        )
    return plan


def _bound_campaign_source_identity_v1(
    *,
    plan: Mapping[str, Any],
    bound_plan: Mapping[str, Any],
    source_lineage: Mapping[str, Any],
    science_lineage: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Build the exact source identity for a bound science-plan execution."""

    if not isinstance(source_lineage, Mapping):
        raise ExecutionFrontDoorError("source_lineage must be an object")
    if not isinstance(science_lineage, Mapping):
        raise ExecutionFrontDoorError("science_lineage must be an object")
    source = dict(source_lineage)
    science = dict(science_lineage)
    compiled_plan_sha256 = str(plan["plan_sha256"])
    bound_plan_sha256 = str(bound_plan["plan_sha256"])
    for name, expected in (
        ("compiled_plan_sha256", compiled_plan_sha256),
        ("bound_plan_sha256", bound_plan_sha256),
    ):
        supplied = source.get(name)
        if supplied is not None and supplied != expected:
            raise ExecutionFrontDoorError(
                f"source_lineage.{name} does not match the bound plan"
            )
    try:
        source_sha256 = canonical_sha256(source)
        science_sha256 = canonical_sha256(science)
    except (TypeError, ValueError) as exc:
        raise ExecutionFrontDoorError(
            "source_lineage and science_lineage must be canonical JSON"
        ) from exc
    identity = {
        "schema_version": "eiel_bound_campaign_source_identity_v1",
        "compiled_plan_sha256": compiled_plan_sha256,
        "bound_plan_sha256": bound_plan_sha256,
        "source_lineage": source,
        "source_lineage_sha256": source_sha256,
        "science_lineage": science,
        "science_lineage_sha256": science_sha256,
        "git": _git_identity(repo_root),
    }
    # Preserve the established Launchpad source fields when the explicit
    # source lineage provides them.  Native CRCA and historical evidence
    # validators consume these top-level projections; the complete lineage
    # remains sealed under ``source_lineage`` above.
    for name in ("campaign_spec_path", "campaign_spec_sha256"):
        if name in source:
            identity[name] = source[name]
    return identity


def _resolve_compiled_plan(
    request: ExecutionRequest,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
    spec_path = Path(request.campaign_spec).resolve()
    if not spec_path.is_file() or spec_path.is_symlink():
        raise ExecutionFrontDoorError(
            f"campaign_spec must be an existing regular file: {spec_path}"
        )
    try:
        source_sha256 = _sha256_file(spec_path)
        spec = load_campaign_spec(spec_path)
        if _is_cuda_full_evolution_workhorse_v2_spec(spec):
            from .cuda_full_evolution_v2_workhorse_provider import (
                cuda_full_evolution_workhorse_v2_profile_registry_v1,
                cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2,
                materialize_workhorse_exact_founder_plan_v1,
            )

            extension = spec.get("analysis", {}).get("extension", {})
            operation = extension.get("contract", {})
            workhorse_contract = operation.get("contract", {})
            operation_name = operation.get("operation")
            shape_id = (
                workhorse_contract.get("shape_id")
                if isinstance(workhorse_contract, Mapping)
                else None
            ) or operation.get("shape_id")
            if not isinstance(shape_id, str):
                raise ExecutionFrontDoorError(
                    "Workhorse operation lacks its compile shape"
                )
            if not isinstance(operation_name, str):
                raise ExecutionFrontDoorError(
                    "Workhorse operation lacks its exact discriminator"
                )
            compiled = compile_campaign(
                spec,
                registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
                    shape_id=shape_id,
                    operation=operation_name,
                ),
                planning_resource_accounting=(
                    cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
                ),
            )
            plan = materialize_workhorse_exact_founder_plan_v1(
                _plan_payload(compiled)
            )
        elif _is_cuda_full_evolution_v2_spec(spec):
            from .cuda_full_evolution_v2_provider import (
                cuda_full_evolution_v2_profile_registry_v1,
                cuda_full_evolution_v2_signed_planning_resource_accounting_v1,
            )

            compiled = compile_campaign(
                spec,
                registry=cuda_full_evolution_v2_profile_registry_v1(),
                planning_resource_accounting=(
                    cuda_full_evolution_v2_signed_planning_resource_accounting_v1
                ),
            )
        elif _is_tensor_world_c1_spec(spec):
            from .tensor_world_c1_provider import (
                tensor_world_c1_profile_registry_v1,
            )

            compiled = compile_campaign(
                spec, registry=tensor_world_c1_profile_registry_v1()
            )
        elif _is_tw_rngv2_graph_fusion_spec(spec):
            from .tw_rngv2_graph_fusion_provider import (
                tw_rngv2_graph_fusion_profile_registry_v1,
            )

            compiled = compile_campaign(
                spec, registry=tw_rngv2_graph_fusion_profile_registry_v1()
            )
        else:
            compiled = compile_campaign(spec)
        if not _is_cuda_full_evolution_workhorse_v2_spec(spec):
            plan = _plan_payload(compiled)
    except Exception as exc:
        if isinstance(exc, ExecutionFrontDoorError):
            raise
        raise ExecutionFrontDoorError(
            f"campaign compilation failed: {type(exc).__name__}: {exc}"
        ) from exc
    compiled_plan_sha256 = str(plan.get("plan_sha256", ""))
    if not compiled_plan_sha256:
        raise ExecutionFrontDoorError("compiled campaign lacks plan_sha256")
    configured_root = request.output_root
    if configured_root is None:
        output = plan.get("output")
        configured_root = (
            str(output.get("runtime_root"))
            if isinstance(output, Mapping) and output.get("runtime_root")
            else None
        )
    if not configured_root:
        raise ExecutionFrontDoorError(
            "execution request and compiled campaign both lack output_root"
        )
    configured_path = Path(configured_root).expanduser()
    if configured_path.is_symlink():
        raise ExecutionFrontDoorError("output_root may not be a symlink")
    runtime_root = configured_path
    if not runtime_root.is_absolute():
        runtime_root = repo_root / runtime_root
    runtime_root = runtime_root.resolve()
    if runtime_root == repo_root:
        raise ExecutionFrontDoorError("output_root may not be the repository root")
    if runtime_root.exists() and runtime_root.is_file():
        raise ExecutionFrontDoorError("output_root is an existing regular file")
    bound = _bind_runtime_root(plan, runtime_root)
    source_identity = {
        "campaign_spec_path": str(spec_path),
        "campaign_spec_sha256": source_sha256,
        "compiled_plan_sha256": compiled_plan_sha256,
        "bound_plan_sha256": bound["plan_sha256"],
        "git": _git_identity(repo_root),
    }
    artifact_lifecycle_explicit = (
        isinstance(spec.get("artifact_lifecycle"), Mapping)
        and spec.get("_legacy_lifecycle_derived") is not True
    )
    return bound, source_identity, str(runtime_root), artifact_lifecycle_explicit


def _is_tensor_world_c1_spec(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    from .tensor_world_c1_provider import is_tensor_world_c1_spec_v1

    return is_tensor_world_c1_spec_v1(value)


def _is_cuda_full_evolution_v2_spec(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    from .cuda_full_evolution_v2_provider import (
        is_cuda_full_evolution_v2_spec_v1,
    )

    return is_cuda_full_evolution_v2_spec_v1(value)


def _is_cuda_full_evolution_workhorse_v2_spec(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    from .cuda_full_evolution_v2_workhorse_provider import (
        is_cuda_full_evolution_workhorse_v2_spec_v1,
    )

    return is_cuda_full_evolution_workhorse_v2_spec_v1(value)


def _is_cuda_full_evolution_v2_plan_v1(value: Mapping[str, Any]) -> bool:
    """Identify the two explicitly unqualified Full Evolution V2 route families."""

    from .cuda_full_evolution_v2_provider import (
        PROFILE_ID as CUDA_FULL_EVOLUTION_V2_PROFILE_ID,
    )
    from .cuda_full_evolution_v2_workhorse_provider import (
        is_cuda_full_evolution_workhorse_v2_profile_v1,
    )

    profile_id = value.get("resolved_profile")
    return profile_id == CUDA_FULL_EVOLUTION_V2_PROFILE_ID or (
        is_cuda_full_evolution_workhorse_v2_profile_v1(profile_id)
    )


def _execution_authority_scope_v1(
    *, purpose: str, plan: Mapping[str, Any]
) -> str:
    """Project one scope identically into preflight and campaign dispatch."""

    if purpose == "qualprep":
        return "qualprep_development_only"
    if purpose in {"engineering", "simulation", "validation", "benchmark"} and (
        _is_cuda_full_evolution_v2_plan_v1(plan)
        or plan.get("resolved_profile") == PRODUCTION_CPU_PROFILE_ID
    ):
        return "engineering_unqualified_real_use"
    return "qualified_execution"


def _is_tw_rngv2_graph_fusion_spec(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    from .tw_rngv2_graph_fusion_provider import is_tw_rngv2_graph_fusion_spec_v1

    return is_tw_rngv2_graph_fusion_spec_v1(value)


def _provider_registry_for_plan_v1(
    plan: Mapping[str, Any],
    *,
    cuda_full_evolution_v2_checkpoint_occurrence_adapter: Any | None = None,
) -> ExecutionProviderRegistryV1:
    from .cuda_full_evolution_v2_provider import (
        PROFILE_ID as CUDA_FULL_EVOLUTION_V2_PROFILE_ID,
        cuda_full_evolution_v2_provider_registry_v1,
    )

    from .cuda_full_evolution_v2_workhorse_provider import (
        cuda_full_evolution_workhorse_v2_provider_registry_v1,
        is_cuda_full_evolution_workhorse_v2_profile_v1,
    )

    if is_cuda_full_evolution_workhorse_v2_profile_v1(
        plan.get("resolved_profile")
    ):
        return cuda_full_evolution_workhorse_v2_provider_registry_v1(
            operation_adapter=(
                cuda_full_evolution_v2_checkpoint_occurrence_adapter
            )
        )

    if plan.get("resolved_profile") == CUDA_FULL_EVOLUTION_V2_PROFILE_ID:
        return cuda_full_evolution_v2_provider_registry_v1(
            checkpoint_occurrence_adapter=(
                cuda_full_evolution_v2_checkpoint_occurrence_adapter
            )
        )
    from .tensor_world_c1_provider import (
        TENSOR_WORLD_C1_PROFILE_ID,
        tensor_world_c1_provider_registry_v1,
    )

    if plan.get("resolved_profile") == TENSOR_WORLD_C1_PROFILE_ID:
        return tensor_world_c1_provider_registry_v1()
    from .tw_rngv2_graph_fusion_provider import (
        PROFILE_ID as GRAPH_FUSION_PROFILE_ID,
        tw_rngv2_graph_fusion_provider_registry_v1,
    )

    if plan.get("resolved_profile") == GRAPH_FUSION_PROFILE_ID:
        return tw_rngv2_graph_fusion_provider_registry_v1()
    return default_execution_provider_registry()


def _profile_registry_for_plan_v1(plan: Mapping[str, Any]) -> Any | None:
    from .cuda_full_evolution_v2_provider import (
        PROFILE_ID as CUDA_FULL_EVOLUTION_V2_PROFILE_ID,
        cuda_full_evolution_v2_profile_registry_v1,
    )

    from .cuda_full_evolution_v2_workhorse_provider import (
        cuda_full_evolution_workhorse_v2_profile_registry_v1,
        is_cuda_full_evolution_workhorse_v2_profile_v1,
    )

    if is_cuda_full_evolution_workhorse_v2_profile_v1(
        plan.get("resolved_profile")
    ):
        suffix = str(plan["resolved_profile"]).split("__", 1)
        if len(suffix) != 2:
            raise ExecutionFrontDoorError("Workhorse profile shape is absent")
        analysis = plan.get("analysis")
        extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
        operation = (
            extension.get("contract", {}).get("operation")
            if isinstance(extension, Mapping)
            and isinstance(extension.get("contract"), Mapping)
            else None
        )
        if not isinstance(operation, str):
            raise ExecutionFrontDoorError("Workhorse plan operation is absent")
        return cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=suffix[1], operation=operation
        )

    if plan.get("resolved_profile") == CUDA_FULL_EVOLUTION_V2_PROFILE_ID:
        return cuda_full_evolution_v2_profile_registry_v1()
    from .tensor_world_c1_provider import (
        TENSOR_WORLD_C1_PROFILE_ID,
        tensor_world_c1_profile_registry_v1,
    )

    if plan.get("resolved_profile") == TENSOR_WORLD_C1_PROFILE_ID:
        return tensor_world_c1_profile_registry_v1()
    from .tw_rngv2_graph_fusion_provider import (
        PROFILE_ID as GRAPH_FUSION_PROFILE_ID,
        tw_rngv2_graph_fusion_profile_registry_v1,
    )

    if plan.get("resolved_profile") == GRAPH_FUSION_PROFILE_ID:
        return tw_rngv2_graph_fusion_profile_registry_v1()
    return None


def _persist_cuda_full_evolution_v2_shadow_v1(
    *,
    request: ExecutionRequestV2,
    plan: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    preflight: Mapping[str, Any],
    runtime_root: Path,
    provider_registry: ExecutionProviderRegistryV1,
    crca_adapter: CudaFullEvolutionV2LaunchpadCrcaAdapterV1 | None,
) -> _CudaFullEvolutionV2ShadowStateV1 | None:
    """Persist one shadow-only CRCA projection for the V2 candidate route."""

    from .preflight import build_execution_request_v1

    from .cuda_full_evolution_v2_provider import (
        PROFILE_ID as CUDA_FULL_EVOLUTION_V2_PROFILE_ID,
    )
    from .cuda_full_evolution_v2_workhorse_provider import (
        is_cuda_full_evolution_workhorse_v2_profile_v1,
    )

    profile_id = plan.get("resolved_profile")
    if (
        profile_id != CUDA_FULL_EVOLUTION_V2_PROFILE_ID
        and not is_cuda_full_evolution_workhorse_v2_profile_v1(profile_id)
    ):
        return None
    if crca_adapter is None:
        raise ExecutionFrontDoorError(
            "CUDA Full Evolution V2 requires an injected canonical CRCA adapter"
        )
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or not batches:
        raise ExecutionFrontDoorError(
            "CUDA Full Evolution V2 CRCA projection lacks physical batches"
        )
    batch_ids = [
        str(batch.get("batch_id"))
        for batch in batches
        if isinstance(batch, Mapping) and isinstance(batch.get("batch_id"), str)
    ]
    if len(batch_ids) != len(batches) or len(batch_ids) != len(set(batch_ids)):
        raise ExecutionFrontDoorError(
            "CUDA Full Evolution V2 CRCA projection has malformed physical batches"
        )
    execution = plan.get("execution")
    gpu_policy = (
        execution.get("gpu_resource_policy")
        if isinstance(execution, Mapping)
        else None
    )
    shared_w32 = (
        isinstance(gpu_policy, Mapping)
        and gpu_policy.get("policy_id") == "w32x32p256c512_gpuv2_shared"
    )
    if not shared_w32 and len(batch_ids) != 1:
        raise ExecutionFrontDoorError(
            "CUDA Full Evolution V2 exclusive CRCA projection requires one physical batch"
        )
    provider = provider_registry.get(str(profile_id))
    if provider is None:
        raise ExecutionFrontDoorError(
            "CUDA Full Evolution V2 CRCA projection lacks its provider"
        )
    bound_requests = [
        build_execution_request_v1(provider, plan, batch_id).payload()
        for batch_id in batch_ids
    ]
    bound_request = bound_requests[0]
    if shared_w32:
        batch_set = bound_request.get("physical_batch_set")
        if (
            not isinstance(batch_set, Mapping)
            or batch_set.get("schema_version")
            != "eiel_cuda_full_evolution_v2_physical_batch_set_v1"
            or batch_set.get("gpu_resource_policy_id")
            != "w32x32p256c512_gpuv2_shared"
            or batch_set.get("batch_ids") != batch_ids
            or batch_set.get("physical_batch_count") != len(batch_ids)
            or any(
                candidate.get("physical_batch_set") != batch_set
                for candidate in bound_requests
            )
            or [candidate.get("source_batch_id") for candidate in bound_requests]
            != batch_ids
            or len(
                {
                    candidate.get("execution_route_sha256")
                    for candidate in bound_requests
                }
            )
            != 1
        ):
            raise ExecutionFrontDoorError(
                "shared W32 CRCA projection differs from its ordered physical batch set"
            )
        expected_bindings = [
            {
                "batch_id": batch_id,
                "resident_request_sha256": canonical_sha256(
                    candidate["resident_request"]
                ),
            }
            for batch_id, candidate in zip(batch_ids, bound_requests, strict=True)
        ]
        if (
            batch_set.get("ordered_resident_request_bindings")
            != expected_bindings
            or batch_set.get("ordered_resident_request_bindings_sha256")
            != canonical_sha256(expected_bindings)
        ):
            raise ExecutionFrontDoorError(
                "shared W32 CRCA request bindings differ from the compiled partition"
            )
    source_builder = getattr(provider, "crca_source_identity_v1", None)
    assurance_builder = getattr(provider, "crca_assurance_v1", None)
    if not callable(source_builder) or not callable(assurance_builder):
        raise ExecutionFrontDoorError(
            "CUDA Full Evolution V2 provider lacks CRCA projection inputs"
        )
    target = runtime_root / "control" / "crca" / "shadow-execution-projection.json"
    projection = crca_adapter.persist_shadow_projection_v1(
        target=target,
        campaign_plan=plan,
        bound_execution_request=bound_request,
        baseline=request.baseline.payload(),
        source_identity=source_builder(plan, source_identity, preflight),
        assurance=assurance_builder(plan, preflight),
        preflight=preflight,
        attempt_nonce=f"launchpad:{request.purpose}:{request.request_sha256}",
        proposed_reason=(
            f"CUDA Full Evolution V2 {request.purpose} candidate shadow projection"
        ),
    )
    if not isinstance(projection, Mapping):
        raise ExecutionFrontDoorError("CRCA adapter returned no shadow projection")
    if (
        not target.is_file()
        or target.is_symlink()
        or _read_object(target, name="CUDA Full Evolution V2 shadow projection")
        != dict(projection)
    ):
        raise ExecutionFrontDoorError(
            "CRCA adapter did not persist its exact shadow projection"
        )
    authority = projection.get("authority")
    projection_sha256 = projection.get("projection_sha256")
    if not isinstance(authority, str) or not authority.startswith("SHADOW_ONLY_"):
        raise ExecutionFrontDoorError("CRCA projection is not shadow-only")
    if (
        not isinstance(projection_sha256, str)
        or len(projection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in projection_sha256)
    ):
        raise ExecutionFrontDoorError("CRCA projection digest is invalid")
    binding = {
            "schema_version": "eiel_cuda_full_evolution_v2_crca_shadow_binding_v1",
            "authority": authority,
            "path": str(target),
            "projection_sha256": projection_sha256,
        }
    if shared_w32:
        binding.update(
            {
                "gpu_resource_policy_id": "w32x32p256c512_gpuv2_shared",
                "physical_batch_set_sha256": canonical_sha256(
                    bound_request["physical_batch_set"]
                ),
            }
        )
    return _CudaFullEvolutionV2ShadowStateV1(
        binding=binding,
        projection=dict(projection),
        bound_execution_request=bound_request,
    )


def _bind_cuda_full_evolution_v2_attempt_authority_v1(
    *,
    request: ExecutionRequestV2,
    plan: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    preflight: Mapping[str, Any],
    runtime_root: Path,
    shadow_state: _CudaFullEvolutionV2ShadowStateV1,
    crca_adapter: CudaFullEvolutionV2LaunchpadCrcaAdapterV1,
) -> dict[str, Any]:
    """Ask the injected Scheduler/composition boundary for real Attempt authority."""

    from .cuda_full_evolution_v2_provider import ATTEMPT_AUTHORITY_NAME

    target = runtime_root / ATTEMPT_AUTHORITY_NAME
    binding = crca_adapter.bind_attempt_authority_v1(
        target=target,
        request=request.payload(),
        campaign_plan=plan,
        bound_execution_request=shadow_state.bound_execution_request,
        source_identity=source_identity,
        preflight=preflight,
        shadow_projection=shadow_state.projection,
    )
    if not isinstance(binding, Mapping):
        raise ExecutionFrontDoorError("CRCA adapter returned no Attempt decision")
    result = dict(binding)
    granted = result.get("execution_authority_granted")
    if type(granted) is not bool:
        raise ExecutionFrontDoorError(
            "CRCA Attempt decision lacks an exact execution-authority boolean"
        )
    if granted:
        if not target.is_file() or target.is_symlink():
            raise ExecutionFrontDoorError(
                "CRCA Attempt authority was granted without a sealed regular bundle"
            )
        declared_path = result.get("attempt_authority_path")
        if not isinstance(declared_path, str) or Path(declared_path).resolve() != (
            target.resolve()
        ):
            raise ExecutionFrontDoorError(
                "CRCA Attempt decision binds another authority bundle"
            )
    elif target.exists() or target.is_symlink():
        raise ExecutionFrontDoorError(
            "non-authoritative CRCA Attempt decision published an authority bundle: "
            + json.dumps(result.get("reason_codes"), sort_keys=True)
        )
    return result


def _reason_codes(preflight: Mapping[str, Any]) -> set[str]:
    codes: set[str] = set()
    for field in ("reasons", "execution_authority_reasons"):
        rows = preflight.get(field)
        if isinstance(rows, list):
            codes.update(
                str(row.get("code"))
                for row in rows
                if isinstance(row, Mapping) and row.get("code") is not None
            )
    return codes


def _preflight_wait_kind(preflight: Mapping[str, Any]) -> str | None:
    codes = _reason_codes(preflight)
    if not codes or not codes.issubset(
        RESOURCE_WAITABLE_CODES | STORAGE_WAITABLE_CODES
    ):
        return None
    if codes & STORAGE_WAITABLE_CODES:
        return "storage"
    return "resource"


def _wait_outcome(wait_kind: str) -> tuple[str, str, str]:
    if wait_kind == "storage":
        return "WAIT_STORAGE", "wait_storage", "STORAGE_WAIT"
    if wait_kind == "resource":
        return "WAIT_RESOURCE", "wait_resource", "RESOURCE_WAIT"
    raise ExecutionFrontDoorError(f"unsupported wait kind: {wait_kind!r}")


def _path_in_root(value: Any, *, run_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if path.is_symlink():
        return None
    path = path.resolve()
    try:
        path.relative_to(run_root.resolve())
    except ValueError:
        return None
    return path


def verify_mechanical_execution_v1(
    *,
    run_root: str | Path,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    execution_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify bindings and hashes while reporting semantic questions separately."""

    root = Path(run_root).expanduser().resolve()
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        )
        if not passed:
            errors.append(f"{name}: {detail}")

    check(
        "preflight_schema",
        preflight.get("schema_version") == "eiel_campaign_preflight_v1",
        "campaign preflight schema is present",
    )
    check(
        "preflight_plan_binding",
        preflight.get("plan_sha256") == plan.get("plan_sha256"),
        "preflight binds the bound plan SHA-256",
    )
    rows = preflight.get("artifacts")
    check(
        "preflight_execution_authority",
        preflight.get("ready_for_execution") is True,
        "preflight grants the requested execution authority",
    )
    artifacts_ok = isinstance(rows, list)
    if artifacts_ok:
        artifact_names = {
            str(row.get("name"))
            for row in rows
            if isinstance(row, Mapping) and row.get("name") is not None
        }
        for binary_name, manifest_name in (
            ("cuda_extension", "extension_manifest"),
            ("native_binary", "native_manifest"),
            ("binary", "manifest"),
        ):
            if (binary_name in artifact_names) != (manifest_name in artifact_names):
                artifacts_ok = False
                errors.append(
                    f"artifact_unit:{binary_name}: binary and required manifest must be resolved together"
                )
        for row in rows:
            if not isinstance(row, Mapping):
                artifacts_ok = False
                errors.append("artifact_row: artifact preflight row is malformed")
                continue
            path = Path(str(row.get("path", ""))).expanduser()
            regular = path.is_file() and not path.is_symlink()
            try:
                exact = regular and _sha256_file(path) == str(
                    row.get("expected_sha256", "")
                )
            except OSError:
                exact = False
            if not regular or not exact:
                artifacts_ok = False
                errors.append(
                    f"artifact:{row.get('name')}: frozen artifact is missing or changed"
                )
        check(
            "frozen_artifacts",
            artifacts_ok,
            "all preflight artifact hashes remain exact",
        )
    else:
        check("frozen_artifacts", False, "preflight artifact rows are malformed")

    lifecycle_value: Mapping[str, Any] | None = None
    lifecycle_path = root / "artifact_lifecycle_contract.json"
    if lifecycle_path.is_file() and not lifecycle_path.is_symlink():
        try:
            lifecycle_value = validate_run_artifact_lifecycle_contract_v1(
                _read_object(lifecycle_path, name="artifact lifecycle contract")
            )
            expected = ensure_lifecycle_contract_v1(plan)
            check(
                "lifecycle_binding",
                lifecycle_value.get("contract_sha256")
                == expected.get("contract_sha256"),
                "materialized lifecycle contract matches the bound plan",
            )
        except (ArtifactLifecycleError, ExecutionFrontDoorError) as exc:
            check("lifecycle_binding", False, str(exc))
    elif execution_receipt is not None:
        check("lifecycle_binding", False, "terminal execution lacks lifecycle contract")

    if execution_receipt is not None:
        check(
            "execution_receipt_schema",
            execution_receipt.get("schema_version")
            == "eiel_campaign_execution_receipt_v1",
            "execution receipt schema is present",
        )
        check(
            "execution_receipt_plan_binding",
            execution_receipt.get("plan_sha256") == plan.get("plan_sha256"),
            "execution receipt binds the bound plan SHA-256",
        )
        required = (
            "campaign_manifest_path",
            "run_passport_path",
            "artifact_disposition_manifest_path",
            "storage_disposition_receipt_path",
        )
        for field in required:
            path = _path_in_root(execution_receipt.get(field), run_root=root)
            present = path is not None and path.is_file() and not path.is_symlink()
            check(field, present, "terminal receipt path is a regular file in run root")
        disposition_path = _path_in_root(
            execution_receipt.get("artifact_disposition_manifest_path"), run_root=root
        )
        if disposition_path is not None and disposition_path.is_file():
            try:
                disposition = validate_artifact_disposition_manifest_v1(
                    _read_object(disposition_path, name="artifact disposition manifest")
                )
                check(
                    "disposition_binding",
                    disposition.get("plan_sha256") == plan.get("plan_sha256"),
                    "disposition manifest binds the bound plan SHA-256",
                )
            except (ArtifactLifecycleError, ExecutionFrontDoorError) as exc:
                check("disposition_binding", False, str(exc))
        storage_path = _path_in_root(
            execution_receipt.get("storage_disposition_receipt_path"), run_root=root
        )
        if storage_path is not None and storage_path.is_file():
            try:
                storage = _read_object(storage_path, name="storage disposition receipt")
                supplied = storage.get("receipt_sha256")
                core = dict(storage)
                core.pop("receipt_sha256", None)
                check(
                    "storage_receipt_hash",
                    isinstance(supplied, str) and supplied == canonical_sha256(core),
                    "storage disposition receipt hash is exact",
                )
                disposition_hash = (
                    None
                    if disposition_path is None or not disposition_path.is_file()
                    else _read_object(
                        disposition_path, name="artifact disposition manifest"
                    ).get("manifest_sha256")
                )
                check(
                    "storage_receipt_binding",
                    storage.get("manifest_sha256") == disposition_hash
                    and storage.get("plan_sha256") == plan.get("plan_sha256"),
                    "storage disposition receipt binds the disposition and plan",
                )
            except ExecutionFrontDoorError as exc:
                check("storage_receipt_hash", False, str(exc))
        passport_path = _path_in_root(
            execution_receipt.get("run_passport_path"), run_root=root
        )
        if passport_path is not None and passport_path.is_file():
            try:
                passport = _read_object(passport_path, name="run passport")
                supplied = passport.get("passport_sha256")
                core = dict(passport)
                core.pop("passport_sha256", None)
                check(
                    "passport_hash",
                    isinstance(supplied, str) and supplied == canonical_sha256(core),
                    "run passport hash is exact",
                )
                check(
                    "passport_plan_binding",
                    passport.get("plan_sha256") == plan.get("plan_sha256"),
                    "run passport binds the bound plan SHA-256",
                )
                plan_artifacts = plan.get("artifacts")
                expected_artifact_hashes = (
                    {
                        str(name): str(row.get("sha256"))
                        for name, row in plan_artifacts.items()
                        if isinstance(row, Mapping)
                        and isinstance(row.get("sha256"), str)
                    }
                    if isinstance(plan_artifacts, Mapping)
                    else {}
                )
                passport_source = passport.get("source_identity")
                observed_artifact_hashes = (
                    passport_source.get("artifact_hashes")
                    if isinstance(passport_source, Mapping)
                    else None
                )
                if expected_artifact_hashes:
                    check(
                        "passport_artifact_binding",
                        observed_artifact_hashes == expected_artifact_hashes,
                        "run passport binds every plan artifact hash",
                    )
            except ExecutionFrontDoorError as exc:
                check("passport_hash", False, str(exc))
    semantic_questions = [
        "Mechanical verification does not establish semantic parity, scientific validity, or performance claims.",
        "Qualification evidence is evaluated only by an explicit qualification profile.",
    ]
    return {
        "schema_version": MECHANICAL_VERIFICATION_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
        "semantic_questions": semantic_questions,
        "terminal_artifacts_checked": execution_receipt is not None,
        "run_root": str(root),
    }


def evaluate_qualification_profile_v1(
    profile: Mapping[str, Any] | None,
    *,
    execution_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Evaluate the first small adapter without turning it into a Q6 runner."""

    if profile is None:
        return None
    if profile.get("type") != "tensor_world_c7":
        return {
            "schema_version": QUALIFICATION_EVALUATION_SCHEMA_VERSION,
            "profile_type": profile.get("type"),
            "status": "INCOMPLETE",
            "decision": "UNSUPPORTED_PROFILE",
            "errors": ["qualification profile is unsupported"],
        }
    if execution_result.get("execution_status") != "completed":
        return {
            "schema_version": QUALIFICATION_EVALUATION_SCHEMA_VERSION,
            "profile_type": "tensor_world_c7",
            "status": "INCOMPLETE",
            "decision": "EXECUTION_NOT_COMPLETED",
            "errors": ["qualification evaluation requires a completed execution"],
        }
    try:
        from engine.tensor_world.c7_qualification import (
            TensorWorldC7EvidenceReceiptV1,
            TensorWorldC7QualificationFreezeV1,
            decide_tensor_world_c7_v1,
        )

        freeze_payload = _read_object(
            Path(str(profile["freeze"])), name="Tensor World C7 freeze"
        )
        evidence_payload = _read_object(
            Path(str(profile["evidence"])), name="Tensor World C7 evidence"
        )
        freeze = TensorWorldC7QualificationFreezeV1(
            **{
                key: value
                for key, value in freeze_payload.items()
                if key != "schema_version"
            },
            schema_version=freeze_payload.get("schema_version", ""),
        )
        evidence = TensorWorldC7EvidenceReceiptV1(
            **{
                key: value
                for key, value in evidence_payload.items()
                if key != "schema_version"
            },
            schema_version=evidence_payload.get("schema_version", ""),
        )
        decision = decide_tensor_world_c7_v1(freeze, evidence)
        decision_text = decision.value
        status = (
            "PASS"
            if decision_text == "TW-C7-QUALIFIED-FOR-ADMISSION-REVIEW"
            else (
                "INCOMPLETE"
                if decision_text == "TW-C7-QUALIFICATION-INVALID"
                else "FAIL"
            )
        )
        return {
            "schema_version": QUALIFICATION_EVALUATION_SCHEMA_VERSION,
            "profile_type": "tensor_world_c7",
            "status": status,
            "decision": decision_text,
            "freeze_fingerprint": freeze.fingerprint(),
            "semantic_questions": [
                "C7 PASS means qualified for admission review only; it is not a formal Q6 qualification claim."
            ],
            "errors": [],
        }
    except (OSError, TypeError, ValueError, KeyError, ExecutionFrontDoorError) as exc:
        return {
            "schema_version": QUALIFICATION_EVALUATION_SCHEMA_VERSION,
            "profile_type": "tensor_world_c7",
            "status": "INCOMPLETE",
            "decision": "QUALIFICATION_INPUT_INVALID",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


def _result_with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("result_sha256", None)
    return {**core, "result_sha256": canonical_sha256(core)}


def _write_result(run_root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    result = _result_with_hash(value)
    atomic_write_json(run_root / "execution_result.json", result)
    return result


def _existing_result(
    run_root: Path,
    *,
    expected_schema_version: str,
    request_sha256: str,
    plan_sha256: str,
    source_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = run_root / "execution_result.json"
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() or path.is_symlink():
        raise ExecutionFrontDoorError("execution result is not a regular bound file")
    result = _read_object(path, name="execution result")
    supplied = result.get("result_sha256")
    core = dict(result)
    core.pop("result_sha256", None)
    if supplied != canonical_sha256(core):
        raise ExecutionFrontDoorError("execution result SHA-256 mismatch")
    if result.get("schema_version") != expected_schema_version:
        raise ExecutionFrontDoorError(
            "execution result schema does not match its bound request"
        )
    if result.get("run_root") != str(run_root):
        raise ExecutionFrontDoorError("execution result run-root binding mismatch")
    if result.get("request_sha256") != request_sha256:
        raise ExecutionFrontDoorError("run root is bound to another execution request")
    if result.get("plan_sha256") != plan_sha256:
        raise ExecutionFrontDoorError("run root is bound to another compiled plan")
    if result.get("source_identity") != dict(source_identity):
        raise ExecutionFrontDoorError("run root is bound to another source identity")
    return result


def _workhorse_terminal_lifecycle_artifacts_v1(
    run_root: Path,
    result: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    finalized = result.get("workhorse_terminalized_operation")
    if finalized is None:
        if result.get("result_class") == "VALID_NEGATIVE":
            raise ExecutionFrontDoorError(
                "valid-negative result lacks its terminal semantic receipt"
            )
        return {}
    if not isinstance(finalized, Mapping):
        raise ExecutionFrontDoorError(
            "Workhorse terminalized operation binding is malformed"
        )
    raw_path = finalized.get("final_receipt_path") or finalized.get(
        "row_receipt_path"
    )
    if not isinstance(raw_path, str) or not raw_path:
        raise ExecutionFrontDoorError(
            "Workhorse terminalized operation lacks its final receipt path"
        )
    receipt_path = Path(raw_path).expanduser().resolve()
    try:
        relative = str(receipt_path.relative_to(run_root)).replace("\\", "/")
    except ValueError as exc:
        raise ExecutionFrontDoorError(
            "Workhorse terminalized receipt left its run root"
        ) from exc
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ExecutionFrontDoorError(
            "Workhorse terminalized receipt is not a regular bound file"
        )
    declared_sha256 = finalized.get("final_receipt_file_sha256") or finalized.get(
        "row_receipt_file_sha256"
    )
    if declared_sha256 != _sha256_file(receipt_path):
        raise ExecutionFrontDoorError(
            "Workhorse terminalized receipt file SHA-256 mismatch"
        )
    operation = finalized.get("operation")
    if result.get("result_class") == "VALID_NEGATIVE" and operation != (
        "SEMANTIC_EVIDENCE_CASE"
    ):
        raise ExecutionFrontDoorError(
            "valid-negative result is not one closed semantic case"
        )
    if operation == "SEMANTIC_EVIDENCE_CASE":
        try:
            from .cuda_full_evolution_v2_workhorse_provider import (
                validate_workhorse_terminalized_semantic_case_receipt_v2,
            )

            semantic_receipt = (
                validate_workhorse_terminalized_semantic_case_receipt_v2(
                    receipt_path
                )
            )
        except Exception as exc:
            raise ExecutionFrontDoorError(
                f"terminal semantic receipt cannot reopen: {exc}"
            ) from exc
        crca_terminal = result.get("crca_attempt_terminal")
        terminal_binding = (
            crca_terminal.get("workhorse_terminal_binding")
            if isinstance(crca_terminal, Mapping)
            else None
        )
        if (
            result.get("workhorse_terminalized_semantic_case") != finalized
            or finalized.get("result_class") != result.get("result_class")
            or finalized.get("semantic_case_receipt_path")
            != str(receipt_path)
            or finalized.get("semantic_case_receipt_file_sha256")
            != semantic_receipt.get("receipt_file_sha256")
            or finalized.get("semantic_case_receipt_content_sha256")
            != semantic_receipt.get("receipt_content_sha256")
            or finalized.get("final_receipt_content_sha256")
            != semantic_receipt.get("receipt_content_sha256")
            or semantic_receipt.get("result_class") != result.get("result_class")
            or not isinstance(terminal_binding, Mapping)
            or semantic_receipt.get("canonical_terminal_binding")
            != terminal_binding
            or (
                result.get("result_class") == "VALID_NEGATIVE"
                and terminal_binding.get("attempt_terminal_status") != "FAILED"
            )
            or (
                result.get("result_class") == "SUCCESS"
                and terminal_binding.get("attempt_terminal_status")
                != "COMPLETED"
            )
        ):
            raise ExecutionFrontDoorError(
                "terminal semantic receipt/result/Scheduler binding changed"
            )
    elif operation == "SCIENCE_COMPATIBILITY_DEMO":
        crca_terminal = result.get("crca_attempt_terminal")
        terminal_binding = (
            crca_terminal.get("workhorse_terminal_binding")
            if isinstance(crca_terminal, Mapping)
            else None
        )
        if (
            result.get("workhorse_terminalized_science_demo") != finalized
            or result.get("result_class") != "SUCCESS"
            or finalized.get("result_class") != "SUCCESS"
            or finalized.get("demo_result_class")
            != "TECHNICAL_COMPATIBILITY_DEMO_PASSED"
            or finalized.get("science_demo_receipt_path") != str(receipt_path)
            or finalized.get("science_demo_receipt_file_sha256")
            != declared_sha256
            or finalized.get("science_demo_receipt_content_sha256")
            != finalized.get("final_receipt_content_sha256")
            or not isinstance(terminal_binding, Mapping)
            or terminal_binding.get("attempt_terminal_status") != "COMPLETED"
        ):
            raise ExecutionFrontDoorError(
                "terminal science-demo receipt/result/Scheduler binding changed"
            )
    elif operation == "INDEXED_ACS_CONTINUATION":
        try:
            from .cuda_full_evolution_v2_workhorse_provider import (
                validate_workhorse_terminalized_indexed_continuation_receipt_v2,
            )

            indexed_receipt = (
                validate_workhorse_terminalized_indexed_continuation_receipt_v2(
                    receipt_path
                )
            )
        except Exception as exc:
            raise ExecutionFrontDoorError(
                f"terminal indexed continuation receipt cannot reopen: {exc}"
            ) from exc
        crca_terminal = result.get("crca_attempt_terminal")
        terminal_binding = (
            crca_terminal.get("workhorse_terminal_binding")
            if isinstance(crca_terminal, Mapping)
            else None
        )
        if (
            result.get("workhorse_terminalized_indexed_continuation")
            != finalized
            or finalized.get("indexed_continuation_receipt_path")
            != str(receipt_path)
            or finalized.get("final_receipt_file_sha256")
            != indexed_receipt.get("receipt_file_sha256")
            or finalized.get("final_receipt_content_sha256")
            != indexed_receipt.get("receipt_content_sha256")
            or not isinstance(terminal_binding, Mapping)
            or indexed_receipt.get("canonical_terminal_binding")
            != terminal_binding
            or terminal_binding.get("attempt_terminal_status") != "COMPLETED"
        ):
            raise ExecutionFrontDoorError(
                "terminal indexed continuation receipt/result/Scheduler binding changed"
            )
    lifecycle = finalized.get("lifecycle_binding")
    if lifecycle is None:
        lifecycle = {
            "artifact_family": "qualification.workhorse_technical_row_v2",
            "retention_class": "R3",
            "owner_writer": (
                "cuda_full_evolution_workhorse_v2_single_row_owner"
            ),
            "release_predicate": (
                "technical row retained for offline matrix aggregation"
            ),
        }
    expected_fields = {
        "artifact_family",
        "retention_class",
        "owner_writer",
        "release_predicate",
    }
    if (
        not isinstance(lifecycle, Mapping)
        or set(lifecycle) != expected_fields
        or lifecycle.get("retention_class") != "R3"
        or any(
            not isinstance(lifecycle.get(field), str) or not lifecycle[field]
            for field in expected_fields
        )
    ):
        raise ExecutionFrontDoorError(
            "Workhorse final receipt lifecycle binding is invalid"
        )
    return {
        relative: {field: str(lifecycle[field]) for field in expected_fields}
    }


def _validate_terminal_lifecycle_material_v1(
    *,
    run_root: Path,
    plan_sha256: str,
    result: Mapping[str, Any],
    additional_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    result_path = run_root / "execution_result.json"
    manifest_path = run_root / "artifact_disposition_manifest.json"
    storage_path = run_root / "storage_disposition_receipt.json"
    for path, label in (
        (result_path, "execution result"),
        (manifest_path, "artifact disposition manifest"),
        (storage_path, "storage disposition receipt"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ExecutionFrontDoorError(
                f"terminal lifecycle {label} is not a regular file"
            )
    current_result = _read_object(result_path, name="terminal execution result")
    result_core = dict(current_result)
    supplied_result_sha256 = result_core.pop("result_sha256", None)
    if (
        supplied_result_sha256 != canonical_sha256(result_core)
        or current_result != dict(result)
        or current_result.get("result_class") not in {"SUCCESS", "VALID_NEGATIVE"}
        or current_result.get("status") != "completed"
        or current_result.get("ok") is not True
    ):
        raise ExecutionFrontDoorError(
            "terminal lifecycle result is not an exact accepted terminal result"
        )
    try:
        manifest = validate_artifact_disposition_manifest_v1(
            _read_object(manifest_path, name="terminal artifact disposition manifest")
        )
    except ArtifactLifecycleError as exc:
        raise ExecutionFrontDoorError(
            f"terminal lifecycle manifest is invalid: {exc}"
        ) from exc
    inventory = manifest.get("inventory")
    if (
        manifest.get("runtime_root") != str(run_root)
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("terminal_status") != "completed"
        or not isinstance(inventory, Mapping)
        or inventory.get("complete") is not True
    ):
        raise ExecutionFrontDoorError(
            "terminal lifecycle manifest is not the exact completed run inventory"
        )
    storage = _read_object(storage_path, name="terminal storage disposition receipt")
    storage_core = dict(storage)
    supplied_storage_sha256 = storage_core.pop("receipt_sha256", None)
    if (
        storage.get("schema_version") != "eiel_storage_disposition_receipt_v1"
        or supplied_storage_sha256 != canonical_sha256(storage_core)
        or storage.get("manifest_sha256") != manifest.get("manifest_sha256")
        or storage.get("plan_sha256") != plan_sha256
        or storage.get("campaign_id") != manifest.get("campaign_id")
        or storage.get("run_id") != manifest.get("run_id")
        or storage.get("lifecycle_contract_sha256")
        != manifest.get("lifecycle_contract_sha256")
    ):
        raise ExecutionFrontDoorError(
            "terminal storage receipt is not source-bound to the current manifest"
        )
    artifact_rows: dict[str, Mapping[str, Any]] = {}
    for raw_row in manifest.get("artifacts", []):
        if not isinstance(raw_row, Mapping):
            raise ExecutionFrontDoorError(
                "terminal lifecycle inventory contains a malformed artifact row"
            )
        relative = raw_row.get("path")
        if not isinstance(relative, str) or relative in artifact_rows:
            raise ExecutionFrontDoorError(
                "terminal lifecycle inventory paths are malformed or duplicated"
            )
        artifact_rows[relative] = raw_row

    required = {
        "execution_result.json": {
            "artifact_family": "lifecycle.control",
            "retention_class": "R5",
            "owner_writer": "execution-front-door",
        },
        **{
            relative: {
                "artifact_family": binding["artifact_family"],
                "retention_class": binding["retention_class"],
                "owner_writer": binding["owner_writer"],
                "release_predicate": binding["release_predicate"],
            }
            for relative, binding in additional_artifacts.items()
        },
    }
    terminal_artifacts: list[dict[str, Any]] = []
    for relative in sorted(required):
        path = run_root / relative
        if path.is_symlink() or not path.is_file():
            raise ExecutionFrontDoorError(
                f"terminal lifecycle artifact is not a regular file: {relative}"
            )
        row = artifact_rows.get(relative)
        expected = required[relative]
        digest = _sha256_file(path)
        if (
            row is None
            or row.get("sha256") != digest
            or row.get("bytes") != path.stat().st_size
            or any(row.get(field) != value for field, value in expected.items())
        ):
            raise ExecutionFrontDoorError(
                f"terminal lifecycle artifact binding changed: {relative}"
            )
        terminal_artifacts.append(
            {
                "path": relative,
                "file_sha256": digest,
                **expected,
            }
        )
    return {
        "result": {
            "path": "execution_result.json",
            "file_sha256": _sha256_file(result_path),
            "result_sha256": supplied_result_sha256,
        },
        "manifest": {
            "path": "artifact_disposition_manifest.json",
            "file_sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "storage_receipt": {
            "path": "storage_disposition_receipt.json",
            "file_sha256": _sha256_file(storage_path),
            "receipt_sha256": supplied_storage_sha256,
        },
        "terminal_artifacts": terminal_artifacts,
    }


def _validate_terminal_lifecycle_commit_v1(
    *,
    run_root: Path,
    request_sha256: str,
    plan_sha256: str,
    source_identity: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    marker_path = run_root / TERMINAL_LIFECYCLE_COMMIT_NAME
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ExecutionFrontDoorError(
            "accepted result is not published: terminal lifecycle commit marker is absent"
        )
    marker = _read_object(marker_path, name="terminal lifecycle commit marker")
    expected_fields = {
        "schema_version",
        "request_sha256",
        "plan_sha256",
        "source_identity_sha256",
        "result",
        "manifest",
        "storage_receipt",
        "terminal_artifacts",
        "committed_at_utc",
        "commit_sha256",
    }
    core = dict(marker)
    supplied = core.pop("commit_sha256", None)
    if (
        set(marker) != expected_fields
        or marker.get("schema_version") != TERMINAL_LIFECYCLE_COMMIT_SCHEMA_VERSION
        or marker.get("request_sha256") != request_sha256
        or marker.get("plan_sha256") != plan_sha256
        or marker.get("source_identity_sha256")
        != canonical_sha256(source_identity)
        or not isinstance(marker.get("committed_at_utc"), str)
        or not marker["committed_at_utc"]
        or supplied != canonical_sha256(core)
    ):
        raise ExecutionFrontDoorError(
            "terminal lifecycle commit marker identity or seal changed"
        )
    additional = _workhorse_terminal_lifecycle_artifacts_v1(run_root, result)
    material = _validate_terminal_lifecycle_material_v1(
        run_root=run_root,
        plan_sha256=plan_sha256,
        result=result,
        additional_artifacts=additional,
    )
    if any(marker.get(field) != material[field] for field in material):
        raise ExecutionFrontDoorError(
            "terminal lifecycle commit marker no longer matches current material"
        )
    return marker


def _publish_terminal_lifecycle_commit_v1(
    *,
    run_root: Path,
    request_sha256: str,
    plan_sha256: str,
    source_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    additional_artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    derived = _workhorse_terminal_lifecycle_artifacts_v1(run_root, result)
    normalized = {
        str(path): {str(key): str(value) for key, value in binding.items()}
        for path, binding in additional_artifacts.items()
    }
    if normalized != derived:
        raise ExecutionFrontDoorError(
            "terminal lifecycle reseal inputs differ from the finalized Workhorse receipt"
        )
    material = _validate_terminal_lifecycle_material_v1(
        run_root=run_root,
        plan_sha256=plan_sha256,
        result=result,
        additional_artifacts=normalized,
    )
    core = {
        "schema_version": TERMINAL_LIFECYCLE_COMMIT_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "plan_sha256": plan_sha256,
        "source_identity_sha256": canonical_sha256(source_identity),
        **material,
        "committed_at_utc": _now(),
    }
    marker = {**core, "commit_sha256": canonical_sha256(core)}
    atomic_write_json(run_root / TERMINAL_LIFECYCLE_COMMIT_NAME, marker)
    return _validate_terminal_lifecycle_commit_v1(
        run_root=run_root,
        request_sha256=request_sha256,
        plan_sha256=plan_sha256,
        source_identity=source_identity,
        result=result,
    )


def _reseal_and_publish_terminal_lifecycle_v1(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    request_sha256: str,
    source_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    additional_artifacts: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    derived = _workhorse_terminal_lifecycle_artifacts_v1(run_root, result)
    normalized = derived if additional_artifacts is None else {
        str(path): {str(key): str(value) for key, value in binding.items()}
        for path, binding in additional_artifacts.items()
    }
    if normalized != derived:
        raise ExecutionFrontDoorError(
            "terminal lifecycle recovery changed its finalized artifact set"
        )
    kwargs: dict[str, Any] = {}
    if normalized:
        kwargs["additional_artifacts"] = normalized
    resealed = reseal_terminal_lifecycle_inventory_v1(
        plan,
        run_root=run_root,
        **kwargs,
    )
    if (
        not isinstance(resealed, Mapping)
        or not isinstance(resealed.get("manifest"), Mapping)
        or not isinstance(resealed.get("storage_receipt"), Mapping)
    ):
        raise ExecutionFrontDoorError(
            "terminal lifecycle reseal returned no verifiable manifest/receipt"
        )
    current_manifest = _read_object(
        run_root / "artifact_disposition_manifest.json",
        name="resealed artifact disposition manifest",
    )
    current_storage = _read_object(
        run_root / "storage_disposition_receipt.json",
        name="resealed storage disposition receipt",
    )
    if (
        dict(resealed["manifest"]) != current_manifest
        or dict(resealed["storage_receipt"]) != current_storage
    ):
        raise ExecutionFrontDoorError(
            "terminal lifecycle reseal response differs from durable material"
        )
    return _publish_terminal_lifecycle_commit_v1(
        run_root=run_root,
        request_sha256=request_sha256,
        plan_sha256=str(plan["plan_sha256"]),
        source_identity=source_identity,
        result=result,
        additional_artifacts=normalized,
    )


def _validate_existing_baseline_result_binding_v1(
    *,
    request: ExecutionRequestV2,
    runtime_root: Path,
    plan: Mapping[str, Any],
    existing: Mapping[str, Any],
    stored_preflight: Mapping[str, Any],
    artifact_lifecycle_explicit: bool,
) -> dict[str, Any]:
    admission_path = runtime_root / "baseline_admission_receipt.json"
    if not admission_path.is_file() or admission_path.is_symlink():
        raise ExecutionFrontDoorError(
            "existing V2 result lacks its Baseline admission receipt"
        )
    admission = validate_baseline_receipt_binding_v1(
        _read_object(admission_path, name="Baseline admission receipt"),
        schema_version=BASELINE_ADMISSION_SCHEMA_VERSION,
        declaration=request.baseline,
        plan=plan,
        request_sha256=request.request_sha256,
        front_door_id=EXECUTION_REQUEST_V2_SCHEMA_VERSION,
        preflight=stored_preflight,
        stage="admission",
        artifact_lifecycle_explicit=artifact_lifecycle_explicit,
        canopy_enabled=True,
    )
    conformance: Mapping[str, Any] | None = None
    if existing.get("execution_status") is not None:
        conformance_path = runtime_root / "baseline_conformance_receipt.json"
        if not conformance_path.is_file() or conformance_path.is_symlink():
            raise ExecutionFrontDoorError(
                "existing V2 result lacks its Baseline conformance receipt"
            )
        canopy_path = runtime_root / "canopy_relay_supervision.json"
        if not canopy_path.is_file() or canopy_path.is_symlink():
            raise ExecutionFrontDoorError(
                "existing V2 result lacks its Canopy supervision receipt"
            )
        canopy = _read_object(canopy_path, name="Canopy relay supervision receipt")
        conformance = validate_baseline_receipt_binding_v1(
            _read_object(conformance_path, name="Baseline conformance receipt"),
            schema_version=BASELINE_CONFORMANCE_SCHEMA_VERSION,
            declaration=request.baseline,
            plan=plan,
            request_sha256=request.request_sha256,
            front_door_id=EXECUTION_REQUEST_V2_SCHEMA_VERSION,
            preflight=stored_preflight,
            stage="terminal",
            artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            canopy_enabled=True,
            canopy_supervision=canopy,
            admission_receipt_sha256=str(admission["receipt_sha256"]),
            execution_status=str(existing.get("execution_status")),
        )
    binding = baseline_result_binding_v1(
        admission_receipt=admission,
        conformance_receipt=conformance,
    )
    if existing.get("baseline") != binding:
        raise ExecutionFrontDoorError(
            "execution result Baseline receipt binding mismatch"
        )
    return binding


def _base_result(
    *,
    request: ExecutionRequest,
    source_identity: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_root: Path,
    preflight: Mapping[str, Any] | None,
    preflight_path: Path,
    invocation_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": (
            EXECUTION_RESULT_V2_SCHEMA_VERSION
            if isinstance(request, ExecutionRequestV2)
            else EXECUTION_RESULT_SCHEMA_VERSION
        ),
        "request_sha256": request.request_sha256,
        "invocation_mode": invocation_mode,
        "purpose": request.purpose,
        "campaign_id": plan.get("campaign_id"),
        "compiled_plan_sha256": source_identity.get("compiled_plan_sha256"),
        "plan_sha256": plan.get("plan_sha256"),
        "profile_id": plan.get("resolved_profile"),
        "source_identity": dict(source_identity),
        "run_root": str(run_root),
        "preflight_receipt_path": str(preflight_path),
        "preflight_receipt_sha256": (
            canonical_sha256(preflight) if preflight is not None else None
        ),
        "preflight": None if preflight is None else dict(preflight),
        "execution_status": None,
        "status": "failed",
        "result_class": "PRECHECK_FAILED",
        "execution_receipt": None,
        "mechanical_verification": None,
        "qualification_status": "NOT_REQUESTED",
        "qualification_evaluation": None,
        "resource_wait": request.resource_wait.payload(),
        "warnings": [],
        "baseline": (
            None
            if isinstance(request, ExecutionRequestV2)
            else {"overall_status": "LEGACY_UNPROFILED"}
        ),
        "created_at_utc": _now(),
    }


def _v1_continuation_authorized_v1(
    run_root: Path,
    *,
    request_sha256: str,
    plan_sha256: str,
    source_identity: Mapping[str, Any],
) -> bool:
    result_path = run_root / "execution_result.json"
    if result_path.exists() or result_path.is_symlink():
        if not result_path.is_file() or result_path.is_symlink():
            return False
        result = _read_object(result_path, name="V1 continuation result")
        supplied = result.get("result_sha256")
        core = dict(result)
        core.pop("result_sha256", None)
        if supplied != canonical_sha256(core):
            raise ExecutionFrontDoorError("V1 continuation result SHA-256 mismatch")
        return bool(
            result.get("invocation_mode") == "execute"
            and (result.get("result_class"), result.get("status"))
            in {
                ("RESOURCE_WAIT", "wait_resource"),
                ("STORAGE_WAIT", "wait_storage"),
            }
            and result.get("ok") is False
            and result.get("request_sha256") == request_sha256
            and result.get("plan_sha256") == plan_sha256
            and result.get("source_identity") == dict(source_identity)
        )
    state_path = run_root / "execution_state.json"
    if state_path.is_file() and not state_path.is_symlink():
        state = _read_object(state_path, name="V1 continuation state")
        if (
            state.get("schema_version") == "eiel_execution_state_v1"
            and state.get("invocation_mode") == "execute"
            and state.get("request_sha256") == request_sha256
            and state.get("plan_sha256") == plan_sha256
            and state.get("source_identity_sha256") == canonical_sha256(source_identity)
            and state.get("status")
            in {"WAIT_RESOURCE", "WAIT_STORAGE", "RESUMING_DELIBERATE_RESTORE"}
        ):
            return True
    return False


def _execute_request_common_v1(
    request: ExecutionRequest,
    *,
    repo_root: str | Path | None = None,
    mode: str = "execute",
    provider_registry: ExecutionProviderRegistryV1 | None = None,
    preflight_fn: Callable[..., Mapping[str, Any]] = preflight_campaign,
    execute_fn: Callable[..., Mapping[str, Any]] = execute_campaign,
    mechanical_verifier_fn: Callable[
        ..., Mapping[str, Any]
    ] = verify_mechanical_execution_v1,
    canopy_start_fn: Callable[..., Mapping[str, Any]] = start_canopy_relays_v1,
    cuda_full_evolution_v2_crca_adapter: (
        CudaFullEvolutionV2LaunchpadCrcaAdapterV1 | None
    ) = None,
    cuda_full_evolution_v2_checkpoint_occurrence_adapter: Any | None = None,
    cuda_memory_release_observer_fn: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _observe_terminal_cuda_memory_release_v1,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    bound_plan: Mapping[str, Any] | None = None,
    bound_source_identity: Mapping[str, Any] | None = None,
    bound_artifact_lifecycle_explicit: bool | None = None,
    nested_storage_parent_run_root: str | Path | None = None,
    before_execute_fn: Callable[[], None] | None = None,
    before_lifecycle_closeout_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute or preflight one request through the canonical campaign path."""

    if mode not in {"execute", "validate"}:
        raise ExecutionFrontDoorError("mode must be 'execute' or 'validate'")
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        raise ExecutionFrontDoorError(f"repo_root is not a directory: {root}")
    if bound_plan is None:
        plan, source_identity, runtime_root_text, artifact_lifecycle_explicit = (
            _resolve_compiled_plan(request, repo_root=root)
        )
    else:
        if bound_source_identity is None:
            raise ExecutionFrontDoorError(
                "bound campaign plan requires an explicit source identity"
            )
        plan = _validate_bound_campaign_plan_v1(bound_plan)
        source_identity = dict(bound_source_identity)
        configured_root = request.output_root
        if configured_root is None:
            output = plan.get("output")
            configured_root = (
                str(output.get("runtime_root"))
                if isinstance(output, Mapping) and output.get("runtime_root")
                else None
            )
        if not configured_root:
            raise ExecutionFrontDoorError(
                "execution request and bound campaign plan both lack output_root"
            )
        configured_path = Path(configured_root).expanduser()
        if configured_path.is_symlink():
            raise ExecutionFrontDoorError("output_root may not be a symlink")
        runtime_root = configured_path
        if not runtime_root.is_absolute():
            runtime_root = root / runtime_root
        runtime_root = runtime_root.resolve()
        if runtime_root == root:
            raise ExecutionFrontDoorError("output_root may not be the repository root")
        if runtime_root.exists() and runtime_root.is_file():
            raise ExecutionFrontDoorError("output_root is an existing regular file")
        compiled_plan_sha256 = str(plan["plan_sha256"])
        plan = _bind_runtime_root(plan, runtime_root)
        if source_identity.get("compiled_plan_sha256") != compiled_plan_sha256:
            raise ExecutionFrontDoorError(
                "bound source identity does not name the supplied compiled plan"
            )
        if source_identity.get("bound_plan_sha256") != plan["plan_sha256"]:
            raise ExecutionFrontDoorError(
                "bound source identity does not name the supplied runtime-bound plan"
            )
        runtime_root_text = str(runtime_root)
        artifact_lifecycle_explicit = (
            isinstance(plan.get("artifact_lifecycle"), Mapping)
            if bound_artifact_lifecycle_explicit is None
            else bound_artifact_lifecycle_explicit
        )
        if type(artifact_lifecycle_explicit) is not bool:
            raise ExecutionFrontDoorError(
                "bound artifact_lifecycle explicitness must be boolean"
            )
    runtime_root = Path(runtime_root_text)
    request_receipt_path = runtime_root / "execution_request.json"
    if (
        isinstance(request, ExecutionRequestV1)
        and mode == "execute"
        and (
            not request_receipt_path.is_file()
            or request_receipt_path.is_symlink()
            or not _v1_continuation_authorized_v1(
                runtime_root,
                request_sha256=request.request_sha256,
                plan_sha256=str(plan["plan_sha256"]),
                source_identity=source_identity,
            )
        )
    ):
        raise ExecutionFrontDoorError(
            "EXECUTION_REQUEST_V2_REQUIRED: fresh execution requires Baseline; "
            "V1 is limited to exact bound nonterminal continuation"
        )
    runtime_root.mkdir(parents=True, exist_ok=True)
    request_receipt = {
        "schema_version": (
            EXECUTION_REQUEST_V2_RECEIPT_SCHEMA_VERSION
            if isinstance(request, ExecutionRequestV2)
            else EXECUTION_REQUEST_RECEIPT_SCHEMA_VERSION
        ),
        "request": request.payload(),
        "request_sha256": request.request_sha256,
        "source_identity": source_identity,
        "bound_plan_sha256": plan["plan_sha256"],
        "run_root": str(runtime_root),
        "created_at_utc": _now(),
    }
    result_path = runtime_root / "execution_result.json"
    if request_receipt_path.exists() or request_receipt_path.is_symlink():
        if not request_receipt_path.is_file() or request_receipt_path.is_symlink():
            raise ExecutionFrontDoorError(
                "execution request receipt is not a regular bound file"
            )
        previous = _read_object(request_receipt_path, name="execution request receipt")
        if previous.get("request_sha256") != request.request_sha256:
            raise ExecutionFrontDoorError(
                "run root is bound to another request receipt"
            )
        if previous.get("bound_plan_sha256") != plan["plan_sha256"]:
            raise ExecutionFrontDoorError("run root is bound to another plan receipt")
        if previous.get("source_identity") != source_identity:
            raise ExecutionFrontDoorError(
                "run root is bound to another source identity"
            )
    else:
        if result_path.is_file():
            raise ExecutionFrontDoorError(
                "run root has an execution result but no request receipt"
            )
        if any(runtime_root.iterdir()):
            raise ExecutionFrontDoorError(
                "output_root contains files but no canonical request receipt"
            )
        atomic_write_json(request_receipt_path, request_receipt)
    plan_path = runtime_root / "execution_plan.json"
    if plan_path.exists() or plan_path.is_symlink():
        if not plan_path.is_file() or plan_path.is_symlink():
            raise ExecutionFrontDoorError("bound execution plan is not a regular file")
        persisted_plan = _read_object(plan_path, name="bound execution plan")
        if canonical_sha256(persisted_plan) != canonical_sha256(plan):
            raise ExecutionFrontDoorError("run root is bound to another execution plan")
    else:
        atomic_write_json(plan_path, plan)
    existing = _existing_result(
        runtime_root,
        expected_schema_version=(
            EXECUTION_RESULT_V2_SCHEMA_VERSION
            if isinstance(request, ExecutionRequestV2)
            else EXECUTION_RESULT_SCHEMA_VERSION
        ),
        request_sha256=request.request_sha256,
        plan_sha256=str(plan["plan_sha256"]),
        source_identity=source_identity,
    )
    existing_restore_boundary = (
        existing.get("restore_process_boundary")
        if isinstance(existing, Mapping)
        and request.restore_process_isolation == "fresh_invocation_per_restore"
        else None
    )
    if existing_restore_boundary is not None and mode == "execute":
        boundary = existing_restore_boundary
        current_process = _validated_restore_process_identity(
            _process_identity(), name="current restore process"
        )
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("schema_version")
            != "eiel_execution_restore_process_boundary_v1"
            or not isinstance(boundary.get("producer_process"), Mapping)
        ):
            raise ExecutionFrontDoorError(
                "restore-pending result lacks its process boundary"
            )
        producer_process = boundary["producer_process"]
        if producer_process.get("process_instance_id") == current_process.get(
            "process_instance_id"
        ):
            raise ExecutionFrontDoorError(
                "restore_process_isolation requires the pending checkpoint to be "
                "resumed by a fresh OS process"
            )
    if (
        existing is not None
        and mode == "execute"
        and existing.get("result_class")
        not in {
            "RESOURCE_WAIT",
            "STORAGE_WAIT",
            "PRECHECK_FAILED",
            "PREFLIGHT_ERROR",
            "PREFLIGHT_PASS",
            "BASELINE_ADMISSION_FAILED",
            "ATTEMPT_ADMISSION_UNAVAILABLE",
            "RESTORE_PENDING",
        }
    ):
        if isinstance(request, ExecutionRequestV2):
            stored_preflight_path = runtime_root / "preflight_receipt.json"
            if (
                not stored_preflight_path.is_file()
                or stored_preflight_path.is_symlink()
            ):
                raise ExecutionFrontDoorError(
                    "existing result lacks a regular preflight receipt"
                )
            stored_preflight = _read_object(
                stored_preflight_path, name="stored preflight receipt"
            )
            _validate_existing_baseline_result_binding_v1(
                request=request,
                runtime_root=runtime_root,
                plan=plan,
                existing=existing,
                stored_preflight=stored_preflight,
                artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            )
        if (
            artifact_lifecycle_explicit
            and existing.get("result_class") in {"SUCCESS", "VALID_NEGATIVE"}
        ):
            try:
                _validate_terminal_lifecycle_commit_v1(
                    run_root=runtime_root,
                    request_sha256=request.request_sha256,
                    plan_sha256=str(plan["plan_sha256"]),
                    source_identity=source_identity,
                    result=existing,
                )
            except ExecutionFrontDoorError:
                _reseal_and_publish_terminal_lifecycle_v1(
                    plan=plan,
                    run_root=runtime_root,
                    request_sha256=request.request_sha256,
                    source_identity=source_identity,
                    result=existing,
                )
        return existing
    if existing is not None and mode == "validate":
        stored_preflight_path = runtime_root / "preflight_receipt.json"
        if not stored_preflight_path.is_file() or stored_preflight_path.is_symlink():
            raise ExecutionFrontDoorError(
                "existing result lacks a regular preflight receipt"
            )
        stored_preflight = _read_object(
            stored_preflight_path, name="stored preflight receipt"
        )
        if (
            artifact_lifecycle_explicit
            and existing.get("result_class") in {"SUCCESS", "VALID_NEGATIVE"}
        ):
            _validate_terminal_lifecycle_commit_v1(
                run_root=runtime_root,
                request_sha256=request.request_sha256,
                plan_sha256=str(plan["plan_sha256"]),
                source_identity=source_identity,
                result=existing,
            )
        execution_receipt = existing.get("execution_receipt")
        verification = mechanical_verifier_fn(
            run_root=runtime_root,
            plan=plan,
            preflight=stored_preflight,
            execution_receipt=(
                execution_receipt if isinstance(execution_receipt, Mapping) else None
            ),
        )
        validated = dict(existing)
        validated["mechanical_verification"] = verification
        validated["validation_only"] = True
        validated["ok"] = (
            bool(existing.get("ok")) and verification.get("status") == "PASS"
        )
        if verification.get("status") != "PASS":
            validated["result_class"] = "MECHANICAL_VERIFICATION_FAILED"
        if isinstance(request, ExecutionRequestV2):
            validated["baseline"] = _validate_existing_baseline_result_binding_v1(
                request=request,
                runtime_root=runtime_root,
                plan=plan,
                existing=existing,
                stored_preflight=stored_preflight,
                artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            )
        else:
            validated["baseline"] = {"overall_status": "LEGACY_UNPROFILED"}
        return _result_with_hash(validated)
    preflight_path = runtime_root / "preflight_receipt.json"
    registry = provider_registry or _provider_registry_for_plan_v1(
        plan,
        cuda_full_evolution_v2_checkpoint_occurrence_adapter=(
            cuda_full_evolution_v2_checkpoint_occurrence_adapter
        ),
    )
    profile_registry = _profile_registry_for_plan_v1(plan)
    baseline_admission: dict[str, Any] | None = None
    if isinstance(request, ExecutionRequestV2):
        baseline_admission = build_baseline_admission_receipt_v1(
            declaration=request.baseline,
            plan=plan,
            request_sha256=request.request_sha256,
            front_door_id=EXECUTION_REQUEST_V2_SCHEMA_VERSION,
            artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            canopy_enabled=True,
        )
        if not _configuration_admission_allowed_v1(
            admission_receipt=baseline_admission,
            request_sha256=request.request_sha256,
            plan_sha256=str(plan["plan_sha256"]),
        ):
            write_baseline_receipt_v1(
                runtime_root / "baseline_admission_receipt.json",
                baseline_admission,
            )
            result = _base_result(
                request=request,
                source_identity=source_identity,
                plan=plan,
                run_root=runtime_root,
                preflight=None,
                preflight_path=preflight_path,
                invocation_mode=mode,
            )
            result.update(
                {
                    "status": "precheck_failed",
                    "result_class": "BASELINE_ADMISSION_FAILED",
                    "baseline": baseline_result_binding_v1(
                        admission_receipt=baseline_admission
                    ),
                    "warnings": list(baseline_admission.get("reason_codes", [])),
                }
            )
            return _write_result(runtime_root, result)
    started = monotonic_fn()
    preflight_poll_count = 0
    preflight: Mapping[str, Any] | None = None
    while True:
        preflight_poll_count += 1
        try:
            execution_authority_scope = _execution_authority_scope_v1(
                purpose=request.purpose,
                plan=plan,
            )
            preflight_kwargs: dict[str, Any] = {
                "provider_registry": registry,
                # Preflight is a read-only diagnostic.  A real execution
                # must reach orchestration, whose RunStorageGuard owns the
                # single manifest-authorized pre-admission reclamation pass;
                # otherwise storage pressure is converted to WAIT_STORAGE
                # before the reform can release an exact queued R0-R2 row.
                "enforce_storage_admission": mode != "execute",
                "qualprep_mode": (
                    execution_authority_scope == "qualprep_development_only"
                ),
                "engineering_mode": (
                    execution_authority_scope
                    == "engineering_unqualified_real_use"
                ),
            }
            if profile_registry is not None:
                preflight_kwargs["registry"] = profile_registry
            if nested_storage_parent_run_root is not None:
                preflight_kwargs[
                    "nested_storage_parent_run_root"
                ] = nested_storage_parent_run_root
            preflight = preflight_fn(plan, **preflight_kwargs)
        except Exception as exc:
            result = _base_result(
                request=request,
                source_identity=source_identity,
                plan=plan,
                run_root=runtime_root,
                preflight=None,
                preflight_path=preflight_path,
                invocation_mode=mode,
            )
            result.update(
                {
                    "status": "failed",
                    "result_class": "PREFLIGHT_ERROR",
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                }
            )
            if baseline_admission is not None:
                write_baseline_receipt_v1(
                    runtime_root / "baseline_admission_receipt.json",
                    baseline_admission,
                )
                result["baseline"] = baseline_result_binding_v1(
                    admission_receipt=baseline_admission
                )
            return _write_result(runtime_root, result)
        if not isinstance(preflight, Mapping):
            raise ExecutionFrontDoorError("preflight did not return an object")
        preflight = dict(preflight)
        atomic_write_json(preflight_path, preflight)
        resource_wait = request.resource_wait
        wait_kind = _preflight_wait_kind(preflight)
        if (
            preflight.get("ready_for_execution") is True
            or not resource_wait.enabled
            or wait_kind is None
        ):
            break
        elapsed = monotonic_fn() - started
        if (
            resource_wait.max_wait_seconds is not None
            and elapsed >= resource_wait.max_wait_seconds
        ):
            break
        wait_state, _, _ = _wait_outcome(wait_kind)
        atomic_write_json(
            runtime_root / "execution_state.json",
            {
                "schema_version": "eiel_execution_state_v1",
                "status": wait_state,
                "invocation_mode": mode,
                "request_sha256": request.request_sha256,
                "plan_sha256": plan["plan_sha256"],
                "source_identity_sha256": canonical_sha256(source_identity),
                "preflight_poll_count": preflight_poll_count,
                "last_preflight_sha256": canonical_sha256(preflight),
                "next_poll_seconds": resource_wait.poll_seconds,
                "updated_at_utc": _now(),
            },
        )
        sleep_fn(resource_wait.poll_seconds)

    if isinstance(request, ExecutionRequestV2):
        baseline_admission = build_baseline_admission_receipt_v1(
            declaration=request.baseline,
            plan=plan,
            request_sha256=request.request_sha256,
            front_door_id=EXECUTION_REQUEST_V2_SCHEMA_VERSION,
            preflight=preflight,
            artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            canopy_enabled=True,
        )
        write_baseline_receipt_v1(
            runtime_root / "baseline_admission_receipt.json", baseline_admission
        )
    result = _base_result(
        request=request,
        source_identity=source_identity,
        plan=plan,
        run_root=runtime_root,
        preflight=preflight,
        preflight_path=preflight_path,
        invocation_mode=mode,
    )
    result["preflight_poll_count"] = preflight_poll_count
    if baseline_admission is not None:
        result["baseline"] = baseline_result_binding_v1(
            admission_receipt=baseline_admission
        )
    if preflight.get("ready_for_execution") is not True:
        wait_kind = _preflight_wait_kind(preflight)
        _, wait_status, wait_result_class = (
            _wait_outcome(wait_kind)
            if wait_kind is not None
            else ("", "precheck_failed", "PRECHECK_FAILED")
        )
        result.update(
            {
                "status": wait_status,
                "result_class": wait_result_class,
                "ok": False,
                "mechanical_verification": mechanical_verifier_fn(
                    run_root=runtime_root,
                    plan=plan,
                    preflight=preflight,
                ),
            }
        )
        if wait_kind is not None:
            wait_state, _, _ = _wait_outcome(wait_kind)
            atomic_write_json(
                runtime_root / "execution_state.json",
                {
                    "schema_version": "eiel_execution_state_v1",
                    "status": wait_state,
                    "invocation_mode": mode,
                    "request_sha256": request.request_sha256,
                    "plan_sha256": plan["plan_sha256"],
                    "source_identity_sha256": canonical_sha256(source_identity),
                    "preflight_poll_count": preflight_poll_count,
                    "last_preflight_sha256": canonical_sha256(preflight),
                    "next_poll_seconds": resource_wait.poll_seconds,
                    "updated_at_utc": _now(),
                },
            )
        return _write_result(runtime_root, result)
    if (
        baseline_admission is not None
        and baseline_admission.get("admitted") is not True
    ):
        result.update(
            {
                "status": "precheck_failed",
                "result_class": "BASELINE_ADMISSION_FAILED",
                "mechanical_verification": mechanical_verifier_fn(
                    run_root=runtime_root,
                    plan=plan,
                    preflight=preflight,
                ),
                "warnings": list(baseline_admission.get("reason_codes", [])),
            }
        )
        return _write_result(runtime_root, result)
    attempt_binding: dict[str, Any] | None = None
    try:
        shadow_state = (
            _persist_cuda_full_evolution_v2_shadow_v1(
                request=request,
                plan=plan,
                source_identity=source_identity,
                preflight=preflight,
                runtime_root=runtime_root,
                provider_registry=registry,
                crca_adapter=cuda_full_evolution_v2_crca_adapter,
            )
            if isinstance(request, ExecutionRequestV2)
            else None
        )
    except Exception as exc:
        result.update(
            {
                "status": "precheck_failed",
                "result_class": "CRCA_SHADOW_PROJECTION_FAILED",
                "ok": False,
                "mechanical_verification": mechanical_verifier_fn(
                    run_root=runtime_root,
                    plan=plan,
                    preflight=preflight,
                ),
                "warnings": [f"{type(exc).__name__}: {exc}"],
            }
        )
        return _write_result(runtime_root, result)
    if shadow_state is not None:
        result["crca_shadow_execution_projection"] = dict(shadow_state.binding)
        if mode == "execute":
            if cuda_full_evolution_v2_crca_adapter is None:
                raise ExecutionFrontDoorError(
                    "CUDA Native V2 CRCA adapter disappeared before Attempt binding"
                )
            try:
                attempt_binding = _bind_cuda_full_evolution_v2_attempt_authority_v1(
                    request=request,
                    plan=plan,
                    source_identity=source_identity,
                    preflight=preflight,
                    runtime_root=runtime_root,
                    shadow_state=shadow_state,
                    crca_adapter=cuda_full_evolution_v2_crca_adapter,
                )
            except Exception as exc:
                attempt_binding = {
                    "execution_authority_granted": False,
                    "reason_codes": [
                        f"CRCA_ATTEMPT_ADAPTER_ERROR:{type(exc).__name__}:{exc}"
                    ],
                }
            result["crca_attempt_authority"] = attempt_binding
            if attempt_binding.get("execution_authority_granted") is True:
                pass
            else:
                reasons = attempt_binding.get("reason_codes")
                warnings = (
                    [str(value) for value in reasons]
                    if isinstance(reasons, list)
                    else [
                        "CUDA Native V2 requires Scheduler-owned canonical CRCA "
                        "attempt admission, resource, continuation, and disposition leases"
                    ]
                )
                result.update(
                    {
                        "status": "precheck_failed",
                        "result_class": "ATTEMPT_ADMISSION_UNAVAILABLE",
                        "ok": False,
                        "mechanical_verification": mechanical_verifier_fn(
                            run_root=runtime_root,
                            plan=plan,
                            preflight=preflight,
                        ),
                        "warnings": warnings,
                    }
                )
                return _write_result(runtime_root, result)
        # Purpose classifies the run; it does not grant execution authority.
        # Once the canonical Scheduler has admitted the exact Attempt above,
        # development simulations, validations, and benchmarks may execute as
        # explicitly non-qualified real-use evidence before QualPrep.
    if mode == "validate":
        verification = mechanical_verifier_fn(
            run_root=runtime_root,
            plan=plan,
            preflight=preflight,
        )
        verification_passed = verification.get("status") == "PASS"
        result.update(
            {
                "status": "preflight_only",
                "result_class": (
                    "PREFLIGHT_PASS"
                    if verification_passed
                    else "MECHANICAL_VERIFICATION_FAILED"
                ),
                "ok": verification_passed,
                "mechanical_verification": verification,
            }
        )
        return _write_result(runtime_root, result)
    canopy_supervision: dict[str, Any] | None = None
    if isinstance(request, ExecutionRequestV2):
        try:
            canopy_supervision = dict(
                canopy_start_fn(
                    plan=plan,
                    run_root=runtime_root,
                    repository_root=root,
                    enabled=True,
                    group_prefix="launchpad",
                )
            )
        except (CanopySupervisionError, OSError, ValueError) as exc:
            canopy_supervision = record_canopy_observability_degraded_v1(
                runtime_root,
                plan=plan,
                reason=f"{type(exc).__name__}: {exc}",
            )
            result["warnings"].append(
                f"BASELINE_OBSERVABILITY_DEGRADED: {type(exc).__name__}: {exc}"
            )
    try:

        def execute_once() -> dict[str, Any]:
            execute_kwargs: dict[str, Any] = {
                "run_root": runtime_root,
                "provider_registry": registry,
                "qualification_mode": request.purpose == "qualification",
                "execution_scope": _execution_authority_scope_v1(
                    purpose=request.purpose,
                    plan=plan,
                ),
            }
            if nested_storage_parent_run_root is not None:
                execute_kwargs[
                    "nested_storage_parent_run_root"
                ] = nested_storage_parent_run_root
            if before_lifecycle_closeout_fn is not None:
                execute_kwargs[
                    "before_lifecycle_closeout_fn"
                ] = before_lifecycle_closeout_fn
            return dict(execute_fn(plan, **execute_kwargs))

        if before_execute_fn is not None:
            if not callable(before_execute_fn):
                raise ExecutionFrontDoorError(
                    "before_execute_fn must be callable"
                )
            # The canonical request and execution-plan receipts have already
            # been bound above.  Preparatory observers may therefore create
            # their batch-local roots without racing the fresh-root invariant.
            # This hook runs once per front-door invocation, before the first
            # producer call and before any deliberate restore continuation.
            before_execute_fn()

        execution_contract = plan.get("execution")
        restore_ticks: tuple[int, ...] = ()
        if isinstance(execution_contract, Mapping):
            raw_restore_ticks = execution_contract.get("deliberate_restore_ticks")
            if isinstance(raw_restore_ticks, list) and all(
                type(value) is int for value in raw_restore_ticks
            ):
                restore_ticks = tuple(int(value) for value in raw_restore_ticks)
            elif type(execution_contract.get("deliberate_restore_tick")) is int:
                restore_ticks = (int(execution_contract["deliberate_restore_tick"]),)
        completed_restore_events: list[dict[str, Any]] = []
        physical_batches = plan.get("physical_batches", ())
        expected_batch_ids = {
            str(row["batch_id"])
            for row in physical_batches
            if isinstance(row, Mapping) and isinstance(row.get("batch_id"), str)
        }
        expected_restore_events = len(restore_ticks) * len(expected_batch_ids)
        fresh_process_restore = (
            request.restore_process_isolation == "fresh_invocation_per_restore"
        )
        pending_restore_event: dict[str, Any] | None = None
        pending_producer_process: dict[str, Any] | None = None
        completed_process_transitions: list[dict[str, Any]] = []
        current_process = _validated_restore_process_identity(
            _process_identity(), name="current restore process"
        )

        def restore_event(value: Any, *, name: str) -> dict[str, Any]:
            if not isinstance(value, Mapping) or set(value) != {
                "batch_id",
                "restore_tick",
            }:
                raise ExecutionFrontDoorError(
                    f"{name} must contain batch_id and restore_tick"
                )
            event = {
                "batch_id": value.get("batch_id"),
                "restore_tick": value.get("restore_tick"),
            }
            if (
                event["batch_id"] not in expected_batch_ids
                or event["restore_tick"] not in restore_ticks
            ):
                raise ExecutionFrontDoorError(
                    f"{name} is outside the deliberate restore schedule"
                )
            return event

        if fresh_process_restore:
            if not restore_ticks or not expected_batch_ids:
                raise ExecutionFrontDoorError(
                    "fresh restore-process isolation requires a deliberate restore "
                    "schedule and physical batches"
                )
            if existing_restore_boundary is not None:
                raw_completed = existing.get("deliberate_restore_events_completed", [])
                if not isinstance(raw_completed, list):
                    raise ExecutionFrontDoorError(
                        "restore-pending result has invalid completed events"
                    )
                completed_restore_events = [
                    restore_event(value, name="completed restore event")
                    for value in raw_completed
                ]
                if len(completed_restore_events) != len(
                    {canonical_sha256(value) for value in completed_restore_events}
                ):
                    raise ExecutionFrontDoorError(
                        "restore-pending result repeats a completed event"
                    )
                pending_restore_event = restore_event(
                    existing.get("pending_restore_event"),
                    name="pending restore event",
                )
                raw_transitions = existing.get(
                    "restore_process_transitions_completed", []
                )
                if not isinstance(raw_transitions, list) or not all(
                    isinstance(value, Mapping) for value in raw_transitions
                ):
                    raise ExecutionFrontDoorError(
                        "restore-pending result has invalid process transitions"
                    )
                completed_process_transitions = [
                    dict(value) for value in raw_transitions
                ]
                if len(completed_process_transitions) != len(completed_restore_events):
                    raise ExecutionFrontDoorError(
                        "restore-pending process transition frontier differs"
                    )
                for transition, completed_event in zip(
                    completed_process_transitions,
                    completed_restore_events,
                    strict=True,
                ):
                    producer = _validated_restore_process_identity(
                        transition.get("producer_process"),
                        name="completed checkpoint producer process",
                    )
                    restorer = _validated_restore_process_identity(
                        transition.get("restore_process"),
                        name="completed checkpoint restore process",
                    )
                    if (
                        set(transition)
                        != {
                            "schema_version",
                            "event",
                            "producer_process",
                            "restore_process",
                        }
                        or transition.get("schema_version")
                        != "eiel_execution_restore_process_transition_v1"
                        or restore_event(
                            transition.get("event"),
                            name="completed process transition event",
                        )
                        != completed_event
                        or producer.get("process_instance_id")
                        == restorer.get("process_instance_id")
                    ):
                        raise ExecutionFrontDoorError(
                            "restore-pending process transition is invalid"
                        )
                boundary = existing.get("restore_process_boundary")
                if not isinstance(boundary, Mapping):
                    raise ExecutionFrontDoorError(
                        "restore-pending result lacks its process boundary"
                    )
                boundary_event = restore_event(
                    boundary.get("pending_event"),
                    name="process-bound pending restore event",
                )
                raw_producer_process = _validated_restore_process_identity(
                    boundary.get("producer_process"),
                    name="pending checkpoint producer process",
                )
                if (
                    boundary.get("schema_version")
                    != "eiel_execution_restore_process_boundary_v1"
                    or boundary_event != pending_restore_event
                ):
                    raise ExecutionFrontDoorError(
                        "restore-pending process boundary is invalid"
                    )
                pending_producer_process = dict(raw_producer_process)
                if (
                    pending_restore_event in completed_restore_events
                    or len(completed_restore_events) >= expected_restore_events
                ):
                    raise ExecutionFrontDoorError(
                        "restore-pending result has an impossible event frontier"
                    )

        execution_receipt = execute_once()
        execution_storage_wait = bool(
            execution_receipt.get("status") == "interrupted"
            and execution_receipt.get("storage_wait") is True
        )
        if pending_restore_event is not None and not execution_storage_wait:
            completed_restore_events.append(pending_restore_event)
            if pending_producer_process is None:
                raise ExecutionFrontDoorError(
                    "pending restore lacks its producer process identity"
                )
            completed_process_transitions.append(
                {
                    "schema_version": "eiel_execution_restore_process_transition_v1",
                    "event": pending_restore_event,
                    "producer_process": pending_producer_process,
                    "restore_process": current_process,
                }
            )
        while (
            execution_receipt.get("status") == "interrupted"
            and not execution_storage_wait
        ):
            checkpoint_tick = execution_receipt.get("interrupted_checkpoint_next_tick")
            batch_id = execution_receipt.get("interrupted_batch_id")
            if (
                checkpoint_tick not in restore_ticks
                or batch_id not in expected_batch_ids
            ):
                raise ExecutionFrontDoorError(
                    "campaign interrupted outside its deliberate restore schedule"
                )
            event = {"batch_id": batch_id, "restore_tick": checkpoint_tick}
            if event in completed_restore_events or len(completed_restore_events) >= (
                expected_restore_events
            ):
                raise ExecutionFrontDoorError(
                    "campaign repeated a completed deliberate restore event"
                )
            if fresh_process_restore:
                if completed_restore_events:
                    execution_receipt["deliberate_restore_events_completed"] = list(
                        completed_restore_events
                    )
                    execution_receipt["resume_invocations"] = len(
                        completed_restore_events
                    )
                    execution_receipt["restore_process_transitions_completed"] = list(
                        completed_process_transitions
                    )
                process_boundary = {
                    "schema_version": "eiel_execution_restore_process_boundary_v1",
                    "pending_event": event,
                    "producer_process": current_process,
                    "created_at_utc": _now(),
                }
                result.update(
                    {
                        "execution_receipt": execution_receipt,
                        "execution_status": "interrupted",
                        "status": "restore_pending",
                        "result_class": "RESTORE_PENDING",
                        "ok": False,
                        "pending_restore_event": event,
                        "deliberate_restore_events_completed": list(
                            completed_restore_events
                        ),
                        "resume_invocations": len(completed_restore_events),
                        "restore_process_transitions_completed": list(
                            completed_process_transitions
                        ),
                        "restore_process_boundary": process_boundary,
                    }
                )
                atomic_write_json(
                    runtime_root / "execution_state.json",
                    {
                        "schema_version": "eiel_execution_state_v1",
                        "status": "RESTORE_PENDING",
                        "preflight_poll_count": preflight_poll_count,
                        "pending_restore_event": event,
                        "completed_restore_events": list(completed_restore_events),
                        "next_action": "invoke the same execution request in a fresh process",
                        "updated_at_utc": _now(),
                    },
                )
                return _write_result(runtime_root, result)
            completed_restore_events.append(event)
            atomic_write_json(
                runtime_root / "execution_state.json",
                {
                    "schema_version": "eiel_execution_state_v1",
                    "status": "RESUMING_DELIBERATE_RESTORE",
                    "invocation_mode": mode,
                    "request_sha256": request.request_sha256,
                    "plan_sha256": plan["plan_sha256"],
                    "source_identity_sha256": canonical_sha256(source_identity),
                    "preflight_poll_count": preflight_poll_count,
                    "completed_restore_events": list(completed_restore_events),
                    "updated_at_utc": _now(),
                },
            )
            execution_receipt = execute_once()
            execution_storage_wait = bool(
                execution_receipt.get("status") == "interrupted"
                and execution_receipt.get("storage_wait") is True
            )
        if execution_storage_wait and pending_restore_event is not None:
            if pending_producer_process is None or not isinstance(
                existing_restore_boundary, Mapping
            ):
                raise ExecutionFrontDoorError(
                    "storage-wait restore continuation lost its process boundary"
                )
            result.update(
                {
                    "pending_restore_event": pending_restore_event,
                    "deliberate_restore_events_completed": list(
                        completed_restore_events
                    ),
                    "resume_invocations": len(completed_restore_events),
                    "restore_process_transitions_completed": list(
                        completed_process_transitions
                    ),
                    "restore_process_boundary": dict(existing_restore_boundary),
                }
            )
        if completed_restore_events and not execution_storage_wait:
            if len(completed_restore_events) != expected_restore_events:
                raise ExecutionFrontDoorError(
                    "campaign completed before every physical restore event"
                )
            execution_receipt["deliberate_restore_ticks_completed"] = list(
                restore_ticks
            )
            execution_receipt["deliberate_restore_events_completed"] = list(
                completed_restore_events
            )
            execution_receipt["resume_invocations"] = len(completed_restore_events)
            if fresh_process_restore:
                if len(completed_process_transitions) != len(completed_restore_events):
                    raise ExecutionFrontDoorError(
                        "fresh restore process transition evidence is incomplete"
                    )
                execution_receipt["restore_process_transitions_completed"] = list(
                    completed_process_transitions
                )
        result["execution_receipt"] = execution_receipt
        result["execution_status"] = execution_receipt.get("status")
    except (Exception, KeyboardInterrupt) as exc:
        if canopy_supervision is not None and canopy_supervision.get("relays"):
            mark_canopy_source_interrupted_v1(
                runtime_root,
                reason=f"{type(exc).__name__}: {exc}",
            )
        result.update(
            {
                "execution_status": "failed",
                "result_class": "EXECUTION_FAILED",
            }
        )
        from .cuda_full_evolution_v2_workhorse_provider import (
            RUNTIME_HEADROOM_REJECTION_CODE_V2,
        )

        if RUNTIME_HEADROOM_REJECTION_CODE_V2 in str(exc):
            result.update(
                {
                    "result_class": RUNTIME_HEADROOM_REJECTION_CODE_V2,
                    "resource_boundary": "RUNTIME_VRAM_HEADROOM",
                    "retry_contract": "FRESH_EXECUTION_REQUEST_V2_REQUIRED",
                }
            )
        result.setdefault("warnings", []).append(f"{type(exc).__name__}: {exc}")
        execution_receipt = None
    if isinstance(request, ExecutionRequestV2) and baseline_admission is not None:
        baseline_conformance = build_baseline_conformance_receipt_v1(
            admission_receipt=baseline_admission,
            execution_status=(
                str(result.get("execution_status"))
                if result.get("execution_status") is not None
                else None
            ),
            declaration=request.baseline,
            plan=plan,
            request_sha256=request.request_sha256,
            front_door_id=EXECUTION_REQUEST_V2_SCHEMA_VERSION,
            preflight=preflight,
            artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            canopy_enabled=True,
            canopy_supervision=canopy_supervision,
        )
        write_baseline_receipt_v1(
            runtime_root / "baseline_conformance_receipt.json",
            baseline_conformance,
        )
        result["baseline"] = baseline_result_binding_v1(
            admission_receipt=baseline_admission,
            conformance_receipt=baseline_conformance,
        )
    verification = mechanical_verifier_fn(
        run_root=runtime_root,
        plan=plan,
        preflight=preflight,
        execution_receipt=execution_receipt,
    )
    result["mechanical_verification"] = verification
    if (
        isinstance(execution_receipt, Mapping)
        and execution_receipt.get("status") == "interrupted"
        and execution_receipt.get("storage_wait") is True
    ):
        result["status"] = "wait_storage"
        result["result_class"] = "STORAGE_WAIT"
    else:
        result["status"] = result.get("execution_status") or "failed"
    additional_terminal_lifecycle_artifacts: dict[str, dict[str, str]] = {}
    if (
        shadow_state is not None
        and attempt_binding is not None
        and attempt_binding.get("execution_authority_granted") is True
        and result.get("execution_status") in {"completed", "failed"}
        and cuda_full_evolution_v2_crca_adapter is not None
    ):
        terminalizer = getattr(
            cuda_full_evolution_v2_crca_adapter,
            "terminalize_attempt_authority_v1",
            None,
        )
        try:
            if not callable(terminalizer):
                raise ExecutionFrontDoorError(
                    "canonical CRCA adapter lacks Attempt terminalization"
                )
            from .cuda_full_evolution_v2_provider import ATTEMPT_AUTHORITY_NAME

            terminal_execution_receipt = (
                dict(execution_receipt)
                if isinstance(execution_receipt, Mapping)
                else None
            )
            if (
                result.get("execution_status") == "failed"
                and terminal_execution_receipt is None
            ):
                semantic_layer = shadow_state.projection.get("layers", {}).get(
                    "semantic", {}
                )
                semantic_payload = (
                    semantic_layer.get("payload")
                    if isinstance(semantic_layer, Mapping)
                    else None
                )
                start_frontier = (
                    semantic_payload.get("absolute_start_next_tick")
                    if isinstance(semantic_payload, Mapping)
                    else None
                )
                if type(start_frontier) is not int or start_frontier < 0:
                    raise ExecutionFrontDoorError(
                        "failed Attempt lacks its canonical start frontier"
                    )
                terminal_execution_receipt = {
                    "schema_version": (
                        "eiel_launchpad_failed_execution_frontiers_v1"
                    ),
                    "status": "failed",
                    "plan_sha256": plan["plan_sha256"],
                    "frontier_semantics": (
                        "last_canonically_committed_frontiers"
                    ),
                    "crca_final_frontiers": {
                        "last_dispatched_next_tick": start_frontier,
                        "last_computed_next_tick": start_frontier,
                        "durable_checkpoint_next_tick": start_frontier,
                        "published_artifact_next_tick": start_frontier,
                    },
                }

            execution = plan.get("execution")
            shared_gpu_policy = (
                execution.get("gpu_resource_policy")
                if isinstance(execution, Mapping)
                else None
            )
            cuda_release_binding: Mapping[str, Any] | None = None
            if shared_gpu_policy is not None:
                release_materializer = getattr(
                    cuda_full_evolution_v2_crca_adapter,
                    "materialize_terminal_cuda_memory_release_v1",
                    None,
                )
                if not callable(release_materializer):
                    raise ExecutionFrontDoorError(
                        "shared CUDA route lacks terminal memory-release authority"
                    )
                cuda_release_binding = dict(
                    release_materializer(
                        attempt_authority_path=(
                            runtime_root / ATTEMPT_AUTHORITY_NAME
                        ),
                        campaign_plan=plan,
                        bound_execution_request=(
                            shadow_state.bound_execution_request
                        ),
                        observer=cuda_memory_release_observer_fn,
                    )
                )
                terminal_execution_receipt = {
                    **dict(terminal_execution_receipt or {}),
                    "cuda_memory_release": dict(cuda_release_binding),
                }
                result["crca_terminal_execution_receipt"] = dict(
                    terminal_execution_receipt
                )
                if isinstance(execution_receipt, Mapping):
                    execution_receipt = terminal_execution_receipt
                    result["execution_receipt"] = dict(execution_receipt)

            terminal_result = dict(
                terminalizer(
                    attempt_authority_path=runtime_root / ATTEMPT_AUTHORITY_NAME,
                    campaign_plan=plan,
                    execution_status=str(result["execution_status"]),
                    execution_receipt=terminal_execution_receipt,
                    result_binding={
                        "request_sha256": request.request_sha256,
                        "plan_sha256": plan["plan_sha256"],
                        "execution_status": result.get("execution_status"),
                        "warnings": list(result.get("warnings", [])),
                        "cuda_memory_release": (
                            None
                            if cuda_release_binding is None
                            else dict(cuda_release_binding)
                        ),
                    },
                )
            )
            result["crca_attempt_terminal"] = terminal_result
            workhorse_terminal = terminal_result.get(
                "workhorse_terminal_binding"
            )
            if (
                result.get("execution_status") == "completed"
                and isinstance(workhorse_terminal, Mapping)
            ):
                provider = registry.get(str(plan.get("resolved_profile")))
                finalizer = getattr(
                    provider, "finalize_attempt_terminal_v2", None
                )
                if not callable(finalizer):
                    raise ExecutionFrontDoorError(
                        "Workhorse provider lacks post-Scheduler row finalization"
                    )
                finalized = dict(
                    finalizer(
                        plan=plan,
                        run_root=runtime_root,
                        canonical_terminal_binding=workhorse_terminal,
                    )
                )
                row_path = Path(
                    str(
                        finalized.get("final_receipt_path")
                        or finalized.get("row_receipt_path")
                        or ""
                    )
                ).resolve()
                try:
                    row_relative = str(row_path.relative_to(runtime_root))
                except ValueError as exc:
                    raise ExecutionFrontDoorError(
                        "Workhorse row receipt left its run root"
                    ) from exc
                operation_name = finalized.get("operation")
                finalized_result_class = finalized.get("result_class", "SUCCESS")
                terminal_status = workhorse_terminal.get(
                    "attempt_terminal_status"
                )
                if (
                    finalized_result_class not in {"SUCCESS", "VALID_NEGATIVE"}
                    or (
                        finalized_result_class == "VALID_NEGATIVE"
                        and (
                            operation_name != "SEMANTIC_EVIDENCE_CASE"
                            or terminal_status != "FAILED"
                        )
                    )
                    or (
                        finalized_result_class == "SUCCESS"
                        and terminal_status != "COMPLETED"
                    )
                ):
                    raise ExecutionFrontDoorError(
                        "Workhorse final result class differs from its Scheduler terminal"
                    )
                result["workhorse_terminalized_operation"] = finalized
                if operation_name in {None, "EXECUTION_ROW"}:
                    result["workhorse_terminalized_row"] = finalized
                elif operation_name == "DEVICE_ENVELOPE_CALIBRATION":
                    result["workhorse_terminalized_calibration"] = finalized
                elif operation_name == "SEMANTIC_EVIDENCE_CASE":
                    result["workhorse_terminalized_semantic_case"] = finalized
                elif operation_name == "SCIENCE_COMPATIBILITY_DEMO":
                    result["workhorse_terminalized_science_demo"] = finalized
                elif operation_name == "INDEXED_ACS_CONTINUATION":
                    result["workhorse_terminalized_indexed_continuation"] = finalized
                else:
                    raise ExecutionFrontDoorError(
                        "Workhorse finalized an unsupported operation"
                    )
                lifecycle_binding = finalized.get("lifecycle_binding")
                if lifecycle_binding is None:
                    lifecycle_binding = {
                        "artifact_family": (
                            "qualification.workhorse_technical_row_v2"
                        ),
                        "retention_class": "R3",
                        "owner_writer": (
                            "cuda_full_evolution_workhorse_v2_single_row_owner"
                        ),
                        "release_predicate": (
                            "technical row retained for offline matrix aggregation"
                        ),
                    }
                if (
                    not isinstance(lifecycle_binding, Mapping)
                    or set(lifecycle_binding)
                    != {
                        "artifact_family",
                        "retention_class",
                        "owner_writer",
                        "release_predicate",
                    }
                    or lifecycle_binding.get("retention_class") != "R3"
                ):
                    raise ExecutionFrontDoorError(
                        "Workhorse final receipt lifecycle binding is invalid"
                    )
                additional_terminal_lifecycle_artifacts[row_relative] = {
                    str(key): str(value)
                    for key, value in lifecycle_binding.items()
                }
        except Exception as exc:
            result.update(
                {
                    "status": "failed",
                    "result_class": "CRCA_ATTEMPT_TERMINALIZATION_FAILED",
                    "ok": False,
                }
            )
            result.setdefault("warnings", []).append(
                f"CRCA_ATTEMPT_TERMINALIZATION_FAILED:{type(exc).__name__}:{exc}"
            )
    if result["status"] == "completed" and verification.get("status") == "PASS":
        terminalized = result.get("workhorse_terminalized_operation")
        terminalized_result_class = (
            terminalized.get("result_class")
            if isinstance(terminalized, Mapping)
            else None
        )
        result["result_class"] = (
            terminalized_result_class
            if terminalized_result_class in {"SUCCESS", "VALID_NEGATIVE"}
            else "SUCCESS"
        )
        result["ok"] = True
    elif result["status"] == "completed":
        result["result_class"] = "MECHANICAL_VERIFICATION_FAILED"
        result["ok"] = False
    else:
        result["ok"] = False
    qualification = evaluate_qualification_profile_v1(
        request.qualification_profile,
        execution_result=_result_with_hash(result),
    )
    if qualification is not None:
        result["qualification_evaluation"] = qualification
        result["qualification_status"] = qualification["status"]
    terminal_state = {
        "schema_version": "eiel_execution_state_v1",
        "status": result["status"],
        "invocation_mode": mode,
        "request_sha256": request.request_sha256,
        "plan_sha256": plan["plan_sha256"],
        "source_identity_sha256": canonical_sha256(source_identity),
        "preflight_poll_count": preflight_poll_count,
        "result_sha256": canonical_sha256(result),
        "updated_at_utc": _now(),
    }
    if isinstance(result.get("restore_process_boundary"), Mapping):
        terminal_state.update(
            {
                "pending_restore_event": result.get("pending_restore_event"),
                "completed_restore_events": result.get(
                    "deliberate_restore_events_completed", []
                ),
                "restore_process_boundary": dict(result["restore_process_boundary"]),
            }
        )
    atomic_write_json(runtime_root / "execution_state.json", terminal_state)
    final_result = _write_result(runtime_root, result)
    if (
        mode == "execute"
        and artifact_lifecycle_explicit
        and result.get("status") == "completed"
    ):
        try:
            _reseal_and_publish_terminal_lifecycle_v1(
                plan=plan,
                run_root=runtime_root,
                request_sha256=request.request_sha256,
                source_identity=source_identity,
                result=final_result,
                additional_artifacts=additional_terminal_lifecycle_artifacts,
            )
        except (
            ArtifactLifecycleError,
            CampaignExecutionError,
            ExecutionFrontDoorError,
            OSError,
            ValueError,
        ) as exc:
            lifecycle_failure = dict(result)
            lifecycle_failure.update(
                {
                    "status": "failed",
                    "result_class": "TERMINAL_LIFECYCLE_RESEAL_FAILED",
                    "ok": False,
                }
            )
            lifecycle_failure.setdefault("warnings", []).append(
                f"TERMINAL_LIFECYCLE_RESEAL_FAILED:{type(exc).__name__}:{exc}"
            )
            failure_state = {
                **terminal_state,
                "status": "failed",
                "result_sha256": canonical_sha256(lifecycle_failure),
                "updated_at_utc": _now(),
            }
            atomic_write_json(runtime_root / "execution_state.json", failure_state)
            _write_result(runtime_root, lifecycle_failure)
            raise ExecutionFrontDoorError(
                f"terminal lifecycle reseal failed: {type(exc).__name__}: {exc}"
            ) from exc
    return final_result


def execute_request_v1(request: ExecutionRequestV1, **kwargs: Any) -> dict[str, Any]:
    """Validate or continue an exact historical V1 request."""

    if not isinstance(request, ExecutionRequestV1):
        raise ExecutionFrontDoorError("execute_request_v1 requires ExecutionRequestV1")
    return _execute_request_common_v1(request, **kwargs)


def execute_request_v2(request: ExecutionRequestV2, **kwargs: Any) -> dict[str, Any]:
    """Validate or execute a Baseline-declared V2 request."""

    if not isinstance(request, ExecutionRequestV2):
        raise ExecutionFrontDoorError("execute_request_v2 requires ExecutionRequestV2")
    return _execute_request_common_v1(request, **kwargs)


def execute_bound_campaign_plan_v2(
    request: ExecutionRequestV2,
    campaign_plan: Mapping[str, Any],
    *,
    source_lineage: Mapping[str, Any],
    science_lineage: Mapping[str, Any],
    repo_root: str | Path | None = None,
    nested_storage_parent_run_root: str | Path | None = None,
    before_execute_fn: Callable[[], None] | None = None,
    before_lifecycle_closeout_fn: Callable[[], None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute one already-materialized science plan through Launchpad.

    ``campaign_plan`` is the canonical ``eiel_campaign_plan_v1`` materialized
    by the science compiler/lowering boundary.  This helper intentionally does
    not load or compile a campaign specification: authoring/source and science
    identities are supplied explicitly and are sealed into the ordinary front
    door request/result receipts.  All execution authority still comes from
    ``_execute_request_common_v1`` (Baseline, preflight, lifecycle/storage,
    provider registration, CRCA Attempt admission, Canopy, and closeout).

    This seam is deliberately narrower than the generic request entrypoint.
    It accepts engineering use and a scientific simulation only when the
    already-materialized plan and lineage both bind the reusable
    ``EIEL_SCIENCE_INSTRUMENT_V1`` scientific-execution contract.  It cannot
    be used for benchmark, QualPrep, or Qualification purposes, and it cannot
    replace the canonical provider or artifact-lifecycle authorities.
    """

    if not isinstance(request, ExecutionRequestV2):
        raise ExecutionFrontDoorError(
            "execute_bound_campaign_plan_v2 requires ExecutionRequestV2"
        )
    if request.purpose not in {"engineering", "simulation"}:
        raise ExecutionFrontDoorError(
            "execute_bound_campaign_plan_v2 permits engineering or bound "
            "scientific simulation purpose only"
        )
    plan = _validate_bound_campaign_plan_v1(campaign_plan)
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        raise ExecutionFrontDoorError(f"repo_root is not a directory: {root}")
    if not isinstance(source_lineage, Mapping):
        raise ExecutionFrontDoorError("source_lineage must be an object")
    if not isinstance(science_lineage, Mapping):
        raise ExecutionFrontDoorError("science_lineage must be an object")
    if request.purpose == "simulation":
        materialization = plan.get("science_materialization")
        if (
            science_lineage.get("instrument_contract")
            != "EIEL_SCIENCE_INSTRUMENT_V1"
            or science_lineage.get("science_use_class")
            != "SCIENTIFIC_EXECUTION"
            or plan.get("science_use_class") != "SCIENTIFIC_EXECUTION"
            or not isinstance(materialization, Mapping)
            or materialization.get("science_use_class")
            != "SCIENTIFIC_EXECUTION"
            or materialization.get("observation_mode")
            not in {"TYPED_SIDECAR_V1", "COMPACT_SIDECAR_V1"}
        ):
            raise ExecutionFrontDoorError(
                "simulation requires exact EIEL_SCIENCE_INSTRUMENT_V1 "
                "scientific-execution lineage and a canonical science sidecar"
            )

    configured_root = request.output_root
    if configured_root is None:
        output = plan.get("output")
        configured_root = (
            str(output.get("runtime_root"))
            if isinstance(output, Mapping) and output.get("runtime_root")
            else None
        )
    if not configured_root:
        raise ExecutionFrontDoorError(
            "execution request and bound campaign plan both lack output_root"
        )
    configured_path = Path(configured_root).expanduser()
    if configured_path.is_symlink():
        raise ExecutionFrontDoorError("output_root may not be a symlink")
    runtime_root = configured_path
    if not runtime_root.is_absolute():
        runtime_root = root / runtime_root
    runtime_root = runtime_root.resolve()
    if runtime_root == root:
        raise ExecutionFrontDoorError("output_root may not be the repository root")
    if runtime_root.exists() and runtime_root.is_file():
        raise ExecutionFrontDoorError("output_root is an existing regular file")

    bound_plan = _bind_runtime_root(plan, runtime_root)
    source_identity = _bound_campaign_source_identity_v1(
        plan=plan,
        bound_plan=bound_plan,
        source_lineage=source_lineage,
        science_lineage=science_lineage,
        repo_root=root,
    )
    if "mode" in kwargs:
        raise ExecutionFrontDoorError(
            "execute_bound_campaign_plan_v2 does not expose validation mode"
        )
    return _execute_request_common_v1(
        request,
        repo_root=root,
        bound_plan=plan,
        bound_source_identity=source_identity,
        bound_artifact_lifecycle_explicit=True,
        nested_storage_parent_run_root=nested_storage_parent_run_root,
        before_execute_fn=before_execute_fn,
        before_lifecycle_closeout_fn=before_lifecycle_closeout_fn,
        **kwargs,
    )


__all__ = [
    "EXECUTION_PURPOSES",
    "EXECUTION_REQUEST_SCHEMA_VERSION",
    "EXECUTION_REQUEST_V2_SCHEMA_VERSION",
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "EXECUTION_RESULT_V2_SCHEMA_VERSION",
    "CudaFullEvolutionV2LaunchpadCrcaAdapterV1",
    "ExecutionFrontDoorError",
    "ExecutionRequestV1",
    "ExecutionRequestV2",
    "ResourceWaitPolicyV1",
    "evaluate_qualification_profile_v1",
    "execute_bound_campaign_plan_v2",
    "execute_request_v1",
    "execute_request_v2",
    "load_execution_request_v1",
    "verify_mechanical_execution_v1",
]
