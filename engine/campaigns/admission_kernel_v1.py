"""Pure CRCA admission-policy convergence kernel.

The kernel consumes observations made by existing authorities.  It does not
resolve a release, perform representability or Baseline evaluation, inspect
storage, grant a resource/continuation lease, write a receipt, or launch work.
Front-door adapters retain those responsibilities and may cut over only after
their direct decision projection is proven exactly equal to this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

from .canonical import canonical_sha256


ADMISSION_KERNEL_REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_crca_admission_kernel_request_v1"
)
ADMISSION_KERNEL_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_crca_admission_kernel_receipt_v1"
)
ADMISSION_KERNEL_POLICY_VERSION: Final[str] = "crca_admission_policy_v1"
ADMISSION_CHECK_ORDER_V1: Final[tuple[str, ...]] = (
    "frozen_release_exact",
    "representability_exact",
    "baseline",
    "artifact_lifecycle",
    "no_claims",
    "qualification_authority",
    "resource_admission",
)
ADMISSION_CHECK_STATES_V1: Final[tuple[str, ...]] = (
    "PASS",
    "FAIL",
    "WAIT",
    "NOT_APPLICABLE",
)
ADMISSION_EFFECT_REQUESTS_V1: Final[tuple[str, ...]] = (
    "NONE",
    "REQUEST_PREFLIGHT_OBSERVATION",
    "REQUEST_RESOURCE_AND_CONTINUATION_LEASE",
)


class AdmissionKernelError(ValueError):
    """The proposed pure admission decision is malformed or not equivalent."""


def _nonempty(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdmissionKernelError(f"{path} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class AdmissionCheckV1:
    """One normalized observation from an existing policy authority."""

    check_id: str
    state: str
    enforced: bool
    evidence_sha256: str | None = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.check_id not in ADMISSION_CHECK_ORDER_V1:
            raise AdmissionKernelError(
                f"unsupported admission check_id: {self.check_id!r}"
            )
        if self.state not in ADMISSION_CHECK_STATES_V1:
            raise AdmissionKernelError(
                f"unsupported admission check state: {self.state!r}"
            )
        if type(self.enforced) is not bool:
            raise AdmissionKernelError("admission check enforced must be boolean")
        if self.enforced and self.state == "NOT_APPLICABLE":
            raise AdmissionKernelError(
                f"enforced admission check cannot be NOT_APPLICABLE: {self.check_id}"
            )
        if self.evidence_sha256 is not None:
            digest = self.evidence_sha256
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise AdmissionKernelError(
                    f"{self.check_id}.evidence_sha256 must be lowercase SHA-256"
                )
        if any(
            not isinstance(code, str) or not code.strip() for code in self.reason_codes
        ):
            raise AdmissionKernelError(
                f"{self.check_id}.reason_codes must contain non-empty strings"
            )
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise AdmissionKernelError(
                f"{self.check_id}.reason_codes must not contain duplicates"
            )
        if self.state in {"FAIL", "WAIT"} and not self.reason_codes:
            raise AdmissionKernelError(
                f"{self.check_id} {self.state} requires a reason code"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "state": self.state,
            "enforced": self.enforced,
            "evidence_sha256": self.evidence_sha256,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class AdmissionKernelRequestV1:
    """Complete immutable input to one pure admission-policy decision."""

    front_door_id: str
    intent: str
    request_sha256: str
    plan_sha256: str
    checks: tuple[AdmissionCheckV1, ...]
    allow_effect_request: str
    policy_version: str = ADMISSION_KERNEL_POLICY_VERSION
    schema_version: str = ADMISSION_KERNEL_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADMISSION_KERNEL_REQUEST_SCHEMA_VERSION:
            raise AdmissionKernelError("unsupported admission kernel request schema")
        if self.policy_version != ADMISSION_KERNEL_POLICY_VERSION:
            raise AdmissionKernelError("unsupported admission kernel policy version")
        for path, value in (
            ("front_door_id", self.front_door_id),
            ("intent", self.intent),
            ("request_sha256", self.request_sha256),
            ("plan_sha256", self.plan_sha256),
        ):
            _nonempty(value, path=path)
        if self.allow_effect_request not in ADMISSION_EFFECT_REQUESTS_V1[1:]:
            raise AdmissionKernelError(
                "allow_effect_request must name an admitted imperative-shell effect"
            )
        observed_order = tuple(check.check_id for check in self.checks)
        if observed_order != ADMISSION_CHECK_ORDER_V1:
            raise AdmissionKernelError(
                "admission checks must be complete, unique, and canonically ordered"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "front_door_id": self.front_door_id,
            "intent": self.intent,
            "request_sha256": self.request_sha256,
            "plan_sha256": self.plan_sha256,
            "checks": [check.payload() for check in self.checks],
            "allow_effect_request": self.allow_effect_request,
        }


def evaluate_admission_kernel_v1(
    request: AdmissionKernelRequestV1,
) -> dict[str, Any]:
    """Return a content-addressed decision without performing any effects."""

    failures = tuple(
        check for check in request.checks if check.enforced and check.state == "FAIL"
    )
    waits = tuple(
        check for check in request.checks if check.enforced and check.state == "WAIT"
    )
    if failures:
        status = "REJECT"
        decisive = failures
        effect_request = "NONE"
    elif waits:
        status = "WAIT_RESOURCE"
        decisive = waits
        effect_request = "NONE"
    else:
        status = "ALLOW"
        decisive = ()
        effect_request = request.allow_effect_request
    decision = {
        "status": status,
        "allowed": status == "ALLOW",
        "waiting": status == "WAIT_RESOURCE",
        "reason_codes": [code for check in decisive for code in check.reason_codes],
        "effect_request": effect_request,
    }
    core = {
        "schema_version": ADMISSION_KERNEL_RECEIPT_SCHEMA_VERSION,
        "request": request.payload(),
        "request_sha256": canonical_sha256(request.payload()),
        "decision": decision,
        "effects_performed": False,
        "authority": "pure_policy_projection_only_v1",
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def assert_admission_policy_parity_v1(
    receipt: Mapping[str, Any],
    *,
    legacy_allowed: bool,
    legacy_waiting: bool,
    legacy_reason_codes: Sequence[str],
    legacy_effect_request: str,
) -> None:
    """Fail closed unless the retained direct policy and kernel are exact."""

    if type(legacy_allowed) is not bool or type(legacy_waiting) is not bool:
        raise AdmissionKernelError("legacy admission projection must be boolean")
    expected = {
        "status": (
            "ALLOW"
            if legacy_allowed
            else ("WAIT_RESOURCE" if legacy_waiting else "REJECT")
        ),
        "allowed": legacy_allowed,
        "waiting": legacy_waiting,
        "reason_codes": list(legacy_reason_codes),
        "effect_request": legacy_effect_request,
    }
    if receipt.get("decision") != expected:
        raise AdmissionKernelError(
            "CRCA admission shadow diverged from the retained direct policy"
        )


def baseline_admission_shadow_v1(
    *,
    front_door_id: str,
    intent: str,
    request_sha256: str,
    plan_sha256: str,
    admission_receipt: Mapping[str, Any],
    execution_required: bool,
) -> dict[str, Any]:
    """Project the existing Baseline decision through the shared kernel.

    Artifact-lifecycle state is retained as an explicit observation, but it is
    not evaluated twice: Baseline remains its owner and its exact direct
    ``configuration_eligible``/``admitted`` result remains the legacy oracle.
    """

    decision_key = "admitted" if execution_required else "configuration_eligible"
    legacy_allowed = admission_receipt.get(decision_key) is True
    receipt_digest = admission_receipt.get("receipt_sha256")
    evidence_sha256 = receipt_digest if isinstance(receipt_digest, str) else None
    legacy_reason_codes = (
        ()
        if legacy_allowed
        else tuple(str(code) for code in admission_receipt.get("reason_codes", ()))
    )
    if not legacy_allowed and not legacy_reason_codes:
        legacy_reason_codes = ("BASELINE_ADMISSION_FAILED",)
    lifecycle_row = next(
        (
            row
            for row in admission_receipt.get("requirements", ())
            if isinstance(row, Mapping)
            and row.get("requirement_id") == "artifacts.lifecycle_contract"
        ),
        None,
    )
    lifecycle_state = (
        "PASS"
        if isinstance(lifecycle_row, Mapping)
        and lifecycle_row.get("configuration_conformant") is True
        else "FAIL"
    )
    lifecycle_reasons = (
        () if lifecycle_state == "PASS" else ("ARTIFACT_LIFECYCLE_NOT_CONFORMANT",)
    )
    checks = (
        AdmissionCheckV1("frozen_release_exact", "NOT_APPLICABLE", False),
        AdmissionCheckV1("representability_exact", "NOT_APPLICABLE", False),
        AdmissionCheckV1(
            "baseline",
            "PASS" if legacy_allowed else "FAIL",
            True,
            evidence_sha256,
            legacy_reason_codes,
        ),
        AdmissionCheckV1(
            "artifact_lifecycle",
            lifecycle_state,
            False,
            evidence_sha256,
            lifecycle_reasons,
        ),
        AdmissionCheckV1("no_claims", "NOT_APPLICABLE", False),
        AdmissionCheckV1("qualification_authority", "NOT_APPLICABLE", False),
        AdmissionCheckV1("resource_admission", "NOT_APPLICABLE", False),
    )
    allow_effect_request = (
        "REQUEST_RESOURCE_AND_CONTINUATION_LEASE"
        if execution_required
        else "REQUEST_PREFLIGHT_OBSERVATION"
    )
    receipt = evaluate_admission_kernel_v1(
        AdmissionKernelRequestV1(
            front_door_id=front_door_id,
            intent=intent,
            request_sha256=request_sha256,
            plan_sha256=plan_sha256,
            checks=checks,
            allow_effect_request=allow_effect_request,
        )
    )
    assert_admission_policy_parity_v1(
        receipt,
        legacy_allowed=legacy_allowed,
        legacy_waiting=False,
        legacy_reason_codes=legacy_reason_codes,
        legacy_effect_request=(allow_effect_request if legacy_allowed else "NONE"),
    )
    return receipt
