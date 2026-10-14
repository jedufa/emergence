from __future__ import annotations

"""Fail-closed authority policy for CUDA Architecture B Dynamic Flat."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


AUTHORITY_POLICY_ID = (
    "CUDA-STAGE3-AUTHORITY-POLICY-A/"
    "CPP-CPU-SOLE-INDEPENDENT-DYNAMIC-SEMANTIC-AUTHORITY"
)
STAGE2_MISSING_DRIVE_NOISE_STATUS = 4


class AuthorityLabel(str, Enum):
    INDEPENDENT_SEMANTIC_AUTHORITY = "independent_semantic_authority"
    INDEPENDENT_BOUNDED_COMPARATOR = "independent_bounded_comparator"
    DERIVED_CORROBORATION = "derived_corroboration"
    CANDIDATE_SELF_CHECK = "candidate_self_check"


class AuthoritySource(str, Enum):
    CPP_CPU = "cpp_cpu"
    IMMUTABLE_STAGE2 = "immutable_stage2"
    CUDA_CANDIDATE = "cuda_candidate"


class ComparatorInputOrigin(str, Enum):
    INDEPENDENT = "independent"
    CPU_LIVE_HISTORY = "cpu_live_history"
    CANDIDATE_LIVE_HISTORY = "candidate_live_history"


@dataclass(frozen=True)
class CapabilityBoundaryReceipt:
    policy_id: str
    source: str
    capability: str
    supported: bool
    status: str
    reason: str


@dataclass(frozen=True)
class AuthorityClassification:
    source: str
    input_origin: str
    authority_label: str | None
    independent_eligible: bool
    capability_boundary: CapabilityBoundaryReceipt | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["policy_id"] = AUTHORITY_POLICY_ID
        return result


@dataclass(frozen=True)
class DynamicFlatPromotionEvidence:
    cpu_oracle_exact: bool
    cpu_repeat_exact: bool
    cpu_checkpoint_continuation_exact: bool
    candidate_same_device_repeat_exact: bool
    candidate_checkpoint_continuation_exact: bool
    invariants_exact: bool
    sanitizer_clean: bool
    stage2_bounded_prefix_exact: bool
    candidate_derived_comparator_inputs: bool = False


def _enum_value(value: str | Enum, enum_type: type[Enum]) -> Enum:
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {enum_type.__name__}: {value!r}") from exc


def _stage2_capability_boundary(status: int | str | None) -> bool:
    return status in {
        STAGE2_MISSING_DRIVE_NOISE_STATUS,
        "STATUS_MISSING_DRIVE_NOISE",
    }


def classify_dynamic_flat_authority(
    source: AuthoritySource | str,
    *,
    input_origin: ComparatorInputOrigin | str = ComparatorInputOrigin.INDEPENDENT,
    stage2_status: int | str | None = None,
    future_live_slots_present: bool = False,
) -> AuthorityClassification:
    """Classify evidence without extending a comparator past its capability."""

    resolved_source = _enum_value(source, AuthoritySource)
    resolved_origin = _enum_value(input_origin, ComparatorInputOrigin)

    if resolved_source is AuthoritySource.CPP_CPU:
        if resolved_origin is not ComparatorInputOrigin.INDEPENDENT:
            raise ValueError("the C++ CPU authority must use independent inputs")
        return AuthorityClassification(
            source=resolved_source.value,
            input_origin=resolved_origin.value,
            authority_label=AuthorityLabel.INDEPENDENT_SEMANTIC_AUTHORITY.value,
            independent_eligible=True,
            capability_boundary=None,
        )

    if resolved_source is AuthoritySource.CUDA_CANDIDATE:
        if resolved_origin is not ComparatorInputOrigin.INDEPENDENT:
            raise ValueError("candidate self-checks must not manufacture comparator inputs")
        return AuthorityClassification(
            source=resolved_source.value,
            input_origin=resolved_origin.value,
            authority_label=AuthorityLabel.CANDIDATE_SELF_CHECK.value,
            independent_eligible=False,
            capability_boundary=None,
        )

    if resolved_origin is ComparatorInputOrigin.CANDIDATE_LIVE_HISTORY:
        raise ValueError("candidate-derived Stage2 inputs are circular")

    if resolved_origin is ComparatorInputOrigin.CPU_LIVE_HISTORY:
        return AuthorityClassification(
            source=resolved_source.value,
            input_origin=resolved_origin.value,
            authority_label=AuthorityLabel.DERIVED_CORROBORATION.value,
            independent_eligible=False,
            capability_boundary=None,
        )

    if future_live_slots_present or _stage2_capability_boundary(stage2_status):
        status = (
            "STATUS_MISSING_DRIVE_NOISE"
            if _stage2_capability_boundary(stage2_status)
            else "FUTURE_LIVE_SLOTS_PRESENT"
        )
        receipt = CapabilityBoundaryReceipt(
            policy_id=AUTHORITY_POLICY_ID,
            source=resolved_source.value,
            capability="independent_dynamic_stochastic_rows",
            supported=False,
            status=status,
            reason=(
                "immutable Stage2 stochastic rows cover only the initial "
                "live-slot set"
            ),
        )
        return AuthorityClassification(
            source=resolved_source.value,
            input_origin=resolved_origin.value,
            authority_label=None,
            independent_eligible=False,
            capability_boundary=receipt,
        )

    return AuthorityClassification(
        source=resolved_source.value,
        input_origin=resolved_origin.value,
        authority_label=AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR.value,
        independent_eligible=True,
        capability_boundary=None,
    )


def require_authority_label(
    classification: AuthorityClassification,
    claimed_label: AuthorityLabel | str,
) -> None:
    resolved_claim = _enum_value(claimed_label, AuthorityLabel).value
    if classification.authority_label != resolved_claim:
        boundary = classification.capability_boundary
        suffix = f" ({boundary.status})" if boundary is not None else ""
        raise ValueError(
            f"authority label {resolved_claim!r} is invalid for "
            f"{classification.source}{suffix}"
        )
    if resolved_claim.startswith("independent_") and not classification.independent_eligible:
        raise ValueError("evidence is outside its independent capability boundary")


def validate_dynamic_flat_promotion(
    evidence: DynamicFlatPromotionEvidence,
) -> Mapping[str, Any]:
    required = {
        "cpu_oracle_exact": evidence.cpu_oracle_exact,
        "cpu_repeat_exact": evidence.cpu_repeat_exact,
        "cpu_checkpoint_continuation_exact": (
            evidence.cpu_checkpoint_continuation_exact
        ),
        "candidate_same_device_repeat_exact": (
            evidence.candidate_same_device_repeat_exact
        ),
        "candidate_checkpoint_continuation_exact": (
            evidence.candidate_checkpoint_continuation_exact
        ),
        "invariants_exact": evidence.invariants_exact,
        "sanitizer_clean": evidence.sanitizer_clean,
        "stage2_bounded_prefix_exact": evidence.stage2_bounded_prefix_exact,
    }
    missing = sorted(name for name, passed in required.items() if not passed)
    if evidence.candidate_derived_comparator_inputs:
        missing.append("non_circular_comparator_inputs")
    if missing:
        raise ValueError(
            "Dynamic Flat promotion evidence is incomplete: " + ", ".join(missing)
        )
    return {
        "policy_id": AUTHORITY_POLICY_ID,
        "promotion_eligible": True,
        "required_checks": sorted(required),
        "independent_semantic_authority": AuthoritySource.CPP_CPU.value,
    }


def authority_capability_matrix() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_id": AUTHORITY_POLICY_ID,
        "sources": {
            AuthoritySource.CPP_CPU.value: {
                "label": AuthorityLabel.INDEPENDENT_SEMANTIC_AUTHORITY.value,
                "dynamic_post_birth": True,
                "promotion_veto": True,
            },
            AuthoritySource.IMMUTABLE_STAGE2.value: {
                "label": AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR.value,
                "initial_live_slot_prefix": True,
                "dynamic_post_birth": False,
                "cpu_history_label": AuthorityLabel.DERIVED_CORROBORATION.value,
                "promotion_veto_after_boundary": False,
            },
            AuthoritySource.CUDA_CANDIDATE.value: {
                "label": AuthorityLabel.CANDIDATE_SELF_CHECK.value,
                "dynamic_post_birth": True,
                "promotion_veto": False,
            },
        },
    }


def authority_comparator_labels() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_id": AUTHORITY_POLICY_ID,
        "labels": {
            AuthorityLabel.INDEPENDENT_SEMANTIC_AUTHORITY.value: (
                "C++ CPU result produced without candidate-derived inputs"
            ),
            AuthorityLabel.INDEPENDENT_BOUNDED_COMPARATOR.value: (
                "immutable Stage2 result within its representable contract"
            ),
            AuthorityLabel.DERIVED_CORROBORATION.value: (
                "Stage2 result using CPU-derived live-slot history"
            ),
            AuthorityLabel.CANDIDATE_SELF_CHECK.value: (
                "same-device repeat, continuation, invariants, and hash checks"
            ),
        },
        "prohibited": {
            "candidate_live_history": "circular comparator input",
            "stage2_after_missing_drive_noise": "outside independent capability",
        },
    }


def authority_policy_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_id": AUTHORITY_POLICY_ID,
        "decision": "ratified",
        "sole_independent_semantic_authority": AuthoritySource.CPP_CPU.value,
        "stage2_boundary_status": "STATUS_MISSING_DRIVE_NOISE",
        "required_post_birth_checks": [
            "cpu_oracle_exact",
            "cpu_repeat_exact",
            "cpu_checkpoint_continuation_exact",
            "candidate_same_device_repeat_exact",
            "candidate_checkpoint_continuation_exact",
            "invariants_exact",
            "sanitizer_clean",
            "stage2_bounded_prefix_exact",
            "non_circular_comparator_inputs",
        ],
    }


def emit_authority_policy_evidence(output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        "dynamic_flat_authority_policy.json": authority_policy_document(),
        "authority_capability_matrix.json": authority_capability_matrix(),
        "authority_comparator_labels.json": authority_comparator_labels(),
    }
    written: dict[str, str] = {}
    for name, payload in documents.items():
        path = output / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[name] = str(path.resolve())

    markdown = output / "dynamic_flat_authority_policy.md"
    markdown.write_text(
        "\n".join(
            (
                "# Dynamic Flat Authority Policy",
                "",
                f"Decision: `{AUTHORITY_POLICY_ID}`",
                "",
                "The accepted C++ CPU owner is the sole independent semantic ",
                "authority for Dynamic Flat. Immutable Stage2 is independent ",
                "only inside its representable initial-live-slot contract. A ",
                "Stage2 run using CPU live-slot history is derived ",
                "corroboration. Candidate-derived comparator input is circular ",
                "and prohibited.",
                "",
                "After Stage2 reports `STATUS_MISSING_DRIVE_NOISE` or future ",
                "newborn slots are present, it has no independent veto. CUDA ",
                "promotion still requires exact CPU comparison, deterministic ",
                "CPU and CUDA repeats, checkpoint continuation, invariants, a ",
                "clean sanitizer, and exact Stage2 bounded-prefix evidence.",
                "",
            )
        ),
        encoding="utf-8",
    )
    written[markdown.name] = str(markdown.resolve())
    return written


__all__ = [
    "AUTHORITY_POLICY_ID",
    "AuthorityClassification",
    "AuthorityLabel",
    "AuthoritySource",
    "CapabilityBoundaryReceipt",
    "ComparatorInputOrigin",
    "DynamicFlatPromotionEvidence",
    "STAGE2_MISSING_DRIVE_NOISE_STATUS",
    "authority_capability_matrix",
    "authority_comparator_labels",
    "authority_policy_document",
    "classify_dynamic_flat_authority",
    "emit_authority_policy_evidence",
    "require_authority_label",
    "validate_dynamic_flat_promotion",
]
