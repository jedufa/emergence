"""Canonical EIEL production Baseline composition and conformance receipts.

Baseline is observational.  It composes the existing Full Evolution, runtime,
provider, WRC variability, tick-native, checkpoint/ACS, lifecycle, preflight,
and Canopy authorities.  It never enables a capability and it never grants
qualification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

# Transitional G4 re-export for QRR's existing Baseline dependency.  G7 owns
# physical dependency extraction; this avoids adding or ratcheting a new legacy
# inter-family import edge during admission-policy convergence.
from .admission_kernel_v1 import (
    baseline_admission_shadow_v1 as baseline_admission_shadow_v1,
)
from .artifact_lifecycle import (
    ArtifactLifecycleError,
    validate_run_artifact_lifecycle_contract_v1,
)
from .canonical import canonical_sha256
from .canopy_supervision_v1 import DEFAULT_CANOPY_RELAY_ROOT
from .orchestration import atomic_write_json
from .production_engine_identity import (
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
)
from .run_contract_v1 import (
    CAPABILITIES as FULL_EVOLUTION_CAPABILITIES,
    RunContractAdmissionError,
    checkpoint_contract_binding_v1,
)
from .wrc_variability import (
    INVARIANT_NAME as WRC_VARIABILITY_INVARIANT_NAME,
    WRCVariabilityEnvelopeError,
    validate_wrc_parameter_variability_envelope_v1,
)


BASELINE_PROFILE_ID: Final[str] = "EIEL_PRODUCTION_BASELINE_V1"
BASELINE_HUMAN_NAME: Final[str] = "Baseline"
BASELINE_PROFILE_VERSION: Final[int] = 2
BASELINE_PROFILE_SCHEMA_VERSION: Final[str] = (
    "eiel_production_baseline_profile_v1"
)
BASELINE_DECLARATION_SCHEMA_VERSION: Final[str] = "eiel_baseline_declaration_v1"
BASELINE_ADMISSION_SCHEMA_VERSION: Final[str] = (
    "eiel_baseline_admission_receipt_v1"
)
BASELINE_CONFORMANCE_SCHEMA_VERSION: Final[str] = (
    "eiel_baseline_conformance_receipt_v1"
)
BASELINE_INTENTS: Final[tuple[str, ...]] = (
    "production",
    "qualification",
    "development",
    "control",
    "ablation",
)
CONFIGURATION_STATUSES: Final[tuple[str, ...]] = (
    "CONFORMANT",
    "VALID_DECLARED_DEVIATION",
    "UNDECLARED_DEVIATION",
    "INVALID",
)
QUALIFICATION_STATUSES: Final[tuple[str, ...]] = (
    "QUALIFIED",
    "PARTIAL",
    "UNQUALIFIED",
    "NOT_EVALUATED",
    "NOT_APPLICABLE",
)
OPERATIONAL_STATUSES: Final[tuple[str, ...]] = (
    "CONFORMANT",
    "PENDING",
    "DEGRADED",
    "NOT_APPLICABLE",
)
OVERALL_STATUSES: Final[tuple[str, ...]] = (
    "BASELINE_QUALIFIED",
    "BASELINE_CONFORMANT",
    "VALID_DECLARED_DEVIATION",
    "NOT_BASELINE_CONFORMANT",
)

_PROFILE_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "_spec/execution_profiles/v1/eiel_production_baseline_v1.yaml"
)
_PROFILE_REPOSITORY_PATH: Final[str] = (
    "_spec/execution_profiles/v1/eiel_production_baseline_v1.yaml"
)
_REQUIREMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"domain", "classification", "resolver", "authority", "qualification"}
)
_PROFILE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "profile_id", "human_name", "version", "requirements"}
)
_SEMANTIC_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    {"REQUIRED", "REQUIRED_MODE", "FORBIDDEN", "OPERATIONAL"}
)
_ROUTE_QUALIFICATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "authority_state",
        "execution_route_sha256",
        "qualification_receipt_sha256",
        "provider_binding_exact",
        "profile_binding_exact",
        "route_hash_exact",
        "artifact_conformance_exact",
        "variability_envelope_exact",
        "covered_obligations",
        "required_obligations",
        "missing_obligations",
    }
)


class BaselineError(ValueError):
    """Baseline profile, declaration, or conformance input is malformed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise BaselineError(f"Baseline YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _exact(value: Mapping[str, Any], allowed: set[str] | frozenset[str], *, path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise BaselineError(f"{path} contains unsupported fields: {unknown}")


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BaselineError(f"{path} must be a non-empty string")
    return value.strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _hash_or_none(value: Any) -> str | None:
    return canonical_sha256(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class BaselineDeviationV1:
    requirement_id: str
    reason: str

    def __post_init__(self) -> None:
        if _text(
            self.requirement_id, path="baseline.deviation.requirement_id"
        ) != self.requirement_id:
            raise BaselineError(
                "baseline.deviation.requirement_id must not contain outer whitespace"
            )
        if _text(self.reason, path="baseline.deviation.reason") != self.reason:
            raise BaselineError(
                "baseline.deviation.reason must not contain outer whitespace"
            )

    def payload(self) -> dict[str, str]:
        return {"requirement_id": self.requirement_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class BaselineDeclarationV1:
    profile_id: str
    intent: str
    deviations: tuple[BaselineDeviationV1, ...]
    schema_version: str = BASELINE_DECLARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BASELINE_DECLARATION_SCHEMA_VERSION:
            raise BaselineError("unsupported baseline declaration schema")
        if self.profile_id != BASELINE_PROFILE_ID:
            raise BaselineError(
                f"baseline.profile_id must be {BASELINE_PROFILE_ID!r}"
            )
        if self.intent not in BASELINE_INTENTS:
            raise BaselineError(
                f"baseline.intent must be one of {BASELINE_INTENTS!r}"
            )
        if not isinstance(self.deviations, tuple) or any(
            not isinstance(row, BaselineDeviationV1) for row in self.deviations
        ):
            raise BaselineError("baseline.deviations must be canonical deviation rows")
        requirement_ids = [row.requirement_id for row in self.deviations]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise BaselineError("baseline deviation requirement ids must be unique")
        unknown = sorted(
            set(requirement_ids) - set(load_baseline_profile_v1()["requirements"])
        )
        if unknown:
            raise BaselineError(
                f"baseline deviation names unknown requirements {unknown!r}"
            )
        if self.intent == "production" and self.deviations:
            raise BaselineError("production Baseline intent forbids deviations")
        if self.intent in {"control", "ablation"} and not self.deviations:
            raise BaselineError(f"{self.intent} Baseline intent requires a deviation")

    @classmethod
    def from_value(cls, value: Any) -> "BaselineDeclarationV1":
        if not isinstance(value, Mapping):
            raise BaselineError("baseline must be an object")
        raw = dict(value)
        _exact(
            raw,
            {"schema_version", "profile_id", "intent", "deviations"},
            path="baseline",
        )
        schema = raw.get("schema_version", BASELINE_DECLARATION_SCHEMA_VERSION)
        if schema != BASELINE_DECLARATION_SCHEMA_VERSION:
            raise BaselineError("unsupported baseline declaration schema")
        profile_id = _text(raw.get("profile_id"), path="baseline.profile_id")
        if profile_id != BASELINE_PROFILE_ID:
            raise BaselineError(
                f"baseline.profile_id must be {BASELINE_PROFILE_ID!r}"
            )
        intent = _text(raw.get("intent"), path="baseline.intent")
        if intent not in BASELINE_INTENTS:
            raise BaselineError(f"baseline.intent must be one of {BASELINE_INTENTS!r}")
        rows = raw.get("deviations", [])
        if not isinstance(rows, list):
            raise BaselineError("baseline.deviations must be a list")
        deviations: list[BaselineDeviationV1] = []
        seen: set[str] = set()
        known = set(load_baseline_profile_v1()["requirements"])
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise BaselineError(f"baseline.deviations[{index}] must be an object")
            item = dict(row)
            _exact(
                item,
                {"requirement_id", "reason"},
                path=f"baseline.deviations[{index}]",
            )
            requirement_id = _text(
                item.get("requirement_id"),
                path=f"baseline.deviations[{index}].requirement_id",
            )
            reason = _text(
                item.get("reason"), path=f"baseline.deviations[{index}].reason"
            )
            if requirement_id not in known:
                raise BaselineError(
                    f"baseline deviation names unknown requirement {requirement_id!r}"
                )
            if requirement_id in seen:
                raise BaselineError(
                    f"baseline deviation duplicates requirement {requirement_id!r}"
                )
            seen.add(requirement_id)
            deviations.append(BaselineDeviationV1(requirement_id, reason))
        if intent == "production" and deviations:
            raise BaselineError("production Baseline intent forbids deviations")
        if intent in {"control", "ablation"} and not deviations:
            raise BaselineError(f"{intent} Baseline intent requires a deviation")
        return cls(profile_id, intent, tuple(deviations))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "intent": self.intent,
            "deviations": [row.payload() for row in self.deviations],
        }


def validate_baseline_purpose_v1(
    declaration: BaselineDeclarationV1, *, purpose: str
) -> None:
    qualification_purpose = purpose == "qualification"
    qualification_intent = declaration.intent == "qualification"
    if qualification_purpose != qualification_intent:
        raise BaselineError(
            "purpose='qualification' and baseline.intent='qualification' must agree"
        )


def load_baseline_profile_v1(path: str | Path | None = None) -> dict[str, Any]:
    profile_path = Path(path or _PROFILE_PATH).expanduser().resolve()
    if not profile_path.is_file() or profile_path.is_symlink():
        raise BaselineError(f"Baseline profile is not a regular file: {profile_path}")
    try:
        raw_value = yaml.load(
            profile_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BaselineError(f"cannot read Baseline profile: {exc}") from exc
    if not isinstance(raw_value, Mapping):
        raise BaselineError("Baseline profile must be an object")
    raw = dict(raw_value)
    _exact(raw, _PROFILE_FIELDS, path="baseline_profile")
    if (
        raw.get("schema_version") != BASELINE_PROFILE_SCHEMA_VERSION
        or raw.get("profile_id") != BASELINE_PROFILE_ID
        or raw.get("human_name") != BASELINE_HUMAN_NAME
        or raw.get("version") != BASELINE_PROFILE_VERSION
    ):
        raise BaselineError("Baseline profile identity is invalid")
    requirements = raw.get("requirements")
    if not isinstance(requirements, Mapping) or not requirements:
        raise BaselineError("Baseline profile requirements must be a non-empty object")
    normalized: dict[str, dict[str, Any]] = {}
    for requirement_id, value in requirements.items():
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            raise BaselineError("Baseline requirement ids must be non-empty strings")
        if not isinstance(value, Mapping):
            raise BaselineError(f"requirement {requirement_id!r} must be an object")
        row = dict(value)
        _exact(row, _REQUIREMENT_FIELDS, path=f"requirements.{requirement_id}")
        domain = _text(row.get("domain"), path=f"requirements.{requirement_id}.domain")
        classification = _text(
            row.get("classification"),
            path=f"requirements.{requirement_id}.classification",
        )
        if classification not in {
            "REQUIRED",
            "REQUIRED_MODE",
            "FORBIDDEN",
            "ADMISSION",
            "OPERATIONAL",
        }:
            raise BaselineError(
                f"requirements.{requirement_id}.classification is unsupported"
            )
        resolver = _text(
            row.get("resolver"), path=f"requirements.{requirement_id}.resolver"
        )
        authority = row.get("authority")
        qualification = row.get("qualification")
        if not isinstance(authority, Mapping) or not isinstance(
            qualification, Mapping
        ):
            raise BaselineError(
                f"requirements.{requirement_id} authority/qualification must be objects"
            )
        qualification_row = dict(qualification)
        _exact(
            qualification_row,
            {"mode", "required_obligations"},
            path=f"requirements.{requirement_id}.qualification",
        )
        mode = qualification_row.get("mode")
        if mode not in {"route_bound", "not_applicable"}:
            raise BaselineError(
                f"requirements.{requirement_id}.qualification.mode is unsupported"
            )
        obligations = qualification_row.get("required_obligations", [])
        if not isinstance(obligations, list) or any(
            not isinstance(item, str) or not item for item in obligations
        ):
            raise BaselineError(
                f"requirements.{requirement_id}.qualification.required_obligations is malformed"
            )
        normalized[str(requirement_id)] = {
            "domain": domain,
            "classification": classification,
            "resolver": resolver,
            "authority": dict(authority),
            "qualification": {
                "mode": mode,
                "required_obligations": list(obligations),
            },
        }
    core = {
        "schema_version": raw["schema_version"],
        "profile_id": raw["profile_id"],
        "human_name": raw["human_name"],
        "version": raw["version"],
        "requirements": normalized,
    }
    return {
        **core,
        "profile_path": str(profile_path),
        "profile_sha256": canonical_sha256(core),
    }


def _route_row(preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(preflight, Mapping):
        return {}
    return _mapping(preflight.get("execution_route"))


def _route_qualified(
    preflight: Mapping[str, Any] | None,
    *,
    required_obligations: Sequence[str],
) -> bool | None:
    if not isinstance(preflight, Mapping):
        return None
    route = _route_row(preflight)
    if route.get("authority_state") != "route_qualified":
        return False
    for name in (
        "provider_binding_exact",
        "profile_binding_exact",
        "route_hash_exact",
        "artifact_conformance_exact",
        "variability_envelope_exact",
    ):
        if route.get(name) is not True:
            return False
    if not isinstance(route.get("execution_route_sha256"), str) or not isinstance(
        route.get("qualification_receipt_sha256"), str
    ):
        return False
    missing = route.get("missing_obligations")
    if not isinstance(missing, list) or missing:
        return False
    covered = route.get("covered_obligations")
    route_required = route.get("required_obligations")
    if not isinstance(covered, list) or not isinstance(route_required, list):
        return False
    required = set(required_obligations)
    return required.issubset({str(value) for value in covered}) and required.issubset(
        {str(value) for value in route_required}
    )


def _provider_identity(
    plan: Mapping[str, Any], preflight: Mapping[str, Any] | None
) -> tuple[str | None, str | None, str | None]:
    if isinstance(preflight, Mapping):
        provider = _mapping(preflight.get("execution_provider"))
        profile = _mapping(preflight.get("profile"))
        qualification_reuse = _mapping(preflight.get("qualification_reuse"))
        identity = _mapping(qualification_reuse.get("identity_binding"))
        mechanics = _mapping(identity.get("mechanics"))
        return (
            str(profile["profile_id"])
            if isinstance(profile.get("profile_id"), str)
            else None,
            str(provider["provider_id"])
            if isinstance(provider.get("provider_id"), str)
            else None,
            str(mechanics["backend"])
            if isinstance(mechanics.get("backend"), str)
            else None,
        )
    profile_id = plan.get("resolved_profile")
    registered_profiles = {
        PRODUCTION_PROFILE_ID: "cuda_resident",
        "eiel-production-runtime-v1-cpu-native": "cpu_native",
    }
    provider_id = (
        PRODUCTION_PROVIDER_ID if profile_id in registered_profiles else None
    )
    backend = registered_profiles.get(profile_id)
    return (
        str(profile_id) if isinstance(profile_id, str) else None,
        provider_id,
        backend,
    )


def _canopy_supervision_binding_exact_v1(
    *,
    plan: Mapping[str, Any],
    supervision: Mapping[str, Any],
    expected: Mapping[str, Any],
    front_door_id: str,
) -> bool:
    supplied = supervision.get("receipt_sha256")
    core = dict(supervision)
    core.pop("receipt_sha256", None)
    output = _mapping(plan.get("output"))
    runtime_root = output.get("runtime_root")
    cells = plan.get("cells")
    relays = supervision.get("relays")
    group_id = supervision.get("relay_group_id")
    group_root_value = supervision.get("relay_group_root")
    if (
        not isinstance(supplied, str)
        or supplied != canonical_sha256(core)
        or supervision.get("schema_version") != expected.get("schema_version")
        or supervision.get("authority") != expected.get("authority")
        or supervision.get("simulation_state_mutation") is not False
        or supervision.get("plan_sha256") != plan.get("plan_sha256")
        or not isinstance(runtime_root, str)
        or supervision.get("run_root")
        != str(Path(runtime_root).expanduser().resolve())
        or not isinstance(cells, list)
        or not isinstance(relays, list)
        or not isinstance(group_id, str)
        or not isinstance(group_root_value, str)
    ):
        return False
    planned_cell_ids = [
        row.get("cell_id") if isinstance(row, Mapping) else None for row in cells
    ]
    relay_cell_ids = [
        row.get("cell_id") if isinstance(row, Mapping) else None for row in relays
    ]
    if (
        any(not isinstance(cell_id, str) for cell_id in planned_cell_ids)
        or relay_cell_ids != planned_cell_ids
    ):
        return False
    relay_binding_sha256 = canonical_sha256(
        {
            "plan_sha256": plan.get("plan_sha256"),
            "run_root": str(Path(runtime_root).expanduser().resolve()),
        }
    )
    prefix_by_front_door = {
        "eiel_execution_request_v2": "launchpad",
        "eiel_production_science_request_v2": "science",
    }
    expected_prefix = prefix_by_front_door.get(front_door_id)
    if (
        expected_prefix is None
        or group_id != f"{expected_prefix}-{relay_binding_sha256[:20]}"
    ):
        return False
    group_root = Path(group_root_value).expanduser().resolve()
    if group_root != DEFAULT_CANOPY_RELAY_ROOT.resolve() / group_id:
        return False
    for cell_id, relay in zip(planned_cell_ids, relays, strict=True):
        assert isinstance(cell_id, str) and isinstance(relay, Mapping)
        output_root = group_root / cell_id
        if (
            relay.get("output_root") != str(output_root)
            or relay.get("manifest_path")
            != str(output_root / "relay_manifest.json")
        ):
            return False
    return True


def _resolve_requirement(
    requirement_id: str,
    definition: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
    front_door_id: str,
    artifact_lifecycle_explicit: bool,
    canopy_enabled: bool,
    canopy_supervision: Mapping[str, Any] | None,
    stage: str,
) -> dict[str, Any]:
    resolver = str(definition["resolver"])
    expected = dict(definition["authority"])
    execution = _mapping(plan.get("execution"))
    route = _route_row(preflight)
    observed: dict[str, Any]
    conformant: bool
    operational = "NOT_APPLICABLE"

    if resolver == "strict_run_contract":
        contract = _mapping(plan.get("run_contract"))
        requested = _mapping(contract.get("requested"))
        try:
            contract_binding = checkpoint_contract_binding_v1(plan)
        except (RunContractAdmissionError, TypeError, ValueError):
            contract_binding = None
        capability_rows = _mapping(contract.get("capabilities"))
        exact_capabilities = (
            set(capability_rows) == set(FULL_EVOLUTION_CAPABILITIES)
            and all(
                _mapping(capability_rows.get(capability)).get("supported") is True
                and _mapping(capability_rows.get(capability)).get("configured")
                is True
                and _mapping(capability_rows.get(capability)).get("initializable")
                is True
                for capability in FULL_EVOLUTION_CAPABILITIES
            )
        )
        observed = {
            "profile_id": requested.get("profile_id"),
            "mode": requested.get("mode"),
            "admission_status": contract.get("admission_status"),
            "failures": contract.get("failures"),
            "contract_hash": contract.get("contract_hash"),
            "required_capabilities": contract.get("required_capabilities"),
        }
        conformant = (
            requested.get("profile_id") == expected.get("profile_id")
            and requested.get("mode") == expected.get("mode")
            and contract.get("admission_status") == "passed"
            and contract.get("failures") == []
            and isinstance(contract_binding, Mapping)
            and contract_binding.get("profile_id") == expected.get("profile_id")
            and contract_binding.get("required_capabilities")
            == list(FULL_EVOLUTION_CAPABILITIES)
            and exact_capabilities
        )
        fingerprint_input = {
            "requested": requested,
            "required_capabilities": contract.get("required_capabilities"),
            "capabilities": contract.get("capabilities"),
        }
        authority_sha256 = contract.get("contract_hash")
    elif resolver == "production_cuda_runtime":
        runtime_id, provider_id, backend = _provider_identity(plan, preflight)
        observed = {
            "runtime_profile_id": runtime_id,
            "provider_id": provider_id,
            "backend": backend,
        }
        conformant = observed == expected
        fingerprint_input = observed
        authority_sha256 = route.get("execution_route_sha256") or _hash_or_none(observed)
    elif resolver == "wrc_variability_envelope":
        qualification_key = _mapping(plan.get("qualification_key"))
        raw_envelope = qualification_key.get("variability_envelope")
        fixed_invariants = _mapping(qualification_key.get("fixed_invariants"))
        try:
            envelope = validate_wrc_parameter_variability_envelope_v1(
                raw_envelope if isinstance(raw_envelope, Mapping) else {}
            )
        except (WRCVariabilityEnvelopeError, TypeError, ValueError):
            envelope = None
        observed = {
            "schema_version": (
                raw_envelope.get("schema_version")
                if isinstance(raw_envelope, Mapping)
                else None
            ),
            "envelope_id": envelope.envelope_id if envelope is not None else None,
            "envelope_sha256": (
                envelope.envelope_sha256 if envelope is not None else None
            ),
            "profile_id": envelope.profile_id if envelope is not None else None,
            "backend": envelope.backend if envelope is not None else None,
            "world": (
                {
                    "base": envelope.world_base,
                    "overlays": list(envelope.world_overlays),
                }
                if envelope is not None
                else None
            ),
            "qualification_invariant_sha256": fixed_invariants.get(
                WRC_VARIABILITY_INVARIANT_NAME
            ),
        }
        expected_observed = {
            **expected,
            "qualification_invariant_sha256": expected.get("envelope_sha256"),
        }
        conformant = envelope is not None and observed == expected_observed
        fingerprint_input = {
            "variability_envelope": raw_envelope,
            "qualification_invariant_sha256": fixed_invariants.get(
                WRC_VARIABILITY_INVARIANT_NAME
            ),
        }
        authority_sha256 = (
            envelope.envelope_sha256
            if envelope is not None
            else _hash_or_none(raw_envelope)
        )
    elif resolver == "absolute_native_tick":
        observed = {
            "execution_route_identity": execution.get("execution_route_identity"),
            "time_authority": execution.get("time_authority"),
        }
        conformant = observed == expected
        fingerprint_input = {
            **observed,
            "tick_native_control": execution.get("tick_native_control"),
        }
        authority_sha256 = route.get("execution_route_sha256") or _hash_or_none(
            fingerprint_input
        )
    elif resolver == "native_streamed_checkpoint":
        observed = {
            "native_streamed_checkpoint": execution.get(
                "native_streamed_checkpoint", False
            )
        }
        conformant = observed == expected
        fingerprint_input = {
            **observed,
            "checkpoint_storage_mode": execution.get("checkpoint_storage_mode"),
            "checkpoint_ticks": execution.get("checkpoint_ticks"),
        }
        authority_sha256 = route.get("execution_route_sha256") or _hash_or_none(
            fingerprint_input
        )
    elif resolver == "acs_two_slot":
        observed = {
            "acs_checkpoint_slots": execution.get("acs_checkpoint_slots", 0),
            "persistence_mode": (
                "bounded_async_single_writer_v1"
                if execution.get("acs_checkpoint_slots", 0)
                else "synchronous_v1"
            ),
        }
        conformant = observed == expected
        fingerprint_input = observed
        authority_sha256 = route.get("execution_route_sha256") or _hash_or_none(observed)
    elif resolver == "artifact_lifecycle":
        raw = plan.get("artifact_lifecycle")
        try:
            lifecycle = validate_run_artifact_lifecycle_contract_v1(raw)
            lifecycle_valid = True
        except (ArtifactLifecycleError, TypeError, ValueError):
            lifecycle = {}
            lifecycle_valid = False
        observed = {
            "schema_version": lifecycle.get("schema_version"),
            "source": (
                "explicit_plan_bound"
                if artifact_lifecycle_explicit
                else "legacy_derived"
            ),
            "contract_sha256": lifecycle.get("contract_sha256"),
        }
        conformant = (
            lifecycle_valid
            and artifact_lifecycle_explicit
            and lifecycle.get("schema_version") == expected.get("schema_version")
        )
        fingerprint_input = lifecycle or raw
        authority_sha256 = lifecycle.get("contract_sha256")
    elif resolver == "canonical_front_door":
        observed = {"front_door_id": front_door_id}
        conformant = front_door_id in expected.get("admitted_front_doors", [])
        fingerprint_input = observed
        authority_sha256 = _hash_or_none(observed)
    elif resolver == "resource_preflight":
        if preflight is None:
            observed = {"state": "PENDING"}
            conformant = True
        else:
            observed = {
                "schema_version": preflight.get("schema_version"),
                "plan_sha256": preflight.get("plan_sha256"),
                "ready_for_execution": preflight.get("ready_for_execution"),
            }
            conformant = (
                preflight.get("schema_version") == expected.get("schema_version")
                and preflight.get("plan_sha256") == plan.get("plan_sha256")
                and preflight.get("ready_for_execution") is True
            )
        fingerprint_input = observed
        authority_sha256 = _hash_or_none(preflight) if preflight is not None else None
    elif resolver == "canopy_supervision":
        observed = {"enabled": canopy_enabled}
        conformant = canopy_enabled is expected.get("enabled")
        if not canopy_enabled:
            operational = "NOT_APPLICABLE"
        elif canopy_supervision is None:
            operational = "PENDING"
        else:
            state = canopy_supervision.get("state")
            binding_exact = _canopy_supervision_binding_exact_v1(
                plan=plan,
                supervision=canopy_supervision,
                expected=expected,
                front_door_id=front_door_id,
            )
            observed.update(
                {
                    "schema_version": canopy_supervision.get("schema_version"),
                    "state": state,
                    "authority": canopy_supervision.get("authority"),
                    "binding_exact": binding_exact,
                }
            )
            if (
                state == "watching"
                and binding_exact
            ):
                operational = "CONFORMANT"
            else:
                operational = "DEGRADED"
        fingerprint_input = {
            "enabled": canopy_enabled,
            "supervision": canopy_supervision,
        }
        authority_sha256 = (
            canopy_supervision.get("receipt_sha256")
            if isinstance(canopy_supervision, Mapping)
            else None
        )
    elif resolver == "generation_authority_forbidden":
        native = execution.get("execution_route_identity") == (
            "tick_native_execution_route_v1"
        )
        leaked = sorted(
            set(execution)
            & {
                "generations",
                "ticks_per_generation",
                "physical_window_generations",
            }
        )
        observed = {"active": not native or bool(leaked), "public_fields": leaked}
        conformant = observed["active"] is False
        fingerprint_input = observed
        authority_sha256 = _hash_or_none(observed)
    elif resolver == "device_fallback_forbidden":
        runtime_id, provider_id, backend = _provider_identity(plan, preflight)
        fallback_allowed = not (
            provider_id == PRODUCTION_PROVIDER_ID
            and runtime_id in {PRODUCTION_PROFILE_ID, "eiel-production-runtime-v1-cpu-native"}
            and backend in {"cuda_resident", "cpu_native"}
        )
        observed = {"fallback_allowed": fallback_allowed}
        conformant = observed == expected
        fingerprint_input = {
            **observed,
            "provider_id": provider_id,
            "runtime_profile_id": runtime_id,
            "backend": backend,
        }
        authority_sha256 = route.get("execution_route_sha256") or _hash_or_none(
            fingerprint_input
        )
    else:
        raise BaselineError(
            f"Baseline requirement {requirement_id!r} has unknown resolver {resolver!r}"
        )

    qualification_definition = _mapping(definition.get("qualification"))
    qualification_mode = qualification_definition.get("mode")
    if qualification_mode == "not_applicable":
        qualification_status = "NOT_APPLICABLE"
    elif not conformant:
        qualification_status = "UNQUALIFIED"
    else:
        route_status = _route_qualified(
            preflight,
            required_obligations=qualification_definition.get(
                "required_obligations", []
            ),
        )
        qualification_status = (
            "NOT_EVALUATED"
            if route_status is None
            else ("QUALIFIED" if route_status else "UNQUALIFIED")
        )
    qualification_binding: dict[str, Any] = {
        "mode": qualification_mode,
        "required_obligations": sorted(
            qualification_definition.get("required_obligations", [])
        ),
        "status": qualification_status,
    }
    if qualification_mode == "route_bound":
        qualification_binding["route"] = {
            key: route.get(key) for key in sorted(_ROUTE_QUALIFICATION_FIELDS)
        }
    requirement_definition_sha256 = canonical_sha256(
        {"requirement_id": requirement_id, **dict(definition)}
    )
    impact_fingerprint = canonical_sha256(
        {
            "requirement_definition_sha256": requirement_definition_sha256,
            "observed_binding": fingerprint_input,
            "configuration_conformant": conformant,
        }
    )
    qualification_binding["configuration_fingerprint"] = impact_fingerprint
    return {
        "requirement_id": requirement_id,
        "domain": definition["domain"],
        "classification": definition["classification"],
        "resolver": resolver,
        "expected": expected,
        "observed": observed,
        "configuration_conformant": conformant,
        "qualification_status": qualification_status,
        "operational_status": operational,
        "authority_sha256": authority_sha256,
        "requirement_definition_sha256": requirement_definition_sha256,
        "impact_fingerprint": impact_fingerprint,
        "qualification_fingerprint": canonical_sha256(qualification_binding),
        "stage": stage,
    }


def _configuration_decision(
    rows: Sequence[Mapping[str, Any]], declaration: BaselineDeclarationV1
) -> tuple[str, list[dict[str, Any]], list[str]]:
    mismatches = {
        str(row["requirement_id"])
        for row in rows
        if row.get("classification") in _SEMANTIC_CLASSIFICATIONS
        and row.get("configuration_conformant") is not True
    }
    declared = {row.requirement_id: row for row in declaration.deviations}
    undeclared = sorted(mismatches - set(declared))
    unused = sorted(set(declared) - mismatches)
    reasons: list[str] = []
    if undeclared:
        reasons.append("UNDECLARED_BASELINE_DEVIATION")
    if unused:
        reasons.append("UNUSED_BASELINE_DEVIATION")
    if declaration.intent == "production" and mismatches:
        reasons.append("PRODUCTION_BASELINE_DEVIATION_FORBIDDEN")
    if declaration.intent in {"control", "ablation"} and not mismatches:
        reasons.append("BASELINE_DEVIATION_REQUIRED")
    if unused or (declaration.intent == "production" and mismatches) or (
        declaration.intent in {"control", "ablation"} and not mismatches
    ):
        status = "INVALID"
    elif undeclared:
        status = "UNDECLARED_DEVIATION"
    elif mismatches:
        status = "VALID_DECLARED_DEVIATION"
    else:
        status = "CONFORMANT"
    deviation_rows = [
        {
            **declared[requirement_id].payload(),
            "expected": next(
                row.get("expected")
                for row in rows
                if row.get("requirement_id") == requirement_id
            ),
            "observed": next(
                row.get("observed")
                for row in rows
                if row.get("requirement_id") == requirement_id
            ),
            "status": "DECLARED",
        }
        for requirement_id in sorted(mismatches & set(declared))
    ]
    deviation_rows.extend(
        {
            "requirement_id": requirement_id,
            "reason": None,
            "expected": next(
                row.get("expected")
                for row in rows
                if row.get("requirement_id") == requirement_id
            ),
            "observed": next(
                row.get("observed")
                for row in rows
                if row.get("requirement_id") == requirement_id
            ),
            "status": "UNDECLARED",
        }
        for requirement_id in undeclared
    )
    deviation_rows.extend(
        {
            **declared[requirement_id].payload(),
            "expected": next(
                row.get("expected")
                for row in rows
                if row.get("requirement_id") == requirement_id
            ),
            "observed": next(
                row.get("observed")
                for row in rows
                if row.get("requirement_id") == requirement_id
            ),
            "status": "UNUSED",
        }
        for requirement_id in unused
    )
    return status, deviation_rows, reasons


def _qualification_decision(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [
        str(row.get("qualification_status"))
        for row in rows
        if str(row.get("qualification_status")) != "NOT_APPLICABLE"
    ]
    if not statuses:
        return "NOT_APPLICABLE"
    if all(status == "NOT_EVALUATED" for status in statuses):
        return "NOT_EVALUATED"
    if all(status == "QUALIFIED" for status in statuses):
        return "QUALIFIED"
    if any(status == "QUALIFIED" for status in statuses):
        return "PARTIAL"
    return "UNQUALIFIED"


def _operational_decision(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [
        str(row.get("operational_status"))
        for row in rows
        if row.get("classification") == "OPERATIONAL"
    ]
    if not statuses or all(status == "NOT_APPLICABLE" for status in statuses):
        return "NOT_APPLICABLE"
    if any(status == "DEGRADED" for status in statuses):
        return "DEGRADED"
    if any(status == "PENDING" for status in statuses):
        return "PENDING"
    return "CONFORMANT"


def _overall_decision(
    *, configuration: str, qualification: str, operations: str
) -> str:
    if configuration == "VALID_DECLARED_DEVIATION":
        return "VALID_DECLARED_DEVIATION"
    if configuration != "CONFORMANT" or operations == "DEGRADED":
        return "NOT_BASELINE_CONFORMANT"
    if qualification == "QUALIFIED" and operations in {
        "CONFORMANT",
        "NOT_APPLICABLE",
    }:
        return "BASELINE_QUALIFIED"
    return "BASELINE_CONFORMANT"


def resolve_baseline_v1(
    *,
    declaration: BaselineDeclarationV1,
    plan: Mapping[str, Any],
    request_sha256: str,
    front_door_id: str,
    preflight: Mapping[str, Any] | None = None,
    artifact_lifecycle_explicit: bool = True,
    canopy_enabled: bool = True,
    canopy_supervision: Mapping[str, Any] | None = None,
    stage: str = "admission",
) -> dict[str, Any]:
    profile = load_baseline_profile_v1()
    rows = [
        _resolve_requirement(
            requirement_id,
            definition,
            plan=plan,
            preflight=preflight,
            front_door_id=front_door_id,
            artifact_lifecycle_explicit=artifact_lifecycle_explicit,
            canopy_enabled=canopy_enabled,
            canopy_supervision=canopy_supervision,
            stage=stage,
        )
        for requirement_id, definition in profile["requirements"].items()
    ]
    configuration, deviations, reason_codes = _configuration_decision(
        rows, declaration
    )
    qualification = _qualification_decision(rows)
    operations = _operational_decision(rows)
    overall = _overall_decision(
        configuration=configuration,
        qualification=qualification,
        operations=operations,
    )
    admission_row = next(
        row
        for row in rows
        if row["requirement_id"] == "provenance.resource_preflight"
    )
    execution_admission_status = (
        "PENDING"
        if preflight is None
        else (
            "ADMITTED"
            if admission_row["configuration_conformant"] is True
            else "REJECTED"
        )
    )
    configuration_eligible = configuration in {
        "CONFORMANT",
        "VALID_DECLARED_DEVIATION",
    }
    admitted = (
        configuration_eligible and execution_admission_status == "ADMITTED"
    )
    domains: dict[str, dict[str, Any]] = {}
    for row in rows:
        domain = str(row["domain"])
        domain_row = domains.setdefault(
            domain,
            {
                "authority_sha256s": [],
                "requirement_fingerprints": {},
                "requirement_qualification_fingerprints": {},
            },
        )
        if row.get("authority_sha256"):
            domain_row["authority_sha256s"].append(row["authority_sha256"])
        domain_row["requirement_fingerprints"][row["requirement_id"]] = row[
            "impact_fingerprint"
        ]
        domain_row["requirement_qualification_fingerprints"][
            row["requirement_id"]
        ] = row["qualification_fingerprint"]
    for domain, row in domains.items():
        row["authority_sha256s"] = sorted(set(row["authority_sha256s"]))
        row["configuration_fingerprint"] = canonical_sha256(
            row["requirement_fingerprints"]
        )
        row["qualification_fingerprint"] = canonical_sha256(
            row["requirement_qualification_fingerprints"]
        )
        row["authority_fingerprint"] = canonical_sha256(
            row["authority_sha256s"]
        )
        row["impact_fingerprint"] = row["configuration_fingerprint"]
        row["domain"] = domain
    return {
        "profile": {
            "profile_id": profile["profile_id"],
            "human_name": profile["human_name"],
            "version": profile["version"],
            "profile_sha256": profile["profile_sha256"],
            "profile_path": _PROFILE_REPOSITORY_PATH,
        },
        "declaration": declaration.payload(),
        "request_sha256": request_sha256,
        "plan_sha256": plan.get("plan_sha256"),
        "preflight_sha256": _hash_or_none(preflight),
        "front_door_id": front_door_id,
        "stage": stage,
        "configuration_status": configuration,
        "qualification_status": qualification,
        "operational_status": operations,
        "overall_status": overall,
        "execution_admission_status": execution_admission_status,
        "configuration_eligible": configuration_eligible,
        "admitted": admitted,
        "reason_codes": reason_codes,
        "deviations": deviations,
        "requirements": rows,
        "domain_fingerprints": {
            name: domains[name] for name in sorted(domains)
        },
    }


def build_baseline_admission_receipt_v1(**kwargs: Any) -> dict[str, Any]:
    resolved = resolve_baseline_v1(stage="admission", **kwargs)
    core = {
        "schema_version": BASELINE_ADMISSION_SCHEMA_VERSION,
        **resolved,
        "created_at_utc": _now(),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def validate_baseline_receipt_binding_v1(
    receipt: Mapping[str, Any],
    *,
    schema_version: str,
    declaration: BaselineDeclarationV1,
    plan: Mapping[str, Any],
    request_sha256: str,
    front_door_id: str,
    preflight: Mapping[str, Any] | None,
    stage: str,
    artifact_lifecycle_explicit: bool,
    canopy_enabled: bool,
    canopy_supervision: Mapping[str, Any] | None = None,
    admission_receipt_sha256: str | None = None,
    execution_status: str | None = None,
) -> dict[str, Any]:
    """Verify an immutable Baseline receipt and its owning execution bindings."""

    if not isinstance(receipt, Mapping):
        raise BaselineError("Baseline receipt must be an object")
    value = dict(receipt)
    supplied = value.get("receipt_sha256")
    core = dict(value)
    core.pop("receipt_sha256", None)
    if not isinstance(supplied, str) or supplied != canonical_sha256(core):
        raise BaselineError("Baseline receipt SHA-256 mismatch")
    if schema_version not in {
        BASELINE_ADMISSION_SCHEMA_VERSION,
        BASELINE_CONFORMANCE_SCHEMA_VERSION,
    } or value.get("schema_version") != schema_version:
        raise BaselineError("Baseline receipt schema mismatch")
    expected_stage = (
        "admission"
        if schema_version == BASELINE_ADMISSION_SCHEMA_VERSION
        else "terminal"
    )
    if stage != expected_stage:
        raise BaselineError("Baseline receipt stage does not match its schema")
    profile = load_baseline_profile_v1()
    observed_profile = _mapping(value.get("profile"))
    expected_profile = {
        "profile_id": profile["profile_id"],
        "human_name": profile["human_name"],
        "version": profile["version"],
        "profile_sha256": profile["profile_sha256"],
    }
    if any(
        observed_profile.get(key) != item
        for key, item in expected_profile.items()
    ):
        raise BaselineError("Baseline receipt profile binding mismatch")
    expected_bindings = {
        "declaration": declaration.payload(),
        "request_sha256": request_sha256,
        "plan_sha256": plan.get("plan_sha256"),
        "preflight_sha256": _hash_or_none(preflight),
        "front_door_id": front_door_id,
        "stage": stage,
    }
    mismatches = sorted(
        key for key, item in expected_bindings.items() if value.get(key) != item
    )
    if mismatches:
        raise BaselineError(
            "Baseline receipt execution binding mismatch: " + ", ".join(mismatches)
        )
    expected_resolution = resolve_baseline_v1(
        declaration=declaration,
        plan=plan,
        request_sha256=request_sha256,
        front_door_id=front_door_id,
        preflight=preflight,
        artifact_lifecycle_explicit=artifact_lifecycle_explicit,
        canopy_enabled=canopy_enabled,
        canopy_supervision=(
            canopy_supervision if stage == "terminal" else None
        ),
        stage=stage,
    )
    semantic_mismatches = sorted(
        key
        for key, item in expected_resolution.items()
        if value.get(key) != item
    )
    if semantic_mismatches:
        raise BaselineError(
            "Baseline receipt semantic mismatch: "
            + ", ".join(semantic_mismatches)
        )
    if schema_version == BASELINE_CONFORMANCE_SCHEMA_VERSION:
        if (
            not isinstance(admission_receipt_sha256, str)
            or value.get("admission_receipt_sha256")
            != admission_receipt_sha256
        ):
            raise BaselineError(
                "Baseline conformance receipt admission binding mismatch"
            )
        if value.get("execution_status") != execution_status:
            raise BaselineError(
                "Baseline conformance receipt execution status mismatch"
            )
    return value


def build_baseline_conformance_receipt_v1(
    *,
    admission_receipt: Mapping[str, Any],
    execution_status: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    validated_admission = validate_baseline_receipt_binding_v1(
        admission_receipt,
        schema_version=BASELINE_ADMISSION_SCHEMA_VERSION,
        declaration=kwargs["declaration"],
        plan=kwargs["plan"],
        request_sha256=kwargs["request_sha256"],
        front_door_id=kwargs["front_door_id"],
        preflight=kwargs.get("preflight"),
        stage="admission",
        artifact_lifecycle_explicit=kwargs.get(
            "artifact_lifecycle_explicit", True
        ),
        canopy_enabled=kwargs.get("canopy_enabled", True),
    )
    admission_sha256 = validated_admission["receipt_sha256"]
    resolved = resolve_baseline_v1(stage="terminal", **kwargs)
    core = {
        "schema_version": BASELINE_CONFORMANCE_SCHEMA_VERSION,
        **resolved,
        "admission_receipt_sha256": admission_sha256,
        "execution_status": execution_status,
        "evaluated_at_utc": _now(),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def write_baseline_receipt_v1(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    supplied = receipt.get("receipt_sha256")
    core = dict(receipt)
    core.pop("receipt_sha256", None)
    if supplied != canonical_sha256(core):
        raise BaselineError("Baseline receipt SHA-256 mismatch")
    atomic_write_json(destination, dict(receipt))
    return destination


def baseline_result_binding_v1(
    *,
    admission_receipt: Mapping[str, Any],
    conformance_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_receipt = conformance_receipt or admission_receipt
    profile = _mapping(effective_receipt.get("profile"))
    return {
        "profile_id": BASELINE_PROFILE_ID,
        "human_name": BASELINE_HUMAN_NAME,
        "profile_version": profile.get("version"),
        "profile_sha256": profile.get("profile_sha256"),
        "configuration_status": effective_receipt.get("configuration_status"),
        "qualification_status": effective_receipt.get("qualification_status"),
        "operational_status": effective_receipt.get("operational_status"),
        "overall_status": effective_receipt.get("overall_status"),
        "execution_admission_status": effective_receipt.get(
            "execution_admission_status"
        ),
        "configuration_eligible": effective_receipt.get("configuration_eligible"),
        "admitted": effective_receipt.get("admitted"),
        "deviations": list(effective_receipt.get("deviations", [])),
        "admission_receipt_sha256": admission_receipt.get("receipt_sha256"),
        "conformance_receipt_sha256": (
            conformance_receipt.get("receipt_sha256")
            if isinstance(conformance_receipt, Mapping)
            else None
        ),
    }


def render_baseline_human_v1(value: Mapping[str, Any]) -> str:
    lines = [
        f"{BASELINE_HUMAN_NAME} ({BASELINE_PROFILE_ID})",
        f"configuration: {value.get('configuration_status')}",
        f"qualification: {value.get('qualification_status')}",
        f"operations: {value.get('operational_status')}",
        f"execution_admission: {value.get('execution_admission_status')}",
        f"overall: {value.get('overall_status')}",
    ]
    for row in value.get("deviations", []):
        if isinstance(row, Mapping):
            lines.append(
                f"deviation: {row.get('requirement_id')} — {row.get('reason')}"
            )
    return "\n".join(lines)


__all__ = [
    "BASELINE_ADMISSION_SCHEMA_VERSION",
    "BASELINE_CONFORMANCE_SCHEMA_VERSION",
    "BASELINE_DECLARATION_SCHEMA_VERSION",
    "BASELINE_HUMAN_NAME",
    "BASELINE_INTENTS",
    "BASELINE_PROFILE_ID",
    "BASELINE_PROFILE_VERSION",
    "BaselineDeclarationV1",
    "BaselineDeviationV1",
    "BaselineError",
    "baseline_result_binding_v1",
    "build_baseline_admission_receipt_v1",
    "build_baseline_conformance_receipt_v1",
    "load_baseline_profile_v1",
    "render_baseline_human_v1",
    "resolve_baseline_v1",
    "validate_baseline_receipt_binding_v1",
    "validate_baseline_purpose_v1",
    "write_baseline_receipt_v1",
]
