"""Admission and diagnostic activation contracts for production campaigns.

This module is intentionally observational.  It resolves existing campaign,
component, lifecycle, and genome authorities; it never edits configuration or
participates in simulation mechanics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Final

from .canonical import canonical_bytes, canonical_sha256
from .production_engine_components import (
    ProductionEngineComponentBindingV1,
    production_engine_component_binding_v1,
)
from .production_engine_configuration import (
    ProductionEngineConfigurationV1,
    production_engine_configuration_v1,
)
from .production_engine_lifecycle_v1 import (
    ProductionEngineLifecycleControlV1,
    production_engine_lifecycle_control_v1,
)
from .providers import CampaignProviderError
from engine.evolution.global_genome_registry_v1 import global_genome_profile_v1

SCHEMA_VERSION: Final[str] = "eiel_resolved_run_contract_v1"
ACTIVATION_SCHEMA_VERSION: Final[str] = "eiel_run_contract_activation_report_v1"
CHECKPOINT_BINDING_SCHEMA_VERSION: Final[str] = (
    "eiel_run_contract_checkpoint_binding_v1"
)
PROFILE_ID: Final[str] = "eiel_full_evolution_v1"
CAPABILITIES: Final[tuple[str, ...]] = (
    "patch",
    "wdl_delayed_hazard",
    "jungle",
    "eed",
    "genome_v2_ordinary_birth",
    "genome_v2_mutation",
    "recombination_profile",
)
_RELEVANT_PREFIXES: Final[tuple[str, ...]] = (
    "patch.",
    "wdl.",
    "jungle.",
    "eed.",
    "genome_upgrade_v2.",
    "production_lifecycle.",
)


class RunContractAdmissionError(RuntimeError):
    """A strict requested profile does not match effective configuration."""

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        self.diagnostic = dict(diagnostic)
        failures = self.diagnostic.get("failures", [])
        compact = "; ".join(
            f"{row.get('capability')}: {row.get('reason')}"
            for row in failures
            if isinstance(row, Mapping)
        )
        super().__init__(
            "RUN CONTRACT ADMISSION: FAILED" + (f" ({compact})" if compact else "")
        )


@dataclass(frozen=True, slots=True)
class _Context:
    plan: Mapping[str, Any]
    configuration: ProductionEngineConfigurationV1
    binding: ProductionEngineComponentBindingV1
    lifecycle: ProductionEngineLifecycleControlV1 | None


Validator = Callable[[_Context], tuple[bool, str, Mapping[str, Any]]]


def _pass(reason: str, **evidence: Any) -> tuple[bool, str, Mapping[str, Any]]:
    return True, reason, evidence


def _fail(reason: str, **evidence: Any) -> tuple[bool, str, Mapping[str, Any]]:
    return False, reason, evidence


def _component_ids(context: _Context) -> set[str]:
    return {
        context.binding.world_base.component_id,
        *(row.component_id for row in context.binding.world_overlays),
        context.binding.genome_inheritance.component_id,
    }


def _patch(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    present = (
        context.configuration.world_base == "patch"
        and "world.patch.v1" in _component_ids(context)
        and context.binding.world_base.production_ready
    )
    if not present:
        return _fail(
            "effective world base is not the production Patch component",
            world_base=context.configuration.world_base,
            component_id=context.binding.world_base.component_id,
        )
    return _pass(
        "production Patch world component resolved",
        component_id="world.patch.v1",
        contract_version=context.binding.world_base.contract_version,
    )


def _overlay(
    context: _Context,
    *,
    overlay_id: str,
    component_id: str,
    label: str,
) -> tuple[bool, str, Mapping[str, Any]]:
    patch_ok, _, _ = _patch(context)
    if not patch_ok:
        return _fail(f"{label} requires the Patch world base")
    components = {row.component_id: row for row in context.binding.world_overlays}
    component = components.get(component_id)
    if overlay_id not in context.configuration.world_overlays or component is None:
        return _fail(
            f"effective configuration lacks the {overlay_id} overlay",
            resolved_overlays=list(context.configuration.world_overlays),
        )
    if not component.production_ready:
        return _fail(f"{label} component is not production ready")
    return _pass(
        f"production {label} overlay resolved",
        component_id=component.component_id,
        contract_version=component.contract_version,
    )


def _wdl(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    return _overlay(
        context,
        overlay_id="wdl_delayed_hazard",
        component_id="world.overlay.wdl_delayed_hazard.v1",
        label="WDL delayed-hazard",
    )


def _jungle(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    return _overlay(
        context,
        overlay_id="jungle",
        component_id="world.overlay.jungle.v1",
        label="Jungle",
    )


def _eed(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    ok, reason, evidence = _overlay(
        context,
        overlay_id="eed_resource_impact",
        component_id="world.overlay.eed_resource_impact.v1",
        label="EED resource-impact",
    )
    if not ok:
        return ok, reason, evidence
    if context.configuration.backend == "cpu_native":
        return _fail(
            "EED is not initializable on the current production CPU route: "
            "that route disables respec0b_resource_specialization_enable, "
            "which eed_resource_impact_enable requires",
            backend=context.configuration.backend,
            dependency="respec0b_resource_specialization_enable",
        )
    return _pass(
        reason,
        **dict(evidence),
        classification="simulation_mechanics_with_read_only_reporting_projection",
    )


def _genome_v2(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    component = context.binding.genome_inheritance
    if (
        component.component_id != "genome.genome_upgrade_v2.v1"
        or component.configuration_id != "genome_upgrade_v2"
        or not component.production_ready
    ):
        return _fail(
            "effective genome component is not Genome V2",
            component_id=component.component_id,
            configuration_id=component.configuration_id,
        )
    profile = global_genome_profile_v1("genome_upgrade_v2")
    return _pass(
        "production Genome V2 component resolved",
        component_id=component.component_id,
        component_contract=component.contract_version,
        inheritance_contract=profile.inheritance_contract,
        mutation_contract=profile.mutation_contract,
    )


def _ordinary_birth(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    genome_ok, _, genome_evidence = _genome_v2(context)
    if not genome_ok:
        return _fail("Genome V2 ordinary birth requires Genome V2")
    if context.lifecycle is None:
        return _fail("Genome V2 ordinary birth requires production lifecycle control")
    if not context.lifecycle.ordinary_births_enabled:
        return _fail("production_lifecycle.ordinary_births_enabled is false")
    return _pass(
        "Genome V2 ordinary-birth route is admitted",
        **dict(genome_evidence),
        lifecycle_schema=context.lifecycle.payload()["schema_version"],
        lifecycle_mode=context.lifecycle.lifecycle_mode,
    )


def _mutation(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    birth_ok, _, birth_evidence = _ordinary_birth(context)
    if not birth_ok:
        return _fail("Genome V2 mutation requires the Genome V2 ordinary-birth route")
    assert context.lifecycle is not None
    if not context.lifecycle.genome_v2_mutation_enabled:
        return _fail(
            "Genome V2 mutation is ineffective because no positive admitted rate/step regime resolved",
            admitted_regimes_q10=[
                {"rate_q10": rate, "step_q10": step}
                for rate, step in context.lifecycle.genome_v2_mutation_regimes_q10
            ],
        )
    return _pass(
        "Genome V2 fixed-Q10 mutation operator is enabled",
        **dict(birth_evidence),
        admitted_regimes_q10=[
            {"rate_q10": rate, "step_q10": step}
            for rate, step in context.lifecycle.genome_v2_mutation_regimes_q10
        ],
    )


def _recombination(context: _Context) -> tuple[bool, str, Mapping[str, Any]]:
    birth_ok, _, birth_evidence = _ordinary_birth(context)
    if not birth_ok:
        return _fail(
            "Genome V2 recombination requires the Genome V2 ordinary-birth route"
        )
    inheritance = global_genome_profile_v1("genome_upgrade_v2").inheritance_contract
    if inheritance != "genome_upgrade_v2_linked_diploid_tape_v1":
        return _fail("Genome V2 linked-diploid recombination contract is unavailable")
    return _pass(
        "fixed Genome V2 linked-product recombination profile resolved",
        **dict(birth_evidence),
        recombination_profile=inheritance,
        linkage_blocks=6,
        products_per_linkage_block=2,
        meiosis_draw_count=12,
    )


_VALIDATORS: Final[dict[str, Validator]] = {
    "patch": _patch,
    "wdl_delayed_hazard": _wdl,
    "jungle": _jungle,
    "eed": _eed,
    "genome_v2_ordinary_birth": _ordinary_birth,
    "genome_v2_mutation": _mutation,
    "recombination_profile": _recombination,
}
_PROFILES: Final[dict[str, tuple[str, ...]]] = {PROFILE_ID: CAPABILITIES}
_HASH_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "requested",
    "required_capabilities",
    "capabilities",
    "resolved_relevant_config",
    "source_identity",
    "device",
    "qualification_key_sha256",
)
_GIT_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")


def _source_identity() -> dict[str, str]:
    override = os.environ.get("EIEL_SOURCE_GIT_COMMIT", "").strip()
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        if override:
            return {"git_commit": override, "discovery": "EIEL_SOURCE_GIT_COMMIT"}
        commit = "unavailable"
    return {"git_commit": commit, "discovery": "repository_checkout"}


def _normalized_source_identity(
    source_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = dict(source_identity) if source_identity is not None else _source_identity()
    commit = raw.get("git_commit")
    if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("source identity requires a 40-character lowercase Git commit")
    return {
        str(name): value
        for name, value in sorted(raw.items())
        if name != "discovery"
    }


def _failure(
    *, profile_id: Any, mode: Any, capability: str, reason: str
) -> dict[str, Any]:
    return {
        "schema_version": "eiel_run_contract_admission_failure_v1",
        "profile_id": profile_id,
        "mode": mode,
        "failures": [{"capability": capability, "reason": reason}],
    }


def _relevant_parameter_variants(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = plan.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        return []
    variants: dict[bytes, dict[str, Any]] = {}
    for cell in cells:
        parameters = cell.get("parameters") if isinstance(cell, Mapping) else None
        if not isinstance(parameters, Mapping):
            continue
        relevant = {
            str(name): value
            for name, value in sorted(parameters.items())
            if str(name).startswith(_RELEVANT_PREFIXES)
        }
        variants[canonical_bytes(relevant)] = relevant
    return [variants[key] for key in sorted(variants)]


def resolve_run_contract_v1(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one requested profile without mutating effective configuration."""

    mode = request.get("mode")
    profile_id = request.get("profile_id")
    if mode not in {"optional", "strict"} or profile_id not in _PROFILES:
        diagnostic = {
            "schema_version": "eiel_run_contract_admission_failure_v1",
            "profile_id": profile_id,
            "mode": mode,
            "failures": [
                {
                    "capability": "profile",
                    "reason": "unsupported run-contract mode or profile",
                }
            ],
        }
        if mode == "strict":
            raise RunContractAdmissionError(diagnostic)
        return {**diagnostic, "admission_status": "advisory_failed"}
    try:
        configuration = production_engine_configuration_v1(plan)
        binding = production_engine_component_binding_v1(configuration)
        lifecycle = production_engine_lifecycle_control_v1(plan)
    except (CampaignProviderError, ValueError) as exc:
        diagnostic = {
            "schema_version": "eiel_run_contract_admission_failure_v1",
            "profile_id": profile_id,
            "mode": mode,
            "failures": [{"capability": "effective_configuration", "reason": str(exc)}],
        }
        if mode == "strict":
            raise RunContractAdmissionError(diagnostic) from exc
        return {**diagnostic, "admission_status": "advisory_failed"}

    context = _Context(plan, configuration, binding, lifecycle)
    capability_rows: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for capability in _PROFILES[str(profile_id)]:
        configured, reason, evidence = _VALIDATORS[capability](context)
        capability_rows[capability] = {
            "supported": True,
            "configured": configured,
            "initializable": configured,
            "reason": reason,
            "evidence": dict(evidence),
        }
        if not configured:
            failures.append({"capability": capability, "reason": reason})

    if failures and mode == "strict":
        raise RunContractAdmissionError(
            {
                "schema_version": "eiel_run_contract_admission_failure_v1",
                "profile_id": profile_id,
                "mode": mode,
                "failures": failures,
            }
        )

    try:
        identity = _normalized_source_identity(source_identity)
    except ValueError as exc:
        diagnostic = _failure(
            profile_id=profile_id,
            mode=mode,
            capability="source_identity",
            reason=str(exc),
        )
        if mode == "strict":
            raise RunContractAdmissionError(diagnostic) from exc
        return {**diagnostic, "admission_status": "advisory_failed"}
    resolved_relevant_config = {
        "production_engine_configuration": configuration.payload(),
        "production_engine_configuration_sha256": configuration.configuration_sha256,
        "component_binding": binding.payload(),
        "component_binding_sha256": binding.binding_sha256,
        "production_lifecycle": None if lifecycle is None else lifecycle.payload(),
        "production_lifecycle_sha256": (
            None if lifecycle is None else lifecycle.control_sha256
        ),
        "parameter_variants": _relevant_parameter_variants(plan),
    }
    hash_input = {
        "schema_version": SCHEMA_VERSION,
        "requested": {"profile_id": profile_id, "mode": mode},
        "required_capabilities": list(_PROFILES[str(profile_id)]),
        "capabilities": capability_rows,
        "resolved_relevant_config": resolved_relevant_config,
        "source_identity": identity,
        "device": configuration.backend,
        "qualification_key_sha256": plan.get("qualification_key_sha256"),
    }
    contract_hash = f"sha256:{canonical_sha256(hash_input)}"
    return {
        **hash_input,
        "admission_status": "passed" if not failures else "advisory_failed",
        "failures": failures,
        "contract_hash": contract_hash,
        "hash_input_schema_version": "eiel_run_contract_hash_input_v1",
    }


def validate_resolved_run_contract_v1(
    plan: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Fail closed if an embedded strict contract is stale or was altered."""

    stored = plan.get("run_contract")
    if not isinstance(stored, Mapping):
        return None
    requested = stored.get("requested")
    mode = requested.get("mode") if isinstance(requested, Mapping) else None
    profile_id = (
        requested.get("profile_id") if isinstance(requested, Mapping) else None
    )
    if mode != "strict":
        return dict(stored)

    def reject(reason: str) -> None:
        raise RunContractAdmissionError(
            _failure(
                profile_id=profile_id,
                mode=mode,
                capability="contract_integrity",
                reason=reason,
            )
        )

    if stored.get("schema_version") != SCHEMA_VERSION:
        reject("stored strict contract has an unsupported schema version")
    if stored.get("admission_status") != "passed" or stored.get("failures") != []:
        reject("stored strict contract does not record passed admission")
    hash_input = {name: stored.get(name) for name in _HASH_INPUT_FIELDS}
    expected_stored_hash = f"sha256:{canonical_sha256(hash_input)}"
    if stored.get("contract_hash") != expected_stored_hash:
        reject("stored strict contract hash does not match its canonical inputs")
    assert isinstance(requested, Mapping)
    current = resolve_run_contract_v1(
        plan,
        requested,
        source_identity=source_identity,
    )
    if current["contract_hash"] != stored["contract_hash"]:
        reject(
            "stored strict contract does not match current source identity or "
            "effective configuration"
        )
    return dict(stored)


def refresh_resolved_run_contract_v1(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Re-resolve an opted-in contract after a bounded plan transformation.

    Route qualification and tick-native materialization derive a different
    executable plan.  They must not reuse a hash resolved for the parent plan.
    The request remains operator-owned; this only re-observes the resulting
    effective configuration and never enables a capability.
    """

    stored = plan.get("run_contract")
    if not isinstance(stored, Mapping):
        return None
    requested = stored.get("requested")
    if not isinstance(requested, Mapping):
        raise RunContractAdmissionError(
            _failure(
                profile_id=None,
                mode=None,
                capability="contract_integrity",
                reason="transformed plan has no valid run-contract request",
            )
        )
    return resolve_run_contract_v1(plan, requested)


def checkpoint_contract_binding_v1(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the compact strict-contract identity stored in a checkpoint.

    The authoritative checkpoint binding owns persistence and restore identity.
    This projection deliberately contains no duplicate effective configuration:
    the canonical resolved contract hash already binds that configuration.
    Legacy and optional runs remain unbound by design.
    """

    contract = plan.get("run_contract")
    if not isinstance(contract, Mapping):
        return None
    requested = contract.get("requested")
    if not isinstance(requested, Mapping) or requested.get("mode") != "strict":
        return None
    # Full source/configuration re-resolution is the public admission/resume
    # gate in ``validate_resolved_run_contract_v1``.  This projection is also
    # used by lower-level provider tests, so it verifies the stored canonical
    # contract without rediscovering the checkout a second time.
    hash_input = {name: contract.get(name) for name in _HASH_INPUT_FIELDS}
    expected_hash = f"sha256:{canonical_sha256(hash_input)}"
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("admission_status") != "passed"
        or contract.get("failures") != []
        or contract.get("contract_hash") != expected_hash
    ):
        raise RunContractAdmissionError(
            _failure(
                profile_id=requested.get("profile_id"),
                mode="strict",
                capability="checkpoint_contract",
                reason="strict run contract hash does not match canonical inputs",
            )
        )
    validated = contract
    capabilities = validated.get("capabilities")
    required = validated.get("required_capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
    ):
        raise RunContractAdmissionError(
            _failure(
                profile_id=requested.get("profile_id"),
                mode="strict",
                capability="checkpoint_contract",
                reason="strict run contract lacks required capability identity",
            )
        )
    capability_contracts: dict[str, dict[str, str]] = {}
    version_keys = (
        "contract_version",
        "component_contract",
        "inheritance_contract",
        "mutation_contract",
        "recombination_profile",
        "lifecycle_schema",
        "lifecycle_mode",
    )
    for name in required:
        if not isinstance(name, str) or not isinstance(capabilities.get(name), Mapping):
            raise RunContractAdmissionError(
                _failure(
                    profile_id=requested.get("profile_id"),
                    mode="strict",
                    capability="checkpoint_contract",
                    reason="strict run contract has malformed capability identity",
                )
            )
        evidence = capabilities[name].get("evidence")
        if not isinstance(evidence, Mapping):
            raise RunContractAdmissionError(
                _failure(
                    profile_id=requested.get("profile_id"),
                    mode="strict",
                    capability="checkpoint_contract",
                    reason="strict run contract lacks capability-version evidence",
                )
            )
        capability_contracts[name] = {
            key: value
            for key in version_keys
            if isinstance((value := evidence.get(key)), str)
        }
    contract_hash = validated.get("contract_hash")
    profile_id = requested.get("profile_id")
    if not isinstance(contract_hash, str) or not isinstance(profile_id, str):
        raise RunContractAdmissionError(
            _failure(
                profile_id=profile_id,
                mode="strict",
                capability="checkpoint_contract",
                reason="strict run contract identity is incomplete",
            )
        )
    return {
        "schema_version": CHECKPOINT_BINDING_SCHEMA_VERSION,
        "profile_id": profile_id,
        "run_contract_schema_version": validated.get("schema_version"),
        "run_contract_hash": contract_hash,
        "required_capabilities": list(required),
        "capability_contracts": capability_contracts,
    }


def _metric_rows(
    batch_manifests: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for manifest in batch_manifests:
        raw = manifest.get("metrics_rows")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            rows.extend(row for row in raw if isinstance(row, Mapping))
    return rows


def _generation_rows(
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for metric in metric_rows:
        raw = metric.get("generation_metrics")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            rows.extend(row for row in raw if isinstance(row, Mapping))
    return rows


def _runner_receipts(
    batch_manifests: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Read provider-owned initialization receipts without affecting runtime."""

    rows: list[Mapping[str, Any]] = []
    for manifest in batch_manifests:
        result = manifest.get("result")
        raw = result.get("runner_receipts") if isinstance(result, Mapping) else None
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            rows.extend(row for row in raw if isinstance(row, Mapping))
    return rows


def _number(value: Any) -> float:
    return float(value) if type(value) in {int, float} else 0.0


def build_activation_report_v1(
    plan: Mapping[str, Any],
    batch_manifests: Sequence[Mapping[str, Any]],
    *,
    campaign_status: str,
    runtime_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Project capability-specific runtime evidence without affecting execution."""

    contract = plan.get("run_contract")
    if not isinstance(contract, Mapping) or contract.get("contract_hash") is None:
        return None
    metric_rows = _metric_rows(batch_manifests)
    generations = _generation_rows(metric_rows)
    runner_receipts = _runner_receipts(batch_manifests)
    metrics = [
        row.get("metrics")
        for row in metric_rows
        if isinstance(row.get("metrics"), Mapping)
    ]
    initialized = bool(metric_rows or generations)
    completed_generations = len(generations)
    successful_births = sum(
        _number(row.get("ordinary_birth_count", row.get("successful_births")))
        for row in generations
    )
    if successful_births == 0:
        successful_births = sum(
            _number(row.get("population.successful_births_cumulative"))
            for row in metrics
        )
    eed_metric_observed = any(
        any(str(name).startswith("eed.") for name in row) for row in metrics
    )
    eed_receipt_observed = any(
        isinstance(row.get("eed"), Mapping)
        and row["eed"].get("component_id")
        == "world.overlay.eed_resource_impact.v1"
        for row in runner_receipts
    )
    eed_observed = eed_metric_observed or eed_receipt_observed
    wdl_observed = any("wdl" in row for row in generations) or any(
        any(str(name).startswith("wdl.") for name in row) for row in metrics
    )
    explicit = runtime_evidence or {}
    capability_rows: dict[str, Any] = {}
    for capability in contract.get("required_capabilities", []):
        admission = contract.get("capabilities", {}).get(capability, {})
        configured = bool(
            isinstance(admission, Mapping) and admission.get("configured") is True
        )
        evidence: dict[str, Any] = {
            "completed_generation_rows": completed_generations,
        }
        evidence.update(dict(explicit.get(str(capability), {})))
        eligible = _number(evidence.get("eligible_events"))
        executions = _number(evidence.get("execution_count"))
        exercised = False
        reason = "runtime has not yet produced activation evidence"
        if capability == "patch":
            exercised = completed_generations > 0
            reason = "generation execution observed" if exercised else reason
        elif capability == "wdl_delayed_hazard":
            exercised = wdl_observed
            reason = "WDL generation projection observed" if exercised else reason
        elif capability == "jungle":
            exercised = completed_generations > 0
            reason = (
                "generation execution traversed the resolved Jungle world route"
                if exercised
                else reason
            )
        elif capability == "eed":
            exercised = eed_observed
            evidence["eed_metric_projection_observed"] = eed_metric_observed
            evidence["eed_runner_receipt_observed"] = eed_receipt_observed
            reason = (
                "EED component initialization receipt or read-only projection observed"
                if exercised
                else reason
            )
        elif capability in {
            "genome_v2_ordinary_birth",
            "genome_v2_mutation",
            "recombination_profile",
        }:
            evidence["eligible_successful_births"] = int(successful_births)
            exercised = successful_births > 0
            reason = (
                "successful Genome V2 birth exercised the admitted combined inheritance route"
                if exercised
                else "no successful Genome V2 birth opportunity occurred"
            )

        if executions > 0:
            exercised = True
            reason = "explicit diagnostic execution counter is positive"

        expected_projection_missing = (
            campaign_status == "completed"
            and completed_generations > 0
            and capability in {"wdl_delayed_hazard", "eed"}
            and not exercised
        )
        expected_missing = (
            eligible > 0 and executions == 0
        ) or expected_projection_missing
        if not configured:
            status = "NOT_APPLICABLE"
            initialized_for_capability = False
            reason = "capability was not configured"
        elif expected_missing:
            status = "EXPECTED_BUT_MISSING"
            initialized_for_capability = initialized
            reason = (
                "completed runtime omitted the capability-specific projection"
                if expected_projection_missing
                else "eligible runtime events were observed without execution"
            )
        elif exercised:
            status = "EXERCISED"
            initialized_for_capability = True
        elif campaign_status in {"completed", "failed"} and not initialized:
            status = "EXPECTED_BUT_MISSING"
            initialized_for_capability = False
            reason = "terminal campaign lacks runtime initialization evidence"
        else:
            status = "NOT_YET_EXERCISED"
            initialized_for_capability = initialized
        capability_rows[str(capability)] = {
            "configured": configured,
            "initialized": initialized_for_capability,
            "exercised": exercised,
            "status": status,
            "reason": reason,
            "evidence": evidence,
        }
    return {
        "schema_version": ACTIVATION_SCHEMA_VERSION,
        "profile_id": contract.get("requested", {}).get("profile_id"),
        "contract_hash": contract.get("contract_hash"),
        "campaign_status": campaign_status,
        "capabilities": capability_rows,
        "claim_boundary": (
            "configuration and expected runtime-path integrity only; not scientific "
            "validation or evidence of utility"
        ),
    }


__all__ = [
    "ACTIVATION_SCHEMA_VERSION",
    "CAPABILITIES",
    "CHECKPOINT_BINDING_SCHEMA_VERSION",
    "PROFILE_ID",
    "RunContractAdmissionError",
    "SCHEMA_VERSION",
    "build_activation_report_v1",
    "checkpoint_contract_binding_v1",
    "refresh_resolved_run_contract_v1",
    "resolve_run_contract_v1",
    "validate_resolved_run_contract_v1",
]
