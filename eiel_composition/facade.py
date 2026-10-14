"""Explicit, opt-in composition facade over the current EIEL provider.

The facade does not select a different execution owner.  It translates a batch
once through the current provider and, when explicitly requested, proves that
the contract plane can describe and unwrap that exact request before returning
the original provider/request pair for execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from emergence_contracts import ContentDigest, RunId

from .adapters.legacy_components import (
    EielLegacyAssemblyFactory,
    LegacyAssemblyParityV1,
)
from .adapters.legacy_runtime import LegacyRuntimeAdapter


@dataclass(frozen=True, slots=True)
class PreparedEielBatchV1:
    """One current provider request with optional contract-plane receipts."""

    plan: object
    provider: object
    resident_request: object
    contract_facade_enabled: bool
    assembly: object | None = None
    assembly_parity: LegacyAssemblyParityV1 | None = None
    contract_request: object | None = None
    contract_prepared: object | None = None


def _plan_payload(plan: object) -> dict[str, object]:
    payload = plan.payload() if callable(getattr(plan, "payload", None)) else plan
    if not isinstance(payload, Mapping):
        raise TypeError("plan must expose a mapping payload")
    return dict(payload)


class EielCompositionFacadeV1:
    """Keep direct execution authoritative while enabling one bounded opt-in."""

    def __init__(self, *, provider_registry: object | None = None) -> None:
        if provider_registry is None:
            from engine.campaigns.providers import (
                default_execution_provider_registry,
            )

            provider_registry = default_execution_provider_registry()
        self._provider_registry = provider_registry

    def prepare_batch(
        self,
        plan: object,
        batch_id: str,
        *,
        use_contract_facade: bool = False,
    ) -> PreparedEielBatchV1:
        """Translate through the current provider, optionally adding views.

        Translation deliberately happens before any compatibility projection so
        current provider errors retain their type and message.
        """

        payload = _plan_payload(plan)
        profile_id = payload.get("resolved_profile")
        provider = self._provider_registry.get(profile_id)
        if provider is None:
            from engine.campaigns.orchestration import CampaignExecutionError

            raise CampaignExecutionError(
                f"profile {profile_id!r} has no registered execution provider"
            )
        resident_request = provider.translate_batch(plan, batch_id)
        if not use_contract_facade:
            return PreparedEielBatchV1(
                plan=plan,
                provider=provider,
                resident_request=resident_request,
                contract_facade_enabled=False,
            )

        assembly, parity = EielLegacyAssemblyFactory().describe(payload)
        backend_id = payload.get("qualification_key", {}).get("backend")
        if not isinstance(backend_id, str):
            raise TypeError("compiled plan qualification backend is absent")
        runtime = LegacyRuntimeAdapter.from_current_provider(
            provider,
            backend_id=backend_id,
        )
        contract_request = runtime.request_from_resident_batch(
            resident_request=resident_request,
            assembly=assembly,
            config_digest=ContentDigest("sha256", parity.configuration_sha256),
            run_id=RunId(str(payload.get("campaign_id"))),
        )
        contract_prepared = runtime.prepare(contract_request, assembly)
        if runtime.legacy_request_payload(contract_prepared) != resident_request.payload():
            raise RuntimeError("contract facade changed the current provider request")
        return PreparedEielBatchV1(
            plan=plan,
            provider=provider,
            resident_request=resident_request,
            contract_facade_enabled=True,
            assembly=assembly,
            assembly_parity=parity,
            contract_request=contract_request,
            contract_prepared=contract_prepared,
        )

    @staticmethod
    def execute_batch(prepared: PreparedEielBatchV1, context: object) -> object:
        """Delegate unchanged to the provider that translated the request."""

        return prepared.provider.execute_batch(
            prepared.plan,
            prepared.resident_request,
            context,
        )


__all__ = ["EielCompositionFacadeV1", "PreparedEielBatchV1"]
