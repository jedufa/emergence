"""Canonical WRC execution-capability envelopes.

An envelope qualifies an instrument surface.  Atlas levels and seeds are
campaign requests and never appear in its identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_sha256
from .contracts import ParameterContractV1, RuntimeProfileV1


SCHEMA_VERSION: Final[str] = "eiel_wrc_parameter_variability_envelope_v1"
INVARIANT_NAME: Final[str] = "wrc.parameter_variability_envelope_sha256"


class WRCVariabilityEnvelopeError(ValueError):
    """The requested WRC capability is not declared by the production route."""


def route_scope_ids_v1(
    world_base: str, world_overlays: tuple[str, ...]
) -> tuple[str, ...]:
    """Translate campaign route metadata to declared capability scopes."""
    if world_base != "patch":
        return ()
    scope_ids = ["patch"]
    aliases = {
        "jungle": "jungle",
        "wdl": "wdl",
        "wdl_delayed_hazard": "wdl",
        "eed_resource_impact": "eed",
    }
    for overlay in world_overlays:
        scope = aliases.get(overlay)
        if scope is None:
            return ()
        scope_ids.append(scope)
    return tuple(sorted(set(scope_ids)))


@dataclass(frozen=True, slots=True)
class WRCParameterVariabilityEnvelopeV1:
    envelope_id: str
    profile_id: str
    backend: str
    world_base: str
    world_overlays: tuple[str, ...]
    parameter_contracts: tuple[Mapping[str, Any], ...]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "envelope_id": self.envelope_id,
            "profile_id": self.profile_id,
            "backend": self.backend,
            "world": {
                "base": self.world_base,
                "overlays": list(self.world_overlays),
            },
            "parameter_contracts": [
                dict(contract) for contract in self.parameter_contracts
            ],
        }

    @property
    def envelope_sha256(self) -> str:
        return canonical_sha256(self.identity_payload())

    def payload(self) -> dict[str, Any]:
        return {**self.identity_payload(), "envelope_sha256": self.envelope_sha256}


def build_wrc_parameter_variability_envelope_v1(
    profile: RuntimeProfileV1,
    *,
    world_base: str,
    world_overlays: tuple[str, ...],
    requested_parameter_names: set[str],
) -> WRCParameterVariabilityEnvelopeV1 | None:
    """Build the full declared surface for a requested route capability.

    The requested names merely select whether WRC mode is needed.  Once it is,
    the envelope contains every declared experiment-variable parameter admitted
    by that route scope, preventing Atlas layout from changing route identity.
    """

    requested = {
        name
        for name in requested_parameter_names
        if (contract := profile.parameter_contracts.get(name)) is not None
        and contract.wrc_axis
        and contract.variability_semantics == "experiment_variable"
    }
    if not requested:
        return None
    scopes = route_scope_ids_v1(world_base, world_overlays)
    if not scopes:
        raise WRCVariabilityEnvelopeError(
            "WRC parameter variability is unsupported for the requested world route"
        )
    contracts: list[Mapping[str, Any]] = []
    admitted_names: set[str] = set()
    for name, contract in sorted(profile.parameter_contracts.items()):
        if (
            contract.wrc_axis
            and contract.variability_semantics == "experiment_variable"
            and set(contract.variability_route_scopes) & set(scopes)
        ):
            if not contract.per_cell or contract.qualification_affecting:
                raise WRCVariabilityEnvelopeError(
                    f"WRC experiment-variable contract {name!r} is inconsistent"
                )
            contracts.append(contract.variability_contract_payload_v1())
            admitted_names.add(name)
    unsupported = sorted(requested - admitted_names)
    if unsupported:
        raise WRCVariabilityEnvelopeError(
            "WRC parameters are outside the declared route capability: "
            f"{unsupported!r}"
        )
    if not contracts:
        raise WRCVariabilityEnvelopeError("route has no declared WRC capability")
    scope_fragment = "-".join(scopes)
    return WRCParameterVariabilityEnvelopeV1(
        envelope_id=(f"wrc-{profile.profile_id}-{scope_fragment}-v1"),
        profile_id=profile.profile_id,
        backend=profile.backend,
        world_base=world_base,
        world_overlays=tuple(sorted(world_overlays)),
        parameter_contracts=tuple(contracts),
    )


def validate_wrc_parameter_variability_envelope_v1(
    value: Mapping[str, Any],
) -> WRCParameterVariabilityEnvelopeV1:
    """Validate an already-bound envelope without granting qualification."""

    if value.get("schema_version") != SCHEMA_VERSION:
        raise WRCVariabilityEnvelopeError("unsupported WRC variability envelope schema")
    contract_rows = value.get("parameter_contracts")
    world = value.get("world")
    if not isinstance(contract_rows, list) or not isinstance(world, Mapping):
        raise WRCVariabilityEnvelopeError("WRC variability envelope is malformed")
    envelope = WRCParameterVariabilityEnvelopeV1(
        envelope_id=str(value.get("envelope_id") or ""),
        profile_id=str(value.get("profile_id") or ""),
        backend=str(value.get("backend") or ""),
        world_base=str(world.get("base") or ""),
        world_overlays=tuple(str(item) for item in world.get("overlays", [])),
        parameter_contracts=tuple(dict(item) for item in contract_rows if isinstance(item, Mapping)),
    )
    if not envelope.envelope_id or len(envelope.parameter_contracts) != len(contract_rows):
        raise WRCVariabilityEnvelopeError("WRC variability envelope is malformed")
    if value.get("envelope_sha256") != envelope.envelope_sha256:
        raise WRCVariabilityEnvelopeError("WRC variability envelope hash is invalid")
    return envelope


__all__ = [
    "INVARIANT_NAME",
    "SCHEMA_VERSION",
    "WRCParameterVariabilityEnvelopeV1",
    "WRCVariabilityEnvelopeError",
    "build_wrc_parameter_variability_envelope_v1",
    "route_scope_ids_v1",
    "validate_wrc_parameter_variability_envelope_v1",
]
