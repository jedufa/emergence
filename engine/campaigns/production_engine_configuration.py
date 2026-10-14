"""Canonical configuration identity consumed by the single production engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Final

from .canonical import canonical_sha256
from .providers import CampaignProviderError
from .wrc_variability import (
    INVARIANT_NAME as WRC_VARIABILITY_INVARIANT_NAME,
    WRCVariabilityEnvelopeError,
    validate_wrc_parameter_variability_envelope_v1,
)


SCHEMA_VERSION: Final[str] = "eiel_production_engine_configuration_v1"
PROVENANCE_ONLY_PARAMETER_NAMES: Final[frozenset[str]] = frozenset(
    {"genome_upgrade_v2.e1_arm_label"}
)
PROVENANCE_ONLY_ARTIFACT_KINDS: Final[frozenset[str]] = frozenset(
    {"preregistration"}
)


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CampaignProviderError(f"production engine configuration lacks {path}")
    return value


@dataclass(frozen=True, slots=True)
class ProductionEngineConfigurationV1:
    """Mechanic identity selected without selecting an execution provider."""

    backend: str
    world_base: str
    world_overlays: tuple[str, ...]
    controller_family: str
    selection_mode: str
    genome_contract: str
    parameter_names: tuple[str, ...]
    artifact_identities: tuple[tuple[str, str], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "world": {
                "base": self.world_base,
                "overlays": list(self.world_overlays),
            },
            "controller": {"family": self.controller_family},
            "selection": {"mode": self.selection_mode},
            "genome": {"contract": self.genome_contract},
            "parameter_names": list(self.parameter_names),
            "artifact_identities": {
                name: digest for name, digest in self.artifact_identities
            },
        }

    def component_binding(self) -> dict[str, Any]:
        """Return the canonical domain binding without changing engine identity."""

        from .production_engine_components import (
            production_engine_component_binding_v1,
        )

        return production_engine_component_binding_v1(self).payload()

    @property
    def configuration_sha256(self) -> str:
        return canonical_sha256(self.payload())


def production_engine_configuration_v1(
    plan: Mapping[str, Any],
) -> ProductionEngineConfigurationV1:
    """Derive world/controller/genome identity from a compiled campaign plan."""

    qualification = plan.get("qualification_key")
    if not isinstance(qualification, Mapping):
        raise CampaignProviderError(
            "compiled plan lacks production engine qualification identity"
        )
    invariants = qualification.get("fixed_invariants")
    artifacts = qualification.get("artifact_hashes")
    artifact_rows = plan.get("artifacts")
    if (
        not isinstance(invariants, Mapping)
        or not isinstance(artifacts, Mapping)
        or not isinstance(artifact_rows, Mapping)
    ):
        raise CampaignProviderError(
            "compiled plan has malformed production engine qualification identity"
        )
    overlays_raw = invariants.get("runtime.world_overlays")
    try:
        overlays = json.loads(_text(overlays_raw, path="runtime.world_overlays"))
    except json.JSONDecodeError as exc:
        raise CampaignProviderError(
            "production engine world overlays are not canonical JSON"
        ) from exc
    if (
        not isinstance(overlays, list)
        or any(not isinstance(value, str) or not value for value in overlays)
        or overlays != sorted(set(overlays))
    ):
        raise CampaignProviderError(
            "production engine world overlays are not sorted unique identifiers"
        )
    cells = plan.get("cells")
    if (
        not isinstance(cells, Sequence)
        or isinstance(cells, (str, bytes))
        or not cells
    ):
        raise CampaignProviderError("compiled plan lacks production engine cells")
    parameter_names: set[str] = set()
    for cell in cells:
        parameters = cell.get("parameters") if isinstance(cell, Mapping) else None
        if not isinstance(parameters, Mapping):
            raise CampaignProviderError(
                "compiled production engine cell lacks parameters"
            )
        parameter_names.update(
            str(name)
            for name in parameters
            if str(name) not in PROVENANCE_ONLY_PARAMETER_NAMES
        )
    variability_envelope = qualification.get("variability_envelope", {})
    if variability_envelope:
        if not isinstance(variability_envelope, Mapping):
            raise CampaignProviderError("WRC variability envelope must be an object")
        try:
            envelope = validate_wrc_parameter_variability_envelope_v1(
                variability_envelope
            )
        except WRCVariabilityEnvelopeError as exc:
            raise CampaignProviderError(str(exc)) from exc
        if invariants.get(WRC_VARIABILITY_INVARIANT_NAME) != envelope.envelope_sha256:
            raise CampaignProviderError(
                "WRC variability envelope invariant does not match its payload"
            )
        parameter_names.update(
            str(contract["name"]) for contract in envelope.parameter_contracts
        )

    artifact_identities = tuple(
        sorted(
            (
                _text(name, path="artifact identity name"),
                _text(digest, path=f"artifact_hashes.{name}"),
            )
            for name, digest in artifacts.items()
            if not (
                isinstance(artifact_rows.get(name), Mapping)
                and artifact_rows[name].get("kind")
                in PROVENANCE_ONLY_ARTIFACT_KINDS
            )
        )
    )
    return ProductionEngineConfigurationV1(
        backend=_text(qualification.get("backend"), path="backend"),
        world_base=_text(
            invariants.get("runtime.world_base"), path="runtime.world_base"
        ),
        world_overlays=tuple(overlays),
        controller_family=_text(
            invariants.get("runtime.policy_family"),
            path="runtime.policy_family",
        ),
        selection_mode=_text(
            invariants.get("runtime.selection_mode"),
            path="runtime.selection_mode",
        ),
        genome_contract=_text(
            invariants.get("semantic_registry"), path="semantic_registry"
        ),
        parameter_names=tuple(sorted(parameter_names)),
        artifact_identities=artifact_identities,
    )


__all__ = [
    "ProductionEngineConfigurationV1",
    "SCHEMA_VERSION",
    "production_engine_configuration_v1",
]
