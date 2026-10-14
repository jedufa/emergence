"""Engine-wide genome, inheritance, and mutation profile ownership.

This registry does not add or alter biology. It gives existing mechanics one
architectural owner and one configuration surface shared by every world and
controller. Resident/native profiles may bind additional ABI metadata, but
they remain profiles of this registry rather than execution providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .genome_upgrade_v2_package_v1 import PRE_ROUTE_PACKAGE_OBLIGATIONS


SCHEMA_VERSION: Final[str] = "eiel_global_genome_registry_v1"
OWNER_ID: Final[str] = "eiel_global_genome_inheritance_v1"


@dataclass(frozen=True, slots=True)
class GlobalGenomeProfileV1:
    profile_id: str
    inheritance_contract: str
    mutation_contract: str
    implementation: str
    historical_only: bool = False
    required_artifact_ids: tuple[str, ...] = ()
    required_obligations: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "inheritance_contract": self.inheritance_contract,
            "mutation_contract": self.mutation_contract,
            "implementation": self.implementation,
            "historical_only": self.historical_only,
            "required_artifact_ids": list(self.required_artifact_ids),
            "required_obligations": list(self.required_obligations),
        }


_PROFILES: Final[dict[str, GlobalGenomeProfileV1]] = {
    "patch_default": GlobalGenomeProfileV1(
        profile_id="patch_default",
        inheritance_contract="eiel_inheritance_registry_v1",
        mutation_contract="existing_patch_mutation_schedule_v1",
        implementation="engine.evolution.reproduction",
    ),
    "inheritance_cross_path": GlobalGenomeProfileV1(
        profile_id="inheritance_cross_path",
        inheritance_contract="inheritance_cross_path_registry_v2",
        mutation_contract="existing_mutation_schedule_after_linkage_v1",
        implementation="engine.evolution.reproduction",
        historical_only=True,
    ),
    "genome_upgrade_v2": GlobalGenomeProfileV1(
        profile_id="genome_upgrade_v2",
        inheritance_contract="genome_upgrade_v2_linked_diploid_tape_v1",
        mutation_contract="genome_upgrade_v2_fixed_q10_mutation_v1",
        implementation="engine.evolution.genome_upgrade_v2",
        required_artifact_ids=(
            "genome_upgrade_v2_executable_contract",
            "genome_upgrade_v2_capability_contract",
            "genome_upgrade_v2_package_contract",
            "genome_upgrade_v2_obligation_dictionary",
            "genome_upgrade_v2_invalidation_rules",
            "genome_upgrade_v2_chronicle_schema",
            "genome_upgrade_v2_reporting_schema",
            "genome_upgrade_v2_observability_storage_budget",
            "genome_upgrade_v2_package_receipt",
        ),
        required_obligations=PRE_ROUTE_PACKAGE_OBLIGATIONS,
    ),
    "recomb72": GlobalGenomeProfileV1(
        profile_id="recomb72",
        inheritance_contract="recomb72_frozen_inheritance_v1",
        mutation_contract="recomb72_audit_mutation_disabled_v1",
        implementation="engine.evolution.recomb72_inheritance",
        historical_only=True,
    ),
}
def global_genome_profile_v1(profile_id: str) -> GlobalGenomeProfileV1:
    """Resolve a profile admitted for current production configuration."""

    try:
        profile = _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown global genome profile: {profile_id!r}") from exc
    if profile.historical_only:
        raise ValueError(
            "P12_RETIRED_GLOBAL_GENOME_PROFILE "
            f"profile={profile_id!r} historical_analysis_only"
        )
    return profile


def global_genome_registry_v1() -> dict[str, object]:
    """Return the canonical architecture-level registry projection."""

    return {
        "schema_version": SCHEMA_VERSION,
        "owner_id": OWNER_ID,
        "profiles": {
            profile_id: _PROFILES[profile_id].payload()
            for profile_id in sorted(_PROFILES)
        },
    }


__all__ = [
    "GlobalGenomeProfileV1",
    "OWNER_ID",
    "SCHEMA_VERSION",
    "global_genome_profile_v1",
    "global_genome_registry_v1",
]
