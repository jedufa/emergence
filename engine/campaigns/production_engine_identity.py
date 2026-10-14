"""Canonical identity for the single EIEL production execution engine."""

from typing import Final


PRODUCTION_PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
PRODUCTION_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
PRODUCTION_CPU_PROFILE_ID: Final[str] = "eiel-production-runtime-v1-cpu-native"
PRODUCTION_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "eiel_production_engine_checkpoint_set_v1"
)
PRODUCTION_LIFECYCLE_OWNER_ID: Final[str] = (
    "eiel_production_engine_lifecycle_v1"
)
PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_production_engine_mechanic_qualification_v1"
)
PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_production_engine_reporting_qualification_v1"
)


__all__ = [
    "PRODUCTION_CHECKPOINT_SCHEMA_VERSION",
    "PRODUCTION_CPU_PROFILE_ID",
    "PRODUCTION_LIFECYCLE_OWNER_ID",
    "PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION",
    "PRODUCTION_PROFILE_ID",
    "PRODUCTION_PROVIDER_ID",
    "PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION",
]
