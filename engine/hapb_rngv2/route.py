"""Default-off HAPB-FRAG RNGV2 selector. Legacy FRAG remains the default."""

from __future__ import annotations

import os
from typing import Any, Final

from .contract import (
    CONTRACT_HASH,
    CONTRACT_IDENTITY,
    PARENT_TW_HASH,
    PARENT_TW_IDENTITY,
    SEMANTIC_EPOCH,
    NAMESPACE_DEFAULT,
)


RNG_SEMANTICS_V1: Final[int] = 1
RNG_SEMANTICS_V2: Final[int] = 2
DEFAULT_HAPB_RNG_SEMANTICS: Final[int] = RNG_SEMANTICS_V1
LEGACY_HAPB_FRAG_ROUTE_ID: Final[str] = "hapb_frag_identity_reuse_v1"
V2_HAPB_FRAG_ROUTE_ID: Final[str] = "hapb_frag_rngv2_v1"
HAPB_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1"
)
HAPB_RNG_SEMANTICS_ENV_V1: Final[str] = "EIEL_HAPB_RNG_SEMANTICS_VERSION"


class HapbRngv2ContractError(ValueError):
    """Fail-closed HAPB RNGV2 selector/contract error."""


def resolve_hapb_rng_route(
    *,
    hapb_rng_semantics_version: int = DEFAULT_HAPB_RNG_SEMANTICS,
) -> dict[str, Any]:
    version = int(hapb_rng_semantics_version)
    if version == RNG_SEMANTICS_V1:
        return {
            "hapb_rng_semantics_version": RNG_SEMANTICS_V1,
            "rng_semantics_version": RNG_SEMANTICS_V1,
            "route_id": LEGACY_HAPB_FRAG_ROUTE_ID,
            "default": True,
            "host_tapes_allowed": True,
            "entrypoint": HAPB_DIRECT_ENTRYPOINT_V1,
        }
    if version == RNG_SEMANTICS_V2:
        return {
            "hapb_rng_semantics_version": RNG_SEMANTICS_V2,
            "rng_semantics_version": RNG_SEMANTICS_V2,
            "route_id": V2_HAPB_FRAG_ROUTE_ID,
            "default": False,
            "host_tapes_allowed": False,
            "entrypoint": HAPB_DIRECT_ENTRYPOINT_V1,
            "contract_identity": CONTRACT_IDENTITY,
            "contract_hash": CONTRACT_HASH,
            "parent_tw_contract_identity": PARENT_TW_IDENTITY,
            "parent_tw_contract_hash": PARENT_TW_HASH,
            "namespace_key": NAMESPACE_DEFAULT,
            "semantic_epoch": SEMANTIC_EPOCH,
        }
    raise HapbRngv2ContractError(
        f"unknown hapb_rng_semantics_version {hapb_rng_semantics_version}"
    )


def reject_v1_v2_mix(*, hapb_rng_semantics_version: int, tape_capacity: int) -> None:
    if hapb_rng_semantics_version == RNG_SEMANTICS_V2 and tape_capacity != 0:
        raise HapbRngv2ContractError("HAPB RNGV2 cannot carry V1 Gaussian tapes")
    if hapb_rng_semantics_version == RNG_SEMANTICS_V1 and tape_capacity == 0:
        raise HapbRngv2ContractError("legacy HAPB FRAG cannot use V2 zero-tape state")


def assert_v2_no_host_tapes(
    *,
    tape_capacity: int = 0,
    primary_bytes: int = 0,
    secondary_bytes: int = 0,
    gaussian_cache_bytes: int = 0,
) -> None:
    if (
        tape_capacity != 0
        or primary_bytes != 0
        or secondary_bytes != 0
        or gaussian_cache_bytes != 0
    ):
        raise HapbRngv2ContractError(
            "HAPB RNGV2 forbids host Gaussian/MT tapes and CPython gauss cache"
        )


def apply_hapb_rng_route(
    sim_cfg: Any,
    *,
    hapb_rng_semantics_version: int = DEFAULT_HAPB_RNG_SEMANTICS,
) -> dict[str, Any]:
    """Bind an explicit route onto a HAPB sim_cfg. Default path never calls this."""

    resolved = resolve_hapb_rng_route(
        hapb_rng_semantics_version=hapb_rng_semantics_version
    )
    setattr(sim_cfg, "hapb_rng_semantics_version", resolved["hapb_rng_semantics_version"])
    setattr(sim_cfg, "rng_semantics_version", resolved["rng_semantics_version"])
    if resolved["hapb_rng_semantics_version"] == RNG_SEMANTICS_V2:
        os.environ[HAPB_RNG_SEMANTICS_ENV_V1] = "2"
    else:
        os.environ.pop(HAPB_RNG_SEMANTICS_ENV_V1, None)
    return resolved


def hapb_rngv2_env_enabled() -> bool:
    value = os.environ.get(HAPB_RNG_SEMANTICS_ENV_V1)
    return value == "2"
