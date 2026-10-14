"""Default-off Tensor World RNG V2 route identity and admit boundary."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Mapping

from engine.tw_rngv2.contract import CONTRACT_HASH as PROTOTYPE_HASH
from engine.tw_rngv2.contract import CONTRACT_IDENTITY as PROTOTYPE_IDENTITY
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH

from .c4_state import C4PythonRandomTensorStateV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONTRACT_PATH: Final[Path] = (
    ROOT / "_spec" / "campaigns" / "v1" / "tw_rngv2" / "production_route_v1.json"
)

RNG_SEMANTICS_V1: Final[int] = 1
RNG_SEMANTICS_V2: Final[int] = 2
DEFAULT_RNG_SEMANTICS: Final[int] = RNG_SEMANTICS_V1
V1_ROUTE_ID: Final[str] = "tensor_world_c0_c7_v3"
V2_ROUTE_ID: Final[str] = "tensor_world_c0_c7_rngv2_v1"
PRODUCTION_SCHEMA: Final[str] = "tw_rngv2_dea1_production_route_contract_v1"


def load_production_contract() -> dict[str, Any]:
    return json.loads(PRODUCTION_CONTRACT_PATH.read_text(encoding="utf-8"))


def production_contract_hash() -> str:
    payload = load_production_contract()
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def production_contract_identity() -> str:
    return f"{V2_ROUTE_ID}@{production_contract_hash()[:16]}"


def resolve_rng_route(
    *,
    rng_semantics_version: int = DEFAULT_RNG_SEMANTICS,
) -> dict[str, Any]:
    if rng_semantics_version == RNG_SEMANTICS_V1:
        return {
            "rng_semantics_version": RNG_SEMANTICS_V1,
            "route_id": V1_ROUTE_ID,
            "default": True,
            "host_tapes_allowed": True,
        }
    if rng_semantics_version == RNG_SEMANTICS_V2:
        return {
            "rng_semantics_version": RNG_SEMANTICS_V2,
            "route_id": V2_ROUTE_ID,
            "default": False,
            "host_tapes_allowed": False,
            "contract_identity": production_contract_identity(),
            "contract_hash": production_contract_hash(),
            "parent_prototype": PROTOTYPE_IDENTITY,
            "parent_prototype_hash": PROTOTYPE_HASH,
            "namespace_key": NAMESPACE_DEFAULT,
            "semantic_epoch": SEMANTIC_EPOCH,
        }
    raise TensorWorldContractError(
        f"unknown rng_semantics_version {rng_semantics_version}"
    )


def assert_v2_no_host_tapes(state: C4PythonRandomTensorStateV1 | Mapping[str, Any]) -> None:
    if isinstance(state, C4PythonRandomTensorStateV1):
        capacity = state.tape_capacity
        primary_bytes = int(state.python_random_gauss_primary_tape.nbytes)
        secondary_bytes = int(state.python_random_gauss_secondary_tape.nbytes)
    else:
        capacity = int(state["tape_capacity"])
        primary_bytes = int(state["primary_bytes"])
        secondary_bytes = int(state["secondary_bytes"])
    if capacity != 0 or primary_bytes != 0 or secondary_bytes != 0:
        raise TensorWorldContractError(
            "V2 route forbids host Gaussian tapes; tape_capacity must be 0"
        )


def reject_v1_v2_mix(*, rng_semantics_version: int, tape_capacity: int) -> None:
    if rng_semantics_version == RNG_SEMANTICS_V2 and tape_capacity != 0:
        raise TensorWorldContractError("V2 run cannot carry V1 Gaussian tapes")
    if rng_semantics_version == RNG_SEMANTICS_V1 and tape_capacity == 0:
        raise TensorWorldContractError("V1 run cannot use V2 zero-tape state")


__all__ = [
    "DEFAULT_RNG_SEMANTICS",
    "PRODUCTION_CONTRACT_PATH",
    "RNG_SEMANTICS_V1",
    "RNG_SEMANTICS_V2",
    "V1_ROUTE_ID",
    "V2_ROUTE_ID",
    "assert_v2_no_host_tapes",
    "load_production_contract",
    "production_contract_hash",
    "production_contract_identity",
    "reject_v1_v2_mix",
    "resolve_rng_route",
]
