"""Strict, host-only identity contract for CUDA-native Full Evolution V2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import re
from typing import Any, Final

CUDA_NATIVE_PROFILE_V2: Final[str] = "EIEL_FULL_EVOLUTION_CUDA_NATIVE_V2"
CUDA_NATIVE_PROFILE_ID_V2: Final[str] = "eiel_full_evolution_cuda_native_v2"
CUDA_NATIVE_ABI_V1: Final[str] = "eiel_full_evolution_cuda_native_abi_v1"
CUDA_NATIVE_ROUTE_V2: Final[str] = "tensor_world_cuda_full_evolution_v2"
CUDA_NATIVE_TRANSITION_V2: Final[str] = "eiel_full_evolution_transition_v2"
CUDA_NATIVE_TIME_AUTHORITY_V1: Final[str] = "absolute_tick_v1"
CUDA_NATIVE_TRANSITION_ORACLE_V1: Final[str] = "cuda_golden_only"
CUDA_NATIVE_WTPS_BASELINE: Final[Decimal] = Decimal("4467.193115159868")
CUDA_NATIVE_PACKED_TPS_BASELINE: Final[Decimal] = Decimal("32.847008199704916")
CUDA_NATIVE_TARGET_WTPS_100X: Final[Decimal] = CUDA_NATIVE_WTPS_BASELINE * Decimal(100)
CUDA_NATIVE_QUALIFIED_BASELINE_BATCH_V2: Final[int] = 136
CUDA_NATIVE_B136_COMPARISON_PACKED_TICKS_PER_SECOND_100X: Final[Decimal] = (
    CUDA_NATIVE_PACKED_TPS_BASELINE * Decimal(100)
)
CUDA_NATIVE_B136_COMPARISON_PACKED_TICK_MS_100X: Final[Decimal] = (
    (Decimal(1000) / CUDA_NATIVE_B136_COMPARISON_PACKED_TICKS_PER_SECOND_100X).quantize(
        Decimal("0.00000000001"), rounding=ROUND_HALF_EVEN
    )
)
# Compatibility aliases inside this unqualified V2 development branch. These are
# B136 comparison quantities, never the closure metric for the tunable winning B.
CUDA_NATIVE_TARGET_PACKED_TICKS_PER_SECOND_100X: Final[Decimal] = (
    CUDA_NATIVE_B136_COMPARISON_PACKED_TICKS_PER_SECOND_100X
)
CUDA_NATIVE_TARGET_PACKED_TICK_MS_100X: Final[Decimal] = (
    CUDA_NATIVE_B136_COMPARISON_PACKED_TICK_MS_100X
)

WORLD_WIDTH_V2: Final[int] = 32
INITIAL_POPULATION_V2: Final[int] = 256
RESIDENT_CAPACITY_C_V2: Final[int] = 512
AGENT_CAPACITY_V2: Final[int] = RESIDENT_CAPACITY_C_V2
LINKED_DIPLOID_LOCI_V2: Final[int] = 128
TEMPORAL_K_DEFAULT_V2: Final[int] = 8
TEMPORAL_K_MIN_V2: Final[int] = 1
TEMPORAL_K_MAX_V2: Final[int] = 64
POLICY_FEATURES_V2: Final[int] = 16
POLICY_ACTIONS_V2: Final[int] = 7
POLICY_ALLELE_BYTES_V2: Final[int] = POLICY_FEATURES_V2 * POLICY_ACTIONS_V2
TRAIT_ALLELES_V2: Final[int] = 16
MAX_SIGNED_INT64: Final[int] = (1 << 63) - 1
MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2: Final[int] = 65_536
MAX_TOTAL_TICKS_COUNTER_SAFE_V2: Final[int] = (
    MAX_SIGNED_INT64 // MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2
)

FULL_EVOLUTION_CAPABILITIES_V2: Final[tuple[str, ...]] = (
    "patch",
    "wdl_delayed_hazard",
    "jungle",
    "eed",
    "genome_v2_ordinary_birth",
    "genome_v2_mutation",
    "recombination_profile",
)
_GENERATION_KEY = re.compile(r"(?:generation|(?:^|[_-])g[0-9]+(?:$|[_-]))", re.I)
_ALLOWED_KEYS = frozenset({
    "profile", "profile_id", "native_abi", "route_id", "transition_id",
    "time_authority", "transition_oracle", "capabilities", "B", "total_ticks", "start_next_tick",
    "temporal_k", "world_width", "initial_population", "agent_capacity",
})


class CudaFullEvolutionContractError(ValueError):
    """Raised when a CUDA-native Full Evolution V2 input is not exact."""


def cuda_native_required_packed_ticks_per_second_100x_v2(batch_size: int) -> Decimal:
    """Packed-tick rate equivalent to the world-rate goal for one exact B."""
    _positive_int(batch_size, "batch_size")
    return CUDA_NATIVE_TARGET_WTPS_100X / Decimal(batch_size)


def cuda_native_required_packed_tick_ms_100x_v2(batch_size: int) -> Decimal:
    """Packed-tick latency equivalent to the world-rate goal for one exact B."""
    return (
        Decimal(1000)
        / cuda_native_required_packed_ticks_per_second_100x_v2(batch_size)
    ).quantize(Decimal("0.00000000001"), rounding=ROUND_HALF_EVEN)


def _positive_int(value: object, name: str) -> None:
    if type(value) is not int or not 0 < value <= MAX_SIGNED_INT64:
        raise CudaFullEvolutionContractError(
            f"{name} must be an integer in [1, {MAX_SIGNED_INT64}]"
        )


def _nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or not 0 <= value <= MAX_SIGNED_INT64:
        raise CudaFullEvolutionContractError(
            f"{name} must be an integer in [0, {MAX_SIGNED_INT64}]"
        )


def _reject_generation_keys(value: object, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise CudaFullEvolutionContractError(f"{path}: mapping keys must be strings")
            if _GENERATION_KEY.search(key):
                raise CudaFullEvolutionContractError(
                    f"{path}.{key}: generation-shaped keys are forbidden; use absolute ticks"
                )
            _reject_generation_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_generation_keys(nested, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionNativeContractV2:
    """V2 public identity. CUDA golden execution is the sole transition oracle."""

    B: int
    total_ticks: int
    start_next_tick: int = 0
    temporal_k: int = TEMPORAL_K_DEFAULT_V2
    profile: str = CUDA_NATIVE_PROFILE_V2
    profile_id: str = CUDA_NATIVE_PROFILE_ID_V2
    native_abi: str = CUDA_NATIVE_ABI_V1
    route_id: str = CUDA_NATIVE_ROUTE_V2
    transition_id: str = CUDA_NATIVE_TRANSITION_V2
    time_authority: str = CUDA_NATIVE_TIME_AUTHORITY_V1
    transition_oracle: str = CUDA_NATIVE_TRANSITION_ORACLE_V1
    capabilities: tuple[str, ...] = FULL_EVOLUTION_CAPABILITIES_V2
    world_width: int = WORLD_WIDTH_V2
    initial_population: int = INITIAL_POPULATION_V2
    agent_capacity: int = AGENT_CAPACITY_V2

    def __post_init__(self) -> None:
        _positive_int(self.B, "B")
        _positive_int(self.total_ticks, "total_ticks")
        if self.total_ticks > MAX_TOTAL_TICKS_COUNTER_SAFE_V2:
            raise CudaFullEvolutionContractError(
                "total_ticks exceeds the worst-case signed-int64 capability-counter bound"
            )
        _nonnegative_int(self.start_next_tick, "start_next_tick")
        if self.start_next_tick > MAX_SIGNED_INT64 - self.total_ticks:
            raise CudaFullEvolutionContractError(
                "start_next_tick + total_ticks exceeds signed int64"
            )
        _positive_int(self.temporal_k, "temporal_k")
        if not TEMPORAL_K_MIN_V2 <= self.temporal_k <= TEMPORAL_K_MAX_V2:
            raise CudaFullEvolutionContractError(
                f"temporal_k must be in [{TEMPORAL_K_MIN_V2}, {TEMPORAL_K_MAX_V2}]"
            )
        fixed = (
            ("world_width", self.world_width, WORLD_WIDTH_V2),
            ("initial_population", self.initial_population, INITIAL_POPULATION_V2),
            ("agent_capacity", self.agent_capacity, AGENT_CAPACITY_V2),
        )
        for name, actual, expected in fixed:
            if type(actual) is not int or actual != expected:
                raise CudaFullEvolutionContractError(f"{name} must equal {expected}")
        required = (
            ("profile", self.profile, CUDA_NATIVE_PROFILE_V2),
            ("profile_id", self.profile_id, CUDA_NATIVE_PROFILE_ID_V2),
            ("native_abi", self.native_abi, CUDA_NATIVE_ABI_V1),
            ("route_id", self.route_id, CUDA_NATIVE_ROUTE_V2),
            ("transition_id", self.transition_id, CUDA_NATIVE_TRANSITION_V2),
            ("time_authority", self.time_authority, CUDA_NATIVE_TIME_AUTHORITY_V1),
            ("transition_oracle", self.transition_oracle, CUDA_NATIVE_TRANSITION_ORACLE_V1),
        )
        for name, actual, expected in required:
            if actual != expected:
                raise CudaFullEvolutionContractError(f"{name} must equal {expected}")
        if type(self.capabilities) is not tuple:
            raise CudaFullEvolutionContractError("capabilities must be an immutable tuple")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise CudaFullEvolutionContractError("capabilities must not contain duplicates")
        if self.capabilities != FULL_EVOLUTION_CAPABILITIES_V2:
            raise CudaFullEvolutionContractError(
                "capabilities must exactly match all seven CUDA Native V2 capabilities"
            )

    @property
    def batch_size(self) -> int:
        return self.B

    @property
    def linked_diploid_loci(self) -> int:
        return LINKED_DIPLOID_LOCI_V2

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile, "profile_id": self.profile_id,
            "native_abi": self.native_abi, "route_id": self.route_id,
            "transition_id": self.transition_id, "time_authority": self.time_authority,
            "transition_oracle": self.transition_oracle, "capabilities": list(self.capabilities),
            "B": self.B, "total_ticks": self.total_ticks,
            "start_next_tick": self.start_next_tick, "temporal_k": self.temporal_k,
            "world_width": self.world_width, "initial_population": self.initial_population,
            "agent_capacity": self.agent_capacity,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CudaFullEvolutionNativeContractV2":
        _reject_generation_keys(value)
        unknown = set(value) - _ALLOWED_KEYS
        if unknown:
            raise CudaFullEvolutionContractError(
                "unsupported CUDA Native V2 contract keys: " + ", ".join(sorted(unknown))
            )
        missing = {"B", "total_ticks"} - set(value)
        if missing:
            raise CudaFullEvolutionContractError(
                "missing CUDA Native V2 contract keys: " + ", ".join(sorted(missing))
            )
        capabilities = value.get("capabilities", FULL_EVOLUTION_CAPABILITIES_V2)
        if type(capabilities) is list:
            capabilities = tuple(capabilities)
        return cls(
            B=value["B"], total_ticks=value["total_ticks"],
            start_next_tick=value.get("start_next_tick", 0),
            temporal_k=value.get("temporal_k", TEMPORAL_K_DEFAULT_V2),
            profile=value.get("profile", CUDA_NATIVE_PROFILE_V2),
            profile_id=value.get("profile_id", CUDA_NATIVE_PROFILE_ID_V2),
            native_abi=value.get("native_abi", CUDA_NATIVE_ABI_V1),
            route_id=value.get("route_id", CUDA_NATIVE_ROUTE_V2),
            transition_id=value.get("transition_id", CUDA_NATIVE_TRANSITION_V2),
            time_authority=value.get("time_authority", CUDA_NATIVE_TIME_AUTHORITY_V1),
            transition_oracle=value.get("transition_oracle", CUDA_NATIVE_TRANSITION_ORACLE_V1),
            capabilities=capabilities,
            world_width=value.get("world_width", WORLD_WIDTH_V2),
            initial_population=value.get("initial_population", INITIAL_POPULATION_V2),
            agent_capacity=value.get("agent_capacity", AGENT_CAPACITY_V2),
        )


def validate_cuda_full_evolution_contract_v2(
    value: CudaFullEvolutionNativeContractV2 | Mapping[str, Any],
) -> CudaFullEvolutionNativeContractV2:
    if isinstance(value, CudaFullEvolutionNativeContractV2):
        return value
    if not isinstance(value, Mapping):
        raise CudaFullEvolutionContractError("CUDA Native V2 contract must be a mapping")
    return CudaFullEvolutionNativeContractV2.from_mapping(value)
