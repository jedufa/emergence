"""Runtime projection of generic Science Instrument worlds onto CUDA V2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import struct
from typing import Any, Final, Mapping, Sequence

from emergence_contracts import (
    ExactNaturalSourceStateError,
    full_genome_bytes_v1,
    materialization_profile_id_v1,
    validate_exact_natural_source_state_ref_v1,
)

from .contract import INITIAL_POPULATION_V2, RESIDENT_CAPACITY_C_V2


NATIVE_WORLD_CONTROL_WIDTH_V1: Final[int] = 22
NATIVE_WORLD_CONTROL_FIELDS_V1: Final[tuple[str, ...]] = (
    "capability_mask",
    "mutation_threshold_u16",
    "mutation_step",
    "resource_carrying_base",
    "resource_carrying_variation",
    "resource_growth_base",
    "resource_growth_variation",
    "resource_multiplier_q16",
    "movement_cost",
    "resource_uptake_base",
    "wdl_damage",
    "jungle_damage_threshold",
    "eed_loss_shift",
    "reproduction_energy_threshold",
    "reproduction_energy_cost",
    "selection_pressure_q16",
    "seed_ecology",
    "seed_founder",
    "seed_policy",
    "seed_mutation",
    "seed_task_environment",
    "seed_initialization",
)
NATIVE_FOUNDER_REGISTRY_RELATIVE_PATH_V1: Final[str] = (
    "_spec/science_instrument/v1/native_founder_profile_registry_v1.json"
)
NATIVE_WORLD_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_science_cuda_v2_native_world_v1"
)
NATIVE_WORLD_CONTROL_SET_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_science_cuda_v2_native_world_control_set_v1"
)

_DEFAULTS: Final[dict[str, int | float | bool]] = {
    "environment.patch.enabled": True,
    "environment.wdl.enabled": True,
    "environment.jungle.enabled": True,
    "environment.eed.enabled": True,
    "reproduction.enabled": True,
    "mutation.enabled": True,
    "recombination.enabled": True,
    "mutation.rate": 2048.0 / 65536.0,
    "mutation.step": 1,
    "environment.resource.carrying_base": 96,
    "environment.resource.carrying_variation": 63,
    "environment.resource.growth_base": 1,
    "environment.resource.growth_variation": 3,
    "environment.resource.multiplier": 1.0,
    "environment.resource_multiplier": 1.0,
    "environment.movement.cost": 1,
    "environment.resource.uptake_base": 4,
    "environment.wdl.damage": 1,
    "environment.jungle.damage_threshold": 24,
    "environment.eed.loss_shift": 5,
    "reproduction.energy_threshold": 100,
    "reproduction.energy_cost": 30,
    "selection.pressure": 1.0,
}
_CAPABILITY_BITS: Final[dict[str, int]] = {
    "environment.patch.enabled": 1 << 0,
    "environment.wdl.enabled": 1 << 1,
    "environment.jungle.enabled": 1 << 2,
    "environment.eed.enabled": 1 << 3,
    "reproduction.enabled": 1 << 4,
    "mutation.enabled": 1 << 5,
    "recombination.enabled": 1 << 6,
}


class ScienceNativeWorldRuntimeError(ValueError):
    """A generic native-world payload cannot be lowered to CUDA tensors."""


@dataclass(frozen=True, slots=True)
class _NativeWorldRuntimeRowV1:
    """Digest-checked runtime view of an already projected logical world."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "_NativeWorldRuntimeRowV1":
        raw = dict(value)
        required = {
            "schema_version",
            "semantic_inputs_sha256",
            "science_ownership_sha256",
            "capability_mask_at_start",
            "world",
            "initialization",
            "seed_bindings",
            "control_bindings",
            "time",
            "control_digest",
        }
        if set(raw) != required or raw.get("schema_version") != (
            NATIVE_WORLD_SCHEMA_VERSION_V1
        ):
            raise ScienceNativeWorldRuntimeError(
                "native world control fields or schema are unsupported"
            )
        digest = raw.get("control_digest")
        if (
            not isinstance(digest, Mapping)
            or set(digest) != {"algorithm", "hex"}
            or digest.get("algorithm") != "sha256"
            or not isinstance(digest.get("hex"), str)
        ):
            raise ScienceNativeWorldRuntimeError(
                "native world control digest is invalid"
            )
        body = {key: raw[key] for key in required - {"control_digest"}}
        if _sha256(body) != digest["hex"]:
            raise ScienceNativeWorldRuntimeError(
                "native world control digest does not match its body"
            )
        for field in (
            "world",
            "initialization",
            "seed_bindings",
            "time",
        ):
            if not isinstance(raw[field], Mapping):
                raise ScienceNativeWorldRuntimeError(
                    f"native world {field} is not an object"
                )
        if not isinstance(raw["control_bindings"], list):
            raise ScienceNativeWorldRuntimeError(
                "native world control_bindings is not an array"
            )
        return cls(payload=raw)

    @property
    def control_digest(self) -> str:
        return str(self.payload["control_digest"]["hex"])


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _native_world_control_set_sha256_v1(
    worlds: Sequence[_NativeWorldRuntimeRowV1],
) -> str:
    if not worlds:
        raise ScienceNativeWorldRuntimeError(
            "native world control set cannot be empty"
        )
    return _sha256(
        {
            "schema_version": NATIVE_WORLD_CONTROL_SET_SCHEMA_VERSION_V1,
            "ordered_control_digests": [world.control_digest for world in worlds],
        }
    )


def _value_at_tick(row: Mapping[str, Any], absolute_tick: int) -> object:
    value = row["resolved_value"]
    schedule = row["schedule"]
    kind = schedule["kind"]
    if kind == "static":
        return value
    if kind == "piecewise_constant_absolute_ticks":
        for segment in schedule["segments"]:
            if segment["start_absolute_tick"] <= absolute_tick:
                value = segment["value"]
            else:
                break
        return value
    if kind == "switch_absolute_tick":
        if type(value) is not bool:
            raise ScienceNativeWorldRuntimeError(
                f"{row['control_id']} uses a non-boolean switch"
            )
        return not value if absolute_tick >= schedule["absolute_tick"] else value
    if kind == "pulse_absolute_ticks":
        start = schedule["start_absolute_tick"]
        if start <= absolute_tick < start + schedule["duration_ticks"]:
            return value
        return _DEFAULTS[row["control_id"]]
    raise ScienceNativeWorldRuntimeError(f"unsupported schedule kind {kind!r}")


def _next_boundary(row: Mapping[str, Any], absolute_tick: int) -> int | None:
    schedule = row["schedule"]
    kind = schedule["kind"]
    candidates: list[int] = []
    if kind == "piecewise_constant_absolute_ticks":
        candidates.extend(
            segment["start_absolute_tick"] for segment in schedule["segments"]
        )
    elif kind == "switch_absolute_tick":
        candidates.append(schedule["absolute_tick"])
    elif kind == "pulse_absolute_ticks":
        candidates.extend(
            (
                schedule["start_absolute_tick"],
                schedule["start_absolute_tick"] + schedule["duration_ticks"],
            )
        )
    return min((value for value in candidates if value > absolute_tick), default=None)


def _control_values(
    world: _NativeWorldRuntimeRowV1, absolute_tick: int
) -> dict[str, object]:
    values: dict[str, object] = dict(_DEFAULTS)
    for row in world.payload["control_bindings"]:
        values[row["control_id"]] = _value_at_tick(row, absolute_tick)
    # The dotted spelling is canonical for the runtime field; retain the older
    # underscore spelling as a declarative alias accepted by existing bundles.
    if any(
        row["control_id"] == "environment.resource_multiplier"
        for row in world.payload["control_bindings"]
    ):
        values["environment.resource.multiplier"] = values[
            "environment.resource_multiplier"
        ]
    return values


def _q16(value: object, *, field: str) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScienceNativeWorldRuntimeError(f"{field} must be numeric")
    result = int(round(float(value) * 65536.0))
    if not 0 <= result < 1 << 31:
        raise ScienceNativeWorldRuntimeError(f"{field} exceeds Q16 range")
    return result


def _seed(seeds: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = seeds.get(name)
        if type(value) is int:
            return value
    return 0


def _runtime_row(
    world: _NativeWorldRuntimeRowV1, absolute_tick: int
) -> tuple[int, ...]:
    values = _control_values(world, absolute_tick)
    mask = 0
    for control_id, bit in _CAPABILITY_BITS.items():
        if values[control_id] is True:
            mask |= bit
    mutation_rate = values["mutation.rate"]
    if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
        raise ScienceNativeWorldRuntimeError("mutation.rate must be numeric")
    threshold = min(65536, max(0, int(round(float(mutation_rate) * 65536.0))))
    seeds = world.payload["seed_bindings"]
    row = (
        mask,
        threshold,
        int(values["mutation.step"]),
        int(values["environment.resource.carrying_base"]),
        int(values["environment.resource.carrying_variation"]),
        int(values["environment.resource.growth_base"]),
        int(values["environment.resource.growth_variation"]),
        _q16(values["environment.resource.multiplier"], field="resource multiplier"),
        int(values["environment.movement.cost"]),
        int(values["environment.resource.uptake_base"]),
        int(values["environment.wdl.damage"]),
        int(values["environment.jungle.damage_threshold"]),
        int(values["environment.eed.loss_shift"]),
        int(values["reproduction.energy_threshold"]),
        int(values["reproduction.energy_cost"]),
        _q16(values["selection.pressure"], field="selection pressure"),
        _seed(seeds, "ecology"),
        _seed(seeds, "founder"),
        _seed(seeds, "policy"),
        _seed(seeds, "mutation"),
        _seed(seeds, "task_environment", "task"),
        _seed(seeds, "initialization"),
    )
    if len(row) != NATIVE_WORLD_CONTROL_WIDTH_V1:
        raise ScienceNativeWorldRuntimeError("native world control width drifted")
    return row


def _load_founder_registry(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ScienceNativeWorldRuntimeError(
            f"native founder registry is unreadable: {path}"
        ) from exc
    if raw.get("schema_version") != "eiel_science_native_founder_profile_registry_v1":
        raise ScienceNativeWorldRuntimeError("native founder registry schema changed")
    rows = raw.get("profiles")
    if raw.get("carrier_preserved_trait_loci") != [0]:
        raise ScienceNativeWorldRuntimeError(
            "native founder registry must preserve the carrier sex trait locus"
        )
    if not isinstance(rows, list) or not rows:
        raise ScienceNativeWorldRuntimeError("native founder registry is empty")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("profile_id"), str):
            raise ScienceNativeWorldRuntimeError("native founder profile is invalid")
        profile_id = row["profile_id"]
        if profile_id in by_id:
            raise ScienceNativeWorldRuntimeError("native founder registry duplicates an ID")
        by_id[profile_id] = row
    return by_id, hashlib.sha256(raw_bytes).hexdigest()


def _resolve_profile(
    registry: Mapping[str, dict[str, Any]], profile_id: str
) -> dict[str, Any]:
    seen: set[str] = set()
    current = profile_id
    while True:
        if current in seen:
            raise ScienceNativeWorldRuntimeError("native founder alias cycle")
        seen.add(current)
        row = registry.get(current)
        if row is None:
            raise ScienceNativeWorldRuntimeError(
                f"native founder profile is not registered: {profile_id}"
            )
        alias = row.get("alias_of")
        if alias is None:
            return row
        if not isinstance(alias, str):
            raise ScienceNativeWorldRuntimeError("native founder alias is invalid")
        current = alias


def _profile_arrays(row: Mapping[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    base = row.get("base")
    if not isinstance(base, Mapping):
        raise ScienceNativeWorldRuntimeError("native founder profile base is absent")
    if base.get("kind") == "constant":
        policy = [[int(base.get("policy", 0)) for _ in range(112)] for _ in range(2)]
        traits = [[int(base.get("traits", 0)) for _ in range(16)] for _ in range(2)]
    elif base.get("kind") == "deterministic_random":
        generator = random.Random(int(base["seed"]))
        policy = [
            [
                generator.randint(int(base["policy_min"]), int(base["policy_max"]))
                for _ in range(112)
            ]
            for _ in range(2)
        ]
        traits = [
            [
                generator.randint(int(base["trait_min"]), int(base["trait_max"]))
                for _ in range(16)
            ]
            for _ in range(2)
        ]
    else:
        raise ScienceNativeWorldRuntimeError("native founder profile base kind is invalid")
    for family, target, limit in (
        ("policy_edits", policy, 112),
        ("trait_edits", traits, 16),
    ):
        edits = row.get(family, [])
        if not isinstance(edits, list):
            raise ScienceNativeWorldRuntimeError(f"{family} must be an array")
        for edit in edits:
            if not isinstance(edit, Mapping) or set(edit) != {
                "alleles",
                "loci",
                "value",
            }:
                raise ScienceNativeWorldRuntimeError(f"{family} row is invalid")
            for allele in edit["alleles"]:
                for locus in edit["loci"]:
                    if allele not in (0, 1) or type(locus) is not int or not 0 <= locus < limit:
                        raise ScienceNativeWorldRuntimeError(f"{family} index is invalid")
                    target[allele][locus] = int(edit["value"])
    return policy, traits


def _exact_source_profile_arrays(
    initialization: Mapping[str, Any],
) -> dict[str, tuple[list[list[int]], list[list[int]]]] | None:
    """Decode only authenticated inline natural states; never synthesize them."""

    states = initialization.get("exact_natural_source_states")
    if states is None:
        return None
    if not isinstance(states, list) or not states:
        raise ScienceNativeWorldRuntimeError(
            "exact natural source states are incomplete"
        )
    result: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
    try:
        for state in states:
            ref = validate_exact_natural_source_state_ref_v1(state)
            profile_id = materialization_profile_id_v1(ref["genome_state_id"])
            genome = full_genome_bytes_v1(ref)
            policy_values = list(struct.unpack("<224b", genome[:224]))
            trait_values = list(struct.unpack("<32h", genome[224:]))
            result[profile_id] = (
                [policy_values[:112], policy_values[112:]],
                [trait_values[:16], trait_values[16:]],
            )
    except (ExactNaturalSourceStateError, struct.error, TypeError, ValueError) as error:
        raise ScienceNativeWorldRuntimeError(
            f"exact natural source state is not materializable: {error}"
        ) from error
    if len(result) != len(states):
        raise ScienceNativeWorldRuntimeError(
            "exact natural source states have colliding materialization identities"
        )
    return result


def founder_profile_policy_id_v1(
    profile_id: str,
    *,
    repository_root: Path | None = None,
) -> str:
    """Return the capture-compatible policy identity for one founder profile.

    The identity is derived from the exact two-allele int8 policy bytes that
    the native capture emits.  It is deliberately independent of any study or
    panel name, so a future science bundle can bind the same policy identity
    to its own declared strategy vocabulary.
    """

    if not isinstance(profile_id, str) or not profile_id:
        raise ScienceNativeWorldRuntimeError("founder profile ID is invalid")
    root = repository_root or Path(__file__).resolve().parents[3]
    registry_path = (root / NATIVE_FOUNDER_REGISTRY_RELATIVE_PATH_V1).resolve()
    registry, _ = _load_founder_registry(registry_path)
    row = _resolve_profile(registry, profile_id)
    policy, _ = _profile_arrays(row)
    encoded = bytes(int(value) & 0xFF for allele in policy for value in allele)
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def founder_profile_policy_bindings_v1(
    profile_ids: Sequence[str],
    *,
    repository_root: Path | None = None,
) -> dict[str, str]:
    """Return a canonical policy-id → declared founder-profile mapping."""

    unique = tuple(sorted(set(profile_ids)))
    if not unique or any(not isinstance(item, str) or not item for item in unique):
        raise ScienceNativeWorldRuntimeError("founder profile IDs are invalid")
    result = {
        founder_profile_policy_id_v1(item, repository_root=repository_root): item
        for item in unique
    }
    if len(result) != len(unique):
        raise ScienceNativeWorldRuntimeError(
            "founder profiles produce colliding policy identities"
        )
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class ScienceNativeWorldRuntimeProjectionV1:
    worlds: tuple[_NativeWorldRuntimeRowV1, ...]
    founder_registry_path: Path
    founder_registry_sha256: str
    control_set_sha256: str

    @classmethod
    def from_controls(
        cls,
        controls: Sequence[Mapping[str, Any]],
        *,
        repository_root: Path | None = None,
    ) -> "ScienceNativeWorldRuntimeProjectionV1":
        worlds = tuple(_NativeWorldRuntimeRowV1.from_dict(row) for row in controls)
        if not worlds:
            raise ScienceNativeWorldRuntimeError("native world controls are empty")
        root = repository_root or Path(__file__).resolve().parents[3]
        registry_path = (root / NATIVE_FOUNDER_REGISTRY_RELATIVE_PATH_V1).resolve()
        _registry, registry_sha256 = _load_founder_registry(registry_path)
        return cls(
            worlds=worlds,
            founder_registry_path=registry_path,
            founder_registry_sha256=registry_sha256,
            control_set_sha256=_native_world_control_set_sha256_v1(worlds),
        )

    def rows_at_tick(self, absolute_tick: int) -> tuple[tuple[int, ...], ...]:
        if type(absolute_tick) is not int or absolute_tick < 0:
            raise ScienceNativeWorldRuntimeError("absolute tick is invalid")
        return tuple(_runtime_row(world, absolute_tick) for world in self.worlds)

    def next_boundary_after(self, absolute_tick: int) -> int | None:
        boundaries = [
            boundary
            for world in self.worlds
            for row in world.payload["control_bindings"]
            for boundary in (_next_boundary(row, absolute_tick),)
            if boundary is not None
        ]
        return min(boundaries, default=None)

    def founder_binding_v1(self) -> dict[str, Any]:
        """Return the deterministic founder binding without touching state."""

        registry, observed_sha256 = _load_founder_registry(self.founder_registry_path)
        if observed_sha256 != self.founder_registry_sha256:
            raise ScienceNativeWorldRuntimeError("native founder registry changed after bind")
        assignments: list[list[str]] = []
        live_counts: list[int] = []
        used_ids: set[str] = set()
        exact_ref_ids: set[str] = set()
        for world in self.worlds:
            initialization = world.payload["initialization"]
            exact_profiles = _exact_source_profile_arrays(initialization)
            exact_profile_ids = set(exact_profiles or {})
            if exact_profiles is not None:
                for state in initialization.get("exact_natural_source_states", []):
                    exact_ref_ids.add(str(state["ref_id"]))
            live = initialization.get("initial_live_count")
            population = initialization.get("initial_population")
            composition = initialization.get("exact_founder_composition")
            if (
                type(live) is not int
                or type(population) is not int
                or not 1 <= live <= INITIAL_POPULATION_V2
                or population != live
                or not isinstance(composition, list)
            ):
                raise ScienceNativeWorldRuntimeError(
                    "native W32 initialization requires 1..256 exact live founders"
                )
            assignment: list[str] = []
            for row in composition:
                profile_id = row.get("founder_profile_id")
                count = row.get("count")
                if (
                    not isinstance(profile_id, str)
                    or type(count) is not int
                    or count < 0
                ):
                    raise ScienceNativeWorldRuntimeError("founder composition is invalid")
                if exact_profiles is not None:
                    if profile_id not in exact_profile_ids:
                        raise ScienceNativeWorldRuntimeError(
                            "exact natural initialization contains a proxy founder profile"
                        )
                else:
                    _resolve_profile(registry, profile_id)
                used_ids.add(profile_id)
                assignment.extend([profile_id] * count)
            if len(assignment) != live:
                raise ScienceNativeWorldRuntimeError(
                    "founder composition does not equal initial live count"
                )
            random.Random(
                _seed(world.payload["seed_bindings"], "founder")
            ).shuffle(assignment)
            assignments.append(assignment)
            live_counts.append(live)
        binding = {
            "schema_version": "eiel_science_native_founder_application_v1",
            "registry_path": str(self.founder_registry_path),
            "registry_sha256": self.founder_registry_sha256,
            "control_set_sha256": self.control_set_sha256,
            "used_profile_ids": sorted(used_ids),
            "exact_natural_source_ref_ids": sorted(exact_ref_ids),
            "live_counts": live_counts,
            "ordered_founder_assignments_sha256": _sha256(assignments),
        }
        return {**binding, "binding_sha256": _sha256(binding)}

    def apply_founders_to_state(self, state: Any) -> dict[str, Any]:
        """Apply exact per-world founder compositions to initialized state."""

        import torch

        if state.dim() != 2 or state.size(0) != len(self.worlds):
            raise ScienceNativeWorldRuntimeError("resident state and worlds differ")
        registry, observed_sha256 = _load_founder_registry(self.founder_registry_path)
        if observed_sha256 != self.founder_registry_sha256:
            raise ScienceNativeWorldRuntimeError("native founder registry changed after bind")
        used_ids = sorted(
            {
                row["founder_profile_id"]
                for world in self.worlds
                for row in world.payload["initialization"]["exact_founder_composition"]
            }
        )
        exact_profiles_by_id: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
        exact_mode_by_world: list[bool] = []
        exact_profile_ids_by_world: list[frozenset[str] | None] = []
        for world in self.worlds:
            exact_profiles = _exact_source_profile_arrays(
                world.payload["initialization"]
            )
            exact_mode_by_world.append(exact_profiles is not None)
            exact_profile_ids_by_world.append(
                None if exact_profiles is None else frozenset(exact_profiles)
            )
            if exact_profiles is not None:
                exact_profiles_by_id.update(exact_profiles)
        exact_profile_ids = frozenset(exact_profiles_by_id)
        for world_index, world in enumerate(self.worlds):
            if exact_profile_ids_by_world[world_index] is not None:
                continue
            legacy_profile_ids = frozenset(
                row["founder_profile_id"]
                for row in world.payload["initialization"][
                    "exact_founder_composition"
                ]
            )
            if legacy_profile_ids & exact_profile_ids:
                raise ScienceNativeWorldRuntimeError(
                    "founder profile identity is ambiguous between exact natural "
                    "and legacy materialization"
                )
        arrays_by_id: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
        for profile_id in used_ids:
            exact_array = exact_profiles_by_id.get(profile_id)
            if exact_array is not None:
                arrays_by_id[profile_id] = exact_array
            else:
                # A physical batch may contain exact-natural worlds alongside
                # legacy registered-profile worlds.  Resolve each remaining
                # profile through the immutable registry rather than allowing
                # the presence of one exact world to change the other worlds'
                # initialization semantics.
                arrays_by_id[profile_id] = _profile_arrays(
                    _resolve_profile(registry, profile_id)
                )
        arrays = [arrays_by_id[profile_id] for profile_id in used_ids]
        policy_registry = torch.tensor(
            [row[0] for row in arrays], dtype=torch.int8, device=state.device
        )
        trait_registry = torch.tensor(
            [row[1] for row in arrays], dtype=torch.int16, device=state.device
        )
        index_by_id = {profile_id: index for index, profile_id in enumerate(used_ids)}
        assignments: list[list[int]] = []
        live_counts: list[int] = []
        for world in self.worlds:
            initialization = world.payload["initialization"]
            live = initialization.get("initial_live_count")
            population = initialization.get("initial_population")
            composition = initialization.get("exact_founder_composition")
            if (
                type(live) is not int
                or type(population) is not int
                or not 1 <= live <= INITIAL_POPULATION_V2
                or population != live
                or not isinstance(composition, list)
            ):
                raise ScienceNativeWorldRuntimeError(
                    "native W32 initialization requires 1..256 exact live founders"
                )
            assignment: list[int] = []
            for row in composition:
                profile_id = row.get("founder_profile_id")
                count = row.get("count")
                if profile_id not in index_by_id or type(count) is not int or count < 0:
                    raise ScienceNativeWorldRuntimeError("founder composition is invalid")
                assignment.extend([index_by_id[profile_id]] * count)
            if len(assignment) != live:
                raise ScienceNativeWorldRuntimeError(
                    "founder composition does not equal initial live count"
                )
            # Composition counts are semantic; slot placement is acquisition-
            # independent and is deterministically randomized by the resolved
            # founder seed.  Paired worlds with the same founder seed therefore
            # receive the same slot assignment even when their controls differ.
            random.Random(
                _seed(world.payload["seed_bindings"], "founder")
            ).shuffle(assignment)
            assignments.append(
                assignment + [assignment[0]] * (INITIAL_POPULATION_V2 - live)
            )
            live_counts.append(live)
        assignment_tensor = torch.tensor(
            assignments, dtype=torch.int64, device=state.device
        )
        selected_policy = policy_registry[assignment_tensor].permute(0, 2, 3, 1)
        selected_traits = trait_registry[assignment_tensor].permute(0, 2, 3, 1)
        policy = state[:, 44288:158976].view(torch.int8).reshape(
            len(self.worlds), 2, 112, RESIDENT_CAPACITY_C_V2
        )
        traits = state[:, 158976:191744].view(torch.int16).reshape(
            len(self.worlds), 2, 16, RESIDENT_CAPACITY_C_V2
        )
        policy[:, :, :, :INITIAL_POPULATION_V2].copy_(selected_policy)
        # Registered founder profiles historically inherit the canonical
        # initializer's carrier/sex locus.  An exact natural source state is
        # different: every policy and trait locus is part of the authenticated
        # hereditary bytes and must be copied verbatim.  Preserve the legacy
        # behavior only for worlds without an exact source-state reference.
        preserved_legacy_sex_locus = (
            traits[:, :, 0, :INITIAL_POPULATION_V2].clone()
            if any(not exact for exact in exact_mode_by_world)
            else None
        )
        traits[:, :, :, :INITIAL_POPULATION_V2].copy_(selected_traits)
        if preserved_legacy_sex_locus is not None:
            for world_index, exact in enumerate(exact_mode_by_world):
                if not exact:
                    traits[world_index, :, 0, :INITIAL_POPULATION_V2].copy_(
                        preserved_legacy_sex_locus[
                            world_index, :, :INITIAL_POPULATION_V2
                        ]
                    )
        expressed_policy = state[:, 191744:249088].view(torch.int32).reshape(
            len(self.worlds), 7, 4, RESIDENT_CAPACITY_C_V2
        )
        midpoint_policy = torch.div(
            policy[:, 0, :, :].to(torch.int16)
            + policy[:, 1, :, :].to(torch.int16),
            2,
            rounding_mode="trunc",
        ).to(torch.int32)
        for action in range(7):
            for group in range(4):
                packed = torch.zeros(
                    (len(self.worlds), RESIDENT_CAPACITY_C_V2),
                    dtype=torch.int32,
                    device=state.device,
                )
                for component in range(4):
                    locus = (group * 4 + component) * 7 + action
                    packed |= (midpoint_policy[:, locus, :] & 0xFF) << (component * 8)
                expressed_policy[:, action, group, :].copy_(packed)
        expressed_traits = state[:, 249088:265472].view(torch.int16).reshape(
            len(self.worlds), 16, RESIDENT_CAPACITY_C_V2
        )
        expressed_traits.copy_(
            torch.div(
                traits[:, 0, :, :].to(torch.int32)
                + traits[:, 1, :, :].to(torch.int32),
                2,
                rounding_mode="trunc",
            ).to(torch.int16)
        )
        # The canonical initializer creates P256.  Per-world populations below
        # that carrier maximum are reduced in place without changing geometry.
        occupancy = state[:, 192:2240].view(torch.uint16).reshape(len(self.worlds), 1024)
        positions = state[:, 23744:24768].view(torch.uint16).reshape(
            len(self.worlds), RESIDENT_CAPACITY_C_V2
        )
        alive = state[:, 31936:32000].view(torch.int32).reshape(len(self.worlds), 16)
        header = state[:, :192].view(torch.int64).reshape(len(self.worlds), 24)
        alive_words = []
        for live in live_counts:
            alive_words.append(
                [
                    -1 if live >= (word + 1) * 32 else (
                        (1 << max(0, live - word * 32)) - 1
                        if live > word * 32
                        else 0
                    )
                    for word in range(16)
                ]
            )
        alive.copy_(torch.tensor(alive_words, dtype=torch.int32, device=state.device))
        for world_index, live in enumerate(live_counts):
            if live < INITIAL_POPULATION_V2:
                inactive_sites = positions[
                    world_index, live:INITIAL_POPULATION_V2
                ].to(torch.int64)
                occupancy[world_index, inactive_sites] = 65535
            header[world_index, 2] = live + 1
            header[world_index, 3] = live
        binding = self.founder_binding_v1()
        if binding["live_counts"] != live_counts:
            raise ScienceNativeWorldRuntimeError(
                "native founder binding live counts changed during application"
            )
        binding_body = {
            key: value for key, value in binding.items() if key != "binding_sha256"
        }
        if all(exact_mode_by_world):
            application_mode = "exact_natural_full_genome_v1"
        elif not any(exact_mode_by_world):
            application_mode = "registered_founder_profile_v1"
        else:
            application_mode = "mixed_per_world_v1"
        binding_body.update(
            {
                "hereditary_identity_application_mode": application_mode,
                "preserved_trait_loci_by_world": [
                    [] if exact else [0] for exact in exact_mode_by_world
                ],
            }
        )
        return {**binding_body, "binding_sha256": _sha256(binding_body)}


__all__ = (
    "NATIVE_FOUNDER_REGISTRY_RELATIVE_PATH_V1",
    "NATIVE_WORLD_CONTROL_FIELDS_V1",
    "NATIVE_WORLD_CONTROL_WIDTH_V1",
    "ScienceNativeWorldRuntimeError",
    "ScienceNativeWorldRuntimeProjectionV1",
    "founder_profile_policy_bindings_v1",
    "founder_profile_policy_id_v1",
)
