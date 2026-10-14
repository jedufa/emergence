"""Deterministic data-only fixtures for the Phase F v2 combination matrix.

The matrix owns case identity and intent.  This module materializes that intent
as sixteen canonical agent records per case: slots 0..7 are the named targets
and slots 8..15 are their intact twins.  It owns no simulation, CUDA launch,
checkpoint, mutation RNG, or result authority.

All structural records cross the existing public canonical genome and modular
storage boundaries.  In particular, this module does not reproduce the Phase E
test helpers or interpret the 57 storage planes itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, Mapping, Sequence

from agents.modular_genome_v1 import (
    B2_CAPACITY_PROFILES,
    MODULAR_GENOME_SCHEMA_VERSION,
    canonical_hash_from_genome_dict,
    canonical_payload_from_genome_dict,
)
from agents.structural_identity_v1 import new_native_identity
from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import (
    CASE_COUNT,
    MATRIX_ID,
    build_phase_f_v2_matrix,
    validate_phase_f_v2_matrix,
)
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    neutral_developmental_expression_record,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    neutral_gated_recurrent_record,
)
from evolution.structural_observation_routing_v1 import (
    ROUTING_BLOCK_KEY,
    normalize_routing_block,
)
from evolution.structural_output_routing_v1 import (
    OUTPUT_ROUTING_BLOCK_KEY,
    normalize_output_routing_block,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    normalize_activity_module_params,
)
from perf.cuda_architecture_b_modular_state_abi import (
    PHYSICAL_CONNECTION_SLOTS,
    PHYSICAL_MODULE_SLOTS,
    PackedModularStorageV1,
    pack_modular_storage,
    validate_modular_storage,
)


SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_fixture_case_v1"
REGISTRY_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_fixture_registry_v1"
PANEL_AGENT_COUNT: Final[int] = 16
TARGET_AGENT_COUNT: Final[int] = 8
INTACT_TWIN_COUNT: Final[int] = 8
MODULAR_RNG_DRAW_COUNT: Final[int] = 0

_INTEGRATOR_OPERATOR: Final[str] = "bounded_scalar_integrator_v1"
_INPUT_SEMANTICS: Final[tuple[str, ...]] = ("self_energy", "self_health")
_OUTPUT_SEMANTICS: Final[tuple[str, ...]] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "rest",
)


class PhaseFV2FixtureError(ValueError):
    """A Phase F case is unknown, malformed, or not the frozen fixture."""


def phase_f_fixture_sha256_v2(value: Any) -> str:
    """Return the Phase F canonical digest for fixture data."""

    try:
        return canonical_sha256(value)
    except Exception as exc:
        raise PhaseFV2FixtureError("fixture is not finite canonical JSON data") from exc


def _profile_code(profile_id: str) -> str:
    if profile_id == "b2-12x32-v1":
        return "b12"
    if profile_id == "b2-16x48-v1":
        return "b16"
    raise PhaseFV2FixtureError(f"unsupported fixture profile {profile_id!r}")


def _matrix_and_case(
    case: str | Mapping[str, Any], matrix: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = validate_phase_f_v2_matrix(
        build_phase_f_v2_matrix() if matrix is None else matrix
    )
    cases = {str(row["case_id"]): row for row in frozen["cases"]}
    if isinstance(case, str):
        case_id = case
    elif isinstance(case, Mapping):
        case_id = str(case.get("case_id", ""))
    else:
        raise PhaseFV2FixtureError("case must be a case ID or exact matrix case")
    if case_id not in cases:
        raise PhaseFV2FixtureError(f"unknown Phase F case {case_id!r}")
    expected = cases[case_id]
    if isinstance(case, Mapping) and dict(case) != expected:
        raise PhaseFV2FixtureError("case does not match its frozen matrix binding")
    return frozen, deepcopy(expected)


def _identity(
    *,
    family_id: str,
    profile_id: str,
    variant: int,
    module_index: int,
    parent_module: Mapping[str, Any] | None = None,
    operator_id: str = "phase_f_v2_fixture_materialization",
) -> dict[str, Any]:
    identity = new_native_identity(
        context={
            "campaign_id": MATRIX_ID,
            "branch_id": f"{family_id}-{profile_id}",
            "generation": 0,
            "tick": 0,
            "child_agent_id": variant,
            "parent_agent_ids": [],
            "lineage_token": f"{family_id}-{profile_id}-v{variant}",
        },
        operator_id=operator_id,
        attempt_index=module_index,
        operator_role=f"v{variant}-m{module_index}",
        parent_module=parent_module,
    )
    # Keep every fixture connection within the production locality bound.
    position = [
        round(0.10 + 0.025 * variant + 0.010 * module_index, 8),
        round(0.20 + 0.020 * module_index, 8),
    ]
    identity["topology_pos"] = position
    return identity


def _development(
    gains: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    prerequisites: Sequence[str] = (),
    latch: bool = False,
) -> dict[str, Any]:
    record = neutral_developmental_expression_record()
    record["stage_gains"] = [float(value) for value in gains]
    record["prerequisite_homology_ids"] = sorted(str(value) for value in prerequisites)
    record["latch_dependency_readiness"] = bool(latch)
    return record


def _module(
    *,
    family_id: str,
    profile_id: str,
    variant: int,
    module_index: int,
    gated: bool = False,
    gated_overrides: Mapping[str, float] | None = None,
    integrator_retention: float | None = None,
    integrator_input_gain: float | None = None,
    integrator_output_gain: float | None = None,
    input_semantic: str | None = None,
    input_gain: float = 0.0,
    output_semantic: str | None = None,
    output_gain: float = 0.0,
    output_bindings: Sequence[tuple[str, float]] | None = None,
    development: Mapping[str, Any] | None = None,
    dormant: bool = False,
    identity_parent: Mapping[str, Any] | None = None,
    identity_operator: str = "phase_f_v2_fixture_materialization",
) -> dict[str, Any]:
    identity = _identity(
        family_id=family_id,
        profile_id=profile_id,
        variant=variant,
        module_index=module_index,
        parent_module=identity_parent,
        operator_id=identity_operator,
    )
    signed = -1.0 if (variant + module_index) % 2 else 1.0
    if gated:
        gated_record = neutral_gated_recurrent_record()
        gated_record.update(
            {
                "reset_bias": 0.25 * signed,
                "reset_input_gain": 0.5,
                "reset_state_gain": -0.25,
                "update_bias": -0.125 * signed,
                "update_input_gain": 0.75,
                "update_state_gain": 0.25,
                "candidate_bias": 0.125 * signed,
                "candidate_input_gain": 1.0,
                "candidate_state_gain": -0.5,
                "output_gain": 0.5 * signed,
            }
        )
        if gated_overrides is not None:
            gated_record.update(
                {str(key): float(value) for key, value in gated_overrides.items()}
            )
        params: dict[str, Any] = {
            "live_operator": GATED_OPERATOR,
            "expression_scale": 0.75,
            GATED_RECORD_KEY: gated_record,
        }
    else:
        params = {
            "live_operator": _INTEGRATOR_OPERATOR,
            "expression_scale": 0.75,
            "live_retention": 0.25 + 0.10 * (variant % 4),
            "live_input_gain": 0.5 * signed,
            "live_output_gain": 0.5,
        }
        if integrator_retention is not None:
            params["live_retention"] = float(integrator_retention)
        if integrator_input_gain is not None:
            params["live_input_gain"] = float(integrator_input_gain)
        if integrator_output_gain is not None:
            params["live_output_gain"] = float(integrator_output_gain)
    if input_semantic is not None:
        params[ROUTING_BLOCK_KEY] = normalize_routing_block(
            {
                "bindings": [
                    {
                        "semantic_id": input_semantic,
                        "target_slot": "in",
                        "gain": input_gain,
                        "enabled": True,
                    }
                ]
            }
        )
    resolved_output_bindings = (
        list(output_bindings)
        if output_bindings is not None
        else ([] if output_semantic is None else [(output_semantic, output_gain)])
    )
    if resolved_output_bindings:
        params[OUTPUT_ROUTING_BLOCK_KEY] = normalize_output_routing_block(
            {
                "bindings": [
                    {
                        "semantic_id": semantic_id,
                        "source_slot": "out",
                        "gain": gain,
                        "enabled": True,
                    }
                    for semantic_id, gain in resolved_output_bindings
                ]
            }
        )
    if development is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = deepcopy(dict(development))
    if dormant:
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        }
    params = normalize_activity_module_params(params)
    return {
        "module_id": f"m{module_index:03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(
    index: int, source: Mapping[str, Any], target: Mapping[str, Any], weight: float
) -> dict[str, Any]:
    return {
        "connection_id": f"c{index:03d}",
        "src_module_id": str(source["module_id"]),
        "dst_module_id": str(target["module_id"]),
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "phase_f_v2_previous_tick",
        "weight": float(weight),
    }


def _base_structure(
    family_id: str, profile_id: str, variant: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_semantic = _INPUT_SEMANTICS[variant % len(_INPUT_SEMANTICS)]
    output_semantic = _OUTPUT_SEMANTICS[variant % len(_OUTPUT_SEMANTICS)]
    signed = -1.0 if variant % 2 else 1.0

    def make(index: int, **kwargs: Any) -> dict[str, Any]:
        return _module(
            family_id=family_id,
            profile_id=profile_id,
            variant=variant,
            module_index=index,
            **kwargs,
        )

    modules: list[dict[str, Any]] = []
    connections: list[dict[str, Any]] = []
    neutral_gated = {
        "reset_bias": 0.0,
        "reset_input_gain": 0.0,
        "reset_state_gain": 0.0,
        "update_bias": 0.0,
        "update_input_gain": 0.0,
        "update_state_gain": 0.0,
        "candidate_bias": 0.0,
        "candidate_input_gain": 0.0,
        "candidate_state_gain": 0.0,
        "output_gain": 0.0,
    }
    if family_id == "f01":
        if variant in {0, 1}:
            return [], []
        if variant == 2:
            modules = [
                make(
                    0,
                    integrator_input_gain=0.0,
                    integrator_output_gain=0.0,
                )
            ]
        elif variant == 3:
            modules = [make(0, gated=True, gated_overrides=neutral_gated)]
        elif variant == 4:
            modules = [
                make(
                    0,
                    input_semantic="self_energy",
                    input_gain=0.0,
                    output_semantic="move_north",
                    output_gain=0.5,
                )
            ]
        elif variant == 5:
            modules = [
                make(
                    0,
                    input_semantic="self_energy",
                    input_gain=0.5,
                    output_semantic="move_north",
                    output_gain=0.0,
                    development=_development((0.0, 0.0, 0.0)),
                )
            ]
        elif variant == 6:
            modules = [
                make(
                    0,
                    gated=True,
                    input_semantic="self_health",
                    input_gain=0.5,
                    output_semantic="rest",
                    output_gain=0.5,
                    dormant=True,
                )
            ]
        else:
            original = make(
                0,
                integrator_input_gain=0.0,
                integrator_output_gain=0.0,
            )
            duplicate = make(
                1,
                integrator_input_gain=0.0,
                integrator_output_gain=0.0,
                identity_parent=original,
                identity_operator="duplicate_with_decay",
            )
            modules = [original, duplicate]
    elif family_id == "f02":
        if variant < 4:
            semantics = (
                ("self_energy", 0.5, 0.25),
                ("self_energy", -0.5, -0.25),
                ("self_health", 0.5, -0.25),
                ("self_health", -0.5, 0.25),
            )[variant]
            modules = [
                make(
                    0,
                    input_semantic=semantics[0],
                    input_gain=semantics[1],
                    output_semantic="move_north",
                    output_gain=0.5,
                )
            ]
        elif variant in {4, 5}:
            target = make(
                0,
                output_semantic="move_east",
                output_gain=0.5,
            )
            modules = [target]
            connections = [
                _connection(0, target, target, 0.5 if variant == 4 else -0.5)
            ]
        elif variant == 6:
            modules = [
                make(
                    0,
                    input_semantic="self_energy",
                    input_gain=0.5,
                    output_semantic="move_west",
                    output_gain=0.5,
                    development=_development((0.5, 0.75, 1.0)),
                )
            ]
        else:
            modules = [
                make(
                    0,
                    input_semantic="self_health",
                    input_gain=-0.5,
                    output_semantic="rest",
                    output_gain=0.5,
                )
            ]
    elif family_id == "f03":
        gated_settings = {
            0: {"update_bias": -4.0, "candidate_state_gain": 0.5},
            1: {"update_bias": 4.0, "candidate_input_gain": 1.0},
            2: {
                "reset_bias": -4.0,
                "update_bias": 4.0,
                "candidate_state_gain": 1.0,
            },
            3: {"candidate_bias": 1.0},
            4: {"candidate_bias": -1.0},
            5: {"candidate_input_gain": 1.0},
            6: {"candidate_input_gain": 1.0},
            7: {"candidate_state_gain": 1.0},
        }[variant]
        if variant < 6:
            modules = [
                make(
                    0,
                    gated=True,
                    gated_overrides=gated_settings,
                    input_semantic="self_energy",
                    input_gain=0.5,
                    output_semantic="move_north",
                    output_gain=0.5,
                    development=_development((1.0, 1.0, 1.0)),
                )
            ]
        else:
            root = make(
                0,
                development=_development((1.0, 1.0, 1.0)),
                dormant=variant == 7,
            )
            leaf = make(
                1,
                gated=True,
                gated_overrides=gated_settings,
                input_semantic="self_health" if variant == 6 else None,
                input_gain=0.5,
                output_semantic="move_south" if variant == 6 else "rest",
                output_gain=0.5,
                development=_development(
                    (0.0, 1.0, 1.0),
                    prerequisites=(root["homology_id"],),
                ),
            )
            modules = [root, leaf]
            connections = [] if variant == 6 else [_connection(0, root, leaf, -0.5)]
    elif family_id == "f04":
        first_output: tuple[str, float] | None = None
        second_output: tuple[str, float] | None = None
        if variant == 0:
            first_output = ("move_north", 0.5)
        elif variant == 1:
            second_output = ("move_north", 0.5)
        elif variant == 2:
            first_output = second_output = ("move_east", 0.5)
        elif variant == 3:
            first_output = ("move_east", 0.5)
            second_output = ("rest", 0.5)
        elif variant == 5:
            first_output = ("move_west", 0.5)
            second_output = ("move_west", -0.5)
        else:
            second_output = ("move_south", 0.5)
        first = make(
            0,
            gated=variant >= 2,
            input_semantic=input_semantic,
            input_gain=0.5,
            output_semantic=None if first_output is None else first_output[0],
            output_gain=0.0 if first_output is None else first_output[1],
            dormant=variant == 6,
        )
        second = make(
            1,
            gated=variant in {1, 3, 4, 5},
            output_semantic=None if second_output is None else second_output[0],
            output_gain=0.0 if second_output is None else second_output[1],
            development=(_development((0.0, 0.0, 0.0)) if variant == 7 else None),
        )
        modules = [first, second]
        connections = [_connection(0, first, second, -0.5 if variant == 4 else 0.5)]
    elif family_id == "f05":
        first = make(
            0,
            gated=False,
            input_semantic="self_energy",
            input_gain=0.5,
            dormant=variant == 4,
            output_bindings=([("move_north", 0.5)] if variant == 7 else None),
        )
        second = make(
            1,
            gated=variant in {4, 6},
            input_semantic="self_health" if variant == 6 else None,
            input_gain=-0.5 if variant == 6 else 0.0,
            output_bindings=([("rest", 0.5)] if variant == 7 else None),
            development=(_development((0.0, 0.0, 0.0)) if variant == 5 else None),
        )
        third_development = (
            _development(
                (1.0, 1.0, 1.0),
                prerequisites=(first["homology_id"], second["homology_id"]),
            )
            if variant == 3
            else None
        )
        third = make(
            2,
            gated=False,
            output_semantic="move_east" if variant == 7 else output_semantic,
            output_gain=0.5 * signed,
            development=third_development,
        )
        modules = [first, second, third]
        if variant in {1, 3}:
            connections = [
                _connection(0, first, third, 0.5),
                _connection(1, second, third, 0.5 * signed),
            ]
        elif variant == 2:
            connections = [
                _connection(0, first, second, 0.5),
                _connection(1, first, third, -0.5),
            ]
        else:
            connections = [
                _connection(0, first, second, 0.5),
                _connection(1, second, third, 0.5 * signed),
            ]
    elif family_id == "f06":
        count = 2 if variant < 3 else 3
        modules = [
            make(
                index,
                gated=(variant + index) % 2 == 0,
                input_semantic=input_semantic if index == 0 else None,
                input_gain=0.5,
                output_semantic=output_semantic if index == count - 1 else None,
                output_gain=0.5,
                dormant=variant == 6 and index == 1,
            )
            for index in range(count)
        ]
        for index in range(count):
            polarity = -1.0 if variant in {1, 4} else 1.0
            if variant == 2 and index == 1:
                polarity = -1.0
            if variant == 5:
                polarity = (1.0, 1.0, -2.0)[index]
            connections.append(
                _connection(
                    index, modules[index], modules[(index + 1) % count], 0.5 * polarity
                )
            )
    elif family_id == "f07":
        root = make(
            0,
            input_semantic=input_semantic,
            input_gain=0.5,
            development=_development(
                (0.0, 0.0, 0.0) if variant == 7 else (1.0, 1.0, 1.0)
            ),
            dormant=variant == 3,
        )
        gains = (
            (1.0, 0.0, 0.0)
            if variant == 0
            else (0.0, 1.0, 1.0)
            if variant in {1, 4, 5, 6}
            else (0.0, 0.0, 1.0)
            if variant == 2
            else (0.5, 0.5, 0.5)
        )
        leaf = make(
            1,
            gated=variant % 2 == 1,
            output_semantic=output_semantic,
            output_gain=0.5,
            development=_development(
                gains,
                prerequisites=(root["homology_id"],),
                latch=variant in {5, 6, 7},
            ),
            dormant=variant == 6,
        )
        modules = [root, leaf]
        connections = [_connection(0, root, leaf, 0.5)]
    elif family_id == "f08":
        first_dormant = variant in {0, 1, 2, 3, 4}
        first = make(
            0,
            gated=variant in {1, 7},
            input_semantic=input_semantic,
            input_gain=0.5,
            output_semantic=output_semantic,
            output_gain=0.5,
            dormant=first_dormant,
        )
        modules = [first]
        if variant in {4, 5}:
            second = make(
                1,
                output_semantic="rest",
                output_gain=0.5,
                dormant=variant == 5,
            )
            modules.append(second)
            connections = [_connection(0, first, second, 0.5)]
    elif family_id == "f09":
        if variant == 3:
            modules = [make(0, integrator_output_gain=0.0)]
        else:
            active = make(
                0,
                input_semantic="self_energy",
                input_gain=0.5,
                output_semantic="move_north",
                output_gain=0.5,
            )
            target = make(
                1,
                gated=variant == 1,
                gated_overrides=({"output_gain": 0.0} if variant == 1 else None),
                integrator_output_gain=0.0,
                input_semantic="self_health" if variant == 4 else None,
                input_gain=0.5 if variant == 4 else 0.0,
                output_semantic="rest" if variant == 4 else None,
                output_gain=0.5 if variant == 4 else 0.0,
                dormant=variant == 0,
            )
            modules = [active, target]
            if variant == 2:
                dependent = make(
                    2,
                    output_semantic="move_east",
                    output_gain=0.5,
                    development=_development(prerequisites=(target["homology_id"],)),
                )
                modules.append(dependent)
            if variant == 5:
                connections = [
                    _connection(0, active, target, 0.5),
                    _connection(1, target, active, -0.5),
                ]
            elif variant == 6:
                third = make(2, integrator_output_gain=0.0)
                fourth = make(3, integrator_output_gain=0.0)
                modules.extend([third, fourth])
            elif variant == 7:
                replacement = make(
                    2,
                    integrator_input_gain=0.0,
                    integrator_output_gain=0.0,
                    identity_parent=target,
                    identity_operator="duplicate_with_decay",
                )
                modules.append(replacement)
    elif family_id == "f10":
        count = 1 if variant in {0, 1} else min(8, 1 + variant)
        for index in range(count):
            development = None
            if variant == 5 and index == count - 1 and modules:
                development = _development(prerequisites=(modules[0]["homology_id"],))
            modules.append(
                make(
                    index,
                    gated=(
                        variant == 1 or (variant >= 2 and (variant + index) % 3 == 0)
                    ),
                    input_semantic=input_semantic if index == 0 else None,
                    input_gain=0.0 if variant in {0, 1, 2, 7} else 0.25,
                    output_semantic=output_semantic if index == count - 1 else None,
                    output_gain=0.0,
                    development=development,
                    dormant=variant == 6 and index == count - 1,
                )
            )
        connections = [
            _connection(index, modules[index], modules[index + 1], 0.0)
            for index in range(max(0, count - 1))
        ]
    elif family_id == "f11":
        first = make(
            0,
            input_semantic="self_energy",
            input_gain=0.5,
            output_semantic="move_north" if variant == 5 else None,
            output_gain=0.5,
            development=(_development((1.0, 1.0, 1.0)) if variant == 6 else None),
            dormant=variant == 6,
        )
        second = make(
            1,
            gated=True,
            input_semantic="self_health",
            input_gain=-0.5,
            output_semantic="rest",
            output_gain=0.5,
            development=(
                _development((0.5, 1.0, 1.0), prerequisites=(first["homology_id"],))
                if variant == 6
                else None
            ),
        )
        modules = [first, second]
        connections = [_connection(0, first, second, 0.5)]
        if variant == 5:
            third = make(
                2,
                output_bindings=[("move_east", 0.5), ("rest", -0.25)],
            )
            modules.append(third)
            connections.append(_connection(1, second, third, 0.25))
    elif family_id == "f12":
        if variant == 0:
            return [], []
        if variant == 1:
            modules = [
                make(
                    0,
                    gated=True,
                    input_semantic="self_energy",
                    input_gain=0.5,
                    output_semantic="rest",
                    output_gain=0.5,
                )
            ]
        elif variant in {2, 7}:
            first = make(0, input_semantic="self_energy", input_gain=0.5)
            second = make(
                1,
                gated=variant == 7,
                output_semantic="move_east",
                output_gain=0.5,
            )
            modules = [first, second]
            connections = [_connection(0, first, second, 0.5)]
        elif variant == 3:
            root = make(0, development=_development())
            leaf = make(
                1,
                output_semantic="rest",
                output_gain=0.5,
                development=_development(
                    (0.0, 1.0, 1.0), prerequisites=(root["homology_id"],)
                ),
            )
            modules = [root, leaf]
            connections = [_connection(0, root, leaf, 0.5)]
        elif variant == 4:
            modules = [
                make(
                    0,
                    gated=True,
                    output_semantic="rest",
                    output_gain=0.5,
                    dormant=True,
                )
            ]
        elif variant == 5:
            modules = [
                make(
                    index,
                    gated=index == 1,
                    gated_overrides=({"output_gain": 0.0} if index == 1 else None),
                    integrator_output_gain=0.0,
                )
                for index in range(4)
            ]
            connections = [
                _connection(index, modules[index], modules[index + 1], 0.0)
                for index in range(3)
            ]
        else:
            active = make(0, output_semantic="move_north", output_gain=0.5)
            eligible = make(1, integrator_output_gain=0.0, dormant=True)
            replacement = make(
                2,
                integrator_input_gain=0.0,
                integrator_output_gain=0.0,
                identity_parent=eligible,
                identity_operator="duplicate_with_decay",
            )
            modules = [active, eligible, replacement]
    else:
        raise PhaseFV2FixtureError(f"unknown fixture family {family_id!r}")
    return modules, connections


def _canonical_genome(
    *,
    profile_id: str,
    modules: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        return canonical_payload_from_genome_dict(
            {
                "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
                "genes": {},
                "modules": deepcopy(list(modules)),
                "connections": deepcopy(list(connections)),
                "b2_caps": deepcopy(B2_CAPACITY_PROFILES[profile_id]),
                "b2_invalid_policy": "reject",
            }
        )
    except Exception as exc:
        raise PhaseFV2FixtureError(
            "fixture genome failed production canonicalization"
        ) from exc


def _runtime_maps(
    genome: Mapping[str, Any], family_id: str, variant: int
) -> tuple[dict[str, float], dict[str, float], dict[str, bool], dict[str, float]]:
    modules = list(genome["modules"])
    states: dict[str, float] = {}
    outputs: dict[str, float] = {}
    latches: dict[str, bool] = {}
    effective: dict[str, float] = {}
    for index, module in enumerate(modules):
        instance_id = str(module["module_instance_id"])
        sign = -1.0 if (variant + index) % 2 else 1.0
        states[instance_id] = float(sign * (0.10 + 0.05 * index))
        outputs[instance_id] = float(sign * (0.05 + 0.025 * index))
        params = dict(module.get("params", {}))
        development = params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        if isinstance(development, Mapping):
            if bool(development.get("latch_dependency_readiness", False)):
                latches[instance_id] = False
            effective[instance_id] = float(development["stage_gains"][0])

    def instance(index: int) -> str:
        return str(modules[index]["module_instance_id"])

    if family_id == "f01":
        if variant in {2, 3, 7}:
            for key in states:
                states[key] = 0.0
                outputs[key] = 0.0
        elif variant == 6 and modules:
            states[instance(0)] = 0.4
            outputs[instance(0)] = 0.3
    elif family_id == "f02" and modules:
        signed_states = {0: 0.25, 1: -0.25, 2: -0.25, 3: 0.25, 7: -0.25}
        if variant in signed_states:
            states[instance(0)] = signed_states[variant]
        if variant in {4, 5}:
            outputs[instance(0)] = 0.4 if variant == 4 else -0.4
        if variant == 6:
            effective[instance(0)] = 0.5
    elif family_id == "f03" and modules:
        states[instance(-1)] = -0.25 if variant == 4 else 0.25
        if variant == 6:
            effective[instance(0)] = 1.0
            effective[instance(1)] = 1.0
        elif variant == 7:
            effective[instance(0)] = 1.0
            effective[instance(1)] = 0.0
    elif family_id == "f04" and variant == 7:
        effective[instance(1)] = 0.0
    elif family_id == "f05":
        if variant == 3:
            effective[instance(2)] = 1.0
        elif variant == 5:
            effective[instance(1)] = 0.0
    elif family_id == "f07":
        root_id = instance(0)
        leaf_id = instance(1)
        if variant == 0:
            effective.update({root_id: 1.0, leaf_id: 1.0})
        elif variant in {1, 4}:
            effective.update({root_id: 1.0, leaf_id: 1.0})
        elif variant == 2:
            effective.update({root_id: 1.0, leaf_id: 0.0})
            states[leaf_id] = 0.4
            outputs[leaf_id] = 0.3
        elif variant == 3:
            effective.update({root_id: 1.0, leaf_id: 0.0})
        elif variant == 5:
            effective.update({root_id: 1.0, leaf_id: 1.0})
            latches[leaf_id] = True
        elif variant == 6:
            effective.update({root_id: 1.0, leaf_id: 1.0})
            latches[leaf_id] = True
        else:
            effective.update({root_id: 0.0, leaf_id: 0.5})
            latches[leaf_id] = True
    elif family_id == "f08" and modules:
        if variant == 0:
            states[instance(0)] = 0.4
        elif variant == 1:
            states[instance(0)] = -0.4
        elif variant == 2:
            outputs[instance(0)] = 0.35
        elif variant == 3:
            states[instance(0)] = 0.25
            outputs[instance(0)] = -0.2
    elif family_id == "f11" and variant == 6:
        effective[instance(0)] = 1.0
        effective[instance(1)] = 0.5
    elif family_id == "f11" and variant == 1:
        states[instance(0)] = 0.1
        states[instance(1)] = -0.15
        outputs[instance(0)] = 0.05
        outputs[instance(1)] = -0.075
    elif family_id == "f12" and variant == 3:
        effective[instance(0)] = 1.0
        effective[instance(1)] = 0.0
    return states, outputs, latches, effective


def _prepared_variant_recipe(
    *,
    family_id: str,
    profile_id: str,
    variant: int,
    genome: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Describe a boundary setup without claiming that the boundary ran."""

    modules = list(genome["modules"])
    module_ids = [str(module["module_id"]) for module in modules]
    recipes: dict[str, tuple[str, ...]] = {
        "f09": (
            "eligible_dormant_prune_candidate",
            "eligible_output_neutral_prune_candidate",
            "dependency_blocked_prune_request",
            "minimum_module_invariant_rejection",
            "incident_binding_cleanup_candidate",
            "incident_connection_cleanup_candidate",
            "deterministic_repack_candidate",
            "freed_slot_new_identity_candidate",
        ),
        "f11": (
            "legacy_profile_mixed_source",
            "no_migration_matched_control",
            "unauthorized_request_rejection",
            "wrong_boundary_or_digest_rejection",
            "canonical_neutral_tail_migration_candidate",
            "routing_connection_migration_candidate",
            "development_dependency_dormancy_cost_migration_candidate",
            "first_continuation_identity_preservation_candidate",
        ),
        "f12": (
            "neutral_empty_checkpoint_representative",
            "single_gated_checkpoint_representative",
            "two_module_chain_checkpoint_representative",
            "development_dependency_checkpoint_representative",
            "dormancy_checkpoint_representative",
            "structural_cost_checkpoint_representative",
            "prune_reuse_checkpoint_representative",
            "capacity_migration_checkpoint_representative",
        ),
    }
    if family_id not in recipes:
        return None
    return {
        "schema_version": "eiel_phase_f_v2_prepared_boundary_recipe_v1",
        "recipe": recipes[family_id][variant],
        "profile_id": profile_id,
        "module_ids": module_ids,
        "connection_ids": [
            str(connection["connection_id"]) for connection in genome["connections"]
        ],
        "prepared_only": True,
        "lifecycle_executed": False,
        "migration_executed": False,
        "checkpoint_authority_claimed": False,
        "modular_rng_draw_count": 0,
    }


def _remove_last_module(
    genome: dict[str, Any], runtime: tuple[dict[str, Any], ...]
) -> tuple[str, str] | None:
    modules = list(genome["modules"])
    if len(modules) <= 1:
        return None
    removed = modules[-1]
    removed_local = str(removed["module_id"])
    removed_instance = str(removed["module_instance_id"])
    genome["modules"] = modules[:-1]
    genome["connections"] = [
        connection
        for connection in genome["connections"]
        if removed_local
        not in {str(connection["src_module_id"]), str(connection["dst_module_id"])}
    ]
    for values in runtime:
        values.pop(removed_instance, None)
    return removed_local, removed_instance


def _apply_intervention(
    *,
    action: str,
    role_id: str,
    family_id: str,
    profile_id: str,
    variant: int,
    genome: dict[str, Any],
    runtime: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []
    lowered = action.lower()
    if family_id in {"f09", "f11", "f12"} or (
        family_id != "f10" and ("prune" in lowered or "reuse" in lowered)
    ):
        return (
            genome,
            runtime,
            [
                {
                    "operation": "prepared_boundary_recipe",
                    "recipe": action,
                    "hereditary_change": False,
                    "executed": False,
                }
            ],
        )
    if role_id in {"r0", "r1"}:
        return genome, runtime, operations

    states, outputs, latches, effective = runtime
    modules = list(genome["modules"])

    if family_id == "f10" and action == "prune_output_neutral_structure":
        removed_identity = _remove_last_module(genome, runtime)
        modules = list(genome["modules"])
        if removed_identity is not None:
            removed_module_id, removed_instance_id = removed_identity
            operations.append(
                {
                    "operation": "prepared_fixture_prune",
                    "removed_module_id": removed_module_id,
                    "removed_module_instance_id": removed_instance_id,
                }
            )

    if (
        family_id == "f01"
        and action == "neutral_duplicate_and_clean_free_slot_panel"
        and modules
    ):
        parent = modules[0]
        duplicate = _module(
            family_id=family_id,
            profile_id=profile_id,
            variant=variant,
            module_index=len(modules),
            integrator_input_gain=0.0,
            integrator_output_gain=0.0,
            identity_parent=parent,
            identity_operator="duplicate_with_decay",
        )
        modules.append(duplicate)
        genome["modules"] = modules
        duplicate_instance_id = str(duplicate["module_instance_id"])
        states[duplicate_instance_id] = 0.0
        outputs[duplicate_instance_id] = 0.0
        operations.append(
            {
                "operation": "add_neutral_duplicate",
                "parent_module_instance_id": str(parent["module_instance_id"]),
                "duplicate_module_id": str(duplicate["module_id"]),
                "duplicate_module_instance_id": duplicate_instance_id,
            }
        )

    if "remove_input" in lowered or "zero_binding" in lowered:
        changed = []
        for module in modules:
            params = dict(module["params"])
            if ROUTING_BLOCK_KEY in params:
                params.pop(ROUTING_BLOCK_KEY)
                module["params"] = params
                changed.append(str(module["module_id"]))
        if changed:
            operations.append(
                {"operation": "remove_input_bindings", "module_ids": changed}
            )

    if (
        "remove_output" in lowered
        or "remove_downstream_output" in lowered
        or "zero_development_and_dormancy" in lowered
    ):
        changed = []
        for module in modules[-1:]:
            params = dict(module["params"])
            if OUTPUT_ROUTING_BLOCK_KEY in params:
                params.pop(OUTPUT_ROUTING_BLOCK_KEY)
                module["params"] = params
                changed.append(str(module["module_id"]))
        if changed:
            operations.append(
                {"operation": "remove_output_bindings", "module_ids": changed}
            )

    if "reset_state" in lowered or "reset_member_state" in lowered:
        reset_ids = list(states)
        if "reset_member_state" in lowered and modules:
            reset_ids = [str(modules[0]["module_instance_id"])]
        for key in reset_ids:
            states[key] = 0.0
            outputs[key] = 0.0
        operations.append(
            {
                "operation": "reset_runtime_state",
                "module_instance_ids": sorted(reset_ids),
            }
        )

    if ("block_dependency" in lowered or "block_prerequisite" in lowered) and modules:
        target = modules[-1]
        params = dict(target["params"])
        development = params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        prerequisite_ids = (
            list(development.get("prerequisite_homology_ids", []))
            if isinstance(development, Mapping)
            else []
        )
        prerequisite = next(
            (
                module
                for module in modules
                if str(module["homology_id"]) in prerequisite_ids
            ),
            None,
        )
        if prerequisite is not None:
            prerequisite_params = dict(prerequisite["params"])
            prerequisite_params[ACTIVITY_RECORD_KEY] = {
                "schema_version": ACTIVITY_SCHEMA_VERSION,
                "dormant": True,
            }
            prerequisite["params"] = normalize_activity_module_params(
                prerequisite_params
            )
            target_instance = str(target["module_instance_id"])
            effective[target_instance] = 0.0
            if target_instance in latches:
                latches[target_instance] = False
            operations.append(
                {
                    "operation": "block_dependency_readiness",
                    "target_module_id": str(target["module_id"]),
                    "prerequisite_module_id": str(prerequisite["module_id"]),
                }
            )

    if (
        "zero_development" in lowered or "set_development_expression_zero" in lowered
    ) and modules:
        target = modules[-1]
        params = dict(target["params"])
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development((0.0, 0.0, 0.0))
        target["params"] = params
        effective[str(target["module_instance_id"])] = 0.0
        operations.append(
            {
                "operation": "zero_developmental_expression",
                "module_id": str(target["module_id"]),
            }
        )

    if "dormancy" in lowered or "dormant" in lowered:
        if modules:
            target = modules[-1]
            params = dict(target["params"])
            params[ACTIVITY_RECORD_KEY] = {
                "schema_version": ACTIVITY_SCHEMA_VERSION,
                "dormant": True,
            }
            target["params"] = normalize_activity_module_params(params)
            operations.append(
                {"operation": "enable_dormancy", "module_id": str(target["module_id"])}
            )

    if (
        "remove_chain_connection" in lowered
        or "remove_first_structural_connection" in lowered
        or "remove_recurrent_connection" in lowered
        or "remove_zero_contribution_connection" in lowered
    ) and genome["connections"]:
        removed = genome["connections"].pop(0)
        operations.append(
            {
                "operation": "remove_connection",
                "connection_id": str(removed["connection_id"]),
            }
        )

    # Several boundary recipes intentionally change only the execution
    # boundary, profile, checkpoint, or reporting request.  Their explicit
    # operation preserves that fact without inventing hereditary changes.
    if not operations:
        operations.append(
            {
                "operation": "boundary_or_reporting_recipe",
                "recipe": action,
                "hereditary_change": False,
            }
        )
    canonical = _canonical_genome(
        profile_id=profile_id,
        modules=genome["modules"],
        connections=genome["connections"],
    )
    valid_ids = {str(module["module_instance_id"]) for module in canonical["modules"]}
    cleaned_runtime = tuple(
        dict(sorted((key, value) for key, value in values.items() if key in valid_ids))
        for values in runtime
    )
    return canonical, cleaned_runtime, operations


def _record(
    *,
    agent_slot: int,
    genome: Mapping[str, Any],
    runtime: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    states, outputs, latches, effective = runtime
    return {
        "agent_id": 10_000 + agent_slot,
        "agent_slot_token": 20_000 + agent_slot,
        "genome": deepcopy(dict(genome)),
        "structural_live_module_state": deepcopy(states),
        "structural_live_module_outputs": deepcopy(outputs),
        "structural_development_latch_by_instance_id": deepcopy(latches),
        "structural_development_effective_gain_by_instance_id": deepcopy(effective),
        "gated_recurrent_enabled": any(
            str(dict(module.get("params", {})).get("live_operator", ""))
            == GATED_OPERATOR
            for module in genome["modules"]
        ),
        "saturated_or_rejected_activity_count": 0,
    }


def _assert_canonical_tail(packed: PackedModularStorageV1) -> None:
    validate_modular_storage(packed)
    layout = packed.layout
    for field in layout.fields:
        tensor = packed.tensors[field.name]
        tail = None
        if len(field.shape) >= 2 and field.shape[1] == PHYSICAL_MODULE_SLOTS:
            tail = tensor[:, layout.logical_module_slots :]
        elif len(field.shape) >= 2 and field.shape[1] == PHYSICAL_CONNECTION_SLOTS:
            tail = tensor[:, layout.logical_connection_slots :]
        if (
            tail is not None
            and tail.numel()
            and not bool((tail == field.fill_value).all().item())
        ):
            raise PhaseFV2FixtureError(f"noncanonical physical tail in {field.name}")


def _materialize_unchecked(
    matrix: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any]:
    family_id = str(case["family_id"])
    profile_id = str(case["profile_id"])
    role_id = str(case["role_id"])
    role_action = str(case["role_contract"]["action"])
    family = next(row for row in matrix["families"] if row["family_id"] == family_id)
    descriptions = {
        str(row["variant_id"]): str(row["description"])
        for row in family["panel_variants"]
    }
    agents: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    agent_metadata: list[dict[str, Any]] = []
    for agent_slot in range(PANEL_AGENT_COUNT):
        variant = agent_slot % TARGET_AGENT_COUNT
        targeted = agent_slot < TARGET_AGENT_COUNT
        modules, connections = _base_structure(family_id, profile_id, variant)
        intact = _canonical_genome(
            profile_id=profile_id, modules=modules, connections=connections
        )
        runtime = _runtime_maps(intact, family_id, variant)
        materialized = deepcopy(intact)
        operations: list[dict[str, Any]] = []
        if targeted:
            materialized, runtime, operations = _apply_intervention(
                action=role_action,
                role_id=role_id,
                family_id=family_id,
                profile_id=profile_id,
                variant=variant,
                genome=materialized,
                runtime=runtime,
            )
        record = _record(agent_slot=agent_slot, genome=materialized, runtime=runtime)
        source_records.append(record)
        agent_metadata.append(
            {
                "agent_slot": agent_slot,
                "panel_half": "target" if targeted else "intact_twin",
                "variant_id": f"v{variant}",
                "variant_description": descriptions[f"v{variant}"],
                "twin_slot": agent_slot + 8 if targeted else agent_slot - 8,
                "intact_genome_sha256": canonical_hash_from_genome_dict(intact),
                "prepared_boundary_recipe": _prepared_variant_recipe(
                    family_id=family_id,
                    profile_id=profile_id,
                    variant=variant,
                    genome=intact,
                ),
                "intervention": {
                    "schema_version": "eiel_phase_f_v2_explicit_intervention_v1",
                    "targeted": targeted,
                    "action": role_action if targeted else "intact_twin",
                    "operations": operations,
                    "modular_rng_draw_count": MODULAR_RNG_DRAW_COUNT,
                },
            }
        )

    packed = pack_modular_storage(
        source_records, agent_capacity=PANEL_AGENT_COUNT, profile_id=profile_id
    )
    _assert_canonical_tail(packed)
    for metadata, normalized_record in zip(
        agent_metadata, packed.records_sidecar, strict=True
    ):
        if normalized_record is None:
            raise PhaseFV2FixtureError(
                "fixture unexpectedly contains an empty agent slot"
            )
        material = {
            **metadata,
            "materialized_genome_sha256": canonical_hash_from_genome_dict(
                normalized_record["genome"]
            ),
            "record": deepcopy(normalized_record),
        }
        material["agent_fixture_sha256"] = phase_f_fixture_sha256_v2(material)
        agents.append(material)

    layout = packed.layout.to_dict()
    material = {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "case_sha256": str(case["case_sha256"]),
        "matrix_id": str(matrix["matrix_id"]),
        "matrix_sha256": str(matrix["matrix_sha256"]),
        "fixture_blueprint_sha256": str(
            case["fixture_blueprint"]["fixture_blueprint_sha256"]
        ),
        "family_id": family_id,
        "profile_id": profile_id,
        "profile_code": _profile_code(profile_id),
        "role_id": role_id,
        "role_action": role_action,
        "panel_agent_count": PANEL_AGENT_COUNT,
        "target_agent_count": TARGET_AGENT_COUNT,
        "intact_twin_count": INTACT_TWIN_COUNT,
        "physical_module_slots": int(layout["physical_module_slots"]),
        "physical_connection_slots": int(layout["physical_connection_slots"]),
        "logical_module_slots": int(layout["logical_module_slots"]),
        "logical_connection_slots": int(layout["logical_connection_slots"]),
        "layout_manifest_sha256": str(layout["manifest_sha256"]),
        "packed_storage_sha256": str(packed.packed_sha256),
        "canonical_tail_verified": True,
        "modular_rng_draw_count": MODULAR_RNG_DRAW_COUNT,
        "execution_recipe": {
            "schema_version": "eiel_phase_f_v2_fixture_execution_recipe_v1",
            "scope": "whole_single_cell_invocation",
            "modular_enabled": role_action != "modular_disabled_legacy",
            "deterministic_actions": True,
            "births": False,
            "deaths": False,
            "turnover": False,
            "simulation_executed": False,
        },
        "agents": agents,
    }
    material["fixture_sha256"] = phase_f_fixture_sha256_v2(material)
    return material


def materialize_phase_f_case_v2(
    case: str | Mapping[str, Any],
    *,
    matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one exact matrix case as a detached data-only panel."""

    frozen, exact_case = _matrix_and_case(case, matrix)
    return _materialize_unchecked(frozen, exact_case)


def validate_phase_f_fixture_case_v2(
    value: Any, *, matrix: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed unless ``value`` is exactly the deterministic case fixture."""

    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhaseFV2FixtureError("fixture case must be a string-keyed mapping")
    case_id = value.get("case_id")
    if not isinstance(case_id, str):
        raise PhaseFV2FixtureError("fixture case_id must be a string")
    expected = materialize_phase_f_case_v2(case_id, matrix=matrix)
    if dict(value) != expected:
        raise PhaseFV2FixtureError(
            "fixture does not match deterministic case materialization"
        )
    return deepcopy(expected)


def _build_registry_unchecked(matrix: Mapping[str, Any]) -> dict[str, Any]:
    cases = [_materialize_unchecked(matrix, case) for case in matrix["cases"]]
    material = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "matrix_id": str(matrix["matrix_id"]),
        "matrix_sha256": str(matrix["matrix_sha256"]),
        "case_count": len(cases),
        "panel_agent_count": PANEL_AGENT_COUNT,
        "target_agent_count_per_case": TARGET_AGENT_COUNT,
        "intact_twin_count_per_case": INTACT_TWIN_COUNT,
        "physical_module_slots": PHYSICAL_MODULE_SLOTS,
        "physical_connection_slots": PHYSICAL_CONNECTION_SLOTS,
        "modular_rng_draw_count": MODULAR_RNG_DRAW_COUNT,
        "cases": cases,
    }
    material["registry_sha256"] = phase_f_fixture_sha256_v2(material)
    return material


def build_phase_f_fixture_registry_v2(
    matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic fixture registry for all 120 matrix cases."""

    frozen = validate_phase_f_v2_matrix(
        build_phase_f_v2_matrix() if matrix is None else matrix
    )
    registry = _build_registry_unchecked(frozen)
    if registry["case_count"] != CASE_COUNT:
        raise PhaseFV2FixtureError("fixture registry case count mismatch")
    return registry


def validate_phase_f_fixture_registry_v2(
    value: Any, *, matrix: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed unless ``value`` is the exact complete fixture registry."""

    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhaseFV2FixtureError("fixture registry must be a string-keyed mapping")
    expected = build_phase_f_fixture_registry_v2(matrix)
    if dict(value) != expected:
        raise PhaseFV2FixtureError(
            "fixture registry is incomplete, altered, or noncanonical"
        )
    return deepcopy(expected)


# The receipt vocabulary calls the complete registry a fixture manifest.  This
# alias is intentionally descriptive; it does not create a second contract.
build_phase_f_fixture_manifest_v2 = build_phase_f_fixture_registry_v2


__all__ = [
    "INTACT_TWIN_COUNT",
    "MODULAR_RNG_DRAW_COUNT",
    "PANEL_AGENT_COUNT",
    "REGISTRY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TARGET_AGENT_COUNT",
    "PhaseFV2FixtureError",
    "build_phase_f_fixture_manifest_v2",
    "build_phase_f_fixture_registry_v2",
    "materialize_phase_f_case_v2",
    "phase_f_fixture_sha256_v2",
    "validate_phase_f_fixture_case_v2",
    "validate_phase_f_fixture_registry_v2",
]
