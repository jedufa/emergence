from __future__ import annotations

"""Strict loader for the frozen EIEL recurrent-native contract manifest."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "eiel_recurrent_state_capacity_epoch_manifest_v1"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).with_name("recurrent_native")
    / "recurrent_state_capacity_epoch_manifest_v1.json"
)


class RecurrentNativeContractError(ValueError):
    """Raised when recurrent state or its manifest violates the frozen ABI."""


@dataclass(frozen=True)
class RecurrentNativeContract:
    payload: Mapping[str, Any]
    path: Path

    @property
    def hidden_width(self) -> int:
        return int(self.payload["stable_gru"]["hidden_width"])

    @property
    def memory_ladder(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["stable_gru"]["memory_ladder"])

    @property
    def stable_tolerance(self) -> tuple[float, float]:
        value = self.payload["precision"]["comparison"]["stable_gru_cpu"]
        return float(value["rtol"]), float(value["atol"])

    @property
    def one_step_tolerance(self) -> dict[str, float]:
        rtol, atol = self.stable_tolerance
        return {"rtol": rtol, "atol": atol}

    @property
    def sequence_tolerance_values(self) -> tuple[float, float]:
        value = self.payload["precision"]["comparison"]["sequence"]
        return float(value["rtol"]), float(value["atol"])

    @property
    def sequence_tolerance(self) -> dict[str, float]:
        rtol, atol = self.sequence_tolerance_values
        return {"rtol": rtol, "atol": atol}

    def packed_width(self, observation_width: int, action_width: int = 7) -> int:
        observation_width = _positive_int(observation_width, "observation_width")
        action_width = _positive_int(action_width, "action_width")
        if action_width != 7:
            raise RecurrentNativeContractError(
                f"action_width={action_width}, expected frozen action width 7"
            )
        hidden = self.hidden_width
        return (
            hidden * observation_width
            + hidden
            + 3 * hidden * hidden
            + 3 * hidden
            + 3 * hidden * hidden
            + 3 * hidden
            + action_width * hidden
            + action_width
        )

    def validate_memory_dim(self, value: Any) -> int:
        dim = _nonnegative_int(value, "memory_dim")
        if dim not in self.memory_ladder:
            raise RecurrentNativeContractError(
                f"memory_dim={dim} is outside frozen ladder {self.memory_ladder}"
            )
        return dim

    def validate_state_shape(
        self,
        *,
        capacity: Any,
        hidden_width: Any,
        hmem_rows: Any,
        active_rows: Any,
    ) -> None:
        capacity_i = _nonnegative_int(capacity, "capacity")
        hidden_i = _positive_int(hidden_width, "hidden_width")
        hmem_i = _nonnegative_int(hmem_rows, "hmem_rows")
        active_i = _nonnegative_int(active_rows, "active_rows")
        if hidden_i != self.hidden_width:
            raise RecurrentNativeContractError(
                f"hidden_width={hidden_i}, expected {self.hidden_width}"
            )
        if hmem_i != capacity_i:
            raise RecurrentNativeContractError(
                f"hmem_rows={hmem_i}, expected capacity={capacity_i}"
            )
        if active_i > capacity_i:
            raise RecurrentNativeContractError(
                f"active_rows={active_i} exceeds capacity={capacity_i}"
            )


def load_recurrent_native_contract(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    project_root: str | Path | None = None,
) -> RecurrentNativeContract:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrentNativeContractError(
            f"cannot load recurrent contract {manifest_path}: {exc}"
        ) from exc
    _validate_manifest(payload, manifest_path=manifest_path, project_root=project_root)
    return RecurrentNativeContract(payload=payload, path=manifest_path)


def _validate_manifest(
    payload: Any,
    *,
    manifest_path: Path,
    project_root: str | Path | None,
) -> None:
    root = _mapping(payload, "manifest")
    _exact_keys(
        root,
        {
            "schema_version",
            "contract_id",
            "precision",
            "stable_gru",
            "modular_recurrent",
            "fail_closed",
            "source_references",
        },
        "manifest",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise RecurrentNativeContractError(
            f"unsupported schema_version={root['schema_version']!r}"
        )
    if root["contract_id"] != "stable_gru_v1_plus_modular_recurrent_v1":
        raise RecurrentNativeContractError("unexpected contract_id")

    precision = _mapping(root["precision"], "precision")
    _exact_keys(precision, {"storage", "compute", "comparison", "ambiguity"}, "precision")
    if precision.get("storage") != "float32" or precision.get("compute") != "float32":
        raise RecurrentNativeContractError("stable GRU storage and compute must be float32")
    comparison = _mapping(precision.get("comparison"), "precision.comparison")
    _exact_keys(
        comparison,
        {"stable_gru_cpu", "sequence", "integer_and_lifecycle"},
        "precision.comparison",
    )
    for name in ("stable_gru_cpu", "sequence"):
        tolerance = _mapping(comparison.get(name), f"precision.comparison.{name}")
        _exact_keys(tolerance, {"rtol", "atol"}, f"precision.comparison.{name}")
        _finite_nonnegative(tolerance.get("rtol"), f"{name}.rtol")
        _finite_nonnegative(tolerance.get("atol"), f"{name}.atol")
    if comparison.get("integer_and_lifecycle") != "exact":
        raise RecurrentNativeContractError("integer_and_lifecycle comparison must be exact")

    stable = _mapping(root["stable_gru"], "stable_gru")
    _exact_keys(
        stable,
        {"hidden_width", "memory_ladder", "gate_order", "equations",
         "packed_parameter_order", "packed_width_formula", "mask", "partition",
         "rng", "lifecycle", "checkpoint", "cost"},
        "stable_gru",
    )
    if _positive_int(stable.get("hidden_width"), "hidden_width") != 64:
        raise RecurrentNativeContractError("stable GRU hidden_width must be 64")
    ladder = stable.get("memory_ladder")
    if ladder != [0, 8, 16, 32, 64]:
        raise RecurrentNativeContractError("memory_ladder must equal [0, 8, 16, 32, 64]")
    if stable.get("gate_order") != ["reset", "update", "new"]:
        raise RecurrentNativeContractError("GRU gate_order must be reset, update, new")
    expected_pack = ["encW", "encb", "Wih", "bih", "Whh", "bhh", "headW", "headb"]
    if stable.get("packed_parameter_order") != expected_pack:
        raise RecurrentNativeContractError("packed_parameter_order does not match bank ABI")
    if stable.get("packed_width_formula") != "H*O+H+3*H*H+3*H+3*H*H+3*H+A*H+A":
        raise RecurrentNativeContractError("packed_width_formula drifted")
    if stable.get("mask") != (
        "Prefix mask index < effective_memory_dim is applied to h_prev before the cell "
        "and h_next before commit."
    ):
        raise RecurrentNativeContractError("stable GRU mask contract drifted")
    if stable.get("partition") != (
        "Rows retain alive/refill order within each MLP/GRU partition; stride misses are "
        "sorted by agent_id before partitioning."
    ):
        raise RecurrentNativeContractError("stable GRU partition contract drifted")
    rng = _mapping(stable.get("rng"), "stable_gru.rng")
    if rng != {
        "stride_one": "MLP [rows,actions] then GRU [rows,actions] from torch policy stream",
        "stride_many": "MLP [rows,stride,actions] then GRU [rows,actions] from policy_np; GRU action is repeated",
        "deterministic": "argmax and zero policy draws",
    }:
        raise RecurrentNativeContractError("stable GRU RNG contract drifted")
    equations = stable.get("equations")
    if not isinstance(equations, list) or len(equations) != 8 or not all(isinstance(v, str) and v for v in equations):
        raise RecurrentNativeContractError("stable_gru.equations must contain eight non-empty equations")

    lifecycle = _mapping(stable.get("lifecycle"), "stable_gru.lifecycle")
    _exact_keys(
        lifecycle,
        {"founder", "newborn", "inheritance", "death", "slot_reuse", "tick_commit",
         "failure", "cancellation"},
        "stable_gru.lifecycle",
    )
    for required in ("founder", "newborn", "inheritance", "death", "slot_reuse", "tick_commit", "failure", "cancellation"):
        if not isinstance(lifecycle.get(required), str) or not lifecycle[required]:
            raise RecurrentNativeContractError(f"missing lifecycle.{required}")
    checkpoint = _mapping(stable.get("checkpoint"), "stable_gru.checkpoint")
    _exact_keys(
        checkpoint, {"authoritative", "cache_only", "continuation"},
        "stable_gru.checkpoint",
    )
    if checkpoint.get("authoritative") != [
        "_Hmem", "_gru_hidden", "_gru_inited", "policy stride action sequence",
        "policy stride token", "policy stride cursor",
    ] or checkpoint.get("cache_only") != [
        "_gru_pack", "per-field GRU parameter banks", "_policy_tick_cache",
    ] or checkpoint.get("continuation") != (
        "Fresh owner rebuilds parameter banks, restores Hmem, and preserves valid stride-cache rows."
    ):
        raise RecurrentNativeContractError("stable GRU checkpoint contract drifted")
    cost = _mapping(stable.get("cost"), "stable_gru.cost")
    _exact_keys(cost, {"formula", "default_base", "default_per_dim", "charge_point"},
                "stable_gru.cost")
    if cost.get("formula") != "max(0, gru_cost_base + gru_cost_per_dim * effective_memory_dim)":
        raise RecurrentNativeContractError("stable GRU cost formula drifted")
    if cost.get("charge_point") != "environment apply after recurrent forward":
        raise RecurrentNativeContractError("stable GRU cost charge point drifted")
    _finite_nonnegative(cost.get("default_base"), "cost.default_base")
    _finite_nonnegative(cost.get("default_per_dim"), "cost.default_per_dim")

    modular = _mapping(root["modular_recurrent"], "modular_recurrent")
    _exact_keys(
        modular,
        {"envelope", "identity_and_order", "distinct_masks", "routing", "gated_scalar",
         "capacity_epoch", "operations", "mutation", "inheritance", "neutral_duplication",
         "dormancy_and_prune", "cost", "checkpoint", "commit"},
        "modular_recurrent",
    )
    envelope = _mapping(modular["envelope"], "modular_recurrent.envelope")
    if envelope != {
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
        "logical_profiles": {"legacy": [12, 32], "expanded": [16, 48]},
        "state_width_per_module": 1,
        "storage": "float32",
        "physical_resize_within_epoch": False,
        "epoch_metadata": ["capacity_epoch_id", "capacity_epoch_generation"],
    }:
        raise RecurrentNativeContractError("modular envelope or capacity_profiles drifted")
    identity = _mapping(modular["identity_and_order"], "modular_recurrent.identity_and_order")
    if identity.get("stable_identities") != ["module_instance_id", "homology_id"] or identity.get(
        "canonical_prefix_sort"
    ) != ["homology_id", "module_instance_id", "module_id"]:
        raise RecurrentNativeContractError("modular stable identity/order drifted")
    expected_masks = ["module_occupied", "module_dormant", "connection_occupied",
        "module_state_present", "module_output_present", "input_binding_enabled",
        "output_binding_enabled", "development_latch_present",
        "development_effective_gain_present"]
    if modular["distinct_masks"] != expected_masks:
        raise RecurrentNativeContractError("modular distinct_masks drifted")
    routing = _mapping(modular["routing"], "modular_recurrent.routing")
    if routing.get("input_semantics") != ["self_energy", "self_health"] or routing.get(
        "output_semantics"
    ) != {"move_north": 0, "move_south": 1, "move_east": 2, "move_west": 3, "rest": 5}:
        raise RecurrentNativeContractError("modular routing vocabulary drifted")
    if routing.get("binding_gain_bounds") != [-1.0, 1.0] or routing.get(
        "output_residual_clip"
    ) != [-0.25, 0.25] or routing.get("routing_selects_action") is not False:
        raise RecurrentNativeContractError("modular routing bounds drifted")
    gated = _mapping(modular["gated_scalar"], "modular_recurrent.gated_scalar")
    expected_parameters = ["reset_bias", "reset_input_gain", "reset_state_gain", "update_bias",
        "update_input_gain", "update_state_gain", "candidate_bias", "candidate_input_gain",
        "candidate_state_gain", "output_gain"]
    if gated.get("operator") != "gated_scalar_recurrent_v1" or gated.get(
        "parameter_order"
    ) != expected_parameters or gated.get("core_parameter_bounds") != [-4.0, 4.0] or gated.get(
        "output_gain_bounds"
    ) != [-2.0, 2.0] or gated.get("preactivation_clip") != [-16.0, 16.0] or gated.get(
        "state_clip"
    ) != [-1.0, 1.0] or gated.get("execution_rng_draws") != 0:
        raise RecurrentNativeContractError("modular gated scalar contract drifted")
    if not isinstance(gated.get("equations"), list) or len(gated["equations"]) != 8:
        raise RecurrentNativeContractError("modular gated equations drifted")
    epoch = _mapping(modular.get("capacity_epoch"), "modular_recurrent.capacity_epoch")
    if epoch.get("ordinary_mutation_can_resize") is not False:
        raise RecurrentNativeContractError("ordinary mutation must not resize recurrent capacity")
    eligibility = _mapping(epoch.get("eligibility"), "modular_recurrent.capacity_epoch.eligibility")
    if eligibility != {"window_generations": 3, "minimum_cap_contacts": 3,
        "minimum_blocked_legal_mutations": 2, "minimum_near_cap_lineages": 2,
        "minimum_active_capacity_fraction_each_sample": 0.8,
        "low_bloat_each_sample": True, "compression_relief_failed_each_sample": True}:
        raise RecurrentNativeContractError("capacity eligibility thresholds drifted")
    migration = _mapping(epoch.get("migration"), "modular_recurrent.capacity_epoch.migration")
    if (
        migration.get("source_profile") != [12, 32]
        or migration.get("target_profile") != [16, 48]
        or migration.get("safe_boundary") != "completed_generation_before_next_generation_rng_draw"
        or migration.get("next_generation_delta") != 1
        or migration.get("migration_rng_draws") != 0
        or migration.get("physical_resize_count") != 0
        or migration.get("new_tail") != "canonical empty"
        or migration.get("old_subspace") != "byte exact"
    ):
        raise RecurrentNativeContractError("capacity migration contract drifted")
    operations = modular.get("operations")
    if operations != ["activate", "duplicate", "route", "specialize", "disable", "prune"]:
        raise RecurrentNativeContractError("modular operation order is not frozen")
    mutation = _mapping(modular.get("mutation"), "modular_recurrent.mutation")
    if mutation.get("attempts") != 1 or mutation.get("externally_owned_draws") != 2 or mutation.get(
        "draw0_operation_bins"
    ) != [[0.0, 0.25, "toggle_dormancy"], [0.25, 0.5, "prune_eligible_module"],
          [0.5, 0.75, "refine_descendant_parameter"],
          [0.75, 1.0, "refine_descendant_routing_or_expression"]] or mutation.get(
        "native_rng_draws"
    ) != 0:
        raise RecurrentNativeContractError("two-draw modular mutation contract drifted")
    coefficients = _mapping(_mapping(modular.get("cost"), "modular_recurrent.cost").get(
        "coefficients"
    ), "modular_recurrent.cost.coefficients")
    if coefficients != {"allocated_module": 1, "executing_integrator": 4,
        "executing_gated": 32, "evaluated_connection": 2, "active_input_binding": 1,
        "active_output_binding": 2, "active_dependency_edge": 1,
        "dormant_module_maintenance": 0}:
        raise RecurrentNativeContractError("modular cost coefficients drifted")
    if "remain active" not in modular.get("neutral_duplication", "") or "output_gain=0" not in modular.get(
        "neutral_duplication", ""
    ):
        raise RecurrentNativeContractError("neutral duplication contract drifted")

    failures = root["fail_closed"]
    if not isinstance(failures, list) or len(failures) < 8 or not all(isinstance(v, str) and v for v in failures):
        raise RecurrentNativeContractError("fail_closed must enumerate malformed-state classes")
    references = root["source_references"]
    if not isinstance(references, list) or not references:
        raise RecurrentNativeContractError("source_references must be non-empty")
    resolved_root = Path(project_root).resolve() if project_root is not None else manifest_path.parents[3]
    for index, reference_value in enumerate(references):
        reference = _mapping(reference_value, f"source_references[{index}]")
        _exact_keys(reference, {"path", "lines", "claim"}, f"source_references[{index}]")
        relative = reference["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RecurrentNativeContractError(f"invalid source reference path at index {index}")
        if not (resolved_root / relative).is_file():
            raise RecurrentNativeContractError(
                f"source reference does not exist: {relative} (root {resolved_root})"
            )
        if not isinstance(reference["lines"], str) or not reference["lines"]:
            raise RecurrentNativeContractError(f"missing source line range at index {index}")
        if not isinstance(reference["claim"], str) or not reference["claim"]:
            raise RecurrentNativeContractError(f"missing source claim at index {index}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RecurrentNativeContractError(f"{name} must be an object")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecurrentNativeContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RecurrentNativeContractError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result <= 0:
        raise RecurrentNativeContractError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentNativeContractError(f"{name} must be a nonnegative integer")
    return int(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentNativeContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RecurrentNativeContractError(f"{name} must be finite and nonnegative")
    return result


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "RecurrentNativeContract",
    "RecurrentNativeContractError",
    "SCHEMA_VERSION",
    "load_recurrent_native_contract",
]
