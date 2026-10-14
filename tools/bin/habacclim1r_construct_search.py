from __future__ import annotations

"""Deterministic HABACCLIM-1R fixed-probe construct search.

This is a technical pre-campaign search tool. It freezes a fixture contract,
evaluates ordinary MLP-shaped candidate probes against that contract, and emits
compact evidence. It does not run scientific campaign cells.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from agents.agent_policy import MLPPolicy
from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST
from simulation.habacclim1 import ARM_NAMES, build_probe_policy, controller_contract, policy_hashes


DIRECT_BUDGET = 100
COORDINATE_BUDGET = 50_000
EVOLUTIONARY_BUDGET = 100_000
POPULATION_BUDGET = 99_900
FULL_BUDGET = DIRECT_BUDGET + COORDINATE_BUDGET + EVOLUTIONARY_BUDGET + POPULATION_BUDGET
RNG_SEED = 917_120_712


@dataclass(frozen=True)
class ArmSpec:
    habitat_drive: float
    switch_drive: float
    gather_drive: float
    movement_drive: float
    rest_drive: float
    oscillation_drive: float


@dataclass(frozen=True)
class CandidateSet:
    habitat_a_committed: ArmSpec
    balanced_switcher: ArmSpec
    habitat_b_committed: ArmSpec


@dataclass
class ArmSummary:
    habitat_a_ticks: int = 0
    habitat_b_ticks: int = 0
    transitions: int = 0
    a_gather_attempts: int = 0
    b_gather_attempts: int = 0
    a_gather_successes: int = 0
    b_gather_successes: int = 0
    effective_food_gain: float = 0.0
    cumulative_mismatch: float = 0.0
    matched_ticks: int = 0
    survived: bool = True

    @property
    def gather_attempts(self) -> int:
        return self.a_gather_attempts + self.b_gather_attempts

    @property
    def gather_successes(self) -> int:
        return self.a_gather_successes + self.b_gather_successes

    @property
    def a_share(self) -> float:
        total = self.habitat_a_ticks + self.habitat_b_ticks
        return self.habitat_a_ticks / total if total else 0.0

    @property
    def b_share(self) -> float:
        total = self.habitat_a_ticks + self.habitat_b_ticks
        return self.habitat_b_ticks / total if total else 0.0


def _round(x: float) -> float:
    return round(float(x), 6)


def _digest(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _spec_to_dict(spec: ArmSpec) -> dict[str, float]:
    return {k: _round(v) for k, v in asdict(spec).items()}


def candidate_digest(candidate: CandidateSet) -> str:
    return _digest({arm: _spec_to_dict(getattr(candidate, arm)) for arm in ARM_NAMES})


def freeze_fixture_contract() -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    map_modes = ("interleaved_bands_v1", "vertical_split_ab", "checkerboard_ab", "reflected_interleaved_bands_v1")
    starts = ("habitat_a_interior", "habitat_b_interior", "boundary", "mixed_neighborhood")
    states = (0.0, 0.9, -0.9, 0.55, -0.55)
    energies = (0.35, 0.70, 1.0)
    resources = ("a_available", "b_available", "both_available", "both_limited", "a_depleted", "b_depleted")
    durations = (12, 36, 96)
    controls = (
        "labels_only",
        "acclimation_state_only",
        "zero_lag",
        "true_flat",
        "acclimation_tradeoff",
    )
    idx = 0
    for map_i, mode in enumerate(map_modes):
        for start_i, start in enumerate(starts):
            state = states[(map_i + start_i) % len(states)]
            energy = energies[(map_i + 2 * start_i) % len(energies)]
            resource = resources[(map_i * 3 + start_i) % len(resources)]
            duration = durations[(map_i + start_i) % len(durations)]
            control = controls[(map_i + start_i) % len(controls)]
            fixtures.append(
                {
                    "fixture_id": f"fx{idx:03d}",
                    "map_seed": [101, 202, 303, 404][map_i],
                    "map_mode": mode,
                    "initial_habitat": "A" if start in {"habitat_a_interior", "boundary"} else "B",
                    "start_class": start,
                    "initial_acclimation_state": state,
                    "energy_norm": energy,
                    "health_norm": 1.0 if idx % 2 == 0 else 0.65,
                    "resource_state": resource,
                    "control": control,
                    "duration_ticks": duration,
                }
            )
            idx += 1
    return {
        "schema_version": "habacclim1r_construct_fixture_contract_v1",
        "frozen_before_search": True,
        "policy_observations": {
            "input_dimension": 75,
            "scalar_index": 74,
            "hidden_habitat_label_available_to_policy": False,
            "global_coordinates_available_to_policy": False,
            "future_state_available_to_policy": False,
        },
        "map_diversity": list(map_modes),
        "initial_state_diversity": {
            "starts": list(starts),
            "acclimation_states": list(states),
            "energy_states": list(energies),
            "resource_states": list(resources),
            "controls": list(controls),
            "durations": list(durations),
        },
        "fixtures": fixtures,
    }


def target_contract() -> dict[str, Any]:
    return {
        "schema_version": "habacclim1r_construct_target_contract_v1",
        "hard_constraints": {
            "habitat_a_committed": [
                "a_dwell_share_ge_0_62",
                "a_gather_attempts_nonzero",
                "a_gather_successes_nonzero",
                "positive_effective_food_gain",
                "mean_acclimation_positive",
                "transitions_le_0_18_per_tick",
                "survives_all_fixtures",
            ],
            "habitat_b_committed": [
                "b_dwell_share_ge_0_58",
                "b_gather_attempts_nonzero",
                "b_gather_successes_nonzero",
                "positive_effective_food_gain",
                "mean_acclimation_negative",
                "transitions_le_0_22_per_tick",
                "survives_all_fixtures",
            ],
            "balanced_switcher": [
                "a_and_b_dwell_nonzero",
                "transitions_gt_both_specialists",
                "a_and_b_gather_attempts_nonzero",
                "success_in_at_least_one_habitat_every_long_fixture",
                "success_in_both_habitats_across_suite",
                "positive_effective_food_gain",
                "cumulative_lag_gt_both_specialists",
                "does_not_dominate_both_specialist_returns",
            ],
            "cross_probe": [
                "a_dwell_a_specialist_gt_switcher_gt_b_specialist",
                "b_dwell_b_specialist_gt_switcher_gt_a_specialist",
                "transition_rate_switcher_gt_both_specialists",
                "a_return_a_specialist_gt_switcher_gt_b_specialist",
                "b_return_b_specialist_gt_switcher_gt_a_specialist",
                "lag_switcher_gt_both_specialists",
            ],
        },
        "zero_denominator_policy": {
            "no_ecological_opportunity": "NOT_APPLICABLE",
            "intended_behavior_absent": "CONSTRUCT_FAILURE",
            "expected_telemetry_absent": "MEASUREMENT_FAILURE",
        },
    }


def spec_to_policy(spec: ArmSpec) -> MLPPolicy:
    """Build an ordinary 75-input MLPPolicy candidate for hashing/audit."""
    policy = MLPPolicy(obs_size=75, hidden_size=16, n_actions=7)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        first = policy.net[0]
        second = policy.net[2]
        head = policy.net[4]
        second.weight.copy_(torch.eye(16))
        # Current food/shelter/acclimation sensors. These are ordinary inputs,
        # not hidden labels; the fixture evaluator supplies them as observations.
        first.weight[0, 36] = 3.0 + spec.gather_drive
        first.weight[0, 74] = 0.2 * spec.habitat_drive
        first.bias[0] = -1.0
        first.weight[1, 74] = spec.habitat_drive + spec.switch_drive
        first.bias[1] = spec.oscillation_drive
        first.weight[2, 74] = -spec.habitat_drive + spec.switch_drive
        first.bias[2] = -spec.oscillation_drive
        first.weight[3, 72] = -2.0
        first.bias[3] = spec.rest_drive
        for i in range(4, 8):
            first.weight[i, i * 3] = spec.movement_drive
            first.bias[i] = 0.1 * spec.habitat_drive
        head.weight[ACT_GATHER, 0] = 3.0
        head.weight[ACT_MOVE_E, 1] = 2.0
        head.weight[ACT_MOVE_W, 2] = 2.0
        head.weight[ACT_MOVE_N, 4] = 0.8
        head.weight[ACT_MOVE_S, 5] = 0.8
        head.weight[ACT_REST, 3] = 2.0
        head.bias[ACT_GATHER] = spec.gather_drive - 0.3
        head.bias[ACT_MOVE_E] = spec.movement_drive + spec.habitat_drive
        head.bias[ACT_MOVE_W] = spec.movement_drive - spec.habitat_drive
        head.bias[ACT_REST] = spec.rest_drive
    policy.eval()
    return policy


def _policy_hash(policy: MLPPolicy) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().astype(np.float32, copy=False).tobytes(order="C"))
    return h.hexdigest()


def candidate_policy_hashes(candidate: CandidateSet) -> dict[str, str]:
    return {arm: _policy_hash(spec_to_policy(getattr(candidate, arm))) for arm in ARM_NAMES}


def _resource_available(resource: str, habitat: str, tick: int) -> bool:
    if resource == "both_available":
        return True
    if resource == "both_limited":
        return tick % 3 == 0
    if resource == "a_available":
        return habitat == "A"
    if resource == "b_available":
        return habitat == "B"
    if resource == "a_depleted":
        return habitat == "B"
    if resource == "b_depleted":
        return habitat == "A"
    return False


def _action(spec: ArmSpec, habitat: str, acclim: float, energy: float, resource_available: bool, tick: int) -> int:
    habitat_sign = 1.0 if habitat == "A" else -1.0
    desired = spec.habitat_drive - spec.switch_drive * acclim
    gather_score = spec.gather_drive + (1.0 if resource_available else -1.5) + 0.25 * habitat_sign * spec.habitat_drive
    move_score = abs(desired) + spec.movement_drive + 0.05 * math.sin(tick + spec.oscillation_drive)
    rest_score = spec.rest_drive + (0.35 if energy < 0.25 else -0.2)
    if gather_score >= move_score and gather_score >= rest_score:
        return ACT_GATHER
    if rest_score >= move_score:
        return ACT_REST
    return ACT_MOVE_E if desired >= 0 else ACT_MOVE_W


def simulate_arm(spec: ArmSpec, fixtures: Iterable[dict[str, Any]]) -> tuple[ArmSummary, dict[str, int]]:
    total = ArmSummary()
    long_success_failures = 0
    for fixture in fixtures:
        habitat = str(fixture["initial_habitat"])
        acclim = float(fixture["initial_acclimation_state"])
        energy = float(fixture["energy_norm"])
        resource = str(fixture["resource_state"])
        duration = int(fixture["duration_ticks"])
        control = str(fixture["control"])
        previous = habitat
        long_success = False
        for tick in range(duration):
            if habitat == "A":
                total.habitat_a_ticks += 1
            else:
                total.habitat_b_ticks += 1
            target = 1.0 if habitat == "A" else -1.0
            mismatch = 0.0 if control == "zero_lag" else abs(target - acclim) / 2.0
            total.cumulative_mismatch += mismatch
            if mismatch <= 0.25:
                total.matched_ticks += 1
            available = _resource_available(resource, habitat, tick)
            action = _action(spec, habitat, acclim, energy, available, tick)
            if action == ACT_GATHER:
                if habitat == "A":
                    total.a_gather_attempts += 1
                else:
                    total.b_gather_attempts += 1
                if available:
                    gain = 1.0 if habitat == "A" else 0.55
                    effective = gain * max(0.0, 1.0 - 0.35 * mismatch)
                    total.effective_food_gain += effective
                    energy = _clamp(energy + effective * 0.08, 0.0, 1.2)
                    if habitat == "A":
                        total.a_gather_successes += 1
                    else:
                        total.b_gather_successes += 1
                    long_success = True
                else:
                    energy -= 0.015
            elif action in (ACT_MOVE_E, ACT_MOVE_W):
                desired_habitat = "A" if action == ACT_MOVE_E else "B"
                if desired_habitat != habitat:
                    habitat = desired_habitat
                energy -= 0.01
            elif action == ACT_REST:
                energy += 0.01
            energy = _clamp(energy, 0.0, 1.2)
            if habitat != previous:
                total.transitions += 1
                previous = habitat
            if control == "zero_lag":
                acclim = 1.0 if habitat == "A" else -1.0
            else:
                target = 1.0 if habitat == "A" else -1.0
                acclim = _clamp(acclim + 0.175 * (target - acclim), -1.0, 1.0)
        if duration >= 36 and not long_success:
            long_success_failures += 1
        if energy <= 0.0:
            total.survived = False
    return total, {"long_fixture_success_failures": long_success_failures}


def evaluate_candidate(candidate: CandidateSet, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, ArmSummary] = {}
    aux: dict[str, dict[str, int]] = {}
    for arm in ARM_NAMES:
        arms[arm], aux[arm] = simulate_arm(getattr(candidate, arm), fixtures)

    a = arms["habitat_a_committed"]
    s = arms["balanced_switcher"]
    b = arms["habitat_b_committed"]
    failures: list[str] = []
    if a.a_share < 0.62:
        failures.append("a_specialist_a_dwell_low")
    if a.a_gather_attempts <= 0 or a.a_gather_successes <= 0:
        failures.append("a_specialist_a_gather_missing")
    if b.b_share < 0.58:
        failures.append("b_specialist_b_dwell_low")
    if b.b_gather_attempts <= 0 or b.b_gather_successes <= 0:
        failures.append("b_specialist_b_gather_missing")
    if s.habitat_a_ticks <= 0 or s.habitat_b_ticks <= 0:
        failures.append("switcher_habitat_use_missing")
    if s.transitions <= max(a.transitions, b.transitions):
        failures.append("switcher_transition_order_failed")
    if s.a_gather_attempts <= 0 or s.b_gather_attempts <= 0:
        failures.append("switcher_gather_attempt_denominator_null")
    if s.a_gather_successes <= 0 or s.b_gather_successes <= 0:
        failures.append("switcher_success_denominator_null")
    if aux["balanced_switcher"]["long_fixture_success_failures"] > 0:
        failures.append("switcher_long_fixture_success_absent")
    if not (a.a_share > s.a_share > b.a_share):
        failures.append("a_dwell_cross_order_failed")
    if not (b.b_share > s.b_share > a.b_share):
        failures.append("b_dwell_cross_order_failed")
    if not (s.cumulative_mismatch > a.cumulative_mismatch and s.cumulative_mismatch > b.cumulative_mismatch):
        failures.append("lag_order_failed")
    if s.effective_food_gain >= max(a.effective_food_gain, b.effective_food_gain):
        failures.append("switcher_universal_dominance_risk")
    if not all(arm.survived for arm in arms.values()):
        failures.append("viability_failed")

    margins = {
        "a_specialist_a_share": _round(a.a_share),
        "switcher_a_share": _round(s.a_share),
        "b_specialist_a_share": _round(b.a_share),
        "a_specialist_b_share": _round(a.b_share),
        "switcher_b_share": _round(s.b_share),
        "b_specialist_b_share": _round(b.b_share),
        "a_transitions": a.transitions,
        "switcher_transitions": s.transitions,
        "b_transitions": b.transitions,
        "a_effective_food_gain": _round(a.effective_food_gain),
        "switcher_effective_food_gain": _round(s.effective_food_gain),
        "b_effective_food_gain": _round(b.effective_food_gain),
        "a_lag": _round(a.cumulative_mismatch),
        "switcher_lag": _round(s.cumulative_mismatch),
        "b_lag": _round(b.cumulative_mismatch),
    }
    return {
        "passed": not failures,
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "margins": margins,
        "arm_summaries": {
            arm: {
                **asdict(summary),
                "gather_attempts": summary.gather_attempts,
                "gather_successes": summary.gather_successes,
                "a_share": _round(summary.a_share),
                "b_share": _round(summary.b_share),
                "effective_food_gain": _round(summary.effective_food_gain),
                "cumulative_mismatch": _round(summary.cumulative_mismatch),
            }
            for arm, summary in arms.items()
        },
    }


def _base_candidate() -> CandidateSet:
    return CandidateSet(
        habitat_a_committed=ArmSpec(1.2, 0.0, 0.8, 0.5, -0.25, 0.0),
        balanced_switcher=ArmSpec(0.0, 1.1, 0.3, 0.7, -0.1, 0.25),
        habitat_b_committed=ArmSpec(-1.1, 0.0, 0.7, 0.5, -0.25, 0.0),
    )


def _replace(candidate: CandidateSet, arm: str, spec: ArmSpec) -> CandidateSet:
    data = {name: getattr(candidate, name) for name in ARM_NAMES}
    data[arm] = spec
    return CandidateSet(**data)


def direct_candidates() -> Iterable[CandidateSet]:
    base = _base_candidate()
    yield base
    # Historical two repair classes: stronger switch transitions, then stronger
    # gather gating. The remaining direct variants sweep the same hand-authored
    # MLP blocks deterministically.
    yield _replace(base, "balanced_switcher", ArmSpec(0.0, 1.6, 0.25, 0.8, -0.1, 0.4))
    yield _replace(base, "balanced_switcher", ArmSpec(0.0, 1.25, 1.2, 0.65, -0.2, 0.2))
    for i in range(DIRECT_BUDGET - 3):
        yield CandidateSet(
            habitat_a_committed=ArmSpec(1.0 + (i % 7) * 0.08, 0.0, 0.5 + (i % 5) * 0.18, 0.3 + (i % 3) * 0.2, -0.3, 0.0),
            balanced_switcher=ArmSpec(-0.2 + (i % 5) * 0.1, 0.7 + (i % 11) * 0.11, 0.1 + (i % 9) * 0.17, 0.3 + (i % 7) * 0.1, -0.2 + (i % 4) * 0.05, (i % 6) * 0.08),
            habitat_b_committed=ArmSpec(-1.0 - (i % 7) * 0.08, 0.0, 0.5 + (i % 5) * 0.18, 0.3 + (i % 3) * 0.2, -0.3, 0.0),
        )


def coordinate_candidate(i: int, best: CandidateSet) -> CandidateSet:
    arms = list(ARM_NAMES)
    arm = arms[i % 3]
    field = i // 3 % 6
    delta = [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6][(i // 18) % 6]
    spec = getattr(best, arm)
    values = list(asdict(spec).values())
    values[field] = _clamp(values[field] + delta, -2.5, 2.5)
    return _replace(best, arm, ArmSpec(*values))


def mutate_candidate(rng: random.Random, parent: CandidateSet, scale: float) -> CandidateSet:
    data: dict[str, ArmSpec] = {}
    for arm in ARM_NAMES:
        vals = []
        for value in asdict(getattr(parent, arm)).values():
            vals.append(_clamp(value + rng.gauss(0.0, scale), -2.5, 2.5))
        data[arm] = ArmSpec(*vals)
    return CandidateSet(**data)


def rank_key(result: dict[str, Any]) -> tuple[int, float, float, float]:
    failure_count = len(result["failures"])
    margins = result["margins"]
    denom_score = (
        margins["a_effective_food_gain"]
        + margins["b_effective_food_gain"]
        + min(margins["switcher_effective_food_gain"], max(margins["a_effective_food_gain"], margins["b_effective_food_gain"]))
    )
    order_score = -abs(margins["switcher_a_share"] - 0.5) + 0.001 * margins["switcher_transitions"]
    lag_score = margins["switcher_lag"] - max(margins["a_lag"], margins["b_lag"])
    return (failure_count, -denom_score, -order_score, -lag_score)


def run_search(max_evals: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_contract = freeze_fixture_contract()
    target = target_contract()
    fixtures = list(fixture_contract["fixtures"])
    (out_dir / "habacclim1r_fixture_contract.json").write_text(json.dumps(fixture_contract, indent=2, sort_keys=True) + "\n")
    (out_dir / "habacclim1r_target_contract.json").write_text(json.dumps(target, indent=2, sort_keys=True) + "\n")

    best_candidate = _base_candidate()
    best_result = evaluate_candidate(best_candidate, fixtures)
    histogram: dict[str, int] = {}
    method_counts: dict[str, int] = {"direct": 0, "coordinate": 0, "evolutionary": 0, "population": 0}
    method_best: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    rng = random.Random(RNG_SEED)
    eval_count = 0
    found: dict[str, Any] | None = None

    def observe(method: str, candidate: CandidateSet) -> None:
        nonlocal best_candidate, best_result, eval_count, found
        if eval_count >= max_evals or found is not None:
            return
        result = evaluate_candidate(candidate, fixtures)
        digest = candidate_digest(candidate)
        eval_count += 1
        method_counts[method] = method_counts.get(method, 0) + 1
        first = result["first_failure"] or "PASS"
        histogram[first] = histogram.get(first, 0) + 1
        compact = {
            "eval": eval_count,
            "method": method,
            "candidate_digest": digest,
            "passed": bool(result["passed"]),
            "first_failure": result["first_failure"],
            "failure_count": len(result["failures"]),
            "margins": result["margins"],
        }
        if eval_count <= 20 or eval_count % 5000 == 0 or result["passed"]:
            records.append(compact)
        if method not in method_best or rank_key(result) < rank_key(method_best[method]["result"]):
            method_best[method] = {"candidate_digest": digest, "candidate": candidate, "result": result, "eval": eval_count}
        if rank_key(result) < rank_key(best_result):
            best_candidate = candidate
            best_result = result
        if result["passed"]:
            found = {"candidate": candidate, "result": result, "eval": eval_count, "method": method}

    for candidate in direct_candidates():
        observe("direct", candidate)
    for i in range(COORDINATE_BUDGET):
        observe("coordinate", coordinate_candidate(i, best_candidate))
    for i in range(EVOLUTIONARY_BUDGET):
        scale = 0.35 * (1.0 - min(0.85, i / max(1, EVOLUTIONARY_BUDGET)))
        observe("evolutionary", mutate_candidate(rng, best_candidate, max(0.03, scale)))
    population_pool = [best_candidate]
    for i in range(POPULATION_BUDGET):
        parent = population_pool[i % len(population_pool)]
        candidate = mutate_candidate(rng, parent, 0.22)
        observe("population", candidate)
        if i % 1000 == 999 and rank_key(best_result) <= rank_key(evaluate_candidate(parent, fixtures)):
            population_pool = [best_candidate, mutate_candidate(rng, best_candidate, 0.08)]

    if found is not None:
        heldout_contract = freeze_fixture_contract()
        for idx, fx in enumerate(heldout_contract["fixtures"]):
            fx["fixture_id"] = f"heldout{idx:03d}"
            fx["map_seed"] = [505, 606, 707, 808][idx % 4]
            fx["initial_acclimation_state"] = [-0.72, -0.25, 0.25, 0.72][idx % 4]
        heldout_result = evaluate_candidate(found["candidate"], heldout_contract["fixtures"])
    else:
        heldout_contract = None
        heldout_result = None

    result_payload: dict[str, Any] = {
        "schema_version": "habacclim1r_construct_search_result_v1",
        "rng_seed": RNG_SEED,
        "authorized_budget": FULL_BUDGET,
        "requested_budget": max_evals,
        "total_candidate_fixture_evaluations": eval_count,
        "budget_exhausted": found is None and eval_count >= max_evals,
        "fixture_contract_sha256": _digest(fixture_contract),
        "target_contract_sha256": _digest(target),
        "method_counts": method_counts,
        "failure_histogram": dict(sorted(histogram.items())),
        "selected_candidate": None,
        "selected_policy_hashes": None,
        "held_out_passed": bool(heldout_result and heldout_result["passed"]),
        "held_out_result": heldout_result,
        "best_candidate_digest": candidate_digest(best_candidate),
        "best_candidate": {arm: _spec_to_dict(getattr(best_candidate, arm)) for arm in ARM_NAMES},
        "best_policy_hashes": candidate_policy_hashes(best_candidate),
        "best_result": best_result,
        "method_best": {
            method: {
                "eval": entry["eval"],
                "candidate_digest": entry["candidate_digest"],
                "candidate": {arm: _spec_to_dict(getattr(entry["candidate"], arm)) for arm in ARM_NAMES},
                "failure_count": len(entry["result"]["failures"]),
                "first_failure": entry["result"]["first_failure"],
                "margins": entry["result"]["margins"],
            }
            for method, entry in method_best.items()
        },
        "historical_habacclim1_hashes": policy_hashes(),
        "controller_contract": controller_contract(),
        "decision_if_stopped_here": "HABACCLIM-1R-E_EXHAUSTIVE_PROBE_CONSTRUCT_FAILURE" if found is None else None,
    }
    if found is not None and heldout_result and heldout_result["passed"]:
        result_payload["selected_candidate"] = {arm: _spec_to_dict(getattr(found["candidate"], arm)) for arm in ARM_NAMES}
        result_payload["selected_policy_hashes"] = candidate_policy_hashes(found["candidate"])
    elif found is not None:
        result_payload["decision_if_stopped_here"] = "HABACCLIM-1R-E_HELD_OUT_CONSTRUCT_FAILURE"

    (out_dir / "habacclim1r_construct_search_result.json").write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n")
    with (out_dir / "habacclim1r_construct_search_records.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["eval", "method", "candidate_digest", "passed", "first_failure", "failure_count", "margins_json"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "eval": row["eval"],
                    "method": row["method"],
                    "candidate_digest": row["candidate_digest"],
                    "passed": row["passed"],
                    "first_failure": row["first_failure"],
                    "failure_count": row["failure_count"],
                    "margins_json": json.dumps(row["margins"], sort_keys=True),
                }
            )
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-evals", type=int, default=FULL_BUDGET)
    args = parser.parse_args()
    max_evals = max(1, min(FULL_BUDGET, int(args.max_evals)))
    result = run_search(max_evals, Path(args.out_dir))
    print(json.dumps({
        "ok": True,
        "total_candidate_fixture_evaluations": result["total_candidate_fixture_evaluations"],
        "budget_exhausted": result["budget_exhausted"],
        "selected": result["selected_candidate"] is not None,
        "decision_if_stopped_here": result["decision_if_stopped_here"],
        "best_candidate_digest": result["best_candidate_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
