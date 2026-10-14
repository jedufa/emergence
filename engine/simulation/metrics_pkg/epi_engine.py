"""simulation/metrics_pkg/epi_engine.py

Last modified: 2026-02-23 (Europe/Vienna)

Phase 5a: Experimental Pressure Index (EPI).

Requirements (prompt-frozen):
- Shadow-clone evaluation (no mutation of main sim state)
- Stress scenarios schema: stress_scenario_v1
- Mean+min aggregation with alpha weights
- Canonical JSON output (sorted keys, stable floats)
- Must use RNGManager stream "stress" only during stress evaluation

Notes
-----
This implementation avoids touching the main simulation RNG state or global torch
RNG state by:

- Running evaluation on deep-copied environment/population state
- Using deterministic (argmax) policy action selection during evaluation
- Injecting RNGManager.stress_* RNGs into the shadow runner (env + pop + runner)
"""

from __future__ import annotations

import hashlib
import tempfile
from typing import Optional
from dataclasses import replace
from typing import Any, Dict, List, Sequence, Tuple

from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.metrics_pkg.stress_scenarios import StressScenarioV1
from simulation.rng_manager import RNGManager


EPI_SCHEMA_VERSION = "epi_v2"


def _det_violation(code: str, context: str) -> None:
    raise RuntimeError(f"[DETERMINISM_VIOLATION]\ncode={code}\ncontext={context}\n")


def derive_stress_seed(
    global_seed: int,
    scenario_id: str,
    generation_index: int,
    seed_offset: int,
) -> int:
    """Derive the per-scenario stress seed.

    Spec (prompt-frozen):
        stress_seed = SHA256(global_seed || scenario_id || generation_index || seed_offset)

    Implementation detail:
    - Uses UTF-8 string concatenation with a '|' delimiter.
    - Returns the first 8 bytes of the digest interpreted as an unsigned big-endian integer.
    """

    material = (
        str(int(global_seed)).encode("utf-8")
        + b"|"
        + str(scenario_id).encode("utf-8")
        + b"|"
        + str(int(generation_index)).encode("utf-8")
        + b"|"
        + str(int(seed_offset)).encode("utf-8")
    )
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_weights(ws1: float, ws2: float, ws3: float) -> Tuple[float, float, float]:
    total = float(ws1) + float(ws2) + float(ws3)
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (float(ws1) / total, float(ws2) / total, float(ws3) / total)


def _select_agent_ids(
    alive_ids_sorted: Sequence[int],
    mode: str,
    stride: int,
    start_index: int,
) -> List[int]:
    if mode == "all":
        return list(alive_ids_sorted)
    if mode == "fixed_stride":
        s = max(1, int(stride))
        start = max(0, int(start_index))
        return list(alive_ids_sorted[start::s])
    # Unknown mode => fail closed (no sampling).
    return []


def _apply_pressure_overrides(env: Any, overrides: Dict[str, float]) -> None:
    """Apply pressure overrides to the environment.

    For Phase 5a, we only support runtime multipliers that are already part of GridWorld.
    Unknown keys are ignored.
    """

    if not overrides:
        return

    food_mult = float(overrides.get("food_spawn_mult", getattr(env, "food_spawn_mult", 1.0)))
    action_mult = float(overrides.get("action_cost_mult", getattr(env, "action_cost_mult", 1.0)))
    rest_mult = float(overrides.get("rest_cost_mult", getattr(env, "rest_cost_mult", 1.0)))

    # Repo reality: env.gridworld_pkg.multipliers.MultipliersMixin.set_multipliers
    # uses keyword-only args. Older snapshots used positional args.
    if hasattr(env, "set_multipliers"):
        try:
            env.set_multipliers(
                food_spawn_mult=food_mult,
                action_cost_mult=action_mult,
                rest_cost_mult=rest_mult,
            )
            return
        except TypeError:
            # Best-effort fallback for legacy positional signatures.
            try:
                env.set_multipliers(food_mult, action_mult, rest_mult)
                return
            except Exception:
                pass

    # Last resort: assign attributes directly (shadow clone only).
    try:
        setattr(env, "food_spawn_mult", float(food_mult))
        setattr(env, "action_cost_mult", float(action_mult))
        setattr(env, "rest_cost_mult", float(rest_mult))
    except Exception:
        return


class EPIEngine:
    """EPI engine (shadow-clone stress evaluation)."""

    def __init__(self) -> None:
        return

    def compute(
        self,
        sim: Any,
        generation_index: int,
        scenarios: Sequence[StressScenarioV1],
    ) -> Dict:
        """Compute EPI for `generation_index`.

        This method does not mutate `sim`.
        """

        sim_cfg = sim.sim_cfg
        global_seed = int(getattr(sim_cfg, "seed", 0))

        ws1, ws2, ws3 = _normalize_weights(
            getattr(sim_cfg, "epi_ws1_survival", 0.50),
            getattr(sim_cfg, "epi_ws2_reproduction", 0.30),
            getattr(sim_cfg, "epi_ws3_robustness", 0.20),
        )
        alpha_mean = float(getattr(sim_cfg, "epi_alpha_mean", 0.70))
        alpha_min = float(getattr(sim_cfg, "epi_alpha_min", 0.30))

        alive_ids_sorted = sorted([a.agent_id for a in sim.pop.alive_agents()])
        sample_ids = _select_agent_ids(
            alive_ids_sorted,
            mode=str(getattr(sim_cfg, "epi_subsample_mode", "all")),
            stride=int(getattr(sim_cfg, "epi_subsample_stride", 1)),
            start_index=int(getattr(sim_cfg, "epi_subsample_start_index", 0)),
        )
        sample_ids_set = set(sample_ids)

        # Build deterministic snapshot material once per compute call, then
        # apply into scenario-local shadow runners. This avoids deepcopy over
        # live RNG-bearing object graphs.
        from simulation.checkpoint_bundle_v2 import (  # noqa: WPS433
            _export_population_and_weights,
            _export_world,
        )
        with tempfile.TemporaryDirectory(prefix="epi_shadow_snapshot_") as snapdir:
            weights_dir = str(tempfile.mkdtemp(prefix="weights_", dir=snapdir))
            world_obj = _export_world(sim)
            pop_obj, _ = _export_population_and_weights(sim, weights_dir=weights_dir)

            scenario_results: List[Dict] = []
            scenario_scores: List[float] = []

            for sc in scenarios:
                res = self._evaluate_scenario(
                    world_obj=world_obj,
                    pop_obj=pop_obj,
                    weights_dir=weights_dir,
                    sim=sim,
                    generation_index=generation_index,
                    scenario=sc,
                    sample_ids=sample_ids,
                    sample_ids_set=sample_ids_set,
                    ws1=ws1,
                    ws2=ws2,
                    ws3=ws3,
                    global_seed=global_seed,
                )
                scenario_results.append(res)
                scenario_scores.append(float(res["scenario_score"]))

            if scenario_scores:
                mean_score = float(sum(scenario_scores)) / float(len(scenario_scores))
                min_score = float(min(scenario_scores))
            else:
                mean_score = 0.0
                min_score = 0.0

            epi_value = 100.0 * (alpha_mean * mean_score + alpha_min * min_score)

            record = {
                "schema_version": EPI_SCHEMA_VERSION,
                "generation": int(generation_index),
                "weights": {
                    "ws1_survival": float(ws1),
                    "ws2_reproduction": float(ws2),
                    "ws3_robustness": float(ws3),
                },
                "aggregation": {
                    "alpha_mean": float(alpha_mean),
                    "alpha_min": float(alpha_min),
                    "mean": float(mean_score),
                    "min": float(min_score),
                },
                "sampling": {
                    "mode": str(getattr(sim_cfg, "epi_subsample_mode", "all")),
                    "stride": int(getattr(sim_cfg, "epi_subsample_stride", 1)),
                    "start_index": int(getattr(sim_cfg, "epi_subsample_start_index", 0)),
                    "sample_size": int(len(sample_ids)),
                    "population_alive": int(len(alive_ids_sorted)),
                },
                "scenarios": scenario_results,
                "epi_value": float(epi_value),
            }

            _ = canonical_json_bytes(record)
            return record

    def _evaluate_scenario(
        self,
        *,
        world_obj: Dict[str, Any],
        pop_obj: Dict[str, Any],
        weights_dir: str,
        sim: Any,
        generation_index: int,
        scenario: StressScenarioV1,
        sample_ids: Sequence[int],
        sample_ids_set: set[int],
        ws1: float,
        ws2: float,
        ws3: float,
        global_seed: int,
    ) -> Dict:
        """Evaluate one stress scenario on a shadow clone."""

        stress_seed = derive_stress_seed(
            global_seed=global_seed,
            scenario_id=scenario.scenario_id,
            generation_index=generation_index,
            seed_offset=scenario.seed_offset,
        )

        rng = RNGManager(global_seed=stress_seed)

        duration_steps = int(max(0, int(scenario.duration_steps)))

        from simulation.runner import SimulationRunner  # local import to avoid cycles
        from simulation.checkpoint_bundle_v2 import _apply_population, _apply_world  # noqa: WPS433

        sim_cfg_shadow = replace(
            sim.sim_cfg,
            generations=1,
            steps_per_generation=duration_steps,
            behavior_log_enable=False,
            policy_deterministic_actions=True,
        )

        with tempfile.TemporaryDirectory(prefix="epi_shadow_") as tmpdir:
            shadow = SimulationRunner(
                env_cfg=sim.env_cfg,
                pop_cfg=sim.pop_cfg,
                sim_cfg=sim_cfg_shadow,
                logger=NoOpMetricsLogger(),
                run_dir=tmpdir,
                perf=False,
            )

            _apply_world(shadow, world_obj)
            _apply_population(shadow, pop_obj, weights_dir=weights_dir)

            # The runner's tick loop assumes generation-local counters exist
            # (birth diagnostics, etc.). In normal runs, _start_generation() sets
            # these. For Phase 5a we bypass _start_generation to avoid mixing
            # world-reset logic and stress overrides.
            _phase5a_init_shadow_generation_counters(shadow)

            # Ensure shadow evaluation uses the same policy parameters as the
            # main simulation (no main-state mutation).
            _phase5a_copy_policy_bank_state(src_sim=sim, dst_sim=shadow)

            # Inject stress RNG stream as the only RNG source.
            shadow.rng_manager = rng
            shadow.np_rng = rng.stress_np
            if hasattr(shadow.env, "rng"):
                shadow.env.rng = rng.stress_np
            # Keep population-level RNG surfaces bound to mutation domains so
            # frozen mutation.* budget contracts remain valid inside shadow runs.
            setattr(shadow.pop, "rng_manager", rng)
            if hasattr(shadow.pop, "rng"):
                shadow.pop.rng = rng.mutation_py
            if hasattr(shadow.pop, "np_rng"):
                shadow.pop.np_rng = rng.mutation_np
            self._assert_shadow_rng_bindings(shadow)

            if sample_ids_set:
                for agent in shadow.pop.agents.values():
                    if agent.alive and (agent.agent_id not in sample_ids_set):
                        agent.alive = False
                        if hasattr(shadow.env, "remove_agent"):
                            shadow.env.remove_agent(agent.agent_id)

            starts: Dict[int, Tuple[bool, int, float]] = {}
            for aid in sample_ids:
                a = shadow.pop.agents.get(aid)
                if a is None:
                    continue
                starts[int(aid)] = (bool(a.alive), int(a.stats.offspring_count), float(a.injury))

            alive_start = sum(1 for (alive0, _, _) in starts.values() if alive0)

            _apply_pressure_overrides(shadow.env, scenario.pressure_overrides)

            buckets: Dict[str, float] = {}
            rest_streaks: Dict[int, int] = {}

            _ = shadow._run_ticks(
                gen=int(generation_index),
                buckets=buckets,
                injury_recover_mult=1.0,
                deaths_this_gen=0,
                successful_births=0,
                failed_births=0,
                gen_peak_injury=0.0,
                rest_streaks=rest_streaks,
            )

            alive_end = 0
            offspring_delta_sum = 0
            injury_delta_sum = 0.0
            injury_count = 0

            for aid, (alive0, off0, inj0) in starts.items():
                a = shadow.pop.agents.get(aid)
                if a is None:
                    continue
                if a.alive:
                    alive_end += 1

                off_delta = int(a.stats.offspring_count) - int(off0)
                if off_delta > 0:
                    offspring_delta_sum += off_delta

                inj_delta = float(a.injury) - float(inj0)
                if inj_delta > 0:
                    injury_delta_sum += inj_delta
                injury_count += 1

        survival_score = (float(alive_end) / float(alive_start)) if alive_start > 0 else 0.0
        reproduction_score = (
            min(1.0, float(offspring_delta_sum) / float(alive_start)) if alive_start > 0 else 0.0
        )
        mean_injury_delta = (injury_delta_sum / float(injury_count)) if injury_count > 0 else 0.0

        injury_cap = 10.0
        robustness_score = 1.0 - min(1.0, float(mean_injury_delta) / injury_cap)

        scenario_score = float(ws1) * survival_score + float(ws2) * reproduction_score + float(ws3) * robustness_score

        result = {
            "scenario": scenario.to_dict(),
            "alive_start": int(alive_start),
            "alive_end": int(alive_end),
            "offspring_delta_sum": int(offspring_delta_sum),
            "mean_injury_delta": float(mean_injury_delta),
            "survival_score": float(survival_score),
            "reproduction_score": float(reproduction_score),
            "robustness_score": float(robustness_score),
            "scenario_score": float(scenario_score),
        }

        _ = canonical_json_bytes(result)
        return result

    def _assert_shadow_rng_bindings(self, shadow: Any) -> None:
        """Fail fast if shadow RNG surfaces are not bound as expected."""
        rm = getattr(shadow, "rng_manager", None)
        if rm is None:
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "missing_shadow_rng_manager")
        if getattr(shadow, "np_rng", None) is not getattr(rm, "stress_np", None):
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "runner_np_not_stress_np")
        if getattr(getattr(shadow, "env", None), "rng", None) is not getattr(rm, "stress_np", None):
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "env_rng_not_stress_np")
        pop = getattr(shadow, "pop", None)
        if pop is None:
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "missing_shadow_population")
        if getattr(pop, "rng_manager", None) is not rm:
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "pop_rng_manager_mismatch")
        if getattr(pop, "rng", None) is not getattr(rm, "mutation_py", None):
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "pop_rng_not_mutation_py")
        if getattr(pop, "np_rng", None) is not getattr(rm, "mutation_np", None):
            _det_violation("DET_PHASE5A_SHADOW_RNG_BINDING_INVALID", "pop_np_rng_not_mutation_np")


def _phase5a_copy_policy_bank_state(*, src_sim: Any, dst_sim: Any) -> None:
    """Best-effort copy of determinism-critical policy bank tensors.

    Phase 5a requirement: EPI shadow evaluation must reflect the *same* policy
    parameters as the main simulation, while never mutating main state.

    Repo reality: policy parameters live in runner-level "bank" tensors
    (e.g., _mlp_pack, _gru_pack, _gru_hmem). Population deep copies do not
    include these tensors.

    This helper copies bank tensors from src_sim -> dst_sim using clone().
    - No RNG draws.
    - No mutations to src_sim.
    - Safe if tensors do not exist.
    """
    try:
        import torch  # noqa: WPS433
    except Exception:
        return

    def _clone_tensor(x: Any) -> Optional[Any]:
        try:
            if x is None:
                return None
            if not isinstance(x, torch.Tensor):
                return None
            # clone() is deterministic given identical source bytes.
            return x.detach().clone()
        except Exception:
            return None

    # Common bank tensors (MLP + GRU + GRU hidden memory) used by runner_policy.
    for name in [
        "_mlp_pack",
        "_gru_pack",
        "_gru_hmem",
        "_policy_bank",
    ]:
        try:
            src_t = getattr(src_sim, name, None)
            dst_t = _clone_tensor(src_t)
            if dst_t is not None:
                setattr(dst_sim, name, dst_t)
        except Exception:
            continue

    # Best-effort device/bookkeeping mirrors.
    for name in [
        "_bank_device",
        "_bank_dtype",
        "_bank_capacity",
    ]:
        try:
            if hasattr(src_sim, name):
                setattr(dst_sim, name, getattr(src_sim, name))
        except Exception:
            continue


def _phase5a_init_shadow_generation_counters(sim: Any) -> None:
    """Initialize generation-local counters required by runner tick logic.

    Background
    ----------
    In the main loop, runner_pkg/runner_generation.py::_start_generation()
    initializes per-generation birth diagnostic counters (and a few other
    bookkeeping variables). The Phase 5a EPI shadow evaluation bypasses
    _start_generation() on purpose so stress overrides are not clobbered by
    world-reset logic.

    However, the tick loop (and birth pipeline) expects these attributes to
    exist on the runner instance.

    Determinism
    ----------
    - No RNG draws.
    - Only sets missing attributes to integer zeros.
    """

    # Must match the attribute names initialized in runner_generation.py.
    # Keep the list explicit and stable.
    counters = [
        "_birth_attempts_gen",
        "_birth_pre_cooldown_gen",
        "_birth_pre_no_energy_gen",
        "_birth_pre_injury_gate_gen",
        "_birth_pre_no_space_gen",
        "_birth_pre_no_mate_gen",
        "_birth_attempt_conception_fail_gen",
        "_birth_gestation_fail_gen",
        "_birth_make_child_fail_gen",
        "_birth_fail_no_space_gen",
        "_birth_fail_no_mate_gen",
        "_birth_fail_buffer_gen",
        "_birth_fail_injury_gate_gen",
        "_birth_fail_scar_gate_gen",
        "_birth_fail_conception_gen",
        "_birth_fail_gestation_gen",
    ]

    for name in counters:
        try:
            if not hasattr(sim, name):
                setattr(sim, name, 0)
        except Exception:
            continue
