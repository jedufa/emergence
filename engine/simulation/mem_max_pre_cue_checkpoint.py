from __future__ import annotations

"""Default-off MEM-MAX pre-cue checkpoint vertical slice.

The recorder is intentionally diagnostic-only. It attaches a hook to the live
delayed-choice substrate and captures an isolated in-memory runner clone before
cue/selected-target treatment is written into the world.
"""

import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch

from env.gridworld_pkg.patch_world import MEM_MAX_DELAYED_CHOICE_ARRAY_NAMES
from evolution.policy_lineage_telemetry import policy_hash as lineage_policy_hash
from evolution.policy_lineage_telemetry import summarize_events as summarize_lineage_events


SCHEMA_VERSION = "mem_max_pre_cue_checkpoint_v1"
HOOK_ATTR = "_mem_max_pre_cue_checkpoint_hook"
FORCED_TARGET_ATTR = "_mem_max_pre_cue_forced_correct_target"


def _json_sha(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha(value: Any) -> str:
    arr = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _tensor_sha(value: Any) -> str:
    if value is None:
        return ""
    try:
        tensor = value.detach().to("cpu").contiguous().float()
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    except Exception:
        return "unavailable"


def _policy_hash(policy: Any) -> str:
    try:
        return str(lineage_policy_hash(policy))
    except Exception:
        return ""


def _agent_rows(runner: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    agents = getattr(getattr(runner, "pop", None), "agents", {}) or {}
    for aid in sorted(int(v) for v in agents.keys()):
        agent = agents[int(aid)]
        genome = getattr(agent, "genome", None)
        rows.append(
            {
                "agent_id": int(aid),
                "alive": bool(getattr(agent, "alive", False)),
                "slot": int(getattr(agent, "slot", -1) or -1),
                "slot_token": int(getattr(agent, "slot_token", 0) or 0),
                "policy_type": type(getattr(agent, "policy", None)).__name__,
                "policy_hash": _policy_hash(getattr(agent, "policy", None)),
                "memory_dim": int(getattr(genome, "memory_dim", 0) or 0) if genome is not None else 0,
                "use_gru": bool(getattr(genome, "use_gru", False)) if genome is not None else False,
            }
        )
    return rows


def runner_state_digest(runner: Any) -> Dict[str, Any]:
    env = getattr(runner, "env", None)
    env_arrays: Dict[str, str] = {}
    if env is not None:
        for name in MEM_MAX_DELAYED_CHOICE_ARRAY_NAMES:
            if hasattr(env, name):
                env_arrays[name] = _array_sha(getattr(env, name))
        for name in ("grid", "agent_alive", "agent_pos_y", "agent_pos_x", "food"):
            if hasattr(env, name):
                env_arrays[name] = _array_sha(getattr(env, name))

    obs_sha = ""
    try:
        alive_ids = np.asarray(
            [int(getattr(agent, "agent_id")) for agent in getattr(runner.pop, "alive_agents")()],
            dtype=np.int32,
        )
        if alive_ids.size:
            obs_sha = _array_sha(runner.env.get_local_observations(alive_ids))
    except Exception:
        obs_sha = "unavailable"

    rng_snapshot: Mapping[str, Any] = {}
    try:
        rng_snapshot = runner.rng_manager.capture_snapshot()
    except Exception:
        rng_snapshot = {}

    tensors = {
        name: _tensor_sha(getattr(runner, name, None))
        for name in (
            "_W1",
            "_b1",
            "_W2",
            "_b2",
            "_W3",
            "_b3",
            "_gru_Wih",
            "_gru_Whh",
            "_gru_bih",
            "_gru_bhh",
            "_gru_Wout",
            "_gru_bout",
            "_Hmem",
        )
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "env_arrays": env_arrays,
        "agents": _agent_rows(runner),
        "rng_snapshot_sha256": _json_sha(rng_snapshot),
        "rng_indices": getattr(getattr(runner, "rng_manager", None), "indices_snapshot", lambda: {})(),
        "tensor_sha256": tensors,
        "observation_sha256": obs_sha,
    }
    payload["state_sha256"] = _json_sha(payload)
    return payload


def _temporarily_clear_live_hooks(runner: Any) -> Dict[str, Any]:
    env = getattr(runner, "env", None)
    saved: Dict[str, Any] = {}
    if env is not None and hasattr(env, HOOK_ATTR):
        saved[f"env.{HOOK_ATTR}"] = getattr(env, HOOK_ATTR)
        setattr(env, HOOK_ATTR, None)
    if hasattr(env, FORCED_TARGET_ATTR):
        saved[f"env.{FORCED_TARGET_ATTR}"] = getattr(env, FORCED_TARGET_ATTR)
        delattr(env, FORCED_TARGET_ATTR)
    if hasattr(runner, "mem_max_pre_cue_checkpoint_recorder"):
        saved["runner.recorder"] = getattr(runner, "mem_max_pre_cue_checkpoint_recorder")
        setattr(runner, "mem_max_pre_cue_checkpoint_recorder", None)
    return saved


def _restore_live_hooks(runner: Any, saved: Mapping[str, Any]) -> None:
    env = getattr(runner, "env", None)
    if env is not None and f"env.{HOOK_ATTR}" in saved:
        setattr(env, HOOK_ATTR, saved[f"env.{HOOK_ATTR}"])
    if env is not None and f"env.{FORCED_TARGET_ATTR}" in saved:
        setattr(env, FORCED_TARGET_ATTR, saved[f"env.{FORCED_TARGET_ATTR}"])
    if "runner.recorder" in saved:
        setattr(runner, "mem_max_pre_cue_checkpoint_recorder", saved["runner.recorder"])


class MemMaxPreCueCheckpointRecorder:
    """Capture default-off pre-cue runner snapshots from the live substrate."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.max_snapshots = max(1, int(getattr(runner.sim_cfg, "mem_max_pre_cue_checkpoint_max_snapshots", 1) or 1))
        root = getattr(runner.sim_cfg, "mem_max_pre_cue_checkpoint_dir", "") or ""
        self.out_dir = Path(root) if root else Path(str(getattr(runner, "run_dir", "."))) / "mem_max_pre_cue_checkpoints"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[Dict[str, Any]] = []
        setattr(runner.env, HOOK_ATTR, self.capture_pre_cue)

    @classmethod
    def maybe_create(cls, runner: Any) -> "MemMaxPreCueCheckpointRecorder | None":
        if not bool(getattr(runner.sim_cfg, "mem_max_pre_cue_checkpoint_enable", False)):
            return None
        return cls(runner)

    def capture_pre_cue(
        self,
        *,
        env: Any,
        abs_step: int,
        event_index: int,
        event_id: int,
        correct_target: int,
        target_a: Tuple[int, int],
        target_b: Tuple[int, int],
    ) -> Dict[str, Any] | None:
        if len(self.snapshots) >= int(self.max_snapshots):
            return None
        runner = self.runner
        before = runner_state_digest(runner)
        saved = _temporarily_clear_live_hooks(runner)
        try:
            runner_clone = copy.deepcopy(runner)
        finally:
            _restore_live_hooks(runner, saved)
        clone_saved = _temporarily_clear_live_hooks(runner_clone)
        _restore_live_hooks(runner_clone, {})
        after_live = runner_state_digest(runner)
        clone_digest = runner_state_digest(runner_clone)
        no_aliasing = _alias_audit(runner, runner_clone)
        snapshot_id = f"pre_cue_abs{int(abs_step):08d}_event{int(event_index):04d}_{len(self.snapshots):03d}"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "abs_step": int(abs_step),
            "event_index": int(event_index),
            "event_id": int(event_id),
            "correct_target": int(correct_target),
            "target_a": [int(target_a[0]), int(target_a[1])],
            "target_b": [int(target_b[0]), int(target_b[1])],
            "state_sha256": before["state_sha256"],
            "clone_state_sha256": clone_digest["state_sha256"],
            "live_noninterference": before["state_sha256"] == after_live["state_sha256"],
            "clone_identity_match": before["state_sha256"] == clone_digest["state_sha256"],
            "no_aliasing": no_aliasing,
            "digest": before,
        }
        snapshot_path = self.out_dir / f"{snapshot_id}.runner.pkl.gz"
        with gzip.open(snapshot_path, "wb") as fh:
            pickle.dump({"metadata": payload, "runner": runner_clone}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        snapshot_sha = _bytes_sha(snapshot_path.read_bytes())
        payload["snapshot_path"] = str(snapshot_path)
        payload["snapshot_sha256"] = snapshot_sha
        payload["roundtrip"] = roundtrip_snapshot(snapshot_path)
        meta_path = self.out_dir / f"{snapshot_id}.json"
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.snapshots.append(payload)
        summary_path = self.out_dir / "pre_cue_checkpoint_summary.json"
        summary_path.write_text(json.dumps({"snapshots": self.snapshots}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload


def _alias_audit(left: Any, right: Any) -> Dict[str, Any]:
    checks: Dict[str, bool] = {"runner_distinct": id(left) != id(right), "env_distinct": id(left.env) != id(right.env)}
    for name in ("agent_alive", "agent_pos_y", "agent_pos_x"):
        checks[f"env_{name}_distinct"] = id(getattr(left.env, name, None)) != id(getattr(right.env, name, None))
    checks["population_distinct"] = id(getattr(left, "pop", None)) != id(getattr(right, "pop", None))
    left_hmem = getattr(left, "_Hmem", None)
    right_hmem = getattr(right, "_Hmem", None)
    checks["hmem_distinct"] = (left_hmem is None and right_hmem is None) or id(left_hmem) != id(right_hmem)
    return {"checks": checks, "passed": bool(all(checks.values()))}


def load_snapshot(snapshot_path: str | Path) -> Dict[str, Any]:
    with gzip.open(Path(snapshot_path), "rb") as fh:
        return pickle.load(fh)


def roundtrip_snapshot(snapshot_path: str | Path) -> Dict[str, Any]:
    loaded = load_snapshot(snapshot_path)
    metadata = dict(loaded.get("metadata") or {})
    runner = loaded.get("runner")
    digest = runner_state_digest(runner)
    return {
        "loaded": runner is not None,
        "state_sha256": digest.get("state_sha256", ""),
        "matches_metadata": bool(digest.get("state_sha256", "") == metadata.get("state_sha256", "")),
    }


def _alive_agents_for_branch(runner: Any) -> List[Any]:
    agents = list(getattr(runner.pop, "alive_agents")())
    agents.sort(key=lambda agent: int(getattr(agent, "agent_id")))
    try:
        runner.np_rng.shuffle(agents)
    except Exception:
        pass
    return agents


def deterministic_focal_policy_sample(runner: Any) -> Dict[str, Any]:
    """Return outcome-independent focal agents by structural policy category."""
    from agents.cognition import effective_memory_dim_for_genome  # local import avoids widening base imports

    rows = _agent_rows(runner)
    by_id = {int(row["agent_id"]): row for row in rows}
    agents = getattr(getattr(runner, "pop", None), "agents", {}) or {}
    gru_memory: List[int] = []
    mlp_memory_zero: List[int] = []
    for aid_raw in sorted(int(v) for v in agents.keys()):
        agent = agents[int(aid_raw)]
        genome = getattr(agent, "genome", None)
        eff = int(effective_memory_dim_for_genome(genome, sim_cfg=getattr(runner, "sim_cfg", None)))
        row = by_id.get(int(aid_raw), {})
        row["effective_memory_dim"] = int(eff)
        if str(row.get("policy_type", "")) == "GRUPolicy" and eff > 0:
            gru_memory.append(int(aid_raw))
        if eff <= 0 or str(row.get("policy_type", "")) != "GRUPolicy":
            mlp_memory_zero.append(int(aid_raw))
    return {
        "schema_version": "mem_max_24_focal_policy_sample_v1",
        "rule": "sorted_agent_id_by_structural_category",
        "gru_memory_positive_agent_ids": gru_memory,
        "mlp_or_memory_zero_agent_ids": mlp_memory_zero,
        "selected_gru_focal_agent_id": int(gru_memory[0]) if gru_memory else None,
        "selected_control_focal_agent_id": int(mlp_memory_zero[0]) if mlp_memory_zero else None,
        "agents": rows,
    }


def _action_masses(probs: np.ndarray, pos: Tuple[int, int], target_a: Tuple[int, int], target_b: Tuple[int, int]) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=np.float32)
    move_probs = probs[:4] if probs.size >= 4 else np.zeros((4,), dtype=np.float32)

    def direction_mass(target: Tuple[int, int]) -> float:
        dy = int(target[0]) - int(pos[0])
        dx = int(target[1]) - int(pos[1])
        masses: List[float] = []
        if dy < 0:
            masses.append(float(move_probs[0]))
        elif dy > 0:
            masses.append(float(move_probs[1]))
        if dx > 0:
            masses.append(float(move_probs[2]))
        elif dx < 0:
            masses.append(float(move_probs[3]))
        return float(max(masses)) if masses else 0.0

    neutral_mass = float(np.sum(probs[4:], dtype=np.float64)) if probs.size > 4 else 0.0
    movement_mass = float(np.sum(move_probs, dtype=np.float64))
    return {
        "movement_mass": movement_mass,
        "neutral_mass": neutral_mass,
        "target_a_mass": direction_mass(target_a),
        "target_b_mass": direction_mass(target_b),
        "movement_vs_neutral_margin": movement_mass - neutral_mass,
    }


def _position(runner: Any, aid: int) -> Tuple[int, int]:
    return (int(runner.env.agent_pos_y[int(aid)]), int(runner.env.agent_pos_x[int(aid)]))


def _at_target(pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
    return int(pos[0]) == int(target[0]) and int(pos[1]) == int(target[1])


def _hmem_row_sha(runner: Any, aid: int) -> str:
    try:
        agent = runner.pop.agents[int(aid)]
        slot = int(getattr(agent, "slot", -1))
        hmem = getattr(runner, "_Hmem", None)
        if hmem is None or slot < 0:
            return ""
        row = hmem[int(slot)].detach().to("cpu").contiguous().float().numpy()
        return _array_sha(row)
    except Exception:
        return "unavailable"


def _delayed_choice_counter_snapshot(runner: Any) -> Dict[str, Any]:
    keys = [
        "_mem_max_delayed_choice_correct_arrival_count_gen",
        "_mem_max_delayed_choice_incorrect_arrival_count_gen",
        "_mem_max_delayed_choice_no_choice_count_gen",
        "_mem_max_delayed_choice_consequence_applied_count_gen",
        "_mem_max_delayed_choice_consequence_eligible_count_gen",
        "_mem_max_delayed_choice_target_a_arrival_count_gen",
        "_mem_max_delayed_choice_target_b_arrival_count_gen",
        "_mem_max_delayed_choice_applied_health_damage_total_gen",
        "_mem_max_delayed_choice_applied_energy_cost_total_gen",
    ]
    return {key: getattr(runner, key, 0) for key in keys}


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: float(after.get(key, 0) or 0) - float(before.get(key, 0) or 0) for key in sorted(set(before) | set(after))}


def execute_mid_generation_branch_continuation(
    snapshot_path: str | Path,
    *,
    forced_correct_target: int,
    focal_agent_id: int,
    memory_ablated: bool = False,
    cue_enabled: bool = True,
    consequence_enabled: bool = True,
    label: str = "",
) -> Dict[str, Any]:
    """Run a restored pre-cue clone through production tick stages and env execution."""
    snapshot_path = Path(snapshot_path)
    loaded = load_snapshot(snapshot_path)
    metadata = dict(loaded.get("metadata") or {})
    runner = loaded["runner"]
    env = runner.env
    target = int(forced_correct_target)
    if target not in (1, 2):
        raise ValueError("forced_correct_target must be 1 or 2")
    setattr(env, FORCED_TARGET_ATTR, target)
    runner.sim_cfg.force_memory_disabled = bool(memory_ablated)
    runner.sim_cfg.reset_memory_each_tick = bool(memory_ablated)
    runner.env_cfg.mem_max_delayed_choice_cue_enable = bool(cue_enabled)
    runner.env_cfg.mem_max_delayed_choice_consequence_enable = bool(consequence_enabled)

    abs_step = int(metadata["abs_step"])
    cue_steps = int(getattr(runner.env_cfg, "mem_max_delayed_choice_cue_steps", 1) or 1)
    hidden_steps = int(getattr(runner.env_cfg, "mem_max_delayed_choice_hidden_steps", 1) or 1)
    active_steps = int(getattr(runner.env_cfg, "mem_max_delayed_choice_active_steps", 1) or 1)
    decision_abs_step = abs_step + cue_steps + hidden_steps
    event_end_abs_step = decision_abs_step + active_steps
    target_a = tuple(int(v) for v in metadata.get("target_a", [0, 0]))
    target_b = tuple(int(v) for v in metadata.get("target_b", [0, 0]))

    initial_digest = runner_state_digest(runner)
    initial_rng = getattr(getattr(runner, "rng_manager", None), "capture_snapshot", lambda: {})()
    initial_hmem = _hmem_row_sha(runner, int(focal_agent_id))
    tick_rows: List[Dict[str, Any]] = []
    decision_row: Dict[str, Any] = {}
    deaths = 0
    peak_injury = 0.0
    counters_before = _delayed_choice_counter_snapshot(runner)

    for tick in range(abs_step, event_end_abs_step + 1):
        runner._eiel_fdp_cur_gen = 0
        runner._eiel_fdp_cur_tick = int(tick)
        runner.env.patch_world_step(int(tick))
        alive = _alive_agents_for_branch(runner)
        agent_ids_np = np.asarray([int(getattr(agent, "agent_id")) for agent in alive], dtype=np.int32)
        pre_pos = _position(runner, int(focal_agent_id))
        pre_health = float(getattr(runner.pop.agents[int(focal_agent_id)], "health", 0.0))
        pre_energy = float(getattr(runner.pop.agents[int(focal_agent_id)], "energy", 0.0))
        actions_np = runner._batched_actions_strided(alive, buckets={})
        policy = getattr(runner, "_mem_max_live_level4_last_policy", {}) or {}
        probs = np.asarray((policy.get("probs_by_agent_id") or {}).get(int(focal_agent_id), np.zeros((7,), dtype=np.float32)), dtype=np.float32)
        logits = np.asarray((policy.get("logits_by_agent_id") or {}).get(int(focal_agent_id), np.zeros((7,), dtype=np.float32)), dtype=np.float32)
        action_by_id = {int(agent_ids_np[i]): int(actions_np[i]) for i in range(int(agent_ids_np.shape[0]))}
        focal_action = int(action_by_id.get(int(focal_agent_id), -1))
        deaths, peak_injury = runner._apply_env_tick(
            alive,
            agent_ids_np,
            np.asarray(actions_np, dtype=np.int32),
            {},
            int(deaths),
            float(peak_injury),
        )
        deaths = runner._apply_mem_max_delayed_choice_consequence(int(deaths))
        post_pos = _position(runner, int(focal_agent_id))
        post_health = float(getattr(runner.pop.agents[int(focal_agent_id)], "health", 0.0))
        post_energy = float(getattr(runner.pop.agents[int(focal_agent_id)], "energy", 0.0))
        pre_at_a = _at_target(pre_pos, target_a)
        pre_at_b = _at_target(pre_pos, target_b)
        post_at_a = _at_target(post_pos, target_a)
        post_at_b = _at_target(post_pos, target_b)
        row = {
            "tick": int(tick),
            "phase": "decision_active" if int(tick) >= int(decision_abs_step) else ("cue" if int(tick) == int(abs_step) else "hidden"),
            "focal_agent_id": int(focal_agent_id),
            "pre_position_y": int(pre_pos[0]),
            "pre_position_x": int(pre_pos[1]),
            "post_position_y": int(post_pos[0]),
            "post_position_x": int(post_pos[1]),
            "chosen_action": int(focal_action),
            "executed_action": int(focal_action),
            "movement_success": bool(pre_pos != post_pos),
            "target_occupancy_before": "A" if pre_at_a else ("B" if pre_at_b else "none"),
            "target_occupancy_after": "A" if post_at_a else ("B" if post_at_b else "none"),
            "new_target_entry": bool((not (pre_at_a or pre_at_b)) and (post_at_a or post_at_b)),
            "new_correct_target_entry": bool((not (pre_at_a or pre_at_b)) and ((target == 1 and post_at_a) or (target == 2 and post_at_b))),
            "health_delta": float(post_health - pre_health),
            "energy_delta": float(post_energy - pre_energy),
            "logits_sha256": _array_sha(logits),
            "probs": [float(v) for v in probs.tolist()],
            **_action_masses(probs, pre_pos, target_a, target_b),
        }
        tick_rows.append(row)
        if int(tick) == int(decision_abs_step):
            decision_row = dict(row)

    counters_after = _delayed_choice_counter_snapshot(runner)
    final_digest = runner_state_digest(runner)
    final_rng = getattr(getattr(runner, "rng_manager", None), "capture_snapshot", lambda: {})()
    agent_row = {int(row["agent_id"]): row for row in _agent_rows(runner)}.get(int(focal_agent_id), {})
    return {
        "schema_version": "mem_max_24_mid_generation_branch_continuation_v1",
        "label": str(label),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": _bytes_sha(snapshot_path.read_bytes()),
        "shared_state_sha256": str(metadata.get("state_sha256", "")),
        "forced_correct_target": int(target),
        "memory_ablated": bool(memory_ablated),
        "cue_enabled": bool(cue_enabled),
        "consequence_enabled": bool(consequence_enabled),
        "focal_agent_id": int(focal_agent_id),
        "focal_policy_type": str(agent_row.get("policy_type", "")),
        "focal_memory_dim": int(agent_row.get("memory_dim", 0) or 0),
        "focal_use_gru": bool(agent_row.get("use_gru", False)),
        "initial_hmem_sha256": initial_hmem,
        "final_hmem_sha256": _hmem_row_sha(runner, int(focal_agent_id)),
        "decision_abs_step": int(decision_abs_step),
        "event_end_abs_step": int(event_end_abs_step),
        "ticks": tick_rows,
        "decision": decision_row,
        "counter_delta": _counter_delta(counters_before, counters_after),
        "initial_state_sha256": initial_digest.get("state_sha256", ""),
        "final_state_sha256": final_digest.get("state_sha256", ""),
        "initial_rng_sha256": _json_sha(initial_rng),
        "final_rng_sha256": _json_sha(final_rng),
        "runner": runner,
    }


def execute_exact_mid_generation_pair(
    snapshot_path: str | Path,
    *,
    focal_agent_id: int,
    memory_ablated: bool = False,
    cue_enabled: bool = True,
    consequence_enabled: bool = True,
    label: str = "",
) -> Dict[str, Any]:
    branches = []
    for target in (1, 2):
        branch = execute_mid_generation_branch_continuation(
            snapshot_path,
            forced_correct_target=target,
            focal_agent_id=int(focal_agent_id),
            memory_ablated=bool(memory_ablated),
            cue_enabled=bool(cue_enabled),
            consequence_enabled=bool(consequence_enabled),
            label=f"{label}_target_{target}",
        )
        branch.pop("runner", None)
        branches.append(branch)
    a = branches[0]["decision"]
    b = branches[1]["decision"]
    return {
        "schema_version": "mem_max_24_exact_mid_generation_pair_v1",
        "label": str(label),
        "focal_agent_id": int(focal_agent_id),
        "memory_ablated": bool(memory_ablated),
        "cue_enabled": bool(cue_enabled),
        "consequence_enabled": bool(consequence_enabled),
        "shared_state_sha256": branches[0].get("shared_state_sha256", ""),
        "branches": branches,
        "logit_divergence": 0.0 if a.get("logits_sha256") == b.get("logits_sha256") else 1.0,
        "hidden_state_separation": 0.0 if branches[0].get("final_hmem_sha256") == branches[1].get("final_hmem_sha256") else 1.0,
        "signed_target_mass_delta": float(
            (float(a.get("target_a_mass", 0.0)) - float(a.get("target_b_mass", 0.0)))
            + (float(b.get("target_b_mass", 0.0)) - float(b.get("target_a_mass", 0.0)))
        ),
        "chosen_action_differs": bool(a.get("chosen_action") != b.get("chosen_action")),
        "position_transition_differs": bool(
            (a.get("post_position_y"), a.get("post_position_x")) != (b.get("post_position_y"), b.get("post_position_x"))
        ),
    }


def _clear_checkpoint_hooks(runner: Any) -> None:
    env = getattr(runner, "env", None)
    if hasattr(runner, "mem_max_pre_cue_checkpoint_recorder"):
        setattr(runner, "mem_max_pre_cue_checkpoint_recorder", None)
    if env is not None and hasattr(env, HOOK_ATTR):
        setattr(env, HOOK_ATTR, None)
    if env is not None and hasattr(env, FORCED_TARGET_ATTR):
        delattr(env, FORCED_TARGET_ATTR)


def _lineage_runtime_semantics(runner: Any) -> Dict[str, Any]:
    return {
        "cue_enabled": bool(getattr(runner.env_cfg, "mem_max_delayed_choice_cue_enable", True)),
        "memory_ablated": bool(getattr(runner.sim_cfg, "force_memory_disabled", False)),
        "hidden_steps": int(getattr(runner.env_cfg, "mem_max_delayed_choice_hidden_steps", 2) or 2),
    }


def _inject_stage_policy(runner: Any, *, focal_agent_id: int, policy: Any, memory_dim: int) -> None:
    agent = runner.pop.agents[int(focal_agent_id)]
    agent.policy = copy.deepcopy(policy)
    genome = getattr(agent, "genome", None)
    policy_type = type(agent.policy).__name__
    if genome is not None:
        if hasattr(genome, "use_gru"):
            setattr(genome, "use_gru", bool(policy_type == "GRUPolicy"))
        if hasattr(genome, "memory_dim"):
            setattr(genome, "memory_dim", int(memory_dim) if policy_type == "GRUPolicy" else 0)
    try:
        if hasattr(runner, "_sync_bank_for_agents"):
            runner._sync_bank_for_agents([agent], buckets=None)
        if hasattr(runner, "_sync_gru_bank_for_agents"):
            runner._sync_gru_bank_for_agents([agent], buckets=None)
        if hasattr(runner, "_reset_memory_slots"):
            slot = int(getattr(agent, "slot", -1))
            if slot >= 0:
                runner._reset_memory_slots(torch.tensor([slot], device=getattr(runner, "_bank_device", "cpu"), dtype=torch.long))
    except Exception:
        raise


def execute_policy_stage_common_state_pair(
    snapshot_path: str | Path,
    *,
    stage_label: str,
    stage_policy: Any,
    focal_agent_id: int,
    memory_dim: int,
) -> Dict[str, Any]:
    """Evaluate one captured policy stage on the same pre-cue state.

    This is a diagnostic-only wrapper around the existing paired branch replay.
    It writes a temporary injected snapshot beside the source snapshot so the
    normal branch execution path remains unchanged.
    """
    if stage_policy is None:
        return {
            "schema_version": "mem_max_25_common_state_policy_stage_pair_v1",
            "stage_label": str(stage_label),
            "available": False,
            "reason": "missing_stage_policy",
        }
    loaded = load_snapshot(snapshot_path)
    runner = loaded["runner"]
    _clear_checkpoint_hooks(runner)
    _inject_stage_policy(runner, focal_agent_id=int(focal_agent_id), policy=stage_policy, memory_dim=int(memory_dim))
    metadata = dict(loaded.get("metadata") or {})
    temp_path = Path(snapshot_path).with_name(f"{Path(snapshot_path).stem}.{stage_label}.mem_max_25.runner.pkl.gz")
    try:
        with gzip.open(temp_path, "wb") as fh:
            pickle.dump({"metadata": metadata, "runner": runner}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        pair = execute_exact_mid_generation_pair(
            temp_path,
            focal_agent_id=int(focal_agent_id),
            label=f"mem_max_25_{stage_label}",
        )
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    pair["schema_version"] = "mem_max_25_common_state_policy_stage_pair_v1"
    pair["stage_label"] = str(stage_label)
    pair["stage_policy_type"] = type(stage_policy).__name__
    pair["stage_policy_hash"] = _policy_hash(stage_policy)
    pair["stage_memory_dim"] = int(memory_dim)
    return pair


def _stage_event_summary(event: Mapping[str, Any], stage_record: Mapping[str, Any] | None) -> Dict[str, Any]:
    stage_record = stage_record or {}
    return {
        "schema_version": "mem_max_25_policy_stage_reconciliation_v1",
        "event_id": int(event.get("event_id", -1)),
        "policy_donor_parent_id": int(event.get("policy_donor_parent_id", -1)),
        "child_agent_id": int(event.get("child_agent_id", -1)),
        "stages_present": {
            "policy_donor_parent": stage_record.get("policy_donor_parent") is not None,
            "child_pre_mutation": stage_record.get("child_pre_mutation") is not None,
            "child_post_mutation": stage_record.get("child_post_mutation") is not None,
            "final_attached_child": stage_record.get("final_attached_child") is not None,
        },
        "hashes": {
            "event_parent_policy_hash": str(event.get("parent_policy_hash", "")),
            "captured_parent_policy_hash": _policy_hash(stage_record.get("policy_donor_parent")),
            "event_child_pre_mutation_policy_hash": str(event.get("child_pre_mutation_policy_hash", "")),
            "captured_child_pre_mutation_policy_hash": _policy_hash(stage_record.get("child_pre_mutation")),
            "event_child_post_mutation_policy_hash": str(event.get("child_post_mutation_policy_hash", "")),
            "captured_child_post_mutation_policy_hash": _policy_hash(stage_record.get("child_post_mutation")),
            "event_child_final_policy_hash": str(event.get("child_final_policy_hash", "")),
            "captured_final_child_policy_hash": _policy_hash(stage_record.get("final_attached_child")),
        },
    }


def _stage_hashes_match(reconciliation: Mapping[str, Any]) -> bool:
    hashes = reconciliation.get("hashes") or {}
    pairs = [
        ("event_parent_policy_hash", "captured_parent_policy_hash"),
        ("event_child_pre_mutation_policy_hash", "captured_child_pre_mutation_policy_hash"),
        ("event_child_post_mutation_policy_hash", "captured_child_post_mutation_policy_hash"),
        ("event_child_final_policy_hash", "captured_final_child_policy_hash"),
    ]
    return bool(all(str(hashes.get(left, "")) and hashes.get(left) == hashes.get(right) for left, right in pairs))


def execute_lineage_producing_generation_resume(
    snapshot_path: str | Path,
    *,
    label: str = "",
    common_state_event_index: int | None = None,
    policy_weight_mutation_enable: bool | None = None,
    policy_weight_mutation_mode: str | None = None,
    policy_weight_mutation_rate: float | None = None,
    policy_weight_mutation_std: float | None = None,
) -> Dict[str, Any]:
    """Resume a pre-cue snapshot through generation end and capture real lineage.

    The resume uses the normal runner loop with skip_init=True, so cull/refill,
    child construction, policy mutation, bank sync, and env/population sync are
    the same production paths used by uninterrupted generation execution.
    """
    loaded = load_snapshot(snapshot_path)
    metadata = dict(loaded.get("metadata") or {})
    runner = loaded["runner"]
    _clear_checkpoint_hooks(runner)
    if policy_weight_mutation_enable is not None:
        runner.sim_cfg.policy_weight_mutation_enable = bool(policy_weight_mutation_enable)
    if policy_weight_mutation_mode is not None:
        runner.sim_cfg.policy_weight_mutation_mode = str(policy_weight_mutation_mode)
    if policy_weight_mutation_rate is not None:
        runner.sim_cfg.policy_weight_mutation_rate = float(policy_weight_mutation_rate)
    if policy_weight_mutation_std is not None:
        runner.sim_cfg.policy_weight_mutation_std = float(policy_weight_mutation_std)
    runner.sim_cfg.policy_lineage_telemetry_enable = True
    runner.sim_cfg.policy_lineage_telemetry_sample_limit = max(
        16,
        int(getattr(runner.sim_cfg, "policy_lineage_telemetry_sample_limit", 0) or 0),
    )
    setattr(runner.sim_cfg, "mem_max_25_stage_policy_capture_enable", True)
    runner.pop.sim_cfg = runner.sim_cfg
    setattr(runner.pop, "_policy_lineage_runtime_semantics", _lineage_runtime_semantics(runner))
    setattr(runner.pop, "_policy_lineage_probe_indices", {})
    before_digest = runner_state_digest(runner)
    before_ids = sorted(int(getattr(agent, "agent_id")) for agent in runner.pop.alive_agents())
    runner.run(start_gen=0, skip_init=True)
    after_digest = runner_state_digest(runner)
    after_ids = sorted(int(getattr(agent, "agent_id")) for agent in runner.pop.alive_agents())
    events = list(getattr(runner.pop, "_policy_lineage_events", []) or [])
    stage_records = list(getattr(runner.pop, "_policy_lineage_stage_records", []) or [])
    stage_by_event = {int(record.get("event_id", -1)): record for record in stage_records}
    selected_event_index = int(common_state_event_index) if common_state_event_index is not None else -1
    if events and selected_event_index < 0:
        for idx, candidate in enumerate(events):
            if int(candidate.get("parent_memory_dim", 0) or 0) > 0 or int(candidate.get("child_post_memory_dim", 0) or 0) > 0:
                selected_event_index = int(idx)
                break
        if selected_event_index < 0:
            selected_event_index = 0
    event = events[selected_event_index] if events and 0 <= selected_event_index < len(events) else {}
    stage_record = stage_by_event.get(int(event.get("event_id", -1)), {}) if event else {}
    reconciliation = _stage_event_summary(event, stage_record) if event else {
        "schema_version": "mem_max_25_policy_stage_reconciliation_v1",
        "event_id": -1,
        "stages_present": {},
        "hashes": {},
    }
    donor_id = int(event.get("policy_donor_parent_id", before_ids[0] if before_ids else -1)) if event else (before_ids[0] if before_ids else -1)
    parent_memory_dim = int(event.get("parent_memory_dim", 0) or 0) if event else 0
    child_memory_dim = int(event.get("child_post_memory_dim", 0) or 0) if event else 0
    common_state_pairs: List[Dict[str, Any]] = []
    if donor_id >= 0 and stage_record:
        stage_specs = [
            ("policy_donor_parent", stage_record.get("policy_donor_parent"), parent_memory_dim),
            ("child_pre_mutation", stage_record.get("child_pre_mutation"), child_memory_dim),
            ("child_post_mutation", stage_record.get("child_post_mutation"), child_memory_dim),
            ("final_attached_child", stage_record.get("final_attached_child"), child_memory_dim),
        ]
        for stage_label, policy, memory_dim in stage_specs:
            common_state_pairs.append(
                execute_policy_stage_common_state_pair(
                    snapshot_path,
                    stage_label=stage_label,
                    stage_policy=policy,
                    focal_agent_id=int(donor_id),
                    memory_dim=int(memory_dim),
                )
            )
    action_rows = []
    for pair in common_state_pairs:
        branches = pair.get("branches") or []
        for branch in branches:
            decision = branch.get("decision") or {}
            action_rows.append(
                {
                    "stage_label": pair.get("stage_label", ""),
                    "forced_correct_target": branch.get("forced_correct_target"),
                    "chosen_action": decision.get("chosen_action"),
                    "movement_mass": decision.get("movement_mass"),
                    "neutral_mass": decision.get("neutral_mass"),
                    "movement_vs_neutral_margin": decision.get("movement_vs_neutral_margin"),
                    "target_a_mass": decision.get("target_a_mass"),
                    "target_b_mass": decision.get("target_b_mass"),
                }
            )
    parity_identity = {
        "snapshot_state_sha256": str(metadata.get("state_sha256", "")),
        "resume_initial_state_sha256": str(before_digest.get("state_sha256", "")),
        "initial_matches_snapshot": bool(str(metadata.get("state_sha256", "")) == str(before_digest.get("state_sha256", ""))),
    }
    return {
        "schema_version": "mem_max_25_lineage_producing_generation_resume_v1",
        "label": str(label),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": _bytes_sha(Path(snapshot_path).read_bytes()),
        "generation_start": 0,
        "generation_end_reached": True,
        "production_resume_api": "runner.run(start_gen=0, skip_init=True)",
        "production_reproduction_path": "Population.reproduce_and_cull",
        "before_alive_agent_ids": before_ids,
        "after_alive_agent_ids": after_ids,
        "new_alive_agent_ids": sorted(set(after_ids) - set(before_ids)),
        "birth_event_count": int(len(events)),
        "selected_common_state_event_index": int(selected_event_index),
        "selected_common_state_event_id": int(event.get("event_id", -1)) if event else -1,
        "policy_stage_record_count": int(len(stage_records)),
        "lineage_events": events,
        "lineage_summary": summarize_lineage_events(events),
        "policy_stage_reconciliation": reconciliation,
        "policy_stage_hashes_match": _stage_hashes_match(reconciliation),
        "common_state_stage_pairs": common_state_pairs,
        "common_state_pair_count": int(len(common_state_pairs)),
        "common_state_all_available": bool(common_state_pairs and all(bool(pair.get("branches")) for pair in common_state_pairs)),
        "action_family_rows": action_rows,
        "identity": parity_identity,
        "before_state_sha256": before_digest.get("state_sha256", ""),
        "after_state_sha256": after_digest.get("state_sha256", ""),
    }


def execute_lineage_resume_repeat_parity(snapshot_path: str | Path) -> Dict[str, Any]:
    left = execute_lineage_producing_generation_resume(snapshot_path, label="repeat_left")
    right = execute_lineage_producing_generation_resume(snapshot_path, label="repeat_right")
    comparable_keys = [
        "birth_event_count",
        "policy_stage_record_count",
        "new_alive_agent_ids",
        "lineage_summary",
        "policy_stage_reconciliation",
        "policy_stage_hashes_match",
        "common_state_pair_count",
        "common_state_all_available",
        "action_family_rows",
        "after_state_sha256",
    ]
    return {
        "schema_version": "mem_max_25_lineage_resume_repeat_parity_v1",
        "passed": bool(all(left.get(key) == right.get(key) for key in comparable_keys)),
        "compared_keys": comparable_keys,
        "left": {key: left.get(key) for key in comparable_keys},
        "right": {key: right.get(key) for key in comparable_keys},
    }


def execute_paired_branch_replay(snapshot_path: str | Path, *, focal_agent_id: int | None = None) -> Dict[str, Any]:
    snapshot_path = Path(snapshot_path)
    loaded = load_snapshot(snapshot_path)
    metadata = dict(loaded.get("metadata") or {})
    snapshot_sha256 = str(metadata.get("snapshot_sha256", "") or "")
    if not snapshot_sha256:
        snapshot_sha256 = _bytes_sha(snapshot_path.read_bytes())
    results = []
    for target in (1, 2):
        branch_loaded = load_snapshot(snapshot_path)
        runner = branch_loaded["runner"]
        env = runner.env
        setattr(env, FORCED_TARGET_ATTR, int(target))
        abs_step = int(metadata["abs_step"])
        cue_steps = int(getattr(runner.env_cfg, "mem_max_delayed_choice_cue_steps", 1) or 1)
        hidden_steps = int(getattr(runner.env_cfg, "mem_max_delayed_choice_hidden_steps", 1) or 1)
        decision_abs_step = abs_step + cue_steps + hidden_steps
        ticks = list(range(abs_step, decision_abs_step + 1))
        last_actions: np.ndarray | None = None
        for tick in ticks:
            runner.env.patch_world_step(int(tick))
            alive = _alive_agents_for_branch(runner)
            last_actions = runner._batched_actions_strided(alive, buckets={})
        alive_ids = [int(getattr(agent, "agent_id")) for agent in _alive_agents_for_branch(runner)]
        focal = int(focal_agent_id) if focal_agent_id is not None else int(alive_ids[0])
        policy = getattr(runner, "_mem_max_live_level4_last_policy", {}) or {}
        probs = np.asarray((policy.get("probs_by_agent_id") or {}).get(focal, np.zeros((7,), dtype=np.float32)), dtype=np.float32)
        logits = np.asarray((policy.get("logits_by_agent_id") or {}).get(focal, np.zeros((7,), dtype=np.float32)), dtype=np.float32)
        pos = (int(runner.env.agent_pos_y[focal]), int(runner.env.agent_pos_x[focal]))
        target_a = tuple(int(v) for v in metadata.get("target_a", [0, 0]))
        target_b = tuple(int(v) for v in metadata.get("target_b", [0, 0]))
        masses = _action_masses(probs, pos, target_a, target_b)
        branch_digest = runner_state_digest(runner)
        results.append(
            {
                "branch": "cue_a_target_a" if target == 1 else "cue_b_target_b",
                "forced_correct_target": int(target),
                "decision_abs_step": int(decision_abs_step),
                "focal_agent_id": int(focal),
                "position": [int(pos[0]), int(pos[1])],
                "top_action": int(np.argmax(probs)) if probs.size else -1,
                "chosen_action": int(last_actions[alive_ids.index(focal)]) if last_actions is not None and focal in alive_ids else -1,
                "logits_sha256": _array_sha(logits),
                "probs": [float(v) for v in probs.tolist()],
                **masses,
                "branch_state_sha256": branch_digest.get("state_sha256", ""),
            }
        )
    a, b = results[0], results[1]
    paired = {
        "schema_version": "mem_max_pre_cue_paired_replay_v1",
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_sha256,
        "shared_state_sha256": metadata.get("state_sha256", ""),
        "branches": results,
        "hidden_window_executed_ticks": int(len(range(int(metadata["abs_step"]), int(metadata["abs_step"]) + int(getattr(loaded["runner"].env_cfg, "mem_max_delayed_choice_cue_steps", 1) or 1) + int(getattr(loaded["runner"].env_cfg, "mem_max_delayed_choice_hidden_steps", 1) or 1) + 1))),
        "logit_divergence": 0.0 if a["logits_sha256"] == b["logits_sha256"] else 1.0,
        "signed_target_mass_delta": float((a["target_a_mass"] - a["target_b_mass"]) + (b["target_b_mass"] - b["target_a_mass"])),
    }
    return paired


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
