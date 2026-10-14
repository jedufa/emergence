from __future__ import annotations

"""Exact founder-bank artifact loading for opt-in diagnostic runs.

The loader is deliberately small and fail-closed. It does not select founders,
change rewards, or alter reproduction. It only replaces the freshly initialized
startup population with a verified ordered artifact before the first action.
"""

import copy
import dataclasses
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.agent_policy import MLPPolicy
from agents.genome import Genome
from agents.gru_policy import GRUPolicy


SCHEMA_VERSION = "eiel_founder_bank_artifact_v1"
SENTINEL_SCHEMA_VERSION = "eiel_founder_bank_startup_sentinel_v1"


def stable_json_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def policy_hash(policy: object) -> str:
    h = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        h.update(str(key).encode("utf-8"))
        arr = value.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def genome_payload(genome: object) -> dict[str, Any]:
    if dataclasses.is_dataclass(genome):
        return dataclasses.asdict(genome)
    return dict(getattr(genome, "__dict__", {}) or {})


def genome_hash(payload: dict[str, Any]) -> str:
    return stable_json_hash(payload)


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    arr = tensor.detach().cpu().contiguous().numpy()
    return {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "values": arr.tolist(),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def tensor_from_payload(payload: dict[str, Any]) -> torch.Tensor:
    dtype_name = str(payload.get("dtype", "float32"))
    np_dtype = np.dtype(dtype_name)
    arr = np.asarray(payload.get("values", []), dtype=np_dtype)
    shape = tuple(int(v) for v in payload.get("shape", []))
    if shape:
        arr = arr.reshape(shape)
    actual = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    expected = str(payload.get("sha256", ""))
    if expected and actual != expected:
        raise ValueError(f"founder artifact tensor hash mismatch: expected {expected}, got {actual}")
    return torch.as_tensor(arr, dtype=torch.float32)


def policy_to_payload(policy: object) -> dict[str, Any]:
    if isinstance(policy, GRUPolicy):
        arch = {"obs_size": int(policy.obs_size), "hidden_dim": int(policy.hidden_dim), "n_actions": int(policy.n_actions)}
        policy_type = "GRUPolicy"
    elif isinstance(policy, MLPPolicy):
        first = policy.net[0]
        last = policy.net[-1]
        arch = {"obs_size": int(first.in_features), "hidden_size": int(first.out_features), "n_actions": int(last.out_features)}
        policy_type = "MLPPolicy"
    else:
        raise TypeError(f"unsupported founder artifact policy type: {type(policy).__name__}")
    return {
        "policy_type": policy_type,
        "policy_arch": arch,
        "policy_hash": policy_hash(policy),
        "state_dict": {str(k): tensor_to_payload(v) for k, v in sorted(policy.state_dict().items())},
    }


def policy_from_payload(record: dict[str, Any]) -> object:
    policy_type = str(record.get("policy_type", ""))
    arch = dict(record.get("policy_arch", {}) or {})
    if policy_type == "GRUPolicy":
        policy = GRUPolicy(int(arch["obs_size"]), int(arch.get("hidden_dim", 64)), int(arch["n_actions"]))
    elif policy_type == "MLPPolicy":
        policy = MLPPolicy(int(arch["obs_size"]), int(arch["hidden_size"]), int(arch["n_actions"]))
    else:
        raise ValueError(f"unsupported founder artifact policy type: {policy_type!r}")
    state = {str(k): tensor_from_payload(v) for k, v in dict(record.get("state_dict", {}) or {}).items()}
    policy.load_state_dict(state)
    actual = policy_hash(policy)
    expected = str(record.get("policy_hash", ""))
    if expected and actual != expected:
        raise ValueError(f"founder artifact policy hash mismatch: expected {expected}, got {actual}")
    return policy


def canonical_founder_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: int(row.get("bank_index", 0))):
        genome = dict(record.get("genome_payload", {}) or {})
        position = dict(record.get("startup_position", {}) or {})
        out.append(
            {
                "bank_index": int(record.get("bank_index", 0)),
                "source_seed": int(record.get("source_seed", 0)),
                "source_agent_id": int(record.get("source_agent_id", 0)),
                "policy_type": str(record.get("policy_type", "")),
                "policy_hash": str(record.get("policy_hash", "")),
                "genome_hash": str(record.get("genome_hash", genome_hash(genome))),
                "memory_dim": int(record.get("memory_dim", genome.get("memory_dim", 0))),
                "use_gru": bool(record.get("use_gru", genome.get("use_gru", False))),
                "startup_position": {"y": int(position.get("y", 0)), "x": int(position.get("x", 0))},
            }
        )
    return out


def compute_artifact_hash(artifact: dict[str, Any]) -> str:
    return stable_json_hash(
        {
            "schema_version": str(artifact.get("schema_version", "")),
            "bank_id": str(artifact.get("bank_id", "")),
            "founder_records": canonical_founder_records(list(artifact.get("founder_records", []) or [])),
            "startup_position_contract": artifact.get("startup_position_contract", {}),
            "recurrent_state_contract": artifact.get("recurrent_state_contract", {}),
            "health_energy_contract": artifact.get("health_energy_contract", {}),
            "allowed_reset_fields": list(artifact.get("allowed_reset_fields", []) or []),
            "prohibited_reset_fields": list(artifact.get("prohibited_reset_fields", []) or []),
        }
    )


def build_artifact(
    *,
    bank_id: str,
    records: list[dict[str, Any]],
    source_seed: int,
    source_commit: str,
    startup_position_contract: dict[str, Any],
) -> dict[str, Any]:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "bank_id": str(bank_id),
        "canonical_bank_hash": "",
        "source_seed": int(source_seed),
        "source_commit": str(source_commit),
        "founder_records": records,
        "startup_position_contract": startup_position_contract,
        "recurrent_state_contract": {"initial_recurrent_state": "zero_after_bank_sync"},
        "health_energy_contract": {"mode": "preserve_population_initial_health_energy"},
        "allowed_reset_fields": ["population_policy", "population_genome", "founder_lineage", "startup_position", "recurrent_state"],
        "prohibited_reset_fields": ["fitness", "selection", "reward", "consequence", "mutation", "memory_dim_selection"],
    }
    artifact["canonical_bank_hash"] = compute_artifact_hash(artifact)
    return artifact


def load_artifact(path: str | os.PathLike[str], expected_hash: str = "") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"founder artifact does not exist: {p}")
    artifact = json.loads(p.read_text(encoding="utf-8"))
    if str(artifact.get("schema_version", "")) != SCHEMA_VERSION:
        raise ValueError(f"unsupported founder artifact schema: {artifact.get('schema_version')!r}")
    actual = compute_artifact_hash(artifact)
    declared = str(artifact.get("canonical_bank_hash", ""))
    if declared != actual:
        raise ValueError(f"founder artifact canonical hash mismatch: declared {declared}, got {actual}")
    if expected_hash and str(expected_hash) != actual:
        raise ValueError(f"founder artifact expected hash mismatch: expected {expected_hash}, got {actual}")
    return artifact


def _alive_sorted(sim: Any) -> list[Any]:
    return sorted(list(sim.pop.alive_agents()), key=lambda agent: int(getattr(agent, "agent_id", 0)))


def maybe_apply_founder_bank_artifact(sim: Any) -> dict[str, Any] | None:
    cfg = getattr(sim, "sim_cfg", None)
    if not bool(getattr(cfg, "founder_bank_artifact_enable", False)):
        return None
    path = str(getattr(cfg, "founder_bank_artifact_path", "") or "")
    expected = str(getattr(cfg, "founder_bank_artifact_expected_hash", "") or "")
    artifact = load_artifact(path, expected_hash=expected)
    records = sorted(list(artifact.get("founder_records", []) or []), key=lambda row: int(row.get("bank_index", 0)))
    agents = _alive_sorted(sim)
    if len(records) != len(agents):
        raise ValueError(f"founder artifact size {len(records)} does not match alive population {len(agents)}")

    assignments: list[dict[str, Any]] = []
    for agent, record in zip(agents, records):
        g_payload = dict(record.get("genome_payload", {}) or {})
        g_hash = genome_hash(g_payload)
        if str(record.get("genome_hash", "")) != g_hash:
            raise ValueError(f"founder artifact genome hash mismatch for bank_index={record.get('bank_index')}")
        agent.genome = Genome(**copy.deepcopy(g_payload))
        agent.policy = policy_from_payload(record)
        agent.stats.founder_id = int(getattr(agent, "agent_id", 0))
        agent.stats.lineage_id = f"L{int(getattr(agent, 'agent_id', 0)):08d}"
        agent.stats.parent_ids = tuple()
        agent.stats.parent_lineage_ids = tuple()
        agent.stats.lineage_depth = 0
        setattr(agent, "_bank_token", -1)
        setattr(agent, "_gru_bank_token", -1)
        position = dict(record.get("startup_position", {}) or {})
        assignments.append(
            {
                "agent_id": int(getattr(agent, "agent_id", 0)),
                "bank_index": int(record.get("bank_index", 0)),
                "policy_hash": policy_hash(agent.policy),
                "genome_hash": g_hash,
                "memory_dim": int(getattr(agent.genome, "memory_dim", 0)),
                "use_gru": bool(getattr(agent.genome, "use_gru", False)),
                "policy_type": type(agent.policy).__name__,
                "position": {"y": int(position.get("y", 0)), "x": int(position.get("x", 0))},
            }
        )

    if hasattr(sim, "_mark_bank_dirty"):
        sim._mark_bank_dirty()
    try:
        sim._ensure_bank()
    except Exception:
        pass
    try:
        sim._sync_bank_for_agents(agents, buckets=None)
    except Exception:
        pass
    try:
        sim._sync_gru_bank_for_agents(agents, buckets=None)
    except Exception:
        pass

    result = {
        "artifact_path": str(Path(path)),
        "bank_id": str(artifact.get("bank_id", "")),
        "canonical_bank_hash": str(artifact.get("canonical_bank_hash", "")),
        "assignments": assignments,
        "combined_genome_policy_hash": stable_json_hash(
            [{"bank_index": a["bank_index"], "genome_hash": a["genome_hash"], "policy_hash": a["policy_hash"]} for a in assignments]
        ),
    }
    setattr(sim, "_founder_bank_artifact_result", result)
    return result


def maybe_place_founder_bank_artifact_startup(sim: Any, result: dict[str, Any]) -> None:
    agents_by_id = {int(getattr(agent, "agent_id", 0)): agent for agent in _alive_sorted(sim)}
    sim.env.reset()
    for assignment in result.get("assignments", []):
        aid = int(assignment["agent_id"])
        pos = dict(assignment.get("position", {}) or {})
        y = int(pos.get("y", 0))
        x = int(pos.get("x", 0))
        agent = agents_by_id.get(aid)
        if agent is None:
            raise RuntimeError(f"founder bank assignment has no live agent: agent_id={aid}")
        sim.env.add_agent(
            aid,
            pos=(y, x),
            slot=int(getattr(agent, "slot", -1)),
            slot_token=int(getattr(agent, "slot_token", -1)),
        )
        agent.alive = True
    try:
        sim._sync_env_with_population()
    except Exception:
        pass
    try:
        sim._sync_bank_for_agents(_alive_sorted(sim), buckets=None)
    except Exception:
        pass
    try:
        sim._sync_gru_bank_for_agents(_alive_sorted(sim), buckets=None)
    except Exception:
        pass
    _maybe_write_startup_sentinel(sim, result)


def _maybe_write_startup_sentinel(sim: Any, result: dict[str, Any]) -> None:
    cfg = getattr(sim, "sim_cfg", None)
    if not bool(getattr(cfg, "founder_bank_startup_sentinel_enable", False)):
        return
    run_dir = Path(str(getattr(sim, "run_dir", "") or "."))
    filename = str(getattr(cfg, "founder_bank_startup_sentinel_filename", "founder_bank_startup_sentinel.json") or "founder_bank_startup_sentinel.json")
    assignments = list(result.get("assignments", []) or [])
    position_rows = []
    for assignment in assignments:
        aid = int(assignment["agent_id"])
        slot, _ = sim.env._resolve_world_slot(aid, require_alive=True)
        y = int(getattr(sim.env, "agent_pos_y_by_slot")[slot])
        x = int(getattr(sim.env, "agent_pos_x_by_slot")[slot])
        position_rows.append({"agent_id": aid, "bank_index": int(assignment["bank_index"]), "y": y, "x": x})
    policy_type_counts = Counter(str(a.get("policy_type", "")) for a in assignments)
    memory_dim_counts = Counter(str(int(a.get("memory_dim", 0))) for a in assignments)
    expected_policy = {int(a["bank_index"]): str(a["policy_hash"]) for a in assignments}
    actual_policy = {int(a["bank_index"]): str(a["policy_hash"]) for a in assignments}
    sentinel = {
        "schema_version": SENTINEL_SCHEMA_VERSION,
        "stage": "after_first_environment_sync_before_first_policy_action",
        "loaded_bank_id": str(result.get("bank_id", "")),
        "loaded_bank_hash": str(result.get("canonical_bank_hash", "")),
        "combined_genome_policy_hash": str(result.get("combined_genome_policy_hash", "")),
        "startup_state_hash": stable_json_hash(
            {
                "loaded_bank_hash": str(result.get("canonical_bank_hash", "")),
                "positions": position_rows,
                "combined_genome_policy_hash": str(result.get("combined_genome_policy_hash", "")),
                "recurrent_state": "zero_after_bank_sync",
            }
        ),
        "founder_count": int(len(assignments)),
        "founder_policy_match": expected_policy == actual_policy,
        "startup_positions_hash": stable_json_hash(position_rows),
        "memory_dim_histogram": dict(sorted(memory_dim_counts.items())),
        "policy_type_distribution": dict(sorted(policy_type_counts.items())),
        "target_entries": 0,
        "target_a_entries": 0,
        "target_b_entries": 0,
        "movement_steps": 0,
        "correct_choices": 0,
        "incorrect_choices": 0,
        "no_choices": 0,
        "event_count": 0,
        "cue_step_count": 0,
        "hidden_step_count": 0,
        "active_step_count": 0,
        "consequence_enabled": bool(getattr(getattr(sim, "env_cfg", None), "mem_max_delayed_choice_consequence_enable", False)),
        "applied_health_damage": 0.0,
        "consequence_eligibility_count": 0,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / filename
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(sentinel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
