from __future__ import annotations

"""Compact parent-child policy lineage telemetry.

Default-off diagnostic support only. The functions here compute hashes, tensor
delta norms, and deterministic cue-action probes without drawing RNG or touching
fitness/reward state.
"""

import dataclasses
import copy
import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np
import torch

from agents.agent_policy import MLPPolicy
from agents.gru_policy import GRUPolicy
from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_W
from evolution.cue_action_taxonomy import classify_cue_action, effective_runtime_cue_action_probe, level_counts


SCHEMA_VERSION = "eiel_policy_lineage_event_v1"


def stable_json_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def genome_payload(genome: object) -> dict[str, Any]:
    if dataclasses.is_dataclass(genome):
        return dataclasses.asdict(genome)
    return dict(getattr(genome, "__dict__", {}) or {})


def genome_hash(genome: object) -> str:
    return stable_json_hash(genome_payload(genome))


def policy_hash(policy: object) -> str:
    h = hashlib.sha256()
    if policy is None or not hasattr(policy, "state_dict"):
        return ""
    for key, value in sorted(policy.state_dict().items()):
        h.update(str(key).encode("utf-8"))
        arr = value.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def policy_type(policy: object) -> str:
    return type(policy).__name__ if policy is not None else "none"


def state_snapshot(policy: object) -> dict[str, torch.Tensor]:
    if policy is None or not hasattr(policy, "state_dict"):
        return {}
    return {str(k): v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def _tensor_delta_norm(before: dict[str, torch.Tensor], after_policy: object, *, selector: str) -> float:
    if after_policy is None or not hasattr(after_policy, "state_dict"):
        return 0.0
    total = 0.0
    for key, value in after_policy.state_dict().items():
        name = str(key)
        if selector == "recurrent" and not any(part in name.lower() for part in ("gru", "weight_hh", "weight_ih")):
            continue
        if selector == "action_head" and not any(part in name.lower() for part in ("head", "net.4")):
            continue
        prior = before.get(name)
        if prior is None:
            continue
        delta = value.detach().cpu().float() - prior.float()
        total += float(torch.sum(delta * delta).item())
    return float(total ** 0.5)


def tensor_delta_summary(before: dict[str, torch.Tensor], after_policy: object) -> dict[str, float]:
    return {
        "recurrent_tensor_delta_norm": _tensor_delta_norm(before, after_policy, selector="recurrent"),
        "action_head_tensor_delta_norm": _tensor_delta_norm(before, after_policy, selector="action_head"),
        "full_policy_tensor_delta_norm": _tensor_delta_norm(before, after_policy, selector="full"),
    }


def cue_action_probe(policy: object, *, obs_size: int) -> dict[str, Any]:
    if policy is None:
        return {"available": False}
    size = max(1, int(obs_size))
    obs_none = torch.zeros(size, dtype=torch.float32)
    obs_a = torch.zeros(size, dtype=torch.float32)
    obs_b = torch.zeros(size, dtype=torch.float32)
    obs_a[-2 if size >= 2 else 0] = 1.0
    obs_b[-1] = 1.0

    def logits(obs: torch.Tensor) -> tuple[torch.Tensor, float]:
        with torch.no_grad():
            if isinstance(policy, GRUPolicy):
                h0 = torch.zeros(int(getattr(policy, "hidden_dim", 64)), dtype=torch.float32)
                out, h = policy(obs, h0)
                return out.detach().cpu().float(), float(torch.norm(h.detach().cpu().float()).item())
            out = policy(obs)
            return out.detach().cpu().float(), 0.0

    la, ha = logits(obs_a)
    lb, hb = logits(obs_b)
    ln, _hn = logits(obs_none)
    pa = torch.softmax(la, dim=0)
    pb = torch.softmax(lb, dim=0)
    pn = torch.softmax(ln, dim=0)
    move_e = ACT_MOVE_E
    move_w = ACT_MOVE_W
    return {
        "available": True,
        "probe_kind": "raw_counterfactual_policy",
        "cue_input_delta": float(torch.norm(obs_a - obs_b).item()),
        "hidden_state_separation_after_cue": float(abs(ha - hb)),
        "hidden_state_separation_hidden_end_proxy": float(abs(ha - hb)),
        "action_logit_l2_divergence": float(torch.norm(la - lb).item()),
        "top1_cue_a": int(torch.argmax(la).item()),
        "top1_cue_b": int(torch.argmax(lb).item()),
        "top1_no_cue": int(torch.argmax(ln).item()),
        "target_a_direction_probability_proxy": float(pa[move_w].item()) if pa.numel() > move_w else 0.0,
        "target_b_direction_probability_proxy": float(pb[move_e].item()) if pb.numel() > move_e else 0.0,
        "cue_consistent_direction_score_proxy": float(
            (pa[move_w].item() if pa.numel() > move_w else 0.0)
            + (pb[move_e].item() if pb.numel() > move_e else 0.0)
            - (pn[move_w].item() if pn.numel() > move_w else 0.0)
            - (pn[move_e].item() if pn.numel() > move_e else 0.0)
        ),
        "cue_sensitive": bool(torch.norm(la - lb).item() > 1e-6 or int(torch.argmax(la).item()) != int(torch.argmax(lb).item())),
    }


def _lineage_id(agent: object) -> str:
    return str(getattr(getattr(agent, "stats", None), "lineage_id", "") or f"L{int(getattr(agent, 'agent_id', 0)):08d}")


def _founder_id(agent: object) -> int:
    return int(getattr(getattr(agent, "stats", None), "founder_id", getattr(agent, "agent_id", -1)) or -1)


def ensure_store(pop: object) -> list[dict[str, Any]]:
    events = getattr(pop, "_policy_lineage_events", None)
    if not isinstance(events, list):
        events = []
        setattr(pop, "_policy_lineage_events", events)
    return events


def _ensure_stage_store(pop: object) -> list[dict[str, Any]]:
    records = getattr(pop, "_policy_lineage_stage_records", None)
    if not isinstance(records, list):
        records = []
        setattr(pop, "_policy_lineage_stage_records", records)
    return records


def _clone_policy(policy: object) -> object | None:
    if policy is None:
        return None
    try:
        return copy.deepcopy(policy)
    except Exception:
        return None


def _clone_pre_mutation_policy(child_policy: object, snapshot: dict[str, torch.Tensor]) -> object | None:
    cloned = _clone_policy(child_policy)
    if cloned is None or not snapshot:
        return cloned
    try:
        cloned.load_state_dict({str(k): v.detach().clone() for k, v in snapshot.items()})
    except Exception:
        return None
    return cloned


def maybe_record_policy_lineage_event(
    pop: object,
    *,
    p1: object,
    p2: object,
    child: object,
    policy_donor: object | None,
    construction_kind: str,
    pre_mutation_hash: str,
    pre_mutation_snapshot: dict[str, torch.Tensor],
    pre_mutation_probe: dict[str, Any],
    mutation_summary: dict[str, Any],
) -> dict[str, Any] | None:
    sim_cfg = getattr(pop, "sim_cfg", None)
    if not bool(getattr(sim_cfg, "policy_lineage_telemetry_enable", False)):
        return None
    events = ensure_store(pop)
    limit = int(getattr(sim_cfg, "policy_lineage_telemetry_sample_limit", 64) or 64)
    event_id = int(getattr(pop, "_policy_lineage_event_counter", 0) or 0) + 1
    setattr(pop, "_policy_lineage_event_counter", event_id)

    donor = policy_donor if policy_donor is not None else p1
    capture_stage_policies = bool(getattr(sim_cfg, "mem_max_25_stage_policy_capture_enable", False))
    if capture_stage_policies:
        stage_records = _ensure_stage_store(pop)
        stage_records.append(
            {
                "event_id": int(event_id),
                "policy_donor_parent_id": int(getattr(donor, "agent_id", -1)),
                "child_agent_id": int(getattr(child, "agent_id", -1)),
                "policy_donor_parent": _clone_policy(getattr(donor, "policy", None)),
                "child_pre_mutation": _clone_pre_mutation_policy(getattr(child, "policy", None), pre_mutation_snapshot),
                "child_post_mutation": _clone_policy(getattr(child, "policy", None)),
                "final_attached_child": _clone_policy(getattr(child, "policy", None)),
            }
        )
    runtime_semantics = dict(getattr(pop, "_policy_lineage_runtime_semantics", {}) or {})
    probe_indices = dict(getattr(pop, "_policy_lineage_probe_indices", {}) or {})
    post_hash = policy_hash(getattr(child, "policy", None))
    delta = tensor_delta_summary(pre_mutation_snapshot, getattr(child, "policy", None))
    parent_probe = cue_action_probe(getattr(donor, "policy", None), obs_size=int(getattr(pop, "obs_size", 1)))
    child_probe = cue_action_probe(getattr(child, "policy", None), obs_size=int(getattr(pop, "obs_size", 1)))
    parent_effective_probe = effective_runtime_cue_action_probe(
        getattr(donor, "policy", None),
        obs_size=int(getattr(pop, "obs_size", 1)),
        cue_a_index=probe_indices.get("cue_a_index"),
        cue_b_index=probe_indices.get("cue_b_index"),
        memory_dim=int(getattr(getattr(donor, "genome", None), "memory_dim", 0) or 0),
        cue_enabled=bool(runtime_semantics.get("cue_enabled", False)),
        memory_ablated=bool(runtime_semantics.get("memory_ablated", False)),
        hidden_steps=int(runtime_semantics.get("hidden_steps", 2) or 2),
    )
    child_effective_probe = effective_runtime_cue_action_probe(
        getattr(child, "policy", None),
        obs_size=int(getattr(pop, "obs_size", 1)),
        cue_a_index=probe_indices.get("cue_a_index"),
        cue_b_index=probe_indices.get("cue_b_index"),
        memory_dim=int(getattr(getattr(child, "genome", None), "memory_dim", 0) or 0),
        cue_enabled=bool(runtime_semantics.get("cue_enabled", False)),
        memory_ablated=bool(runtime_semantics.get("memory_ablated", False)),
        hidden_steps=int(runtime_semantics.get("hidden_steps", 2) or 2),
    )
    sampled = len(events) < limit
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "generation": int(getattr(pop, "_current_generation", getattr(pop, "generation", 0)) or 0),
        "child_agent_id": int(getattr(child, "agent_id", -1)),
        "child_biological_lineage_id": _lineage_id(child),
        "founder_ancestry_id": int(_founder_id(child)),
        "biological_parent_ids": [int(getattr(p1, "agent_id", -1)), int(getattr(p2, "agent_id", -1))],
        "policy_donor_parent_id": int(getattr(donor, "agent_id", -1)),
        "policy_donor_lineage_id": _lineage_id(donor),
        "policy_construction_kind": str(construction_kind),
        "runtime_semantics": runtime_semantics,
        "parent_policy_type": policy_type(getattr(donor, "policy", None)),
        "child_policy_type": policy_type(getattr(child, "policy", None)),
        "parent_policy_hash": policy_hash(getattr(donor, "policy", None)),
        "child_pre_mutation_policy_hash": str(pre_mutation_hash),
        "child_post_mutation_policy_hash": post_hash,
        "child_final_policy_hash": post_hash,
        "parent_genome_hash": genome_hash(getattr(donor, "genome", None)),
        "child_genome_hash": genome_hash(getattr(child, "genome", None)),
        "parent_memory_dim": int(getattr(getattr(donor, "genome", None), "memory_dim", 0) or 0),
        "child_pre_memory_dim": int(getattr(getattr(child, "genome", None), "memory_dim", 0) or 0),
        "child_post_memory_dim": int(getattr(getattr(child, "genome", None), "memory_dim", 0) or 0),
        "mutation_enabled": bool(mutation_summary.get("enabled", False)),
        "mutation_mutated": bool(mutation_summary.get("mutated", False)),
        "policy_mutation_mode": str(mutation_summary.get("mode", "dense")),
        "policy_mutation_rate": float(mutation_summary.get("rate", 1.0) or 0.0),
        "policy_mutation_std": float(mutation_summary.get("std", 0.0) or 0.0),
        "mutation_parameter_count": int(mutation_summary.get("parameter_count", 0) or 0),
        "mutation_mutated_parameter_count": int(mutation_summary.get("mutated_parameter_count", 0) or 0),
        "mutation_trace_id": stable_json_hash(
            {
                "event_id": event_id,
                "child": int(getattr(child, "agent_id", -1)),
                "pre": str(pre_mutation_hash),
                "post": post_hash,
                "std": float(mutation_summary.get("std", 0.0) or 0.0),
            }
        ),
        "checkpoint_replay_sequence_id": f"policy_lineage_event_{event_id:08d}",
        "sampled_for_probe": bool(sampled),
        "parent_cue_probe": parent_probe if sampled else {"available": False, "sample_skipped": True},
        "child_pre_cue_probe": pre_mutation_probe if sampled else {"available": False, "sample_skipped": True},
        "child_post_cue_probe": child_probe if sampled else {"available": False, "sample_skipped": True},
        "final_child_cue_probe": child_probe if sampled else {"available": False, "sample_skipped": True},
        "parent_effective_cue_action_probe": parent_effective_probe if sampled else {"available": False, "sample_skipped": True},
        "child_post_effective_cue_action_probe": child_effective_probe if sampled else {"available": False, "sample_skipped": True},
        **delta,
    }
    events.append(record)
    return record


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    sampled = [event for event in events if bool(event.get("sampled_for_probe", False))]
    parent_sensitive = [
        event for event in sampled if bool((event.get("parent_cue_probe") or {}).get("cue_sensitive", False))
    ]
    child_sensitive = [
        event for event in sampled if bool((event.get("child_post_cue_probe") or {}).get("cue_sensitive", False))
    ]
    retained = [
        event
        for event in sampled
        if bool((event.get("parent_cue_probe") or {}).get("cue_sensitive", False))
        and bool((event.get("child_post_cue_probe") or {}).get("cue_sensitive", False))
    ]
    gained = [
        event
        for event in sampled
        if not bool((event.get("parent_cue_probe") or {}).get("cue_sensitive", False))
        and bool((event.get("child_post_cue_probe") or {}).get("cue_sensitive", False))
    ]
    lost = [
        event
        for event in sampled
        if bool((event.get("parent_cue_probe") or {}).get("cue_sensitive", False))
        and not bool((event.get("child_post_cue_probe") or {}).get("cue_sensitive", False))
    ]
    construction = Counter(str(event.get("policy_construction_kind", "")) for event in events)
    parent_levels = [
        classify_cue_action(
            raw_probe=event.get("parent_cue_probe") or {},
            effective_probe=event.get("parent_effective_cue_action_probe") or {},
            runtime_semantics=event.get("runtime_semantics") or {},
            policy_type=str(event.get("parent_policy_type", "")),
            memory_dim=int(event.get("parent_memory_dim", 0) or 0),
        )
        for event in sampled
    ]
    child_levels = [
        classify_cue_action(
            raw_probe=event.get("child_post_cue_probe") or {},
            effective_probe=event.get("child_post_effective_cue_action_probe") or {},
            runtime_semantics=event.get("runtime_semantics") or {},
            policy_type=str(event.get("child_policy_type", "")),
            memory_dim=int(event.get("child_post_memory_dim", 0) or 0),
        )
        for event in sampled
    ]
    return {
        "schema_version": "eiel_policy_lineage_event_summary_v1",
        "event_count": int(len(events)),
        "sampled_event_count": int(len(sampled)),
        "parent_cue_sensitive_count": int(len(parent_sensitive)),
        "child_post_cue_sensitive_count": int(len(child_sensitive)),
        "cue_sensitive_retained_count": int(len(retained)),
        "cue_sensitive_gained_count": int(len(gained)),
        "cue_sensitive_lost_count": int(len(lost)),
        "mutation_mutated_count": int(sum(1 for event in events if bool(event.get("mutation_mutated", False)))),
        "construction_kind_counts": dict(sorted(construction.items())),
        "parent_cue_action_level_counts": level_counts(parent_levels),
        "child_post_cue_action_level_counts": level_counts(child_levels),
        "complete_parent_child_map": bool(events and all(int(event.get("child_agent_id", -1)) >= 0 for event in events)),
    }
