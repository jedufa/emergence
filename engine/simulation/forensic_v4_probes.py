"""engine.simulation.forensic_v4_probes

Forensic V4 probes (A–C)
------------------------
This module provides lightweight, deterministic-safe instrumentation used during
Phase 3 resume determinism investigations.

Goals:
- A) Policy/bank digest probes at resume_apply_post + tick000.start
- B) Ordering proof: forward-call ordering + batch composition + action digest
- C) Missing state hunt scaffolding: agent attribute key signatures

Design constraints:
- MUST NOT draw RNG
- MUST NOT mutate simulation state (except writing debug artifacts)
- SHOULD avoid heavy overhead outside debug mode; all entrypoints are gated
  by env var EIEL_FORENSIC_V4=1.

All outputs are written under: <run_dir>/debug/
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _is_enabled() -> bool:
    """Return True if Forensic V4 probes are enabled."""
    return str(os.getenv("EIEL_FORENSIC_V4", "0")).strip() in ("1", "true", "TRUE", "yes", "YES")


def _ensure_debug_dir(run_dir: str) -> str:
    """Ensure <run_dir>/debug exists and return its path."""
    dbg = os.path.join(str(run_dir), "debug")
    os.makedirs(dbg, exist_ok=True)
    return dbg


def _sha256_bytes(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _write_jsonl(run_dir: str, filename: str, record: Dict[str, Any]) -> None:
    dbg = _ensure_debug_dir(run_dir)
    path = os.path.join(dbg, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        f.write("\n")


def _write_json(run_dir: str, filename: str, obj: Any) -> None:
    dbg = _ensure_debug_dir(run_dir)
    path = os.path.join(dbg, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def agent_attr_signature(
    agents: Sequence[Any],
    *,
    whitelist_prefixes: Tuple[str, ...] = ("_",),
    include_per_agent: bool = False,
    max_agents: int = 256,
) -> Dict[str, Any]:
    """Compute a deterministic signature of agent __dict__ keys.

    This is NOT a full deep state dump. It answers:
    - Do the two runs have the same attribute sets on Agent objects?
    - If not, what keys are unexpected / missing?

    We ignore callables and private fields by default (prefix '_'), because many are caches.
    You can widen/adjust the filter later, but for Phase 3 the goal is to catch
    missing public state or unexpected public state.
    """
    union: set[str] = set()
    per_agent: List[Dict[str, Any]] = []
    n = 0
    for a in agents:
        if n >= max_agents:
            break
        n += 1
        try:
            d = getattr(a, "__dict__", {}) or {}
        except Exception:
            d = {}
        keys = []
        for k in d.keys():
            if whitelist_prefixes and any(str(k).startswith(p) for p in whitelist_prefixes):
                continue
            keys.append(str(k))
        keys_sorted = sorted(set(keys))
        union.update(keys_sorted)
        if include_per_agent:
            per_agent.append({
                "agent_id": str(getattr(a, "agent_id", "?")),
                "keys": keys_sorted,
                "keys_sha256": _sha256_json(keys_sorted),
            })
    union_sorted = sorted(union)
    out: Dict[str, Any] = {
        "schema_version": "agent_attr_signature_v1",
        "agents_considered": n,
        "union_keys": union_sorted,
        "union_keys_sha256": _sha256_json(union_sorted),
    }
    if include_per_agent:
        out["per_agent"] = per_agent
    return out


def emit_agent_attr_signature(
    run_dir: str,
    *,
    tag: str,
    generation_index: Optional[int],
    tick: Optional[int],
    agents: Sequence[Any],
) -> None:
    """Write agent attribute signature record to debug/agent_attr_signature.jsonl."""
    if not _is_enabled():
        return
    sig = agent_attr_signature(agents, include_per_agent=False)
    rec = {
        "schema_version": "agent_attr_signature_record_v1",
        "tag": str(tag),
        "generation_index": generation_index,
        "tick": tick,
        "union_keys_sha256": sig["union_keys_sha256"],
        "agents_considered": sig["agents_considered"],
        # Keep union keys out of jsonl to avoid huge logs; store separately per tag.
    }
    _write_jsonl(run_dir, "agent_attr_signature.jsonl", rec)
    # Also write full union list for the latest tag (overwrites; small and intentional).
    _write_json(run_dir, "agent_attr_signature__latest.json", sig)


def emit_forward_order_probe(
    run_dir: str,
    *,
    tag: str,
    generation_index: int,
    tick: int,
    agent_ids: Sequence[int],
    batch_slices: Optional[List[Tuple[int, int]]] = None,
    actions_digest: Optional[str] = None,
) -> None:
    """Emit ordering proof record to debug/forward_order_probe.jsonl."""
    if not _is_enabled():
        return
    ids_list = [int(x) for x in agent_ids]
    ids_head = ids_list[:32]
    ids_tail = ids_list[-32:] if len(ids_list) > 32 else []
    rec = {
        "schema_version": "forward_order_probe_v1",
        "tag": str(tag),
        "generation_index": int(generation_index),
        "tick": int(tick),
        "n": int(len(ids_list)),
        "ids_sha256": _sha256_json(ids_list),
        "ids_head": ids_head,
        "ids_tail": ids_tail,
        "batch_slices": batch_slices or [],
        "actions_digest": actions_digest,
    }
    _write_jsonl(run_dir, "forward_order_probe.jsonl", rec)


def tensor_sha256_cpu_float32(t: Any) -> str:
    """Compute a stable sha256 over a torch tensor's CPU float32 bytes.

    - Synchronization is the caller's responsibility (CUDA async).
    - We do NOT import torch at module import time to avoid side effects.
    """
    import torch  # local import
    if t is None:
        return "none"
    if not isinstance(t, torch.Tensor):
        return "not_a_tensor"
    tt = t.detach()
    tt = tt.to("cpu").contiguous()
    # For hashing, cast to float32 when possible to reduce dtype drift.
    if tt.dtype != torch.float32:
        try:
            tt = tt.float()
        except Exception:
            pass
    b = tt.numpy().tobytes(order="C")
    return hashlib.sha256(b).hexdigest()


def emit_policy_bank_digest(
    run_dir: str,
    *,
    tag: str,
    generation_index: Optional[int],
    tick: Optional[int],
    bank_info: Dict[str, Any],
) -> None:
    """Write policy/bank digest record to debug/policy_bank_probe.jsonl."""
    if not _is_enabled():
        return
    rec = {
        "schema_version": "policy_bank_probe_v1",
        "tag": str(tag),
        "generation_index": generation_index,
        "tick": tick,
        **bank_info,
    }
    _write_jsonl(run_dir, "policy_bank_probe.jsonl", rec)


def emit_policy_bank_sync_digest(
    run_dir: str,
    *,
    tag: str,
    generation_index: Optional[int],
    tick: Optional[int],
    kind: str,
    stage: str,
    sync_info: Dict[str, Any],
) -> None:
    """Write policy/bank sync decision record to debug/policy_bank_sync_probe.jsonl.

    Purpose: isolate resume vs fresh differences inside bank sync (token gating, capacity growth, agent ordering).
    """
    if not _is_enabled():
        return
    rec = {
        "schema_version": "policy_bank_sync_probe_v1",
        "tag": str(tag),
        "generation_index": generation_index,
        "tick": tick,
        "kind": str(kind),
        "stage": str(stage),
        **sync_info,
    }
    _write_jsonl(run_dir, "policy_bank_sync_probe.jsonl", rec)
