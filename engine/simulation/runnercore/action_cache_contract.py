"""Schema-aware lookup for policy stride-cache entries."""

from __future__ import annotations

from typing import Any


def is_slot_keyed_policy_cache_entry(value: Any) -> bool:
    if isinstance(value, dict):
        return "agent_id" in value
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return False
    try:
        int(value[0])
        int(value[1])
    except (TypeError, ValueError):
        return False
    return hasattr(value[2], "__len__") and not isinstance(value[2], (str, bytes))


def resolve_policy_action_cache_entry(
    cache: Any,
    *,
    slot: int,
    agent_id: int,
) -> Any | None:
    if not isinstance(cache, dict):
        return None

    seen: set[int] = set()
    for key in (int(agent_id), int(slot)):
        if key in seen:
            continue
        seen.add(key)
        value = cache.get(key)
        if value is None:
            continue
        can_be_agent_keyed = key == int(agent_id)
        can_be_slot_keyed = key == int(slot)
        if isinstance(value, dict) and "agent_id" not in value:
            # The oldest resident-state schema used mappings keyed by slot.
            if can_be_slot_keyed:
                return value
            continue
        slot_keyed = is_slot_keyed_policy_cache_entry(value)
        if slot_keyed:
            try:
                cached_agent_id = int(
                    value.get("agent_id", -1) if isinstance(value, dict) else value[0]
                )
            except (TypeError, ValueError):
                continue
            if cached_agent_id == int(agent_id):
                return value
            continue
        if can_be_agent_keyed:
            return value
    return None
