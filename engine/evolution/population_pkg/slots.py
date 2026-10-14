from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

@dataclass
class SlotPool:
    max_slots: int
    free_slots: Deque[int]
    slot_tokens: List[int]

    @classmethod
    def build(cls, base_population: int, slot_factor: int, max_slots_cap: int):
        base = int(base_population)
        factor = int(slot_factor)
        if factor < 2:
            factor = 2
        cap = int(max_slots_cap or 0)
        if cap > 0:
            if cap < base:
                raise ValueError("max_slots_cap must be >= base_population")
            max_slots = min(base * factor, cap)
        else:
            max_slots = max(base * factor, base * 2)

        free = deque(range(max_slots))
        tokens = [0 for _ in range(max_slots)]
        return cls(max_slots=max_slots, free_slots=free, slot_tokens=tokens)

    def reset(self):
        self.free_slots.clear()
        self.free_slots.extend(range(self.max_slots))
        for i in range(self.max_slots):
            self.slot_tokens[i] = 0

    def alloc(self) -> Optional[Tuple[int, int]]:
        if not self.free_slots:
            return None
        slot = int(self.free_slots.popleft())
        self.slot_tokens[slot] += 1
        return slot, int(self.slot_tokens[slot])

    def free(self, slot: int) -> None:
        if slot < 0 or slot >= self.max_slots:
            return
        self.free_slots.append(int(slot))
