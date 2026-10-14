from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from . import SCHEMA_VERSION
from .schema import validate_playback


TargetFn = Callable[[int, str, str], int]


@dataclass
class _Agent:
    agent_id: str
    lineage_id: str
    branch_id: str
    birth_tick: int
    death_tick: int | None
    parent_agent_ids: list[str]
    generation_born: int | None = None
    generation_died: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "lineage_id": self.lineage_id,
            "branch_id": self.branch_id,
            "birth_tick": self.birth_tick,
            "death_tick": self.death_tick,
            "parent_agent_ids": self.parent_agent_ids,
            "generation_born": self.generation_born,
            "generation_died": self.generation_died,
        }


def _ramp(tick: int, start: int, end: int, low: float, high: float) -> float:
    if tick <= start:
        return low
    if tick >= end:
        return high
    u = (tick - start) / (end - start)
    u = u * u * (3 - 2 * u)
    return low + (high - low) * u


def _build_dataset(
    dataset_id: str,
    seed: int,
    last_tick: int,
    lineages: list[str],
    branch_defs: dict[str, list[tuple[str, int, str | None, str]]],
    target: TargetFn,
    notes: str,
    profiles: dict[str, dict[str, float | int | str]] | None = None,
) -> dict[str, object]:
    rng = random.Random(seed)
    lineage_records = [
        {
            "lineage_id": lineage_id,
            "parent_lineage_id": None,
            "founder_lineage_id": lineage_id,
            "creation_tick": 0,
            "extinction_tick": None,
            "display_label": lineage_id.replace("lineage_", "Lineage "),
        }
        for lineage_id in lineages
    ]
    branch_records = []
    branch_to_lineage = {}
    for lineage_id, defs in branch_defs.items():
        for branch_id, creation_tick, parent_branch_id, label in defs:
            branch_records.append(
                {
                    "branch_id": branch_id,
                    "lineage_id": lineage_id,
                    "parent_branch_id": parent_branch_id,
                    "creation_tick": creation_tick,
                    "extinction_tick": None,
                    "display_label": label,
                }
            )
            branch_to_lineage[branch_id] = lineage_id

    agents: list[_Agent] = []
    alive_by_branch: dict[str, list[str]] = {branch["branch_id"]: [] for branch in branch_records}
    agent_by_id: dict[str, _Agent] = {}
    next_agent = 1

    for tick in range(0, last_tick + 1):
        for branch in branch_records:
            branch_id = branch["branch_id"]
            if tick < branch["creation_tick"]:
                continue
            lineage_id = branch_to_lineage[branch_id]
            desired = target(tick, lineage_id, branch_id)
            alive = [agent_id for agent_id in alive_by_branch[branch_id] if agent_by_id[agent_id].death_tick is None]
            alive_by_branch[branch_id] = alive
            profile = (profiles or {}).get(branch_id, {})
            birth_rate = float(profile.get("birth_rate", 0.24))
            death_rate = float(profile.get("death_rate", 0.28))
            variation = float(profile.get("bounded_stochastic_variation", 0.025))
            wobble = 1.0 + (rng.random() * 2 - 1) * variation
            if len(alive) > desired:
                max_deaths = max(1, math.ceil(len(alive) * death_rate * wobble))
                deaths = min(len(alive) - desired, max_deaths)
                rng.shuffle(alive)
                for agent_id in alive[:deaths]:
                    agent_by_id[agent_id].death_tick = tick
                    agent_by_id[agent_id].generation_died = tick // 20
                alive_by_branch[branch_id] = alive[deaths:]
            elif len(alive) < desired:
                max_births = max(1, math.ceil(max(1, desired) * birth_rate * wobble))
                births = min(desired - len(alive), max_births)
                lineage_alive = [
                    agent_id
                    for ids in alive_by_branch.values()
                    for agent_id in ids
                    if agent_by_id[agent_id].lineage_id == lineage_id
                    and agent_by_id[agent_id].death_tick is None
                ]
                for _ in range(births):
                    parent_pool = alive_by_branch[branch_id] or lineage_alive
                    parents = [rng.choice(parent_pool)] if parent_pool else []
                    if len(parent_pool) > 1 and rng.random() < 0.42:
                        mate = rng.choice(parent_pool)
                        if mate not in parents:
                            parents.append(mate)
                    agent_id = f"{dataset_id}_a{next_agent:06d}"
                    next_agent += 1
                    agent = _Agent(
                        agent_id=agent_id,
                        lineage_id=lineage_id,
                        branch_id=branch_id,
                        birth_tick=tick,
                        death_tick=None,
                        parent_agent_ids=parents,
                        generation_born=tick // 20,
                    )
                    agents.append(agent)
                    agent_by_id[agent_id] = agent
                    alive_by_branch[branch_id].append(agent_id)

    for lineage in lineage_records:
        lineage_id = lineage["lineage_id"]
        alive_ticks = [
            agent.death_tick if agent.death_tick is not None else last_tick + 1
            for agent in agents
            if agent.lineage_id == lineage_id
        ]
        if alive_ticks and max(alive_ticks) <= last_tick:
            lineage["extinction_tick"] = max(alive_ticks)
    for branch in branch_records:
        branch_id = branch["branch_id"]
        alive_ticks = [
            agent.death_tick if agent.death_tick is not None else last_tick + 1
            for agent in agents
            if agent.branch_id == branch_id
        ]
        if alive_ticks and max(alive_ticks) <= last_tick:
            branch["extinction_tick"] = max(alive_ticks)

    data = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "source_kind": "synthetic",
            "source_path": None,
            "provenance_classification": "deterministic synthetic V0 fixture",
            "visualization_seed": seed,
            "time_unit": "abstract_tick",
            "first_tick": 0,
            "last_tick": last_tick,
            "generation_mapping": {"ticks_per_generation": 20},
            "semantics_mode": "actual_agents",
            "parent_reference_policy": "strict",
            "notes": notes,
            "synthetic_profile_label": "synthetic demographic dynamics; not scientific fitness",
            "synthetic_demographic_profiles": profiles or {},
        },
        "lineages": lineage_records,
        "branches": branch_records,
        "agents": [agent.to_json() for agent in agents],
    }
    validate_playback(data)
    return data


def balanced(seed: int = 1701) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 9)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 45 + i * 4, f"{lineage}_root", "soft branch"),
        ]
        for i, lineage in enumerate(lineages)
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        phase = idx * 0.71
        if branch.endswith("_root"):
            return max(18, round(48 + 12 * math.sin(tick / 32 + phase)))
        return round(_ramp(tick, 40 + idx * 4, 140 + idx * 3, 0, 36 + (idx % 3) * 9))

    profiles = {
        branch_id: {
            "birth_rate": 0.18 if branch_id.endswith("_root") else 0.12,
            "death_rate": 0.20,
            "growth_delay": 45 if branch_id.endswith("_branch_a") else 0,
            "decline_onset": 180,
            "branch_modifier": 1.0,
            "bounded_stochastic_variation": 0.02,
        }
        for defs in branches.values()
        for branch_id, *_ in defs
    }
    return _build_dataset("balanced_coexistence", seed, 220, lineages, branches, target, "Eight founder canopies coexist with modest branch lobes.", profiles)


def dominant_sweep(seed: int = 2701) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 8)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 70, f"{lineage}_root", "descendant lobe"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        if idx == 1:
            base = round(_ramp(tick, 35, 220, 56, 520))
            return round(base * (0.72 if branch.endswith("_root") else 0.28))
        end = 0 if idx in {5, 7} else 24
        base = round(_ramp(tick, 30, 220, 86 - idx * 4, end))
        return round(base * (0.82 if branch.endswith("_root") else 0.18))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.20 if idx == 1 else 0.11,
                "death_rate": 0.10 if idx == 1 else 0.26,
                "growth_delay": 35 if idx == 1 else 0,
                "decline_onset": 80 if idx != 1 else 230,
                "branch_modifier": 1.14 if branch_id.endswith("_branch_a") and idx == 1 else 0.92,
                "bounded_stochastic_variation": 0.025,
            }
    return _build_dataset("dominant_sweep", seed, 260, lineages, branches, target, "One founder canopy grows gradually while several lineages contract and two go extinct.", profiles)


def branch_split(seed: int = 3701) -> dict[str, object]:
    lineages = ["lineage_1", "lineage_2", "lineage_3"]
    branches = {
        "lineage_1": [
            ("lineage_1_root", 0, None, "root"),
            ("lineage_1_branch_a", 45, "lineage_1_root", "branch A"),
            ("lineage_1_branch_b", 65, "lineage_1_root", "branch B"),
        ],
        "lineage_2": [("lineage_2_root", 0, None, "neighbor")],
        "lineage_3": [("lineage_3_root", 0, None, "neighbor")],
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        if branch == "lineage_1_root":
            return round(_ramp(tick, 0, 130, 90, 50))
        if branch == "lineage_1_branch_a":
            return round(_ramp(tick, 45, 150, 0, 170))
        if branch == "lineage_1_branch_b":
            return round(_ramp(tick, 65, 170, 0, 230))
        return round(82 + 9 * math.sin(tick / 42))

    profiles = {
        "lineage_1_root": {"birth_rate": 0.12, "death_rate": 0.18, "growth_delay": 0, "decline_onset": 120, "branch_modifier": 0.78, "bounded_stochastic_variation": 0.02},
        "lineage_1_branch_a": {"birth_rate": 0.17, "death_rate": 0.12, "growth_delay": 45, "decline_onset": 190, "branch_modifier": 1.0, "bounded_stochastic_variation": 0.02},
        "lineage_1_branch_b": {"birth_rate": 0.21, "death_rate": 0.09, "growth_delay": 65, "decline_onset": 220, "branch_modifier": 1.28, "bounded_stochastic_variation": 0.02},
        "lineage_2_root": {"birth_rate": 0.12, "death_rate": 0.12, "growth_delay": 0, "decline_onset": 210, "branch_modifier": 1.0, "bounded_stochastic_variation": 0.02},
        "lineage_3_root": {"birth_rate": 0.12, "death_rate": 0.12, "growth_delay": 0, "decline_onset": 210, "branch_modifier": 1.0, "bounded_stochastic_variation": 0.02},
    }
    return _build_dataset("branch_split", seed, 230, lineages, branches, target, "A major lineage splits into two surviving branch lobes, then branch B becomes stronger.", profiles)


def weak_extinction(seed: int = 3901) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 7)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 60, f"{lineage}_root", "secondary lobe"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        if idx == 6:
            base = round(_ramp(tick, 20, 185, 78, 0))
            return round(base * (0.85 if branch.endswith("_root") else 0.15))
        base = round(_ramp(tick, 40, 210, 58 + idx * 4, 92 + idx * 9))
        return round(base * (0.82 if branch.endswith("_root") else 0.18))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.10 if idx == 6 else 0.16,
                "death_rate": 0.38 if idx == 6 else 0.12,
                "growth_delay": 60 if branch_id.endswith("_branch_a") else 0,
                "decline_onset": 60 if idx == 6 else 240,
                "branch_modifier": 0.62 if idx == 6 else 1.0,
                "bounded_stochastic_variation": 0.025,
            }
    return _build_dataset("weak_lineage_extinction", seed, 220, lineages, branches, target, "One weak lineage declines to extinction while neighboring canopies continue expanding.", profiles)


def late_bloomer(seed: int = 4101) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 6)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 90, f"{lineage}_root", "late lobe"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        if idx == 4:
            base = round(_ramp(tick, 120, 240, 12, 330))
            return round(base * (0.55 if branch.endswith("_root") else 0.45))
        decline = _ramp(tick, 135, 260, 86 + idx * 5, 44 + idx * 3)
        return round(decline * (0.86 if branch.endswith("_root") else 0.14))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.24 if idx == 4 else 0.12,
                "death_rate": 0.08 if idx == 4 else 0.20,
                "growth_delay": 120 if idx == 4 else 0,
                "decline_onset": 999 if idx == 4 else 140,
                "branch_modifier": 1.18 if idx == 4 and branch_id.endswith("_branch_a") else 0.94,
                "bounded_stochastic_variation": 0.025,
            }
    return _build_dataset("late_bloomer", seed, 270, lineages, branches, target, "A small lineage remains quiet until a late bloom makes it visually prominent.", profiles)


def boom_collapse(seed: int = 4701) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 6)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 55, f"{lineage}_root", "branch"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        if idx == 2:
            up = _ramp(tick, 20, 110, 44, 470)
            down = _ramp(tick, 120, 235, up, 0)
            base = round(down)
            return round(base * (0.62 if branch.endswith("_root") else 0.38))
        base = round(_ramp(tick, 130, 230, 58 + idx * 5, 98 + idx * 7))
        return round(base * (0.86 if branch.endswith("_root") else 0.14))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.24 if idx == 2 else 0.13,
                "death_rate": 0.34 if idx == 2 else 0.11,
                "growth_delay": 20 if idx == 2 else 120,
                "decline_onset": 125 if idx == 2 else 220,
                "branch_modifier": 1.12 if branch_id.endswith("_branch_a") else 1.0,
                "bounded_stochastic_variation": 0.03,
            }
    return _build_dataset("boom_collapse", seed, 250, lineages, branches, target, "One canopy booms and then collapses while neighbors reclaim visual space.", profiles)


def dominance_reversal(seed: int = 5101) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 7)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 50, f"{lineage}_root", "branch"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        if idx == 2:
            base = round(_ramp(tick, 0, 100, 150, 360))
            base = round(_ramp(tick, 110, 255, base, 58))
        elif idx == 5:
            base = round(_ramp(tick, 85, 260, 32, 410))
        else:
            base = round(72 + idx * 8 + 10 * math.sin(tick / 36 + idx))
        return round(base * (0.72 if branch.endswith("_root") else 0.28))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.23 if idx == 5 else 0.15,
                "death_rate": 0.30 if idx == 2 else 0.11,
                "growth_delay": 85 if idx == 5 else 0,
                "decline_onset": 120 if idx == 2 else 999,
                "branch_modifier": 1.12 if idx == 5 else 1.0,
                "bounded_stochastic_variation": 0.022,
            }
    return _build_dataset("dominance_reversal", seed, 280, lineages, branches, target, "The early dominant lineage contracts while a different lineage overtakes it.", profiles)


def cyclic_fortunes(seed: int = 5301) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 5)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 70, f"{lineage}_root", "branch"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        phase = idx * math.pi * 0.5
        wave = 0.5 + 0.5 * math.sin(tick / 38 + phase)
        envelope = _ramp(tick, 0, 80, 0.55, 1.0)
        total = round((54 + wave * 158) * envelope)
        return round(total * (0.74 if branch.endswith("_root") else 0.26))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.19,
                "death_rate": 0.20,
                "growth_delay": 70 if branch_id.endswith("_branch_a") else 0,
                "decline_onset": 0,
                "branch_modifier": 1.0 + (idx - 2.5) * 0.03,
                "bounded_stochastic_variation": 0.028,
            }
    return _build_dataset("cyclic_lineage_fortunes", seed, 300, lineages, branches, target, "Lineages take turns expanding and contracting in bounded cyclic demographic waves.", profiles)


def founder_bottleneck(seed: int = 5501) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 5)]
    branches = {
        "lineage_1": [
            ("lineage_1_root", 0, None, "founder root"),
            ("lineage_1_branch_a", 115, "lineage_1_root", "survivor branch"),
            ("lineage_1_branch_b", 150, "lineage_1_root", "recovery branch"),
        ],
        "lineage_2": [("lineage_2_root", 0, None, "neighbor")],
        "lineage_3": [("lineage_3_root", 0, None, "neighbor")],
        "lineage_4": [("lineage_4_root", 0, None, "neighbor")],
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        if lineage == "lineage_1":
            if branch == "lineage_1_root":
                down = _ramp(tick, 30, 105, 210, 18)
                return round(_ramp(tick, 150, 240, down, 46))
            if branch == "lineage_1_branch_a":
                return round(_ramp(tick, 115, 230, 0, 150))
            return round(_ramp(tick, 150, 260, 0, 190))
        idx = int(lineage.rsplit("_", 1)[1])
        return round(_ramp(tick, 60, 210, 72 + idx * 6, 105 + idx * 9))

    profiles = {
        "lineage_1_root": {"birth_rate": 0.08, "death_rate": 0.34, "growth_delay": 0, "decline_onset": 35, "branch_modifier": 0.7, "bounded_stochastic_variation": 0.022},
        "lineage_1_branch_a": {"birth_rate": 0.21, "death_rate": 0.08, "growth_delay": 115, "decline_onset": 999, "branch_modifier": 1.1, "bounded_stochastic_variation": 0.022},
        "lineage_1_branch_b": {"birth_rate": 0.24, "death_rate": 0.08, "growth_delay": 150, "decline_onset": 999, "branch_modifier": 1.18, "bounded_stochastic_variation": 0.022},
    }
    for lineage in ["lineage_2", "lineage_3", "lineage_4"]:
      profiles[f"{lineage}_root"] = {"birth_rate": 0.14, "death_rate": 0.11, "growth_delay": 0, "decline_onset": 260, "branch_modifier": 1.0, "bounded_stochastic_variation": 0.02}
    return _build_dataset("founder_bottleneck", seed, 275, lineages, branches, target, "A founder lineage narrows to a bottleneck, then survivor branches regrow into the field.", profiles)


def sibling_branch_competition(seed: int = 5703) -> dict[str, object]:
    data = branch_split(seed)
    data["metadata"]["dataset_id"] = "sibling_branch_competition"
    data["metadata"]["notes"] = "Sibling branches from one lineage diverge, then one descendant branch becomes visibly stronger than its sibling."
    validate_playback(data)
    return data


def scale_stress(seed: int = 5701) -> dict[str, object]:
    lineages = [f"lineage_{i}" for i in range(1, 13)]
    branches = {
        lineage: [
            (f"{lineage}_root", 0, None, "root"),
            (f"{lineage}_branch_a", 35, f"{lineage}_root", "branch A"),
            (f"{lineage}_branch_b", 70, f"{lineage}_root", "branch B"),
        ]
        for lineage in lineages
    }

    def target(tick: int, lineage: str, branch: str) -> int:
        idx = int(lineage.rsplit("_", 1)[1])
        peak_total = 342 + (idx % 4) * 37
        if idx in {3, 8}:
            peak_total += 180
        growth = _ramp(tick, 0, 135, 55, peak_total)
        churn = 1.0 + 0.08 * math.sin(tick / 24 + idx)
        total = round(growth * churn)
        if branch.endswith("_root"):
            return round(total * 0.48)
        if branch.endswith("_branch_a"):
            return round(total * 0.29)
        return max(0, round(total * 0.23))

    profiles = {}
    for lineage, defs in branches.items():
        idx = int(lineage.rsplit("_", 1)[1])
        for branch_id, *_ in defs:
            profiles[branch_id] = {
                "birth_rate": 0.22 if idx in {3, 8} else 0.18,
                "death_rate": 0.09,
                "growth_delay": 35 if branch_id.endswith("_branch_a") else 70 if branch_id.endswith("_branch_b") else 0,
                "decline_onset": 999,
                "branch_modifier": 1.18 if branch_id.endswith("_branch_b") and idx in {3, 8} else 1.0,
                "bounded_stochastic_variation": 0.018,
            }
    return _build_dataset("scale_stress_5000", seed, 190, lineages, branches, target, "Stress fixture reaches roughly 5,000 simultaneous living particles.", profiles)


SCENARIOS = {
    "balanced": balanced,
    "dominant": dominant_sweep,
    "weak_extinction": weak_extinction,
    "late_bloomer": late_bloomer,
    "branch_split": branch_split,
    "sibling_competition": sibling_branch_competition,
    "boom_collapse": boom_collapse,
    "dominance_reversal": dominance_reversal,
    "cyclic_fortunes": cyclic_fortunes,
    "founder_bottleneck": founder_bottleneck,
    "stress": scale_stress,
}
