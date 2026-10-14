from __future__ import annotations

"""Reporting and founder hooks for the live RECOMB-72 SimulationRunner lane.

The inheritance mechanics themselves stay at the production boundary in
``evolution.reproduction.crossover_genome``.  This module only prepares the
neutral founders and observes the real population after each generation.
"""

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolution.recomb72_inheritance import (
    AUDITED_LOCI,
    LOCUS_BOUNDS,
    MUTATION_OFF_CONTRACT,
    NEUTRAL_PACKAGE,
    founder_packages,
    mode_sha256,
    registry_sha256,
)
from evolution.recomb72_metrics import (
    GENERATION_METRICS_SCHEMA_VERSION,
    ancestry_concentration,
    locus_inheritance_statistics,
    locus_statistics,
    package_statistics,
    reproduction_concentration,
)


LIVE_FIXTURE_SCHEMA_VERSION = "recomb72_live_fixture_v2"
LIVE_METRICS_SCHEMA_VERSION = GENERATION_METRICS_SCHEMA_VERSION


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _enabled(runner: Any) -> bool:
    return bool(getattr(getattr(runner, "sim_cfg", None), "recomb72_audit_enable", False))


def _agent_package(agent: Any) -> dict[str, float]:
    genome = getattr(agent, "genome", None)
    if genome is None:
        raise RuntimeError("RECOMB72_LIVE_AGENT_GENOME_MISSING")
    package: dict[str, float] = {}
    for locus in AUDITED_LOCI:
        value = float(getattr(genome, locus))
        lower, upper = LOCUS_BOUNDS[locus]
        if value < lower - 1.0e-6 or value > upper + 1.0e-6:
            raise RuntimeError(
                f"RECOMB72_LIVE_LOCUS_OUT_OF_BOUNDS locus={locus} value={value}"
            )
        package[locus] = min(float(upper), max(float(lower), value))
    return package


def _policy_sha256(policy: Any) -> str:
    digest = hashlib.sha256(type(policy).__name__.encode("utf-8"))
    parameters = getattr(policy, "parameters", None)
    if not callable(parameters):
        raise RuntimeError("RECOMB72_LIVE_POLICY_PARAMETERS_MISSING")
    for parameter in parameters():
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(int(value) for value in tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _neutrality_receipt(runner: Any) -> dict[str, Any]:
    env = runner.env_cfg
    cfg = runner.sim_cfg
    expected = {
        "food_spawn_per_step": 0,
        "max_food": 0,
        "move_cost": 0.0,
        "rest_cost": 0.0,
        "gather_cost": 0.0,
        "attack_cost": 0.0,
        "food_energy": 0.0,
        "enable_biomes": False,
        "enable_combat": False,
        "enable_injury": False,
        "enable_scar": False,
        "patch_world_enable": False,
    }
    mismatches = {
        name: {"expected": wanted, "actual": getattr(env, name, None)}
        for name, wanted in expected.items()
        if getattr(env, name, None) != wanted
    }
    if (
        mismatches
        or not bool(getattr(cfg, "recomb72_neutral_actions_enable", False))
        or int(getattr(cfg, "recomb72_neutral_action_id", -1)) != 5
        or not bool(
            getattr(cfg, "recomb72_neutral_environment_enable", False)
        )
    ):
        raise RuntimeError(
            f"RECOMB72_LIVE_NEUTRALITY_CONTRACT_MISMATCH fields={mismatches!r}"
        )
    agents = list(runner.pop.alive_agents())
    policy_hashes = {_policy_sha256(agent.policy) for agent in agents}
    normalized_genomes = []
    lifecycle_rows = []
    for agent in agents:
        genome = copy.deepcopy(agent.genome.to_dict())
        for locus in AUDITED_LOCI:
            genome[locus] = NEUTRAL_PACKAGE[locus]
        genome["mutation_rate"] = 0.0
        normalized_genomes.append(_sha256(genome))
        lifecycle_rows.append(
            {
                "energy": float(agent.energy),
                "health": float(agent.health),
                "injury": float(getattr(agent, "injury", 0.0)),
                "scar": float(getattr(agent, "scar", 0.0)),
                "age": int(getattr(agent, "age", 0)),
                "repro_cooldown": int(getattr(agent, "repro_cooldown", 0)),
                "alive": bool(agent.alive),
            }
        )
    genome_hashes = set(normalized_genomes)
    lifecycle_hashes = {_sha256(row) for row in lifecycle_rows}
    if len(policy_hashes) != 1 or len(genome_hashes) != 1 or len(lifecycle_hashes) != 1:
        raise RuntimeError(
            "RECOMB72_LIVE_FOUNDER_HETEROGENEITY "
            f"policy={len(policy_hashes)} genome={len(genome_hashes)} "
            f"lifecycle={len(lifecycle_hashes)}"
        )
    return {
        "schema_version": "recomb72_live_neutrality_receipt_v1",
        "trait_specific_action_authority": False,
        "ecological_selection": False,
        "genotype_sensitive_world_mechanics": False,
        "neutral_action": "ACT_REST",
        "neutral_environment_commit": True,
        "environment_contract": expected,
        "founder_policy_sha256": next(iter(policy_hashes)),
        "founder_non_audited_genome_sha256": next(iter(genome_hashes)),
        "founder_lifecycle_sha256": next(iter(lifecycle_hashes)),
        "founder_policy_unique_count": len(policy_hashes),
        "founder_non_audited_genome_unique_count": len(genome_hashes),
        "founder_lifecycle_unique_count": len(lifecycle_hashes),
    }


def _ancestry_group(ancestry: Mapping[str, Any]) -> str:
    a_mass = 0.0
    b_mass = 0.0
    for founder, raw_weight in ancestry.items():
        token = str(founder)
        weight = float(raw_weight)
        if token.startswith("founder_"):
            ordinal = int(token.rsplit("_", 1)[-1])
            if ordinal < 32:
                a_mass += weight
            else:
                b_mass += weight
    if a_mass > b_mass:
        return "a"
    if b_mass > a_mass:
        return "b"
    return "mixed"


def _lineage_metrics(agents: Sequence[Any]) -> dict[str, Any]:
    ancestry_by_agent = {
        int(agent.agent_id): dict(getattr(agent, "_recomb72_ancestry"))
        for agent in agents
    }
    concentration = ancestry_concentration(
        [int(agent.agent_id) for agent in agents],
        ancestry_by_agent,
    )
    grouped: dict[str, list[Any]] = {"a": [], "b": [], "mixed": []}
    for agent in agents:
        grouped[_ancestry_group(ancestry_by_agent[int(agent.agent_id)])].append(agent)
    centroids: dict[str, dict[str, float]] = {}
    for group in ("a", "b"):
        if grouped[group]:
            centroids[group] = {
                locus: sum(_agent_package(agent)[locus] for agent in grouped[group])
                / len(grouped[group])
                for locus in AUDITED_LOCI
            }
    if set(centroids) == {"a", "b"}:
        distance = math.sqrt(
            sum(
                (
                    (centroids["a"][locus] - centroids["b"][locus])
                    / (LOCUS_BOUNDS[locus][1] - LOCUS_BOUNDS[locus][0])
                )
                ** 2
                for locus in AUDITED_LOCI
            )
            / len(AUDITED_LOCI)
        )
    else:
        distance = 0.0
    genealogy_persists = bool(grouped["a"] and grouped["b"])
    return {
        **concentration,
        "descendant_branch_counts": {
            group: len(grouped[group]) for group in ("a", "b", "mixed")
        },
        "descendant_branch_diversity": sum(bool(grouped[group]) for group in grouped),
        "effective_alive_lineages": concentration["effective_founder_ancestry_count"],
        "trait_distance_between_branches": float(distance),
        "package_distance_between_branches": float(distance),
        "genealogy_persists": genealogy_persists,
        "genetic_distinction_lost_threshold": 0.10,
        "genealogy_persists_while_genetic_distinction_disappears": (
            genealogy_persists and distance < 0.10
        ),
    }


def _not_applicable_packages(population_count: int) -> dict[str, Any]:
    return {
        "population_count": int(population_count),
        "intact_package_count": 0,
        "intact_package_fraction": 0.0,
        "recombinant_count": 0,
        "recombinant_fraction": 0.0,
        "neither_package_count": int(population_count),
        "neither_package_fraction": 1.0,
        "not_applicable": True,
    }


def _generation_row(
    runner: Any,
    *,
    generation: int,
    boundary_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cfg = runner.sim_cfg
    agents = list(runner.pop.alive_agents())
    if len(agents) != 64:
        raise RuntimeError(
            f"RECOMB72_LIVE_POPULATION_MISMATCH generation={generation} count={len(agents)}"
        )
    packages = [_agent_package(agent) for agent in agents]
    pattern = str(cfg.recomb72_founder_pattern)
    mode = str(cfg.recomb72_inheritance_mode)
    package_a, package_b = founder_packages(pattern)
    loci = {
        locus: locus_statistics(
            [row[locus] for row in packages],
            lower_bound=LOCUS_BOUNDS[locus][0],
            upper_bound=LOCUS_BOUNDS[locus][1],
            neutral_value=NEUTRAL_PACKAGE[locus],
        )
        for locus in AUDITED_LOCI
    }
    packages_row = (
        package_statistics(
            packages,
            package_a=package_a,
            package_b=package_b,
            bounds=LOCUS_BOUNDS,
        )
        if pattern == "correlated_multi_trait_packages"
        else _not_applicable_packages(len(packages))
    )

    event = dict(boundary_event or {})
    births = list(event.get("reproduction_events", ()) or ())
    contributions: Counter[int] = Counter()
    pair_categories: Counter[str] = Counter()
    child_categories: Counter[str] = Counter()
    inheritance_events: dict[str, Any] = {}
    for birth in births:
        parent_a_id = int(birth["parent_a_id"])
        parent_b_id = int(birth["parent_b_id"])
        contributions[parent_a_id] += 1
        contributions[parent_b_id] += 1
        parent_a_group = str(birth.get("parent_a_group") or "")
        parent_b_group = str(birth.get("parent_b_group") or "")
        child_group = str(birth.get("child_group") or "")
        if not parent_a_group:
            parent_a_group = _ancestry_group(birth["parent_a_ancestry"])
        if not parent_b_group:
            parent_b_group = _ancestry_group(birth["parent_b_ancestry"])
        if not child_group:
            child_group = _ancestry_group(birth["child_ancestry"])
        pair = "-".join(sorted((parent_a_group, parent_b_group)))
        pair_categories[pair] += 1
        child_categories[f"{pair}->{child_group}"] += 1
    if births:
        for locus in AUDITED_LOCI:
            inheritance_events[locus] = locus_inheritance_statistics(
                [
                    {
                        "parent_a": birth["parent_a_package"][locus],
                        "parent_b": birth["parent_b_package"][locus],
                        "child": birth["child_package"][locus],
                    }
                    for birth in births
                ],
                neutral_value=NEUTRAL_PACKAGE[locus],
            )

    before_ids = list(event.get("before_ids", ())) if event else [a.agent_id for a in agents]
    survivor_ids = list(event.get("survivor_ids", ())) if event else [a.agent_id for a in agents]
    mutation_events = int(event.get("mutation_events", 0))
    ordinary_births = int(event.get("ordinary_tick_births", 0))
    current_production_path_count = sum(
        dict(birth.get("inheritance") or {}).get("production_current_path") is True
        for birth in births
    )
    alternative_audit_path_count = sum(
        dict(birth.get("inheritance") or {}).get("non_audited_surface")
        == "exact_parent_a_copy"
        for birth in births
    )
    row = {
        "schema_version": LIVE_METRICS_SCHEMA_VERSION,
        "authority": "production_python_reference_simulation_runner",
        "cell_id": str(getattr(cfg, "chronicle_run_id", "") or getattr(cfg, "campaign_id", "")),
        "generation": int(generation),
        "mode": mode,
        "pattern": pattern,
        "paired_seed": int(cfg.recomb72_audit_seed),
        "population": {
            "before_cull": len(before_ids),
            "eligible_parents": len(survivor_ids),
            "selected_parent_events": sum(contributions.values()),
            "pairings": len(births),
            "children": len(births),
            "after_refill": len(agents),
            "failed_refill_events": int(event.get("failed_refill_events", 0)),
            "ordinary_births": ordinary_births,
            "mutation_events": mutation_events,
        },
        "reproduction": {
            **reproduction_concentration(contributions),
            "eligible_parent_count": len(survivor_ids),
            "selected_unique_parent_count": len(contributions),
            "same_and_cross_group_pairings": dict(sorted(pair_categories.items())),
            "offspring_category_by_parental_category": dict(sorted(child_categories.items())),
        },
        "ancestry": _lineage_metrics(agents),
        "loci": loci,
        "inheritance_events": inheritance_events,
        "packages": packages_row,
        "integrity": {
            "live_fixture_schema_version": LIVE_FIXTURE_SCHEMA_VERSION,
            "registry_sha256": registry_sha256(),
            "mode_sha256": mode_sha256(mode),
            "mutation_off_contract": MUTATION_OFF_CONTRACT,
            "founder_receipt": copy.deepcopy(
                getattr(runner.pop, "_recomb72_founder_receipt", None)
            ),
            "neutrality_receipt": copy.deepcopy(
                getattr(runner, "_recomb72_neutrality_receipt", None)
            ),
            "genotype_blind_cull": True,
            "all_survivors_eligible": True,
            "actual_ticks_per_generation": int(cfg.steps_per_generation),
            "real_runner": True,
            "chronicle_enabled": bool(cfg.chronicle_enable),
            "production_current_path_count": current_production_path_count,
            "alternative_audit_path_count": alternative_audit_path_count,
        },
    }
    row["row_sha256"] = _sha256(row)
    return row


def apply_recomb72_live_founders_v1(runner: Any) -> dict[str, Any] | None:
    """Install identical founder baselines and the frozen 32/32 packages."""

    if not _enabled(runner):
        return None
    cfg = runner.sim_cfg
    if int(cfg.steps_per_generation) != 80:
        raise RuntimeError("RECOMB72_LIVE_REQUIRES_T80")
    if len(runner.pop.alive_agents()) != 64:
        raise RuntimeError("RECOMB72_LIVE_REQUIRES_P64")
    agents = list(runner.pop.alive_agents())
    canonical_genome = copy.deepcopy(agents[0].genome)
    canonical_policy = copy.deepcopy(agents[0].policy)
    for agent in agents:
        agent.genome = copy.deepcopy(canonical_genome)
        agent.policy = copy.deepcopy(canonical_policy)
    receipt = runner.pop.apply_recomb72_founder_packages()
    runner._recomb72_generation_rows = []
    runner._recomb72_real_tick_count = 0
    runner._recomb72_neutral_action_agent_ticks = 0
    runner._recomb72_neutrality_receipt = _neutrality_receipt(runner)
    row = _generation_row(runner, generation=0, boundary_event=None)
    runner._recomb72_generation_rows.append(row)
    if bool(getattr(runner, "_recomb72_reporting_enabled", True)):
        _write_rows(runner)
    return {
        "schema_version": LIVE_FIXTURE_SCHEMA_VERSION,
        "founder_receipt": copy.deepcopy(receipt),
        "neutrality_receipt": copy.deepcopy(runner._recomb72_neutrality_receipt),
        "g0_row_sha256": row["row_sha256"],
    }


def _write_rows(runner: Any) -> None:
    path = Path(str(runner.run_dir)) / "recomb72_generation_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(getattr(runner, "_recomb72_generation_rows", ()) or ())
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def record_recomb72_live_generation_v1(
    runner: Any,
    *,
    generation: int,
) -> dict[str, Any] | None:
    """Observe one completed real generation boundary without consuming RNG."""

    if not _enabled(runner):
        return None
    event = getattr(runner.pop, "_recomb72_generation_boundary_event", None)
    if not isinstance(event, Mapping):
        raise RuntimeError("RECOMB72_LIVE_BOUNDARY_EVENT_MISSING")
    expected = int(generation)
    observed = int(event.get("generation", expected))
    if observed != expected:
        raise RuntimeError(
            f"RECOMB72_LIVE_BOUNDARY_GENERATION_MISMATCH expected={expected} observed={observed}"
        )
    runner._recomb72_real_tick_count = int(
        getattr(runner, "_recomb72_real_tick_count", 0)
    ) + int(runner.sim_cfg.steps_per_generation)
    row = _generation_row(runner, generation=expected, boundary_event=event)
    rows = list(getattr(runner, "_recomb72_generation_rows", ()) or ())
    if rows and int(rows[-1]["generation"]) >= expected:
        raise RuntimeError("RECOMB72_LIVE_DUPLICATE_GENERATION_ROW")
    rows.append(row)
    runner._recomb72_generation_rows = rows
    if bool(getattr(runner, "_recomb72_reporting_enabled", True)):
        _write_rows(runner)
    return row


def terminal_recomb72_live_receipt_v1(runner: Any) -> dict[str, Any]:
    """Return the authoritative live-cell receipt used by the campaign provider."""

    rows = list(getattr(runner, "_recomb72_generation_rows", ()) or ())
    terminal_generation = int(rows[-1]["generation"]) if rows else -1
    expected = list(range(terminal_generation + 1))
    observed = [int(row["generation"]) for row in rows]
    if observed != expected:
        raise RuntimeError(
            f"RECOMB72_LIVE_GENERATION_COVERAGE_MISMATCH expected={expected!r} observed={observed!r}"
        )
    expected_ticks = terminal_generation * int(runner.sim_cfg.steps_per_generation)
    recorded_ticks = int(getattr(runner, "_recomb72_real_tick_count", -1))
    agent_ticks = int(
        getattr(runner, "_recomb72_neutral_action_agent_ticks", -1)
    )
    if agent_ticks % 64:
        raise RuntimeError(
            f"RECOMB72_LIVE_AGENT_TICK_COUNT_INVALID observed={agent_ticks}"
        )
    actual_ticks = agent_ticks // 64
    if recorded_ticks != actual_ticks:
        raise RuntimeError(
            "RECOMB72_LIVE_TICK_OBSERVERS_DISAGREE "
            f"boundary={recorded_ticks} action={actual_ticks}"
        )
    if actual_ticks != expected_ticks:
        raise RuntimeError(
            f"RECOMB72_LIVE_TICK_COUNT_MISMATCH expected={expected_ticks} observed={actual_ticks}"
        )
    return {
        "schema_version": "recomb72_live_cell_receipt_v2",
        "authority": "production_python_reference_simulation_runner",
        "generation_count": terminal_generation,
        "ticks_per_generation": int(runner.sim_cfg.steps_per_generation),
        "actual_tick_count": actual_ticks,
        "neutral_action_agent_tick_count": agent_ticks,
        "generation_row_count": len(rows),
        "generation_rows_sha256": _sha256(rows),
        "registry_sha256": registry_sha256(),
        "mode_sha256": mode_sha256(str(runner.sim_cfg.recomb72_inheritance_mode)),
        "chronicle_enabled": bool(runner.sim_cfg.chronicle_enable),
        "chronicle_root": str(Path(str(runner.run_dir)) / "chronicle"),
        "neutrality_receipt": copy.deepcopy(
            getattr(runner, "_recomb72_neutrality_receipt", None)
        ),
    }


__all__ = [
    "LIVE_FIXTURE_SCHEMA_VERSION",
    "apply_recomb72_live_founders_v1",
    "record_recomb72_live_generation_v1",
    "terminal_recomb72_live_receipt_v1",
]
