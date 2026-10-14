from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import pytest

from agents.genome import Genome
from agents.structural_identity_v1 import (
    IDENTITY_ORIGIN_LEGACY,
    IDENTITY_ORIGIN_NATIVE,
    STRUCTURAL_IDENTITY_SCHEMA_VERSION,
    StructuralIdentityError,
    migrate_legacy_structural_payload,
    new_native_identity,
)
from evolution.capacity_contract import build_capacity_contract, structural_bootstrap_reachability
from evolution.operators_c1 import apply_operator_pipeline, structural_operator_dependency_graph
from evolution.recombination_c2 import (
    C2_FROZEN_ALIGNMENT_KEY,
    C2_RECOMBINATION_VERSION,
    recombine_lineage_payloads,
    recombine_self_lineage_payload,
)
from evolution.structural_bootstrap_probe import run_structural_bootstrap_technical_probe
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


class SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return float(next(self.values))


def _empty() -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [],
        "connections": [],
    }


def _context(child: int, generation: int = 1, *, lineage: str = "") -> dict:
    return {
        "campaign_id": "escap2-test",
        "branch_id": "capacity",
        "generation": int(generation),
        "tick": 0,
        "child_agent_id": int(child),
        "parent_agent_ids": [1, 2],
        "lineage_token": str(lineage),
    }


def _apply_one(
    payload: dict,
    *,
    schedule_draw: float,
    child: int,
    generation: int,
    enabled: bool = True,
) -> tuple[dict, dict, int]:
    rng = SequenceRng([schedule_draw, 0.1, 0.2, 0.3])
    out, telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=rng,
        max_ops=1,
        structural_bootstrap_enable=bool(enabled),
        structural_identity_context=_context(child, generation),
    )
    return out, telemetry, rng.draw_count


def _minimal_graph() -> dict:
    payload, _, draws_1 = _apply_one(_empty(), schedule_draw=0.20, child=10, generation=1)
    payload, _, draws_2 = _apply_one(payload, schedule_draw=0.20, child=11, generation=2)
    payload, _, draws_3 = _apply_one(payload, schedule_draw=0.05, child=12, generation=3)
    assert (draws_1, draws_2, draws_3) == (4, 4, 4)
    return payload


def _repack(payload: dict, order: list[int]) -> dict:
    source_modules = list(payload["modules"])
    chosen = [deepcopy(source_modules[index]) for index in order]
    old_to_new: dict[str, str] = {}
    for index, module in enumerate(chosen):
        old = str(module["module_id"])
        new = f"slot_{index:02d}"
        module["module_id"] = new
        old_to_new[old] = new
    connections = []
    for connection in payload.get("connections", []):
        out = deepcopy(connection)
        out["src_module_id"] = old_to_new[str(connection["src_module_id"])]
        out["dst_module_id"] = old_to_new[str(connection["dst_module_id"])]
        connections.append(out)
    out = deepcopy(payload)
    out["modules"] = chosen
    out["connections"] = connections
    return Genome.from_dict(out).to_dict()


def test_empty_founder_dependency_graph_is_feature_gated() -> None:
    founder = Genome.random(random.Random(4))
    assert founder.modules == []
    assert founder.connections == []
    disabled = structural_operator_dependency_graph(bootstrap_enabled=False)
    enabled = structural_operator_dependency_graph(bootstrap_enabled=True)
    assert disabled["shortest_legal_path"] == []
    assert disabled["impossible_transitions"]
    assert enabled["shortest_legal_path"][-1] == "minimal_connected_structure"
    assert enabled["operators"]["duplicate_with_decay"]["eligible_from_empty"] is True
    assert enabled["requirement_classes"]["eligible_from_empty"] == ["duplicate_with_decay"]
    assert "promoter_expression_mutation" in enabled["requirement_classes"]["requires_at_least_one_module"]
    assert "motif_duplication" in enabled["requirement_classes"]["requires_two_or_more_modules"]
    assert "gate_drift" in enabled["requirement_classes"]["requires_existing_connection"]


def test_forced_normal_operator_path_reaches_minimal_structure_without_seeding() -> None:
    payload = _minimal_graph()
    assert len(payload["modules"]) == 2
    assert len(payload["connections"]) == 1
    assert all(module["identity_origin"] == IDENTITY_ORIGIN_NATIVE for module in payload["modules"])
    assert payload["connections"][0]["weight"] == 0.0
    assert payload["connections"][0]["channel"] == "c1_bootstrap"
    assert all(module["params"]["expression_scale"] == 0.0 for module in payload["modules"])


def test_bootstrap_fixed_draw_budget_and_rejected_proposal_retains_structure() -> None:
    disabled, telemetry_disabled, disabled_draws = _apply_one(
        _empty(), schedule_draw=0.20, child=20, generation=1, enabled=False
    )
    accepted, telemetry_accepted, accepted_draws = _apply_one(
        _empty(), schedule_draw=0.20, child=20, generation=1, enabled=True
    )
    ineligible, telemetry_ineligible, ineligible_draws = _apply_one(
        accepted, schedule_draw=0.05, child=21, generation=2, enabled=True
    )
    full = deepcopy(accepted)
    full["modules"] = []
    for index in range(12):
        identity = new_native_identity(
            context=_context(100 + index, 1),
            operator_id="fixture",
            attempt_index=index,
            operator_role=f"locus_{index}",
        )
        full["modules"].append(
            {
                "module_id": f"m{index:03d}",
                "module_type": "generic",
                "params": {"expression_scale": 0.0},
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
                **identity,
                "latent_pos": identity["topology_pos"],
            }
        )
    full = Genome.from_dict(full).to_dict()
    rejected, telemetry_rejected, rejected_draws = _apply_one(
        full, schedule_draw=0.20, child=300, generation=2, enabled=True
    )
    assert disabled_draws == accepted_draws == ineligible_draws == rejected_draws == 4
    assert disabled["modules"] == []
    assert telemetry_disabled["noop_reason_counts"] == {"bootstrap_disabled_empty_structure": 1}
    assert telemetry_accepted["structural_bootstrap_accepted_count"] == 1
    assert telemetry_ineligible["noop_reason_counts"] == {"requires_two_modules": 1}
    assert rejected["modules"] == full["modules"]
    assert telemetry_rejected["capacity_blocked_count"] == 1


def test_repeat_bootstrap_is_byte_and_event_deterministic() -> None:
    a, ta, da = _apply_one(_empty(), schedule_draw=0.20, child=30, generation=1)
    b, tb, db = _apply_one(_empty(), schedule_draw=0.20, child=30, generation=1)
    assert Genome.from_dict(a).canonical_bytes() == Genome.from_dict(b).canonical_bytes()
    assert ta == tb
    assert da == db == 4


def test_independent_local_slot_zero_bootstraps_are_not_homologous() -> None:
    a, _, _ = _apply_one(_empty(), schedule_draw=0.20, child=0, generation=1)
    b, _, _ = _apply_one(_empty(), schedule_draw=0.20, child=1, generation=1)
    assert a["modules"][0]["module_id"] == b["modules"][0]["module_id"] == "m000"
    assert a["modules"][0]["module_instance_id"] != b["modules"][0]["module_instance_id"]
    assert a["modules"][0]["homology_id"] != b["modules"][0]["homology_id"]
    child, summary = recombine_lineage_payloads(
        parent_a_payload=a,
        parent_b_payload=b,
        rng=SequenceRng([]),
    )
    assert summary.matched_modules == 0
    assert len(child["modules"]) == 2


def test_identity_ancestry_and_topology_survive_repacking_and_functional_mutation() -> None:
    graph = _minimal_graph()
    repacked = _repack(graph, [1, 0])
    before = {
        module["homology_id"]: (
            module["module_instance_id"],
            module["topology_pos"],
            module["ancestry"],
        )
        for module in repacked["modules"]
    }
    # promoter_expression_mutation is the final seventh scheduler bucket.
    rng = SequenceRng([0.95, 0.25, 0.75])
    mutated, _ = apply_operator_pipeline(
        payload=repacked,
        rng=rng,
        max_ops=1,
        structural_bootstrap_enable=True,
        structural_identity_context=_context(50, 4),
    )
    after = {
        module["homology_id"]: (
            module["module_instance_id"],
            module["topology_pos"],
            module["ancestry"],
        )
        for module in mutated["modules"]
    }
    assert before == after
    assert rng.draw_count == 3


def test_duplicate_gets_new_instance_and_homology_with_parent_ancestry() -> None:
    one, _, _ = _apply_one(_empty(), schedule_draw=0.20, child=60, generation=1)
    two, _, _ = _apply_one(one, schedule_draw=0.20, child=61, generation=2)
    parent, duplicate = sorted(two["modules"], key=lambda module: module["ancestry"]["origin_generation"])
    assert duplicate["module_instance_id"] != parent["module_instance_id"]
    assert duplicate["homology_id"] != parent["homology_id"]
    assert duplicate["ancestry"]["parent_module_instance_ids"] == [parent["module_instance_id"]]
    assert duplicate["ancestry"]["origin_operator"] == "duplicate_with_decay"


def test_homology_recombination_aligns_across_slots_and_remaps_connections() -> None:
    graph = _minimal_graph()
    parent_a = _repack(graph, [0, 1])
    parent_b = _repack(graph, [1, 0])
    parent_b["modules"][0]["params"]["expression_scale"] = 0.25
    rng = SequenceRng([0.25, 0.75, 0.25])
    child, summary = recombine_lineage_payloads(
        parent_a_payload=parent_a,
        parent_b_payload=parent_b,
        rng=rng,
    )
    assert summary.alignment_key == C2_FROZEN_ALIGNMENT_KEY == "homology_id"
    assert summary.matched_modules == 2
    assert summary.matched_connections == 1
    assert len(child["modules"]) == 2
    assert len(child["connections"]) == 1
    valid = {module["module_id"] for module in child["modules"]}
    assert child["connections"][0]["src_module_id"] in valid
    assert child["connections"][0]["dst_module_id"] in valid
    assert child["genes"]["c2_recombination_version"] == C2_RECOMBINATION_VERSION


def test_shared_duplicated_paralog_remains_a_distinct_alignable_locus() -> None:
    ancestor = _minimal_graph()
    duplicated, _, _ = _apply_one(
        ancestor,
        schedule_draw=0.20,
        child=65,
        generation=4,
    )
    assert len(duplicated["modules"]) == 3
    parent_a = _repack(duplicated, [0, 1, 2])
    parent_b = _repack(duplicated, [2, 0, 1])
    child, summary = recombine_lineage_payloads(
        parent_a_payload=parent_a,
        parent_b_payload=parent_b,
        rng=SequenceRng([0.1] * 16),
    )
    assert summary.matched_modules == 3
    assert len(child["modules"]) == 3
    assert len({module["homology_id"] for module in child["modules"]}) == 3
    assert any(
        module["ancestry"]["origin_operator"] == "duplicate_with_decay"
        and module["ancestry"]["parent_module_instance_ids"]
        for module in child["modules"]
    )


def test_parent_order_is_canonical_and_unmatched_handling_is_deterministic() -> None:
    shared = _minimal_graph()
    independent, _, _ = _apply_one(_empty(), schedule_draw=0.20, child=70, generation=1)
    parent_b = deepcopy(shared)
    parent_b["modules"].extend(deepcopy(independent["modules"]))
    parent_b = Genome.from_dict(parent_b).to_dict()
    rng_a = SequenceRng([0.25, 0.75, 0.25])
    rng_b = SequenceRng([0.25, 0.75, 0.25])
    child_ab, summary_ab = recombine_lineage_payloads(
        parent_a_payload=shared,
        parent_b_payload=parent_b,
        rng=rng_a,
    )
    child_ba, summary_ba = recombine_lineage_payloads(
        parent_a_payload=parent_b,
        parent_b_payload=shared,
        rng=rng_b,
    )
    assert child_ab == child_ba
    assert summary_ab == summary_ba
    assert len(child_ab["modules"]) == 3


def test_homology_recombination_fails_closed_when_unmatched_modules_exceed_cap() -> None:
    def parent(scope: int) -> dict:
        modules = []
        for index in range(7):
            identity = new_native_identity(
                context=_context(scope * 100 + index, 1),
                operator_id="fixture",
                attempt_index=index,
                operator_role=f"parent_{scope}_locus_{index}",
            )
            modules.append(
                {
                    "module_id": f"m{index:03d}",
                    "module_type": "generic",
                    "params": {"expression_scale": 0.0},
                    "io": {
                        "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                        "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                        "caps": {"max_fanin": 6, "max_fanout": 6},
                    },
                    **identity,
                    "latent_pos": identity["topology_pos"],
                }
            )
        return Genome.from_dict({**_empty(), "modules": modules}).to_dict()

    with pytest.raises(RuntimeError, match="DET_GENOME_B2_BUDGET_EXCEEDED"):
        recombine_lineage_payloads(
            parent_a_payload=parent(1),
            parent_b_payload=parent(2),
            rng=SequenceRng([]),
        )


def test_legacy_migration_is_zero_draw_conservative_and_preserves_source() -> None:
    legacy = {
        "schema_version": "modular_genome_v1",
        "modules": [{"module_id": "m000", "module_type": "generic", "params": {"p": 1.0}}],
        "connections": [],
    }
    migration = migrate_legacy_structural_payload(legacy, strict=True)
    assert migration["source_payload"] == legacy
    assert migration["rng_draws"] == 0
    module = migration["migrated_payload"]["modules"][0]
    assert module["identity_origin"] == IDENTITY_ORIGIN_LEGACY
    assert module["legacy_identity_source"]["homology_status"] == "unresolved_do_not_cross_align"
    first = Genome.from_dict(migration["migrated_payload"]).canonical_bytes()
    second = Genome.from_dict(migration["migrated_payload"]).canonical_bytes()
    assert first == second
    migrated_payload = Genome.from_dict(migration["migrated_payload"]).to_dict()
    legacy_instance = migrated_payload["modules"][0]["module_instance_id"]
    child, summary = recombine_lineage_payloads(
        parent_a_payload=migrated_payload,
        parent_b_payload=migrated_payload,
        rng=SequenceRng([]),
    )
    continued, continued_summary = recombine_lineage_payloads(
        parent_a_payload=child,
        parent_b_payload=child,
        rng=SequenceRng([]),
    )
    assert summary.matched_modules == continued_summary.matched_modules == 0
    assert child["modules"][0]["module_instance_id"] == legacy_instance
    assert continued["modules"][0]["module_instance_id"] == legacy_instance
    partial = deepcopy(legacy)
    partial["modules"][0]["module_instance_id"] = "claimed-native-without-homology"
    with pytest.raises(StructuralIdentityError):
        migrate_legacy_structural_payload(partial, strict=True)


def test_capacity_contract_reports_real_flagged_reachability_and_schema_versions() -> None:
    disabled = SimConfig()
    enabled = SimConfig()
    enabled.structural_bootstrap_enable = True
    assert structural_bootstrap_reachability(disabled)["classification"] == "not_bootstrap_reachable"
    assert structural_bootstrap_reachability(enabled)["classification"] == "bootstrap_reachable_feature_gated"
    contract = build_capacity_contract(sim_cfg=enabled)
    assert contract["structure"]["identity_schema_version"] == STRUCTURAL_IDENTITY_SCHEMA_VERSION
    assert contract["structure"]["recombination_alignment_key"] == "homology_id"
    assert (
        contract["structure"]["legacy_unresolved_recombination_policy"]
        == "canonical_parent_set_no_cross_alignment"
    )
    assert contract["structure"]["reachability"]["operator_dependency_graph"]["shortest_legal_path"]


def test_checkpoint_bundle_v2_roundtrip_preserves_identity_and_branch_continuation(tmp_path: Path) -> None:
    def make_sim(root: Path) -> SimulationRunner:
        cfg = SimConfig(
            seed=88021,
            generations=1,
            steps_per_generation=2,
            device="cpu",
            bank_device="cpu",
            behavior_log_enable=False,
        )
        cfg.branch_id = "escap2-checkpoint-branch"
        cfg.campaign_id = "escap2-checkpoint-campaign"
        cfg.checkpoint_bundle_v2_period_gen = 0
        return SimulationRunner(
            EnvConfig(width=8, height=8),
            PopConfig(population_size=4),
            cfg,
            NoOpMetricsLogger(),
            str(root),
            perf=False,
        )

    graph = _minimal_graph()
    sim_a = make_sim(tmp_path / "source")
    sim_a.run(start_gen=0, skip_init=False)
    for agent in sim_a.pop.agents.values():
        agent.genome = Genome.from_dict(graph)
    bundle = tmp_path / "source" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    assert loaded["metadata"]["structural_identity"]["migration_rng_draws"] == 0

    sim_b = make_sim(tmp_path / "restored")
    checkpoint_bundle_v2_apply(sim_b, loaded, restore_rng=True)
    source = {
        int(agent_id): agent.genome.canonical_bytes()
        for agent_id, agent in sim_a.pop.agents.items()
    }
    restored = {
        int(agent_id): agent.genome.canonical_bytes()
        for agent_id, agent in sim_b.pop.agents.items()
    }
    assert source == restored
    for agent in sim_b.pop.agents.values():
        assert all(module["identity_origin"] == IDENTITY_ORIGIN_NATIVE for module in agent.genome.modules)
        identities_before = {
            module["homology_id"]: module["module_instance_id"]
            for module in agent.genome.modules
        }
        continued, _ = recombine_self_lineage_payload(
            parent_payload=agent.genome.to_dict(),
            rng=SequenceRng([0.0] * 16),
        )
        identities_after = {
            module["homology_id"]: module["module_instance_id"]
            for module in continued["modules"]
        }
        assert identities_after == identities_before


def test_bounded_technical_probe_reaches_with_matched_disabled_controls_and_repeat_parity() -> None:
    result = run_structural_bootstrap_technical_probe()
    assert result["status"] == "passed"
    assert result["repeat_parity"] is True
    assert all(cell["reached"] for cell in result["treatment_cells"])
    assert all(not cell["reached"] for cell in result["bootstrap_disabled_controls"])
    assert result["claims"]["ecological_value"] == "not_tested"
