from __future__ import annotations

from copy import deepcopy

from eiel_composition.adaptive_expedition.state_materialization import (
    ATLAS_POLICY_FACTS_BUDGET_BYTES,
    _atlas_entry_v1,
    _frame_maps_v1,
)
from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import ObservationFrameV1
from emergence_contracts.canonical import canonical_sha256
from tests.test_eiel_adaptive_expedition_coordinator_v1 import _frame


def test_atlas_policy_facts_are_bounded_and_source_bound() -> None:
    raw = deepcopy(_frame().to_dict())
    large_facts = {
        "genotype_multiset_sha256": "fixture-genotype-digest",
        "locus_value_histograms": {"locus-0": list(range(10_000))},
    }
    raw["policy_facing_facts"]["payload"]["active_genotype_summary"] = large_facts
    frame = ObservationFrameV1.from_dict(raw)
    frames_by_id, _ = _frame_maps_v1((frame,))

    entry = _atlas_entry_v1(
        {
            "entry_id": "atlas-entry",
            "frame_id": f"OF-{frame.observation_frame_sha256().hex}",
            "trajectory_id": frame.world_trajectory_id,
            "niche_id": "fixture-niche",
            "representative_role": "ROLE",
            "frontier_ordinal": 3,
            "observation_absolute_tick": frame.next_absolute_tick,
            "descriptor_values": {"opaque": 1},
            "parent_entry_ids": [],
        },
        frontier_ordinal=3,
        frames_by_id=frames_by_id,
    )

    projection = entry["policy_facing_facts"]
    assert len(canonical_json_bytes(projection)) <= ATLAS_POLICY_FACTS_BUDGET_BYTES
    assert projection["schema_version"] == (
        "eiel_adaptive_science_atlas_policy_facts_projection_v1"
    )
    payload = projection["payload"]
    source = frame.policy_facing_facts.to_value()
    assert payload["source_schema_version"] == "coordinator_policy_facts_v1"
    assert payload["source_payload_sha256"] == canonical_sha256(source).to_dict()
    assert payload["source_payload_bytes"] == len(canonical_json_bytes(source))
    assert payload["facts"]["active_count"] == 2
    omitted = payload["facts"]["active_genotype_summary"]
    assert omitted["schema_version"] == (
        "eiel_adaptive_science_opaque_policy_fact_subtree_v1"
    )
    assert omitted["source_sha256"] == canonical_sha256(large_facts).to_dict()
    assert omitted["source_bytes"] == len(canonical_json_bytes(large_facts))
