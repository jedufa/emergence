from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import struct

import pytest

from emergence_contracts.science_compact_capture import (
    ScienceCompactCaptureContractError,
    reconstruct_compact_birth_genome_v1,
)
from emergence_evidence.native_compact_reducer_v1 import (
    ARTIFACT_FILE_NAME,
    NativeCompactReducerArtifactV1,
    NativeCompactReducerError,
    accumulate_death_metrics_native_v1,
    build_native_compact_reducer_v1,
    create_native_compact_reducer_state_v1,
    default_native_compact_reducer_binding_v1,
    load_native_compact_reducer_v1,
    reconstruct_packed_birth_genome_native_v1,
    reconstruct_packed_birth_genomes_native_v1,
)
from emergence_evidence.science_strategy_attribution_v1 import (
    validate_genotype_reproductive_attribution_v1,
)


def _pack_genome(policy: list[int], trait: list[int]) -> bytes:
    return struct.pack("<224b32h", *policy, *trait)


def _unpack_genome(raw: bytes) -> dict[str, list[int]]:
    values = struct.unpack("<224b32h", raw)
    return {"policy": list(values[:224]), "trait": list(values[224:])}


def _expand_provenance(packed: bytes) -> bytes:
    return (
        bytes(
            (packed[linked_locus // 2] >> (4 * (linked_locus % 2))) & 0x0F
            for linked_locus in range(128)
        )
        + packed[64:68]
    )


def _pack_provenance(codes: list[tuple[int, int]], recombination: int) -> bytes:
    packed = bytearray(68)
    for linked_locus, (allele_zero, allele_one) in enumerate(codes):
        nibble = allele_zero | (allele_one << 2)
        packed[linked_locus // 2] |= nibble << (4 * (linked_locus % 2))
    packed[64:68] = recombination.to_bytes(4, "little")
    return bytes(packed)


@pytest.fixture(scope="module")
def native_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> NativeCompactReducerArtifactV1:
    artifact, manifest = build_native_compact_reducer_v1(
        tmp_path_factory.mktemp("native-compact-reducer")
    )
    return load_native_compact_reducer_v1(
        artifact, manifest, require_production_qualified=False
    )


def test_native_birth_reconstruction_matches_edge_oracle(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    parent_a_policy = [10] * 112 + [20] * 112
    parent_a_policy[113] = 127
    parent_b_policy = [30] * 112 + [40] * 112
    parent_a_trait = [100] * 16 + [200] * 16
    parent_b_trait = [300] * 16 + [400] * 16
    codes = [(0, 0)] * 128
    codes[0] = (1, 2)
    codes[1] = (1, 0)
    packed = _pack_provenance(codes, 1)
    result = reconstruct_packed_birth_genome_native_v1(
        native_artifact,
        parent_a_genome=_pack_genome(parent_a_policy, parent_a_trait),
        parent_b_genome=_pack_genome(parent_b_policy, parent_b_trait),
        packed_birth_provenance=packed,
        mutation_step=3,
    )
    oracle = reconstruct_compact_birth_genome_v1(
        parent_a_genome={"policy": parent_a_policy, "trait": parent_a_trait},
        parent_b_genome={"policy": parent_b_policy, "trait": parent_b_trait},
        birth_provenance=_expand_provenance(packed),
        mutation_step=3,
    )
    assert _unpack_genome(result.child_genomes) == oracle["child_genome"]
    assert result.reconstructed_birth_count == 1
    assert result.mutation_origin_count == oracle["mutation_origin_count"] == 3
    assert result.saturated_noop_count == oracle["saturated_noop_count"] == 1
    assert result.recombination_nonzero_pattern_count == 1
    assert result.recombination_choice_one_counts[0] == 1
    assert sum(result.recombination_choice_one_counts) == 1


def test_native_birth_batch_matches_randomized_python_oracle(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    rng = random.Random(0xE1E1100)
    parent_a_rows: list[bytes] = []
    parent_b_rows: list[bytes] = []
    provenance_rows: list[bytes] = []
    mutation_steps: list[int] = []
    expected_children: list[bytes] = []
    expected_origins = [0] * 128
    expected_deltas = [0] * 128
    expected_recombination_bits = [0] * 32
    expected_mutation_count = 0
    expected_saturated_count = 0
    expected_nonzero_recombination = 0
    for row in range(257):
        parent_a_policy = [rng.randint(-128, 127) for _ in range(224)]
        parent_b_policy = [rng.randint(-128, 127) for _ in range(224)]
        parent_a_trait = [rng.randint(-32768, 32767) for _ in range(32)]
        parent_b_trait = [rng.randint(-32768, 32767) for _ in range(32)]
        if row == 0:
            parent_a_policy[:4] = [-128, -127, 126, 127]
            parent_b_trait[:4] = [-32768, -32767, 32766, 32767]
        codes = [(rng.randrange(3), rng.randrange(3)) for _ in range(128)]
        recombination = rng.getrandbits(32)
        packed = _pack_provenance(codes, recombination)
        mutation_step = rng.randint(1, 127)
        parent_a = {"policy": parent_a_policy, "trait": parent_a_trait}
        parent_b = {"policy": parent_b_policy, "trait": parent_b_trait}
        oracle = reconstruct_compact_birth_genome_v1(
            parent_a_genome=parent_a,
            parent_b_genome=parent_b,
            birth_provenance=_expand_provenance(packed),
            mutation_step=mutation_step,
        )
        parent_a_rows.append(_pack_genome(parent_a_policy, parent_a_trait))
        parent_b_rows.append(_pack_genome(parent_b_policy, parent_b_trait))
        provenance_rows.append(packed)
        mutation_steps.append(mutation_step)
        expected_children.append(
            _pack_genome(
                oracle["child_genome"]["policy"], oracle["child_genome"]["trait"]
            )
        )
        expected_mutation_count += oracle["mutation_origin_count"]
        expected_saturated_count += oracle["saturated_noop_count"]
        expected_nonzero_recombination += int(recombination != 0)
        for bit in range(32):
            expected_recombination_bits[bit] += (recombination >> bit) & 1
        for change in oracle["changed_alleles"]:
            linked_locus = change["linked_locus"]
            expected_origins[linked_locus] += 1
            expected_deltas[linked_locus] += change["delta"]
    result = reconstruct_packed_birth_genomes_native_v1(
        native_artifact,
        parent_a_genomes=b"".join(parent_a_rows),
        parent_b_genomes=b"".join(parent_b_rows),
        packed_birth_provenance_rows=b"".join(provenance_rows),
        mutation_steps=mutation_steps,
    )
    assert result.child_genomes == b"".join(expected_children)
    assert result.reconstructed_birth_count == len(mutation_steps)
    assert result.mutation_origin_count == expected_mutation_count
    assert result.saturated_noop_count == expected_saturated_count
    assert result.recombination_nonzero_pattern_count == expected_nonzero_recombination
    assert result.mutation_origin_counts_by_linked_locus == tuple(expected_origins)
    assert result.mutation_delta_totals_by_linked_locus == tuple(expected_deltas)
    assert result.recombination_choice_one_counts == tuple(expected_recombination_bits)


def test_native_birth_rejects_reserved_code_and_invalid_extent(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    genome = _pack_genome([0] * 224, [0] * 32)
    reserved = _pack_provenance([(3, 0)] + [(0, 0)] * 127, 0)
    with pytest.raises(NativeCompactReducerError, match="reserved_mutation_code"):
        reconstruct_packed_birth_genome_native_v1(
            native_artifact,
            parent_a_genome=genome,
            parent_b_genome=genome,
            packed_birth_provenance=reserved,
            mutation_step=1,
        )
    with pytest.raises(NativeCompactReducerError, match="invalid extents"):
        reconstruct_packed_birth_genomes_native_v1(
            native_artifact,
            parent_a_genomes=genome[:-1],
            parent_b_genomes=genome,
            packed_birth_provenance_rows=bytes(68),
            mutation_steps=(1,),
        )
    with pytest.raises(ScienceCompactCaptureContractError, match="reserved"):
        reconstruct_compact_birth_genome_v1(
            parent_a_genome={"policy": [0] * 224, "trait": [0] * 32},
            parent_b_genome={"policy": [0] * 224, "trait": [0] * 32},
            birth_provenance=_expand_provenance(reserved),
            mutation_step=1,
        )


def _pack_metric_row(values: list[int]) -> bytes:
    output = bytearray()
    for field, value in enumerate(values):
        output.extend(struct.pack("<q" if field == 17 else "<I", value))
    assert len(output) == 88
    return bytes(output)


def test_native_death_metric_accumulator_preserves_wide_exact_totals(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    rows = [
        [2**32 - 1] * 17 + [-(2**63)] + [2**32 - 1] * 3,
        [2**32 - 1] * 17 + [2**63 - 1] + [2**32 - 1] * 3,
        list(range(17)) + [-41] + [18, 19, 20],
    ]
    totals = accumulate_death_metrics_native_v1(
        native_artifact, b"".join(_pack_metric_row(row) for row in rows)
    )
    assert totals == tuple(sum(row[field] for row in rows) for field in range(21))
    assert accumulate_death_metrics_native_v1(native_artifact, b"") == (0,) * 21
    with pytest.raises(NativeCompactReducerError, match="invalid extent"):
        accumulate_death_metrics_native_v1(native_artifact, b"x")


def test_prebuilt_identity_is_deterministic_and_fails_closed(tmp_path: Path) -> None:
    first_artifact, first_manifest = build_native_compact_reducer_v1(tmp_path / "a")
    second_artifact, second_manifest = build_native_compact_reducer_v1(tmp_path / "b")
    assert first_artifact.read_bytes() == second_artifact.read_bytes()
    first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert first_payload == second_payload
    with pytest.raises(NativeCompactReducerError, match="plan-bound manifest hash"):
        load_native_compact_reducer_v1(first_artifact, first_manifest)
    with pytest.raises(NativeCompactReducerError, match="manifest hash mismatch"):
        load_native_compact_reducer_v1(
            first_artifact,
            first_manifest,
            expected_manifest_sha256="0" * 64,
        )
    tampered = tmp_path / "tampered" / ARTIFACT_FILE_NAME
    tampered.parent.mkdir()
    shutil.copyfile(first_artifact, tampered)
    raw = bytearray(tampered.read_bytes())
    raw[-1] ^= 1
    tampered.write_bytes(raw)
    with pytest.raises(NativeCompactReducerError, match="artifact hash mismatch"):
        load_native_compact_reducer_v1(
            tampered, first_manifest, require_production_qualified=False
        )


def test_default_binding_is_prebuilt_and_source_hash_bound() -> None:
    artifact, manifest, manifest_sha256 = (
        default_native_compact_reducer_binding_v1()
    )
    loaded = load_native_compact_reducer_v1(
        artifact,
        manifest,
        require_production_qualified=False,
        expected_manifest_sha256=manifest_sha256,
    )
    assert loaded.artifact_path == artifact
    assert loaded.manifest_path == manifest
    assert loaded.manifest_sha256 == manifest_sha256
    assert loaded.production_routing_qualified is False


def _event(agent_id: int, parent_a: int, parent_b: int, kind: int, tick: int) -> bytes:
    return struct.pack("<QQQQ", agent_id, parent_a, parent_b, (kind << 6) | tick)


def test_persistent_native_state_reduces_disjoint_worlds_exactly(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    founder_a = _pack_genome([10] * 224, [100] * 32)
    founder_b = _pack_genome([20] * 224, [200] * 32)
    founder_c = _pack_genome([-5] * 224, [-50] * 32)
    first_codes = [(1, 0)] + [(0, 0)] * 127
    first_provenance = _pack_provenance(first_codes, 3)
    first_metrics = list(range(17)) + [-101] + [18, 19, 20]
    with create_native_compact_reducer_state_v1(
        native_artifact,
        resident_capacities=(3, 2),
        start_next_absolute_ticks=(0, 0),
        mutation_segments=(((0, 3),), ((0, 1),)),
        max_worker_threads=2,
    ) as state:
        state.initialize_world(
            0, agent_ids=(1, 2), packed_genomes=founder_a + founder_b
        )
        state.initialize_world(1, agent_ids=(1,), packed_genomes=founder_c)
        state.process_dispatch(
            start_next_absolute_tick=0,
            end_next_absolute_tick=1,
            event_rows=_event(3, 1, 2, 2, 0) + _event(1, 0, 0, 1, 0),
            death_metric_rows=_pack_metric_row(first_metrics),
            packed_birth_provenance_rows=first_provenance,
            world_offsets=(0, 1, 2),
        )
        world_zero = state.world_receipt(0)
        world_one = state.world_receipt(1)
        assert (world_zero.birth_count, world_zero.death_count) == (1, 0)
        assert world_zero.births.reconstructed_birth_count == 1
        assert world_zero.births.mutation_origin_count == 1
        assert world_zero.births.recombination_nonzero_pattern_count == 1
        assert world_zero.active_agent_count == 3
        assert world_one.completed_lifetime_metric_totals == tuple(first_metrics)
        assert world_one.active_agent_count == 0
        active = {record.agent_id: record for record in state.export_active(0)}
        expected_child = reconstruct_packed_birth_genome_native_v1(
            native_artifact,
            parent_a_genome=founder_a,
            parent_b_genome=founder_b,
            packed_birth_provenance=first_provenance,
            mutation_step=3,
        ).child_genomes
        assert active[3].genome == expected_child
        assert active[3].generation == 1
        assert active[3].mutation_lineage is True
        assert active[3].born is True

        second_metrics = [100 + index for index in range(17)] + [71] + [118, 119, 120]
        second_provenance = _pack_provenance([(0, 0)] * 128, 0)
        state.process_dispatch(
            start_next_absolute_tick=1,
            end_next_absolute_tick=2,
            event_rows=_event(1, 0, 0, 1, 0) + _event(4, 2, 3, 2, 0),
            death_metric_rows=_pack_metric_row(second_metrics),
            packed_birth_provenance_rows=second_provenance,
            world_offsets=(0, 2, 2),
        )
        world_zero = state.world_receipt(0)
        assert (world_zero.birth_count, world_zero.death_count) == (2, 1)
        assert world_zero.carrier_reproduction_birth_count == 1
        assert world_zero.mutation_carrier_reproduction_birth_count == 1
        assert world_zero.unique_born_agents_reproduced_count == 1
        assert world_zero.unique_mutation_carriers_reproduced_count == 1
        assert world_zero.maximum_lineage_depth == 2
        assert world_zero.completed_lifetime_metric_totals == tuple(second_metrics)
        assert world_zero.dispatch_count == 2
        assert world_zero.dispatch_tick_count == 2
        assert state.world_receipt(1).dispatch_count == 2
        active = {record.agent_id: record for record in state.export_active(0)}
        assert set(active) == {2, 3, 4}
        assert active[4].generation == 2
        assert active[4].mutation_lineage is True


def test_persistent_native_state_poisoned_failure_is_not_publishable(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    genome = _pack_genome([0] * 224, [0] * 32)
    state = create_native_compact_reducer_state_v1(
        native_artifact,
        resident_capacities=(2,),
        start_next_absolute_ticks=(0,),
        mutation_segments=(((0, 1),),),
        max_worker_threads=1,
    )
    try:
        state.initialize_world(0, agent_ids=(1,), packed_genomes=genome)
        with pytest.raises(
            NativeCompactReducerError,
            match=r"inactive_agent; world=0, tick=0, row=0, agent=2, parents=\(1,999\)",
        ):
            state.process_dispatch(
                start_next_absolute_tick=0,
                end_next_absolute_tick=1,
                event_rows=_event(2, 1, 999, 2, 0),
                death_metric_rows=b"",
                packed_birth_provenance_rows=bytes(68),
                world_offsets=(0, 1),
            )
        with pytest.raises(NativeCompactReducerError, match="poisoned"):
            state.world_receipt(0)
    finally:
        state.close()


def test_persistent_native_state_rejects_reused_or_nonmonotonic_agent_id(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    genome = _pack_genome([0] * 224, [0] * 32)
    state = create_native_compact_reducer_state_v1(
        native_artifact,
        resident_capacities=(2,),
        start_next_absolute_ticks=(0,),
        mutation_segments=(((0, 1),),),
        max_worker_threads=1,
    )
    try:
        state.initialize_world(0, agent_ids=(1, 2), packed_genomes=genome * 2)
        state.process_dispatch(
            start_next_absolute_tick=0,
            end_next_absolute_tick=1,
            event_rows=_event(2, 0, 0, 1, 0),
            death_metric_rows=_pack_metric_row([0] * 21),
            packed_birth_provenance_rows=b"",
            world_offsets=(0, 1),
        )
        with pytest.raises(
            NativeCompactReducerError,
            match=r"duplicate_agent; world=0, tick=1, row=0, agent=2",
        ):
            state.process_dispatch(
                start_next_absolute_tick=1,
                end_next_absolute_tick=2,
                event_rows=_event(2, 1, 1, 2, 0),
                death_metric_rows=b"",
                packed_birth_provenance_rows=_pack_provenance(
                    [(0, 0)] * 128, 0
                ),
                world_offsets=(0, 1),
            )
        with pytest.raises(NativeCompactReducerError, match="poisoned"):
            state.world_receipt(0)
    finally:
        state.close()


def test_strategy_sufficient_statistics_are_fused_into_native_event_pass(
    native_artifact: NativeCompactReducerArtifactV1,
) -> None:
    genome = _pack_genome([0] * 224, [0] * 32)
    zero_provenance = _pack_provenance([(0, 0)] * 128, 0)
    founder_three_metrics = list(range(21))
    founder_one_metrics = [100 + value for value in range(21)]
    terminal_metrics = {
        2: [200 + value for value in range(21)],
        4: [300 + value for value in range(21)],
        5: [400 + value for value in range(21)],
    }
    with create_native_compact_reducer_state_v1(
        native_artifact,
        resident_capacities=(3,),
        start_next_absolute_ticks=(0,),
        mutation_segments=(((0, 1),),),
        max_worker_threads=1,
    ) as state:
        state.initialize_world(
            0,
            agent_ids=(1, 2, 3),
            packed_genomes=genome * 3,
        )
        state.configure_strategy(
            0,
            strategy_count=2,
            founder_agent_ids=(1, 2, 3),
            founder_strategy_masks=(1, 2, 1),
            threshold_ticks=(0, 1, 2),
            early_life_tick=1,
        )
        state.process_dispatch(
            start_next_absolute_tick=0,
            end_next_absolute_tick=1,
            event_rows=_event(3, 0, 0, 1, 0) + _event(4, 1, 2, 2, 0),
            death_metric_rows=_pack_metric_row(founder_three_metrics),
            packed_birth_provenance_rows=zero_provenance,
            world_offsets=(0, 2),
        )
        state.process_dispatch(
            start_next_absolute_tick=1,
            end_next_absolute_tick=2,
            event_rows=_event(1, 0, 0, 1, 0) + _event(5, 2, 4, 2, 0),
            death_metric_rows=_pack_metric_row(founder_one_metrics),
            packed_birth_provenance_rows=zero_provenance,
            world_offsets=(0, 2),
        )
        state.observe_strategy_snapshot(
            0,
            active_agent_ids=(2, 4, 5),
            active_metric_rows=b"".join(
                _pack_metric_row(terminal_metrics[agent_id]) for agent_id in (2, 4, 5)
            ),
        )
        state.finalize_strategy(0, terminal_tick=2)
        receipt = state.strategy_receipt(0)
        assert receipt.enabled is True
        assert receipt.finalized is True
        assert receipt.threshold_ticks == (0, 1, 2)
        assert receipt.early_life_tick == 1
        assert receipt.first_birth_tick == 0
        assert receipt.first_death_tick == 0
        assert receipt.extinction_tick is None
        assert receipt.total_agent_count == 5
        assert receipt.founder_count == 3
        assert (receipt.birth_count, receipt.death_count) == (2, 2)
        assert receipt.maximum_lineage_depth == 2
        assert receipt.mixed_parent_birth_count == 2
        assert receipt.agent_counts_by_strategy == (4, 3)
        assert receipt.offspring_born_by_strategy == (2, 2)
        assert receipt.offspring_survived_by_strategy_threshold == (
            (2, 2, 2),
            (2, 2, 2),
        )
        assert receipt.maximum_lineage_depth_by_strategy == (2, 2)
        assert receipt.parent_birth_events_by_strategy == (2, 3)
        assert receipt.death_events_by_strategy == (2, 0)
        assert receipt.persistence_to_early_life_by_strategy == (2, 3)
        assert receipt.terminal_alive_by_strategy == (2, 3)
        assert receipt.offspring_group_counts == ((3, 2),)
        assert receipt.parent_pair_counts == ((1, 2, 1), (2, 3, 1))
        genotype = receipt.genotype_reproductive_attribution
        assert genotype is not None
        validate_genotype_reproductive_attribution_v1(genotype)
        assert genotype["group_count"] == 1
        genotype_group = genotype["groups"][0]
        assert genotype_group["agent_count"] == 5
        assert genotype_group["founder_count"] == 3
        assert genotype_group["offspring_birth_count"] == 2
        assert genotype_group["parent_birth_event_count"] == 4
        assert genotype_group["reproductive_agent_count"] == 3
        assert genotype_group["death_count"] == 2
        assert genotype_group["terminal_alive_count"] == 3
        expected_zero = tuple(
            founder_three_metrics[field]
            + founder_one_metrics[field]
            + terminal_metrics[4][field]
            + terminal_metrics[5][field]
            for field in range(21)
        )
        expected_one = tuple(
            terminal_metrics[2][field]
            + terminal_metrics[4][field]
            + terminal_metrics[5][field]
            for field in range(21)
        )
        assert receipt.terminal_metric_totals_by_strategy == (
            expected_zero,
            expected_one,
        )
        active = {record.agent_id: record for record in state.export_active(0)}
        assert active[2].strategy_mask == 2
        assert active[2].founder is True
        assert active[4].strategy_mask == 3
        assert active[4].birth_tick == 0
        assert active[5].strategy_mask == 3
        assert active[5].birth_tick == 1
