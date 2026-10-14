from __future__ import annotations

from engine.hapb_rngv2.contract import (
    CONTRACT,
    CONTRACT_HASH,
    INVENTORY,
    PARENT_TW_IDENTITY,
    PARENT_TW_IDENTITY_DECLARED,
    SHARED_PARENT_DOMAINS,
)
from engine.hapb_rngv2.identity import HapbIdentityMap
from engine.hapb_rngv2.oracle import HapbEvent, HapbOracle
from engine.hapb_rngv2.policy import fill_policy_window_v2
from engine.tensor_world.c5_rngv2_cpu import mutate_planned_births_v2
from engine.tensor_world.c5_rngv2_route import production_contract_identity
from engine.tw_rngv2.address import EventAddress
from engine.tw_rngv2.oracle import PrototypeOracle
from engine.tw_rngv2.philox import PHILOX_KATS, philox4x32_10
from engine.tw_rngv2.status import Status


def test_official_philox4x32_10_kats() -> None:
    assert CONTRACT_HASH
    for rounds, counter, key, expected in PHILOX_KATS:
        assert philox4x32_10(counter, key, rounds=rounds) == expected


def test_parent_tw_contract_is_declared_or_recorded_successor() -> None:
    live = production_contract_identity()
    assert live == PARENT_TW_IDENTITY
    assert live == PARENT_TW_IDENTITY_DECLARED or live.startswith(
        "tensor_world_c0_c7_rngv2_v1@"
    )
    assert CONTRACT["parent_tw_contract_identity"] == PARENT_TW_IDENTITY_DECLARED


def test_inventory_has_no_unknown_domains() -> None:
    rows = INVENTORY["domains"]
    assert rows
    assert INVENTORY["status"] == "RESOLVED_NO_UNKNOWN"
    assert {row["classification"] for row in rows} <= {
        "MIGRATE_TO_V2",
        "PRESERVE_COMPACT_VALID_DOMAIN",
        "DETERMINISTIC_NO_RNG",
        "NOT_REACHED_BY_HAPB_FRAG",
    }
    assert all(row["classification"] != "UNKNOWN" for row in rows)


def test_shared_tensorworld_hapb_fixture_bitwise_parity() -> None:
    parent = PrototypeOracle()
    hapb = HapbOracle()
    addresses = [
        EventAddress(0, 0x0330, 11, 4, 0),
        EventAddress(0, 0x0331, 11, 4, 2),
        EventAddress(0, 0x0332, 11, 4, 3),
        EventAddress(3, 0x0310, 8, 9, 1),
        EventAddress(1, 0x0210, 4, 2, 0),
    ]
    for address in addresses:
        assert int(address.domain_id) in SHARED_PARENT_DOMAINS
        tw = parent.generate(address, n=7)
        hapb_result = hapb.generate(
            HapbEvent(
                address.world_uid,
                address.domain_id,
                address.subject_uid,
                address.absolute_tick,
                address.operation_id,
            ),
            n=7,
        )
        assert hapb_result.status is tw.status
        assert hapb_result.words == tw.words
        assert hapb_result.bounded == tw.bounded
        assert hapb_result.accepted_ordinal == tw.accepted_ordinal
        assert hapb_result.z0_bits == tw.z0_bits
        assert hapb_result.z1_bits == tw.z1_bits
        assert hapb_result.uniform_f32_bits == tw.uniform_f32_bits


def test_bounded_integer_retry_and_cacheless_normal_bits() -> None:
    oracle = HapbOracle()
    result = oracle.generate(HapbEvent(0, 0x0332, 5, 1, 0), n=7)
    assert result.status is Status.RNG_OK
    assert 0 <= result.bounded < 7
    assert result.accepted_ordinal >= 0
    retry = oracle.generate(HapbEvent(0, 0x0332, 5, 1, 0), n=7)
    assert retry.bounded == result.bounded
    assert retry.accepted_ordinal == result.accepted_ordinal
    normal = oracle.generate(HapbEvent(0, 0x0331, 5, 1, 0), n=7)
    again = oracle.generate(HapbEvent(0, 0x0331, 5, 1, 0), n=7)
    assert normal.z0_bits == again.z0_bits
    assert normal.z1_bits == again.z1_bits
    assert normal.z0_bits != 0 or normal.z1_bits != 0


def test_directed_event_identity_and_cross_domain_uniqueness() -> None:
    oracle = HapbOracle()
    birth = oracle.generate(HapbEvent(0, 0x0310, 3, 8, 0), n=7)
    mutation = oracle.generate(HapbEvent(0, 0x0330, 3, 8, 0), n=7)
    action = oracle.generate(HapbEvent(0, 0x0210, 3, 8, 0), n=7)
    assert birth.words != mutation.words
    assert mutation.words != action.words
    other_subject = oracle.generate(HapbEvent(0, 0x0310, 4, 8, 0), n=7)
    assert birth.words != other_subject.words


def test_semantic_address_independent_of_physical_slot_and_page() -> None:
    mapper = HapbIdentityMap(cells=1, slots=4)
    uid = mapper.allocate(0, 1)
    oracle = HapbOracle()
    first = oracle.generate(HapbEvent(0, 0x0331, uid, 12, 1), n=7)
    mapper.remap_physical(0, 1, 3)
    mapper.move_host_page(0, 3, 99)
    mapper.compact(0)
    assert mapper.uid_at(0, 0) == uid
    assert mapper.page_of_slot[0][0] == 99
    after = oracle.generate(HapbEvent(0, 0x0331, mapper.uid_at(0, 0), 12, 1), n=7)
    assert after.words == first.words
    assert after.z0_bits == first.z0_bits
    other = mapper.allocate(0, 2)
    distinct = oracle.generate(HapbEvent(0, 0x0331, other, 12, 1), n=7)
    assert distinct.words != first.words


def test_compact_keeps_page_mapping_with_live_identities() -> None:
    mapper = HapbIdentityMap(cells=1, slots=4)
    first = mapper.allocate(0, 2)
    second = mapper.allocate(0, 0)
    mapper.move_host_page(0, 2, 17)
    mapper.move_host_page(0, 0, 4)
    mapper.compact(0)
    assert mapper.uid_at(0, 0) == second
    assert mapper.uid_at(0, 1) == first
    assert mapper.page_of_slot[0][0] == 4
    assert mapper.page_of_slot[0][1] == 17
    assert mapper.uid_at(0, 2) is None


def test_inventory_preserves_compact_pcg_domains() -> None:
    by_name = {row["domain"]: row["classification"] for row in INVENTORY["domains"]}
    assert by_name["action_selection"] == "PRESERVE_COMPACT_VALID_DOMAIN"
    assert by_name["food_resource_spawn"] == "PRESERVE_COMPACT_VALID_DOMAIN"
    assert by_name["lifecycle_drive_noise"] == "PRESERVE_COMPACT_VALID_DOMAIN"
    assert by_name["ordinary_birth_scalar_mutation"] == "MIGRATE_TO_V2"
    assert by_name["policy_newborn_and_window_crossover"] == "MIGRATE_TO_V2"


def test_policy_window_is_addressed_by_semantic_episode_not_slot() -> None:
    first = fill_policy_window_v2(
        world_uid=2, semantic_origin=10, episodes=2, width=8
    )
    remapped_same_origin = fill_policy_window_v2(
        world_uid=2, semantic_origin=10, episodes=2, width=8
    )
    assert first.status is Status.RNG_OK
    assert first.host_tape_bytes == 0
    assert first.host_gaussian_cache_bytes == 0
    assert first.crossover == remapped_same_origin.crossover
    shifted = fill_policy_window_v2(
        world_uid=2, semantic_origin=11, episodes=2, width=8
    )
    assert shifted.crossover != first.crossover


def test_shared_birth_mutation_cpu_oracle_is_tensorworld_function() -> None:
    genes = (0.1, -0.2, 10.0, 1.0, 0.4, 0.02, 0.5, 0.1)
    plans = ((7, 3, genes, genes, 0.2, 0.2, 0),)
    first = mutate_planned_births_v2(world_uid=0, absolute_tick=4, plans=plans)
    reused = mutate_planned_births_v2(world_uid=0, absolute_tick=4, plans=plans)
    assert first[0].child_genes == reused[0].child_genes
    assert first[0].status is Status.RNG_OK
