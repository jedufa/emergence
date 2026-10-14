from __future__ import annotations

from engine.tw_rngv2.address import DOMAIN_BY_ID, EventAddress
from engine.tw_rngv2.contract import CONTRACT_HASH, MASK48
from engine.tw_rngv2.oracle import PrototypeOracle
from engine.tw_rngv2.status import Status


def _structured_addresses(count: int) -> list[EventAddress]:
    domains = tuple(DOMAIN_BY_ID)
    addresses: list[EventAddress] = []
    index = 0
    # Boundaries first.
    for domain in domains:
        for subject, tick, op in (
            (0, 0, 0),
            (1, 1, 1),
            (MASK48, MASK48, 0xFFFF),
        ):
            addresses.append(EventAddress(0, domain, subject, tick, op))
            index += 1
    while len(addresses) < count:
        world = index % 64
        domain = domains[index % len(domains)]
        subject = index  # unique subject keeps the corpus collision-free by construction
        tick = (index * 0xBF58476D1CE4E5B9) & MASK48
        op = (index * 17) & 0xFFFF
        if index & 2:
            tick ^= 1 << ((index // 2) % 48)
        addresses.append(EventAddress(world, domain, subject, tick, op))
        index += 1
    return addresses


def test_two_to_the_twenty_adversarial_cpu_corpus() -> None:
    assert CONTRACT_HASH
    count = 1 << 20
    addresses = _structured_addresses(count)
    oracle = PrototypeOracle()
    seen = set()
    mismatches = 0
    non_ok = 0
    digest_results = []
    for address in addresses:
        identity = address.identity_tuple(0, 0)
        assert identity not in seen
        seen.add(identity)
        result = oracle.generate(address, n=7)
        if result.status is not Status.RNG_OK:
            non_ok += 1
        digest_results.append(result)
    # Cross-domain: same subject/tick/op, different domains must differ.
    a = EventAddress(0, 0x0330, 99, 8, 3)
    b = EventAddress(0, 0x0331, 99, 8, 3)
    ra = PrototypeOracle().generate(a)
    rb = PrototypeOracle().generate(b)
    if ra.words == rb.words:
        mismatches += 1
    digest = oracle.digest(digest_results)
    assert len(seen) == count
    assert non_ok == 0
    assert mismatches == 0
    assert len(digest) == 64


def test_reordering_and_untaken_branch_do_not_alias() -> None:
    taken = EventAddress(3, 0x0210, 4, 10, 2)
    other = EventAddress(3, 0x0200, 4, 10, 2)
    oracle = PrototypeOracle()
    first = oracle.generate(taken)
    # Untaken branch generates nothing; other domain remains independently addressed.
    second = oracle.generate(other)
    replay = PrototypeOracle()
    assert replay.generate(other).words == second.words
    assert replay.generate(taken).words == first.words
    assert first.words != second.words
