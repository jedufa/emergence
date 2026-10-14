from __future__ import annotations

from engine.tw_rngv2.address import EventAddress
from engine.tw_rngv2.bounded import bounded_uint32, lemire_threshold
from engine.tw_rngv2.contract import CONTRACT_HASH
from engine.tw_rngv2.philox import philox4x32_10
from engine.tw_rngv2.status import Status


def _words(ordinal: int, address: EventAddress) -> int:
    block, lane = divmod(ordinal, 4)
    return philox4x32_10(
        address.counter_words(block),
        address.key_words(0xABCDEF0123456789),
    )[lane]


def test_lemire_edge_classes() -> None:
    assert CONTRACT_HASH
    address = EventAddress(0, 0x0332, 11, 4, 3)
    classes = [1, 2, 3, 5, 7, 15, 16, 17, (1 << 16) - 1, 1 << 16, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32]
    for n in classes:
        value, ordinal, status = bounded_uint32(n, lambda r, n=n: _words(r, address))
        assert status is Status.RNG_OK
        if n == 1:
            assert value == 0 and ordinal == 0
        elif n == 1 << 32:
            assert 0 <= value <= 0xFFFFFFFF
            assert ordinal == 0
        else:
            assert 0 <= value < n
            assert ordinal >= 0


def test_forced_rejection_uses_next_addressed_ordinal() -> None:
    n = 7
    threshold = lemire_threshold(n)
    reject = 0
    while ((reject * n) & 0xFFFFFFFF) >= threshold or ((reject * n) & 0xFFFFFFFF) >= n:
        reject += 1
        if reject > 10000:
            raise AssertionError("could not find a rejecting sample")
    accept = 1
    while ((accept * n) & 0xFFFFFFFF) < threshold:
        accept += 1

    def words(ordinal: int) -> int:
        return reject if ordinal == 0 else accept

    value, ordinal, status = bounded_uint32(n, words)
    assert status is Status.RNG_OK
    assert ordinal == 1
    assert 0 <= value < n


def test_invalid_n_and_threshold_match_uint32() -> None:
    address = EventAddress(1, 0x0200, 1, 1, 1)
    _, _, status = bounded_uint32(0, lambda r: _words(r, address))
    assert status is Status.RNG_INVALID_N
    assert lemire_threshold(7) == ((-7) & 0xFFFFFFFF) % 7
    assert lemire_threshold(8) == 0
