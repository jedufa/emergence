from __future__ import annotations

from engine.tw_rngv2.contract import CONTRACT, CONTRACT_HASH, CONTRACT_IDENTITY, contract_hash


def test_contract_identity_is_frozen_and_hashable() -> None:
    assert CONTRACT["prototype_identity"] == "TW-RNGV2-DEA1-PROTO-1"
    assert CONTRACT["rng_semantics_version"] == 2
    assert CONTRACT["core"]["rounds"] == 10
    assert CONTRACT["core"]["constants_hex"]["M0"] == "d2511f53"
    assert len(CONTRACT_HASH) == 64
    assert CONTRACT_IDENTITY.startswith("TW-RNGV2-DEA1-PROTO-1@")
    assert contract_hash(CONTRACT) == CONTRACT_HASH
    assert CONTRACT["status"].startswith("FROZEN_FOR_STANDALONE_PROTOTYPE")
