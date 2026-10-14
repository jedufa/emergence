from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pytest

from engine.authoritative_state.conversions import (
    UINT64_MAX,
    checked_finite_float32,
    checked_finite_positive_float32,
    checked_int32,
    checked_lineage_founder_id,
    checked_uint64,
    uint64_from_json_decimal,
    uint64_to_json_decimal,
)
from engine.authoritative_state.errors import AuthoritativeStateContractError
from engine.authoritative_state.identity import (
    BINARY_IDENTITY_SCHEMA_V1,
    BinaryIdentityV1,
    canonical_sha256,
    require_identity_equal,
)


def _abi_rows() -> list[dict[str, object]]:
    return [
        {
            "entrypoint": "cpu_owner_v1",
            "abi_version": "cpu_abi_v1",
            "argument_count": 17,
            "argument_order_sha256": "1" * 64,
        },
        {
            "entrypoint": "cuda_owner_v1",
            "abi_version": "cuda_abi_v1",
            "argument_count": 161,
            "argument_order_sha256": "2" * 64,
        },
    ]


def _identity_payload() -> dict[str, object]:
    abi = _abi_rows()
    return {
        "contract_schema_version": BINARY_IDENTITY_SCHEMA_V1,
        "state_schema_version": "eiel_authoritative_state_schema_v1",
        "state_schema_sha256": "3" * 64,
        "checkpoint_schema_version": "checkpoint_bundle_v2",
        "checkpoint_schema_sha256": "4" * 64,
        "abi_variants": abi,
        "abi_set_sha256": canonical_sha256(abi),
        "source_commit": "5" * 40,
        "source_inventory_sha256": "6" * 64,
        "build_identity_sha256": "7" * 64,
        "supported_route_contracts": ["patch_fixed_v1", "wdl_v2"],
        "capabilities": ["checkpoint_roundtrip", "fresh_owner_restore"],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-(2**31), -(2**31)),
        (2**31 - 1, 2**31 - 1),
        (np.int64(42), 42),
    ],
)
def test_checked_int32_accepts_exact_boundaries(value: object, expected: int) -> None:
    assert checked_int32(value, path="state.counter") == expected


@pytest.mark.parametrize("value", [-(2**31) - 1, 2**31, True, 1.0, "1"])
def test_checked_int32_rejects_type_and_range(value: object) -> None:
    with pytest.raises(AuthoritativeStateContractError) as caught:
        checked_int32(value, path="state.counter")
    assert caught.value.code in {"E_STATE_TYPE", "E_STATE_INT_RANGE"}
    assert caught.value.path == "state.counter"


@pytest.mark.parametrize("value", [-1, 0, 2**31 - 1, np.int32(7)])
def test_founder_id_accepts_sentinel_and_nonnegative_int32(value: object) -> None:
    assert checked_lineage_founder_id(value, path="identity.lineage_founder_id") == int(
        value
    )


@pytest.mark.parametrize("value", [-2, 2**31])
def test_founder_id_rejects_values_outside_semantic_storage_range(value: int) -> None:
    with pytest.raises(
        AuthoritativeStateContractError,
        match=(
            r"E_STATE_LINEAGE_FOUNDER_RANGE:identity\.lineage_founder_id: "
            r"must be -1 or an integer in \[0, 2147483647\]"
        ),
    ):
        checked_lineage_founder_id(value, path="identity.lineage_founder_id")


@pytest.mark.parametrize(
    "value",
    [0, 2**32 - 1, 2**32, 2**53 - 1, 2**53 + 1, 2**63, UINT64_MAX],
)
def test_uint64_decimal_json_round_trip_is_lossless(value: int) -> None:
    encoded = uint64_to_json_decimal(value, path="rng.word")
    assert isinstance(encoded, str)
    assert json.loads(json.dumps(encoded)) == encoded
    assert uint64_from_json_decimal(encoded, path="rng.word") == value


@pytest.mark.parametrize(
    "value", [-1, UINT64_MAX + 1, True, 1.0, "1"]
)
def test_checked_uint64_rejects_non_uint64_values(value: object) -> None:
    with pytest.raises(AuthoritativeStateContractError):
        checked_uint64(value, path="rng.word")


@pytest.mark.parametrize(
    "value", [0, 1, "01", "+1", "-1", " 1", str(UINT64_MAX + 1)]
)
def test_uint64_json_decoder_rejects_noncanonical_or_out_of_range(
    value: object,
) -> None:
    with pytest.raises(AuthoritativeStateContractError) as caught:
        uint64_from_json_decimal(value, path="rng.word")
    assert caught.value.code == "E_STATE_UINT64_JSON"


def test_float32_narrowing_returns_exact_binary32_value() -> None:
    source = 1.00000006
    expected = struct.unpack("!f", struct.pack("!f", source))[0]
    assert checked_finite_float32(source, path="runtime.value") == expected


@pytest.mark.parametrize(
    "value",
    [
        np.nextafter(np.float32(0), np.float32(1), dtype=np.float32),
        np.finfo(np.float32).tiny,
        1.0,
        np.finfo(np.float32).max,
    ],
)
def test_positive_float32_accepts_representable_boundaries(value: object) -> None:
    narrowed = checked_finite_positive_float32(value, path="runtime.max_energy")
    assert np.isfinite(np.float32(narrowed))
    assert narrowed > 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_positive_float32_rejects_nonfinite(value: float) -> None:
    with pytest.raises(AuthoritativeStateContractError) as caught:
        checked_finite_positive_float32(value, path="runtime.max_energy")
    assert caught.value.code == "E_STATE_FLOAT_FINITE"


@pytest.mark.parametrize("value", [0.0, -0.0, -1.0])
def test_positive_float32_rejects_nonpositive(value: float) -> None:
    with pytest.raises(AuthoritativeStateContractError) as caught:
        checked_finite_positive_float32(value, path="runtime.start_health")
    assert caught.value.code == "E_STATE_FLOAT_POSITIVE"


@pytest.mark.parametrize(
    "value",
    [
        float(np.finfo(np.float32).max) * 2.0,
        float(np.nextafter(np.float32(0), np.float32(1))) / 2.0,
    ],
)
def test_positive_float32_rejects_overflow_and_underflow(value: float) -> None:
    with pytest.raises(AuthoritativeStateContractError) as caught:
        checked_finite_positive_float32(value, path="runtime.max_energy")
    assert caught.value.code == "E_STATE_FLOAT32_RANGE"


def test_binary_identity_round_trip_and_hash_are_canonical() -> None:
    payload = _identity_payload()
    identity = BinaryIdentityV1.from_mapping(payload)
    assert identity.payload() == payload
    assert identity.identity_sha256 == canonical_sha256(payload)
    assert BinaryIdentityV1.from_mapping(identity.payload()) == identity


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("state_schema_sha256"),
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value.__setitem__("state_schema_sha256", "A" * 64),
        lambda value: value.__setitem__("source_commit", "5" * 39),
        lambda value: value.__setitem__(
            "capabilities", list(reversed(value["capabilities"]))
        ),
        lambda value: value.__setitem__(
            "abi_variants", list(reversed(value["abi_variants"]))
        ),
        lambda value: value.__setitem__("abi_set_sha256", "8" * 64),
    ],
)
def test_binary_identity_rejects_structural_and_hash_drift(mutate) -> None:
    payload = deepcopy(_identity_payload())
    mutate(payload)
    with pytest.raises(AuthoritativeStateContractError):
        BinaryIdentityV1.from_mapping(payload)


def test_binary_identity_rejects_boolean_argument_count() -> None:
    payload = _identity_payload()
    payload["abi_variants"][0]["argument_count"] = True
    payload["abi_set_sha256"] = canonical_sha256(payload["abi_variants"])
    with pytest.raises(AuthoritativeStateContractError) as caught:
        BinaryIdentityV1.from_mapping(payload)
    assert caught.value.code == "E_STATE_TYPE"


def test_require_identity_equal_accepts_exact_and_rejects_one_hash_change() -> None:
    expected = _identity_payload()
    assert require_identity_equal(expected, deepcopy(expected), path="artifact").payload() == (
        expected
    )
    actual = deepcopy(expected)
    actual["build_identity_sha256"] = "8" * 64
    with pytest.raises(AuthoritativeStateContractError) as caught:
        require_identity_equal(expected, actual, path="artifact")
    assert caught.value.code == "E_STATE_IDENTITY_MISMATCH"
    assert "expected=" in caught.value.detail
    assert "actual=" in caught.value.detail


def test_cpp_binary_identity_header_compiles_and_validates(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    project = Path(__file__).resolve().parents[1]
    source = tmp_path / "identity_probe.cpp"
    source.write_text(
        """
#include "engine/native_runtime/authoritative_state/binary_identity_v1.hpp"

using namespace eiel::authoritative_state;

int main() {
    static constexpr AbiVariantIdentityV1 abi[] = {
        {"cuda_owner_v1", "cuda_abi_v1", 161U,
         "2222222222222222222222222222222222222222222222222222222222222222"}
    };
    static constexpr std::string_view routes[] = {"patch_fixed_v1", "wdl_v2"};
    static constexpr std::string_view capabilities[] = {
        "checkpoint_roundtrip", "fresh_owner_restore"
    };
    static constexpr BinaryIdentityV1 identity{
        kBinaryIdentitySchemaV1,
        "state_schema_v1",
        "3333333333333333333333333333333333333333333333333333333333333333",
        "checkpoint_v2",
        "4444444444444444444444444444444444444444444444444444444444444444",
        abi,
        1U,
        "5555555555555555555555555555555555555555555555555555555555555555",
        "6666666666666666666666666666666666666666",
        "7777777777777777777777777777777777777777777777777777777777777777",
        "8888888888888888888888888888888888888888888888888888888888888888",
        routes,
        2U,
        capabilities,
        2U
    };
    static_assert(validate_binary_identity_v1(identity) == IdentityErrorCodeV1::kOk);
    static_assert(is_canonical_uint64_decimal("18446744073709551615"));
    static_assert(!is_canonical_uint64_decimal("18446744073709551616"));
    static_assert(!is_canonical_uint64_decimal("01"));
    return 0;
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
            str(project),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
