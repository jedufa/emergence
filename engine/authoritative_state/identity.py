"""Canonical binary self-description model for authoritative state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Final

from .conversions import checked_integer
from .errors import (
    IDENTITY_FIELDS_ERROR,
    IDENTITY_MISMATCH_ERROR,
    IDENTITY_ORDER_ERROR,
    IDENTITY_VALUE_ERROR,
    fail,
)


BINARY_IDENTITY_SCHEMA_V1: Final[str] = (
    "eiel_authoritative_state_binary_identity_v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_ABI_FIELDS = frozenset(
    {"entrypoint", "abi_version", "argument_count", "argument_order_sha256"}
)
_IDENTITY_FIELDS = frozenset(
    {
        "contract_schema_version",
        "state_schema_version",
        "state_schema_sha256",
        "checkpoint_schema_version",
        "checkpoint_schema_sha256",
        "abi_variants",
        "abi_set_sha256",
        "source_commit",
        "source_inventory_sha256",
        "build_identity_sha256",
        "supported_route_contracts",
        "capabilities",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_fields(value: Any, *, expected: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise fail(IDENTITY_FIELDS_ERROR, path, "must be an object")
    actual = set(value)
    if actual != expected:
        raise fail(
            IDENTITY_FIELDS_ERROR,
            path,
            f"fields differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}",
        )
    return value


def _string(value: Any, *, path: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise fail(IDENTITY_VALUE_ERROR, path, "has invalid format")
    return value


def _ordered_tokens(value: Any, *, path: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(not isinstance(item, str) or _TOKEN.fullmatch(item) is None for item in value)
    ):
        raise fail(IDENTITY_VALUE_ERROR, path, "must be an array of tokens")
    tokens = tuple(value)
    if tokens != tuple(sorted(set(tokens))):
        raise fail(IDENTITY_ORDER_ERROR, path, "must be sorted and unique")
    return tokens


@dataclass(frozen=True, slots=True)
class AbiVariantIdentityV1:
    entrypoint: str
    abi_version: str
    argument_count: int
    argument_order_sha256: str

    @classmethod
    def from_mapping(cls, value: Any, *, path: str) -> "AbiVariantIdentityV1":
        row = _exact_fields(value, expected=_ABI_FIELDS, path=path)
        return cls(
            entrypoint=_string(
                row["entrypoint"], path=f"{path}.entrypoint", pattern=_TOKEN
            ),
            abi_version=_string(
                row["abi_version"], path=f"{path}.abi_version", pattern=_TOKEN
            ),
            argument_count=checked_integer(
                row["argument_count"],
                bits=32,
                signed=False,
                path=f"{path}.argument_count",
            ),
            argument_order_sha256=_string(
                row["argument_order_sha256"],
                path=f"{path}.argument_order_sha256",
                pattern=_SHA256,
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "entrypoint": self.entrypoint,
            "abi_version": self.abi_version,
            "argument_count": self.argument_count,
            "argument_order_sha256": self.argument_order_sha256,
        }


@dataclass(frozen=True, slots=True)
class BinaryIdentityV1:
    contract_schema_version: str
    state_schema_version: str
    state_schema_sha256: str
    checkpoint_schema_version: str
    checkpoint_schema_sha256: str
    abi_variants: tuple[AbiVariantIdentityV1, ...]
    abi_set_sha256: str
    source_commit: str
    source_inventory_sha256: str
    build_identity_sha256: str
    supported_route_contracts: tuple[str, ...]
    capabilities: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, *, path: str = "identity") -> "BinaryIdentityV1":
        data = _exact_fields(value, expected=_IDENTITY_FIELDS, path=path)
        if data["contract_schema_version"] != BINARY_IDENTITY_SCHEMA_V1:
            raise fail(
                IDENTITY_VALUE_ERROR,
                f"{path}.contract_schema_version",
                f"must equal {BINARY_IDENTITY_SCHEMA_V1}",
            )
        raw_variants = data["abi_variants"]
        if (
            isinstance(raw_variants, (str, bytes))
            or not isinstance(raw_variants, Sequence)
            or not raw_variants
        ):
            raise fail(
                IDENTITY_VALUE_ERROR,
                f"{path}.abi_variants",
                "must be a nonempty array",
            )
        variants = tuple(
            AbiVariantIdentityV1.from_mapping(
                row, path=f"{path}.abi_variants[{index}]"
            )
            for index, row in enumerate(raw_variants)
        )
        keys = tuple((row.entrypoint, row.abi_version) for row in variants)
        if keys != tuple(sorted(set(keys))):
            raise fail(
                IDENTITY_ORDER_ERROR,
                f"{path}.abi_variants",
                "must be sorted and unique by entrypoint and ABI version",
            )
        supplied_abi_hash = _string(
            data["abi_set_sha256"],
            path=f"{path}.abi_set_sha256",
            pattern=_SHA256,
        )
        expected_abi_hash = canonical_sha256([row.payload() for row in variants])
        if supplied_abi_hash != expected_abi_hash:
            raise fail(
                IDENTITY_VALUE_ERROR,
                f"{path}.abi_set_sha256",
                "does not match canonical ABI variants",
            )
        return cls(
            contract_schema_version=BINARY_IDENTITY_SCHEMA_V1,
            state_schema_version=_string(
                data["state_schema_version"],
                path=f"{path}.state_schema_version",
                pattern=_TOKEN,
            ),
            state_schema_sha256=_string(
                data["state_schema_sha256"],
                path=f"{path}.state_schema_sha256",
                pattern=_SHA256,
            ),
            checkpoint_schema_version=_string(
                data["checkpoint_schema_version"],
                path=f"{path}.checkpoint_schema_version",
                pattern=_TOKEN,
            ),
            checkpoint_schema_sha256=_string(
                data["checkpoint_schema_sha256"],
                path=f"{path}.checkpoint_schema_sha256",
                pattern=_SHA256,
            ),
            abi_variants=variants,
            abi_set_sha256=supplied_abi_hash,
            source_commit=_string(
                data["source_commit"], path=f"{path}.source_commit", pattern=_COMMIT
            ),
            source_inventory_sha256=_string(
                data["source_inventory_sha256"],
                path=f"{path}.source_inventory_sha256",
                pattern=_SHA256,
            ),
            build_identity_sha256=_string(
                data["build_identity_sha256"],
                path=f"{path}.build_identity_sha256",
                pattern=_SHA256,
            ),
            supported_route_contracts=_ordered_tokens(
                data["supported_route_contracts"],
                path=f"{path}.supported_route_contracts",
            ),
            capabilities=_ordered_tokens(
                data["capabilities"], path=f"{path}.capabilities"
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "contract_schema_version": self.contract_schema_version,
            "state_schema_version": self.state_schema_version,
            "state_schema_sha256": self.state_schema_sha256,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "checkpoint_schema_sha256": self.checkpoint_schema_sha256,
            "abi_variants": [row.payload() for row in self.abi_variants],
            "abi_set_sha256": self.abi_set_sha256,
            "source_commit": self.source_commit,
            "source_inventory_sha256": self.source_inventory_sha256,
            "build_identity_sha256": self.build_identity_sha256,
            "supported_route_contracts": list(self.supported_route_contracts),
            "capabilities": list(self.capabilities),
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.payload())


def require_identity_equal(
    expected: BinaryIdentityV1 | Mapping[str, Any],
    actual: BinaryIdentityV1 | Mapping[str, Any],
    *,
    path: str,
) -> BinaryIdentityV1:
    """Require exact canonical identity equality and return the admitted value."""

    left = (
        expected
        if isinstance(expected, BinaryIdentityV1)
        else BinaryIdentityV1.from_mapping(expected, path=f"{path}.expected")
    )
    right = (
        actual
        if isinstance(actual, BinaryIdentityV1)
        else BinaryIdentityV1.from_mapping(actual, path=f"{path}.actual")
    )
    if left != right:
        raise fail(
            IDENTITY_MISMATCH_ERROR,
            path,
            f"identity differs: expected={left.identity_sha256} actual={right.identity_sha256}",
        )
    return right
