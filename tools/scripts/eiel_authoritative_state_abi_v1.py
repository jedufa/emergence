#!/usr/bin/env python3
"""Generate and check the canonical EIEL native state ABI order contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT: Final[Path] = (
    ROOT / "_spec/authoritative_state/v1/abi_variants.json"
)
ALLOCATOR: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_direct_resident_source_allocator_v1.py"
)
BINDING: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
)
CONTROLLER_HPP: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1.hpp"
)
TURNOVER_CUH: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_turnover_v1.cuh"
)
PREPARATION: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_direct_resident_preparation_v1.py"
)
OWNER: Final[Path] = (
    ROOT / "engine/perf/cxxbackbone10_resident_state_owner.py"
)
SCHEMA_VERSION: Final[str] = "eiel_authoritative_state_abi_v1"
ORDER_HASH_ALGORITHM: Final[str] = "sha256(canonical-json-array-of-names)"

BASE_PARTS: Final[tuple[str, ...]] = (
    "ORDINARY_BIRTH_IDENTITY_FIELDS",
    "ORDINARY_BIRTH_SCALAR_RNG_FIELDS",
    "ORDINARY_BIRTH_GENOME_INT32_FIELDS",
    "policy_mem_dim",
    "policy_use_gru",
    "ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS",
    "ORDINARY_BIRTH_DRIVE_FIELDS",
    "ORDINARY_BIRTH_POLICY_FIELDS",
    "TURNOVER_STATE_FIELDS",
    "policy_birth_tape_cursor_base",
    "python_random_raw_draw_base",
)


class AbiContractError(ValueError):
    """Raised when an ABI source or observed order violates the contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(names),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _string_tuple_constants(path: Path) -> dict[str, tuple[str, ...]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name):
            continue
        try:
            value = eval(
                compile(ast.Expression(node.value), str(path), "eval"),
                {"__builtins__": {}},
                values,
            )
        except Exception:
            continue
        values[target.id] = value
    return {
        name: tuple(value)
        for name, value in values.items()
        if isinstance(value, tuple)
        and all(isinstance(item, str) for item in value)
    }


def _ordinary_base(constants: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    names: list[str] = []
    for part in BASE_PARTS:
        if part in constants:
            names.extend(constants[part])
        else:
            names.append(part)
    return tuple(names)


def _accepted_ordinary_counts(path: Path) -> tuple[int, ...]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"ordinary_birth_state_tensors\.size\(\)\s*"
        r"==\s*\(lifecycle_state_enabled(?P<body>.*?)"
        r'"ordinary_birth_state_tensors must be empty',
        text,
        re.DOTALL,
    )
    if match is None:
        raise AbiContractError("ordinary-birth binding count contract is absent")
    counts = sorted({int(value) for value in re.findall(r"\b\d{2,3}\b", match.group("body"))})
    if not counts:
        raise AbiContractError("ordinary-birth binding admits no non-empty variants")
    return tuple(counts)


def _require_disabled_variants(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required = (
        "ordinary_birth_state_tensors must be empty when lifecycle is disabled",
        "turnover mutable tensor groups must be empty when disabled",
    )
    missing = [message for message in required if message not in text]
    if missing:
        raise AbiContractError(
            f"disabled ABI contract text is absent: {missing}"
        )


def _diploid_suffix(path: Path, *, suffix_count: int) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"struct FlatOrdinaryBirthState\s*\{(?P<body>.*?)\n\};",
        text,
        re.DOTALL,
    )
    if match is None:
        raise AbiContractError("FlatOrdinaryBirthState is absent")
    body = match.group("body")
    marker = "Present only for the versioned diploid route"
    if marker not in body:
        raise AbiContractError("diploid ABI marker is absent")
    tail = body.split(marker, 1)[1]
    names = list(
        re.findall(
            r"^\s*(?:const\s+)?[A-Za-z0-9_:<>]+\s*\*\s*([a-z][a-z0-9_]*)\s*;",
            tail,
            re.MULTILINE,
        )
    )
    allocator = ALLOCATOR.read_text(encoding="utf-8")
    if (
        suffix_count >= 9
        and "diploid_bank.canonical_hereditary_digest" in allocator
    ):
        names.insert(8, "canonical_hereditary_digest")
    if len(names) > suffix_count:
        names = names[:suffix_count]
    if len(names) < suffix_count:
        raise AbiContractError(
            "diploid suffix cannot be named exactly from current sources"
        )
    return tuple(names)


def _turnover_enum(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"enum TurnoverMutableTensorIndex\s*:[^{]+\{(?P<body>.*?)"
        r"kTurnoverMutableTensorCount\s*,?\s*\};",
        text,
        re.DOTALL,
    )
    if match is None:
        raise AbiContractError("turnover C++ enum is absent")
    return tuple(re.findall(r"\b(k[A-Z][A-Za-z0-9]+)\b", match.group("body")))


def _assigned_list_names(
    path: Path,
    target_name: str,
    *,
    starred: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    starred = dict(starred or {})
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in node.targets
        ) or not isinstance(node.value, ast.List):
            continue
        names: list[str] = []
        for item in node.value.elts:
            if isinstance(item, ast.Starred) and isinstance(item.value, ast.Name):
                try:
                    names.extend(starred[item.value.id])
                except KeyError as exc:
                    raise AbiContractError(
                        f"{target_name}: unsupported starred expansion {item.value.id}"
                    ) from exc
            elif isinstance(item, ast.Subscript):
                key = item.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.append(key.value)
                else:
                    raise AbiContractError(f"{target_name}: dynamic subscript")
            elif isinstance(item, ast.Name):
                names.append(item.id)
            else:
                raise AbiContractError(
                    f"{target_name}: unsupported list expression {ast.dump(item)}"
                )
        return tuple(names)
    raise AbiContractError(f"{target_name}: ordered source list is absent")


def _preparation_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    for path in (ROOT, ROOT / "engine"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
        PATCH_P2_STATE_FIELDS_V1,
    )

    return (
        tuple(DIRECT_RESIDENT_ARGUMENT_NAMES_V1),
        tuple(PATCH_P2_STATE_FIELDS_V1),
    )


def _unsupported_metadata(
    *, family: str, name: str, index: int, occurrence: int
) -> dict[str, Any]:
    field_id = (
        f"abi.{family}.{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        f".occurrence_{occurrence}.v1"
    )
    return {
        "semantic_field_id": field_id,
        "physical_type": "unsupported_not_proven",
        "dtype": "unsupported_not_proven",
        "signedness": "unsupported_not_proven",
        "rank": "unsupported_not_proven",
        "shape_formula": "unsupported_not_proven",
        "valid_row_count_formula": "unsupported_not_proven",
        "capacity_formula": "unsupported_not_proven",
        "mutability": "unsupported_not_proven",
        "semantic_owner": "unsupported_not_proven",
        "optionality": "unsupported_not_proven",
        "checkpoint_participation": "unsupported_not_proven",
        "chronicle_participation": "unsupported_not_proven",
        "profile_availability": ["unsupported_not_proven"],
        "metadata_status": "unsupported_family",
    }


def _scratch_metadata(name: str) -> dict[str, Any]:
    return {
        "semantic_field_id": f"abi.ordinary_birth_scratch.{name}.v1",
        "physical_type": "torch.Tensor",
        "dtype": "uint8",
        "signedness": "unsigned",
        "rank": 2,
        "shape_formula": "[cells,required_lifecycle_scratch_bytes_v1]",
        "valid_row_count_formula": "cells",
        "capacity_formula": "required_lifecycle_scratch_bytes_v1",
        "mutability": "mutable_transaction_scratch",
        "semantic_owner": "flattened_birth_and_turnover_lifecycle",
        "optionality": "required_when_lifecycle_scratch_enabled",
        "checkpoint_participation": "none_freshly_allocated_launch_scratch",
        "chronicle_participation": "none",
        "profile_availability": [
            "ordinary_birth",
            "generation_turnover",
            "full_patch_lifecycle",
        ],
        "metadata_status": "supported_exact",
    }


def _load_exact_contracts() -> dict[str, tuple[dict[str, Any], ...]]:
    modules = (
        (
            "ordinary_turnover_v1.py",
            "build_ordinary_turnover_contracts_v1",
        ),
        (
            "full_tick_event_v1.py",
            "build_full_tick_event_contracts_v1",
        ),
        (
            "outer_output_patch_v1.py",
            "build_outer_output_patch_contracts_v1",
        ),
    )
    contracts: dict[str, tuple[dict[str, Any], ...]] = {}
    root = ROOT / "engine/authoritative_state/contracts"
    for filename, builder_name in modules:
        path = root / filename
        spec = importlib.util.spec_from_file_location(
            f"eiel_authoritative_state_{path.stem}", path
        )
        if spec is None or spec.loader is None:
            raise AbiContractError(f"cannot load exact ABI contract module {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        builder = getattr(module, builder_name, None)
        if not callable(builder):
            raise AbiContractError(f"{path}: {builder_name} is absent")
        built = builder()
        if not isinstance(built, Mapping):
            raise AbiContractError(f"{path}: contract builder returned non-mapping")
        for family, rows in built.items():
            if family in contracts:
                raise AbiContractError(f"duplicate exact ABI family {family}")
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise AbiContractError(f"{family}: exact contract rows are invalid")
            contracts[str(family)] = tuple(dict(row) for row in rows)
    return contracts


def _signedness(dtype: str) -> str:
    if dtype == "bool":
        return "unsigned"
    if dtype.startswith("uint"):
        return "unsigned"
    if dtype.startswith("int"):
        return "signed"
    if dtype.startswith("float"):
        return "ieee754"
    return "not_applicable"


def _bit_width(dtype: str) -> int | str:
    match = re.search(r"(8|16|32|64)$", dtype)
    return int(match.group(1)) if match else 1 if dtype == "bool" else "container"


def _exact_metadata(
    *,
    family: str,
    name: str,
    index: int,
    occurrence: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("name") != name:
        raise AbiContractError(
            f"{family}[{index}]: contract name {contract.get('name')!r} "
            f"differs from ABI name {name!r}"
        )
    if contract.get("metadata_status") != "supported_exact":
        raise AbiContractError(f"{family}[{index}]: contract is not exact")
    dtype = str(contract["dtype"])
    profile = contract.get("profile_availability", ())
    profiles = (
        [part.strip() for part in profile.split(",") if part.strip()]
        if isinstance(profile, str)
        else [str(part) for part in profile]
    )
    field_id = (
        f"abi.{family}.{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}"
        f".occurrence_{occurrence}.v1"
    )
    stale_tail = contract.get(
        "stale_tail_semantics", contract.get("stale_tail_contract")
    )
    provider = contract.get(
        "provider_participation", contract.get("provider_projection")
    )
    analysis = contract.get(
        "analysis_participation", contract.get("analysis_projection")
    )
    boundary = contract.get(
        "boundary", contract.get("boundary_test_profile", "source_bound")
    )
    source = contract.get("source")
    if source is None:
        anchors = contract.get("source_anchors", ())
        source = ";".join(str(anchor) for anchor in anchors)
    required = {
        "stale_tail_policy": stale_tail,
        "provider_participation": provider,
        "analysis_participation": analysis,
        "source_declaration": source,
        "valid_range": contract.get("valid_range"),
        "sentinels": contract.get("sentinels"),
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise AbiContractError(
            f"{family}[{index}]: exact contract lacks {missing}"
        )
    return {
        "semantic_field_id": field_id,
        "physical_type": str(contract.get("kind", "tensor")),
        "dtype": dtype,
        "signedness": _signedness(dtype),
        "bit_width": _bit_width(dtype),
        "valid_range": contract["valid_range"],
        "sentinels": list(contract["sentinels"]),
        "rank": contract.get("rank"),
        "shape_formula": str(contract["shape_formula"]),
        "valid_row_count_formula": str(contract["valid_row_count_formula"]),
        "capacity_formula": str(contract["capacity_formula"]),
        "mutability": str(contract["mutability"]),
        "stale_tail_policy": str(stale_tail),
        "semantic_owner": str(contract["semantic_owner"]),
        "optionality": "required_for_declared_variant",
        "checkpoint_participation": str(contract["checkpoint_participation"]),
        "chronicle_participation": str(contract["chronicle_participation"]),
        "provider_participation": str(provider),
        "analysis_participation": str(analysis),
        "profile_availability": profiles,
        "boundary_test_profile": str(boundary),
        "source_declaration": str(source),
        "metadata_status": "supported_exact",
        "authority_contract": json.loads(
            json.dumps(dict(contract), allow_nan=False, sort_keys=True)
        ),
    }


def _variant(
    *,
    variant_id: str,
    family: str,
    names: Sequence[str],
    groups: Sequence[str],
    cpp_enum_names: Sequence[str] | None = None,
    supported: bool = False,
    contract_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if len(names) != len(groups):
        raise AbiContractError(f"{variant_id}: group expansion length disagrees")
    if contract_rows is not None and len(contract_rows) != len(names):
        raise AbiContractError(
            f"{variant_id}: exact contract count {len(contract_rows)} "
            f"differs from ABI count {len(names)}"
        )
    entries = []
    occurrences: dict[str, int] = {}
    for index, (name, group) in enumerate(zip(names, groups, strict=True)):
        occurrences[name] = occurrences.get(name, 0) + 1
        entry: dict[str, Any] = {
            "index": index,
            "name": name,
            "expansion_group": group,
        }
        if cpp_enum_names is not None:
            entry["cpp_enum_name"] = cpp_enum_names[index]
        if names:
            entry.update(
                _exact_metadata(
                    family=family,
                    name=name,
                    index=index,
                    occurrence=occurrences[name],
                    contract=contract_rows[index],
                )
                if contract_rows is not None
                else _scratch_metadata(name)
                if supported and family == "ordinary_birth_scratch"
                else _unsupported_metadata(
                    family=family,
                    name=name,
                    index=index,
                    occurrence=occurrences[name],
                )
            )
        entries.append(entry)
    return {
        "variant_id": variant_id,
        "abi_family": family,
        "count": len(names),
        "order_hash_algorithm": ORDER_HASH_ALGORITHM,
        "order_sha256": order_sha256(names),
        "support_status": (
            "supported_exact_metadata"
            if supported or contract_rows is not None or not names
            else "unsupported_metadata_incomplete"
        ),
        "entries": entries,
    }


def build_contract() -> dict[str, Any]:
    constants = _string_tuple_constants(ALLOCATOR)
    _require_disabled_variants(BINDING)
    base = _ordinary_base(constants)
    if len(base) < 1:
        raise AbiContractError("ordinary-birth base is empty")
    accepted = _accepted_ordinary_counts(BINDING)
    if len(base) not in accepted:
        raise AbiContractError(
            f"allocator base count {len(base)} is not binding-admitted {accepted}"
        )
    extra_counts = tuple(count for count in accepted if count != len(base))
    if not extra_counts:
        raise AbiContractError(
            f"expected at least one non-legacy ordinary-birth variant, observed {accepted}"
        )
    suffixes = {
        count: _diploid_suffix(
            CONTROLLER_HPP, suffix_count=count - len(base)
        )
        for count in extra_counts
    }

    group_sizes = [
        (part, len(constants[part]) if part in constants else 1)
        for part in BASE_PARTS
    ]
    base_groups = tuple(
        group
        for group, size in group_sizes
        for _ in range(size)
    )
    turnover = constants.get("TURNOVER_MUTABLE_STATE_FIELDS")
    if turnover is None:
        raise AbiContractError("TURNOVER_MUTABLE_STATE_FIELDS is absent")
    turnover_enum = _turnover_enum(TURNOVER_CUH)
    if len(turnover) != len(turnover_enum):
        raise AbiContractError(
            "Python turnover list and C++ turnover enum counts disagree"
        )

    owner_constants = _string_tuple_constants(OWNER)
    event_columns = (
        owner_constants["EVENT_TAPE_TENSOR_NAMES"]
        + owner_constants["TYPED_LIFECYCLE_EVENT_TENSOR_NAMES"]
    )
    event_names = _assigned_list_names(
        ALLOCATOR,
        "event_state_tensors",
        starred={"event_columns": event_columns},
    )
    full_tick_names = _assigned_list_names(ALLOCATOR, "full_tick_state")
    output_names = constants["ORDINARY_BIRTH_OUTPUT_FIELDS"]
    turnover_output_names = constants["TURNOVER_OUTPUT_FIELDS"]
    outer_names, patch_names = _preparation_names()
    exact_contracts = _load_exact_contracts()
    if (
        len(full_tick_names) != 29
        or len(event_names) != 31
        or len(output_names) != 63
        or len(turnover_output_names) != 16
        or len(patch_names) != 10
        or len(outer_names) != 172
    ):
        raise AbiContractError("one or more ordered production ABI counts changed")

    variants = [
        _variant(
            variant_id="ordinary_birth_disabled_0",
            family="ordinary_birth_state",
            names=(),
            groups=(),
        ),
        _variant(
            variant_id=f"ordinary_birth_legacy_{len(base)}",
            family="ordinary_birth_state",
            names=base,
            groups=base_groups,
            contract_rows=exact_contracts["ordinary_birth_state"][: len(base)],
        ),
        *(
            _variant(
                variant_id=(
                    f"ordinary_birth_diploid_{count}"
                    if count == min(extra_counts)
                    else f"ordinary_birth_genome_v2_{count}"
                ),
                family="ordinary_birth_state",
                names=base + suffixes[count],
                groups=base_groups
                + (
                    ("DIPLOID_SUFFIX",) * 9
                    + ("GENOME_UPGRADE_V2_SUFFIX",) * (len(suffixes[count]) - 9)
                    if len(suffixes[count]) >= 9
                    else ("DIPLOID_SUFFIX",) * len(suffixes[count])
                ),
                contract_rows=exact_contracts["ordinary_birth_state"][:count],
            )
            for count in extra_counts
        ),
        _variant(
            variant_id="turnover_disabled_0",
            family="turnover_mutable_state",
            names=(),
            groups=(),
            cpp_enum_names=(),
        ),
        _variant(
            variant_id=f"turnover_mutable_state_{len(turnover)}",
            family="turnover_mutable_state",
            names=turnover,
            groups=("TURNOVER_MUTABLE_STATE_FIELDS",) * len(turnover),
            cpp_enum_names=turnover_enum,
            contract_rows=exact_contracts["turnover_mutable_state"],
        ),
        _variant(
            variant_id="full_tick_state_29",
            family="full_tick_state",
            names=full_tick_names,
            groups=("full_tick_state",) * len(full_tick_names),
            contract_rows=exact_contracts["full_tick_state"],
        ),
        _variant(
            variant_id="event_state_31",
            family="event_state",
            names=event_names,
            groups=("event_state",) * len(event_names),
            contract_rows=exact_contracts["event_state"],
        ),
        _variant(
            variant_id="ordinary_birth_output_disabled_0",
            family="ordinary_birth_output",
            names=(),
            groups=(),
        ),
        _variant(
            variant_id=f"ordinary_birth_output_{len(output_names)}",
            family="ordinary_birth_output",
            names=output_names,
            groups=("ORDINARY_BIRTH_OUTPUT_FIELDS",) * len(output_names),
            contract_rows=exact_contracts["ordinary_birth_output"],
        ),
        _variant(
            variant_id="turnover_output_disabled_0",
            family="turnover_output",
            names=(),
            groups=(),
        ),
        _variant(
            variant_id=f"turnover_output_{len(turnover_output_names)}",
            family="turnover_output",
            names=turnover_output_names,
            groups=("TURNOVER_OUTPUT_FIELDS",) * len(turnover_output_names),
            contract_rows=exact_contracts["turnover_output"],
        ),
        _variant(
            variant_id="ordinary_birth_scratch_disabled_0",
            family="ordinary_birth_scratch",
            names=(),
            groups=(),
            supported=True,
        ),
        _variant(
            variant_id="ordinary_birth_scratch_1",
            family="ordinary_birth_scratch",
            names=("lifecycle_scratch_storage",),
            groups=("ordinary_birth_scratch_tensors",),
            contract_rows=exact_contracts["ordinary_birth_scratch"],
        ),
        _variant(
            variant_id="patch_p2_state_disabled_0",
            family="patch_p2_state",
            names=(),
            groups=(),
        ),
        _variant(
            variant_id="patch_p2_state_context_7",
            family="patch_p2_state",
            names=patch_names[:7],
            groups=("PATCH_P2_STATE_FIELDS_V1",) * 7,
            contract_rows=exact_contracts["patch_p2_state"][:7],
        ),
        _variant(
            variant_id="patch_p2_state_full_10",
            family="patch_p2_state",
            names=patch_names,
            groups=("PATCH_P2_STATE_FIELDS_V1",) * 10,
            contract_rows=exact_contracts["patch_p2_state"],
        ),
        _variant(
            variant_id=f"direct_resident_outer_{len(outer_names)}",
            family="direct_resident_outer",
            names=outer_names,
            groups=("DIRECT_RESIDENT_ARGUMENT_NAMES_V1",) * len(outer_names),
            contract_rows=exact_contracts["direct_resident_outer"],
        ),
    ]
    unsupported_families: list[dict[str, str]] = []
    source_paths = (
        ALLOCATOR,
        BINDING,
        CONTROLLER_HPP,
        TURNOVER_CUH,
        PREPARATION,
        OWNER,
        *(
            ROOT / "engine/authoritative_state/contracts" / filename
            for filename in (
                "ordinary_turnover_v1.py",
                "full_tick_event_v1.py",
                "outer_output_patch_v1.py",
            )
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "canonical_supported_abi_variants",
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "unsupported_families": unsupported_families,
        "variants": variants,
    }
    payload["contract_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def validate_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AbiContractError("ABI contract must be a JSON object")
    observed = dict(value)
    expected = build_contract()
    if observed != expected:
        raise AbiContractError(
            "ABI contract differs from the deterministic current-source contract"
        )
    return observed


def validate_order(
    contract: Mapping[str, Any],
    *,
    family: str,
    names: Sequence[str],
) -> str:
    candidates = [
        item
        for item in contract.get("variants", [])
        if item.get("abi_family") == family
    ]
    same_count = [item for item in candidates if item.get("count") == len(names)]
    if not same_count:
        supported = sorted({int(item["count"]) for item in candidates})
        raise AbiContractError(
            f"{family}: unsupported count {len(names)}; supported counts={supported}"
        )
    observed_hash = order_sha256(names)
    for item in same_count:
        expected_names = [entry["name"] for entry in item["entries"]]
        if list(names) == expected_names and observed_hash == item["order_sha256"]:
            return str(item["variant_id"])
    expected = ", ".join(str(item["order_sha256"]) for item in same_count)
    raise AbiContractError(
        f"{family}: count {len(names)} is known but order hash {observed_hash} "
        f"does not match canonical hash(es) {expected}"
    )


def validate_sequence(
    contract: Mapping[str, Any],
    *,
    family: str,
    names: Sequence[str],
) -> str:
    unsupported = {
        item["abi_family"]: item["reason"]
        for item in contract.get("unsupported_families", [])
    }
    if family in unsupported:
        raise AbiContractError(
            f"{family}: unsupported production ABI family: {unsupported[family]}"
        )
    return validate_order(contract, family=family, names=names)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check = subparsers.add_parser("check")
    check.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT)
    sequence = subparsers.add_parser("check-sequence")
    sequence.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT)
    sequence.add_argument("--family", required=True)
    sequence.add_argument("--names-file", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "generate":
        value = build_contract()
        _write(args.output, value)
        print(json.dumps({
            "contract_sha256": value["contract_sha256"],
            "output": str(args.output),
            "variants": {
                item["variant_id"]: item["order_sha256"]
                for item in value["variants"]
            },
        }, sort_keys=True))
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    if args.command == "check":
        print(json.dumps({
            "contract_sha256": contract["contract_sha256"],
            "ok": True,
        }, sort_keys=True))
        return
    names_value = json.loads(args.names_file.read_text(encoding="utf-8"))
    if not isinstance(names_value, list) or not all(
        isinstance(item, str) for item in names_value
    ):
        raise AbiContractError("names file must be a JSON array of strings")
    variant_id = validate_sequence(
        contract, family=args.family, names=names_value
    )
    print(json.dumps({"ok": True, "variant_id": variant_id}, sort_keys=True))


if __name__ == "__main__":
    main()
