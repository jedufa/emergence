"""Exact contracts for the ordinary-birth and turnover mutable tensor ABIs.

The positional order is owned by the source allocator.  Most resident tensor
shapes, semantic owners, and checkpoint modes are owned by the accepted
device-neutral state-bank manifest.  Runtime-dependent widths are expressed
symbolically and can be reconciled against the pre-dispatch prepared-owner
receipt without launching a native tick.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    ORDINARY_BIRTH_DRIVE_FIELDS,
    ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
    ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    ORDINARY_BIRTH_IDENTITY_FIELDS,
    ORDINARY_BIRTH_POLICY_FIELDS,
    ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
    TURNOVER_MUTABLE_STATE_FIELDS,
    TURNOVER_STATE_FIELDS,
)
from engine.perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
)
from engine.perf.cuda_architecture_b_state_bank import (
    load_accepted_state_bank_manifest,
)


ORDINARY_BIRTH_LEGACY_COUNT_V1: Final[int] = 113
ORDINARY_BIRTH_DIPLOID_COUNT_V1: Final[int] = 122
ORDINARY_BIRTH_GENOME_UPGRADE_V2_COUNT_V1: Final[int] = 139
TURNOVER_MUTABLE_COUNT_V1: Final[int] = 122

_ALLOCATOR_ANCHOR = (
    "engine/perf/"
    "cuda_architecture_b_direct_resident_source_allocator_v1.py:2221-2659"
)
_BINDING_ANCHOR = (
    "engine/perf/"
    "cuda_architecture_b_flattened_controller_v1_binding.cpp:638-969,"
    "2564-2634,3053-3151,3575-3608"
)
_TURNOVER_ANCHOR = (
    "engine/perf/cuda_architecture_b_flattened_turnover_v1.cuh:20-230;"
    "engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh:"
    "180-436,560-579,1068-1089,2143-2209"
)
_CHECKPOINT_ANCHOR = (
    "engine/perf/cuda_architecture_b_checkpoint.py:"
    "601-650,1196-1245,1360-1400,1560-1623"
)
_DIPLOID_ANCHOR = (
    "engine/perf/cuda_diploid_state_bank_v1.py:177-224,285-316"
)
_CHRONICLE_ANCHOR = (
    "simulation/chronicle/recorder.py:1255-1290;"
    "engine/native_runtime/compact_summary_authority_adapter.py:661"
)

_DIPLOID_SUFFIX: Final[tuple[str, ...]] = (
    "inheritance_mode",
    "inheritance_mutation_enabled",
    "resource_handling_enabled",
    "mutation_sigma_multiplier",
    "diploid_allele_storage",
    "genome_resource_handling_axis",
    "diploid_allele_shadow_storage",
    "genome_resource_handling_axis_shadow",
    "canonical_hereditary_digest",
)
_GENOME_UPGRADE_V2_SUFFIX: Final[tuple[str, ...]] = (
    "genome_upgrade_v2_enabled",
    "genome_upgrade_v2_mutation_rate_q10",
    "genome_upgrade_v2_mutation_step_q10",
    "genome_upgrade_v2_alleles_q10",
    "genome_upgrade_v2_modifiers_q10",
    "genome_upgrade_v2_products_q10",
    "genome_upgrade_v2_phenotypes_q10",
    "genome_upgrade_v2_expression_cost_q10",
    "genome_upgrade_v2_active_product_count",
    "genome_upgrade_v2_saturated_product_count",
    "genome_upgrade_v2_alleles_q10_shadow",
    "genome_upgrade_v2_modifiers_q10_shadow",
    "genome_upgrade_v2_products_q10_shadow",
    "genome_upgrade_v2_phenotypes_q10_shadow",
    "genome_upgrade_v2_expression_cost_q10_shadow",
    "genome_upgrade_v2_active_product_count_shadow",
    "genome_upgrade_v2_saturated_product_count_shadow",
)

_DTYPE_TRAITS: Final[dict[str, tuple[str, int]]] = {
    "bool": ("boolean", 8),
    "float32": ("not_applicable", 32),
    "float64": ("not_applicable", 64),
    "int32": ("signed", 32),
    "int64": ("signed", 64),
    "uint8": ("unsigned", 8),
    "uint32": ("unsigned", 32),
    "uint64": ("unsigned", 64),
}

# Fields not present in the accepted state-bank manifest.  These shapes are
# independently enforced by the allocator, packed-state runtime templates, or
# the native binding.
_SUPPLEMENT: Final[dict[str, tuple[str, tuple[str | int, ...], str, str]]] = {
    "agent_id_to_slot": (
        "int32",
        ("cell", "capacity"),
        "resident population identity and slot owner",
        "bounded ABI compatibility shadow; compact identity pairs are checkpoint authority",
    ),
    "active_agent_ids": (
        "int64",
        ("cell", "capacity"),
        "resident population identity and slot owner",
        "derived canonical alive-order identity projection",
    ),
    "active_agent_slots": (
        "int32",
        ("cell", "capacity"),
        "resident population identity and slot owner",
        "derived capacity-bounded companion to active_agent_ids",
    ),
    "active_count_state": (
        "int64",
        ("cell",),
        "resident population identity and slot owner",
        "derived from canonical live identity rows",
    ),
    "pending_free_agent_ids": (
        "int64",
        ("cell", "capacity"),
        "generation-turnover transaction owner",
        "runtime-prefix pending free identity list",
    ),
    "pending_free_slots": (
        "int64",
        ("cell", "capacity"),
        "generation-turnover transaction owner",
        "runtime-prefix pending free slot list",
    ),
    "pending_free_count": (
        "int64",
        ("cell",),
        "generation-turnover transaction owner",
        "derived length of runtime-prefix pending free lists",
    ),
    "slot_free_queue_count": (
        "int64",
        ("cell",),
        "resident population identity and slot owner",
        "encoded with canonical free-slot queue",
    ),
    "population_next_id": (
        "int64",
        ("cell", 1),
        "resident population identity and slot owner",
        "canonical population identity scalar",
    ),
    "python_random_words": (
        "uint32",
        ("cell", 624),
        "Python random-stream owner",
        "structured python_random_state_v1",
    ),
    "python_random_index": (
        "int64",
        ("cell", 1),
        "Python random-stream owner",
        "structured python_random_state_v1",
    ),
    "python_random_has_gauss": (
        "bool",
        ("cell", 1),
        "Python random-stream owner",
        "structured python_random_state_v1",
    ),
    "python_random_gauss_next": (
        "float64",
        ("cell", 1),
        "Python random-stream owner",
        "structured python_random_state_v1",
    ),
    "python_random_gauss_primary_tape": (
        "float64",
        ("cell", "cpython_gauss_tape_capacity"),
        "Python random-stream owner",
        "regenerated runtime tape; counter state checkpointed",
    ),
    "python_random_gauss_secondary_tape": (
        "float64",
        ("cell", "cpython_gauss_tape_capacity"),
        "Python random-stream owner",
        "regenerated runtime tape; counter state checkpointed",
    ),
    "python_random_raw_draws": (
        "int64",
        ("cell", 1),
        "Python random-stream owner",
        "structured python_random_state_v1",
    ),
    "mutation_py_semantic_draws": (
        "int64",
        ("cell", 1),
        "mutation random-stream owner",
        "canonical semantic draw counter",
    ),
    "pred_lr": (
        "float32",
        ("cell", "capacity"),
        "resident organism state owner",
        "included exact tensor clone",
    ),
    "population_pcg64_state_words": (
        "uint64",
        ("cell", 4),
        "population random-stream owner",
        "structured PCG64 state",
    ),
    "population_pcg64_buffer_words": (
        "int64",
        ("cell", 2),
        "population random-stream owner",
        "structured PCG64 buffered word state",
    ),
    "mutation_np_semantic_draws": (
        "int64",
        ("cell", 1),
        "mutation random-stream owner",
        "canonical semantic draw counter",
    ),
    "policy_birth_delta_tape": (
        "float32",
        ("cell", "ordinary_birth_capacity", "mlp_pack_width"),
        "mutation random-stream owner",
        "regenerated runtime tape; counter and RNG state checkpointed",
    ),
    "policy_birth_mutation_mask_tape": (
        "bool",
        ("cell", "ordinary_birth_capacity", "mlp_pack_width"),
        "mutation random-stream owner",
        "regenerated runtime tape; counter and RNG state checkpointed",
    ),
    "policy_birth_crossover_mask_packed_tape": (
        "uint8",
        ("cell", "ordinary_birth_capacity", "policy_crossover_packed_row_bytes"),
        "mutation random-stream owner",
        "bitpack_lsb0_v1 regenerated runtime tape; counter and RNG state checkpointed",
    ),
    "policy_birth_post_rng_state_packed_tape": (
        "uint8",
        ("cell", "ordinary_birth_capacity", TORCH_CPU_RNG_STATE_PACKED_BYTES_V1),
        "mutation random-stream owner",
        "torch_cpu_rng_state_5056_u32_middle_v1 regenerated runtime tape; terminal canonical RNG state checkpointed",
    ),
    "policy_birth_semantic_draws_tape": (
        "int64",
        ("cell", "ordinary_birth_capacity"),
        "mutation random-stream owner",
        "regenerated runtime tape; semantic draw counter checkpointed",
    ),
    "policy_birth_tape_cursor": (
        "int64",
        ("cell", 1),
        "mutation random-stream owner",
        "canonical policy-birth tape cursor",
    ),
    "mutation_torch_cpu_semantic_draws": (
        "int64",
        ("cell", 1),
        "mutation random-stream owner",
        "canonical semantic draw counter",
    ),
    "chemistry_observation_fidelity": (
        "float32",
        ("cell", "capacity"),
        "resident organism state owner",
        "included exact tensor clone",
    ),
    "chemistry_observation_bias": (
        "float32",
        ("cell", "capacity"),
        "resident organism state owner",
        "included exact tensor clone",
    ),
    "pcg64_state_words": (
        "uint64",
        ("cell", 4),
        "shared environment-policy random-stream owner",
        "structured PCG64 state",
    ),
    "pcg64_buffer_words": (
        "int64",
        ("cell", 2),
        "shared environment-policy random-stream owner",
        "structured PCG64 buffered word state",
    ),
    "policy_np_semantic_draws": (
        "int64",
        ("cell",),
        "policy random-stream owner",
        "canonical semantic draw counter",
    ),
    "policy_birth_tape_cursor_base": (
        "int64",
        ("cell",),
        "prepared direct-resident source owner",
        "not checkpointed separately; derived from checkpointed cursor",
    ),
    "python_random_raw_draw_base": (
        "int64",
        ("cell",),
        "prepared direct-resident source owner",
        "not checkpointed separately; derived from checkpointed counter",
    ),
}

_DIPLOID: Final[dict[str, tuple[str, tuple[str | int, ...], str]]] = {
    "inheritance_mode": ("int32", ("cell",), "configuration scalar"),
    "inheritance_mutation_enabled": ("bool", ("cell",), "configuration scalar"),
    "resource_handling_enabled": ("bool", ("cell",), "configuration scalar"),
    "mutation_sigma_multiplier": ("float64", ("cell",), "configuration scalar"),
    "diploid_allele_storage": (
        "float32",
        ("cell", 34, "capacity"),
        "included as 34 named diploid allele planes",
    ),
    "genome_resource_handling_axis": (
        "float32",
        ("cell", "capacity"),
        "included exact diploid resource axis",
    ),
    "diploid_allele_shadow_storage": (
        "float32",
        ("cell", 34, "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_resource_handling_axis_shadow": (
        "float32",
        ("cell", "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "canonical_hereditary_digest": (
        "uint8",
        ("cell", "capacity", 32),
        "included exact canonical hereditary digest",
    ),
}
_GENOME_UPGRADE_V2: Final[
    dict[str, tuple[str, tuple[str | int, ...], str]]
] = {
    "genome_upgrade_v2_enabled": (
        "bool",
        ("cell",),
        "configuration scalar included in the checkpoint envelope",
    ),
    "genome_upgrade_v2_mutation_rate_q10": (
        "int64",
        ("cell",),
        "configuration scalar included in the checkpoint envelope",
    ),
    "genome_upgrade_v2_mutation_step_q10": (
        "int64",
        ("cell",),
        "configuration scalar included in the checkpoint envelope",
    ),
    "genome_upgrade_v2_alleles_q10": (
        "int64",
        ("cell", 2, 12, 4, "capacity"),
        "included exact fixed-Q10 hereditary plane",
    ),
    "genome_upgrade_v2_modifiers_q10": (
        "int64",
        ("cell", 2, 12, "capacity"),
        "included exact fixed-Q10 hereditary plane",
    ),
    "genome_upgrade_v2_products_q10": (
        "int64",
        ("cell", 12, "capacity"),
        "included exact fixed-Q10 expressed product plane",
    ),
    "genome_upgrade_v2_phenotypes_q10": (
        "int64",
        ("cell", 10, "capacity"),
        "included exact fixed-Q10 expressed phenotype plane",
    ),
    "genome_upgrade_v2_expression_cost_q10": (
        "int64",
        ("cell", "capacity"),
        "included exact fixed-Q10 expression cost plane",
    ),
    "genome_upgrade_v2_active_product_count": (
        "int64",
        ("cell", "capacity"),
        "included exact expression telemetry plane",
    ),
    "genome_upgrade_v2_saturated_product_count": (
        "int64",
        ("cell", "capacity"),
        "included exact expression telemetry plane",
    ),
    "genome_upgrade_v2_alleles_q10_shadow": (
        "int64",
        ("cell", 2, 12, 4, "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_upgrade_v2_modifiers_q10_shadow": (
        "int64",
        ("cell", 2, 12, "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_upgrade_v2_products_q10_shadow": (
        "int64",
        ("cell", 12, "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_upgrade_v2_phenotypes_q10_shadow": (
        "int64",
        ("cell", 10, "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_upgrade_v2_expression_cost_q10_shadow": (
        "int64",
        ("cell", "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_upgrade_v2_active_product_count_shadow": (
        "int64",
        ("cell", "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
    "genome_upgrade_v2_saturated_product_count_shadow": (
        "int64",
        ("cell", "capacity"),
        "transaction shadow; not checkpoint authority",
    ),
}


def ordinary_birth_state_names_v1(count: int) -> tuple[str, ...]:
    legacy = (
        *ORDINARY_BIRTH_IDENTITY_FIELDS,
        *ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
        *ORDINARY_BIRTH_GENOME_INT32_FIELDS,
        "policy_mem_dim",
        "policy_use_gru",
        *ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
        *ORDINARY_BIRTH_DRIVE_FIELDS,
        *ORDINARY_BIRTH_POLICY_FIELDS,
        *TURNOVER_STATE_FIELDS,
        "policy_birth_tape_cursor_base",
        "python_random_raw_draw_base",
    )
    if len(legacy) != ORDINARY_BIRTH_LEGACY_COUNT_V1:
        raise RuntimeError("ordinary-birth legacy ABI source count drifted")
    if count == ORDINARY_BIRTH_LEGACY_COUNT_V1:
        return legacy
    if count == ORDINARY_BIRTH_DIPLOID_COUNT_V1:
        return (*legacy, *_DIPLOID_SUFFIX)
    if count == ORDINARY_BIRTH_GENOME_UPGRADE_V2_COUNT_V1:
        return (*legacy, *_DIPLOID_SUFFIX, *_GENOME_UPGRADE_V2_SUFFIX)
    raise ValueError("ordinary_variant_count must be exactly 113, 122, or 139")


def _manifest_contracts() -> dict[str, Any]:
    return {
        field.name: field
        for field in load_accepted_state_bank_manifest().fields
    }


def _shape_formula(shape: Sequence[str | int]) -> str:
    return "[" + ",".join(str(value) for value in shape) + "]"


def _valid_formula(name: str, shape: Sequence[str | int]) -> str:
    if name == "active_count_state":
        return "cell"
    if name == "agent_id_to_slot":
        return "count(value>=0)"
    if name in {"slot_free_queue", "pending_free_agent_ids", "pending_free_slots"}:
        counter = (
            "slot_free_queue_count_by_cell"
            if name == "slot_free_queue"
            else "pending_free_count_by_cell"
        )
        return f"sum({counter})"
    if "capacity" in shape and name != "agent_grid":
        trailing = tuple(shape[shape.index("capacity") + 1 :])
        suffix = (
            " * " + " * ".join(str(value) for value in trailing)
            if trailing
            else ""
        )
        return f"sum(active_count_by_cell){suffix}"
    return "product(shape)"


def _tail_contract(name: str, shape: Sequence[str | int]) -> str:
    if name == "agent_id_to_slot":
        return "negative sentinel is invalid; nonnegative entries are valid"
    if name == "slot_free_queue":
        return "queue prefix is valid; tail is ignored and may be stale"
    if name in {"pending_free_agent_ids", "pending_free_slots"}:
        return "pending-free prefix is valid; tail is ignored and may be stale"
    if "capacity" in shape and name != "agent_grid":
        return (
            "non-live capacity rows are preserved unless reused; newborn rows "
            "are initialized before commit"
        )
    return "full storage participates; no stale tail"


def _valid_range(name: str, dtype: str) -> str:
    if dtype == "bool":
        return "{false,true}"
    if name == "active_count_state":
        return "[0,capacity]"
    if name in {"pos_y"}:
        return "-1 sentinel or [0,height)"
    if name in {"pos_x"}:
        return "-1 sentinel or [0,width)"
    if name in {
        "agent_id_to_slot",
        "active_agent_slots",
        "pending_free_slots",
        "slot_free_queue",
    }:
        return "-1 sentinel or [0,capacity)"
    if name == "agent_grid":
        return "-1 sentinel or nonnegative agent identity"
    if name == "python_random_index":
        return "[0,624]"
    if dtype.startswith("uint"):
        return f"[0,2^{_DTYPE_TRAITS[dtype][1]}-1]"
    if dtype.startswith("float"):
        return "finite on valid rows; semantic bounds owned by producer"
    return "signed dtype domain; semantic bounds owned by producer"


def _sentinels(name: str) -> tuple[int | str, ...]:
    if (
        name in {
            "agent_grid",
            "agent_id_to_slot",
            "slot_to_agent_id",
            "active_agent_ids",
            "active_agent_slots",
            "pending_free_agent_ids",
            "pending_free_slots",
            "slot_free_queue",
            "action_cache_actions",
            "action_cache_token",
            "action_cache_cursor",
            "action_cache_next",
            "last_action",
            "last_last_action",
            "first_repro_step",
            "lineage_founder_id",
            "patch_habitat_lineage_id",
        }
        or name.endswith("_agent_ids")
        or name.endswith("_slots")
    ):
        return (-1,)
    return ("none_documented_for_valid_rows",)


def _base_row(
    *,
    family: str,
    index: int,
    name: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = manifest.get(name)
    diploid = _DIPLOID.get(name)
    genome_upgrade_v2 = _GENOME_UPGRADE_V2.get(name)
    if genome_upgrade_v2 is not None:
        dtype, shape, checkpoint = genome_upgrade_v2
        owner = "CUDA GENOME-UPGRADE-V2 fixed-Q10 state-bank owner"
        anchor = (
            "engine/perf/cuda_genome_upgrade_v2_state_bank.py;"
            "engine/perf/genome_upgrade_v2_q10_v1.cuh"
        )
    elif diploid is not None:
        dtype, shape, checkpoint = diploid
        owner = "CUDA diploid state-bank owner"
        anchor = _DIPLOID_ANCHOR
    elif accepted is not None:
        dtype = accepted.dtype_name
        shape = accepted.device_neutral_shape_variables
        checkpoint = accepted.checkpoint_mode
        owner = accepted.semantic_owner
        anchor = (
            "engine/perf/flat_patch_device_neutral_state_manifest_v4.json;"
            + _ALLOCATOR_ANCHOR
        )
    else:
        try:
            dtype, shape, owner, checkpoint = _SUPPLEMENT[name]
        except KeyError as exc:
            raise RuntimeError(f"no exact physical contract for {family}.{name}") from exc
        anchor = _ALLOCATOR_ANCHOR
    signedness, bit_width = _DTYPE_TRAITS[dtype]
    shape = tuple(shape)
    mutability = (
        "transactional_source_read_write"
        if family == "turnover_mutable_state"
        else "resident_source_read_write"
    )
    if name in {
        "diploid_allele_shadow_storage",
        "genome_resource_handling_axis_shadow",
    } or name.endswith("_shadow"):
        mutability = "transaction_shadow_write_then_commit"
    return {
        "index": index,
        "name": name,
        "metadata_status": "supported_exact",
        "dtype": dtype,
        "physical_type": dtype,
        "signedness": signedness,
        "bit_width": bit_width,
        "valid_range": _valid_range(name, dtype),
        "sentinels": _sentinels(name),
        "rank": len(shape),
        "shape_formula": _shape_formula(shape),
        "shape_dimensions": shape,
        "capacity_formula": "product(shape)",
        "valid_row_count_formula": _valid_formula(name, shape),
        "mutability": mutability,
        "stale_tail_contract": _tail_contract(name, shape),
        "checkpoint_participation": checkpoint,
        "chronicle_participation": (
            "no direct positional projection; selected semantics may contribute "
            "to generation summaries"
        ),
        "provider_projection": "no direct positional ABI projection",
        "analysis_projection": "no direct positional ABI projection",
        "profile_availability": (
            "patch_fixed_mlp_direct_resident_v1",
            "patch_gru_direct_resident_v1",
        ),
        "semantic_owner": owner,
        "source_anchors": (
            anchor,
            _BINDING_ANCHOR,
            _CHECKPOINT_ANCHOR,
            _CHRONICLE_ANCHOR,
        ),
        "boundary_test_profile": (
            "prepared_owner_pre_dispatch_exact_v1"
            if diploid is None and genome_upgrade_v2 is None
            else (
                "genome_upgrade_v2_state_bank_exact_v1"
                if genome_upgrade_v2 is not None
                else "diploid_state_bank_exact_v1"
            )
        ),
    }


def _receipt_rows(
    receipt: Mapping[str, Any] | None,
    prefix: str,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    if receipt is None:
        return {}
    tensors = receipt.get("tensors")
    if not isinstance(tensors, Sequence) or isinstance(tensors, (str, bytes)):
        raise ValueError("prepared-owner receipt tensors must be a sequence")
    marker = prefix + "."
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in tensors:
        if not isinstance(row, Mapping):
            continue
        path = str(row.get("path", ""))
        if path.startswith(marker):
            grouped.setdefault(path[len(marker) :], []).append(row)
    return {name: tuple(rows) for name, rows in grouped.items()}


def _observed_occurrence(
    observed: Mapping[str, tuple[Mapping[str, Any], ...]],
    counters: dict[str, int],
    name: str,
) -> Mapping[str, Any] | None:
    occurrence = counters.get(name, 0)
    counters[name] = occurrence + 1
    rows = observed.get(name, ())
    return rows[occurrence] if occurrence < len(rows) else None


def _reconcile_row(
    row: dict[str, Any],
    observed: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if observed is None:
        row["prepared_owner_reconciliation"] = "symbolic_profile_contract"
        return row, ()
    problems: list[str] = []
    dtype = observed.get("dtype")
    rank = observed.get("rank")
    shape = observed.get("shape")
    if dtype != row["dtype"]:
        problems.append(f"dtype expected={row['dtype']} observed={dtype}")
    if rank != row["rank"]:
        problems.append(f"rank expected={row['rank']} observed={rank}")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != row["rank"]
    ):
        problems.append(f"shape/rank invalid observed={shape}")
    row["prepared_owner_observation"] = {
        "dtype": dtype,
        "rank": rank,
        "shape": list(shape) if isinstance(shape, Sequence) else shape,
        "capacity_count": observed.get("capacity_count"),
        "valid_count": observed.get("valid_count"),
        "valid_count_semantics": observed.get("valid_count_semantics"),
        "mutability": observed.get("mutability"),
    }
    row["prepared_owner_reconciliation"] = (
        "exact_match" if not problems else "contradiction"
    )
    return row, tuple(problems)


def build_ordinary_turnover_contracts_v1(
    receipt: Mapping[str, Any] | None = None,
    *,
    ordinary_variant_count: int = ORDINARY_BIRTH_GENOME_UPGRADE_V2_COUNT_V1,
    turnover_variant_count: int = TURNOVER_MUTABLE_COUNT_V1,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Build exact ordered contracts, optionally bound to a prepared owner.

    A legacy 113 or diploid 122 receipt may be used while requesting the 139
    variant: absent suffix rows remain exact from their state-bank boundaries
    and are intentionally not fabricated as receipt observations.
    """

    manifest = _manifest_contracts()
    ordinary_observed = _receipt_rows(receipt, "ordinary_birth_state_tensors")
    turnover_observed = _receipt_rows(receipt, "turnover_state_tensors")
    ordinary_occurrences: dict[str, int] = {}
    ordinary = []
    for index, name in enumerate(
        ordinary_birth_state_names_v1(ordinary_variant_count)
    ):
        row = _base_row(
            family="ordinary_birth_state",
            index=index,
            name=name,
            manifest=manifest,
        )
        row, problems = _reconcile_row(
            row,
            _observed_occurrence(
                ordinary_observed, ordinary_occurrences, name
            ),
        )
        if problems:
            raise ValueError(
                f"ordinary_birth_state[{index}] {name}: " + "; ".join(problems)
            )
        ordinary.append(row)
    turnover = []
    turnover_occurrences: dict[str, int] = {}
    if len(TURNOVER_MUTABLE_STATE_FIELDS) != TURNOVER_MUTABLE_COUNT_V1:
        raise RuntimeError("turnover mutable ABI source count drifted")
    if turnover_variant_count not in {0, TURNOVER_MUTABLE_COUNT_V1}:
        raise ValueError(
            "turnover_variant_count must be 0 or "
            f"{TURNOVER_MUTABLE_COUNT_V1}, got {turnover_variant_count}"
        )
    if turnover_variant_count:
        for index, name in enumerate(TURNOVER_MUTABLE_STATE_FIELDS):
            row = _base_row(
                family="turnover_mutable_state",
                index=index,
                name=name,
                manifest=manifest,
            )
            row, problems = _reconcile_row(
                row,
                _observed_occurrence(
                    turnover_observed, turnover_occurrences, name
                ),
            )
            if problems:
                raise ValueError(
                    f"turnover_mutable_state[{index}] {name}: "
                    + "; ".join(problems)
                )
            turnover.append(row)
    return {
        "ordinary_birth_state": tuple(ordinary),
        "turnover_mutable_state": tuple(turnover),
    }


def reconcile_ordinary_turnover_contracts_v1(
    receipt: Mapping[str, Any],
    *,
    ordinary_variant_count: int,
    turnover_variant_count: int = TURNOVER_MUTABLE_COUNT_V1,
) -> dict[str, Any]:
    """Return contracts plus explicit missing/contradictory receipt ambiguities."""

    contracts = build_ordinary_turnover_contracts_v1(
        None,
        ordinary_variant_count=ordinary_variant_count,
        turnover_variant_count=turnover_variant_count,
    )
    observed_by_family = {
        "ordinary_birth_state": _receipt_rows(
            receipt, "ordinary_birth_state_tensors"
        ),
        "turnover_mutable_state": _receipt_rows(receipt, "turnover_state_tensors"),
    }
    reconciled: dict[str, tuple[dict[str, Any], ...]] = {}
    ambiguities: list[dict[str, Any]] = []
    for family, rows in contracts.items():
        observed = observed_by_family[family]
        occurrences: dict[str, int] = {}
        output = []
        for row in rows:
            observation = _observed_occurrence(
                observed, occurrences, row["name"]
            )
            candidate, problems = _reconcile_row(dict(row), observation)
            if problems:
                ambiguities.append(
                    {
                        "family": family,
                        "index": row["index"],
                        "name": row["name"],
                        "reasons": problems,
                    }
                )
            elif observation is None and row["name"] not in _DIPLOID:
                ambiguities.append(
                    {
                        "family": family,
                        "index": row["index"],
                        "name": row["name"],
                        "reasons": ("missing prepared-owner tensor row",),
                    }
                )
            output.append(candidate)
        reconciled[family] = tuple(output)
    return {"contracts": reconciled, "ambiguities": tuple(ambiguities)}


__all__ = [
    "ORDINARY_BIRTH_DIPLOID_COUNT_V1",
    "ORDINARY_BIRTH_LEGACY_COUNT_V1",
    "TURNOVER_MUTABLE_COUNT_V1",
    "build_ordinary_turnover_contracts_v1",
    "ordinary_birth_state_names_v1",
    "reconcile_ordinary_turnover_contracts_v1",
]
