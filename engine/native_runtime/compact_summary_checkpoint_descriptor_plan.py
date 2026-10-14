"""Frozen descriptor plan for device-side compact checkpoint tensor hashing.

The plan composes existing checkpoint authority and digest-policy registries. It
does not capture tensors, invoke CUDA, or modify controller state.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import copy
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from . import compact_summary_digests as compact_digests
from .checkpoint_digests import (
    DEFAULT_FIELD_POLICIES,
    FIELD_POLICY_SCHEMA_VERSION,
    SCHEMA_VERSION as CHECKPOINT_DIGEST_SCHEMA_VERSION,
    SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
    SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
    SOURCE_PROJECTION_SCHEMA_VERSION,
    SOURCE_RNG_TENSOR_FIELDS,
    DigestLayer,
    FieldDigestPolicy,
)
from perf.cuda_architecture_b_state_bank import (
    ACCEPTED_MANIFEST_CONTRACT_STATUS,
    ACCEPTED_MANIFEST_SCHEMA_VERSION,
    ACCEPTED_MANIFEST_SHA256,
    ACCEPTED_TENSOR_FIELD_COUNT,
    AcceptedStateBankManifest,
    StateBankFieldContract,
    WDL_MANIFEST_CONTRACT_STATUS,
    WDL_MANIFEST_SCHEMA_VERSION,
    WDL_MANIFEST_SHA256,
    WDL_TENSOR_FIELD_COUNT,
    load_accepted_state_bank_manifest,
    load_wdl_state_bank_manifest,
)
from perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1,
    TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1,
    TORCH_CPU_RNG_STATE_ENCODING_V1,
    TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
)


PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_checkpoint_descriptor_plan_v1"
)
DESCRIPTOR_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_checkpoint_tensor_descriptor_v1"
)
CUDA_HASH_CONTRACT_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_summary_checkpoint_hash_v1"
)
CUDA_HASH_BATCH_CAPACITY: Final[int] = 16

# Phase E is deliberately an overlay instead of a revision of the frozen v1
# descriptor plan.  Keeping these contracts separate preserves the accepted v1
# JSON bytes and every existing descriptor index when modular storage is absent.
PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_checkpoint_phase_e_modular_descriptor_overlay_v1"
)
PHASE_E_MODULAR_OVERLAY_DESCRIPTOR_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_checkpoint_phase_e_modular_tensor_descriptor_v1"
)
PHASE_E_MODULAR_OVERLAY_STATUS: Final[str] = "phase_e_modular_storage_ready"

WORLD_FLAT: Final[str] = "flat"
WORLD_PATCH: Final[str] = "patch"
WORLD_WDL: Final[str] = "wdl"
PATCH_P2_PROFILE: Final[str] = "P2"
SUPPORTED_DESCRIPTOR_WORLDS: Final[frozenset[str]] = frozenset(
    {WORLD_FLAT, WORLD_PATCH, WORLD_WDL}
)
WDL_STATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "patch_delayed_hazard_active",
        "patch_delayed_hazard_age",
        "patch_delayed_hazard_cue",
        "patch_delayed_hazard_event_id",
        "patch_delayed_hazard_intensity",
        "patch_delayed_hazard_prior_cue",
        "patch_delayed_hazard_timer",
    }
)
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

MODE_EXACT: Final[str] = "exact"
MODE_DEAD_ROW_ZERO: Final[str] = "dead-row-zero"
MODE_ACTIVE_PREFIX: Final[str] = "active-prefix"
MODE_TORCH_RNG_PACKED_PREFIX: Final[str] = (
    "torch-cpu-mt19937-packed-rows-to-dense-active-prefix"
)
MODE_CODES: Final[dict[str, int]] = {
    MODE_EXACT: 0,
    MODE_DEAD_ROW_ZERO: 1,
    MODE_ACTIVE_PREFIX: 2,
    MODE_TORCH_RNG_PACKED_PREFIX: 3,
}
CUDA_HASH_MODE_NAMES: Final[dict[str, str]] = {
    MODE_EXACT: "exact_state_raw_bytes",
    MODE_DEAD_ROW_ZERO: "live_rows_dead_zero",
    MODE_ACTIVE_PREFIX: "active_prefix",
    MODE_TORCH_RNG_PACKED_PREFIX: (
        "torch_cpu_mt19937_packed_rows_to_dense_active_prefix"
    ),
}

FLAT_PROJECTION_FIXTURE: Final[Path] = Path(
    public_runtime_path('evidence_raw/eiel_native_coverage_expansion_20260715/flat_final_closeout/smoke/seed101_g2_steps1_pop4_abi_qualified_r1/flat_generation_four_way_parity_runs/seed_000101/sources/native/bundle_gen000001/native_runtime_checkpoint_projection.json')
)
FLAT_PROJECTION_FIXTURE_SHA256: Final[str] = (
    "ea99cc2378f2b6e1773f0e693549c41e26c85140f3a83458dffb8797a7af6ba8"
)
FLAT_PROJECTION_FIXTURE_TENSOR_COUNT: Final[int] = 125
FLAT_PROJECTION_CURRENT_ADDITIONS: Final[tuple[str, ...]] = (
    "early_energy_collected_exact",
    "energy_collected_exact",
)

_LIVE_ROW_CANONICALIZER: Final[str] = (
    "field-policy row masking only; never blanket-clear every dead/free tensor"
)
_EVENT_PREFIX_CANONICALIZER: Final[str] = (
    "active_event_count prefix exact; inactive tail normalized by policy only"
)
_EXACT_CANONICALIZERS: Final[frozenset[str]] = frozenset(
    {
        "preserve_exact_stored",
        "preserve_exact_stored_grid",
        "excluded from digest layers; retained in projection payload",
    }
)
_NUMPY_DTYPE_BYTES: Final[dict[str, int]] = {
    "bool": 1,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}
_TORCH_MANIFEST_DTYPE: Final[dict[str, str]] = {
    "bool": "Bool",
    "float32": "Float",
    "float64": "Double",
    "int8": "Char",
    "int32": "Int",
    "int64": "Long",
    "uint8": "Byte",
}

_FLAT_EXCLUDED_PATCH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    }
)
_ACTIVE_ID_FIELD: Final[str] = "agent_ids_by_alive_order"
_ACTIVE_ID_PHYSICAL_SOURCE: Final[str] = "active_agent_ids"
_ACTIVE_ID_LENGTH_SOURCE: Final[str] = "active_count_state"
_POLICY_POST_RNG_CANONICAL_FIELD: Final[str] = "policy_birth_post_rng_state_tape"
_POLICY_POST_RNG_PACKED_FIELD: Final[str] = (
    "policy_birth_post_rng_state_packed_tape"
)
_POLICY_POST_RNG_CANONICAL_ROW_SYMBOL: Final[str] = "mutation_rng_state_bytes"
_POLICY_POST_RNG_PACKED_ROW_SYMBOL: Final[str] = "packed_torch_rng_state_bytes"
_PHYSICAL_SOURCE_OVERRIDES: Final[dict[str, str]] = {
    "agent_id_to_slot_agent_ids": _ACTIVE_ID_PHYSICAL_SOURCE,
    "agent_id_to_slot_slots": "active_agent_slots",
    _ACTIVE_ID_FIELD: _ACTIVE_ID_PHYSICAL_SOURCE,
}
_IDENTITY_PREFIX_AUTHORITIES: Final[dict[str, tuple[str, str]]] = {
    "agent_id_to_slot_agent_ids": ("metadata.alive_count", "alive_count"),
    "agent_id_to_slot_slots": ("metadata.alive_count", "alive_count"),
    "agent_ids_by_alive_order": (
        _ACTIVE_ID_LENGTH_SOURCE,
        "device active_count_state selects [0, alive_count)",
    ),
    "slot_free_queue": (
        "identity.slot_free_queue_count",
        "capacity - alive_count",
    ),
}
_POLICY_TAPE_SHAPES: Final[dict[str, tuple[str, ...]]] = {
    "policy_birth_crossover_mask_tape": (
        "policy_birth_tape_capacity",
        "mlp_pack_width",
    ),
    "policy_birth_delta_tape": (
        "policy_birth_tape_capacity",
        "mlp_pack_width",
    ),
    "policy_birth_mutation_mask_tape": (
        "policy_birth_tape_capacity",
        "mlp_pack_width",
    ),
    _POLICY_POST_RNG_CANONICAL_FIELD: (
        "policy_birth_tape_capacity",
        _POLICY_POST_RNG_PACKED_ROW_SYMBOL,
    ),
    "policy_birth_semantic_draws_tape": ("policy_birth_tape_capacity",),
}


class CompactSummaryCheckpointDescriptorPlanError(ValueError):
    """Raised when a source registry or generated descriptor plan drifts."""


def _fail(path: str, message: str) -> None:
    raise CompactSummaryCheckpointDescriptorPlanError(f"{path}: {message}")


def _phase_e_modular_descriptor_source(field: Any) -> dict[str, Any]:
    from perf.cuda_architecture_b_modular_state_abi import (  # noqa: WPS433
        MODULAR_STORAGE_PROJECTION_PREFIX,
    )

    projected_name = f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}"
    if field.dtype_name not in _NUMPY_DTYPE_BYTES:
        _fail(
            f"phase_e_modular_storage.fields.{field.name}.dtype",
            "unsupported by compact-summary hash runtime",
        )
    shape = [int(extent) for extent in field.shape]
    if not shape or any(extent <= 0 for extent in shape):
        _fail(
            f"phase_e_modular_storage.fields.{field.name}.shape",
            "must be a fixed positive shape",
        )
    return {
        "schema_version": PHASE_E_MODULAR_OVERLAY_DESCRIPTOR_SCHEMA_VERSION,
        "descriptor_id": f"phase_e_modular_storage:{field.name}",
        "tensor_name": projected_name,
        "logical_tensor_name": str(field.name),
        "source_scope": "phase_e_modular_storage_projection",
        "physical_source_name": projected_name,
        "hash_mode": CUDA_HASH_MODE_NAMES[MODE_EXACT],
        "active_length_source": None,
        "dtype": str(field.dtype_name),
        "dtype_bytes": _NUMPY_DTYPE_BYTES[field.dtype_name],
        "torch_tensor_manifest_dtype": _TORCH_MANIFEST_DTYPE[field.dtype_name],
        "fixed_shape": shape,
        "rank": len(shape),
        "raw_canonical_mode": MODE_EXACT,
        "raw_canonical_mode_code": MODE_CODES[MODE_EXACT],
        "alive_authority": None,
        "length_authority": None,
        "length_policy": None,
        "digest_inclusion": {
            "phase_e_modular_storage": True,
            "phase_e_modular_storage_output": "raw",
            "base_v1_descriptor_plan": False,
        },
    }


def maybe_generate_phase_e_modular_checkpoint_descriptor_overlay(
    projection_metadata: Mapping[str, Any],
    projection_tensors: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a validated modular-only descriptor overlay, or ``None``.

    Absence is represented by no object at all.  Any partial, mixed-version, or
    tampered modular projection is rejected by the committed storage ABI before
    descriptor metadata is emitted.
    """

    if not isinstance(projection_metadata, Mapping):
        raise TypeError("projection_metadata must be a mapping")
    if not isinstance(projection_tensors, Mapping):
        raise TypeError("projection_tensors must be a mapping")
    from perf.cuda_architecture_b_modular_state_abi import (  # noqa: WPS433
        MODULAR_STORAGE_METADATA_KEY,
        MODULAR_STORAGE_PROJECTION_PREFIX,
        modular_storage_from_projection,
    )

    packed = modular_storage_from_projection(projection_metadata, projection_tensors)
    if packed is None:
        return None
    sources = sorted(
        (_phase_e_modular_descriptor_source(field) for field in packed.layout.fields),
        key=lambda row: row["tensor_name"],
    )
    expected_names = [row["tensor_name"] for row in sources]
    selected_names = sorted(
        str(name)
        for name in projection_tensors
        if str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    if selected_names != expected_names:
        _fail(
            "phase_e_modular_storage.projection_tensors",
            f"field mismatch expected={expected_names!r} actual={selected_names!r}",
        )
    descriptors: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        batch_index, batch_slot = divmod(index, CUDA_HASH_BATCH_CAPACITY)
        descriptors.append(
            {
                **source,
                "descriptor_index": index,
                "raw_output_index": index,
                "canonical_output_index": index,
                "batch_index": batch_index,
                "batch_slot": batch_slot,
            }
        )
    batches = []
    for start in range(0, len(descriptors), CUDA_HASH_BATCH_CAPACITY):
        rows = descriptors[start : start + CUDA_HASH_BATCH_CAPACITY]
        batches.append(
            {
                "batch_index": len(batches),
                "count": len(rows),
                "descriptor_indices": [row["descriptor_index"] for row in rows],
                "tensor_names": [row["tensor_name"] for row in rows],
            }
        )
    envelope = projection_metadata.get(MODULAR_STORAGE_METADATA_KEY)
    if not isinstance(envelope, Mapping):
        _fail("phase_e_modular_storage.projection_metadata", "missing exact envelope")
    return {
        "schema_version": PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION,
        "status": PHASE_E_MODULAR_OVERLAY_STATUS,
        "base_descriptor_plan_schema_version": PLAN_SCHEMA_VERSION,
        "ordering": "strict lexicographic projected tensor_name",
        "cuda_hash_contract": {
            "schema_version": CUDA_HASH_CONTRACT_VERSION,
            "max_descriptors_per_batch": CUDA_HASH_BATCH_CAPACITY,
            "mode_codes": {MODE_EXACT: MODE_CODES[MODE_EXACT]},
            "hash_mode_names": {MODE_EXACT: CUDA_HASH_MODE_NAMES[MODE_EXACT]},
        },
        "source_contract": {
            "projection_metadata": {
                MODULAR_STORAGE_METADATA_KEY: copy.deepcopy(dict(envelope))
            },
            "projection_metadata_key": MODULAR_STORAGE_METADATA_KEY,
            "projection_tensor_prefix": MODULAR_STORAGE_PROJECTION_PREFIX,
            "storage_schema_version": packed.layout.schema_version,
            "profile_id": packed.layout.profile_id,
            "layout": packed.layout.to_dict(),
            "selected_projection_tensor_names": expected_names,
        },
        "counts": {
            "projection_metadata_fields": 1,
            "descriptors": len(descriptors),
            "batches": len(batches),
            "max_descriptors_per_batch": CUDA_HASH_BATCH_CAPACITY,
        },
        "descriptors": descriptors,
        "batches": batches,
    }


def generate_phase_e_modular_checkpoint_descriptor_overlay(
    projection_metadata: Mapping[str, Any],
    projection_tensors: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate the required Phase E overlay and reject an absent projection."""

    overlay = maybe_generate_phase_e_modular_checkpoint_descriptor_overlay(
        projection_metadata,
        projection_tensors,
    )
    if overlay is None:
        _fail("phase_e_modular_storage", "projection is absent")
    return overlay


def validate_phase_e_modular_checkpoint_descriptor_overlay(
    overlay: Mapping[str, Any],
    projection_metadata: Mapping[str, Any],
    projection_tensors: Mapping[str, Any],
) -> None:
    """Fail closed unless the overlay exactly describes the supplied authority."""

    if not isinstance(overlay, Mapping):
        raise TypeError("phase_e modular descriptor overlay must be a mapping")
    expected = generate_phase_e_modular_checkpoint_descriptor_overlay(
        projection_metadata,
        projection_tensors,
    )
    if dict(overlay) != expected:
        _fail(
            "phase_e_modular_descriptor_overlay",
            "does not exactly match its validated projection authority",
        )


def _validated_world(world: str) -> str:
    if type(world) is not str or world not in SUPPORTED_DESCRIPTOR_WORLDS:
        _fail("world", "must be exactly 'flat', 'patch', or 'wdl'")
    return world


def _policy_tape_registry(
    registry: Mapping[str, tuple[np.dtype[Any], int]] | None = None,
) -> dict[str, tuple[str, int]]:
    source = (
        compact_digests._POLICY_TAPE_DTYPES_AND_RANKS
        if registry is None
        else registry
    )
    expected_names = set(_POLICY_TAPE_SHAPES)
    if set(source) != expected_names:
        _fail(
            "policy_tape_registry",
            f"field mismatch missing={sorted(expected_names - set(source))} "
            f"extra={sorted(set(source) - expected_names)}",
        )
    rows: dict[str, tuple[str, int]] = {}
    for name, raw in source.items():
        if not isinstance(raw, tuple) or len(raw) != 2:
            _fail(f"policy_tape_registry.{name}", "must be a dtype/rank pair")
        dtype, rank = str(np.dtype(raw[0])), raw[1]
        if dtype not in _NUMPY_DTYPE_BYTES or type(rank) is not int or rank < 1:
            _fail(f"policy_tape_registry.{name}", "unsupported dtype or rank")
        if rank != len(_POLICY_TAPE_SHAPES[name]):
            _fail(f"policy_tape_registry.{name}", "rank contradicts shape policy")
        rows[name] = (dtype, rank)
    return rows


def _validated_flat_fields(
    manifest: AcceptedStateBankManifest,
    policies: Mapping[str, FieldDigestPolicy],
) -> tuple[StateBankFieldContract, ...]:
    if manifest.schema_version != ACCEPTED_MANIFEST_SCHEMA_VERSION:
        _fail("state_bank.schema_version", "not the accepted authority registry")
    if manifest.contract_status != ACCEPTED_MANIFEST_CONTRACT_STATUS:
        _fail("state_bank.contract_status", "not accepted for CUDA Stage 1")
    if manifest.source_sha256 != ACCEPTED_MANIFEST_SHA256:
        _fail("state_bank.source_sha256", "accepted authority digest drift")
    names = [field.name for field in manifest.fields]
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    if duplicates:
        _fail("state_bank.fields", f"duplicate authorities {duplicates}")
    if len(manifest.fields) != ACCEPTED_TENSOR_FIELD_COUNT:
        _fail("state_bank.fields", "missing or extra accepted tensor authorities")
    missing_policies = sorted(set(names) - set(policies))
    if missing_policies:
        _fail("checkpoint_field_policies", f"unclassified fields {missing_policies}")
    for name, policy in policies.items():
        if name != policy.field:
            _fail(
                f"checkpoint_field_policies.{name}",
                f"registry key contradicts policy field {policy.field!r}",
            )
    missing_patch = sorted(_FLAT_EXCLUDED_PATCH_FIELDS - set(names))
    if missing_patch:
        _fail("state_bank.fields", f"missing classified Patch fields {missing_patch}")
    fields = tuple(
        sorted(
            (
                field
                for field in manifest.fields
                if field.name not in _FLAT_EXCLUDED_PATCH_FIELDS
            ),
            key=lambda field: field.name,
        )
    )
    if len(fields) != ACCEPTED_TENSOR_FIELD_COUNT - len(_FLAT_EXCLUDED_PATCH_FIELDS):
        _fail("flat_projection.fields", "Flat authority count contradiction")
    return fields


def _validated_patch_p2_fields(
    manifest: AcceptedStateBankManifest,
    policies: Mapping[str, FieldDigestPolicy],
) -> tuple[StateBankFieldContract, ...]:
    # Run the frozen Flat validation first so Patch cannot bypass any accepted
    # manifest, policy, duplicate, or classified-field check.
    _validated_flat_fields(manifest, policies)
    fields = tuple(sorted(manifest.fields, key=lambda field: field.name))
    if len(fields) != ACCEPTED_TENSOR_FIELD_COUNT:
        _fail("patch_p2_projection.fields", "Patch P2 authority count contradiction")
    return fields


def _validated_wdl_fields(
    manifest: AcceptedStateBankManifest,
    policies: Mapping[str, FieldDigestPolicy],
) -> tuple[StateBankFieldContract, ...]:
    if manifest.schema_version != WDL_MANIFEST_SCHEMA_VERSION:
        _fail("state_bank.schema_version", "not the accepted WDL authority registry")
    if manifest.contract_status != WDL_MANIFEST_CONTRACT_STATUS:
        _fail("state_bank.contract_status", "not accepted for WDL storage")
    if manifest.source_sha256 != WDL_MANIFEST_SHA256:
        _fail("state_bank.source_sha256", "accepted WDL authority digest drift")
    if len(manifest.fields) != WDL_TENSOR_FIELD_COUNT:
        _fail("state_bank.fields", "missing or extra accepted WDL authorities")
    flat_patch = load_accepted_state_bank_manifest()
    if manifest.fields[:ACCEPTED_TENSOR_FIELD_COUNT] != flat_patch.fields:
        _fail("state_bank.fields", "WDL profile changed the Flat/Patch authority prefix")
    names = [field.name for field in manifest.fields]
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    if duplicates:
        _fail("state_bank.fields", f"duplicate authorities {duplicates}")
    if set(names[ACCEPTED_TENSOR_FIELD_COUNT:]) != WDL_STATE_FIELDS:
        _fail("state_bank.fields", "WDL overlay field set is not exact")
    missing_policies = sorted(set(names) - set(policies))
    if missing_policies:
        _fail("checkpoint_field_policies", f"unclassified fields {missing_policies}")
    return tuple(sorted(manifest.fields, key=lambda field: field.name))


def _mode_and_authority(
    field: StateBankFieldContract,
    policy: FieldDigestPolicy,
) -> tuple[str, str | None, str | None, str | None]:
    name = field.name
    canonicalizer = policy.canonicalization
    if name in _IDENTITY_PREFIX_AUTHORITIES:
        if canonicalizer != "preserve_exact_stored":
            _fail(f"checkpoint_field_policies.{name}", "identity prefix must be exact stored")
        source, expression = _IDENTITY_PREFIX_AUTHORITIES[name]
        return MODE_ACTIVE_PREFIX, None, source, expression
    if canonicalizer == _EVENT_PREFIX_CANONICALIZER:
        if not (
            name.startswith("event_tape_")
            or name.startswith("typed_lifecycle_event_")
        ):
            _fail(f"checkpoint_field_policies.{name}", "unclassified prefix field")
        source = (
            "metadata.typed_lifecycle_event_active_count"
            if name.startswith("typed_lifecycle_event_")
            else "metadata.event_tape_active_count"
        )
        return MODE_ACTIVE_PREFIX, None, source, "active event prefix length"
    if field.checkpoint_mode == "included_active_prefix":
        _fail(f"state_bank.fields.{name}", "active-prefix mode lacks prefix policy")
    if canonicalizer == _LIVE_ROW_CANONICALIZER:
        return MODE_DEAD_ROW_ZERO, "tensors.alive_mask", None, None
    if name in SOURCE_PATCH_DERIVED_TENSOR_FIELDS:
        expected = DEFAULT_FIELD_POLICIES[name].canonicalization
        if canonicalizer != expected:
            _fail(
                f"checkpoint_field_policies.{name}",
                "derived Patch canonicalizer contradicts the authoritative registry",
            )
        return MODE_EXACT, None, None, None
    if canonicalizer in _EXACT_CANONICALIZERS:
        return MODE_EXACT, None, None, None
    _fail(
        f"checkpoint_field_policies.{name}.canonicalization",
        f"unclassified value {canonicalizer!r}",
    )
    raise AssertionError("unreachable")


def _digest_inclusion(
    *,
    name: str,
    source_scope: str,
    mode: str,
    policy: FieldDigestPolicy | None,
) -> dict[str, Any]:
    if source_scope == "policy_mutation_tape":
        return {
            "source_projection": False,
            "source_projection_output": None,
            "checkpoint_layers": [],
            "checkpoint_layers_output": None,
            "checkpoint_rng": False,
            "checkpoint_rng_output": None,
            "active_agent_ids": False,
            "active_agent_ids_output": None,
            "policy_mutation_tape": True,
            "policy_mutation_tape_output": "canonical",
        }
    assert policy is not None
    layers = [str(layer) for layer in DigestLayer if policy.includes(layer)]
    source_output = "canonical" if mode == MODE_ACTIVE_PREFIX else "raw"
    return {
        "source_projection": True,
        "source_projection_output": source_output,
        "checkpoint_layers": layers,
        "checkpoint_layers_output": "canonical" if layers else None,
        "checkpoint_rng": name in SOURCE_RNG_TENSOR_FIELDS,
        "checkpoint_rng_output": "raw" if name in SOURCE_RNG_TENSOR_FIELDS else None,
        "active_agent_ids": name == _ACTIVE_ID_FIELD,
        "active_agent_ids_output": "canonical" if name == _ACTIVE_ID_FIELD else None,
        "policy_mutation_tape": False,
        "policy_mutation_tape_output": None,
    }


def _checkpoint_descriptor_source(
    field: StateBankFieldContract,
    policy: FieldDigestPolicy,
) -> dict[str, Any]:
    resident_shape = list(field.device_neutral_shape_variables)
    if (
        field.device_neutral_rank != len(resident_shape)
        or not resident_shape
        or resident_shape[0] != "cell"
    ):
        _fail(f"state_bank.fields.{field.name}.shape", "must be cell-leading")
    if field.dtype_name not in _NUMPY_DTYPE_BYTES:
        _fail(f"state_bank.fields.{field.name}.dtype", "unsupported by hash kernel")
    raw_shape = resident_shape[1:]
    mode, alive_authority, length_authority, length_policy = _mode_and_authority(
        field, policy
    )
    if mode == MODE_DEAD_ROW_ZERO and (not raw_shape or raw_shape[0] != "capacity"):
        _fail(f"state_bank.fields.{field.name}.shape", "live-row field is not capacity-leading")
    canonical_shape = list(raw_shape)
    if mode == MODE_ACTIVE_PREFIX:
        if not raw_shape:
            _fail(f"state_bank.fields.{field.name}.shape", "prefix field has no row axis")
        if field.name == "slot_free_queue":
            canonical_shape[0] = "slot_free_queue_count"
        elif field.name in _IDENTITY_PREFIX_AUTHORITIES:
            canonical_shape[0] = "alive_count"
        elif field.name.startswith("typed_lifecycle_event_"):
            canonical_shape[0] = "typed_lifecycle_event_active_count"
        else:
            canonical_shape[0] = "event_tape_active_count"
    physical_source_name = _PHYSICAL_SOURCE_OVERRIDES.get(field.name, field.name)
    return {
        "descriptor_id": f"checkpoint:{field.name}",
        "tensor_name": field.name,
        "source_scope": "checkpoint_projection",
        "physical_source_name": physical_source_name,
        "hash_mode": CUDA_HASH_MODE_NAMES[mode],
        "active_length_source": length_authority if mode == MODE_ACTIVE_PREFIX else None,
        "dtype": field.dtype_name,
        "dtype_bytes": _NUMPY_DTYPE_BYTES[field.dtype_name],
        "torch_tensor_manifest_dtype": _TORCH_MANIFEST_DTYPE[field.dtype_name],
        "resident_shape_symbols": resident_shape,
        "resident_rank": field.device_neutral_rank,
        "raw_shape_symbols": raw_shape,
        "raw_rank": len(raw_shape),
        "canonical_shape_symbols": canonical_shape,
        "canonical_rank": len(canonical_shape),
        "raw_canonical_mode": mode,
        "raw_canonical_mode_code": MODE_CODES[mode],
        "alive_authority": alive_authority,
        "length_authority": length_authority,
        "length_policy": length_policy,
        "state_bank_authority_class": field.authority_class,
        "state_bank_checkpoint_mode": field.checkpoint_mode,
        "field_policy_canonicalizer": policy.canonicalization,
        "digest_inclusion": _digest_inclusion(
            name=field.name,
            source_scope="checkpoint_projection",
            mode=mode,
            policy=policy,
        ),
    }


def _tape_descriptor_source(name: str, dtype: str, rank: int) -> dict[str, Any]:
    raw_shape = list(_POLICY_TAPE_SHAPES[name])
    canonical_shape = list(raw_shape)
    canonical_shape[0] = "policy_birth_tape_cursor"
    physical_source_name = name
    physical_encoding = None
    physical_row_bytes = None
    canonical_row_bytes = None
    canonical_hash_identity = None
    canonical_materialization_rule = None
    mode = MODE_ACTIVE_PREFIX
    if name == _POLICY_POST_RNG_CANONICAL_FIELD:
        physical_source_name = _POLICY_POST_RNG_PACKED_FIELD
        canonical_shape[1] = _POLICY_POST_RNG_CANONICAL_ROW_SYMBOL
        physical_encoding = TORCH_CPU_RNG_STATE_ENCODING_V1
        physical_row_bytes = TORCH_CPU_RNG_STATE_PACKED_BYTES_V1
        canonical_row_bytes = TORCH_CPU_RNG_STATE_FULL_BYTES_V1
        canonical_hash_identity = _POLICY_POST_RNG_CANONICAL_FIELD
        canonical_materialization_rule = (
            "bounded row decode only; complete dense post-state tape materialization "
            "is forbidden"
        )
        mode = MODE_TORCH_RNG_PACKED_PREFIX
    return {
        "descriptor_id": f"policy_mutation_tape:{name}",
        "tensor_name": name,
        "source_scope": "policy_mutation_tape",
        "physical_source_name": physical_source_name,
        "hash_mode": CUDA_HASH_MODE_NAMES[mode],
        "active_length_source": "policy_mutation_tape.policy_birth_tape_cursor",
        "dtype": dtype,
        "dtype_bytes": _NUMPY_DTYPE_BYTES[dtype],
        "torch_tensor_manifest_dtype": _TORCH_MANIFEST_DTYPE[dtype],
        "resident_shape_symbols": ["cell", *raw_shape],
        "resident_rank": rank + 1,
        "raw_shape_symbols": raw_shape,
        "raw_rank": rank,
        "canonical_shape_symbols": canonical_shape,
        "canonical_rank": rank,
        "raw_canonical_mode": mode,
        "raw_canonical_mode_code": MODE_CODES[mode],
        "alive_authority": None,
        "length_authority": "policy_mutation_tape.policy_birth_tape_cursor",
        "length_policy": "consumed half-open prefix [0, cursor)",
        "state_bank_authority_class": "runtime_auxiliary",
        "state_bank_checkpoint_mode": None,
        "field_policy_canonicalizer": None,
        "physical_representation": (
            {
                "field_name": physical_source_name,
                "encoding": physical_encoding,
                "abi_schema_version": TORCH_CPU_RNG_STATE_ABI_SCHEMA_VERSION_V1,
                "abi_fingerprint_sha256": (
                    TORCH_CPU_RNG_STATE_ABI_FINGERPRINT_SHA256_V1
                ),
                "row_bytes": physical_row_bytes,
                "canonical_semantic_field": canonical_hash_identity,
                "canonical_row_bytes": canonical_row_bytes,
                "canonical_hash_bytes": (
                    "exact reconstructed 5056-byte Torch CPU RNG state rows"
                ),
                "full_dense_tape_materialization_allowed": False,
                "canonical_materialization_rule": canonical_materialization_rule,
                "codec_route_binding": "execution_route_sha256",
            }
            if name == _POLICY_POST_RNG_CANONICAL_FIELD
            else None
        ),
        "digest_inclusion": _digest_inclusion(
            name=name,
            source_scope="policy_mutation_tape",
            mode=mode,
            policy=None,
        ),
    }


def _outer_manifest_semantics() -> dict[str, Any]:
    return {
        "source_projection_prehashed_tensor_manifest": {
            "fields": ["device", "dtype", "shape", "sha256"],
            "device": "cpu",
            "dtype": "NumPy spelling from descriptor.dtype",
            "shape_and_sha256": (
                "canonical output for active-prefix; raw output for exact and dead-row-zero"
            ),
        },
        "stored_projection_tensor_manifest": {
            "fields": [
                "shape",
                "dtype",
                "device",
                "numel",
                "element_size",
                "bytes",
                "is_cuda",
                "is_contiguous",
            ],
            "device": "cpu",
            "dtype": "C++ Torch spelling from descriptor.torch_tensor_manifest_dtype",
            "is_cuda": False,
            "is_contiguous": True,
            "shape": "same source-projection output selection",
        },
        "checkpoint_layer_tensor_manifest": {
            "fields": ["field", "dtype", "shape", "sha256"],
            "device_field": "omitted",
            "dtype": "NumPy spelling from descriptor.dtype",
            "shape_and_sha256": "canonical output",
            "ordering": "strict lexicographic included tensor_name",
        },
        "active_agent_ids_tensor_manifest": {
            "fields": ["field", "dtype", "shape", "sha256"],
            "device_field": "omitted",
            "dtype": "int64",
            "shape_and_sha256": "canonical agent_ids_by_alive_order prefix",
            "runtime_descriptor": "checkpoint:agent_ids_by_alive_order",
            "physical_source_name": _ACTIVE_ID_PHYSICAL_SOURCE,
            "physical_shape": "int64[cell, capacity]",
            "hash_mode": CUDA_HASH_MODE_NAMES[MODE_ACTIVE_PREFIX],
            "active_length_source": _ACTIVE_ID_LENGTH_SOURCE,
            "canonical_shape": "int64[alive_count]",
            "digest_equivalence": (
                "sha256(active_agent_ids[:active_count_state]) equals "
                "sha256(agent_ids_by_alive_order)"
            ),
        },
        "policy_mutation_tape_tensor_manifest": {
            "fields": ["field", "dtype", "shape", "sha256"],
            "device_field": "omitted",
            "dtype": "NumPy spelling from descriptor.dtype",
            "shape_and_sha256": "canonical consumed prefix",
            "ordering": "strict lexicographic tensor_name",
            "packed_torch_rng_post_state": {
                "physical_field": _POLICY_POST_RNG_PACKED_FIELD,
                "physical_row_bytes": TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
                "canonical_semantic_field": _POLICY_POST_RNG_CANONICAL_FIELD,
                "canonical_row_bytes": TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
                "hash_method": "bounded row decode",
                "full_dense_tape_materialization_allowed": False,
            },
        },
        "host_finalization": {
            "status": "required_and_supported_for_flat",
            "behavior": (
                "retain non-tensor projection fields and metadata, reconstruct the stored "
                "tensor_manifest, then call existing host outer-digest helpers"
            ),
            "device_hash_interpretation": (
                "logical checkpoint tensors remain CPU projections even when bytes "
                "are hashed on CUDA"
            ),
        },
    }


def _build_plan(
    *,
    manifest: AcceptedStateBankManifest,
    policies: Mapping[str, FieldDigestPolicy],
    tape_registry: Mapping[str, tuple[np.dtype[Any], int]] | None,
    world: str,
) -> dict[str, Any]:
    resolved_world = _validated_world(world)
    checkpoint_fields = (
        _validated_flat_fields(manifest, policies)
        if resolved_world == WORLD_FLAT
        else (
            _validated_patch_p2_fields(manifest, policies)
            if resolved_world == WORLD_PATCH
            else _validated_wdl_fields(manifest, policies)
        )
    )
    tapes = _policy_tape_registry(tape_registry)
    sources = [
        _checkpoint_descriptor_source(field, policies[field.name])
        for field in checkpoint_fields
    ]
    sources.extend(
        _tape_descriptor_source(name, dtype, rank)
        for name, (dtype, rank) in tapes.items()
    )
    names = [row["tensor_name"] for row in sources]
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    if duplicates:
        _fail("descriptors", f"duplicate tensor names {duplicates}")
    sources.sort(key=lambda row: row["tensor_name"])
    descriptors = []
    for index, source in enumerate(sources):
        batch_index, batch_slot = divmod(index, CUDA_HASH_BATCH_CAPACITY)
        descriptors.append(
            {
                "schema_version": DESCRIPTOR_SCHEMA_VERSION,
                **source,
                "descriptor_index": index,
                "raw_output_index": index,
                "canonical_output_index": index,
                "batch_index": batch_index,
                "batch_slot": batch_slot,
            }
        )
    batches = []
    for start in range(0, len(descriptors), CUDA_HASH_BATCH_CAPACITY):
        rows = descriptors[start : start + CUDA_HASH_BATCH_CAPACITY]
        batches.append(
            {
                "batch_index": len(batches),
                "count": len(rows),
                "descriptor_indices": [row["descriptor_index"] for row in rows],
                "tensor_names": [row["tensor_name"] for row in rows],
            }
        )
    mode_counts = Counter(row["raw_canonical_mode"] for row in descriptors)
    inclusion_counts = {
        key: sum(bool(row["digest_inclusion"][key]) for row in descriptors)
        for key in (
            "source_projection",
            "checkpoint_rng",
            "active_agent_ids",
            "policy_mutation_tape",
        )
    }
    inclusion_counts["checkpoint_layer_each"] = sum(
        len(row["digest_inclusion"]["checkpoint_layers"]) == len(DigestLayer)
        for row in descriptors
    )
    if resolved_world == WORLD_FLAT:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "status": "flat_ready_patch_source_projection_pending",
            "ordering": "strict global lexicographic tensor_name",
            "cuda_hash_contract": {
                "schema_version": CUDA_HASH_CONTRACT_VERSION,
                "max_descriptors_per_batch": CUDA_HASH_BATCH_CAPACITY,
                "mode_codes": MODE_CODES,
                "hash_mode_names": CUDA_HASH_MODE_NAMES,
            },
            "source_contract": {
                "state_bank_manifest": {
                    "source_path": manifest.source_path,
                    "source_sha256": manifest.source_sha256,
                    "schema_version": manifest.schema_version,
                    "contract_status": manifest.contract_status,
                    "tensor_field_count": len(manifest.fields),
                },
                "field_policy_registry": {
                    "schema_version": FIELD_POLICY_SCHEMA_VERSION,
                    "digest_layer_schema_version": CHECKPOINT_DIGEST_SCHEMA_VERSION,
                },
                "source_projection_schema_version": SOURCE_PROJECTION_SCHEMA_VERSION,
                "accepted_flat_projection_fixture": {
                    "path": str(FLAT_PROJECTION_FIXTURE),
                    "sha256": FLAT_PROJECTION_FIXTURE_SHA256,
                    "stored_tensor_count": FLAT_PROJECTION_FIXTURE_TENSOR_COUNT,
                    "current_registry_additions": list(
                        FLAT_PROJECTION_CURRENT_ADDITIONS
                    ),
                },
                "flat_excluded_patch_fields": sorted(_FLAT_EXCLUDED_PATCH_FIELDS),
            },
            "outer_manifest_semantics": _outer_manifest_semantics(),
            "unresolved_metadata": [
                {
                    "scope": "patch_source_projection",
                    "status": "explicit_projection_pending",
                    "reason": (
                        "checkpoint_source_projection_digest_from_tensor_manifest "
                        "currently accepts Flat only; no Patch source projection is "
                        "guessed"
                    ),
                }
            ],
            "counts": {
                "accepted_state_bank_authorities": len(manifest.fields),
                "flat_excluded_patch_authorities": len(_FLAT_EXCLUDED_PATCH_FIELDS),
                "flat_checkpoint_tensor_authorities": len(checkpoint_fields),
                "policy_mutation_tape_fields": len(tapes),
                "descriptors": len(descriptors),
                "batches": len(batches),
                "mode_counts": {mode: mode_counts[mode] for mode in MODE_CODES},
                "digest_inclusion_counts": inclusion_counts,
                "unresolved_metadata": 1,
            },
            "descriptors": descriptors,
            "batches": batches,
        }

    wdl_profile = resolved_world == WORLD_WDL
    manifest_source_path = manifest.source_path
    if wdl_profile:
        try:
            manifest_source_path = str(
                Path(manifest.source_path).resolve().relative_to(_PROJECT_ROOT)
            )
        except ValueError:
            _fail(
                "state_bank.source_path",
                "WDL manifest must be repository-bundled for reproducible generation",
            )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": (
            "flat_patch_wdl_storage_ready" if wdl_profile else "flat_patch_p2_ready"
        ),
        "ordering": "strict global lexicographic tensor_name",
        "cuda_hash_contract": {
            "schema_version": CUDA_HASH_CONTRACT_VERSION,
            "max_descriptors_per_batch": CUDA_HASH_BATCH_CAPACITY,
            "mode_codes": MODE_CODES,
            "hash_mode_names": CUDA_HASH_MODE_NAMES,
        },
        "source_contract": {
            "state_bank_manifest": {
                "source_path": manifest_source_path,
                "source_sha256": manifest.source_sha256,
                "schema_version": manifest.schema_version,
                "contract_status": manifest.contract_status,
                "tensor_field_count": len(manifest.fields),
            },
            "field_policy_registry": {
                "schema_version": FIELD_POLICY_SCHEMA_VERSION,
                "digest_layer_schema_version": CHECKPOINT_DIGEST_SCHEMA_VERSION,
            },
            "source_projection_schema_version": SOURCE_PROJECTION_SCHEMA_VERSION,
            "accepted_flat_projection_fixture": {
                "path": str(FLAT_PROJECTION_FIXTURE),
                "sha256": FLAT_PROJECTION_FIXTURE_SHA256,
                "stored_tensor_count": FLAT_PROJECTION_FIXTURE_TENSOR_COUNT,
                "current_registry_additions": list(FLAT_PROJECTION_CURRENT_ADDITIONS),
            },
            "flat_excluded_patch_fields": sorted(_FLAT_EXCLUDED_PATCH_FIELDS),
            "descriptor_world": resolved_world,
            "patch_profile": PATCH_P2_PROFILE,
            "included_patch_p2_fields": sorted(_FLAT_EXCLUDED_PATCH_FIELDS),
            "derived_patch_tensor_fields": sorted(
                SOURCE_PATCH_DERIVED_TENSOR_FIELDS
            ),
            **(
                {"included_wdl_fields": sorted(WDL_STATE_FIELDS)}
                if wdl_profile
                else {}
            ),
        },
        "outer_manifest_semantics": {
            **_outer_manifest_semantics(),
            "host_finalization": {
                **_outer_manifest_semantics()["host_finalization"],
                "status": (
                    "required_and_supported_for_flat_patch_p2_and_wdl_storage"
                    if wdl_profile
                    else "required_and_supported_for_flat_and_patch_p2"
                ),
            },
        },
        "unresolved_metadata": [],
        "counts": {
            "accepted_state_bank_authorities": len(manifest.fields),
            "flat_excluded_patch_authorities": len(_FLAT_EXCLUDED_PATCH_FIELDS),
            (
                "wdl_checkpoint_tensor_authorities"
                if wdl_profile
                else "patch_p2_checkpoint_tensor_authorities"
            ): len(checkpoint_fields),
            "policy_mutation_tape_fields": len(tapes),
            "descriptors": len(descriptors),
            "batches": len(batches),
            "mode_counts": {mode: mode_counts[mode] for mode in MODE_CODES},
            "digest_inclusion_counts": inclusion_counts,
            "unresolved_metadata": 0,
        },
        "descriptors": descriptors,
        "batches": batches,
    }


def generate_compact_summary_checkpoint_descriptor_plan(
    *,
    manifest: AcceptedStateBankManifest | None = None,
    policies: Mapping[str, FieldDigestPolicy] | None = None,
    policy_tape_registry: Mapping[str, tuple[np.dtype[Any], int]] | None = None,
    world: str = WORLD_FLAT,
) -> dict[str, Any]:
    """Generate and validate the frozen descriptor plan from source registries."""

    resolved_world = _validated_world(world)
    resolved_manifest = (
        load_wdl_state_bank_manifest()
        if manifest is None and resolved_world == WORLD_WDL
        else load_accepted_state_bank_manifest()
        if manifest is None
        else manifest
    )
    resolved_policies = DEFAULT_FIELD_POLICIES if policies is None else policies
    plan = _build_plan(
        manifest=resolved_manifest,
        policies=resolved_policies,
        tape_registry=policy_tape_registry,
        world=resolved_world,
    )
    validate_compact_summary_checkpoint_descriptor_plan(
        plan,
        manifest=resolved_manifest,
        policies=resolved_policies,
        policy_tape_registry=policy_tape_registry,
        world=resolved_world,
    )
    return plan


def validate_compact_summary_checkpoint_descriptor_plan(
    plan: Mapping[str, Any],
    *,
    manifest: AcceptedStateBankManifest | None = None,
    policies: Mapping[str, FieldDigestPolicy] | None = None,
    policy_tape_registry: Mapping[str, tuple[np.dtype[Any], int]] | None = None,
    world: str | None = None,
) -> None:
    """Fail closed unless plan exactly matches the current accepted registries."""

    if not isinstance(plan, Mapping):
        raise TypeError("descriptor plan must be a mapping")
    source_contract = plan.get("source_contract")
    descriptor_world = (
        source_contract.get("descriptor_world")
        if isinstance(source_contract, Mapping)
        else None
    )
    if world is None:
        resolved_world = WORLD_FLAT if descriptor_world is None else descriptor_world
    else:
        resolved_world = world
        if descriptor_world is not None and descriptor_world != resolved_world:
            _fail("descriptor_plan.source_contract.descriptor_world", "world mismatch")
    resolved_world = _validated_world(resolved_world)
    resolved_manifest = (
        load_wdl_state_bank_manifest()
        if manifest is None and resolved_world == WORLD_WDL
        else load_accepted_state_bank_manifest()
        if manifest is None
        else manifest
    )
    resolved_policies = DEFAULT_FIELD_POLICIES if policies is None else policies
    expected = _build_plan(
        manifest=resolved_manifest,
        policies=resolved_policies,
        tape_registry=policy_tape_registry,
        world=resolved_world,
    )
    if dict(plan) != expected:
        _fail("descriptor_plan", "does not exactly match generated registry authority")


def descriptor_plan_json_bytes(
    plan: Mapping[str, Any] | None = None,
    *,
    world: str | None = None,
) -> bytes:
    resolved = (
        generate_compact_summary_checkpoint_descriptor_plan(
            world=WORLD_FLAT if world is None else world
        )
        if plan is None
        else dict(plan)
    )
    validate_compact_summary_checkpoint_descriptor_plan(resolved, world=world)
    return (
        json.dumps(
            resolved,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_compact_summary_checkpoint_descriptor_plan(
    path: str | Path,
    *,
    world: str = WORLD_FLAT,
) -> Path:
    destination = Path(path)
    destination.write_bytes(descriptor_plan_json_bytes(world=world))
    return destination


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    parser.add_argument(
        "--world",
        choices=sorted(SUPPORTED_DESCRIPTOR_WORLDS),
        default=WORLD_FLAT,
    )
    arguments = parser.parse_args()
    payload = descriptor_plan_json_bytes(world=arguments.world)
    if arguments.write is None:
        print(payload.decode("utf-8"), end="")
    else:
        arguments.write.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "CUDA_HASH_BATCH_CAPACITY",
    "DESCRIPTOR_SCHEMA_VERSION",
    "FLAT_PROJECTION_FIXTURE",
    "FLAT_PROJECTION_FIXTURE_SHA256",
    "MODE_ACTIVE_PREFIX",
    "MODE_DEAD_ROW_ZERO",
    "MODE_EXACT",
    "MODE_TORCH_RNG_PACKED_PREFIX",
    "PATCH_P2_PROFILE",
    "PHASE_E_MODULAR_OVERLAY_DESCRIPTOR_SCHEMA_VERSION",
    "PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION",
    "PHASE_E_MODULAR_OVERLAY_STATUS",
    "PLAN_SCHEMA_VERSION",
    "SUPPORTED_DESCRIPTOR_WORLDS",
    "WORLD_FLAT",
    "WORLD_PATCH",
    "WORLD_WDL",
    "WDL_STATE_FIELDS",
    "CompactSummaryCheckpointDescriptorPlanError",
    "descriptor_plan_json_bytes",
    "generate_phase_e_modular_checkpoint_descriptor_overlay",
    "generate_compact_summary_checkpoint_descriptor_plan",
    "maybe_generate_phase_e_modular_checkpoint_descriptor_overlay",
    "validate_compact_summary_checkpoint_descriptor_plan",
    "validate_phase_e_modular_checkpoint_descriptor_overlay",
    "write_compact_summary_checkpoint_descriptor_plan",
]
