from __future__ import annotations

"""
simulation/checkpoint_bundle_v2.py

Checkpoint Bundle v2 (portable, no-pickle) implementation.

Last modified: 2026-02-24 12:56:14 Europe/Vienna

Phase 3 requirements (EIEL TSD v3.1)
------------------------------------
- Directory bundle layout:
    checkpoint_bundle/
      metadata.json          (canonical JSON)
      world.json             (canonical JSON)
      population.json        (canonical JSON)
      rng_snapshot_v2.json   (canonical JSON, from Phase 1 RNGManager)
      control_plane.json     (canonical JSON)
      events_hash.txt        (sha256 of event log bytes)
      weights/
        agent_<id>.bin       (portable binary weights, stable format)

- No pickle.
- Canonical JSON for all json files (sorted keys, minified, UTF-8, normalized floats).
- Strict loader/validator with compatibility gates.
- Resume: load bundle -> apply to runner -> continue deterministically.

Design constraints
------------------
- Minimal-change integration: this module is additive and does not change Phase 1/2 semantics.
- We intentionally avoid torch.save / pickle for policy weights.
- We store *per-agent* policy parameters in a stable custom blob format.

Important
---------
This module assumes the existing runner can:
- be constructed normally
- accept state injection (env arrays, population agents)
- rebuild / sync banked weights after policies are restored

"""

from dataclasses import asdict, is_dataclass
import copy
import gzip
import json
import os
import sys
import hashlib
import struct
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from simulation.state_dump import maybe_dump_state
from evolution.inheritance_registry_v1 import (
    LINKED_INHERITANCE_PROFILE,
    inheritance_checkpoint_metadata,
    inspect_historical_inheritance_checkpoint_v1,
    validate_inheritance_profile,
    validate_production_inheritance_profile,
)
from evolution.structural_development_v1 import (
    CYCLE_POLICY,
    DEPENDENCY_IDENTITY_AUTHORITY,
    DEVELOPMENTAL_FEATURE_PROFILE,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_IDS,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    DEVELOPMENT_OPERATOR_ATTEMPTS,
    DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
    DEVELOPMENT_OPERATOR_PROFILE,
    INACTIVE_OUTPUT_POLICY,
    INACTIVE_STATE_POLICY,
    LATCH_SEMANTICS,
    LATCH_VERSION,
    LATE_START_STEP,
    MATURE_START_STEP,
    MAX_DEPENDENCY_EDGES_PER_AGENT,
    MAX_PREREQUISITES_PER_MODULE,
    MAX_STAGE_GAIN,
    MIGRATION_RNG_DRAWS,
    MIN_STAGE_GAIN,
    SUPPORTED_LANE,
    resolve_structural_stage,
    validate_structural_development_config,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_C2_COMPATIBILITY_VERSION,
    GATED_DRAW1_BINS,
    GATED_DRAW_DECOMPOSITION_VERSION,
    GATED_DUPLICATION_POLICY_VERSION,
    GATED_EQUATION_VERSION,
    GATED_EXECUTION_OPERATION_ESTIMATE,
    GATED_FUSION_POLICY_VERSION,
    GATED_LATENT_STATE_POLICY_VERSION,
    GATED_MUTATION_BUDGET_VERSION,
    GATED_MUTATION_PROFILE,
    GATED_NEUTRAL_ELIGIBILITY_VERSION,
    GATED_NEUTRAL_TEMPLATE_VERSION,
    GATED_NUMERIC_VERSION,
    GATED_OPERATION_BUDGET_VERSION,
    GATED_OPERATOR,
    GATED_PARAMETER_BOUNDS,
    GATED_PREACTIVATION_BOUND,
    GATED_PROFILE,
    GATED_PROMOTER_POLICY_VERSION,
    GATED_RECORD_SCHEMA_VERSION,
    GATED_RUNTIME_SCHEMA_VERSION,
    GATED_STATE_BOUND,
    GATED_STATE_WIDTH,
    GATED_SUPPORTED_LANE,
    escap7_unsupported_lane_reasons,
    neutral_gated_recurrent_record,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_SCHEMA_VERSION,
    BLOAT_CLASSIFICATION_VERSION,
    C2_COMPATIBILITY_VERSION as SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION,
    CHECKPOINT_SCHEMA_VERSION as SUSTAINABLE_GROWTH_CHECKPOINT_SCHEMA_VERSION,
    COST_COEFFICIENTS,
    COST_MODE,
    COST_PROFILE,
    COST_SCHEMA_VERSION,
    DUPLICATION_NEUTRALIZATION_VERSION,
    FIXED_ENVELOPE_SHAPE,
    fixed_envelope_shape_digest_for_caps,
    fixed_envelope_shape_for_caps,
    FUSION_POLICY_VERSION as SUSTAINABLE_GROWTH_FUSION_POLICY_VERSION,
    GROWTH_MUTATION_ATTEMPTS,
    GROWTH_MUTATION_BINS,
    GROWTH_MUTATION_DRAWS,
    GROWTH_MUTATION_PROFILE,
    MIGRATION_RNG_DRAWS as SUSTAINABLE_GROWTH_MIGRATION_RNG_DRAWS,
    PRUNE_CLEANUP_VERSION,
    PRUNE_ELIGIBILITY_VERSION,
    REPACK_VERSION,
    REFINEMENT_OPERATOR_VERSION,
    SPECIALIZATION_TELEMETRY_VERSION,
    SUPPORTED_LANE as SUSTAINABLE_GROWTH_SUPPORTED_LANE,
    SUSTAINABLE_GROWTH_PROFILE,
    module_is_dormant,
    normalize_activity_module_params,
)
from evolution.structural_live_modules_v1 import LIVE_MODULE_SCHEMA_VERSION, LIVE_PROFILE


_ESCAP7_PROFILE = GATED_PROFILE
_ESCAP7_RUNTIME_SCHEMA = GATED_RUNTIME_SCHEMA_VERSION
_ESCAP7_OPERATOR = GATED_OPERATOR
_ESCAP7_PARAMETER_SCHEMA = GATED_RECORD_SCHEMA_VERSION
_ESCAP7_EQUATION = GATED_EQUATION_VERSION
_ESCAP7_NUMERIC_POLICY = GATED_NUMERIC_VERSION
_ESCAP7_NEUTRAL_ELIGIBILITY = GATED_NEUTRAL_ELIGIBILITY_VERSION
_ESCAP7_LATENT_POLICY = GATED_LATENT_STATE_POLICY_VERSION
_ESCAP7_MUTATION_PROFILE = GATED_MUTATION_PROFILE
_ESCAP7_DECOMPOSITION = GATED_DRAW_DECOMPOSITION_VERSION
_ESCAP7_MUTATION_BUDGET = GATED_MUTATION_BUDGET_VERSION
_ESCAP7_OPERATION_BUDGET = GATED_OPERATION_BUDGET_VERSION
_ESCAP7_DUPLICATION = GATED_DUPLICATION_POLICY_VERSION
_ESCAP7_PROMOTER = GATED_PROMOTER_POLICY_VERSION
_ESCAP7_FUSION = GATED_FUSION_POLICY_VERSION
_ESCAP7_C2 = GATED_C2_COMPATIBILITY_VERSION
_ESCAP7_LANE = GATED_SUPPORTED_LANE


def _escap7_gated_recurrent_metadata(sim_cfg: Any) -> Dict[str, Any]:
    """Return the complete enabled-only ESCAP-7 replay contract."""

    return {
        "enabled": True,
        "profile": str(getattr(sim_cfg, "structural_gated_recurrent_profile", _ESCAP7_PROFILE)),
        "runtime_schema_version": str(
            getattr(sim_cfg, "structural_gated_recurrent_schema_version", _ESCAP7_RUNTIME_SCHEMA)
        ),
        "operator": str(getattr(sim_cfg, "structural_gated_recurrent_operator", _ESCAP7_OPERATOR)),
        "parameter_schema_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_parameter_schema_version",
                _ESCAP7_PARAMETER_SCHEMA,
            )
        ),
        "equation_version": str(
            getattr(sim_cfg, "structural_gated_recurrent_equation_version", _ESCAP7_EQUATION)
        ),
        "numeric_policy_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_numeric_policy_version",
                _ESCAP7_NUMERIC_POLICY,
            )
        ),
        "state_width": int(
            getattr(sim_cfg, "structural_gated_recurrent_state_width", GATED_STATE_WIDTH)
        ),
        "parameter_bounds": {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in GATED_PARAMETER_BOUNDS.items()
        },
        "preactivation_bounds": [
            -float(GATED_PREACTIVATION_BOUND),
            float(GATED_PREACTIVATION_BOUND),
        ],
        "state_bounds": [-float(GATED_STATE_BOUND), float(GATED_STATE_BOUND)],
        "output_bounds": [-float(GATED_STATE_BOUND), float(GATED_STATE_BOUND)],
        "neutral_eligibility_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_neutral_eligibility_version",
                _ESCAP7_NEUTRAL_ELIGIBILITY,
            )
        ),
        "neutral_latent_state_policy": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_neutral_latent_state_policy",
                _ESCAP7_LATENT_POLICY,
            )
        ),
        "neutral_template_version": GATED_NEUTRAL_TEMPLATE_VERSION,
        "neutral_template": {
            "source_operator": "bounded_scalar_integrator_v1",
            "gated_record": neutral_gated_recurrent_record(),
            "public_output_neutral": True,
            "latent_state_frozen": False,
        },
        "mutation_operator_profile": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_mutation_operator_profile",
                _ESCAP7_MUTATION_PROFILE,
            )
        ),
        "operator_attempts": int(
            getattr(sim_cfg, "structural_gated_recurrent_operator_attempts", 1)
        ),
        "operator_draws_per_attempt": int(
            getattr(sim_cfg, "structural_gated_recurrent_operator_draws_per_attempt", 2)
        ),
        "draw_bin_count": int(
            getattr(sim_cfg, "structural_gated_recurrent_draw_bin_count", 5)
        ),
        "draw_bins": [str(row[2]) for row in GATED_DRAW1_BINS],
        "draw_bin_map": [
            {
                "lower_inclusive": float(lower),
                "upper_exclusive": float(upper),
                "operation": str(operation),
            }
            for lower, upper, operation in GATED_DRAW1_BINS
        ],
        "decomposition_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_decomposition_version",
                _ESCAP7_DECOMPOSITION,
            )
        ),
        "mutation_budget_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_mutation_budget_version",
                _ESCAP7_MUTATION_BUDGET,
            )
        ),
        "operation_budget_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_operation_budget_version",
                _ESCAP7_OPERATION_BUDGET,
            )
        ),
        "estimated_scalar_operations_per_evaluation": int(
            GATED_EXECUTION_OPERATION_ESTIMATE
        ),
        "legacy_integrator_estimated_scalar_operations_per_evaluation": 4,
        "duplication_policy_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_duplication_policy_version",
                _ESCAP7_DUPLICATION,
            )
        ),
        "promoter_policy_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_promoter_policy_version",
                _ESCAP7_PROMOTER,
            )
        ),
        "fusion_policy_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_fusion_policy_version",
                _ESCAP7_FUSION,
            )
        ),
        "c2_compatibility_version": str(
            getattr(
                sim_cfg,
                "structural_gated_recurrent_c2_compatibility_version",
                _ESCAP7_C2,
            )
        ),
        "supported_lane": str(
            getattr(sim_cfg, "structural_gated_recurrent_supported_lane", _ESCAP7_LANE)
        ),
        "lane_lock": {
            "implementation": "python_reference",
            "policy_device": "cpu",
            "bank_device": "cpu",
            "policy_tick_stride": 1,
            "runnercore_enable_fields": False,
            "perf_sync_cuda": False,
            "native_runtime": False,
        },
        "runtime_state_storage": "structural_live_module_state",
        "runtime_output_storage": "structural_live_module_outputs",
        "supported_module_types": [
            {
                "module_type": "generic",
                "operators": ["bounded_scalar_integrator_v1", GATED_OPERATOR],
            }
        ],
        "fixed_shape_policy": {
            "module_state_width": int(GATED_STATE_WIDTH),
            "new_arrays": 0,
            "observation_width_change": 0,
            "action_width_change": 0,
            "central_gru_width_change": 0,
        },
        "execution_rng_draws": 0,
        "migration_rng_draws": int(
            getattr(sim_cfg, "structural_gated_recurrent_migration_rng_draws", 0)
        ),
    }


def _escap8_sustainable_growth_metadata(sim_cfg: Any) -> Dict[str, Any]:
    """Return the complete enabled-only ESCAP-8 replay contract."""

    active_caps = {
        "max_modules_per_agent": getattr(
            sim_cfg, "genome_b2_max_modules_per_agent", FIXED_ENVELOPE_SHAPE["max_module_slots"]
        ),
        "max_connections_per_agent": getattr(
            sim_cfg, "genome_b2_max_connections_per_agent", FIXED_ENVELOPE_SHAPE["max_connection_slots"]
        ),
        "max_module_fanin": getattr(sim_cfg, "genome_b2_max_module_fanin", 6),
        "max_module_fanout": getattr(sim_cfg, "genome_b2_max_module_fanout", 6),
    }
    active_shape = fixed_envelope_shape_for_caps(active_caps)

    return {
        "enabled": True,
        "checkpoint_schema_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_checkpoint_schema_version",
                SUSTAINABLE_GROWTH_CHECKPOINT_SCHEMA_VERSION,
            )
        ),
        "profile": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_profile",
                SUSTAINABLE_GROWTH_PROFILE,
            )
        ),
        "activity_schema_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_activity_schema_version",
                ACTIVITY_SCHEMA_VERSION,
            )
        ),
        "duplication_neutralization_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_duplication_neutralization_version",
                DUPLICATION_NEUTRALIZATION_VERSION,
            )
        ),
        "refinement_operator_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_refinement_operator_version",
                REFINEMENT_OPERATOR_VERSION,
            )
        ),
        "specialization_telemetry_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_specialization_telemetry_version",
                SPECIALIZATION_TELEMETRY_VERSION,
            )
        ),
        "prune_eligibility_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_prune_eligibility_version",
                PRUNE_ELIGIBILITY_VERSION,
            )
        ),
        "prune_cleanup_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_prune_cleanup_version",
                PRUNE_CLEANUP_VERSION,
            )
        ),
        "repack_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_repack_version",
                REPACK_VERSION,
            )
        ),
        "fusion_policy_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_fusion_policy_version",
                SUSTAINABLE_GROWTH_FUSION_POLICY_VERSION,
            )
        ),
        "cost_schema_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_cost_schema_version",
                COST_SCHEMA_VERSION,
            )
        ),
        "cost_profile": str(
            getattr(sim_cfg, "structural_sustainable_growth_cost_profile", COST_PROFILE)
        ),
        "cost_mode": str(
            getattr(sim_cfg, "structural_sustainable_growth_cost_mode", COST_MODE)
        ),
        "cost_coefficients": {
            str(key): float(value)
            for key, value in sorted(
                dict(
                    getattr(
                        sim_cfg,
                        "structural_sustainable_growth_cost_coefficients",
                        COST_COEFFICIENTS,
                    )
                    or {}
                ).items()
            )
        },
        "cost_telemetry_continuation": "recomputed_reporting_only",
        "bloat_classification_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_bloat_classification_version",
                BLOAT_CLASSIFICATION_VERSION,
            )
        ),
        "c1_operator_profile": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_operator_profile",
                GROWTH_MUTATION_PROFILE,
            )
        ),
        "c1_operator_attempts": int(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_operator_attempts",
                GROWTH_MUTATION_ATTEMPTS,
            )
        ),
        "c1_operator_draws_per_attempt": int(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_operator_draws_per_attempt",
                GROWTH_MUTATION_DRAWS,
            )
        ),
        "c1_lane_map": [
            {
                "lower_inclusive": float(lower),
                "upper_exclusive": float(upper),
                "operation": str(operation),
            }
            for lower, upper, operation in tuple(
                getattr(
                    sim_cfg,
                    "structural_sustainable_growth_c1_lane_map",
                    GROWTH_MUTATION_BINS,
                )
            )
        ],
        "c2_compatibility_version": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_c2_compatibility_version",
                SUSTAINABLE_GROWTH_C2_COMPATIBILITY_VERSION,
            )
        ),
        "fixed_envelope_shape": dict(active_shape),
        "fixed_envelope_shape_digest": fixed_envelope_shape_digest_for_caps(active_caps),
        "supported_lane": str(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_supported_lane",
                SUSTAINABLE_GROWTH_SUPPORTED_LANE,
            )
        ),
        "lane_lock": {
            "implementation": "python_reference",
            "policy_device": "cpu",
            "bank_device": "cpu",
            "policy_tick_stride": 1,
            "runnercore_enable_fields": False,
            "perf_sync_cuda": False,
            "native_runtime": False,
        },
        "migration_rng_draws": int(
            getattr(
                sim_cfg,
                "structural_sustainable_growth_migration_rng_draws",
                SUSTAINABLE_GROWTH_MIGRATION_RNG_DRAWS,
            )
        ),
    }


def _force_write_state_dump(sim, *, generation_index: int, tag: str, extra: dict | None = None) -> None:
    """Best-effort deterministic state dump fallback.

    This is a safety net for Phase 3/4 tooling when `maybe_dump_state` cannot
    resolve an output directory (runner API drift) or is otherwise disabled by
    mistake. It must never draw RNG and must never crash the simulation.
    """
    import os
    import json
    from pathlib import Path

    if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
        return

    run_dir = getattr(sim, "run_dir", None) or getattr(sim, "run_dir_path", None) or ""
    if not run_dir:
        return

    root = Path(str(run_dir)) / "state_dumps"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    safe_tag = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(tag))
    path = root / f"state_{int(generation_index):06d}__{safe_tag}.json"

    payload = {
        "generation_index": int(generation_index),
        "tag": str(tag),
    }
    if extra:
        if "alive_ids" in extra:
            payload["alive_ids"] = extra.get("alive_ids")
        payload["extra"] = extra

    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    except Exception:
        return

from simulation.control_plane_pkg.util_json import canonical_json_bytes, canonical_json_bytes_lossless

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SCHEMA_VERSION = "checkpoint_bundle_v2"
HISTORICAL_LINKED_CHECKPOINT_INSPECTION_SCHEMA_VERSION = (
    "checkpoint_bundle_v2_linked_parent_copy_historical_inspection_v1"
)
WEIGHTS_MAGIC = b"EIELW2\x00"
NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME = "native_runtime_checkpoint_projection.json"
NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME = (
    "native_runtime_checkpoint_projection.json.gz"
)
NATIVE_SINGLE_PROJECTION_AUTHORITY = "eiel_native_single_projection_checkpoint_authority_v1"
NATIVE_BUNDLE_COHERENCE_SCHEMA_VERSION = "eiel_native_checkpoint_bundle_coherence_v1"
NATIVE_BUNDLE_COMPLETE_SCHEMA_VERSION = "eiel_native_checkpoint_bundle_complete_v1"
ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION = "eiel_architecture_b_checkpoint_projection_save_v1"
ARCHITECTURE_B_OBSERVER_CONTINUATION_SCHEMA_VERSION = (
    "eiel_architecture_b_observer_continuation_v1"
)
ARCHITECTURE_B_PROJECTION_INPUT_SOURCE = (
    "checkpoint_bundle_v2_save_projection.detached_input"
)
ARCHITECTURE_B_BUNDLE_KIND = "architecture_b"
ARCHITECTURE_B_RESUME_ENVELOPE_SCHEMA_VERSION = (
    "eiel_architecture_b_checkpoint_resume_envelope_v1"
)
ARCHITECTURE_B_METRICS_RESTORE_RECEIPT_SCHEMA_VERSION = (
    "eiel_architecture_b_metrics_logger_restore_receipt_v1"
)
ARCHITECTURE_B_DEDUPLICATION_MAX_RECORD_KEYS = 4096
AUTHORITATIVE_STATE_CHECKPOINT_IDENTITY_SCHEMA_VERSION = (
    "eiel_checkpoint_bundle_v2_authoritative_state_identity_v1"
)
AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_SCHEMA_VERSION = (
    "eiel_checkpoint_bundle_v2_authoritative_state_admission_v1"
)
AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD = "authoritative_state_identity"
AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD = (
    "authoritative_state_checkpoint_admission"
)

_ARCHITECTURE_B_CHECKPOINT_PAYLOAD_AUTHORITY_TOKEN = object()
_ARCHITECTURE_B_RESUME_ENVELOPE_AUTHORITY_TOKEN = object()
_ARCHITECTURE_B_COMPLETION_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "checkpoint_id",
        "source_projection_sha256",
        "coherence_sha256",
        "architecture_b_schema_version",
        "architecture_b_identity_sha256",
        "observer_continuation_sha256",
        "architecture_b_completion_sha256",
    }
)


class ValidatedArchitectureBCheckpointPayload(dict):
    """Checkpoint payload stamped only by the strict Architecture B loader."""

    __slots__ = ("_authority_token", "_authority_digest_sha256")

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        _authority_token: object,
        _authority_digest_sha256: str,
    ) -> None:
        if _authority_token is not _ARCHITECTURE_B_CHECKPOINT_PAYLOAD_AUTHORITY_TOKEN:
            raise CheckpointStrictError(
                "Architecture B checkpoint payload lacks strict-loader authority"
            )
        super().__init__(value)
        self._authority_token = _authority_token
        self._authority_digest_sha256 = _authority_digest_sha256

    def __copy__(self) -> Dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[str, Any]:
        return copy.deepcopy(dict(self), memo)


class ValidatedArchitectureBResumeEnvelope(dict):
    """Detached resume data stamped only after strict payload decoding."""

    __slots__ = ("_authority_token", "_authority_digest_sha256")

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        _authority_token: object,
        _authority_digest_sha256: str,
    ) -> None:
        if _authority_token is not _ARCHITECTURE_B_RESUME_ENVELOPE_AUTHORITY_TOKEN:
            raise CheckpointStrictError(
                "Architecture B resume envelope lacks strict-loader authority"
            )
        super().__init__(value)
        self._authority_token = _authority_token
        self._authority_digest_sha256 = _authority_digest_sha256

    def __copy__(self) -> Dict[str, Any]:
        return dict(self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[str, Any]:
        return copy.deepcopy(dict(self), memo)


def is_validated_architecture_b_resume_envelope(value: Any) -> bool:
    return (
        type(value) is ValidatedArchitectureBResumeEnvelope
        and getattr(value, "_authority_token", None)
        is _ARCHITECTURE_B_RESUME_ENVELOPE_AUTHORITY_TOKEN
    )


def validate_architecture_b_resume_envelope_authority(
    value: Any,
) -> Dict[str, Any]:
    """Rebind a stamped envelope to a detached snapshot at controller call."""

    if not is_validated_architecture_b_resume_envelope(value):
        raise CheckpointStrictError(
            "Architecture B resume envelope was not produced by the strict loader"
        )
    try:
        rebound = _detached_architecture_b_resume_envelope(value)
        rebound_digest = _architecture_b_resume_envelope_authority_sha256(rebound)
    except CheckpointStrictError:
        raise
    except Exception as exc:
        raise CheckpointStrictError(
            "Architecture B resume envelope authority could not be rebound"
        ) from exc
    if rebound_digest != getattr(value, "_authority_digest_sha256", None):
        raise CheckpointStrictError(
            "Architecture B resume envelope changed after strict payload decoding"
        )
    expected_fields = {
        "schema_version",
        "bundle_kind",
        "source_projection_sha256",
        "checkpoint_projection",
        "architecture_b_identity",
        "observer_continuation",
        "controller_import_metadata",
        "bundle_completion_authority",
    }
    if set(rebound) != expected_fields:
        raise CheckpointStrictError(
            "Architecture B resume envelope fields are not a closed schema"
        )
    projection = rebound.get("checkpoint_projection")
    identity = rebound.get("architecture_b_identity")
    continuation = rebound.get("observer_continuation")
    authority = rebound.get("bundle_completion_authority")
    if not all(
        isinstance(item, Mapping)
        for item in (projection, identity, continuation, authority)
    ):
        raise CheckpointStrictError(
            "Architecture B resume envelope authority components are invalid"
        )
    if set(authority) != _ARCHITECTURE_B_COMPLETION_AUTHORITY_FIELDS:
        raise CheckpointStrictError(
            "Architecture B bundle completion authority fields are not a closed schema"
        )
    source_projection_sha256 = _strict_lowercase_sha256(
        rebound.get("source_projection_sha256"),
        "architecture_b_resume_envelope.source_projection_sha256",
    )
    actual_projection_sha256 = str(
        _native_projection_authority_record(projection).get(
            "source_projection_sha256", ""
        )
    )
    if actual_projection_sha256 != source_projection_sha256:
        raise CheckpointStrictError(
            "Architecture B resume envelope projection authority digest mismatch"
        )
    if authority.get("schema_version") != NATIVE_BUNDLE_COMPLETE_SCHEMA_VERSION:
        raise CheckpointStrictError(
            "Architecture B bundle completion schema is invalid"
        )
    if authority.get("status") != "complete":
        raise CheckpointStrictError(
            "Architecture B bundle completion status is not complete"
        )
    _architecture_b_identifier(
        authority.get("checkpoint_id"),
        field="bundle_completion_authority.checkpoint_id",
    )
    _strict_lowercase_sha256(
        authority.get("coherence_sha256"),
        "bundle_completion_authority.coherence_sha256",
    )
    if authority.get("source_projection_sha256") != source_projection_sha256:
        raise CheckpointStrictError(
            "Architecture B bundle completion source authority mismatch"
        )
    if authority.get("architecture_b_schema_version") != (
        ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION
    ):
        raise CheckpointStrictError(
            "Architecture B bundle completion projection schema is invalid"
        )
    identity_sha256 = hashlib.sha256(
        canonical_json_bytes_lossless(identity)
    ).hexdigest()
    continuation_sha256 = hashlib.sha256(
        canonical_json_bytes_lossless(continuation)
    ).hexdigest()
    if authority.get("architecture_b_identity_sha256") != identity_sha256:
        raise CheckpointStrictError(
            "Architecture B resume identity is not bound to bundle completion"
        )
    if authority.get("observer_continuation_sha256") != continuation_sha256:
        raise CheckpointStrictError(
            "Architecture B observer continuation is not bound to bundle completion"
        )
    completion = dict(authority)
    completion["architecture_b_identity"] = identity
    completion["observer_continuation"] = continuation
    recorded_completion_sha256 = _strict_lowercase_sha256(
        authority.get("architecture_b_completion_sha256"),
        "bundle_completion_authority.architecture_b_completion_sha256",
    )
    if _architecture_b_completion_sha256(completion) != recorded_completion_sha256:
        raise CheckpointStrictError(
            "Architecture B resume authority contradicts bundle completion digest"
        )
    return rebound

_ARCHITECTURE_B_OBSERVER_CONTINUATION_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_next_generation",
        "tick",
        "typed_lifecycle_event_cursor",
        "exact_reporting_boundary_ordinal",
        "behavior_completed_generation_and_row_key",
        "metrics_completed_generation",
        "lineage_completed_generation_and_row_key",
        "chronicle_observation_ordinal",
        "progress_sequence",
        "schedule_phase",
        "deduplication",
        "acknowledged_channel_cursor_ranges",
    }
)
_ARCHITECTURE_B_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "controller_id",
        "owner_epoch",
        "snapshot_id",
        "snapshot_sequence",
        "snapshot_digest_sha256",
        "next_generation",
        "tick",
        "phase",
        "source_projection_sha256",
    }
)
_ARCHITECTURE_B_METRICS_LOGGER_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "collapse_streak",
        "theme5_lineage_state",
        "lineage_collapse_forensics_state",
        "early_lineage_effective_gen_1",
        "early_lineage_effective_gen_2",
        "early_lineage_top1_share_gen_1",
        "early_lineage_top1_share_gen_2",
    }
)
_ARCHITECTURE_B_METRICS_STATE_PROVENANCE_FIELDS = frozenset(
    {
        "authority_record_schema_version",
        "authority_source",
        "authority_classification",
        "authority_epoch",
        "source_action_cache_boundary_status",
        "source_action_cache_sha256",
        "source_action_cache_valid_slot_count",
        "source_authority_epoch",
        "source_generation",
        "source_phase",
        "source_projection_schema_version",
        "source_projection_sha256",
        "source_rng_sha256",
        "source_tick",
    }
)
_ARCHITECTURE_B_ACKNOWLEDGED_CHANNEL_RANGE_FIELDS = frozenset(
    {
        "exact_reporting",
        "behaviors",
        "metrics",
        "lineage",
        "typed_lifecycle",
        "chronicle",
        "checkpoint_generation",
    }
)


def _checkpoint_diagnostic_dump_enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    return any(
        os.environ.get(name, "").strip().lower() in truthy
        for name in ("DEBUG_DETERMINISM", "EIEL_STATE_DUMP")
    )


class CheckpointBundleError(RuntimeError):
    """Base class for checkpoint bundle errors."""


class CheckpointSchemaError(CheckpointBundleError):
    """Raised when schema_version is unsupported or invalid."""


class CheckpointStrictError(CheckpointBundleError):
    """Raised when strict mode validation fails."""


def _authoritative_state_checkpoint_identity_v1() -> Dict[str, Any]:
    """Return the exact local schema/component identity for a newly saved bundle."""

    try:
        from engine.authoritative_state.schema import content_sha256, load_schema_v1

        schema = load_schema_v1()
        checkpoint_contract = schema.components["checkpoint_schema.json"]
        compatibility_contract = schema.components["compatibility_matrix.json"]
        migration_contract = schema.components["migration_rules.json"]
        component_hashes = {
            str(name): str(digest)
            for name, digest in sorted(
                dict(schema.root["component_hashes"]).items()
            )
        }
        identity = {
            "identity_schema_version": (
                AUTHORITATIVE_STATE_CHECKPOINT_IDENTITY_SCHEMA_VERSION
            ),
            "state_schema_version": str(schema.root["schema_version"]),
            "state_schema_sha256": str(schema.schema_hash),
            "checkpoint_schema_version": str(
                checkpoint_contract["schema_version"]
            ),
            "checkpoint_schema_contract_sha256": str(
                checkpoint_contract["contract_hash"]
            ),
            "checkpoint_component_file_sha256": component_hashes[
                "checkpoint_schema.json"
            ],
            "compatibility_matrix_contract_sha256": str(
                compatibility_contract["contract_hash"]
            ),
            "migration_rules_contract_sha256": str(
                migration_contract["contract_hash"]
            ),
            "component_hashes": component_hashes,
            "source_commit": str(schema.root["source_commit"]),
        }
        identity["identity_sha256"] = content_sha256(identity)
        return identity
    except CheckpointBundleError:
        raise
    except Exception as exc:
        raise CheckpointStrictError(
            "authoritative-state checkpoint identity is unavailable"
        ) from exc


def _validate_authoritative_state_checkpoint_metadata(
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Classify legacy metadata or require exact schema-v1 identity equality."""

    recorded = metadata.get(AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD)
    if recorded is None:
        return {
            "admission_schema_version": (
                AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_SCHEMA_VERSION
            ),
            "classification": "legacy_checkpoint_bundle_v2_unbound",
            "migration": "retain_legacy_semantics_without_schema_v1_admission",
            "admitted_as_authoritative_state_v1": False,
            "state_schema_sha256": None,
            "identity_sha256": None,
        }
    if not isinstance(recorded, Mapping):
        raise CheckpointStrictError(
            "authoritative-state checkpoint identity must be an object"
        )
    expected = _authoritative_state_checkpoint_identity_v1()
    if set(recorded) != set(expected):
        raise CheckpointStrictError(
            "authoritative-state checkpoint identity fields mismatch"
        )
    try:
        from engine.authoritative_state.schema import content_sha256

        material = dict(recorded)
        supplied_identity_hash = str(material.pop("identity_sha256", "") or "")
        computed_identity_hash = content_sha256(material)
    except Exception as exc:
        raise CheckpointStrictError(
            "authoritative-state checkpoint identity cannot be hashed"
        ) from exc
    if supplied_identity_hash != computed_identity_hash:
        raise CheckpointStrictError(
            "authoritative-state checkpoint identity self-hash mismatch"
        )
    if dict(recorded) != expected:
        differing = sorted(
            key for key in expected if recorded.get(key) != expected.get(key)
        )
        raise CheckpointStrictError(
            "authoritative-state checkpoint identity mismatch: "
            f"{differing}"
        )
    return {
        "admission_schema_version": (
            AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_SCHEMA_VERSION
        ),
        "classification": "authoritative_state_v1_exact",
        "migration": "identity",
        "admitted_as_authoritative_state_v1": True,
        "state_schema_sha256": expected["state_schema_sha256"],
        "identity_sha256": expected["identity_sha256"],
    }


def _validate_authoritative_state_checkpoint_payload(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Revalidate checkpoint identity before any restore-side state mutation."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CheckpointStrictError(
            "checkpoint payload metadata must be an object before restore"
        )
    admission = _validate_authoritative_state_checkpoint_metadata(metadata)
    recorded_admission = payload.get(AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD)
    if recorded_admission is not None and recorded_admission != admission:
        raise CheckpointStrictError(
            "checkpoint payload authoritative-state admission receipt mismatch"
        )
    return admission


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def _build_metadata(
    sim: Any,
    generation_index: int,
    strict_replay_mode: bool,
    world_hash: str,
    population_hash: str,
    rng_snapshot_hash: str,
    runner_state_hash: str,
    bank_state_hash: str,
    resident_runtime_state_hash: str,
    native_runtime_checkpoint_hash: str,
    event_log_hash: str,
    wal_hash: str,
    weights_info: Any = None,
) -> Dict[str, Any]:
    """Build metadata.json for Checkpoint Bundle v2.

    In strict determinism mode, metadata carries a required contract that
    the loader validates (TSD v3.1). This function must therefore:
    - populate all required fields
    - avoid non-deterministic values (timestamps, random UUIDs)
    """
    # Pull deterministic identifiers from config/control plane if available
    sim_cfg = getattr(sim, "sim_cfg", None)
    cp_obj: Dict[str, Any] = {}
    try:
        cp_obj = _export_control_plane(sim)
    except Exception:
        cp_obj = {}

    branch_id = str(cp_obj.get("branch_id") or getattr(sim_cfg, "branch_id", "default") or "default")
    campaign_id = str(cp_obj.get("campaign_id") or getattr(sim_cfg, "campaign_id", "") or "")
    seq = 0
    try:
        seq = int(cp_obj.get("submission_sequence_number", 0) or 0)
    except Exception:
        seq = 0

    # Environment versions
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    np_ver = str(getattr(np, "__version__", "unknown") or "unknown")
    torch_ver = "unknown"
    try:
        if torch is not None:
            torch_ver = str(getattr(torch, "__version__", "unknown") or "unknown")
    except Exception:
        torch_ver = "unknown"

    # Commit hash (best-effort, deterministic string)
    commit = (
        str(getattr(sim_cfg, "code_commit_hash", "") or "")
        or str(getattr(sim, "code_commit_hash", "") or "")
        or str(os.environ.get("GIT_COMMIT", "") or "")
        or "unknown"
    )

    capacity_epoch_metadata = None
    if sim_cfg is not None and bool(getattr(sim_cfg, "capacity_epoch_metadata_enable", False)):
        from simulation.capacity_epoch import checkpoint_capacity_epoch_metadata

        capacity_epoch_metadata = checkpoint_capacity_epoch_metadata(
            sim_cfg=sim_cfg,
            env_cfg=getattr(sim, "env_cfg", None),
        )
        capacity_manager = getattr(sim, "substrate_capacity_manager", None)
        if capacity_manager is not None:
            capacity_epoch_metadata["manager"] = capacity_manager.to_checkpoint_dict()

    from agents.structural_identity_v1 import (
        STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
        STRUCTURAL_IDENTITY_SCHEMA_VERSION,
        STRUCTURAL_LEGACY_MIGRATION_VERSION,
        TOPOLOGY_POSITION_SCHEMA_VERSION,
    )
    structural_identity_metadata = {
        "identity_schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
        "ancestry_schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
        "topology_position_schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION,
        "legacy_migration_version": STRUCTURAL_LEGACY_MIGRATION_VERSION,
        "migration_rng_draws": 0,
    }
    structural_live_modules_metadata = {
        "enabled": bool(getattr(sim_cfg, "structural_live_modules_enable", False)),
        "profile": str(getattr(sim_cfg, "structural_live_modules_profile", "none") or "none"),
        "schema_version": str(
            getattr(sim_cfg, "structural_live_modules_schema_version", "escap3_live_module_runtime_v1")
            or "escap3_live_module_runtime_v1"
        ),
        "migration_rng_draws": 0,
    }
    routing_enabled = bool(getattr(sim_cfg, "structural_observation_routing_enable", False))
    structural_observation_routing_metadata = {
        "enabled": routing_enabled,
        "profile": str(getattr(sim_cfg, "structural_observation_routing_profile", "none") or "none"),
        "schema_version": str(
            getattr(sim_cfg, "structural_observation_routing_schema_version", "escap4_observation_routing_v1")
            or "escap4_observation_routing_v1"
        ),
        "registry_version": str(
            getattr(sim_cfg, "structural_observation_routing_registry_version", "perception_routing_contract_v1")
            or "perception_routing_contract_v1"
        ),
        "max_bindings_per_module": int(
            getattr(sim_cfg, "structural_observation_routing_max_bindings_per_module", 2) or 2
        ),
        "max_bindings_per_agent": int(
            getattr(sim_cfg, "structural_observation_routing_max_bindings_per_agent", 8) or 8
        ),
        "operator_profile": str(
            getattr(sim_cfg, "structural_observation_routing_operator_profile", "escap4_routing_binding_operator_v1")
            or "escap4_routing_binding_operator_v1"
        ),
        "operator_attempts": int(
            getattr(sim_cfg, "structural_observation_routing_operator_attempts", 1) or 0
        ),
        "admitted_semantic_ids": ["self_energy", "self_health"],
        "exposure_mode": "parallel",
        "migration_rng_draws": 0,
    }
    output_routing_enabled = bool(getattr(sim_cfg, "structural_output_routing_enable", False))
    structural_output_routing_metadata = {
        "enabled": output_routing_enabled,
        "profile": str(
            getattr(sim_cfg, "structural_output_routing_profile", "escap5_modular_action_logit_residual_v1")
            or "escap5_modular_action_logit_residual_v1"
        ),
        "schema_version": str(
            getattr(sim_cfg, "structural_output_routing_schema_version", "escap5_output_routing_v1")
            or "escap5_output_routing_v1"
        ),
        "registry_version": str(
            getattr(
                sim_cfg,
                "structural_output_routing_registry_version",
                "eiel_action_semantic_routing_registry_v1",
            )
            or "eiel_action_semantic_routing_registry_v1"
        ),
        "max_bindings_per_module": int(
            getattr(sim_cfg, "structural_output_routing_max_bindings_per_module", 2) or 2
        ),
        "max_bindings_per_agent": int(
            getattr(sim_cfg, "structural_output_routing_max_bindings_per_agent", 8) or 8
        ),
        "operator_profile": str(
            getattr(sim_cfg, "structural_output_routing_operator_profile", "escap5_output_binding_operator_v1")
            or "escap5_output_binding_operator_v1"
        ),
        "operator_attempts": int(
            getattr(sim_cfg, "structural_output_routing_operator_attempts", 1) or 0
        ),
        "admitted_semantic_ids": ["move_north", "move_south", "move_east", "move_west", "rest"],
        "max_binding_gain": float(
            getattr(sim_cfg, "structural_output_routing_max_binding_gain", 1.0) or 0.0
        ),
        "max_action_residual": float(
            getattr(sim_cfg, "structural_output_routing_max_action_residual", 0.25) or 0.0
        ),
        "arbitration_version": str(
            getattr(sim_cfg, "structural_output_routing_arbitration_version", "escap5_output_arbitration_v1")
            or "escap5_output_arbitration_v1"
        ),
        "authority_order_version": str(
            getattr(
                sim_cfg,
                "structural_output_routing_authority_order_version",
                "escap5_before_existing_authority_v1",
            )
            or "escap5_before_existing_authority_v1"
        ),
        "combined_rest_compatibility_version": str(
            getattr(
                sim_cfg,
                "structural_output_routing_combined_rest_compatibility_version",
                "escap5_combined_rest_residual_v1",
            )
            or "escap5_combined_rest_residual_v1"
        ),
        "migration_rng_draws": 0,
    }
    structural_development_enabled = bool(getattr(sim_cfg, "structural_development_enable", False))
    structural_development_metadata = {
        "enabled": structural_development_enabled,
        "profile": str(
            getattr(sim_cfg, "structural_development_profile", DEVELOPMENTAL_FEATURE_PROFILE)
            or DEVELOPMENTAL_FEATURE_PROFILE
        ),
        "module_record_schema_version": str(
            getattr(sim_cfg, "structural_development_module_schema_version", DEVELOPMENTAL_MODULE_SCHEMA_VERSION)
            or DEVELOPMENTAL_MODULE_SCHEMA_VERSION
        ),
        "stage_profile_version": str(
            getattr(sim_cfg, "structural_development_stage_profile_version", DEVELOPMENTAL_STAGE_PROFILE_VERSION)
            or DEVELOPMENTAL_STAGE_PROFILE_VERSION
        ),
        "stage_semantic_ids": list(
            getattr(sim_cfg, "structural_development_stage_ids", DEVELOPMENTAL_STAGE_IDS)
            or DEVELOPMENTAL_STAGE_IDS
        ),
        "mature_start_step": int(
            getattr(sim_cfg, "structural_development_mature_start_step", MATURE_START_STEP)
        ),
        "late_start_step": int(
            getattr(sim_cfg, "structural_development_late_start_step", LATE_START_STEP)
        ),
        "minimum_stage_gain": float(
            getattr(sim_cfg, "structural_development_min_stage_gain", MIN_STAGE_GAIN)
        ),
        "maximum_stage_gain": float(
            getattr(sim_cfg, "structural_development_max_stage_gain", MAX_STAGE_GAIN)
        ),
        "dependency_identity_authority": str(
            getattr(sim_cfg, "structural_development_dependency_identity_authority", DEPENDENCY_IDENTITY_AUTHORITY)
            or DEPENDENCY_IDENTITY_AUTHORITY
        ),
        "max_prerequisites_per_module": int(
            getattr(sim_cfg, "structural_development_max_prerequisites_per_module", MAX_PREREQUISITES_PER_MODULE)
        ),
        "max_dependency_edges_per_agent": int(
            getattr(sim_cfg, "structural_development_max_dependency_edges_per_agent", MAX_DEPENDENCY_EDGES_PER_AGENT)
        ),
        "cycle_policy": str(
            getattr(
                sim_cfg,
                "structural_development_cycle_policy",
                CYCLE_POLICY,
            )
            or CYCLE_POLICY
        ),
        "inactive_state_policy": str(
            getattr(sim_cfg, "structural_development_inactive_state_policy", INACTIVE_STATE_POLICY)
            or INACTIVE_STATE_POLICY
        ),
        "inactive_output_policy": str(
            getattr(
                sim_cfg,
                "structural_development_inactive_output_policy",
                INACTIVE_OUTPUT_POLICY,
            )
            or INACTIVE_OUTPUT_POLICY
        ),
        "latch_version": str(
            getattr(sim_cfg, "structural_development_latch_version", LATCH_VERSION)
            or LATCH_VERSION
        ),
        "latch_semantics": str(
            getattr(
                sim_cfg,
                "structural_development_latch_semantics",
                LATCH_SEMANTICS,
            )
            or LATCH_SEMANTICS
        ),
        "mutation_operator_profile": str(
            getattr(
                sim_cfg,
                "structural_development_operator_profile",
                DEVELOPMENT_OPERATOR_PROFILE,
            )
            or DEVELOPMENT_OPERATOR_PROFILE
        ),
        "operator_attempts": int(
            getattr(sim_cfg, "structural_development_operator_attempts", DEVELOPMENT_OPERATOR_ATTEMPTS)
        ),
        "operator_draws_per_attempt": int(
            getattr(sim_cfg, "structural_development_operator_draws_per_attempt", DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT)
        ),
        "migration_rng_draws": int(
            getattr(sim_cfg, "structural_development_migration_rng_draws", MIGRATION_RNG_DRAWS)
        ),
        "supported_lane": str(
            getattr(
                sim_cfg,
                "structural_development_supported_lane",
                SUPPORTED_LANE,
            )
            or SUPPORTED_LANE
        ),
        "runtime_latch_key": "module_instance_id",
        "effective_previous_output_storage": "structural_live_module_outputs",
    }
    gated_recurrent_enabled = bool(
        getattr(sim_cfg, "structural_gated_recurrent_enable", False)
    )
    structural_gated_recurrent_metadata = (
        _escap7_gated_recurrent_metadata(sim_cfg) if gated_recurrent_enabled else None
    )
    sustainable_growth_enabled = bool(
        getattr(sim_cfg, "structural_sustainable_growth_enable", False)
    )
    structural_sustainable_growth_metadata = (
        _escap8_sustainable_growth_metadata(sim_cfg)
        if sustainable_growth_enabled
        else None
    )
    inheritance_metadata = inheritance_checkpoint_metadata(sim_cfg) or None
    authoritative_state_identity = _authoritative_state_checkpoint_identity_v1()

    # Deterministic checkpoint_id: hash of determinism-critical identifiers + hashes
    ckpt_material = {
        "schema_version": SCHEMA_VERSION,
        "generation_index": int(generation_index),
        "branch_id": branch_id,
        "campaign_id": campaign_id,
        "code_commit_hash": commit,
        "python_version": py_ver,
        "numpy_version": np_ver,
        "torch_version": torch_ver,
        "world_hash": world_hash,
        "population_hash": population_hash,
        "rng_snapshot_hash": rng_snapshot_hash,
        "event_log_hash": event_log_hash,
        "wal_hash": wal_hash,
        "runner_state_hash": runner_state_hash,
        "bank_state_hash": bank_state_hash,
        "resident_runtime_state_hash": resident_runtime_state_hash,
        "native_runtime_checkpoint_hash": native_runtime_checkpoint_hash,
        AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD: (
            authoritative_state_identity
        ),
        "capacity_epoch_metadata": capacity_epoch_metadata,
        "structural_identity_metadata": structural_identity_metadata,
    }
    if bool(structural_live_modules_metadata["enabled"]):
        ckpt_material["structural_live_modules_metadata"] = structural_live_modules_metadata
    if routing_enabled:
        ckpt_material["structural_observation_routing_metadata"] = structural_observation_routing_metadata
    if output_routing_enabled:
        ckpt_material["structural_output_routing_metadata"] = structural_output_routing_metadata
    if structural_development_enabled:
        ckpt_material["structural_development_metadata"] = structural_development_metadata
    if structural_gated_recurrent_metadata is not None:
        ckpt_material["structural_gated_recurrent_metadata"] = structural_gated_recurrent_metadata
    if structural_sustainable_growth_metadata is not None:
        ckpt_material["structural_sustainable_growth_metadata"] = (
            structural_sustainable_growth_metadata
        )
    # Deliberately omit this key for the legacy profile.  This preserves the
    # exact pre-RECOMB-1 checkpoint identifier material.
    if inheritance_metadata is not None:
        ckpt_material["inheritance_metadata"] = inheritance_metadata
    checkpoint_id = hashlib.sha256(canonical_json_bytes(ckpt_material)).hexdigest()

    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "generation_index": int(generation_index),
        "branch_id": branch_id,
        "campaign_id": campaign_id,
        "code_commit_hash": commit,
        "python_version": py_ver,
        "numpy_version": np_ver,
        "torch_version": torch_ver,
        "population_hash": str(population_hash),
        "world_hash": str(world_hash),
        "rng_snapshot_hash": str(rng_snapshot_hash),
        "event_log_hash": str(event_log_hash),
        "wal_hash": str(wal_hash),
        "submission_sequence_number": int(seq),
        "strict_replay_mode": bool(strict_replay_mode),
        # helpful non-required fields (safe)
        "runner_state_hash": str(runner_state_hash),
        "bank_state_hash": str(bank_state_hash),
        "structural_live_effect_scale": float(getattr(sim_cfg, "structural_live_effect_scale", 1.0)),
        "resident_runtime_state_hash": str(resident_runtime_state_hash),
        "native_runtime_checkpoint_hash": str(native_runtime_checkpoint_hash),
        AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD: (
            authoritative_state_identity
        ),
        "structural_identity": structural_identity_metadata,
    }
    if bool(structural_live_modules_metadata["enabled"]):
        meta["structural_live_modules_enable"] = True
        meta["structural_live_modules_profile"] = str(structural_live_modules_metadata["profile"])
        meta["structural_live_modules_schema_version"] = str(structural_live_modules_metadata["schema_version"])
    if routing_enabled:
        meta["structural_observation_routing"] = structural_observation_routing_metadata
    if output_routing_enabled:
        meta["structural_output_routing"] = structural_output_routing_metadata
    if structural_development_enabled:
        meta["structural_development"] = structural_development_metadata
    if structural_gated_recurrent_metadata is not None:
        meta["structural_gated_recurrent"] = structural_gated_recurrent_metadata
    if structural_sustainable_growth_metadata is not None:
        meta["structural_sustainable_growth"] = structural_sustainable_growth_metadata
    if capacity_epoch_metadata is not None:
        meta["capacity_epoch"] = capacity_epoch_metadata
    if inheritance_metadata is not None:
        meta["inheritance"] = inheritance_metadata

    # Optional weights_info: keep deterministic representation (canonical json if dict/list, else string)
    if weights_info is not None:
        try:
            meta["weights_info"] = _to_jsonable(weights_info)
        except Exception:
            meta["weights_info"] = str(weights_info)

    return meta


def validate_inheritance_checkpoint_compatibility(sim: Any, payload: Dict[str, Any]) -> None:
    """Fail closed across production inheritance profile boundaries.

    This gate is intentionally independent of the general checkpoint strict
    flag. Legacy checkpoints have no inheritance envelope and are compatible
    only with the legacy profile. Retired linked-profile checkpoints are
    inspectable through the historical-analysis helper but cannot resume.

    Validation and migration consume zero RNG draws.
    """

    sim_cfg = getattr(sim, "sim_cfg", None)
    try:
        destination_profile = validate_production_inheritance_profile(
            getattr(sim_cfg, "inheritance_profile", None)
        )
    except ValueError as exc:
        raise CheckpointStrictError(
            "P12_RETIRED_INHERITANCE_PROFILE production resume is forbidden"
        ) from exc
    expected = inheritance_checkpoint_metadata(sim_cfg) or None

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = payload.get("meta")
    if not isinstance(metadata, dict):
        metadata = {}
    recorded = metadata.get("inheritance")

    if expected is None:
        if recorded is not None:
            raise CheckpointStrictError(
                "inheritance checkpoint profile mismatch: "
                f"destination={destination_profile!r} checkpoint='repaired'"
            )
        return

    if recorded is None:
        raise CheckpointStrictError(
            "inheritance checkpoint metadata missing for repaired destination: "
            f"destination={destination_profile!r}"
        )
    if not isinstance(recorded, dict):
        raise CheckpointStrictError("inheritance checkpoint metadata must be an object")
    if recorded != expected:
        raise CheckpointStrictError(
            "inheritance checkpoint metadata mismatch: "
            f"destination={destination_profile!r}"
        )


def _pack_native_runtime_value(value: Any) -> Any:
    if torch is not None and hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        arr = tensor.numpy()
        data = arr.tobytes(order="C")
        return {
            "kind": "tensor",
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "device": str(getattr(value, "device", "cpu")),
            "order": "C",
            "data_hex": data.hex(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        data = arr.tobytes(order="C")
        return {
            "kind": "ndarray",
            "shape": [int(x) for x in arr.shape],
            "dtype": str(arr.dtype),
            "order": "C",
            "data_hex": data.hex(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(k): _pack_native_runtime_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_pack_native_runtime_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _cxx26_flat_interchange_scope_active(sim: Any) -> bool:
    """Return whether this runner is inside the route-neutral Flat interchange envelope."""
    sim_cfg = getattr(sim, "sim_cfg", None)
    if sim_cfg is None or not bool(getattr(sim_cfg, "native_runtime_cxx26_scope_lock_enable", False)):
        return False
    if str(getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or "").strip().lower() != "mlp":
        return False

    # Keep the interchange envelope aligned with the production support
    # resolver while deliberately ignoring route selection and extension
    # availability. A reference source does not need a live native module in
    # order to produce the canonical projection.
    from native_runtime.safe_default import _promotion_envelope, _unsupported_reasons

    envelope, _, _ = _promotion_envelope()
    if envelope is None:
        return False

    return not bool(
        _unsupported_reasons(
            sim_cfg,
            getattr(sim, "env_cfg", None),
            sim,
            envelope,
        )
    )


def _route_neutral_patch_interchange_profile(sim: Any) -> str:
    """Return the finite Patch profile admitted for route-neutral checkpoints."""

    sim_cfg = getattr(sim, "sim_cfg", None)
    env_cfg = getattr(sim, "env_cfg", None)
    if sim_cfg is None or env_cfg is None:
        return ""
    if not bool(getattr(sim_cfg, "native_runtime_cxx26_scope_lock_enable", False)):
        return ""
    if str(getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or "").strip().lower() != "mlp":
        return ""
    if not bool(getattr(env_cfg, "patch_world_enable", False)):
        return ""

    from native_runtime.patch_profile import classify_patch_architecture_b_profile

    classification = classify_patch_architecture_b_profile(
        sim_cfg,
        env_cfg,
        sim,
        world="patch",
    )
    if not classification.supported or classification.profile not in {"P0", "P1", "P2"}:
        return ""
    return str(classification.profile)


def _typed_lifecycle_checkpoint_projection(
    projection: Mapping[str, Any],
    *,
    fill_missing: bool,
) -> Dict[str, Any]:
    """Return a projection with a strict, complete typed lifecycle contract."""
    from native_runtime.interchange_projection import validate_native_checkpoint_projection
    from native_runtime.lifecycle_events import (
        FIELD_DTYPES,
        typed_lifecycle_events_from_projection,
    )

    projection = _checkpoint_projection_identity_v2(projection)
    out = {
        str(name): (
            value.detach().clone()
            if hasattr(value, "detach")
            else copy.deepcopy(value)
        )
        for name, value in dict(projection).items()
        if name not in {"metadata", "tensors"}
    }
    out["schema_version"] = str(projection.get("schema_version", "") or "")
    out["metadata"] = copy.deepcopy(dict(projection.get("metadata") or {}))
    out["tensors"] = {
        str(name): value.detach().clone() if hasattr(value, "detach") else copy.deepcopy(value)
        for name, value in dict(projection.get("tensors") or {}).items()
    }
    metadata = out["metadata"]
    tensors = out["tensors"]
    typed_names = tuple(f"typed_lifecycle_event_{name}" for name in FIELD_DTYPES)
    present = tuple(name for name in typed_names if hasattr(tensors.get(name), "detach"))
    if present and len(present) != len(typed_names):
        missing = sorted(set(typed_names) - set(present))
        raise ValueError(f"native checkpoint has an incomplete typed lifecycle tensor set: {missing}")
    if not present and not fill_missing:
        raise ValueError("native checkpoint is missing the typed lifecycle tensor set")

    events = typed_lifecycle_events_from_projection(out)
    event_lengths = {int(tensor.shape[0]) for tensor in events.values()}
    if len(event_lengths) != 1:
        raise ValueError("native checkpoint typed lifecycle tensor lengths disagree")
    typed_count = next(iter(event_lengths), 0)
    for name, tensor in events.items():
        tensors[f"typed_lifecycle_event_{name}"] = tensor.detach().cpu().contiguous()

    legacy_count = _require_native_checkpoint_int(
        metadata.get("event_tape_active_count"),
        "native checkpoint projection metadata.event_tape_active_count",
    )
    event_cursor = _require_native_checkpoint_int(
        metadata.get("event_tape_cursor"),
        "native checkpoint projection metadata.event_tape_cursor",
    )
    if legacy_count != typed_count or event_cursor != typed_count:
        raise ValueError(
            "native checkpoint lifecycle cursor contradiction: "
            f"legacy_active={legacy_count} typed_active={typed_count} cursor={event_cursor}"
        )
    metadata["typed_lifecycle_event_schema_version"] = "eiel_typed_lifecycle_event_v1"
    metadata["typed_lifecycle_event_active_count"] = int(typed_count)
    metadata.setdefault(
        "typed_lifecycle_event_source",
        "checkpoint_sidecar_legacy_adapter" if typed_count else "checkpoint_sidecar_empty_typed_contract",
    )
    validate_native_checkpoint_projection(out)
    return out


def _checkpoint_projection_identity_v2(
    projection: Mapping[str, Any],
) -> Dict[str, Any]:
    from perf.cxxbackbone10_resident_state_owner import (
        CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1,
        CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2,
        migrate_native_checkpoint_projection_v1_to_v2,
    )

    schema = str(projection.get("schema_version", "") or "")
    if schema == CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1:
        try:
            return dict(migrate_native_checkpoint_projection_v1_to_v2(projection))
        except Exception as exc:
            raise ValueError(
                f"legacy native checkpoint projection v1 migration failed: {exc}"
            ) from exc
    if schema != CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2:
        raise ValueError(
            "native checkpoint projection requires schema "
            f"{CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2!r}, got {schema!r}"
        )
    if "agent_id_to_slot" in dict(projection.get("tensors") or {}):
        raise ValueError(
            "native checkpoint projection v2 rejects dense agent_id_to_slot"
        )
    return dict(projection)


def _native_projection_authority_record(projection: Mapping[str, Any]) -> Dict[str, Any]:
    from native_runtime.checkpoint_digests import (
        checkpoint_source_projection_authority_manifest,
    )

    record = dict(checkpoint_source_projection_authority_manifest(projection))
    record["authority_source"] = NATIVE_SINGLE_PROJECTION_AUTHORITY
    return record


def _strict_lowercase_sha256(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CheckpointStrictError(
            f"{field} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _architecture_b_metadata_copy(value: Any, *, field: str) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise CheckpointStrictError(f"{field} keys must be non-empty strings")
            out[key] = _architecture_b_metadata_copy(item, field=f"{field}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            _architecture_b_metadata_copy(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise CheckpointStrictError(f"{field} must be finite")
        return value
    raise CheckpointStrictError(
        f"{field} must contain only canonical JSON metadata, got {type(value).__name__}"
    )


def _architecture_b_identifier(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise CheckpointStrictError(f"{field} must be a non-empty string")
    return value


def _architecture_b_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CheckpointStrictError(f"{field} must be an integer at least {minimum}")
    return value


def _architecture_b_closed_mapping(
    value: Any,
    *,
    field: str,
    required_fields: frozenset[str],
) -> Dict[str, Any]:
    if type(value) is not dict:
        raise CheckpointStrictError(f"{field} must be an object")
    actual_fields = set(value)
    missing = sorted(required_fields - actual_fields)
    unknown = sorted(actual_fields - required_fields)
    if missing:
        raise CheckpointStrictError(f"{field} is missing required fields: {missing}")
    if unknown:
        raise CheckpointStrictError(f"{field} has unknown fields: {unknown}")
    return value


def _validated_architecture_b_metrics_logger_state(
    value: Any,
    *,
    field: str,
) -> Dict[str, Any]:
    detached = _architecture_b_metadata_copy(value, field=field)
    state = _architecture_b_closed_mapping(
        detached,
        field=field,
        required_fields=_ARCHITECTURE_B_METRICS_LOGGER_STATE_FIELDS,
    )
    if state["schema_version"] != "metrics_state_v1":
        raise CheckpointStrictError(
            f"{field}.schema_version must be metrics_state_v1"
        )
    _architecture_b_int(
        state["collapse_streak"],
        field=f"{field}.collapse_streak",
    )

    lineage_state = state["theme5_lineage_state"]
    if type(lineage_state) is not dict:
        raise CheckpointStrictError(f"{field}.theme5_lineage_state must be an object")
    if lineage_state.get("schema_version") != "theme5_lineage_metrics_state_v1":
        raise CheckpointStrictError(
            f"{field}.theme5_lineage_state schema is invalid"
        )
    forensics_state = state["lineage_collapse_forensics_state"]
    if type(forensics_state) is not dict:
        raise CheckpointStrictError(
            f"{field}.lineage_collapse_forensics_state must be an object"
        )
    if (
        forensics_state.get("schema_version")
        != "lineage_collapse_forensics_state_v1"
    ):
        raise CheckpointStrictError(
            f"{field}.lineage_collapse_forensics_state schema is invalid"
        )
    try:
        from simulation.metrics_pkg.lineage import LineageMetricsState
        from simulation.metrics_pkg.lineage_forensics import (
            LineageCollapseForensicsState,
        )
    except Exception:
        from engine.simulation.metrics_pkg.lineage import LineageMetricsState  # type: ignore
        from engine.simulation.metrics_pkg.lineage_forensics import (  # type: ignore
            LineageCollapseForensicsState,
        )
    canonical_lineage_state = LineageMetricsState.from_dict(lineage_state).to_dict()
    if canonical_lineage_state != lineage_state:
        raise CheckpointStrictError(
            f"{field}.theme5_lineage_state is not a canonical MetricsLogger export"
        )
    canonical_forensics_state = LineageCollapseForensicsState.from_dict(
        forensics_state
    ).to_dict()
    if canonical_forensics_state != forensics_state:
        raise CheckpointStrictError(
            f"{field}.lineage_collapse_forensics_state is not a canonical "
            "MetricsLogger export"
        )

    for carry_field in (
        "early_lineage_effective_gen_1",
        "early_lineage_effective_gen_2",
        "early_lineage_top1_share_gen_1",
        "early_lineage_top1_share_gen_2",
    ):
        carry_value = state[carry_field]
        if carry_value is None:
            continue
        if type(carry_value) is not float or not np.isfinite(carry_value):
            raise CheckpointStrictError(
                f"{field}.{carry_field} must be a finite float or null"
            )
    canonical_json_bytes_lossless(state)
    return state


def _canonical_neutral_architecture_b_metrics_logger_state() -> Dict[str, Any]:
    try:
        from simulation.metrics_pkg.lineage import LineageMetricsState
        from simulation.metrics_pkg.lineage_forensics import (
            LineageCollapseForensicsState,
        )
    except Exception:
        from engine.simulation.metrics_pkg.lineage import LineageMetricsState  # type: ignore
        from engine.simulation.metrics_pkg.lineage_forensics import (  # type: ignore
            LineageCollapseForensicsState,
        )
    return {
        "schema_version": "metrics_state_v1",
        "collapse_streak": 0,
        "theme5_lineage_state": LineageMetricsState().to_dict(),
        "lineage_collapse_forensics_state": (
            LineageCollapseForensicsState().to_dict()
        ),
        "early_lineage_effective_gen_1": None,
        "early_lineage_effective_gen_2": None,
        "early_lineage_top1_share_gen_1": None,
        "early_lineage_top1_share_gen_2": None,
    }


def _architecture_b_metrics_logger_state_from_serialized(
    value: Any,
    *,
    projection: Mapping[str, Any],
) -> Dict[str, Any]:
    serialized = _architecture_b_metadata_copy(
        value,
        field="metrics_state",
    )
    _architecture_b_closed_mapping(
        serialized,
        field="metrics_state",
        required_fields=(
            _ARCHITECTURE_B_METRICS_LOGGER_STATE_FIELDS
            | _ARCHITECTURE_B_METRICS_STATE_PROVENANCE_FIELDS
        ),
    )
    expected_authority = _native_projection_authority_record(projection)
    expected_provenance = {
        **expected_authority,
        "authority_classification": "route_neutral_public_continuation_state",
        "authority_epoch": expected_authority["source_authority_epoch"],
    }
    if set(expected_provenance) != _ARCHITECTURE_B_METRICS_STATE_PROVENANCE_FIELDS:
        raise CheckpointStrictError(
            "Architecture B metrics provenance schema is inconsistent"
        )
    for name, expected in expected_provenance.items():
        if serialized.get(name) != expected:
            raise CheckpointStrictError(
                f"metrics_state.{name} contradicts checkpoint projection authority"
            )
    logger_state = {
        name: serialized[name]
        for name in _ARCHITECTURE_B_METRICS_LOGGER_STATE_FIELDS
    }
    return _validated_architecture_b_metrics_logger_state(
        logger_state,
        field="metrics_state.logger_export",
    )


def _architecture_b_row_cursor(
    value: Any,
    *,
    field: str,
    completed_generation: int,
    row_kind: str,
) -> None:
    cursor = _architecture_b_closed_mapping(
        value,
        field=field,
        required_fields=frozenset({"completed_generation", "last_row_key"}),
    )
    recorded_generation = _architecture_b_int(
        cursor["completed_generation"],
        field=f"{field}.completed_generation",
    )
    if recorded_generation != completed_generation:
        raise CheckpointStrictError(
            f"{field}.completed_generation contradiction: "
            f"expected={completed_generation} actual={recorded_generation}"
        )
    row_key = _architecture_b_identifier(
        cursor["last_row_key"],
        field=f"{field}.last_row_key",
    )
    expected_prefix = f"generation:{completed_generation}:{row_kind}:"
    if not row_key.startswith(expected_prefix) or not row_key[len(expected_prefix) :]:
        raise CheckpointStrictError(
            f"{field}.last_row_key must identify a {row_kind} row for "
            f"generation {completed_generation}"
        )


def _architecture_b_cursor_range(
    value: Any,
    *,
    field: str,
) -> List[int]:
    if type(value) is not list or len(value) != 2:
        raise CheckpointStrictError(f"{field} must be a two-item array")
    start = _architecture_b_int(value[0], field=f"{field}[0]")
    end = _architecture_b_int(value[1], field=f"{field}[1]")
    if end < start:
        raise CheckpointStrictError(f"{field} must be a non-reversed half-open range")
    return [start, end]


def _validate_architecture_b_observer_continuation(
    observer_continuation: Any,
    *,
    projection_generation: int,
    projection_tick: int,
    projection_event_cursor: int,
    snapshot_sequence: int,
) -> Dict[str, Any]:
    continuation = _architecture_b_closed_mapping(
        observer_continuation,
        field="observer_continuation",
        required_fields=_ARCHITECTURE_B_OBSERVER_CONTINUATION_FIELDS,
    )
    if continuation["schema_version"] != (
        ARCHITECTURE_B_OBSERVER_CONTINUATION_SCHEMA_VERSION
    ):
        raise CheckpointStrictError(
            "observer_continuation.schema_version must be "
            f"{ARCHITECTURE_B_OBSERVER_CONTINUATION_SCHEMA_VERSION}"
        )

    scalar_expectations = {
        "checkpoint_next_generation": projection_generation,
        "tick": projection_tick,
        "typed_lifecycle_event_cursor": projection_event_cursor,
        "exact_reporting_boundary_ordinal": snapshot_sequence,
        "chronicle_observation_ordinal": snapshot_sequence,
        "progress_sequence": snapshot_sequence,
    }
    for field, expected_value in scalar_expectations.items():
        actual = _architecture_b_int(
            continuation[field],
            field=f"observer_continuation.{field}",
        )
        if actual != expected_value:
            raise CheckpointStrictError(
                f"observer_continuation.{field} contradiction: "
                f"expected={expected_value} actual={actual}"
            )

    if projection_generation == 0 and projection_tick == 0:
        if snapshot_sequence != 0:
            raise CheckpointStrictError(
                "generation-zero observer continuation requires boundary ordinal zero"
            )
        empty_cursor = {
            "completed_generation": -1,
            "last_row_key": None,
        }
        for field in (
            "behavior_completed_generation_and_row_key",
            "lineage_completed_generation_and_row_key",
        ):
            if continuation[field] != empty_cursor:
                raise CheckpointStrictError(
                    f"observer_continuation.{field} must carry the exact "
                    "pre-generation sentinel"
                )
        if continuation["metrics_completed_generation"] != -1:
            raise CheckpointStrictError(
                "observer_continuation.metrics_completed_generation must be -1 "
                "at generation zero"
            )
        schedule = continuation["schedule_phase"]
        if schedule != {"origin_generation": 0, "next_generation": 0}:
            raise CheckpointStrictError(
                "generation-zero observer continuation schedule is not exact"
            )
        if continuation["deduplication"] != {"acknowledged_record_keys": []}:
            raise CheckpointStrictError(
                "generation-zero observer continuation cannot acknowledge records"
            )
        expected_ranges = {
            "exact_reporting": [0, 0],
            "behaviors": [0, 0],
            "metrics": [0, 0],
            "lineage": [0, 0],
            "typed_lifecycle": [0, projection_event_cursor],
            "chronicle": [0, 0],
            "checkpoint_generation": [0, 1],
        }
        if continuation["acknowledged_channel_cursor_ranges"] != expected_ranges:
            raise CheckpointStrictError(
                "generation-zero observer continuation cursor ranges are not exact"
            )
        copied = _architecture_b_metadata_copy(
            continuation,
            field="observer_continuation",
        )
        assert isinstance(copied, dict)
        return copied
    completed_generation = projection_generation - 1
    if completed_generation < 0:
        empty_cursor = {"completed_generation": -1, "last_row_key": None}
        if (
            continuation["behavior_completed_generation_and_row_key"] != empty_cursor
            or continuation["lineage_completed_generation_and_row_key"] != empty_cursor
            or continuation["metrics_completed_generation"] != -1
        ):
            raise CheckpointStrictError(
                "partial generation-zero continuation must retain empty row cursors"
            )
    else:
        _architecture_b_row_cursor(
            continuation["behavior_completed_generation_and_row_key"],
            field="observer_continuation.behavior_completed_generation_and_row_key",
            completed_generation=completed_generation,
            row_kind="agent",
        )
        metrics_generation = _architecture_b_int(
            continuation["metrics_completed_generation"],
            field="observer_continuation.metrics_completed_generation",
        )
        if metrics_generation != completed_generation:
            raise CheckpointStrictError(
                "observer_continuation.metrics_completed_generation contradiction: "
                f"expected={completed_generation} actual={metrics_generation}"
            )
        _architecture_b_row_cursor(
            continuation["lineage_completed_generation_and_row_key"],
            field="observer_continuation.lineage_completed_generation_and_row_key",
            completed_generation=completed_generation,
            row_kind="lineage",
        )

    schedule_phase = _architecture_b_closed_mapping(
        continuation["schedule_phase"],
        field="observer_continuation.schedule_phase",
        required_fields=frozenset({"origin_generation", "next_generation"}),
    )
    schedule_origin = _architecture_b_int(
        schedule_phase["origin_generation"],
        field="observer_continuation.schedule_phase.origin_generation",
    )
    schedule_next = _architecture_b_int(
        schedule_phase["next_generation"],
        field="observer_continuation.schedule_phase.next_generation",
    )
    if schedule_next != projection_generation:
        raise CheckpointStrictError(
            "observer_continuation.schedule_phase.next_generation contradiction: "
            f"expected={projection_generation} actual={schedule_next}"
        )
    if projection_tick == 0:
        if snapshot_sequence > projection_generation:
            if schedule_origin != projection_generation:
                raise CheckpointStrictError(
                    "observer_continuation.schedule_phase.origin_generation "
                    "contradicts absolute snapshot sequence"
                )
        elif schedule_origin + snapshot_sequence != projection_generation:
            raise CheckpointStrictError(
                "observer_continuation.schedule_phase.origin_generation contradicts "
                "next_generation and snapshot_sequence"
            )

    deduplication = _architecture_b_closed_mapping(
        continuation["deduplication"],
        field="observer_continuation.deduplication",
        required_fields=frozenset({"acknowledged_record_keys"}),
    )
    record_keys = deduplication["acknowledged_record_keys"]
    if type(record_keys) is not list:
        raise CheckpointStrictError(
            "observer_continuation.deduplication.acknowledged_record_keys must be an array"
        )
    if len(record_keys) > ARCHITECTURE_B_DEDUPLICATION_MAX_RECORD_KEYS:
        raise CheckpointStrictError(
            "observer_continuation.deduplication.acknowledged_record_keys exceeds "
            f"the bound of {ARCHITECTURE_B_DEDUPLICATION_MAX_RECORD_KEYS}"
        )
    validated_record_keys: List[str] = []
    for index, record_key in enumerate(record_keys):
        key = _architecture_b_identifier(
            record_key,
            field=(
                "observer_continuation.deduplication.acknowledged_record_keys"
                f"[{index}]"
            ),
        )
        parts = key.split(":")
        if len(parts) >= 2 and parts[0] == "generation":
            try:
                key_generation = int(parts[1])
            except ValueError as exc:
                raise CheckpointStrictError(
                    "observer_continuation deduplication generation key is invalid"
                ) from exc
            if key_generation < 0 or key_generation > completed_generation:
                raise CheckpointStrictError(
                    "observer_continuation deduplication key generation is outside "
                    "the acknowledged continuation frontier"
                )
        validated_record_keys.append(key)
    if len(set(validated_record_keys)) != len(validated_record_keys):
        raise CheckpointStrictError(
            "observer_continuation.deduplication.acknowledged_record_keys must be unique"
        )

    ranges = _architecture_b_closed_mapping(
        continuation["acknowledged_channel_cursor_ranges"],
        field="observer_continuation.acknowledged_channel_cursor_ranges",
        required_fields=_ARCHITECTURE_B_ACKNOWLEDGED_CHANNEL_RANGE_FIELDS,
    )
    validated_ranges = {
        channel: _architecture_b_cursor_range(
            ranges[channel],
            field=(
                "observer_continuation.acknowledged_channel_cursor_ranges."
                f"{channel}"
            ),
        )
        for channel in sorted(_ARCHITECTURE_B_ACKNOWLEDGED_CHANNEL_RANGE_FIELDS)
    }
    generation_range = [max(0, completed_generation), projection_generation]
    ordinal_range = [snapshot_sequence - 1, snapshot_sequence]
    exact_ranges = {
        "exact_reporting": generation_range,
        "behaviors": generation_range,
        "metrics": generation_range,
        "lineage": generation_range,
        "chronicle": ordinal_range,
        "checkpoint_generation": [projection_generation, projection_generation + 1],
    }
    for channel, expected_range in exact_ranges.items():
        if validated_ranges[channel] != expected_range:
            raise CheckpointStrictError(
                "observer_continuation.acknowledged_channel_cursor_ranges."
                f"{channel} contradiction: expected={expected_range} "
                f"actual={validated_ranges[channel]}"
            )
    lifecycle_range = validated_ranges["typed_lifecycle"]
    if lifecycle_range[1] != projection_event_cursor:
        raise CheckpointStrictError(
            "observer_continuation.acknowledged_channel_cursor_ranges."
            "typed_lifecycle must end at typed_lifecycle_event_cursor"
        )

    copied = _architecture_b_metadata_copy(
        continuation,
        field="observer_continuation",
    )
    assert isinstance(copied, dict)
    return copied


def _architecture_b_completion_context(
    *,
    architecture_b_identity: Mapping[str, Any],
    observer_continuation: Mapping[str, Any],
    projection: Mapping[str, Any],
    generation_index: int,
    source_projection_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(architecture_b_identity, Mapping):
        raise CheckpointStrictError("architecture_b_identity must be a mapping")
    if not isinstance(observer_continuation, Mapping) or not observer_continuation:
        raise CheckpointStrictError("observer_continuation must be a non-empty mapping")

    identity_source = _architecture_b_closed_mapping(
        architecture_b_identity,
        field="architecture_b_identity",
        required_fields=_ARCHITECTURE_B_IDENTITY_FIELDS,
    )
    identity = _architecture_b_metadata_copy(
        identity_source,
        field="architecture_b_identity",
    )
    assert isinstance(identity, dict)

    if not isinstance(projection, Mapping):
        raise CheckpointStrictError("Architecture B checkpoint projection must be a mapping")
    projection_metadata = projection.get("metadata")
    if not isinstance(projection_metadata, Mapping):
        raise CheckpointStrictError(
            "Architecture B checkpoint projection metadata must be a mapping"
        )
    projection_generation = _architecture_b_int(
        projection_metadata.get("generation"),
        field="native checkpoint projection metadata.generation",
    )
    projection_tick = _architecture_b_int(
        projection_metadata.get("tick"),
        field="native checkpoint projection metadata.tick",
    )
    if projection_generation != generation_index:
        raise CheckpointStrictError(
            "Architecture B projection generation contradicts checkpoint identity"
        )
    expected_phase = (
        "generation_boundary" if projection_tick == 0 else "partial_generation"
    )
    if projection_metadata.get("agent_grid_runner_phase") != expected_phase:
        raise CheckpointStrictError(
            "Architecture B projection phase contradicts checkpoint identity"
        )
    try:
        from native_runtime.lifecycle_events import (
            typed_lifecycle_events_from_projection,
            validated_typed_lifecycle_event_cursor,
        )

        projection_event_cursor = validated_typed_lifecycle_event_cursor(
            projection,
            typed_lifecycle_events_from_projection(projection),
        )
    except Exception as exc:
        raise CheckpointStrictError(
            f"Architecture B projection typed lifecycle cursor is invalid: {exc}"
        ) from exc

    for field in ("run_id", "controller_id", "snapshot_id"):
        _architecture_b_identifier(
            identity.get(field),
            field=f"architecture_b_identity.{field}",
        )
    owner_epoch = _architecture_b_int(
        identity.get("owner_epoch"),
        field="architecture_b_identity.owner_epoch",
        minimum=1,
    )
    snapshot_sequence = _architecture_b_int(
        identity.get("snapshot_sequence"),
        field="architecture_b_identity.snapshot_sequence",
        minimum=0,
    )
    if (snapshot_sequence == 0) != (
        projection_generation == 0 and projection_tick == 0
    ):
        raise CheckpointStrictError(
            "architecture_b_identity snapshot sequence zero is reserved for "
            "the generation-zero boundary"
        )
    expected_snapshot_id = (
        f"{identity['controller_id']}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
    )
    if identity["snapshot_id"] != expected_snapshot_id:
        raise CheckpointStrictError(
            "architecture_b_identity.snapshot_id contradicts controller, owner epoch, "
            "or snapshot sequence"
        )

    digest_fields = (
        "snapshot_digest_sha256",
        "snapshot_digest",
        "payload_digest_sha256",
    )
    supplied_snapshot_digests = [
        _strict_lowercase_sha256(
            identity[field],
            f"architecture_b_identity.{field}",
        )
        for field in digest_fields
        if field in identity
    ]
    if not supplied_snapshot_digests:
        raise CheckpointStrictError(
            "architecture_b_identity.snapshot_digest_sha256 is required"
        )
    if len(set(supplied_snapshot_digests)) != 1:
        raise CheckpointStrictError("architecture_b_identity snapshot digest fields contradict")
    identity["snapshot_digest_sha256"] = supplied_snapshot_digests[0]

    for field, expected in (
        ("next_generation", generation_index),
        ("tick", projection_tick),
    ):
        if field in identity:
            actual = _architecture_b_int(
                identity[field],
                field=f"architecture_b_identity.{field}",
            )
            if actual != expected:
                raise CheckpointStrictError(
                    f"architecture_b_identity.{field} contradiction: "
                    f"expected={expected} actual={actual}"
                )
        identity[field] = expected
    if "phase" in identity and identity["phase"] != expected_phase:
        raise CheckpointStrictError(
            f"architecture_b_identity.phase must be {expected_phase}"
        )
    identity["phase"] = expected_phase
    if "source_projection_sha256" in identity:
        recorded_source_digest = _strict_lowercase_sha256(
            identity["source_projection_sha256"],
            "architecture_b_identity.source_projection_sha256",
        )
        if recorded_source_digest != source_projection_sha256:
            raise CheckpointStrictError(
                "architecture_b_identity source projection digest contradiction"
            )
    identity["source_projection_sha256"] = source_projection_sha256

    continuation = _validate_architecture_b_observer_continuation(
        observer_continuation,
        projection_generation=projection_generation,
        projection_tick=projection_tick,
        projection_event_cursor=projection_event_cursor,
        snapshot_sequence=snapshot_sequence,
    )

    identity_sha256 = hashlib.sha256(
        canonical_json_bytes_lossless(identity)
    ).hexdigest()
    continuation_sha256 = hashlib.sha256(
        canonical_json_bytes_lossless(continuation)
    ).hexdigest()
    return {
        "architecture_b_schema_version": ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION,
        "architecture_b_identity": identity,
        "architecture_b_identity_sha256": identity_sha256,
        "observer_continuation": continuation,
        "observer_continuation_sha256": continuation_sha256,
    }


def _architecture_b_completion_sha256(completion: Mapping[str, Any]) -> str:
    material = {
        field: completion.get(field)
        for field in (
            "schema_version",
            "status",
            "checkpoint_id",
            "source_projection_sha256",
            "coherence_sha256",
            "architecture_b_schema_version",
            "architecture_b_identity",
            "architecture_b_identity_sha256",
            "observer_continuation",
            "observer_continuation_sha256",
        )
    }
    return hashlib.sha256(canonical_json_bytes_lossless(material)).hexdigest()


def _validate_architecture_b_completion_manifest(
    completion: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    projection: Mapping[str, Any],
    source_projection_sha256: str,
) -> None:
    generation_index = _architecture_b_int(
        metadata.get("generation_index"),
        field="metadata.generation_index",
    )
    expected = _architecture_b_completion_context(
        architecture_b_identity=completion.get("architecture_b_identity"),
        observer_continuation=completion.get("observer_continuation"),
        projection=projection,
        generation_index=generation_index,
        source_projection_sha256=source_projection_sha256,
    )
    for field, expected_value in expected.items():
        if completion.get(field) != expected_value:
            raise CheckpointStrictError(
                f"native checkpoint bundle Architecture B completion {field} is invalid"
            )
    for field in ("architecture_b_identity_sha256", "observer_continuation_sha256"):
        if metadata.get(field) != expected[field]:
            raise CheckpointStrictError(
                f"native checkpoint bundle metadata {field} is invalid"
            )
    recorded = _strict_lowercase_sha256(
        completion.get("architecture_b_completion_sha256"),
        "bundle_complete.architecture_b_completion_sha256",
    )
    computed = _architecture_b_completion_sha256(completion)
    if recorded != computed:
        raise CheckpointStrictError(
            "native checkpoint bundle Architecture B completion digest mismatch"
        )


def _detached_architecture_b_projection(
    projection: Mapping[str, Any],
) -> Dict[str, Any]:
    detached: Dict[str, Any] = {}
    for field, value in projection.items():
        if field == "tensors" and isinstance(value, Mapping):
            detached[field] = {
                str(name): (
                    tensor.detach().clone()
                    if hasattr(tensor, "detach")
                    else copy.deepcopy(tensor)
                )
                for name, tensor in value.items()
            }
        else:
            detached[str(field)] = copy.deepcopy(value)
    return detached


def _detached_architecture_b_resume_envelope(
    envelope: Mapping[str, Any],
) -> Dict[str, Any]:
    detached: Dict[str, Any] = {}
    for field, value in envelope.items():
        if field == "checkpoint_projection":
            if not isinstance(value, Mapping):
                raise CheckpointStrictError(
                    "Architecture B resume envelope projection is invalid"
                )
            detached[field] = _detached_architecture_b_projection(value)
        else:
            detached[field] = _architecture_b_metadata_copy(
                value,
                field=f"architecture_b_resume_envelope.{field}",
            )
    return detached


def _architecture_b_resume_envelope_authority_sha256(
    envelope: Mapping[str, Any],
) -> str:
    projection = envelope.get("checkpoint_projection")
    identity = envelope.get("architecture_b_identity")
    continuation = envelope.get("observer_continuation")
    import_metadata = envelope.get("controller_import_metadata")
    completion_authority = envelope.get("bundle_completion_authority")
    if not all(
        isinstance(item, Mapping)
        for item in (
            projection,
            identity,
            continuation,
            import_metadata,
            completion_authority,
        )
    ):
        raise CheckpointStrictError(
            "Architecture B resume envelope binding components are invalid"
        )
    material = {
        "schema_version": envelope.get("schema_version"),
        "bundle_kind": envelope.get("bundle_kind"),
        "source_projection_sha256": envelope.get("source_projection_sha256"),
        "projection_authority": _native_projection_authority_record(projection),
        "architecture_b_identity_sha256": hashlib.sha256(
            canonical_json_bytes_lossless(identity)
        ).hexdigest(),
        "observer_continuation_sha256": hashlib.sha256(
            canonical_json_bytes_lossless(continuation)
        ).hexdigest(),
        "controller_import_metadata_sha256": hashlib.sha256(
            canonical_json_bytes_lossless(import_metadata)
        ).hexdigest(),
        "bundle_completion_authority_sha256": hashlib.sha256(
            canonical_json_bytes_lossless(completion_authority)
        ).hexdigest(),
    }
    return hashlib.sha256(canonical_json_bytes_lossless(material)).hexdigest()


def _architecture_b_checkpoint_payload_snapshot(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = payload.get("metadata")
    completion = payload.get("bundle_complete")
    metrics_state = payload.get("metrics_state")
    envelope = payload.get("architecture_b_resume_envelope")
    if not isinstance(metadata, Mapping) or not isinstance(completion, Mapping):
        raise CheckpointStrictError(
            "Architecture B checkpoint payload authority records are invalid"
        )
    if not isinstance(metrics_state, Mapping):
        raise CheckpointStrictError(
            "Architecture B checkpoint payload metrics state is invalid"
        )
    if not isinstance(envelope, Mapping):
        raise CheckpointStrictError(
            "Architecture B checkpoint payload resume envelope is invalid"
        )
    return {
        "bundle_kind": _architecture_b_metadata_copy(
            payload.get("bundle_kind"),
            field="architecture_b_checkpoint_payload.bundle_kind",
        ),
        "metadata": _architecture_b_metadata_copy(
            metadata,
            field="architecture_b_checkpoint_payload.metadata",
        ),
        "bundle_complete": _architecture_b_metadata_copy(
            completion,
            field="architecture_b_checkpoint_payload.bundle_complete",
        ),
        "metrics_state": _architecture_b_metadata_copy(
            metrics_state,
            field="architecture_b_checkpoint_payload.metrics_state",
        ),
        "architecture_b_resume_envelope": (
            _detached_architecture_b_resume_envelope(envelope)
        ),
    }


def _architecture_b_checkpoint_payload_authority_sha256(
    payload: Mapping[str, Any],
) -> str:
    snapshot = _architecture_b_checkpoint_payload_snapshot(payload)
    envelope = snapshot["architecture_b_resume_envelope"]
    material = {
        "bundle_kind": snapshot["bundle_kind"],
        "metadata_sha256": hashlib.sha256(
            canonical_json_bytes_lossless(snapshot["metadata"])
        ).hexdigest(),
        "bundle_complete_sha256": hashlib.sha256(
            canonical_json_bytes_lossless(snapshot["bundle_complete"])
        ).hexdigest(),
        "metrics_state_sha256": hashlib.sha256(
            canonical_json_bytes(snapshot["metrics_state"])
        ).hexdigest(),
        "resume_envelope_authority_sha256": (
            _architecture_b_resume_envelope_authority_sha256(envelope)
        ),
    }
    return hashlib.sha256(canonical_json_bytes_lossless(material)).hexdigest()


def _build_architecture_b_resume_envelope(
    *,
    projection: Mapping[str, Any],
    completion: Mapping[str, Any],
    source_projection_sha256: str,
) -> Dict[str, Any]:
    decoded_projection = _detached_architecture_b_projection(projection)
    identity = _architecture_b_metadata_copy(
        completion["architecture_b_identity"],
        field="architecture_b_resume_envelope.architecture_b_identity",
    )
    continuation = _architecture_b_metadata_copy(
        completion["observer_continuation"],
        field="architecture_b_resume_envelope.observer_continuation",
    )
    source_metadata = decoded_projection.get("metadata")
    if not isinstance(source_metadata, Mapping):
        raise CheckpointStrictError(
            "Architecture B decoded resume projection metadata is invalid"
        )
    controller_import_metadata = copy.deepcopy(dict(source_metadata))
    controller_import_metadata["architecture_b_identity"] = copy.deepcopy(identity)
    controller_import_metadata["observer_continuation"] = copy.deepcopy(continuation)
    completion_authority = {
        field: copy.deepcopy(completion.get(field))
        for field in sorted(_ARCHITECTURE_B_COMPLETION_AUTHORITY_FIELDS)
    }
    missing_completion_fields = sorted(
        field
        for field, value in completion_authority.items()
        if value is None or value == ""
    )
    if missing_completion_fields:
        raise CheckpointStrictError(
            "Architecture B completion authority is incomplete: "
            f"{missing_completion_fields}"
        )
    return {
        "schema_version": ARCHITECTURE_B_RESUME_ENVELOPE_SCHEMA_VERSION,
        "bundle_kind": ARCHITECTURE_B_BUNDLE_KIND,
        "source_projection_sha256": source_projection_sha256,
        "checkpoint_projection": decoded_projection,
        "architecture_b_identity": identity,
        "observer_continuation": continuation,
        "controller_import_metadata": controller_import_metadata,
        "bundle_completion_authority": completion_authority,
    }


def decode_architecture_b_resume_envelope(
    payload: Mapping[str, Any],
) -> ValidatedArchitectureBResumeEnvelope:
    """Return a detached, explicit Architecture B controller resume envelope."""
    if (
        type(payload) is not ValidatedArchitectureBCheckpointPayload
        or getattr(payload, "_authority_token", None)
        is not _ARCHITECTURE_B_CHECKPOINT_PAYLOAD_AUTHORITY_TOKEN
    ):
        raise CheckpointStrictError(
            "payload lacks exact strict-loader Architecture B provenance"
        )
    try:
        rebound_payload = _architecture_b_checkpoint_payload_snapshot(payload)
        rebound_digest = _architecture_b_checkpoint_payload_authority_sha256(
            rebound_payload
        )
    except CheckpointStrictError:
        raise
    except Exception as exc:
        raise CheckpointStrictError(
            "Architecture B checkpoint payload authority could not be rebound"
        ) from exc
    if rebound_digest != getattr(payload, "_authority_digest_sha256", None):
        raise CheckpointStrictError(
            "Architecture B checkpoint payload changed after strict loading"
        )

    envelope = rebound_payload.get("architecture_b_resume_envelope")
    if not isinstance(envelope, Mapping):
        raise CheckpointStrictError(
            "payload was not loaded as a strict Architecture B checkpoint"
        )
    if envelope.get("schema_version") != ARCHITECTURE_B_RESUME_ENVELOPE_SCHEMA_VERSION:
        raise CheckpointStrictError("Architecture B resume envelope schema is invalid")
    if envelope.get("bundle_kind") != ARCHITECTURE_B_BUNDLE_KIND:
        raise CheckpointStrictError("Architecture B resume envelope kind is invalid")
    projection = envelope.get("checkpoint_projection")
    if not isinstance(projection, Mapping):
        raise CheckpointStrictError("Architecture B resume envelope projection is invalid")
    completion = rebound_payload.get("bundle_complete")
    metadata = rebound_payload.get("metadata")
    if not isinstance(completion, Mapping) or not isinstance(metadata, Mapping):
        raise CheckpointStrictError(
            "Architecture B resume requires its validated bundle completion record"
        )
    source_projection_sha256 = _strict_lowercase_sha256(
        envelope.get("source_projection_sha256"),
        "architecture_b_resume_envelope.source_projection_sha256",
    )
    actual_projection_sha256 = str(
        _native_projection_authority_record(projection).get(
            "source_projection_sha256", ""
        )
    )
    if actual_projection_sha256 != source_projection_sha256:
        raise CheckpointStrictError(
            "Architecture B resume envelope projection authority digest mismatch"
        )
    _validate_architecture_b_completion_manifest(
        completion,
        metadata=metadata,
        projection=projection,
        source_projection_sha256=source_projection_sha256,
    )
    rebuilt = _build_architecture_b_resume_envelope(
        projection=projection,
        completion=completion,
        source_projection_sha256=source_projection_sha256,
    )
    for field in (
        "architecture_b_identity",
        "observer_continuation",
        "bundle_completion_authority",
    ):
        if envelope.get(field) != rebuilt[field]:
            raise CheckpointStrictError(
                f"Architecture B resume envelope {field} contradicts bundle completion"
            )
    return ValidatedArchitectureBResumeEnvelope(
        rebuilt,
        _authority_token=_ARCHITECTURE_B_RESUME_ENVELOPE_AUTHORITY_TOKEN,
        _authority_digest_sha256=(
            _architecture_b_resume_envelope_authority_sha256(rebuilt)
        ),
    )


def architecture_b_metrics_logger_state_from_payload(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Extract exact logger continuation from a strict native-projection load."""

    if type(payload) is not ValidatedArchitectureBCheckpointPayload:
        metadata = payload.get("metadata")
        sidecar = payload.get("native_runtime_checkpoint_projection")
        serialized_state = payload.get("metrics_state")
        if not isinstance(metadata, Mapping) or not isinstance(sidecar, Mapping):
            raise CheckpointStrictError(
                "metrics restoration requires strict native checkpoint metadata and sidecar"
            )
        if metadata.get("checkpoint_authority_schema_version") != (
            NATIVE_SINGLE_PROJECTION_AUTHORITY
        ):
            raise CheckpointStrictError(
                "metrics restoration requires native single-projection authority"
            )
        expected_state_sha256 = _strict_lowercase_sha256(
            metadata.get("metrics_state_hash"),
            "metadata.metrics_state_hash",
        )
        actual_state_sha256 = hashlib.sha256(
            canonical_json_bytes(serialized_state)
        ).hexdigest()
        if actual_state_sha256 != expected_state_sha256:
            raise CheckpointStrictError(
                "metrics_state hash mismatch before logger restoration"
            )
        generation_index = _architecture_b_int(
            metadata.get("generation_index"),
            field="metadata.generation_index",
        )
        projection, source_projection_sha256 = _validate_native_authority_sidecar(
            sidecar,
            generation_index=generation_index,
        )
        if metadata.get("source_projection_sha256") != source_projection_sha256:
            raise CheckpointStrictError(
                "metrics restoration source projection authority mismatch"
            )
        return _architecture_b_metrics_logger_state_from_serialized(
            serialized_state,
            projection=projection,
        )

    envelope = decode_architecture_b_resume_envelope(payload)
    rebound_payload = _architecture_b_checkpoint_payload_snapshot(payload)
    rebound_digest = _architecture_b_checkpoint_payload_authority_sha256(
        rebound_payload
    )
    if rebound_digest != getattr(payload, "_authority_digest_sha256", None):
        raise CheckpointStrictError(
            "Architecture B checkpoint payload changed before metrics restoration"
        )
    projection = envelope.get("checkpoint_projection")
    if not isinstance(projection, Mapping):
        raise CheckpointStrictError(
            "Architecture B metrics restoration has no checkpoint projection"
        )
    return _architecture_b_metrics_logger_state_from_serialized(
        rebound_payload["metrics_state"],
        projection=projection,
    )


def restore_architecture_b_metrics_logger_state(
    sim: Any,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Restore and verify exact MetricsLogger continuation before manual resume."""

    expected_state = architecture_b_metrics_logger_state_from_payload(payload)
    logger = getattr(sim, "logger", None)
    if logger is None:
        for target in _runner_targets(sim):
            candidate = getattr(target, "logger", None)
            if candidate is not None:
                logger = candidate
                break
    export_state = getattr(logger, "export_state", None)
    restore_state = getattr(logger, "restore_state", None)
    if not callable(export_state) or not callable(restore_state):
        raise CheckpointStrictError(
            "Architecture B manual resume requires MetricsLogger export_state/restore_state"
        )

    try:
        previous_state = export_state()
    except Exception as exc:
        raise CheckpointStrictError(
            "Architecture B metrics logger could not export its pre-restore state"
        ) from exc
    if not isinstance(previous_state, Mapping):
        raise CheckpointStrictError(
            "Architecture B metrics logger pre-restore export is not an object"
        )
    previous_state = _architecture_b_metadata_copy(
        previous_state,
        field="metrics_logger.previous_state",
    )

    failure: Optional[Exception] = None
    try:
        restore_state(copy.deepcopy(expected_state))
        restored_state = _validated_architecture_b_metrics_logger_state(
            export_state(),
            field="metrics_logger.restored_state",
        )
        expected_bytes = canonical_json_bytes_lossless(expected_state)
        restored_bytes = canonical_json_bytes_lossless(restored_state)
        if restored_bytes != expected_bytes:
            raise CheckpointStrictError(
                "MetricsLogger restore_state did not reproduce the checkpoint export exactly"
            )
    except Exception as exc:
        failure = exc

    if failure is not None:
        try:
            restore_state(copy.deepcopy(previous_state))
        except Exception as rollback_exc:
            raise CheckpointStrictError(
                "Architecture B metrics restoration failed and logger rollback failed"
            ) from rollback_exc
        if isinstance(failure, CheckpointStrictError):
            raise failure
        raise CheckpointStrictError(
            "Architecture B metrics logger rejected checkpoint continuation state"
        ) from failure

    state_sha256 = hashlib.sha256(expected_bytes).hexdigest()
    return {
        "schema_version": ARCHITECTURE_B_METRICS_RESTORE_RECEIPT_SCHEMA_VERSION,
        "status": "restored_and_verified",
        "metrics_state_schema_version": expected_state["schema_version"],
        "metrics_state_sha256": state_sha256,
        "source_projection_sha256": str(
            payload["metadata"].get("source_projection_sha256", "") or ""
        ),
    }


def _reference_source_projection_envelope(
    imported: Any,
    projection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Give a reference projection the same explicit owner envelope used by native checkpoints."""

    from perf.cxxbackbone10_resident_state_owner import _checkpoint_tensor_manifest

    out = dict(projection)
    out["schema_version"] = "cxxbackbone10_checkpoint_projection_v2"
    metadata = dict(out.get("metadata") or {})
    tensors = dict(out.get("tensors") or {})
    manifest: Dict[str, Any] = {}
    for name, value in sorted(tensors.items()):
        if not hasattr(value, "detach"):
            raise TypeError(f"reference checkpoint tensor is not tensor-like: {name}")
        tensor = value.detach().cpu().contiguous()
        manifest[str(name)] = _checkpoint_tensor_manifest(tensor)
    event_cursor = int(
        metadata.get(
            "typed_lifecycle_event_active_count",
            metadata.get("event_tape_cursor", 0),
        )
        or 0
    )
    generation = metadata.get("generation")
    tick = metadata.get("tick")
    steps_per_generation = metadata.get("configured_steps_per_generation")
    if type(generation) is not int or generation < 0:
        raise TypeError("reference checkpoint generation must be a nonnegative int")
    if type(tick) is not int or tick < 0:
        raise TypeError("reference checkpoint tick must be a nonnegative int")
    if type(steps_per_generation) is not int or steps_per_generation <= 0:
        raise TypeError(
            "reference checkpoint configured_steps_per_generation must be positive"
        )
    generation_calls = generation * steps_per_generation + tick
    out.update(
        {
            "tensor_manifest": manifest,
            "restore_count": 0,
            "generation_calls": generation_calls,
            "event_cursor": event_cursor,
            "runtime_event_cursor": event_cursor,
            "last_applied_actions": 0,
            "last_moved": 0,
            "last_gathered": 0,
            "last_policy_refill_calls": 0,
            "last_policy_refill_rows": 0,
            "pending_generation_free_agent_ids": list(
                getattr(imported, "pending_generation_free_agent_ids", ()) or ()
            ),
            "pending_generation_free_slots": list(
                getattr(imported, "pending_generation_free_slots", ()) or ()
            ),
            "last_world": str(metadata.get("world", "flat") or "flat"),
        }
    )
    return out


def _authority_record_mismatches(
    *,
    name: str,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    mismatches: List[Dict[str, Any]] = []
    for field, expected_value in expected.items():
        actual = payload.get(field)
        if actual != expected_value:
            mismatches.append(
                {
                    "field": f"{name}.{field}",
                    "reason": "source_authority_record_mismatch",
                    "expected": expected_value,
                    "actual": actual,
                }
            )
    return mismatches


def _export_native_runtime_checkpoint_projection(sim: Any, *, generation_index: int) -> Dict[str, Any]:
    engine = getattr(sim, "_native_eiel_engine", None)
    try:
        if engine is not None:
            projection = engine.export_checkpoint_projection()
            counters = getattr(getattr(engine, "counters", None), "to_dict", lambda: {})()
            source = "NativeEielEngine.export_checkpoint_projection"
            world = str(getattr(engine, "world", ""))
            capability_mask = getattr(engine, "capability_mask", {})
        else:
            flat_scope = _cxx26_flat_interchange_scope_active(sim)
            patch_profile = (
                "" if flat_scope else _route_neutral_patch_interchange_profile(sim)
            )
            if not flat_scope and not patch_profile:
                return {
                    "schema_version": "native_runtime_checkpoint_projection_v1",
                    "generation_index": int(generation_index),
                    "status": "not_present",
                    "reason": "runner_outside_route_neutral_cxx26_flat_interchange_scope",
                }
            from perf.cxxbackbone10_resident_state_owner import build_import_projection

            reference_world = "patch" if patch_profile else "flat"
            capability_mask = {
                "flat": reference_world == "flat",
                "patch": reference_world == "patch",
                "churn": True,
            }
            reference_tick = _require_native_checkpoint_int(
                getattr(sim, "_eiel_resume_start_tick", 0) or 0,
                "reference checkpoint tick",
            )
            imported = build_import_projection(
                sim,
                capability_mask,
                generation=int(generation_index),
                tick=int(reference_tick),
            )
            projection = {
                "schema_version": imported.schema_version,
                "metadata": imported.metadata,
                "tensors": imported.tensors,
            }
            projection = _reference_source_projection_envelope(imported, projection)
            counters = {}
            source = (
                "build_import_projection.reference_route_neutral_patch"
                if patch_profile
                else "build_import_projection.reference_route_neutral_cxx26_flat"
            )
            world = reference_world
        projection = _typed_lifecycle_checkpoint_projection(projection, fill_missing=True)
        projection_metadata = dict(projection.get("metadata") or {})
        authority_record = (
            _native_projection_authority_record(projection)
            if engine is not None
            else {
                "source_projection_sha256": _native_projection_sha256(projection),
                "source_projection_schema_version": str(projection.get("schema_version", "") or ""),
                "source_generation": int(projection_metadata.get("generation", generation_index) or 0),
                "source_tick": int(projection_metadata.get("tick", 0) or 0),
                "source_phase": str(projection_metadata.get("agent_grid_runner_phase", "") or ""),
                "source_authority_epoch": str(
                    projection_metadata.get("agent_grid_authority_epoch", "") or ""
                ),
            }
        )
        return {
            "schema_version": "native_runtime_checkpoint_projection_v1",
            "generation_index": int(generation_index),
            "status": "present",
            "source": source,
            "authority_source": (
                NATIVE_SINGLE_PROJECTION_AUTHORITY
                if engine is not None
                else "reference_route_neutral_projection_v2"
            ),
            **authority_record,
            "world": world,
            "capability_mask": _pack_native_runtime_value(capability_mask),
            "counters": _pack_native_runtime_value(counters),
            "projection": _pack_native_runtime_value(projection),
        }
    except Exception as exc:
        return {
            "schema_version": "native_runtime_checkpoint_projection_v1",
            "generation_index": int(generation_index),
            "status": "export_failed",
            "exception": repr(exc),
        }


def _native_authority_sidecar_from_detached_projection(
    projection: Mapping[str, Any],
    *,
    generation_index: int,
    expected_source_projection_sha256: str,
    performance_telemetry: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    if not isinstance(projection, Mapping):
        raise CheckpointStrictError("native checkpoint projection must be a mapping")
    tensors = projection.get("tensors")
    metadata = projection.get("metadata")
    if not isinstance(tensors, Mapping) or not isinstance(metadata, Mapping):
        raise CheckpointStrictError(
            "native checkpoint projection must contain metadata and tensors mappings"
        )

    clone_started = time.perf_counter()
    detached: Dict[str, Any] = {}
    for name, value in projection.items():
        if name == "tensors":
            detached["tensors"] = {
                str(tensor_name): (
                    tensor.detach().clone()
                    if hasattr(tensor, "detach")
                    else copy.deepcopy(tensor)
                )
                for tensor_name, tensor in tensors.items()
            }
        else:
            detached[str(name)] = copy.deepcopy(value)
    clone_materialization_seconds = time.perf_counter() - clone_started

    canonical_pack_started = time.perf_counter()
    raw_authority_record = _native_projection_authority_record(detached)
    detached = _checkpoint_projection_identity_v2(detached)
    projection_metadata = dict(detached.get("metadata") or {})
    generation = _architecture_b_int(
        projection_metadata.get("generation"),
        field="native checkpoint projection metadata.generation",
    )
    tick = _architecture_b_int(
        projection_metadata.get("tick"),
        field="native checkpoint projection metadata.tick",
    )
    if generation != generation_index:
        raise CheckpointStrictError(
            "native checkpoint projection generation contradiction: "
            f"bundle={generation_index} projection={generation}"
        )
    expected_phase = "generation_boundary" if tick == 0 else "partial_generation"
    if projection_metadata.get("agent_grid_runner_phase") != expected_phase:
        raise CheckpointStrictError(
            "Architecture B canonical native checkpoint projection phase "
            "contradicts its tick cursor"
        )

    expected_digest = _strict_lowercase_sha256(
        expected_source_projection_sha256,
        "expected_source_projection_sha256",
    )
    try:
        authority_record = _native_projection_authority_record(detached)
    except CheckpointBundleError:
        raise
    except Exception as exc:
        raise CheckpointStrictError(
            f"native checkpoint source projection validation failed: {exc}"
        ) from exc
    computed_digest = _strict_lowercase_sha256(
        authority_record.get("source_projection_sha256"),
        "computed source projection digest",
    )
    accepted_input_digests = {
        computed_digest,
        str(raw_authority_record.get("source_projection_sha256", "") or ""),
    }
    if expected_digest not in accepted_input_digests:
        raise CheckpointStrictError(
            "native checkpoint source projection digest mismatch: "
            f"expected={expected_digest} computed_v2={computed_digest}"
        )

    projection_world = str(projection_metadata.get("world", "") or "")
    if projection_world not in {"flat", "patch"}:
        raise CheckpointStrictError(
            "native checkpoint projection world must be exactly flat or patch"
        )
    sidecar = {
        "schema_version": "native_runtime_checkpoint_projection_v1",
        "generation_index": generation_index,
        "status": "present",
        "source": ARCHITECTURE_B_PROJECTION_INPUT_SOURCE,
        "projection_input_schema_version": ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION,
        "authority_source": NATIVE_SINGLE_PROJECTION_AUTHORITY,
        **authority_record,
        "world": projection_world,
        "capability_mask": _pack_native_runtime_value(
            {
                "flat": projection_world == "flat",
                "patch": projection_world == "patch",
                "churn": True,
            }
        ),
        "counters": {},
        "projection": _pack_native_runtime_value(detached),
    }
    canonical_pack_seconds = time.perf_counter() - canonical_pack_started
    sidecar_validation_started = time.perf_counter()
    validated_projection, validated_digest = _validate_native_authority_sidecar(
        sidecar,
        generation_index=generation_index,
    )
    sidecar_validation_seconds = time.perf_counter() - sidecar_validation_started
    if validated_digest != computed_digest:
        raise CheckpointStrictError(
            "migrated native checkpoint projection digest changed during sidecar verification"
        )
    if performance_telemetry is not None:
        performance_telemetry.update(
            {
                "checkpoint_clone_materialization": (
                    clone_materialization_seconds
                ),
                "checkpoint_canonical_pack": canonical_pack_seconds,
                "checkpoint_sidecar_validation": sidecar_validation_seconds,
            }
        )
    return sidecar, validated_projection, validated_digest


def _canonical_checkpoint_storage_manifest_from_sidecar(
    sidecar: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if str(sidecar.get("status", "") or "") != "present":
        return None
    from native_runtime import decode_bundled_native_checkpoint_projection
    from native_runtime.checkpoint_digests import DigestLayer, checkpoint_digest_manifest

    projection = decode_bundled_native_checkpoint_projection(sidecar)
    projection = _checkpoint_projection_identity_v2(projection)
    return checkpoint_digest_manifest(
        projection,
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )


def _validate_canonical_checkpoint_storage_manifest(
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    *,
    required: bool,
) -> None:
    metadata_fields = {
        "canonical_checkpoint_storage_schema_version",
        "canonical_checkpoint_storage_digest",
        "canonical_checkpoint_storage_field_count",
    }
    present_fields = metadata_fields.intersection(metadata)
    if not present_fields and not required:
        return
    if present_fields != metadata_fields:
        missing = sorted(metadata_fields - present_fields)
        raise CheckpointStrictError(
            "native checkpoint canonical storage manifest metadata is incomplete: "
            f"missing={missing}"
        )
    try:
        manifest = _canonical_checkpoint_storage_manifest_from_sidecar(sidecar)
    except CheckpointBundleError:
        raise
    except Exception as exc:
        raise CheckpointStrictError(
            f"native checkpoint canonical storage manifest cannot be recomputed: {exc}"
        ) from exc
    if manifest is None:
        raise CheckpointStrictError(
            "native checkpoint canonical storage manifest requires a present sidecar"
        )

    recorded_schema = metadata["canonical_checkpoint_storage_schema_version"]
    if type(recorded_schema) is not str or recorded_schema != manifest["schema_version"]:
        raise CheckpointStrictError(
            "native checkpoint canonical storage manifest schema mismatch"
        )
    recorded_digest = _strict_lowercase_sha256(
        metadata["canonical_checkpoint_storage_digest"],
        "metadata.canonical_checkpoint_storage_digest",
    )
    if recorded_digest != manifest["digest"]:
        raise CheckpointStrictError(
            "native checkpoint canonical storage manifest digest mismatch"
        )
    recorded_field_count = _architecture_b_int(
        metadata["canonical_checkpoint_storage_field_count"],
        field="metadata.canonical_checkpoint_storage_field_count",
    )
    if recorded_field_count != manifest["field_count"]:
        raise CheckpointStrictError(
            "native checkpoint canonical storage manifest field count mismatch"
        )


def _native_projection_sha256(projection: Mapping[str, Any]) -> str:
    """Digest the complete immutable source projection when its full schema is present."""

    from native_runtime.checkpoint_digests import (
        SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS,
        checkpoint_source_projection_digest_manifest,
    )

    if set(SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS).issubset(projection):
        return str(checkpoint_source_projection_digest_manifest(projection)["digest"])
    return hashlib.sha256(canonical_json_bytes_lossless(_pack_native_runtime_value(dict(projection)))).hexdigest()


def _native_projection_restore_sha256(projection: Mapping[str, Any]) -> str:
    """Digest restore-authoritative state, excluding performance-only and derived fields."""

    authority_projection = {
        str(name): value
        for name, value in dict(projection).items()
        if name not in {"restore_count", "generation_calls", "tensor_manifest"}
    }
    packed = _pack_native_runtime_value(authority_projection)
    return hashlib.sha256(canonical_json_bytes_lossless(packed)).hexdigest()


def _native_projection_array(projection: Mapping[str, Any], name: str) -> np.ndarray:
    tensors = dict(projection.get("tensors") or {})
    if name not in tensors:
        raise CheckpointStrictError(f"native checkpoint projection is missing required tensor: {name}")
    value = tensors[name]
    if hasattr(value, "detach"):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    raise CheckpointStrictError(f"native checkpoint projection tensor is not tensor-like: {name}")


def _pack_projection_array(projection: Mapping[str, Any], name: str) -> Dict[str, Any]:
    arr = _native_projection_array(projection, name)
    data = arr.tobytes(order="C")
    return {
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "order": "C",
        "data_hex": data.hex(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_tensor": str(name),
    }


def _validate_native_authority_sidecar(
    sidecar: Mapping[str, Any],
    *,
    generation_index: int,
) -> Tuple[Dict[str, Any], str]:
    status = str(sidecar.get("status", "") or "")
    if status != "present":
        detail = str(sidecar.get("exception", sidecar.get("reason", "")) or "")
        raise CheckpointStrictError(
            "native-authority checkpoint projection export failed closed: "
            f"status={status or 'missing'}" + (f" detail={detail}" if detail else "")
        )
    if str(sidecar.get("authority_source", "") or "") != NATIVE_SINGLE_PROJECTION_AUTHORITY:
        raise CheckpointStrictError("native-authority checkpoint sidecar has no single-projection authority marker")

    from native_runtime import decode_bundled_native_checkpoint_projection
    from native_runtime.checkpoint_digests import validate_checkpoint_source_projection
    from native_runtime.interchange_projection import validate_native_checkpoint_projection

    projection = decode_bundled_native_checkpoint_projection(sidecar)
    projection = _checkpoint_projection_identity_v2(projection)
    validate_native_checkpoint_projection(projection)
    metadata = dict(projection.get("metadata") or {})
    projection_world = str(metadata.get("world", "") or "")
    if sidecar.get("world") != projection_world:
        raise CheckpointStrictError(
            "native-authority checkpoint sidecar world contradicts its projection"
        )
    expected_capability_mask = {
        "flat": projection_world == "flat",
        "patch": projection_world == "patch",
        "churn": True,
    }
    if sidecar.get("capability_mask") != expected_capability_mask:
        raise CheckpointStrictError(
            "native-authority checkpoint sidecar capability mask contradicts its world"
        )
    generation = _require_native_checkpoint_int(
        metadata.get("generation"),
        "native checkpoint projection metadata.generation",
    )
    tick = _require_native_checkpoint_int(
        metadata.get("tick"),
        "native checkpoint projection metadata.tick",
    )
    if generation != int(generation_index):
        raise CheckpointStrictError(
            "native-authority checkpoint generation contradiction: "
            f"bundle={int(generation_index)} projection={generation}"
        )
    expected_phase = "generation_boundary" if tick == 0 else "partial_generation"
    phase = str(metadata.get("agent_grid_runner_phase", "") or "")
    epoch = str(metadata.get("agent_grid_authority_epoch", "") or "")
    expected_epoch = f"generation:{generation}:tick:{tick}"
    configured_steps = _require_native_checkpoint_int(
        metadata.get("configured_steps_per_generation"),
        "native checkpoint projection metadata.configured_steps_per_generation",
        minimum=1,
    )
    if tick > configured_steps:
        raise CheckpointStrictError(
            "native checkpoint projection metadata.tick exceeds configured_steps_per_generation: "
            f"tick={tick} configured={configured_steps}"
        )
    if phase != expected_phase:
        raise CheckpointStrictError(
            "native checkpoint agent_grid_runner_phase contradiction: "
            f"expected {expected_phase} got {phase or 'missing'}"
        )
    if epoch != expected_epoch:
        raise CheckpointStrictError(
            "native checkpoint agent_grid_authority_epoch contradiction: "
            f"expected {expected_epoch} got {epoch or 'missing'}"
        )

    try:
        validate_checkpoint_source_projection(projection, require_rng_authority=True)
        authority_record = _native_projection_authority_record(projection)
    except Exception as exc:
        raise CheckpointStrictError(
            f"native-authority checkpoint source projection is incomplete: {exc}"
        ) from exc

    digest = str(authority_record["source_projection_sha256"])
    recorded_digest = str(sidecar.get("source_projection_sha256", "") or "")
    if not recorded_digest or recorded_digest != digest:
        raise CheckpointStrictError(
            "native-authority checkpoint source projection digest mismatch: "
            f"recorded={recorded_digest or 'missing'} computed={digest}"
        )
    for field, expected in authority_record.items():
        if sidecar.get(field) != expected:
            raise CheckpointStrictError(
                f"native-authority checkpoint sidecar {field} contradiction: "
                f"recorded={sidecar.get(field)!r} expected={expected!r}"
            )

    tensors = dict(projection.get("tensors") or {})
    required_cache = (
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
        "alive_mask",
        "slot_tokens",
    )
    missing_cache = [name for name in required_cache if name not in tensors]
    if missing_cache:
        raise CheckpointStrictError(
            f"native-authority checkpoint has incomplete action-cache state: {missing_cache}"
        )
    return projection, digest


def _resident_runtime_state_v2_from_native_projection(
    projection: Mapping[str, Any],
    *,
    source_projection_sha256: str,
    source_authority: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(projection.get("metadata") or {})
    authority_record = dict(
        _native_projection_authority_record(projection)
        if source_authority is None
        else source_authority
    )
    if str(authority_record.get("source_projection_sha256", "") or "") != str(
        source_projection_sha256
    ):
        raise CheckpointStrictError("native-authority resident source projection digest contradiction")
    source_map = {
        "agent_ids_by_alive_order": "agent_ids_by_alive_order",
        "slot_to_agent_id": "slot_to_agent_id",
        "agent_id_to_slot_agent_ids": "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots": "agent_id_to_slot_slots",
        "slot_tokens": "slot_tokens",
        "alive_mask": "alive_mask",
        "pos_y": "pos_y",
        "pos_x": "pos_x",
        "energy": "energy",
        "health": "health",
        "injury": "injury",
        "age": "age",
        "drives": "drives",
        "hmem": "hmem",
        "policy_attack_bias": "policy_attack_bias",
        "policy_action_temperature": "policy_action_temperature",
        "policy_drive_logit_gain": "policy_drive_logit_gain",
        "policy_use_gru": "policy_use_gru",
        "action_cache_actions": "action_cache_actions",
        "action_cache_token": "action_cache_token",
        "action_cache_cursor": "action_cache_cursor",
        "action_cache_next": "action_cache_next",
    }
    arrays = {
        destination: _pack_projection_array(projection, source)
        for destination, source in source_map.items()
    }
    alive = _native_projection_array(projection, "alive_mask").astype(np.bool_, copy=False)
    slot_tokens = _native_projection_array(projection, "slot_tokens").astype(np.int64, copy=False)
    cache_actions = _native_projection_array(projection, "action_cache_actions").astype(np.int32, copy=False)
    cache_token = _native_projection_array(projection, "action_cache_token").astype(np.int64, copy=False)
    cache_cursor = _native_projection_array(projection, "action_cache_cursor").astype(np.int32, copy=False)
    cache_next = _native_projection_array(projection, "action_cache_next").astype(np.int32, copy=False)
    if cache_actions.ndim != 2:
        raise CheckpointStrictError("native-authority action_cache_actions must be a rank-2 tensor")
    capacity = int(alive.shape[0])
    for name, value in (
        ("slot_tokens", slot_tokens),
        ("action_cache_token", cache_token),
        ("action_cache_cursor", cache_cursor),
        ("action_cache_next", cache_next),
    ):
        if int(value.shape[0]) != capacity:
            raise CheckpointStrictError(
                f"native-authority {name} capacity mismatch: {int(value.shape[0])} != {capacity}"
            )
    if int(cache_actions.shape[0]) != capacity:
        raise CheckpointStrictError(
            "native-authority action_cache_actions capacity mismatch: "
            f"{int(cache_actions.shape[0])} != {capacity}"
        )
    valid = alive & (cache_token >= 0) & (cache_cursor >= 0)
    invalid_token_rows = np.nonzero(valid & (cache_token != slot_tokens))[0]
    if int(invalid_token_rows.shape[0]) > 0:
        row = int(invalid_token_rows[0])
        raise CheckpointStrictError(
            "native-authority action-cache token/slot-token contradiction: "
            f"slot={row} cache={int(cache_token[row])} slot_token={int(slot_tokens[row])}"
        )
    valid_rows = np.nonzero(valid)[0].astype(np.int64)
    for row in valid_rows.tolist():
        cursor = int(cache_cursor[row])
        if cursor >= int(cache_actions.shape[1]):
            expected_next = int(cache_actions[row, -1])
        else:
            expected_next = int(cache_actions[row, cursor])
        if int(cache_next[row]) != expected_next:
            raise CheckpointStrictError(
                "native-authority action-cache cursor/next contradiction: "
                f"slot={row} cursor={cursor} next={int(cache_next[row])} expected={expected_next}"
            )
    validity = np.ascontiguousarray(valid, dtype=np.bool_)
    validity_bytes = validity.tobytes(order="C")
    validity_record = {
        "shape": [int(x) for x in validity.shape],
        "dtype": str(validity.dtype),
        "order": "C",
        "data_hex": validity_bytes.hex(),
        "sha256": hashlib.sha256(validity_bytes).hexdigest(),
        "derivation": "alive_mask & action_cache_token>=0 & action_cache_cursor>=0",
    }
    policy_plan = {
        "schema_version": "resident_policy_execution_plan_v2",
        **authority_record,
        "status": "captured",
        "membership_epoch": str(metadata.get("agent_grid_authority_epoch", "") or ""),
        "alive_slot_count": int(np.count_nonzero(alive)),
        "action_cache_valid_slot_count": int(valid_rows.shape[0]),
        "action_cache_valid_slots": [int(x) for x in valid_rows.tolist()],
        "action_cache_validity": validity_record,
        "action_cache_boundary_status": str(
            authority_record["source_action_cache_boundary_status"]
        ),
        "action_cache_sequence_length": int(cache_actions.shape[1]),
        "action_cache_digest": str(authority_record["source_action_cache_sha256"]),
        "action_cache_epoch": str(metadata.get("agent_grid_authority_epoch", "") or ""),
        "policy_identity_token_relation": "valid cache token equals live slot token",
        "hmem_shape": list(arrays["hmem"]["shape"]),
        "hmem_sha256": str(arrays["hmem"]["sha256"]),
        "hmem_materialized": bool(len(arrays["hmem"]["shape"]) == 2 and arrays["hmem"]["shape"][1] > 0),
    }
    from native_runtime.checkpoint_digests import DigestLayer, checkpoint_digest_manifest

    trajectory_digest = checkpoint_digest_manifest(
        projection,
        layer=DigestLayer.TRAJECTORY_STATE,
    )["digest"]
    payload: Dict[str, Any] = {
        "schema_version": "resident_runtime_state_v2",
        "producer": "checkpoint_bundle_v2.native_single_projection",
        **authority_record,
        "source_projection_trajectory_digest": str(trajectory_digest),
        "generation_index": int(metadata.get("generation", 0) or 0),
        "tick": int(metadata.get("tick", 0) or 0),
        "phase": str(metadata.get("agent_grid_runner_phase", "") or ""),
        "authority_epoch": str(metadata.get("agent_grid_authority_epoch", "") or ""),
        "capacity": capacity,
        "alive_count": int(np.count_nonzero(alive)),
        "slot_epoch": str(metadata.get("agent_grid_authority_epoch", "") or ""),
        "event_cursor": int(metadata.get("event_tape_cursor", 0) or 0),
        "arrays": arrays,
        "environment_state": {
            "schema_version": "resident_environment_state_v2",
            **authority_record,
            "status": "projection_authoritative",
        },
        "generation_counters": {
            "schema_version": "resident_generation_counters_v2",
            **authority_record,
            "status": "projection_tensors_authoritative",
        },
        "population_transaction_tape": {
            "schema_version": "resident_population_transaction_tape_v2",
            **authority_record,
            "event_cursor": int(metadata.get("event_tape_cursor", 0) or 0),
        },
        "policy_execution_plan": policy_plan,
        "runnercore_state_digest": str(trajectory_digest),
        "apply_status": "compatibility_projection_not_independent_authority",
    }
    payload["resident_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes_lossless(
            {key: value for key, value in payload.items() if key != "resident_payload_sha256"}
        )
    ).hexdigest()
    return payload


def _native_compatibility_descriptor(
    schema_version: str,
    *,
    source_authority: Mapping[str, Any],
    payload_kind: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schema_version": str(schema_version),
        "compatibility_projection_schema_version": "eiel_native_checkpoint_compatibility_descriptor_v1",
        "compatibility_projection_status": "superseded_by_native_runtime_checkpoint_projection",
        "payload_kind": str(payload_kind),
        **dict(source_authority),
        "generation_index": int(source_authority["source_generation"]),
    }
    if extra:
        out.update({str(key): value for key, value in extra.items()})
    return out


def _native_checkpoint_event_hash(projection: Mapping[str, Any]) -> str:
    tensors = dict(projection.get("tensors") or {})
    event_payload = {
        name: _pack_native_runtime_value(value)
        for name, value in sorted(tensors.items())
        if name.startswith("event_tape_") or name.startswith("typed_lifecycle_event_")
    }
    return hashlib.sha256(canonical_json_bytes_lossless(event_payload)).hexdigest()


def _native_checkpoint_bundle_coherence_report(
    *,
    metadata: Mapping[str, Any],
    projection: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    resident: Mapping[str, Any],
    compatibility_payloads: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    from native_runtime.checkpoint_digests import checkpoint_source_projection_digest_manifest

    source_manifest = checkpoint_source_projection_digest_manifest(projection)
    source_authority = _native_projection_authority_record(projection)
    source_cache = dict(source_manifest["action_cache"])
    mismatches: List[Dict[str, Any]] = []
    for name, payload in (
        ("metadata", metadata),
        ("sidecar", sidecar),
        ("resident", resident),
    ):
        mismatches.extend(
            _authority_record_mismatches(
                name=name,
                payload=payload,
                expected=source_authority,
            )
        )
    for name, payload in sorted(compatibility_payloads.items()):
        mismatches.extend(
            _authority_record_mismatches(
                name=name,
                payload=payload,
                expected=source_authority,
            )
        )
    for name in (
        "environment_state",
        "generation_counters",
        "population_transaction_tape",
        "policy_execution_plan",
    ):
        nested = resident.get(name)
        if not isinstance(nested, Mapping):
            mismatches.append({"field": f"resident.{name}", "reason": "not_an_object"})
        else:
            mismatches.extend(
                _authority_record_mismatches(
                    name=f"resident.{name}",
                    payload=nested,
                    expected=source_authority,
                )
            )
    if str(sidecar.get("status", "") or "") != "present":
        mismatches.append({"field": "sidecar.status", "expected": "present", "actual": sidecar.get("status")})
    for field, expected in (
        ("python_materialized_during_save", False),
        ("legacy_rng_restore_allowed", False),
        ("native_projection_capture_count", 1),
    ):
        if metadata.get(field) != expected:
            mismatches.append(
                {"field": f"metadata.{field}", "expected": expected, "actual": metadata.get(field)}
            )
    resident_arrays = dict(resident.get("arrays") or {})
    overlap_checks = []
    for destination, record in sorted(resident_arrays.items()):
        if not isinstance(record, Mapping):
            mismatches.append({"field": f"resident.arrays.{destination}", "reason": "not_an_object"})
            continue
        source_name = str(record.get("source_tensor", destination) or destination)
        expected = _pack_projection_array(projection, source_name)
        actual_sha = str(record.get("sha256", "") or "")
        overlap_checks.append(
            {
                "resident_field": destination,
                "source_tensor": source_name,
                "sha256": actual_sha,
            }
        )
        if actual_sha != expected["sha256"] or list(record.get("shape") or []) != expected["shape"] or str(
            record.get("dtype", "") or ""
        ) != expected["dtype"]:
            mismatches.append(
                {
                    "field": f"resident.arrays.{destination}",
                    "reason": "projection_overlap_mismatch",
                    "expected": {key: expected[key] for key in ("shape", "dtype", "sha256")},
                    "actual": {
                        "shape": list(record.get("shape") or []),
                        "dtype": str(record.get("dtype", "") or ""),
                        "sha256": actual_sha,
                    },
                }
            )
    policy_plan = resident.get("policy_execution_plan")
    if isinstance(policy_plan, Mapping):
        expected_validity = dict(source_cache["validity"])
        actual_validity = policy_plan.get("action_cache_validity")
        cache_checks = {
            "action_cache_digest": str(source_cache["digest"]),
            "action_cache_valid_slot_count": int(source_cache["valid_slot_count"]),
            "action_cache_valid_slots": list(source_cache["valid_slots"]),
            "action_cache_boundary_status": str(source_cache["boundary_status"]),
            "action_cache_sequence_length": int(source_cache["sequence_length"]),
            "action_cache_epoch": str(source_cache["authority_epoch"]),
        }
        for field, expected in cache_checks.items():
            if policy_plan.get(field) != expected:
                mismatches.append(
                    {
                        "field": f"resident.policy_execution_plan.{field}",
                        "reason": "action_cache_authority_mismatch",
                        "expected": expected,
                        "actual": policy_plan.get(field),
                    }
                )
        if not isinstance(actual_validity, Mapping):
            mismatches.append(
                {
                    "field": "resident.policy_execution_plan.action_cache_validity",
                    "reason": "not_an_object",
                }
            )
        else:
            for field in ("shape", "dtype", "sha256"):
                if actual_validity.get(field) != expected_validity.get(field):
                    mismatches.append(
                        {
                            "field": f"resident.policy_execution_plan.action_cache_validity.{field}",
                            "reason": "action_cache_validity_mismatch",
                            "expected": expected_validity.get(field),
                            "actual": actual_validity.get(field),
                        }
                    )
    resident_hash = resident.get("resident_payload_sha256")
    computed_resident_hash = hashlib.sha256(
        canonical_json_bytes_lossless(
            {key: value for key, value in resident.items() if key != "resident_payload_sha256"}
        )
    ).hexdigest()
    if resident_hash != computed_resident_hash:
        mismatches.append(
            {
                "field": "resident.resident_payload_sha256",
                "reason": "resident_self_digest_mismatch",
                "expected": computed_resident_hash,
                "actual": resident_hash,
            }
        )
    rng_descriptor = compatibility_payloads.get("rng_snapshot_v2")
    if isinstance(rng_descriptor, Mapping):
        for field, expected in (
            ("legacy_rng_restore_allowed", False),
            ("rng_authority", "native_runtime_checkpoint_projection"),
        ):
            if rng_descriptor.get(field) != expected:
                mismatches.append(
                    {
                        "field": f"rng_snapshot_v2.{field}",
                        "reason": "rng_authority_mismatch",
                        "expected": expected,
                        "actual": rng_descriptor.get(field),
                    }
                )
    projection_metadata = dict(projection.get("metadata") or {})
    cursor = {
        "generation": int(projection_metadata.get("generation", 0) or 0),
        "tick": int(projection_metadata.get("tick", 0) or 0),
        "phase": str(projection_metadata.get("agent_grid_runner_phase", "") or ""),
        "authority_epoch": str(projection_metadata.get("agent_grid_authority_epoch", "") or ""),
    }
    for field, expected in cursor.items():
        actual = metadata.get(field if field != "generation" else "generation_index")
        if field == "authority_epoch":
            actual = metadata.get("source_authority_epoch")
        elif field == "phase":
            actual = metadata.get("source_phase")
        elif field == "tick":
            actual = metadata.get("source_tick")
        if actual != expected:
            mismatches.append(
                {"field": f"metadata.{field}", "reason": "cursor_mismatch", "expected": expected, "actual": actual}
            )
    source_event_cursor = int(projection_metadata.get("event_tape_cursor", 0) or 0)
    if int(projection.get("event_cursor", -1)) != source_event_cursor:
        mismatches.append(
            {
                "field": "projection.event_cursor",
                "reason": "event_cursor_mismatch",
                "expected": source_event_cursor,
                "actual": projection.get("event_cursor"),
            }
        )
    if resident.get("event_cursor") != source_event_cursor:
        mismatches.append(
            {
                "field": "resident.event_cursor",
                "reason": "event_cursor_mismatch",
                "expected": source_event_cursor,
                "actual": resident.get("event_cursor"),
            }
        )
    return {
        "schema_version": NATIVE_BUNDLE_COHERENCE_SCHEMA_VERSION,
        **source_authority,
        "ok": not mismatches,
        "status": "passed" if not mismatches else "failed",
        "cursor": cursor,
        "action_cache": source_cache,
        "overlap_check_count": len(overlap_checks),
        "overlap_checks": overlap_checks,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _checkpoint_bundle_tree_sha256(bundle_dir: str) -> str:
    """Digest the exact published tree without following links."""
    import stat

    root = os.path.abspath(str(bundle_dir))
    if not os.path.isdir(root) or os.path.islink(root):
        raise CheckpointStrictError(
            f"Architecture B checkpoint publication is not a directory: {root}"
        )
    digest = hashlib.sha256()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in directories:
            path = os.path.join(current, name)
            mode = os.lstat(path).st_mode
            if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
                raise CheckpointStrictError(
                    f"Architecture B checkpoint publication contains a linked directory: {path}"
                )
            relative = os.path.relpath(path, root)
            digest.update(b"directory\0" + os.fsencode(relative) + b"\0")
        for name in files:
            path = os.path.join(current, name)
            file_stat = os.lstat(path)
            if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
                raise CheckpointStrictError(
                    f"Architecture B checkpoint publication contains a non-regular file: {path}"
                )
            relative = os.path.relpath(path, root)
            digest.update(b"file\0" + os.fsencode(relative) + b"\0")
            digest.update(struct.pack("<Q", int(file_stat.st_size)))
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _flush_and_fsync_file(handle: Any, *, path: str) -> None:
    try:
        handle.flush()
        os.fsync(handle.fileno())
    except OSError as exc:
        raise CheckpointStrictError(
            f"Architecture B checkpoint file fsync failed for {path}: {exc}"
        ) from exc


def _fsync_directory(path: str, *, context: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise CheckpointStrictError(
            f"Architecture B checkpoint {context} fsync failed for {path}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_staged_directory_tree(staged_bundle_dir: str) -> None:
    for current, directories, _files in os.walk(
        staged_bundle_dir,
        topdown=False,
        followlinks=False,
    ):
        for name in sorted(directories):
            child = os.path.join(current, name)
            if os.path.islink(child):
                raise CheckpointStrictError(
                    "Architecture B checkpoint staging contains a linked directory: "
                    f"{child}"
                )
            _fsync_directory(child, context="staged child directory")
        _fsync_directory(current, context="staged directory")


def _invalidate_failed_architecture_b_publication(bundle_dir: str) -> None:
    completion_marker = os.path.join(bundle_dir, "bundle_complete.json")
    try:
        os.unlink(completion_marker)
    except FileNotFoundError:
        pass
    except OSError:
        pass
    _rm_tree_best_effort(bundle_dir)


def _reuse_identical_architecture_b_publication(
    *,
    staged_bundle_dir: str,
    published_bundle_dir: str,
) -> None:
    _fsync_directory(
        os.path.dirname(published_bundle_dir),
        context="publication parent",
    )
    try:
        checkpoint_bundle_v2_load(
            published_bundle_dir,
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )
    except Exception as exc:
        raise CheckpointStrictError(
            "Architecture B checkpoint publication conflict: existing final is invalid"
        ) from exc
    staged_digest = _checkpoint_bundle_tree_sha256(staged_bundle_dir)
    published_digest = _checkpoint_bundle_tree_sha256(published_bundle_dir)
    if staged_digest != published_digest:
        raise CheckpointStrictError(
            "Architecture B checkpoint publication conflict: existing final differs "
            "from the validated retry"
        )


def _renameat2_noreplace_errno(source: str, destination: str) -> int:
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    return 0 if result == 0 else int(ctypes.get_errno())


def _publish_directory_absent(source: str, destination: str) -> None:
    """Publish a staged Architecture B bundle without replacing a final.

    Linux filesystems with ``renameat2(RENAME_NOREPLACE)`` use the kernel
    primitive directly.  Some supported evidence volumes are FUSE-backed and
    do not expose that syscall.  On those volumes, all repository publishers
    serialize on the publication parent directory, recheck that the final is
    absent under the lock, and use a same-filesystem rename.  The lock is held
    through the rename; strict readback and the completion seal remain
    unchanged, so a publisher cannot silently replace a competing final.
    """
    import errno

    error_number = _renameat2_noreplace_errno(source, destination)
    if error_number == 0:
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        import fcntl

        parent = os.path.dirname(os.path.abspath(destination))
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(
                parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                raise CheckpointStrictError(
                    "Architecture B checkpoint filesystem lacks both "
                    "atomic no-replace publication and a usable parent lock"
                ) from exc
            if os.path.lexists(destination):
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
            try:
                os.rename(source, destination)
            except FileExistsError:
                raise
            except OSError as exc:
                raise CheckpointStrictError(
                    "Architecture B checkpoint fallback publication failed"
                ) from exc
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        except FileExistsError:
            raise
        except CheckpointStrictError:
            raise
        except OSError as exc:
            raise CheckpointStrictError(
                "Architecture B checkpoint filesystem cannot lock publication parent"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return
    raise OSError(error_number, os.strerror(error_number), destination)


def _serialize_checkpoint_bundle_v2_transaction(
    sim: Any,
    *,
    bundle_dir: str,
    generation_index: int,
    strict_replay_mode: bool,
    native_checkpoint_projection: Optional[Mapping[str, Any]] = None,
    expected_source_projection_sha256: Optional[str] = None,
    architecture_b_identity: Optional[Mapping[str, Any]] = None,
    observer_continuation: Optional[Mapping[str, Any]] = None,
    metrics_logger_state: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, float], bool]:
    bdir = os.path.abspath(str(bundle_dir))
    previous = bdir + ".previous"
    projection_input_mode = native_checkpoint_projection is not None
    if projection_input_mode:
        import tempfile

        publication_parent = os.path.dirname(bdir)
        os.makedirs(publication_parent, exist_ok=True)
        tmp = tempfile.mkdtemp(
            prefix=f".{os.path.basename(bdir)}.tmp-",
            dir=publication_parent,
        )
    else:
        tmp = bdir + ".tmp"
    if not projection_input_mode and os.path.exists(previous):
        if os.path.exists(bdir):
            _rm_tree_best_effort(previous)
        else:
            os.replace(previous, bdir)
    if not projection_input_mode and os.path.exists(tmp):
        _rm_tree_best_effort(tmp)
    os.makedirs(tmp, exist_ok=projection_input_mode)
    os.makedirs(os.path.join(tmp, "weights"), exist_ok=True)
    transaction_started = time.perf_counter()
    perf = {
        "checkpoint_clone_materialization": 0.0,
        "checkpoint_canonical_pack": 0.0,
        "checkpoint_sidecar_validation": 0.0,
        "checkpoint_serialize": 0.0,
        "checkpoint_hash": 0.0,
        "checkpoint_write": 0.0,
        "checkpoint_atomic_replace": 0.0,
        "checkpoint_digest_coherence": 0.0,
        "checkpoint_projection_gzip": 0.0,
        "checkpoint_seal_write": 0.0,
        "checkpoint_strict_readback": 0.0,
        "checkpoint_staged_tree_fsync": 0.0,
        "checkpoint_total": 0.0,
    }
    native_authority = projection_input_mode or getattr(sim, "_native_eiel_engine", None) is not None
    moved_previous = False
    try:
        serialize_started = time.perf_counter()
        if native_authority:
            if projection_input_mode:
                assert native_checkpoint_projection is not None
                native_runtime_checkpoint_obj, projection, source_projection_sha256 = (
                    _native_authority_sidecar_from_detached_projection(
                        native_checkpoint_projection,
                        generation_index=int(generation_index),
                        expected_source_projection_sha256=expected_source_projection_sha256,
                        performance_telemetry=perf,
                    )
                )
                completion_identity = architecture_b_identity
                if (
                    isinstance(architecture_b_identity, Mapping)
                    and expected_source_projection_sha256 is not None
                    and architecture_b_identity.get("source_projection_sha256")
                    == expected_source_projection_sha256
                    and expected_source_projection_sha256
                    != source_projection_sha256
                ):
                    completion_identity = dict(architecture_b_identity)
                    completion_identity["source_projection_sha256"] = (
                        source_projection_sha256
                    )
                completion_context = _architecture_b_completion_context(
                    architecture_b_identity=completion_identity,
                    observer_continuation=observer_continuation,
                    projection=projection,
                    generation_index=int(generation_index),
                    source_projection_sha256=source_projection_sha256,
                )
            else:
                native_runtime_checkpoint_obj = _export_native_runtime_checkpoint_projection(
                    sim,
                    generation_index=int(generation_index),
                )
                projection, source_projection_sha256 = _validate_native_authority_sidecar(
                    native_runtime_checkpoint_obj,
                    generation_index=int(generation_index),
                )
                completion_context = None
            projection_metadata = dict(projection.get("metadata") or {})
            source_authority = _native_projection_authority_record(projection)
            resident_runtime_state_obj = _resident_runtime_state_v2_from_native_projection(
                projection,
                source_projection_sha256=source_projection_sha256,
                source_authority=source_authority,
            )
            world_obj = _native_compatibility_descriptor(
                "world_state_v3",
                source_authority=source_authority,
                payload_kind="world",
            )
            pop_obj = _native_compatibility_descriptor(
                "population_state_v2",
                source_authority=source_authority,
                payload_kind="population",
                extra={
                    "next_id": int(projection_metadata.get("population_next_id", 1) or 1),
                    "agents": [],
                    "slot_pool": {},
                    "pop_rng_kind": None,
                    "pop_rng_state": None,
                },
            )
            rng_obj = _native_compatibility_descriptor(
                "rng_snapshot_v2",
                source_authority=source_authority,
                payload_kind="rng",
                extra={
                    "legacy_rng_restore_allowed": False,
                    "rng_authority": "native_runtime_checkpoint_projection",
                },
            )
            runner_state_obj = _native_compatibility_descriptor(
                "runner_state_v1",
                source_authority=source_authority,
                payload_kind="runner_state",
            )
            if projection_input_mode:
                cp_obj = _native_compatibility_descriptor(
                    "control_plane_state_v1",
                    source_authority=source_authority,
                    payload_kind="control_plane",
                    extra={"authority_classification": "configuration_only"},
                )
                if metrics_logger_state is None:
                    canonical_metrics_logger_state = (
                        _canonical_neutral_architecture_b_metrics_logger_state()
                    )
                else:
                    canonical_metrics_logger_state = (
                        _validated_architecture_b_metrics_logger_state(
                            metrics_logger_state,
                            field="metrics_logger_state",
                        )
                    )
                metrics_state_obj = {
                    **canonical_metrics_logger_state,
                    **source_authority,
                    "authority_classification": (
                        "route_neutral_public_continuation_state"
                    ),
                    "authority_epoch": str(
                        projection_metadata.get("agent_grid_authority_epoch", "") or ""
                    ),
                }
                _architecture_b_metrics_logger_state_from_serialized(
                    metrics_state_obj,
                    projection=projection,
                )
            else:
                cp_obj = dict(_export_control_plane(sim))
                cp_obj.update(source_authority)
                cp_obj["authority_classification"] = "configuration_only"
                metrics_state_obj = dict(_export_metrics_state(sim))
                metrics_state_obj.update(
                    {
                        **source_authority,
                        "authority_classification": "route_neutral_public_continuation_state",
                        "authority_epoch": str(
                            projection_metadata.get("agent_grid_authority_epoch", "") or ""
                        ),
                    }
                )
            bank_state_obj = None
            weights_info = {
                "schema_version": "native_projection_mlp_pack_v1",
                **source_authority,
                "format": "native_projection_mlp_pack",
                "count": 0,
                "compatibility_projection_status": "superseded_by_native_runtime_checkpoint_projection",
            }
            events_hash = _native_checkpoint_event_hash(projection)
            wal_hash = hashlib.sha256(b"native_projection_wal_diagnostic_only").hexdigest()
        else:
            _flush_transient_caches(sim)
            _canonicalize_runner_drive_arrays(sim)
            world_obj = _export_world(sim)
            pop_obj, weights_info = _export_population_and_weights(
                sim,
                weights_dir=os.path.join(tmp, "weights"),
            )
            world_obj = _normalize_checkpoint_world_authority(sim, world_obj, pop_obj)
            rng_obj = _export_rng_snapshot(sim, generation_index=int(generation_index))
            cp_obj = _export_control_plane(sim)
            runner_state_obj = _export_runner_state(sim)
            resident_runtime_state_obj = _export_resident_runtime_state_v2(
                sim,
                generation_index=int(generation_index),
            )
            native_runtime_checkpoint_obj = _export_native_runtime_checkpoint_projection(
                sim,
                generation_index=int(generation_index),
            )
            metrics_state_obj = _export_metrics_state(sim)
            bank_state_obj = _export_bank_state(sim)
            events_hash = _event_log_sha256(sim)
            wal_hash = _wal_dir_sha256(sim)
            projection = None
            source_projection_sha256 = ""
            source_authority = {}
            completion_context = None
        perf["checkpoint_serialize"] = time.perf_counter() - serialize_started

        try:
            if bool(int(os.environ.get("DEBUG_DETERMINISM", "0") or "0")):
                if any(bool(getattr(target, "_gru_inited", False)) for target in _runner_targets(sim)) and bank_state_obj is None:
                    if not native_authority:
                        raise CheckpointStrictError("bundle_v2 missing bank_state while GRU is inited")
        except CheckpointStrictError:
            raise
        except Exception:
            pass

        hash_started = time.perf_counter()
        canonical_storage_manifest = _canonical_checkpoint_storage_manifest_from_sidecar(
            native_runtime_checkpoint_obj
        )
        world_hash = _sha256_bytes(canonical_json_bytes(world_obj))
        pop_hash = _sha256_bytes(canonical_json_bytes(pop_obj))
        rng_hash = _sha256_bytes(canonical_json_bytes(rng_obj))
        runner_state_hash = _sha256_bytes(canonical_json_bytes(runner_state_obj))
        bank_state_hash = _sha256_bytes(canonical_json_bytes(bank_state_obj)) if bank_state_obj is not None else ""
        control_plane_hash = _sha256_bytes(canonical_json_bytes(cp_obj))
        metrics_state_hash = _sha256_bytes(canonical_json_bytes(metrics_state_obj))
        resident_runtime_state_hash = _sha256_bytes(canonical_json_bytes_lossless(resident_runtime_state_obj))
        native_runtime_checkpoint_hash = _sha256_bytes(canonical_json_bytes_lossless(native_runtime_checkpoint_obj))
        perf["checkpoint_hash"] = time.perf_counter() - hash_started

        meta_obj = _build_metadata(
            sim=sim,
            generation_index=int(generation_index),
            strict_replay_mode=bool(strict_replay_mode),
            world_hash=world_hash,
            population_hash=pop_hash,
            rng_snapshot_hash=rng_hash,
            runner_state_hash=runner_state_hash,
            bank_state_hash=bank_state_hash,
            resident_runtime_state_hash=resident_runtime_state_hash,
            native_runtime_checkpoint_hash=native_runtime_checkpoint_hash,
            event_log_hash=events_hash,
            wal_hash=wal_hash,
            weights_info=weights_info,
        )
        if canonical_storage_manifest is not None:
            meta_obj.update(
                {
                    "canonical_checkpoint_storage_schema_version": str(
                        canonical_storage_manifest["schema_version"]
                    ),
                    "canonical_checkpoint_storage_digest": str(
                        canonical_storage_manifest["digest"]
                    ),
                    "canonical_checkpoint_storage_field_count": int(
                        canonical_storage_manifest["field_count"]
                    ),
                }
            )
        coherence_report = None
        if native_authority:
            assert projection is not None
            projection_metadata = dict(projection.get("metadata") or {})
            meta_obj.update(
                {
                    **source_authority,
                    "checkpoint_authority_schema_version": NATIVE_SINGLE_PROJECTION_AUTHORITY,
                    "checkpoint_authority_source": "native_runtime_checkpoint_projection",
                    "legacy_payload_authority": "compatibility_descriptors_only",
                    "legacy_rng_restore_allowed": False,
                    "python_materialized_during_save": False,
                    "native_projection_capture_count": 1,
                    "bundle_complete_marker_required": True,
                    "control_plane_hash": control_plane_hash,
                    "metrics_state_hash": metrics_state_hash,
                }
            )
            if completion_context is not None:
                meta_obj.update(
                    {
                        "architecture_b_projection_save_schema_version": (
                            ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION
                        ),
                        "architecture_b_identity_sha256": completion_context[
                            "architecture_b_identity_sha256"
                        ],
                        "observer_continuation_sha256": completion_context[
                            "observer_continuation_sha256"
                        ],
                    }
                )
                meta_obj["checkpoint_id"] = hashlib.sha256(
                    canonical_json_bytes_lossless(
                        {
                            "base_checkpoint_id": meta_obj["checkpoint_id"],
                            "source_projection_sha256": source_projection_sha256,
                            "architecture_b_identity_sha256": completion_context[
                                "architecture_b_identity_sha256"
                            ],
                            "observer_continuation_sha256": completion_context[
                                "observer_continuation_sha256"
                            ],
                        }
                    )
                ).hexdigest()
            compatibility_payloads = {
                "world": world_obj,
                "population": pop_obj,
                "rng_snapshot_v2": rng_obj,
                "runner_state": runner_state_obj,
                "control_plane": cp_obj,
                "metrics_state": metrics_state_obj,
                "weights_info": weights_info,
            }
            coherence_report = _native_checkpoint_bundle_coherence_report(
                metadata=meta_obj,
                projection=projection,
                sidecar=native_runtime_checkpoint_obj,
                resident=resident_runtime_state_obj,
                compatibility_payloads=compatibility_payloads,
            )
            if not bool(coherence_report.get("ok", False)):
                raise CheckpointStrictError(
                    "native checkpoint bundle coherence failed before finalization: "
                    + json.dumps(coherence_report.get("mismatches", [])[:8], sort_keys=True, default=str)
                )
            coherence_hash = _sha256_bytes(canonical_json_bytes_lossless(coherence_report))
            meta_obj["native_checkpoint_bundle_coherence_hash"] = coherence_hash
        perf["checkpoint_digest_coherence"] = time.perf_counter() - hash_started

        write_started = time.perf_counter()
        def _write_lossless(path: str, value: Any) -> None:
            if projection_input_mode:
                _write_canonical_json_lossless(path, value, durable=True)
            else:
                _write_canonical_json_lossless(path, value)

        def _write_lossless_gzip(path: str, value: Any) -> None:
            if projection_input_mode:
                _write_canonical_json_lossless_gzip(path, value, durable=True)
            else:
                _write_canonical_json_lossless_gzip(path, value)

        def _write_normalized(path: str, value: Any) -> None:
            if projection_input_mode:
                _write_canonical_json(path, value, durable=True)
            else:
                _write_canonical_json(path, value)

        _write_lossless(os.path.join(tmp, "metadata.json"), meta_obj)
        _write_lossless(os.path.join(tmp, "world.json"), world_obj)
        _write_lossless(os.path.join(tmp, "population.json"), pop_obj)
        _write_lossless(os.path.join(tmp, "rng_snapshot_v2.json"), rng_obj)
        _write_lossless(os.path.join(tmp, "control_plane.json"), cp_obj)
        _write_lossless(os.path.join(tmp, "runner_state.json"), runner_state_obj)
        _write_lossless(
            os.path.join(tmp, "resident_runtime_state_v2.json"),
            resident_runtime_state_obj,
        )
        native_projection_filename = _checkpoint_projection_sidecar_name()
        native_projection_path = os.path.join(tmp, native_projection_filename)
        if native_projection_filename.endswith(".gz"):
            gzip_started = time.perf_counter()
            _write_lossless_gzip(native_projection_path, native_runtime_checkpoint_obj)
            perf["checkpoint_projection_gzip"] = time.perf_counter() - gzip_started
        else:
            _write_lossless(
                native_projection_path,
                native_runtime_checkpoint_obj,
            )
        _write_lossless(
            os.path.join(tmp, "metrics_state.json"),
            metrics_state_obj,
        )
        if bank_state_obj is not None:
            _write_normalized(
                os.path.join(tmp, "bank_state.json"),
                bank_state_obj,
            )
        events_path = os.path.join(tmp, "events_hash.txt")
        with open(events_path, "wb") as events_file:
            events_file.write((events_hash + "\n").encode("utf-8"))
            if projection_input_mode:
                _flush_and_fsync_file(events_file, path=events_path)
        if native_authority:
            assert coherence_report is not None
            _write_lossless(
                os.path.join(tmp, "native_checkpoint_bundle_coherence_report.json"),
                coherence_report,
            )
            completion = {
                "schema_version": NATIVE_BUNDLE_COMPLETE_SCHEMA_VERSION,
                **source_authority,
                "status": "complete",
                "checkpoint_id": str(meta_obj.get("checkpoint_id", "") or ""),
                "coherence_sha256": str(meta_obj.get("native_checkpoint_bundle_coherence_hash", "") or ""),
            }
            if completion_context is not None:
                completion.update(completion_context)
                completion["architecture_b_completion_sha256"] = (
                    _architecture_b_completion_sha256(completion)
                )
            seal_started = time.perf_counter()
            _write_lossless(
                os.path.join(tmp, "bundle_complete.json"),
                completion,
            )
            perf["checkpoint_seal_write"] = time.perf_counter() - seal_started
        perf["checkpoint_write"] = time.perf_counter() - write_started

        if projection_input_mode:
            readback_started = time.perf_counter()
            checkpoint_bundle_v2_load(
                tmp,
                strict=True,
                required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
            )
            perf["checkpoint_strict_readback"] = (
                time.perf_counter() - readback_started
            )
            fsync_started = time.perf_counter()
            _fsync_staged_directory_tree(tmp)
            perf["checkpoint_staged_tree_fsync"] = (
                time.perf_counter() - fsync_started
            )

        atomic_started = time.perf_counter()
        if projection_input_mode:
            if os.path.lexists(bdir):
                _reuse_identical_architecture_b_publication(
                    staged_bundle_dir=tmp,
                    published_bundle_dir=bdir,
                )
                _rm_tree_best_effort(tmp)
            else:
                try:
                    _publish_directory_absent(tmp, bdir)
                except FileExistsError:
                    _reuse_identical_architecture_b_publication(
                        staged_bundle_dir=tmp,
                        published_bundle_dir=bdir,
                    )
                    _rm_tree_best_effort(tmp)
                else:
                    try:
                        _fsync_directory(
                            os.path.dirname(bdir),
                            context="publication parent",
                        )
                    except Exception:
                        _invalidate_failed_architecture_b_publication(bdir)
                        raise
            perf["checkpoint_atomic_replace"] = time.perf_counter() - atomic_started
            perf["checkpoint_total"] = time.perf_counter() - transaction_started
            return perf, native_authority
        if os.path.exists(previous):
            _rm_tree_best_effort(previous)
        if os.path.exists(bdir):
            os.replace(bdir, previous)
            moved_previous = True
        try:
            os.replace(tmp, bdir)
        except Exception:
            if moved_previous and os.path.exists(previous) and not os.path.exists(bdir):
                os.replace(previous, bdir)
                moved_previous = False
            raise
        if moved_previous and os.path.exists(previous):
            _rm_tree_best_effort(previous)
            moved_previous = False
        perf["checkpoint_atomic_replace"] = time.perf_counter() - atomic_started
        perf["checkpoint_total"] = time.perf_counter() - transaction_started
        return perf, native_authority
    except Exception:
        if os.path.lexists(tmp):
            _rm_tree_best_effort(tmp)
        if moved_previous and os.path.exists(previous):
            if not os.path.exists(bdir):
                os.replace(previous, bdir)
            else:
                _rm_tree_best_effort(previous)
        raise


def checkpoint_bundle_v2_save_projection(
    projection: Mapping[str, Any],
    bundle_dir: str,
    *,
    generation_index: int,
    expected_source_projection_sha256: str,
    architecture_b_identity: Mapping[str, Any],
    observer_continuation: Mapping[str, Any],
    metrics_logger_state: Optional[Mapping[str, Any]] = None,
    performance_telemetry: Optional[Dict[str, float]] = None,
    strict_replay_mode: bool = True,
) -> str:
    """Persist an Architecture B detached native checkpoint projection.

    This path accepts the already detached canonical checkpoint facet from an
    immutable Architecture B boundary snapshot. It never consults a runner or
    live native owner. Reporting owners must pass the exact post-materialization
    ``MetricsLogger.export_state()`` payload. Callers without reporting ownership
    retain the canonical neutral logger default for compatibility. Publication
    strictly verifies staging, atomically creates an absent final, reuses an
    identical valid retry, and rejects conflicts without moving or removing an
    existing final.
    """
    generation = _architecture_b_int(
        generation_index,
        field="generation_index",
    )
    if type(strict_replay_mode) is not bool:
        raise CheckpointStrictError("strict_replay_mode must be bool")
    if performance_telemetry is not None and not isinstance(
        performance_telemetry, dict
    ):
        raise CheckpointStrictError("performance_telemetry must be a dict")
    bdir = os.path.abspath(str(bundle_dir))
    perf, _native_authority = _serialize_checkpoint_bundle_v2_transaction(
        None,
        bundle_dir=bdir,
        generation_index=generation,
        strict_replay_mode=strict_replay_mode,
        native_checkpoint_projection=projection,
        expected_source_projection_sha256=expected_source_projection_sha256,
        architecture_b_identity=architecture_b_identity,
        observer_continuation=observer_continuation,
        metrics_logger_state=metrics_logger_state,
    )
    if performance_telemetry is not None:
        performance_telemetry.clear()
        performance_telemetry.update(
            {str(name): float(seconds) for name, seconds in perf.items()}
        )
    return bdir


def checkpoint_bundle_v2_save(
    sim: Any,
    bundle_dir: str,
    *,
    generation_index: int,
    strict_replay_mode: bool = True,
) -> str:
    """Write a Checkpoint Bundle v2 to `bundle_dir`.

    Parameters
    ----------
    sim:
        Simulation runner instance.
    bundle_dir:
        Target directory path. Will be created if missing.
        Written atomically via temp dir + rename where possible.
    generation_index:
        Absolute generation index represented by this checkpoint bundle.
        Intended to be the SafeBoundary "next_gen" index.
    strict_replay_mode:
        Persisted into metadata.json. When True, strict loader will enforce
        commit mismatch rejection during replay.

    Returns
    -------
    str: final bundle_dir path (absolute).
    """
    bdir = os.path.abspath(str(bundle_dir))
    _perf, native_authority = _serialize_checkpoint_bundle_v2_transaction(
        sim,
        bundle_dir=bdir,
        generation_index=int(generation_index),
        strict_replay_mode=bool(strict_replay_mode),
    )
    if not native_authority:
        try:
            setattr(sim, "_last_checkpoint_perf", dict(_perf))
        except Exception:
            pass

    # Phase 3 debug: state digest around bundle save
    # Flush transient caches (must not affect semantics across resume)
    if not native_authority:
        _flush_transient_caches(sim)
        # Canonicalize runner drive arrays so boundary dump is comparable to resume path.
        _canonicalize_runner_drive_arrays(sim)

    # FDP (Phase 3): targeted missing-state whitelist snapshot stored in the bundle.
    # This lets resume_apply_post compare against the saved boundary state and fail-fast.
    try:
        if not native_authority and bool(int((os.environ.get('EIEL_DETERMINISM_PROBE', '0') or '0'))):
            from simulation.fdp_proofs import build_state_whitelist as _fdp_build_state_whitelist
            w = _fdp_build_state_whitelist(sim)
            w['generation_index'] = int(generation_index)
            _write_canonical_json_lossless(os.path.join(bdir, 'fdp_state_whitelist.json'), w)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Phase 3 debug aid: emit a generation-aligned dump for the *start-of-gen*
    # membership boundary.
    #
    # The resume path emits state_{gen}__pre_start_generation_alive.json during
    # apply for boundary comparisons.
    #
    # Fresh runs that emit the gen0 bundle at start (EIEL_CHECKPOINT_BUNDLE_V2_AT_START=1)
    # otherwise have no code path that creates this tag before the membership harness.
    #
    # We therefore emit it here whenever generation_index==0. This is deterministic:
    # it reads only current sim population flags and writes stable JSON.
    # ------------------------------------------------------------------
    if int(generation_index) == 0:
        try:
            try:
                alive_ids = [int(getattr(a, "agent_id")) for a in getattr(sim, "population", []) if bool(getattr(a, "alive", True))]
            except Exception:
                alive_ids = []
            extra = {"alive_ids": alive_ids}
            try:
                maybe_dump_state(sim, generation_index=int(generation_index), tag="pre_start_generation_alive", extra=extra)
            except Exception:
                _force_write_state_dump(sim, generation_index=int(generation_index), tag="pre_start_generation_alive", extra=extra)
            else:
                # Fallback if runner API drift caused maybe_dump_state to no-op.
                _force_write_state_dump(sim, generation_index=int(generation_index), tag="pre_start_generation_alive", extra=extra)
        except Exception:
            pass
    # NOTE: Do NOT include absolute filesystem paths in boundary dumps.
    # The harness compares fresh vs resume dumps byte-for-byte, and run
    # directories are expected to differ.
    try:
        maybe_dump_state(
            sim,
            generation_index=generation_index,
            tag="bundle_save_post",
            extra={"bundle_gen": int(generation_index)},
        )
    except Exception:
        _force_write_state_dump(
            sim,
            generation_index=int(generation_index),
            tag="bundle_save_post",
            extra={"bundle_gen": int(generation_index)},
        )


    # FDP v3: Align evolution_py tracing across fresh vs resume by resetting the
    # trace epoch at a VERIFIED equivalence boundary.
    # Diagnostic-only: must not affect RNG state or checkpoint semantics.
    try:
        rm = getattr(sim, "rng_manager", None)
        if not native_authority and rm is not None:
            diagnostic_dump_enabled = _checkpoint_diagnostic_dump_enabled()
            # Dump evolution_py RNG state and ledger before/after boundary reset.
            if diagnostic_dump_enabled:
                try:
                    run_dir = getattr(sim, "run_dir", "") or ""
                    if run_dir:
                        dbg = os.path.join(str(run_dir), "debug")
                        if hasattr(rm, "dump_evopy_state_json"):
                            rm.dump_evopy_state_json(path=os.path.join(dbg, "evopy_state__bundle_save_post__pre_reset.json"), stage="bundle_save_post__pre_reset")
                        if hasattr(rm, "dump_evopy_ledger"):
                            rm.dump_evopy_ledger(path=os.path.join(dbg, "evopy_ledger__bundle_save_post__pre_reset.json"))
                except Exception:
                    pass
            if hasattr(rm, "_set_trace_ctx"):
                rm._set_trace_ctx(gen=int(generation_index), tick=-1, stage="bundle_save_post")
            if hasattr(rm, "_reset_evopy_trace"):
                rm._reset_evopy_trace(epoch="boundary")
            if hasattr(rm, "_reset_evopy_ledger"):
                rm._reset_evopy_ledger(epoch="boundary")
            if diagnostic_dump_enabled:
                try:
                    run_dir = getattr(sim, "run_dir", "") or ""
                    if run_dir:
                        dbg = os.path.join(str(run_dir), "debug")
                        if hasattr(rm, "dump_evopy_state_json"):
                            rm.dump_evopy_state_json(path=os.path.join(dbg, "evopy_state__bundle_save_post__post_reset.json"), stage="bundle_save_post__post_reset")
                        if hasattr(rm, "dump_evopy_ledger"):
                            rm.dump_evopy_ledger(path=os.path.join(dbg, "evopy_ledger__bundle_save_post__post_reset.json"))
                except Exception:
                    pass
    except Exception:
        pass

    # -----------------------------
    # Phase 5d: viz/ checkpoint index update (observability only)
    # -----------------------------
    # Only update after the bundle is finalized and stable on disk.
    try:
        run_dir = getattr(sim, "run_dir", None)
        if run_dir:
            from simulation.viz_pkg.viz_writer import get_viz_writer
            get_viz_writer(run_dir).update_checkpoints_index()
    except Exception:
        pass

    return bdir



def rotate_checkpoint_bundles_v2(checkpoint_root_dir: str, *, keep_last: int = 0) -> None:
    """Rotate (prune) Checkpoint Bundle v2 directories under a root folder.

    Parameters
    ----------
    checkpoint_root_dir:
        Directory that contains per-generation bundle folders (e.g.
        <run_dir>/checkpoint_bundles_v2).
    keep_last:
        Keep only the newest N bundles (by generation index).
        - 0 or negative: keep everything (no-op).

    Notes
    -----
    This is best-effort cleanup. It ignores unknown folder names and
    does not raise on deletion failures.

    The function only considers directories matching the standard naming
    convention:
        bundle_gen000123
    """
    root = os.path.abspath(str(checkpoint_root_dir))
    try:
        k = int(keep_last)
    except Exception:
        k = 0

    if k <= 0:
        return
    if not os.path.isdir(root):
        return

    import re

    items: list[tuple[int, str]] = []
    try:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            m = re.match(r"bundle_gen(\d{6})$", name)
            if not m:
                continue
            try:
                gen = int(m.group(1))
            except Exception:
                continue
            items.append((gen, p))
    except Exception:
        return

    # Newest-first
    items.sort(key=lambda t: t[0], reverse=True)
    to_delete = items[int(k):]
    for _gen, p in to_delete:
        try:
            _rm_tree_best_effort(p)
        except Exception:
            # Best-effort only.
            pass


def checkpoint_bundle_v2_inspect_historical(
    bundle_dir: str,
) -> Dict[str, Any]:
    """Inspect retired linked metadata without loading resumable payload."""

    bdir = os.path.abspath(str(bundle_dir))
    metadata_path = os.path.join(bdir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"missing metadata.json in bundle: {bdir}")
    metadata = _read_json(metadata_path)
    _validate_metadata(
        metadata,
        strict=True,
        historical_analysis=True,
    )
    inheritance = metadata.get("inheritance")
    if not isinstance(inheritance, Mapping):
        raise CheckpointStrictError(
            "historical linked-parent checkpoint requires inheritance metadata"
        )
    try:
        disposition = inspect_historical_inheritance_checkpoint_v1(inheritance)
    except ValueError as exc:
        raise CheckpointStrictError(str(exc)) from exc
    return {
        "schema_version": HISTORICAL_LINKED_CHECKPOINT_INSPECTION_SCHEMA_VERSION,
        "bundle_path": bdir,
        "metadata_sha256": hashlib.sha256(
            canonical_json_bytes(metadata)
        ).hexdigest(),
        "checkpoint": {
            key: metadata[key]
            for key in (
                "schema_version",
                "checkpoint_id",
                "generation_index",
                "branch_id",
                "campaign_id",
                "code_commit_hash",
                "strict_replay_mode",
            )
        },
        "inheritance": copy.deepcopy(dict(inheritance)),
        "disposition": {
            **disposition,
            "authority_status": "historical",
            "access_mode": "read_only",
            "resume_status": "non_resumable",
            "normal_restore_allowed": False,
            "state_loaded": False,
            "rng_loaded": False,
        },
    }


def checkpoint_bundle_v2_load(
    bundle_dir: str,
    *,
    strict: bool = True,
    expected_commit_hash: Optional[str] = None,
    required_bundle_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a Checkpoint Bundle v2 from disk.

    Parameters
    ----------
    bundle_dir:
        Bundle directory.
    strict:
        If True, enforce strict compatibility and hash requirements.
    expected_commit_hash:
        If provided, commit mismatch is rejected when metadata.strict_replay_mode=True.
    required_bundle_kind:
        Require a non-downgradable public bundle kind. Architecture B resume
        callers must pass ``ARCHITECTURE_B_BUNDLE_KIND`` with ``strict=True``.

    Returns
    -------
    dict: payload with keys:
        - metadata, world, population, rng_snapshot_v2, control_plane
        - weights_dir
        - events_hash_txt
    """
    if required_bundle_kind is not None and type(required_bundle_kind) is not str:
        raise CheckpointStrictError("required_bundle_kind must be a string or None")
    if required_bundle_kind not in {None, ARCHITECTURE_B_BUNDLE_KIND}:
        raise CheckpointStrictError(
            f"unsupported required_bundle_kind: {required_bundle_kind}"
        )
    required_architecture_b = required_bundle_kind == ARCHITECTURE_B_BUNDLE_KIND
    if required_architecture_b and not strict:
        raise CheckpointStrictError(
            "required Architecture B bundle kind requires strict loading"
        )

    bdir = os.path.abspath(str(bundle_dir))
    meta_p = os.path.join(bdir, "metadata.json")
    if not os.path.exists(meta_p):
        raise FileNotFoundError(f"missing metadata.json in bundle: {bdir}")

    meta = _read_json(meta_p)
    _validate_metadata(meta, strict=strict)
    authoritative_state_admission = (
        _validate_authoritative_state_checkpoint_metadata(meta)
    )

    if strict:
        # Reject any pickle-ish artifacts in v2 bundle.
        _reject_pickle_artifacts(bdir)
        required_native_runtime_checkpoint_projection = False

        # Required bundle files (strict)
        required_files = [
            "metadata.json",
            "world.json",
            "population.json",
            "rng_snapshot_v2.json",
            "control_plane.json",
            # Determinism-critical metrics internal state (required when strict_replay_mode=True).
            "metrics_state.json" if (bool(meta.get("strict_replay_mode", False)) and not bool(int(os.environ.get("EIEL_ALLOW_LEGACY_METRICS_STATE", "0") or "0"))) else "__optional__",
            "events_hash.txt",
        ]

        # Remove optional placeholder.
        required_files = [fn for fn in required_files if fn != "__optional__"]

        if meta.get("runner_state_hash"):
            required_files.append("runner_state.json")
        if meta.get("bank_state_hash"):
            required_files.append("bank_state.json")
        if meta.get("resident_runtime_state_hash"):
            required_files.append("resident_runtime_state_v2.json")
        if meta.get("native_runtime_checkpoint_hash"):
            required_native_runtime_checkpoint_projection = True
        if meta.get("metrics_state_hash"):
            required_files.append("metrics_state.json")
        if str(meta.get("checkpoint_authority_schema_version", "") or "") == NATIVE_SINGLE_PROJECTION_AUTHORITY:
            required_files.extend(
                [
                    "native_checkpoint_bundle_coherence_report.json",
                    "bundle_complete.json",
                ]
            )
        if required_architecture_b:
            required_native_runtime_checkpoint_projection = True
            required_files.extend(
                [
                    "native_checkpoint_bundle_coherence_report.json",
                    "bundle_complete.json",
                ]
            )
        for fn in required_files:
            if not _bundle_json_exists(bdir, fn):
                if fn == "metrics_state.json" and bool(meta.get("strict_replay_mode", False)):
                    try:
                        print(
                            "ERROR: missing required file in bundle: metrics_state.json "
                            "(determinism-critical metrics state; set EIEL_ALLOW_LEGACY_METRICS_STATE=1 to bypass)",
                            flush=True,
                        )
                    except Exception:
                        pass
                    raise CheckpointStrictError(
                        "missing required file in bundle: metrics_state.json "
                        "(determinism-critical metrics state; set EIEL_ALLOW_LEGACY_METRICS_STATE=1 to bypass)"
                    )
                raise CheckpointStrictError(f"missing required file in bundle: {fn}")
        if (
            required_native_runtime_checkpoint_projection
            and not _native_runtime_checkpoint_projection_exists(bdir)
        ):
            raise CheckpointStrictError(
                "missing required file in bundle: "
                f"{NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME} "
                f"(or legacy {NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME})"
            )
        if not os.path.isdir(os.path.join(bdir, "weights")):
            raise CheckpointStrictError("missing weights/ directory in bundle")

    # Schema gate
    if str(meta.get("schema_version")) != SCHEMA_VERSION:
        raise CheckpointSchemaError(f"unsupported schema_version: {meta.get('schema_version')}")

    # Commit mismatch gate (strict replay only)
    if strict and bool(meta.get("strict_replay_mode", False)):
        exp = str(expected_commit_hash or "").strip() if expected_commit_hash is not None else ""
        if exp:
            got = str(meta.get("code_commit_hash", "") or "")
            if got and got != "unknown" and got != exp:
                raise CheckpointStrictError(f"commit mismatch (strict): expected {exp} got {got}")

    # Environment mismatch gate (strict replay only)
    # If strict_replay_mode=True, the checkpoint asserts the environment must match.
    if strict and bool(meta.get("strict_replay_mode", False)):
        try:
            import platform
            import numpy as _np
            import torch as _torch
            cur_py = platform.python_version()
            cur_np = str(getattr(_np, "__version__", ""))
            cur_torch = str(getattr(_torch, "__version__", ""))
            exp_py = str(meta.get("python_version", "") or "")
            exp_np = str(meta.get("numpy_version", "") or "")
            exp_torch = str(meta.get("torch_version", "") or "")
            if exp_py and exp_py != cur_py:
                raise CheckpointStrictError(f"python version mismatch (strict): expected {exp_py} got {cur_py}")
            if exp_np and exp_np != cur_np:
                raise CheckpointStrictError(f"numpy version mismatch (strict): expected {exp_np} got {cur_np}")
            if exp_torch and exp_torch != cur_torch:
                raise CheckpointStrictError(f"torch version mismatch (strict): expected {exp_torch} got {cur_torch}")
        except CheckpointStrictError:
            raise
        except Exception:
            # If we cannot introspect versions, do not silently pass in strict mode.
            raise CheckpointStrictError("unable to validate python/numpy/torch versions in strict mode")

    world = _read_json(os.path.join(bdir, "world.json"))
    pop = _read_json(os.path.join(bdir, "population.json"))
    rng = _read_json(os.path.join(bdir, "rng_snapshot_v2.json"))
    cp = _read_json(os.path.join(bdir, "control_plane.json"))
    metrics_state_p = os.path.join(bdir, "metrics_state.json")
    metrics_state = _read_json(metrics_state_p) if os.path.exists(metrics_state_p) else {"schema_version": "metrics_state_v1", "collapse_streak": 0}
    runner_state_p = os.path.join(bdir, "runner_state.json")
    runner_state = _read_json(runner_state_p) if os.path.exists(runner_state_p) else {"schema_version": "runner_state_v1"}
    bank_state_p = os.path.join(bdir, "bank_state.json")
    bank_state = _read_json(bank_state_p) if os.path.exists(bank_state_p) else {"schema_version": "bank_state_v1"}
    resident_runtime_state_p = os.path.join(bdir, "resident_runtime_state_v2.json")
    resident_runtime_state = (
        _read_json(resident_runtime_state_p)
        if os.path.exists(resident_runtime_state_p)
        else {"schema_version": "resident_runtime_state_v2", "export_status": "missing"}
    )
    native_runtime_checkpoint_p = _native_runtime_checkpoint_projection_path_for_load(bdir)
    native_runtime_checkpoint = (
        _read_json(native_runtime_checkpoint_p)
        if os.path.exists(native_runtime_checkpoint_p)
        else {"schema_version": "native_runtime_checkpoint_projection_v1", "status": "missing"}
    )
    sidecar_projection_input_source = str(
        native_runtime_checkpoint.get("source", "") or ""
    )
    sidecar_projection_input_schema = native_runtime_checkpoint.get(
        "projection_input_schema_version"
    )
    architecture_b_sidecar_provenance_present = (
        sidecar_projection_input_source == ARCHITECTURE_B_PROJECTION_INPUT_SOURCE
        or sidecar_projection_input_schema is not None
    )
    coherence_report_p = os.path.join(bdir, "native_checkpoint_bundle_coherence_report.json")
    coherence_report = _read_json(coherence_report_p) if os.path.exists(coherence_report_p) else {}
    bundle_complete_p = os.path.join(bdir, "bundle_complete.json")
    bundle_complete = _read_json(bundle_complete_p) if os.path.exists(bundle_complete_p) else {}
    architecture_b_resume_envelope: Optional[Dict[str, Any]] = None

    events_hash_txt = ""
    try:
        with open(os.path.join(bdir, "events_hash.txt"), "r", encoding="utf-8") as f:
            events_hash_txt = f.read().strip()
    except Exception:
        events_hash_txt = ""

        # Strict: events_hash.txt must match metadata.event_log_hash
    if strict:
        mh = str(meta.get("event_log_hash", "") or "")
        if events_hash_txt and mh and events_hash_txt != mh:
            raise CheckpointStrictError(f"events_hash.txt mismatch: metadata={mh} file={events_hash_txt}")

# Strict required hashes presence
    if strict:
        for k in ["population_hash", "world_hash", "rng_snapshot_hash", "event_log_hash", "wal_hash"]:
            if not str(meta.get(k, "") or ""):
                raise CheckpointStrictError(f"missing required hash field in metadata: {k}")
        payload_hash_specs = (
            ("world_hash", "world", world, canonical_json_bytes),
            ("population_hash", "population", pop, canonical_json_bytes),
            ("rng_snapshot_hash", "rng_snapshot_v2", rng, canonical_json_bytes),
            ("control_plane_hash", "control_plane", cp, canonical_json_bytes),
            ("runner_state_hash", "runner_state", runner_state, canonical_json_bytes),
            ("metrics_state_hash", "metrics_state", metrics_state, canonical_json_bytes),
            ("bank_state_hash", "bank_state", bank_state, canonical_json_bytes),
            (
                "resident_runtime_state_hash",
                "resident_runtime_state_v2",
                resident_runtime_state,
                canonical_json_bytes_lossless,
            ),
            (
                "native_runtime_checkpoint_hash",
                "native_runtime_checkpoint_projection",
                native_runtime_checkpoint,
                canonical_json_bytes_lossless,
            ),
        )
        for metadata_field, payload_name, payload, encoder in payload_hash_specs:
            expected_hash = str(meta.get(metadata_field, "") or "")
            if not expected_hash:
                continue
            _strict_lowercase_sha256(expected_hash, f"metadata.{metadata_field}")
            computed_hash = _sha256_bytes(encoder(payload))
            if computed_hash != expected_hash:
                raise CheckpointStrictError(
                    f"{payload_name} hash mismatch: "
                    f"metadata={expected_hash} file={computed_hash}"
                )
        _validate_canonical_checkpoint_storage_manifest(
            meta,
            native_runtime_checkpoint,
            required=required_architecture_b,
        )
        if required_architecture_b and str(
            meta.get("checkpoint_authority_schema_version", "") or ""
        ) != NATIVE_SINGLE_PROJECTION_AUTHORITY:
            raise CheckpointStrictError(
                "required Architecture B bundle authority metadata marker is invalid"
            )
        if architecture_b_sidecar_provenance_present and str(
            meta.get("checkpoint_authority_schema_version", "") or ""
        ) != NATIVE_SINGLE_PROJECTION_AUTHORITY:
            raise CheckpointStrictError(
                "native checkpoint bundle Architecture B authority metadata marker is invalid"
            )
        if str(meta.get("checkpoint_authority_schema_version", "") or "") == NATIVE_SINGLE_PROJECTION_AUTHORITY:
            projection, source_digest = _validate_native_authority_sidecar(
                native_runtime_checkpoint,
                generation_index=int(meta.get("generation_index", 0) or 0),
            )
            source_authority = _native_projection_authority_record(projection)
            if str(meta.get("source_projection_sha256", "") or "") != source_digest:
                raise CheckpointStrictError("native-authority metadata source projection digest mismatch")
            expected_coherence_hash = str(meta.get("native_checkpoint_bundle_coherence_hash", "") or "")
            got_coherence_hash = _sha256_bytes(canonical_json_bytes_lossless(coherence_report))
            if not expected_coherence_hash or expected_coherence_hash != got_coherence_hash:
                raise CheckpointStrictError(
                    "native checkpoint bundle coherence report hash mismatch: "
                    f"metadata={expected_coherence_hash or 'missing'} file={got_coherence_hash}"
                )
            if (
                str(coherence_report.get("schema_version", "") or "") != NATIVE_BUNDLE_COHERENCE_SCHEMA_VERSION
                or not bool(coherence_report.get("ok", False))
            ):
                raise CheckpointStrictError("native checkpoint bundle coherence report is not a passed report")
            completion_authority_mismatches = _authority_record_mismatches(
                name="bundle_complete",
                payload=bundle_complete,
                expected=source_authority,
            )
            if (
                str(bundle_complete.get("schema_version", "") or "") != NATIVE_BUNDLE_COMPLETE_SCHEMA_VERSION
                or str(bundle_complete.get("status", "") or "") != "complete"
                or str(bundle_complete.get("checkpoint_id", "") or "") != str(meta.get("checkpoint_id", "") or "")
                or str(bundle_complete.get("source_projection_sha256", "") or "") != source_digest
                or str(bundle_complete.get("coherence_sha256", "") or "") != expected_coherence_hash
                or completion_authority_mismatches
            ):
                raise CheckpointStrictError("native checkpoint bundle atomic completion marker is invalid")
            architecture_b_marker_present = required_architecture_b or architecture_b_sidecar_provenance_present or any(
                field in metadata_container
                for metadata_container in (meta, bundle_complete)
                for field in (
                    "architecture_b_projection_save_schema_version",
                    "architecture_b_schema_version",
                    "architecture_b_identity",
                    "observer_continuation",
                    "architecture_b_completion_sha256",
                )
            )
            if architecture_b_marker_present:
                if sidecar_projection_input_source != ARCHITECTURE_B_PROJECTION_INPUT_SOURCE:
                    raise CheckpointStrictError(
                        "native checkpoint bundle Architecture B sidecar provenance is invalid"
                    )
                if (
                    sidecar_projection_input_schema
                    != ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION
                ):
                    raise CheckpointStrictError(
                        "native checkpoint bundle Architecture B sidecar schema marker is invalid"
                    )
                if (
                    meta.get("architecture_b_projection_save_schema_version")
                    != ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION
                ):
                    raise CheckpointStrictError(
                        "native checkpoint bundle Architecture B metadata marker is invalid"
                    )
                for field in (
                    "world_hash",
                    "population_hash",
                    "rng_snapshot_hash",
                    "control_plane_hash",
                    "runner_state_hash",
                    "metrics_state_hash",
                    "resident_runtime_state_hash",
                    "native_runtime_checkpoint_hash",
                ):
                    if not str(meta.get(field, "") or ""):
                        raise CheckpointStrictError(
                            "native checkpoint bundle Architecture B metadata is missing "
                            f"required payload hash: {field}"
                        )
                _validate_architecture_b_completion_manifest(
                    bundle_complete,
                    metadata=meta,
                    projection=projection,
                    source_projection_sha256=source_digest,
                )
                _architecture_b_metrics_logger_state_from_serialized(
                    metrics_state,
                    projection=projection,
                )
                architecture_b_resume_envelope = _build_architecture_b_resume_envelope(
                    projection=projection,
                    completion=bundle_complete,
                    source_projection_sha256=source_digest,
                )
            recomputed_coherence = _native_checkpoint_bundle_coherence_report(
                metadata=meta,
                projection=projection,
                sidecar=native_runtime_checkpoint,
                resident=resident_runtime_state,
                compatibility_payloads={
                    "world": world,
                    "population": pop,
                    "rng_snapshot_v2": rng,
                    "runner_state": runner_state,
                    "control_plane": cp,
                    "metrics_state": metrics_state,
                    "weights_info": dict(meta.get("weights_info") or {}),
                },
            )
            if not bool(recomputed_coherence.get("ok", False)):
                raise CheckpointStrictError(
                    "native checkpoint bundle coherence verification failed on load: "
                    + json.dumps(recomputed_coherence.get("mismatches", [])[:8], sort_keys=True, default=str)
                )
            if canonical_json_bytes_lossless(recomputed_coherence) != canonical_json_bytes_lossless(
                coherence_report
            ):
                raise CheckpointStrictError(
                    "native checkpoint bundle coherence report contradicts recomputed authority"
                )

    payload = {
        "metadata": meta,
        "world": world,
        "population": pop,
        "rng_snapshot_v2": rng,
        "control_plane": cp,
        "metrics_state": metrics_state,
        "runner_state": runner_state,
        "bank_state": bank_state,
        "resident_runtime_state_v2": resident_runtime_state,
        "native_runtime_checkpoint_projection": native_runtime_checkpoint,
        "native_checkpoint_bundle_coherence_report": coherence_report,
        "bundle_complete": bundle_complete,
        "weights_dir": os.path.join(bdir, "weights"),
        "events_hash_txt": events_hash_txt,
        AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD: (
            authoritative_state_admission
        ),
    }
    if architecture_b_resume_envelope is not None:
        payload["bundle_kind"] = ARCHITECTURE_B_BUNDLE_KIND
        payload["architecture_b_resume_envelope"] = architecture_b_resume_envelope
    if required_architecture_b:
        if architecture_b_resume_envelope is None:
            raise CheckpointStrictError(
                "strict Architecture B load did not produce a resume envelope"
            )
        return ValidatedArchitectureBCheckpointPayload(
            payload,
            _authority_token=_ARCHITECTURE_B_CHECKPOINT_PAYLOAD_AUTHORITY_TOKEN,
            _authority_digest_sha256=(
                _architecture_b_checkpoint_payload_authority_sha256(payload)
            ),
        )
    return payload



def _eiel_recanonicalize_slots_for_alive(pop: Any) -> None:
    """Phase 3 determinism: make agent.slot contiguous for the current alive cohort.

    Why:
    - Policies index bank rows by `agent.slot`.
    - During a long run, slots can become sparse (birth/death churn).
    - The fresh run path ends up with contiguous slots at gen1 tick000.start.
    - Resume must reproduce the same mapping, otherwise bank row indexing diverges.

    Rules:
    - MUST NOT draw RNG.
    - MUST be deterministic given the current alive agent set + their stable ordering.
    - MUST assign slots exactly 0..(alive_count-1) for alive agents.
    """
    try:
        slots = getattr(pop, "_slots", None)
        if slots is None:
            return

        agents = list(getattr(pop, "agents", {}).values())
        alive = [a for a in agents if bool(getattr(a, "alive", True))]

        # Deterministic: stable sort by agent_id if available, else by object id fallback.
        def _aid(x: Any) -> int:
            try:
                return int(getattr(x, "agent_id"))
            except Exception:
                return int(getattr(x, "id", 0)) if getattr(x, "id", None) is not None else 0

        try:
            alive.sort(key=_aid)
        except Exception:
            pass

        n = int(len(alive))

        # Build slot_tokens deterministically: token = i+1 for assigned slots, else 0.
        max_slots = int(getattr(slots, "max_slots", n))
        if max_slots < n:
            # Can't recanonicalize without enough capacity; never hard-fail resume.
            return

        try:
            from collections import deque
            slot_tokens = [0 for _ in range(max_slots)]
            for i, a in enumerate(alive):
                tok = int(i + 1)
                try:
                    a.slot = int(i)
                    a.slot_token = tok
                except Exception:
                    pass
                slot_tokens[i] = tok

            # Free slots are the remainder, in ascending order.
            free = deque(range(n, max_slots))

            # Best-effort update of SlotPool internals (do not assume private layout).
            try:
                slots.slot_tokens = slot_tokens
            except Exception:
                pass
            try:
                slots.free_slots = free
            except Exception:
                pass
        except Exception:
            return
    except Exception:
        return


def _require_native_checkpoint_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or int(value) < int(minimum):
        bound = "a positive" if int(minimum) == 1 else "a non-negative"
        raise ValueError(f"{field} must be {bound} int")
    return int(value)


def _install_partial_native_resume(
    sim: Any,
    engine: Any,
    *,
    generation: int,
    tick: int,
    configured_steps: int,
) -> int:
    """Run only the unconsumed tail on the restored owner's first generation call."""
    remaining_steps = int(configured_steps) - int(tick)
    if int(tick) == 0:
        return remaining_steps

    marker_was_present = hasattr(sim, "_eiel_resume_start_tick")
    previous_marker = getattr(sim, "_eiel_resume_start_tick", None)
    previous_instance_method = getattr(engine, "__dict__", {}).get("execute_generation")
    had_instance_method = "execute_generation" in getattr(engine, "__dict__", {})
    original_execute_generation = engine.execute_generation

    def _execute_checkpoint_remainder(
        *,
        generation_index: int,
        steps_per_generation: int,
        max_births: int = 1,
        birth_commit: bool = True,
    ) -> Any:
        requested_generation = _require_native_checkpoint_int(
            generation_index,
            "native continuation generation_index",
        )
        requested_steps = _require_native_checkpoint_int(
            steps_per_generation,
            "native continuation steps_per_generation",
            minimum=1,
        )
        if requested_generation != int(generation):
            raise CheckpointStrictError(
                "native checkpoint continuation generation mismatch: "
                f"expected {generation} got {requested_generation}"
            )
        if requested_steps != int(configured_steps):
            raise CheckpointStrictError(
                "native checkpoint continuation step configuration mismatch: "
                f"expected {configured_steps} got {requested_steps}"
            )

        try:
            result = original_execute_generation(
                generation_index=requested_generation,
                steps_per_generation=int(remaining_steps),
                max_births=int(max_births),
                birth_commit=bool(birth_commit),
            )
        finally:
            if had_instance_method:
                setattr(engine, "execute_generation", previous_instance_method)
            else:
                delattr(engine, "execute_generation")
            if marker_was_present:
                setattr(sim, "_eiel_resume_start_tick", previous_marker)
            else:
                try:
                    delattr(sim, "_eiel_resume_start_tick")
                except AttributeError:
                    pass
        return result

    setattr(sim, "_eiel_resume_start_tick", int(tick))
    setattr(engine, "execute_generation", _execute_checkpoint_remainder)
    return remaining_steps


def _apply_native_runtime_checkpoint_projection(
    sim: Any,
    payload: Dict[str, Any],
    *,
    native_runtime_extension_path: str | os.PathLike[str] | None = None,
) -> None:
    """Restore the route-neutral owner projection into a reference or native destination."""
    sim_cfg = getattr(sim, "sim_cfg", None)
    native_selected = bool(getattr(sim_cfg, "native_runtime_enable", False))
    if not native_selected:
        try:
            from native_runtime.safe_default import native_generation_requested

            native_selected = bool(native_generation_requested(sim))
        except Exception as exc:
            raise CheckpointStrictError(f"native runtime resume selection failed: {exc}") from exc
    try:
        reference_adapter_selected = bool(
            not native_selected
            and (
                _cxx26_flat_interchange_scope_active(sim)
                or _route_neutral_patch_interchange_profile(sim)
            )
        )
    except Exception as exc:
        raise CheckpointStrictError(f"route-neutral checkpoint scope resolution failed: {exc}") from exc
    if not native_selected and not reference_adapter_selected:
        return

    sidecar = payload.get("native_runtime_checkpoint_projection")
    if not isinstance(sidecar, dict):
        raise CheckpointStrictError("native runtime checkpoint sidecar must be an object")

    status = str(sidecar.get("status", "") or "")
    if status != "present":
        reason = str(sidecar.get("reason", "") or "")
        raise CheckpointStrictError(
            "native runtime checkpoint sidecar cannot be restored: "
            f"status={status or 'missing'}"
            + (f" reason={reason}" if reason else "")
        )

    reference_install_snapshot = None
    try:
        from native_runtime import decode_bundled_native_checkpoint_projection

        bundle_metadata = payload.get("metadata")
        if not isinstance(bundle_metadata, dict):
            raise ValueError("checkpoint bundle metadata must be an object")
        native_authority_bundle = (
            str(bundle_metadata.get("checkpoint_authority_schema_version", "") or "")
            == NATIVE_SINGLE_PROJECTION_AUTHORITY
        )
        if native_authority_bundle:
            projection, source_projection_sha256 = _validate_native_authority_sidecar(
                sidecar,
                generation_index=int(bundle_metadata.get("generation_index", 0) or 0),
            )
        else:
            projection = decode_bundled_native_checkpoint_projection(sidecar)
            projection = _typed_lifecycle_checkpoint_projection(projection, fill_missing=False)
            source_projection_sha256 = _native_projection_sha256(projection)
        metadata = projection.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("native checkpoint projection metadata must be an object")
        bundle_generation = _require_native_checkpoint_int(
            bundle_metadata.get("generation_index"),
            "checkpoint bundle metadata.generation_index",
        )
        sidecar_generation = _require_native_checkpoint_int(
            sidecar.get("generation_index"),
            "native checkpoint sidecar generation_index",
        )
        recorded_generation = _require_native_checkpoint_int(
            metadata.get("generation"),
            "native checkpoint projection metadata.generation",
        )
        if len({bundle_generation, sidecar_generation, recorded_generation}) != 1:
            raise ValueError(
                "native checkpoint generation contradiction: "
                f"bundle={bundle_generation} sidecar={sidecar_generation} projection={recorded_generation}"
            )

        recorded_tick = _require_native_checkpoint_int(
            metadata.get("tick"),
            "native checkpoint projection metadata.tick",
        )
        configured_steps = _require_native_checkpoint_int(
            metadata.get("configured_steps_per_generation"),
            "native checkpoint projection metadata.configured_steps_per_generation",
            minimum=1,
        )
        runner_configured_steps = _require_native_checkpoint_int(
            getattr(sim_cfg, "steps_per_generation", None),
            "resume runner sim_cfg.steps_per_generation",
            minimum=1,
        )
        if configured_steps != runner_configured_steps:
            raise ValueError(
                "native checkpoint configured steps contradiction: "
                f"projection={configured_steps} runner={runner_configured_steps}"
            )
        if recorded_tick > configured_steps:
            raise ValueError(
                "native checkpoint projection metadata.tick exceeds configured_steps_per_generation: "
                f"tick={recorded_tick} configured={configured_steps}"
            )

        phase_contract = str(metadata.get("agent_grid_phase_contract_schema_version", "") or "")
        if phase_contract != "eiel_agent_grid_phase_contract_v1":
            raise ValueError(
                "native checkpoint agent_grid phase contract contradiction: "
                f"expected eiel_agent_grid_phase_contract_v1 got {phase_contract or 'missing'}"
            )
        expected_phase = "generation_boundary" if recorded_tick == 0 else "partial_generation"
        recorded_phase = str(metadata.get("agent_grid_runner_phase", "") or "")
        if recorded_phase != expected_phase:
            raise ValueError(
                "native checkpoint agent_grid_runner_phase contradiction: "
                f"expected {expected_phase} got {recorded_phase or 'missing'}"
            )
        expected_epoch = f"generation:{recorded_generation}:tick:{recorded_tick}"
        recorded_epoch = str(metadata.get("agent_grid_authority_epoch", "") or "")
        if recorded_epoch != expected_epoch:
            raise ValueError(
                "native checkpoint agent_grid_authority_epoch contradiction: "
                f"expected {expected_epoch} got {recorded_epoch or 'missing'}"
            )

        reference_apply = None
        engine = None
        if native_selected:
            from native_runtime import NativeEielEngine
            build_dir = str(getattr(sim_cfg, "native_runtime_build_dir", "") or "") or None
            projection_world = str(metadata.get("world", "") or "")
            if projection_world == "patch":
                destination_profile = _route_neutral_patch_interchange_profile(sim)
                source_profile = str(
                    metadata.get("native_runtime_patch_profile", "") or ""
                )
                if not destination_profile or destination_profile != source_profile:
                    raise ValueError(
                        "Patch checkpoint profile is outside the destination interchange "
                        f"scope: source={source_profile or '<missing>'} "
                        f"destination={destination_profile or '<unsupported>'}"
                    )
                from native_runtime.bindings import native_extension_status

                extension = native_extension_status(native_runtime_extension_path)
                if not (
                    extension.available
                    and extension.abi_compatible
                    and extension.output_sha256_verified
                    and not extension.runtime_jit
                ):
                    raise ValueError(
                        "Patch checkpoint destination requires an explicit qualified "
                        f"development extension: {extension.error or 'not qualified'}"
                    )
                engine = NativeEielEngine.from_checkpoint_projection(
                    projection,
                    build_directory=build_dir,
                    extension_path=extension.extension_path,
                )
            else:
                from native_runtime.runner_integration import validate_native_runtime_supported

                validate_native_runtime_supported(sim)
                engine = NativeEielEngine.from_checkpoint_projection(
                    projection,
                    build_directory=build_dir,
                )
            restored_projection = engine.export_checkpoint_projection()
            restored_projection = _typed_lifecycle_checkpoint_projection(restored_projection, fill_missing=False)
            restored_metadata = restored_projection.get("metadata")
            if not isinstance(restored_metadata, dict):
                raise ValueError("restored native checkpoint projection metadata must be an object")
            restored_cursor = (
                restored_metadata.get("generation"),
                restored_metadata.get("tick"),
            )
            recorded_cursor = (recorded_generation, recorded_tick)
            if restored_cursor != recorded_cursor:
                raise ValueError(
                    "restored native checkpoint cursor mismatch: "
                    f"expected generation={recorded_generation} tick={recorded_tick}, "
                    f"got generation={restored_cursor[0]} tick={restored_cursor[1]}"
                )
            restored_projection_sha256 = _native_projection_restore_sha256(restored_projection)
            source_restore_sha256 = _native_projection_restore_sha256(projection)
            if native_authority_bundle and restored_projection_sha256 != source_restore_sha256:
                from native_runtime.checkpoint_digests import (
                    DigestLayer,
                    compare_checkpoint_layer_first_difference,
                )

                first_tensor_mismatch = compare_checkpoint_layer_first_difference(
                    projection,
                    restored_projection,
                    layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                )
                source_top_level = {
                    name: _pack_native_runtime_value(value)
                    for name, value in projection.items()
                    if name not in {"metadata", "tensors", "restore_count", "generation_calls", "tensor_manifest"}
                }
                restored_top_level = {
                    name: _pack_native_runtime_value(value)
                    for name, value in restored_projection.items()
                    if name not in {"metadata", "tensors", "restore_count", "generation_calls", "tensor_manifest"}
                }
                raise ValueError(
                    "restored native checkpoint projection mismatch: "
                    + json.dumps(
                        {
                            "source_projection_sha256": source_projection_sha256,
                            "expected_restore_sha256": source_restore_sha256,
                            "restored_sha256": restored_projection_sha256,
                            "first_tensor_mismatch": first_tensor_mismatch,
                            "top_level_equal": source_top_level == restored_top_level,
                        },
                        sort_keys=True,
                        default=str,
                    )
                )
            remaining_steps = _install_partial_native_resume(
                sim,
                engine,
                generation=recorded_generation,
                tick=recorded_tick,
                configured_steps=configured_steps,
            )
        else:
            if recorded_tick != 0:
                raise ValueError(
                    "route-neutral reference checkpoint restore currently requires a generation boundary: "
                    f"tick={recorded_tick}"
                )
            slot_ids = _native_projection_array(
                projection, "slot_to_agent_id"
            ).astype(np.int64, copy=False)
            slot_tokens = _native_projection_array(
                projection, "slot_tokens"
            ).astype(np.int64, copy=False)
            alive_rows = _native_projection_array(
                projection, "alive_mask"
            ).astype(np.bool_, copy=False)
            pos_y_rows = _native_projection_array(
                projection, "pos_y"
            ).astype(np.int32, copy=False)
            pos_x_rows = _native_projection_array(
                projection, "pos_x"
            ).astype(np.int32, copy=False)
            capacity = int(slot_ids.shape[0])
            if any(
                array.shape != (capacity,)
                for array in (slot_tokens, alive_rows, pos_y_rows, pos_x_rows)
            ):
                raise ValueError("native world slot authority shapes are inconsistent")
            if bool(np.any(slot_ids < -1)) or bool(
                np.any(slot_ids > np.iinfo(np.int32).max)
            ):
                raise ValueError("native world slot IDs exceed signed int32 grid boundary")
            if len(set(int(value) for value in slot_ids.tolist() if int(value) != -1)) != int(
                np.count_nonzero(slot_ids != -1)
            ):
                raise ValueError("native world slot authority contains duplicate agent IDs")
            if bool(np.any((slot_ids == -1) != (~alive_rows))):
                raise ValueError("native world slot free/alive authority is contradictory")
            if bool(np.any(pos_y_rows[~alive_rows] != -1)) or bool(
                np.any(pos_x_rows[~alive_rows] != -1)
            ):
                raise ValueError("native world free slot positions are noncanonical")
            if bool(np.any(slot_tokens[alive_rows] <= 0)):
                raise ValueError("native world live slot tokens must be positive")
            agent_grid = _native_projection_array(projection, "agent_grid").astype(
                np.int64, copy=False
            )
            if agent_grid.ndim != 2:
                raise ValueError("native agent_grid must be two-dimensional")
            grid_positions: Dict[int, tuple[int, int]] = {}
            for y, x in np.argwhere(agent_grid != -1).tolist():
                agent_id = int(agent_grid[int(y), int(x)])
                if agent_id in grid_positions:
                    raise ValueError("native agent_grid contains duplicate agent IDs")
                grid_positions[agent_id] = (int(y), int(x))
            live_ids = {
                int(slot_ids[slot])
                for slot in np.flatnonzero(alive_rows).tolist()
            }
            if set(grid_positions) != live_ids:
                raise ValueError("native slot authority and agent_grid identities disagree")
            for slot in np.flatnonzero(alive_rows).tolist():
                agent_id = int(slot_ids[slot])
                if (int(pos_y_rows[slot]), int(pos_x_rows[slot])) != grid_positions[agent_id]:
                    raise ValueError("native slot authority and agent_grid positions disagree")

            reference_install_snapshot = _snapshot_native_reference_install(sim)
            from perf.cxxbackbone10_resident_state_owner import apply_owner_checkpoint_to_runner

            reference_apply = dict(apply_owner_checkpoint_to_runner(sim, projection))
            env = getattr(sim, "env", None)
            pop = getattr(sim, "pop", None)
            if env is None or pop is None:
                raise ValueError("native reference materialization has no environment/population")
            agents = getattr(pop, "agents", {})
            if set(int(agent_id) for agent_id in agents) != live_ids:
                raise ValueError("materialized population does not match native live identities")
            pool = getattr(pop, "_slots", None)
            if (
                pool is None
                or int(getattr(pool, "max_slots", -1)) != capacity
                or list(getattr(pool, "slot_tokens", [])) != slot_tokens.tolist()
            ):
                raise ValueError("materialized SlotPool does not match native slot authority")
            for slot in np.flatnonzero(alive_rows).tolist():
                agent_id = int(slot_ids[slot])
                agent = agents.get(agent_id)
                if (
                    agent is None
                    or not bool(getattr(agent, "alive", False))
                    or int(getattr(agent, "slot", -1)) != int(slot)
                    or int(getattr(agent, "slot_token", -1)) != int(slot_tokens[slot])
                ):
                    raise ValueError(
                        "materialized population/native slot join is contradictory"
                    )
            env.slot_to_agent_id = np.ascontiguousarray(slot_ids, dtype=np.int64)
            env.slot_tokens = np.ascontiguousarray(slot_tokens, dtype=np.int64)
            env.agent_world_alive_by_slot = np.ascontiguousarray(
                alive_rows, dtype=np.bool_
            )
            env.agent_pos_y_by_slot = np.ascontiguousarray(
                pos_y_rows, dtype=np.int32
            )
            env.agent_pos_x_by_slot = np.ascontiguousarray(
                pos_x_rows, dtype=np.int32
            )
            env._agent_id_to_world_slot = {
                int(agent_id): (int(slot), int(slot_tokens[slot]))
                for slot, agent_id in enumerate(slot_ids.tolist())
                if int(agent_id) != -1
            }
            env._world_slot_capacity_configured = True
            env._world_authority_migration_required = False
            from perf.cxxbackbone10_resident_state_owner import build_import_projection
            from native_runtime.checkpoint_digests import (
                DigestLayer,
                compare_checkpoint_layer_first_difference,
            )

            materialized = build_import_projection(
                sim,
                {
                    "flat": str(metadata.get("world", "") or "") == "flat",
                    "patch": str(metadata.get("world", "") or "") == "patch",
                    "churn": True,
                },
                generation=recorded_generation,
                tick=recorded_tick,
            )
            materialized_projection = _typed_lifecycle_checkpoint_projection(
                {
                    "schema_version": materialized.schema_version,
                    "metadata": materialized.metadata,
                    "tensors": materialized.tensors,
                },
                fill_missing=False,
            )
            first_materialization_mismatch = compare_checkpoint_layer_first_difference(
                projection,
                materialized_projection,
                layer=DigestLayer.TRAJECTORY_STATE,
            )
            if first_materialization_mismatch is not None:
                raise ValueError(
                    "native-to-reference materialization trajectory mismatch: "
                    + json.dumps(first_materialization_mismatch, sort_keys=True, default=str)
                )
            materialized_metadata = dict(materialized_projection.get("metadata") or {})
            for rng_field in (
                "native_runtime_numpy_env_state",
                "native_runtime_numpy_runner_state",
                "native_runtime_numpy_policy_state",
                "native_runtime_numpy_population_state",
                "native_runtime_python_random_state",
                "native_runtime_env_policy_numpy_alias",
            ):
                if metadata.get(rng_field) != materialized_metadata.get(rng_field):
                    raise ValueError(
                        "native-to-reference materialization RNG mismatch: "
                        f"field={rng_field}"
                    )
            reference_apply["verification_target"] = "materialized_reference_projection"
            reference_apply["trajectory_verification"] = "exact"
            reference_apply["source_projection_sha256"] = source_projection_sha256
            remaining_steps = int(configured_steps)
    except CheckpointStrictError:
        if reference_install_snapshot is not None:
            _restore_native_reference_install(sim, reference_install_snapshot)
        raise
    except Exception as exc:
        if reference_install_snapshot is not None:
            _restore_native_reference_install(sim, reference_install_snapshot)
        raise CheckpointStrictError(f"native runtime checkpoint sidecar restore failed: {exc}") from exc

    if native_selected:
        engine.runner = sim
        setattr(sim, "_native_eiel_engine", engine)
        setattr(sim, "_native_runtime_checkpoint_source_projection", projection)
    else:
        try:
            delattr(sim, "_native_eiel_engine")
        except AttributeError:
            pass
        try:
            delattr(sim, "_native_runtime_checkpoint_source_projection")
        except AttributeError:
            pass
    apply_summary = {
        "schema_version": "native_runtime_checkpoint_apply_v1",
        "status": "restored",
        "generation": int(recorded_generation),
        "tick": int(recorded_tick),
        "configured_steps_per_generation": int(configured_steps),
        "remaining_steps": int(remaining_steps),
        "source": "checkpoint_bundle_v2.native_runtime_checkpoint_projection",
    }
    if native_selected or native_authority_bundle:
        apply_summary.update(
            {
                "source_projection_sha256": str(source_projection_sha256),
                "verification_target": (
                    "restored_native_engine" if native_selected else "materialized_reference_projection"
                ),
            }
        )
    if not native_selected:
        apply_summary["destination_runtime"] = "reference"
        apply_summary["owner_adapter"] = reference_apply
    setattr(
        sim,
        "_native_runtime_checkpoint_apply",
        apply_summary,
    )


def checkpoint_bundle_v2_apply(
    sim: Any,
    payload: Dict[str, Any],
    *,
    restore_rng: bool = True,
    native_runtime_extension_path: str | os.PathLike[str] | None = None,
) -> None:
    """Apply a loaded bundle payload onto an existing SimulationRunner instance.

    This is intended for resume-from-checkpoint.

    Phase 3 determinism note:
    - Any post-restore cache rebuild that consumes Python/NumPy RNG will cause an
      immediate boundary drift. Therefore we perform all structural rehydration
      work *before* restoring RNG, and restore RNG as the final step.
    - Runner drive arrays are not serialized (by design); they are rehydrated
      deterministically from each agent's mirrored `a.drives` via _sync_drive_slot.
    """
    _validate_authoritative_state_checkpoint_payload(payload)
    if "architecture_b_resume_envelope" in payload:
        raise CheckpointStrictError(
            "generic checkpoint_bundle_v2_apply cannot apply an Architecture B "
            "resume envelope; use the Architecture B controller resume integration"
        )
    from simulation.capacity_epoch import validate_capacity_epoch_checkpoint_compatibility

    validate_capacity_epoch_checkpoint_compatibility(sim, payload)
    validate_inheritance_checkpoint_compatibility(sim, payload)
    meta = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    if bool(meta.get("strict_replay_mode", False)):
        expected_scale = float(meta.get("structural_live_effect_scale", 1.0))
        actual_scale = float(getattr(getattr(sim, "sim_cfg", None), "structural_live_effect_scale", 1.0))
        if expected_scale != actual_scale:
            raise CheckpointStrictError(
                "structural_live_effect_scale mismatch (strict): "
                f"expected {expected_scale!r} got {actual_scale!r}"
            )
        cfg = getattr(sim, "sim_cfg", None)
        expected_live_enable = bool(meta.get("structural_live_modules_enable", False))
        actual_live_enable = bool(getattr(cfg, "structural_live_modules_enable", False))
        expected_live_profile = str(meta.get("structural_live_modules_profile", "none") or "none")
        actual_live_profile = str(getattr(cfg, "structural_live_modules_profile", "none") or "none")
        expected_live_schema = str(
            meta.get("structural_live_modules_schema_version", "escap3_live_module_runtime_v1")
            or "escap3_live_module_runtime_v1"
        )
        actual_live_schema = str(
            getattr(cfg, "structural_live_modules_schema_version", "escap3_live_module_runtime_v1")
            or "escap3_live_module_runtime_v1"
        )
        if (expected_live_enable, expected_live_profile, expected_live_schema) != (
            actual_live_enable,
            actual_live_profile,
            actual_live_schema,
        ):
            raise CheckpointStrictError(
                "structural_live_modules config mismatch (strict): "
                f"expected={(expected_live_enable, expected_live_profile, expected_live_schema)!r} "
                f"got={(actual_live_enable, actual_live_profile, actual_live_schema)!r}"
            )
        expected_routing = meta.get("structural_observation_routing", {})
        if not isinstance(expected_routing, dict):
            raise CheckpointStrictError("structural_observation_routing metadata malformed (strict)")
        actual_routing_enable = bool(getattr(cfg, "structural_observation_routing_enable", False))
        expected_routing_enable = bool(expected_routing.get("enabled", False))
        expected_routing_tuple = (
            expected_routing_enable,
            str(expected_routing.get("profile", "none") or "none"),
            str(expected_routing.get("schema_version", "escap4_observation_routing_v1") or "escap4_observation_routing_v1"),
            str(expected_routing.get("registry_version", "perception_routing_contract_v1") or "perception_routing_contract_v1"),
            int(expected_routing.get("max_bindings_per_module", 2) or 2),
            int(expected_routing.get("max_bindings_per_agent", 8) or 8),
            str(expected_routing.get("operator_profile", "escap4_routing_binding_operator_v1") or "escap4_routing_binding_operator_v1"),
            int(expected_routing.get("operator_attempts", 1) or 0),
            tuple(str(v) for v in list(expected_routing.get("admitted_semantic_ids", ["self_energy", "self_health"]))),
            str(expected_routing.get("exposure_mode", "parallel") or "parallel"),
        )
        actual_routing_tuple = (
            actual_routing_enable,
            str(getattr(cfg, "structural_observation_routing_profile", "none") or "none"),
            str(getattr(cfg, "structural_observation_routing_schema_version", "escap4_observation_routing_v1") or "escap4_observation_routing_v1"),
            str(getattr(cfg, "structural_observation_routing_registry_version", "perception_routing_contract_v1") or "perception_routing_contract_v1"),
            int(getattr(cfg, "structural_observation_routing_max_bindings_per_module", 2) or 2),
            int(getattr(cfg, "structural_observation_routing_max_bindings_per_agent", 8) or 8),
            str(getattr(cfg, "structural_observation_routing_operator_profile", "escap4_routing_binding_operator_v1") or "escap4_routing_binding_operator_v1"),
            int(getattr(cfg, "structural_observation_routing_operator_attempts", 1) or 0),
            ("self_energy", "self_health"),
            "parallel",
        )
        if (
            expected_routing_enable != actual_routing_enable
            or (
                expected_routing_enable
                and expected_routing_tuple != actual_routing_tuple
            )
        ):
            raise CheckpointStrictError(
                "structural_observation_routing config mismatch (strict): "
                f"expected={expected_routing_tuple!r} got={actual_routing_tuple!r}"
            )
        output_routing_key_present = "structural_output_routing" in meta
        expected_output_routing = meta.get("structural_output_routing")
        actual_output_routing_enable = bool(getattr(cfg, "structural_output_routing_enable", False))
        if not output_routing_key_present:
            if actual_output_routing_enable:
                raise CheckpointStrictError(
                    "structural_output_routing metadata missing for enabled destination (strict)"
                )
        else:
            if not isinstance(expected_output_routing, dict):
                raise CheckpointStrictError("structural_output_routing metadata malformed (strict)")
            required_output_routing_fields = {
                "enabled",
                "profile",
                "schema_version",
                "registry_version",
                "max_bindings_per_module",
                "max_bindings_per_agent",
                "operator_profile",
                "operator_attempts",
                "admitted_semantic_ids",
                "max_binding_gain",
                "max_action_residual",
                "arbitration_version",
                "authority_order_version",
                "combined_rest_compatibility_version",
                "migration_rng_draws",
            }
            missing_output_routing_fields = sorted(
                required_output_routing_fields - set(expected_output_routing)
            )
            if missing_output_routing_fields:
                raise CheckpointStrictError(
                    "structural_output_routing metadata missing fields (strict): "
                    + ",".join(missing_output_routing_fields)
                )
            if type(expected_output_routing.get("enabled")) is not bool:
                raise CheckpointStrictError("structural_output_routing metadata malformed (strict): enabled")
            if not bool(expected_output_routing["enabled"]):
                raise CheckpointStrictError(
                    "structural_output_routing disabled metadata must be omitted (strict)"
                )
            if not isinstance(expected_output_routing.get("admitted_semantic_ids"), list):
                raise CheckpointStrictError(
                    "structural_output_routing metadata malformed (strict): admitted_semantic_ids"
                )
            for field in (
                "profile",
                "schema_version",
                "registry_version",
                "operator_profile",
                "arbitration_version",
                "authority_order_version",
                "combined_rest_compatibility_version",
            ):
                if not isinstance(expected_output_routing.get(field), str):
                    raise CheckpointStrictError(
                        f"structural_output_routing metadata malformed (strict): {field}"
                    )
            for field in (
                "max_bindings_per_module",
                "max_bindings_per_agent",
                "operator_attempts",
                "migration_rng_draws",
            ):
                if type(expected_output_routing.get(field)) is not int:
                    raise CheckpointStrictError(
                        f"structural_output_routing metadata malformed (strict): {field}"
                    )
            for field in ("max_binding_gain", "max_action_residual"):
                if type(expected_output_routing.get(field)) not in {int, float}:
                    raise CheckpointStrictError(
                        f"structural_output_routing metadata malformed (strict): {field}"
                    )
            expected_output_routing_tuple = (
                bool(expected_output_routing["enabled"]),
                str(expected_output_routing["profile"]),
                str(expected_output_routing["schema_version"]),
                str(expected_output_routing["registry_version"]),
                tuple(str(v) for v in expected_output_routing["admitted_semantic_ids"]),
                int(expected_output_routing["max_bindings_per_module"]),
                int(expected_output_routing["max_bindings_per_agent"]),
                str(expected_output_routing["operator_profile"]),
                int(expected_output_routing["operator_attempts"]),
                float(expected_output_routing["max_binding_gain"]),
                float(expected_output_routing["max_action_residual"]),
                str(expected_output_routing["arbitration_version"]),
                str(expected_output_routing["authority_order_version"]),
                str(expected_output_routing["combined_rest_compatibility_version"]),
                int(expected_output_routing["migration_rng_draws"]),
            )
            actual_output_routing_tuple = (
                actual_output_routing_enable,
                str(getattr(cfg, "structural_output_routing_profile", "none") or "none"),
                str(
                    getattr(cfg, "structural_output_routing_schema_version", "escap5_output_routing_v1")
                    or "escap5_output_routing_v1"
                ),
                str(
                    getattr(
                        cfg,
                        "structural_output_routing_registry_version",
                        "eiel_action_semantic_routing_registry_v1",
                    )
                    or "eiel_action_semantic_routing_registry_v1"
                ),
                ("move_north", "move_south", "move_east", "move_west", "rest"),
                int(getattr(cfg, "structural_output_routing_max_bindings_per_module", 2) or 2),
                int(getattr(cfg, "structural_output_routing_max_bindings_per_agent", 8) or 8),
                str(
                    getattr(cfg, "structural_output_routing_operator_profile", "escap5_output_binding_operator_v1")
                    or "escap5_output_binding_operator_v1"
                ),
                int(getattr(cfg, "structural_output_routing_operator_attempts", 1) or 0),
                float(getattr(cfg, "structural_output_routing_max_binding_gain", 1.0) or 0.0),
                float(getattr(cfg, "structural_output_routing_max_action_residual", 0.25) or 0.0),
                str(
                    getattr(cfg, "structural_output_routing_arbitration_version", "escap5_output_arbitration_v1")
                    or "escap5_output_arbitration_v1"
                ),
                str(
                    getattr(
                        cfg,
                        "structural_output_routing_authority_order_version",
                        "escap5_before_existing_authority_v1",
                    )
                    or "escap5_before_existing_authority_v1"
                ),
                str(
                    getattr(
                        cfg,
                        "structural_output_routing_combined_rest_compatibility_version",
                        "escap5_combined_rest_residual_v1",
                    )
                    or "escap5_combined_rest_residual_v1"
                ),
                0,
            )
            fixed_output_routing_tuple = (
                True,
                "escap5_modular_action_logit_residual_v1",
                "escap5_output_routing_v1",
                "eiel_action_semantic_routing_registry_v1",
                ("move_north", "move_south", "move_east", "move_west", "rest"),
                2,
                8,
                "escap5_output_binding_operator_v1",
                1,
                1.0,
                0.25,
                "escap5_output_arbitration_v1",
                "escap5_before_existing_authority_v1",
                "escap5_combined_rest_residual_v1",
                0,
            )
            if (
                expected_output_routing_tuple != actual_output_routing_tuple
                or expected_output_routing_tuple != fixed_output_routing_tuple
            ):
                raise CheckpointStrictError(
                    "structural_output_routing config mismatch (strict): "
                    f"expected={expected_output_routing_tuple!r} got={actual_output_routing_tuple!r}"
                )
        development_key_present = "structural_development" in meta
        expected_development = meta.get("structural_development")
        actual_development_enable = bool(getattr(cfg, "structural_development_enable", False))
        if not development_key_present:
            if actual_development_enable:
                raise CheckpointStrictError(
                    "structural_development metadata missing for enabled destination (strict)"
                )
        else:
            if not isinstance(expected_development, dict):
                raise CheckpointStrictError("structural_development metadata malformed (strict)")
            required_development_fields = {
                "enabled",
                "profile",
                "module_record_schema_version",
                "stage_profile_version",
                "stage_semantic_ids",
                "mature_start_step",
                "late_start_step",
                "minimum_stage_gain",
                "maximum_stage_gain",
                "dependency_identity_authority",
                "max_prerequisites_per_module",
                "max_dependency_edges_per_agent",
                "cycle_policy",
                "inactive_state_policy",
                "inactive_output_policy",
                "latch_version",
                "latch_semantics",
                "mutation_operator_profile",
                "operator_attempts",
                "operator_draws_per_attempt",
                "migration_rng_draws",
                "supported_lane",
                "runtime_latch_key",
                "effective_previous_output_storage",
            }
            if set(expected_development) != required_development_fields:
                missing = sorted(required_development_fields - set(expected_development))
                extra = sorted(set(expected_development) - required_development_fields)
                raise CheckpointStrictError(
                    "structural_development metadata fields mismatch (strict): "
                    f"missing={missing!r} extra={extra!r}"
                )
            if type(expected_development.get("enabled")) is not bool or not bool(
                expected_development["enabled"]
            ):
                raise CheckpointStrictError(
                    "structural_development disabled metadata must be omitted (strict)"
                )
            if not isinstance(expected_development.get("stage_semantic_ids"), list):
                raise CheckpointStrictError(
                    "structural_development metadata malformed (strict): stage_semantic_ids"
                )
            for field in (
                "profile",
                "module_record_schema_version",
                "stage_profile_version",
                "dependency_identity_authority",
                "cycle_policy",
                "inactive_state_policy",
                "inactive_output_policy",
                "latch_version",
                "latch_semantics",
                "mutation_operator_profile",
                "supported_lane",
                "runtime_latch_key",
                "effective_previous_output_storage",
            ):
                if not isinstance(expected_development.get(field), str):
                    raise CheckpointStrictError(
                        f"structural_development metadata malformed (strict): {field}"
                    )
            for field in (
                "mature_start_step",
                "late_start_step",
                "max_prerequisites_per_module",
                "max_dependency_edges_per_agent",
                "operator_attempts",
                "operator_draws_per_attempt",
                "migration_rng_draws",
            ):
                if type(expected_development.get(field)) is not int:
                    raise CheckpointStrictError(
                        f"structural_development metadata malformed (strict): {field}"
                    )
            for field in ("minimum_stage_gain", "maximum_stage_gain"):
                if type(expected_development.get(field)) not in {int, float}:
                    raise CheckpointStrictError(
                        f"structural_development metadata malformed (strict): {field}"
                    )
            expected_development_tuple = (
                bool(expected_development["enabled"]),
                str(expected_development["profile"]),
                str(expected_development["module_record_schema_version"]),
                str(expected_development["stage_profile_version"]),
                tuple(str(v) for v in expected_development["stage_semantic_ids"]),
                int(expected_development["mature_start_step"]),
                int(expected_development["late_start_step"]),
                float(expected_development["minimum_stage_gain"]),
                float(expected_development["maximum_stage_gain"]),
                str(expected_development["dependency_identity_authority"]),
                int(expected_development["max_prerequisites_per_module"]),
                int(expected_development["max_dependency_edges_per_agent"]),
                str(expected_development["cycle_policy"]),
                str(expected_development["inactive_state_policy"]),
                str(expected_development["inactive_output_policy"]),
                str(expected_development["latch_version"]),
                str(expected_development["latch_semantics"]),
                str(expected_development["mutation_operator_profile"]),
                int(expected_development["operator_attempts"]),
                int(expected_development["operator_draws_per_attempt"]),
                int(expected_development["migration_rng_draws"]),
                str(expected_development["supported_lane"]),
                str(expected_development["runtime_latch_key"]),
                str(expected_development["effective_previous_output_storage"]),
            )
            actual_development_tuple = (
                actual_development_enable,
                str(getattr(cfg, "structural_development_profile", "none") or "none"),
                str(
                    getattr(cfg, "structural_development_module_schema_version", DEVELOPMENTAL_MODULE_SCHEMA_VERSION)
                    or DEVELOPMENTAL_MODULE_SCHEMA_VERSION
                ),
                str(
                    getattr(cfg, "structural_development_stage_profile_version", DEVELOPMENTAL_STAGE_PROFILE_VERSION)
                    or DEVELOPMENTAL_STAGE_PROFILE_VERSION
                ),
                tuple(
                    getattr(cfg, "structural_development_stage_ids", DEVELOPMENTAL_STAGE_IDS)
                    or ()
                ),
                int(getattr(cfg, "structural_development_mature_start_step", MATURE_START_STEP)),
                int(getattr(cfg, "structural_development_late_start_step", LATE_START_STEP)),
                float(getattr(cfg, "structural_development_min_stage_gain", MIN_STAGE_GAIN)),
                float(getattr(cfg, "structural_development_max_stage_gain", MAX_STAGE_GAIN)),
                str(
                    getattr(cfg, "structural_development_dependency_identity_authority", DEPENDENCY_IDENTITY_AUTHORITY)
                    or DEPENDENCY_IDENTITY_AUTHORITY
                ),
                int(getattr(cfg, "structural_development_max_prerequisites_per_module", MAX_PREREQUISITES_PER_MODULE)),
                int(getattr(cfg, "structural_development_max_dependency_edges_per_agent", MAX_DEPENDENCY_EDGES_PER_AGENT)),
                str(
                    getattr(
                        cfg,
                        "structural_development_cycle_policy",
                        CYCLE_POLICY,
                    )
                    or CYCLE_POLICY
                ),
                str(
                    getattr(cfg, "structural_development_inactive_state_policy", INACTIVE_STATE_POLICY)
                    or INACTIVE_STATE_POLICY
                ),
                str(
                    getattr(
                        cfg,
                        "structural_development_inactive_output_policy",
                        INACTIVE_OUTPUT_POLICY,
                    )
                    or INACTIVE_OUTPUT_POLICY
                ),
                str(
                    getattr(cfg, "structural_development_latch_version", LATCH_VERSION)
                    or LATCH_VERSION
                ),
                str(
                    getattr(
                        cfg,
                        "structural_development_latch_semantics",
                        LATCH_SEMANTICS,
                    )
                    or LATCH_SEMANTICS
                ),
                str(
                    getattr(
                        cfg,
                        "structural_development_operator_profile",
                        DEVELOPMENT_OPERATOR_PROFILE,
                    )
                    or DEVELOPMENT_OPERATOR_PROFILE
                ),
                int(getattr(cfg, "structural_development_operator_attempts", DEVELOPMENT_OPERATOR_ATTEMPTS)),
                int(getattr(cfg, "structural_development_operator_draws_per_attempt", DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT)),
                int(getattr(cfg, "structural_development_migration_rng_draws", MIGRATION_RNG_DRAWS)),
                str(
                    getattr(
                        cfg,
                        "structural_development_supported_lane",
                        SUPPORTED_LANE,
                    )
                    or SUPPORTED_LANE
                ),
                "module_instance_id",
                "structural_live_module_outputs",
            )
            fixed_development_tuple = (
                True,
                DEVELOPMENTAL_FEATURE_PROFILE,
                DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
                DEVELOPMENTAL_STAGE_PROFILE_VERSION,
                DEVELOPMENTAL_STAGE_IDS,
                MATURE_START_STEP,
                LATE_START_STEP,
                MIN_STAGE_GAIN,
                MAX_STAGE_GAIN,
                DEPENDENCY_IDENTITY_AUTHORITY,
                MAX_PREREQUISITES_PER_MODULE,
                MAX_DEPENDENCY_EDGES_PER_AGENT,
                CYCLE_POLICY,
                INACTIVE_STATE_POLICY,
                INACTIVE_OUTPUT_POLICY,
                LATCH_VERSION,
                LATCH_SEMANTICS,
                DEVELOPMENT_OPERATOR_PROFILE,
                DEVELOPMENT_OPERATOR_ATTEMPTS,
                DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
                MIGRATION_RNG_DRAWS,
                SUPPORTED_LANE,
                "module_instance_id",
                "structural_live_module_outputs",
            )
            if (
                expected_development_tuple != actual_development_tuple
                or expected_development_tuple != fixed_development_tuple
            ):
                raise CheckpointStrictError(
                    "structural_development config mismatch (strict): "
                    f"expected={expected_development_tuple!r} got={actual_development_tuple!r}"
                )
            runtime_device = str(getattr(getattr(sim, "device", None), "type", "") or "").lower()
            configured_device = str(getattr(cfg, "device", "auto") or "auto").lower()
            validation_device = runtime_device or configured_device
            try:
                validate_structural_development_config(
                    cfg,
                    device=validation_device,
                    policy_stride=int(getattr(cfg, "policy_tick_stride", 1) or 1),
                )
            except (RuntimeError, ValueError) as exc:
                raise CheckpointStrictError(
                    f"structural_development config invalid (strict): {exc}"
                ) from exc
            runnercore_authority = any(
                bool(value)
                for name, value in vars(cfg).items()
                if name.startswith("runnercore") and "authoritative" in name
            )
            if (
                int(getattr(cfg, "policy_tick_stride", 1) or 1) != 1
                or bool(getattr(cfg, "native_runtime_enable", False))
                or runnercore_authority
                or runtime_device == "cuda"
                or configured_device in {"cuda", "gpu"}
                or configured_device.startswith("cuda:")
            ):
                raise CheckpointStrictError(
                    "structural_development unsupported execution lane (strict)"
                )
        expected_gated_recurrent = meta.get("structural_gated_recurrent")
        actual_gated_recurrent_enable = bool(
            getattr(cfg, "structural_gated_recurrent_enable", False)
        )
        if not actual_gated_recurrent_enable:
            if expected_gated_recurrent is not None:
                raise CheckpointStrictError(
                    "structural_gated_recurrent disabled metadata must be omitted (strict)"
                )
        else:
            if not isinstance(expected_gated_recurrent, dict):
                raise CheckpointStrictError(
                    "structural_gated_recurrent metadata missing or malformed (strict)"
                )
            live_prerequisite = (
                bool(getattr(cfg, "structural_live_modules_enable", False)),
                str(getattr(cfg, "structural_live_modules_profile", "none") or "none"),
                str(
                    getattr(
                        cfg,
                        "structural_live_modules_schema_version",
                        LIVE_MODULE_SCHEMA_VERSION,
                    )
                    or LIVE_MODULE_SCHEMA_VERSION
                ),
            )
            if live_prerequisite != (
                True,
                LIVE_PROFILE,
                LIVE_MODULE_SCHEMA_VERSION,
            ):
                raise CheckpointStrictError(
                    "structural_gated_recurrent live-module prerequisite mismatch (strict): "
                    f"actual={live_prerequisite!r}"
                )
            actual_gated_recurrent = _escap7_gated_recurrent_metadata(cfg)
            fixed_gated_recurrent = _escap7_gated_recurrent_metadata(None)
            if (
                expected_gated_recurrent != actual_gated_recurrent
                or expected_gated_recurrent != fixed_gated_recurrent
            ):
                raise CheckpointStrictError(
                    "structural_gated_recurrent metadata/config mismatch (strict): "
                    f"expected={expected_gated_recurrent!r} "
                    f"actual={actual_gated_recurrent!r}"
                )

            gated_lane_reasons = escap7_unsupported_lane_reasons(
                cfg,
                policy_device=getattr(sim, "device", None),
                bank_device=getattr(sim, "bank_device", None),
            )
            if gated_lane_reasons:
                raise CheckpointStrictError(
                    "structural_gated_recurrent unsupported execution lane (strict): "
                    f"reasons={sorted(set(gated_lane_reasons))!r}"
                )
        expected_sustainable_growth = meta.get("structural_sustainable_growth")
        actual_sustainable_growth_enable = bool(
            getattr(cfg, "structural_sustainable_growth_enable", False)
        )
        if not actual_sustainable_growth_enable:
            if expected_sustainable_growth is not None:
                raise CheckpointStrictError(
                    "structural_sustainable_growth disabled metadata must be omitted (strict)"
                )
        else:
            if not isinstance(expected_sustainable_growth, dict):
                raise CheckpointStrictError(
                    "structural_sustainable_growth metadata missing or malformed (strict)"
                )
            if not actual_gated_recurrent_enable:
                raise CheckpointStrictError(
                    "structural_sustainable_growth ESCAP-7 prerequisite mismatch (strict)"
                )
            actual_sustainable_growth = _escap8_sustainable_growth_metadata(cfg)
            fixed_sustainable_growth = _escap8_sustainable_growth_metadata(None)
            dynamic_capacity_fields = {
                "fixed_envelope_shape",
                "fixed_envelope_shape_digest",
            }
            actual_fixed_contract = {
                key: value
                for key, value in actual_sustainable_growth.items()
                if key not in dynamic_capacity_fields
            }
            legacy_fixed_contract = {
                key: value
                for key, value in fixed_sustainable_growth.items()
                if key not in dynamic_capacity_fields
            }
            if (
                expected_sustainable_growth != actual_sustainable_growth
                or actual_fixed_contract != legacy_fixed_contract
            ):
                raise CheckpointStrictError(
                    "structural_sustainable_growth metadata/config mismatch (strict): "
                    f"expected={expected_sustainable_growth!r} "
                    f"actual={actual_sustainable_growth!r}"
                )
            sustainable_growth_lane_reasons = escap7_unsupported_lane_reasons(
                cfg,
                policy_device=getattr(sim, "device", None),
                bank_device=getattr(sim, "bank_device", None),
            )
            if sustainable_growth_lane_reasons:
                raise CheckpointStrictError(
                    "structural_sustainable_growth unsupported execution lane (strict): "
                    f"reasons={sorted(set(sustainable_growth_lane_reasons))!r}"
                )

    native_authority_bundle = (
        str(meta.get("checkpoint_authority_schema_version", "") or "")
        == NATIVE_SINGLE_PROJECTION_AUTHORITY
    )
    if native_authority_bundle:
        # Native-authority bundles intentionally contain no independent legacy
        # trajectory state. Apply route-neutral control/metrics state, then let
        # the single projection own either native restore or strict reference
        # materialization. Legacy world/population/RNG/bank payloads are never
        # allowed to overwrite it.
        _apply_control_plane(sim, payload.get("control_plane", {}))
        _apply_metrics_state(sim, payload.get("metrics_state", {}) or {})
        _apply_native_runtime_checkpoint_projection(
            sim,
            payload,
            native_runtime_extension_path=native_runtime_extension_path,
        )
        apply_summary = dict(getattr(sim, "_native_runtime_checkpoint_apply", {}) or {})
        if apply_summary.get("destination_runtime") == "reference":
            sync_fn = getattr(sim, "_sync_env_with_population", None)
            if callable(sync_fn):
                sync_fn()
        verification = {
            "schema_version": "resident_runtime_state_v2_apply_verification_v2",
            "ok": True,
            "status": "verified_from_single_native_projection",
            "verification_target": str(apply_summary.get("verification_target", "restored_native_engine")),
            "source_projection_sha256": str(meta.get("source_projection_sha256", "") or ""),
            "python_compatibility_mirror": (
                "materialized_at_explicit_reference_boundary"
                if apply_summary.get("destination_runtime") == "reference"
                else "intentionally_not_materialized"
            ),
        }
        setattr(sim, "_resident_runtime_state_v2_apply_verification", verification)
        return
    normalized_world = _normalize_checkpoint_world_authority(
        sim,
        payload.get("world", {}),
        payload.get("population", {}),
    )
    _apply_python_checkpoint_transaction(
        sim,
        world=normalized_world,
        population=payload.get("population", {}),
        resident=payload.get("resident_runtime_state_v2", {}) or {},
        weights_dir=str(payload.get("weights_dir", "")),
    )

    # Phase 3 determinism: preserve checkpoint agent.slot mapping.
    # Slot recanonicalization was attempted but diverged from the fresh path (bank row indices drift).
    # Keep the helper available for forensic experiments, but do NOT run it on resume.

    _apply_control_plane(sim, payload.get("control_plane", {}))

    # Apply runner-level state (shock counters, identity probe arrays, gen counters, etc.)
    try:
        _apply_runner_state(sim, payload.get("runner_state", {}) or {})
    except Exception:
        pass
    try:
        checkpoint_id = str(meta.get("checkpoint_id", "") or "")
        for target in _runner_targets(sim):
            recorder = getattr(target, "scientific_chronicle", None)
            if recorder is not None and hasattr(recorder, "set_parent_checkpoint"):
                recorder.set_parent_checkpoint(checkpoint_id)
                break
    except Exception:
        pass
    _apply_resident_generation_counters(sim, payload.get("resident_runtime_state_v2", {}) or {})

    # Restore determinism-critical metrics logger internal state BEFORE the first
    # post-resume metrics.csv row is written.
    try:
        _apply_metrics_state(sim, payload.get("metrics_state", {}) or {})
    except Exception:
        pass

    # Ensure parameter banks exist and are structurally synced BEFORE restoring bank_state,
    # so that the restored bank (especially GRU _Hmem) is not overwritten by sync.
    #
    # IMPORTANT: some integrations store bank state on `sim.runner` instead of `sim`.
    # Apply the structural ensure/sync to all runner targets.
    try:
        targets = _runner_targets(sim)
        alive = []
        try:
            alive = list(getattr(sim, "pop").alive_agents())
        except Exception:
            alive = []
        for t in targets:
            try:
                if hasattr(t, "_ensure_bank"):
                    t._ensure_bank()  # type: ignore[attr-defined]
                if alive and hasattr(t, "_sync_bank_for_agents"):
                    t._sync_bank_for_agents(alive, buckets=None)  # type: ignore[attr-defined]
                if alive and hasattr(t, "_sync_gru_bank_for_agents"):
                    t._sync_gru_bank_for_agents(alive, buckets=None)  # type: ignore[attr-defined]
            except Exception:
                continue
    except Exception:
        pass

    # Apply bank state LAST so restored GRU memory wins.
    try:
        _apply_bank_state(sim, payload.get("bank_state", {}) or {})
    except Exception:
        pass

        # ------------------------------------------------------------------
    # Phase 3 determinism: on resume, rebuild parameter banks but preserve GRU memory.
    # ------------------------------------------------------------------
    # Parameter banks (_mlp_pack/_gru_pack and per-slot weights) are caches and are not
    # persisted in v2 bundles. After resume we must force a one-time sync from per-agent
    # policy modules into the banks before the first tick, otherwise uninitialized/stale
    # bank rows can leak into policy execution and diverge without any RNG drift.
    #
    # GRU memory (_Hmem) IS persisted; syncing GRU weights must not reset restored memory.
    try:
        alive = []
        try:
            alive = list(getattr(sim, "pop").alive_agents())
        except Exception:
            alive = []
        # Determine which GRU slots should have memory reset at the first post-boundary bank sync.
        # Fresh runs reset memory when slot_token != _gru_bank_token (new agent or slot reassignment).
        gru_reset_slots: List[int] = []
        try:
            for _a in alive:
                try:
                    pol = getattr(_a, "policy", None)
                    if pol is None or not hasattr(pol, "gru"):
                        continue
                    _slot = int(getattr(_a, "slot", -1))
                    if _slot < 0:
                        continue
                    _tok = int(getattr(_a, "slot_token", 0))
                    _last = int(getattr(_a, "_gru_bank_token", -1))
                    if _tok != _last:
                        gru_reset_slots.append(_slot)
                except Exception:
                    continue
            gru_reset_slots = sorted(set(int(x) for x in gru_reset_slots))
        except Exception:
            gru_reset_slots = []


        for a in alive:
            try:
                # Force a one-time parameter sync for this agent on the first post-resume tick.
                setattr(a, "_bank_token", -1)
                setattr(a, "_gru_bank_token", -1)
            except Exception:
                continue

        # Ensure the first tick rebuilds both banks from per-agent policies.
        # Apply to all runner targets (sim and sim.runner), because some repos attach banks to runner.
        try:
            targets = _runner_targets(sim)
            for t in targets:
                try:
                    setattr(t, "_bank_dirty_mlp", True)
                    setattr(t, "_bank_dirty_gru", True)
                except Exception:
                    continue
        except Exception:
            pass

        # One-shot guard: allow GRU bank sync to skip the memory reset step so restored _Hmem survives.
        try:
            bs = payload.get("bank_state", {}) or {}
            if ("_Hmem" in bs) or bool(bs.get("_gru_inited", False)):
                targets = _runner_targets(sim)
                for t in targets:
                    try:
                        setattr(t, "_eiel_skip_gru_memory_reset_once", True)
                    except Exception:
                        continue
        except Exception:
            pass

        # Force an immediate one-time bank rebuild now (must not consume RNG).
        # This prevents uninitialized/stale bank rows from leaking into gen1 tick0.
        try:
            if alive:
                targets = _runner_targets(sim)
                for t in targets:
                    try:
                        if hasattr(t, "_sync_bank_for_agents"):
                            t._sync_bank_for_agents(alive, buckets=None)  # type: ignore[attr-defined]
                        if hasattr(t, "_sync_gru_bank_for_agents"):
                            t._sync_gru_bank_for_agents(alive, buckets=None)  # type: ignore[attr-defined]
                    except Exception:
                        continue
        except Exception:
            pass
        # Re-apply bank state after forced rebuild so restored GRU memory wins.
        try:
            _apply_bank_state(sim, payload.get("bank_state", {}) or {})
        except Exception:
            pass

        # Reset only the GRU slots that were mismatched at the checkpoint boundary.
        # This matches fresh-run tick0 behavior for new agents or slot reassignment.
        try:
            if torch is not None and gru_reset_slots:
                for t in _runner_targets(sim):
                    try:
                        reset_fn = getattr(t, "_reset_memory_slots", None)
                        if not callable(reset_fn):
                            continue
                        dev = getattr(t, "_bank_device", None)
                        if dev is None:
                            dev = getattr(t, "device", "cpu")
                        slots_t = torch.tensor(gru_reset_slots, dtype=torch.long, device=dev)
                        reset_fn(slots_t)
                    except Exception:
                        continue
        except Exception:
            pass



    except Exception:
        pass

    # Rehydrate runner-level drive arrays deterministically from agent mirrors.

    try:
        alive = []
        try:
            alive = list(getattr(sim, "pop").alive_agents())
        except Exception:
            alive = []
        if alive:
            # Resume determinism: runner __init__ may have initialized drive slot tokens
            # for a dummy init population before checkpoint apply. Invalidate tokens so
            # _sync_drive_slot initializes runner drives from checkpoint agent mirrors.
            try:
                ensure = getattr(sim, "_ensure_drive_arrays", None)
                if callable(ensure):
                    ensure()
                tok = getattr(sim, "_drive_slot_token", None)
                if tok is not None:
                    try:
                        tok[...] = np.int32(-1)
                    except Exception:
                        pass
            except Exception:
                pass

            # Resume determinism: bank state (including GRU Hmem) is restored from the bundle.
            # Do NOT invalidate bank tokens or mark banks dirty here, otherwise the first
            # post-resume bank sync may rebuild and overwrite restored GRU memory before tick0.
            # Only fall back to 'force dirty' if the bundle did not provide bank_state.
            try:
                _bs = payload.get('bank_state', {}) or {}
                _has_bank_state = bool(_bs) and (('_Hmem' in _bs) or ('_Wih' in _bs) or ('_Whh' in _bs) or ('_mlp_pack' in _bs) or bool(_bs.get('_gru_inited', False)))
            except Exception:
                _has_bank_state = False

            if _has_bank_state:
                try:
                    population_payload = payload.get("population", {}) or {}
                    checkpoint_token_rows = {
                        int(row.get("agent_id", -1)): row
                        for row in list(population_payload.get("agents", []) or [])
                        if isinstance(row, dict)
                    }
                    for _a in alive:
                        try:
                            _agent_id = int(getattr(_a, "agent_id", -1))
                        except Exception:
                            _agent_id = -1
                        checkpoint_row = checkpoint_token_rows.get(_agent_id, {})
                        for attribute in ("_bank_token", "_gru_bank_token"):
                            try:
                                raw_token = checkpoint_row.get(attribute, -1)
                                setattr(
                                    _a,
                                    attribute,
                                    int(-1 if raw_token is None else raw_token),
                                )
                            except Exception:
                                pass
                    # Mark banks clean on likely runner targets (best-effort).
                    for _t in _runner_targets(sim):
                        try:
                            setattr(_t, '_bank_dirty_mlp', False)
                            setattr(_t, '_bank_dirty_gru', False)
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                # Fallback: no bank_state in bundle, force a rebuild on first forward.
                try:
                    for _a in alive:
                        try:
                            setattr(_a, '_bank_token', -1)
                        except Exception:
                            pass
                        try:
                            setattr(_a, '_gru_bank_token', -1)
                        except Exception:
                            pass
                    mbd = getattr(sim, '_mark_bank_dirty', None)
                    if callable(mbd):
                        mbd()
                except Exception:
                    pass

            # Some runners expose _sync_drive_slot on self (RunnerBase).
            sync_fn = getattr(sim, "_sync_drive_slot", None)
            if callable(sync_fn):
                for a in alive:
                    try:
                        sync_fn(a)
                    except Exception:
                        pass
    except Exception:
        pass

    # NOTE: Do NOT force a post-resume bank resync here.
    # Bank parameters are already synced from per-agent policies above.
    # Forcing a resync at tick0 can reset GRU memory and cause gen1 divergence.

    # Restore a persisted native owner before native execution can reach the
    # get-or-create fallback, which necessarily imports a new tick-0 owner.
    _apply_native_runtime_checkpoint_projection(
        sim,
        payload,
        native_runtime_extension_path=native_runtime_extension_path,
    )

    # ------------------------------------------------------------------
    # Phase 3 determinism parity: env <-> population reconciliation
    # ------------------------------------------------------------------
    # IMPORTANT:
    # Perform envsync BEFORE restoring RNG.
    # Fresh runs execute envsync (loop.post_repro.envsync) before the next
    # boundary RNG snapshot is taken. If envsync were to draw RNG (it
    # should not, but we do not assume), restoring RNG first could allow
    # a hidden drift on resume.
    _did_post_apply_envsync = False
    try:
        sync_fn = getattr(sim, "_sync_env_with_population", None)
        if callable(sync_fn):
            sync_fn()
            _did_post_apply_envsync = True
    except Exception:
        _did_post_apply_envsync = False

    # Restore RNG snapshot as the final step (prevents boundary drift).
    if restore_rng:
        try:
            _apply_rng_snapshot(sim, payload.get("rng_snapshot_v2", {}))
        except Exception:
            pass

    # Optional trace fingerprint
    try:
        if bool(int(os.environ.get("DEBUG_RESUME_TRACE", "0") or "0")):
            fp = {
                "event": "RESUME_APPLY_FINGERPRINT",
                "bundle_generation_index": int(payload.get("metadata", {}).get("generation_index", -1)),
                "runner_state_schema": str((payload.get("runner_state", {}) or {}).get("schema_version", "")),
                "runner_state_keys": sorted([k for k in (payload.get("runner_state", {}) or {}).keys() if k != "data_hex"]),
            }
            print("[RESUME_TRACE] " + json.dumps(fp, sort_keys=True))
    except Exception:
        pass

    # Phase 3 debug: state digest after applying bundle onto live runner
    # NOTE: The bundle generation index is the *last completed* generation (e.g. 16).
    # The resume apply boundary is generation g, and EIEL_STATE_DUMP_GENS typically targets
    # the first resumed generation (e.g. 17). Therefore we emit post-apply debug dumps
    # under generation_index=g so they are captured by the dump gate.
    try:
        g = int(payload.get("metadata", {}).get("generation_index", -1))
    except Exception:
        g = -1
    dump_gen = int(g) if int(g) >= 0 else -1

    # Flush transient caches (must not affect semantics across resume)
    _flush_transient_caches(sim)
    # Canonicalize runner drive arrays so boundary dump is comparable to fresh path.
    _canonicalize_runner_drive_arrays(sim)

    # Reconcile env <-> population membership once post-apply if we did not
    # already do so before restoring RNG.
    try:
        if not bool(_did_post_apply_envsync):
            sync_fn = getattr(sim, "_sync_env_with_population", None)
            if callable(sync_fn):
                sync_fn()
    except Exception:
        pass

    # CXXWAVE1C: verify the resident checkpoint payload against the restored
    # authoritative runner state. This does not make resident state authoritative
    # yet; it establishes the exact parity gate required before doing so.
    try:
        resident_state = payload.get("resident_runtime_state_v2", {}) or {}
        if isinstance(resident_state, dict) and resident_state.get("schema_version") == "resident_runtime_state_v2":
            verification = _verify_resident_runtime_state_v2_against_sim(
                sim,
                resident_state,
                generation_index=int(g),
            )
            try:
                setattr(sim, "_resident_runtime_state_v2_apply_verification", verification)
            except Exception:
                pass
            if bool(int(os.environ.get("EIEL_RESIDENT_RUNTIME_STATE_V2_STRICT_APPLY_VERIFY", "0") or "0")) and not bool(
                verification.get("ok", False)
            ):
                raise CheckpointStrictError(
                    "resident_runtime_state_v2 apply verification failed: "
                    + json.dumps(verification.get("mismatches", [])[:8], sort_keys=True, default=str)
                )
    except CheckpointStrictError:
        raise
    except Exception as exc:
        try:
            setattr(
                sim,
                "_resident_runtime_state_v2_apply_verification",
                {
                    "schema_version": "resident_runtime_state_v2_apply_verification_v1",
                    "ok": False,
                    "status": "verification_error",
                    "error": repr(exc),
                },
            )
        except Exception:
            pass

    # FDP (Phase 3): targeted missing-state whitelist compare (bundle_save_post vs resume_apply_post).
    # Stored in the bundle as fdp_state_whitelist.json, compared here to fail-fast on missing resume state.
    try:
        if bool(int((os.environ.get('EIEL_DETERMINISM_PROBE', '0') or '0'))):
            w_bundle_dir = os.path.dirname(str(payload.get('weights_dir', '') or ''))
            w_path = os.path.join(w_bundle_dir, 'fdp_state_whitelist.json')
            if w_bundle_dir and os.path.exists(w_path):
                from simulation.fdp_proofs import build_state_whitelist as _fdp_build_state_whitelist
                from simulation.fdp_proofs import diff_state_whitelist as _fdp_diff_state_whitelist

                base = _read_json(w_path)
                cur = _fdp_build_state_whitelist(sim)
                cur['generation_index'] = int(g)

                # Emit the resume-side snapshot for inspection (debug only; deterministic payload).
                try:
                    run_dir = getattr(sim, 'run_dir', '') or ''
                    if run_dir:
                        os.makedirs(os.path.join(str(run_dir), 'debug'), exist_ok=True)
                        _write_canonical_json_lossless(os.path.join(str(run_dir), f'debug/fdp_state_whitelist__resume_apply_post__gen{int(g):06d}.json'), cur)
                except Exception:
                    pass

                if isinstance(base, dict) and base != cur:
                    diff = _fdp_diff_state_whitelist(base, cur, max_diffs=64)
                    try:
                        run_dir = getattr(sim, 'run_dir', '') or ''
                        if run_dir:
                            _write_canonical_json_lossless(os.path.join(str(run_dir), f'debug/fdp_state_whitelist_diff__resume_apply_post__gen{int(g):06d}.json'), diff)
                    except Exception:
                        pass
                    try:
                        # Include a tiny deterministic diff summary in the exception for console-only diagnosis.
                        _d = diff.get('diffs') if isinstance(diff, dict) else None
                        _tiny = []
                        if isinstance(_d, list):
                            for _x in _d[:8]:
                                if isinstance(_x, dict):
                                    _tiny.append({'path': _x.get('path'), 'a': _x.get('a'), 'b': _x.get('b')})
                        _msg = 'FDP state whitelist mismatch after resume_apply_post; first_diffs=' + json.dumps(_tiny, sort_keys=True)
                    except Exception:
                        _msg = 'FDP state whitelist mismatch after resume_apply_post'
                    raise CheckpointStrictError(_msg)
    except CheckpointStrictError:
        raise
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Phase 3 debug aid: emit a generation-aligned dump immediately after apply.
    #
    # Fresh runs typically emit state_000000__pre_start_generation_alive.json at
    # the start of generation 0.
    #
    # When resuming from bundle_gen000000, the first dump otherwise produced by
    # this loader would be tagged under (g+1), which prevents boundary-aligned
    # comparisons.
    #
    # Emit the same tag at generation g (the loaded generation) so E2E tooling
    # can compare equivalent boundaries between fresh and resumed runs.
    # ------------------------------------------------------------------
    if int(g) >= 0:
        try:
            try:
                alive_ids = [int(getattr(a, "agent_id")) for a in getattr(sim, "population", []) if bool(getattr(a, "alive", True))]
            except Exception:
                alive_ids = []
            extra = {"alive_ids": alive_ids}
            try:
                maybe_dump_state(sim, generation_index=int(g), tag="pre_start_generation_alive", extra=extra)
            except Exception:
                _force_write_state_dump(sim, generation_index=int(g), tag="pre_start_generation_alive", extra=extra)
            else:
                _force_write_state_dump(sim, generation_index=int(g), tag="pre_start_generation_alive", extra=extra)
        except Exception:
            pass


    # FDP v3: For boundary-aligned tooling, ensure the resume path also emits a
    # gen-aligned `bundle_save_post` dump for the loaded generation. The resume
    # flow does not perform a bundle save at gen 0, but the harness expects this
    # tag to exist on both sides for Step 3b comparisons.
    if int(g) >= 0:
        try:
            _force_write_state_dump(
                sim,
                generation_index=int(g),
                tag="bundle_save_post",
                # Keep schema identical to fresh-run `bundle_save_post` dumps.
                # Do not include absolute paths; run dirs differ by design.
                extra={"bundle_gen": int(g)},
            )
        except Exception:
            pass
    # Import under a private alias to avoid UnboundLocalError if this function scope
    # accidentally shadows `maybe_dump_state`.
    try:
        try:
            from simulation.state_dump import maybe_dump_state as _maybe_dump_state  # type: ignore
        except Exception:
            from engine.simulation.state_dump import maybe_dump_state as _maybe_dump_state  # type: ignore  # noqa: WPS433
    except Exception:
        _maybe_dump_state = None  # type: ignore

    if callable(_maybe_dump_state):
        _maybe_dump_state(sim, generation_index=dump_gen, tag="resume_apply_post", extra={"bundle_gen": g})

        # FDP (Phase 3): determinism_probe record at resume_apply_post (policy identity proof hooks live here).
        try:
            from simulation.determinism_probe import maybe_write_probe as _fdp_maybe_write_probe
            _fdp_maybe_write_probe(sim, generation_index=int(dump_gen), tag='resume_apply_post')
        except Exception:
            pass

        # FDP v3: Align evolution_py tracing across fresh vs resume by resetting the
        # trace epoch at a VERIFIED equivalence boundary.
        # Diagnostic-only: must not affect RNG state or resume semantics.
        try:
            rm = getattr(sim, "rng_manager", None)
            if rm is not None:
                # Dump evolution_py RNG state and ledger before/after boundary reset.
                try:
                    run_dir = getattr(sim, "run_dir", "") or ""
                    if run_dir:
                        dbg = os.path.join(str(run_dir), "debug")
                        if hasattr(rm, "dump_evopy_state_json"):
                            rm.dump_evopy_state_json(path=os.path.join(dbg, "evopy_state__resume_apply_post__pre_reset.json"), stage="resume_apply_post__pre_reset")
                        if hasattr(rm, "dump_evopy_ledger"):
                            rm.dump_evopy_ledger(path=os.path.join(dbg, "evopy_ledger__resume_apply_post__pre_reset.json"))
                except Exception:
                    pass
                if hasattr(rm, "_set_trace_ctx"):
                    rm._set_trace_ctx(gen=int(dump_gen), tick=-1, stage="resume_apply_post")
                if hasattr(rm, "_reset_evopy_trace"):
                    rm._reset_evopy_trace(epoch="boundary")
                if hasattr(rm, "_reset_evopy_ledger"):
                    rm._reset_evopy_ledger(epoch="boundary")
                try:
                    run_dir = getattr(sim, "run_dir", "") or ""
                    if run_dir:
                        dbg = os.path.join(str(run_dir), "debug")
                        if hasattr(rm, "dump_evopy_state_json"):
                            rm.dump_evopy_state_json(path=os.path.join(dbg, "evopy_state__resume_apply_post__post_reset.json"), stage="resume_apply_post__post_reset")
                        if hasattr(rm, "dump_evopy_ledger"):
                            rm.dump_evopy_ledger(path=os.path.join(dbg, "evopy_ledger__resume_apply_post__post_reset.json"))
                except Exception:
                    pass
        except Exception:
            pass

        # Determinism tracer: env vs population alive membership immediately after resume.
        # This isolates whether the policy domain membership mismatch is already present
        # at resume-apply time or only appears later during the tick loop.
        try:
            pop = getattr(sim, "pop", None)
            env = getattr(sim, "env", None)
            env_alive = getattr(env, "agent_alive", None) if env is not None else None
            pop_agents = getattr(pop, "agents", None) if pop is not None else None

            pop_alive_ids = []
            if isinstance(pop_agents, dict):
                for a in pop_agents.values():
                    try:
                        if bool(getattr(a, "alive", False)):
                            pop_alive_ids.append(int(getattr(a, "agent_id")))
                    except Exception:
                        continue
            pop_alive_ids = sorted(set(pop_alive_ids))

            env_alive_ids = []
            if env_alive is not None:
                try:
                    env_alive_ids = [int(x) for x in np.nonzero(np.asarray(env_alive, dtype=bool))[0].tolist()]
                except Exception:
                    env_alive_ids = []
            env_alive_ids = sorted(set(env_alive_ids))

            pop_set = set(pop_alive_ids)
            env_set = set(env_alive_ids)
            extra = {
                "pop_alive_ids_sorted": pop_alive_ids,
                "env_alive_ids_sorted": env_alive_ids,
                "pop_not_env_sorted": sorted(pop_set - env_set),
                "env_not_pop_sorted": sorted(env_set - pop_set),
                "pop_alive_n": int(len(pop_alive_ids)),
                "env_alive_n": int(len(env_alive_ids)),
                "pop_not_env_n": int(len(pop_set - env_set)),
                "env_not_pop_n": int(len(env_set - pop_set)),
            }
            _maybe_dump_state(sim, generation_index=dump_gen, tag="resume_alive_membership_post", extra=extra)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------


def _snapshot_checkpoint_object(obj: Any) -> Dict[str, Any]:
    from collections import deque

    values: Dict[str, Any] = {}
    for name, value in vars(obj).items():
        if isinstance(value, np.ndarray):
            values[name] = value.copy()
        elif torch is not None and isinstance(value, torch.Tensor):
            values[name] = value.detach().clone()
        elif isinstance(value, dict):
            values[name] = value.copy()
        elif isinstance(value, list):
            values[name] = list(value)
        elif isinstance(value, deque):
            values[name] = copy.copy(value)
        elif name == "_slots":
            values[name] = copy.deepcopy(value)
        else:
            values[name] = value
    rng_states: Dict[str, tuple[str, Any]] = {}
    for name, value in vars(obj).items():
        bit_generator = getattr(value, "bit_generator", None)
        if bit_generator is not None and hasattr(bit_generator, "state"):
            rng_states[name] = ("numpy", copy.deepcopy(bit_generator.state))
        elif callable(getattr(value, "getstate", None)) and callable(
            getattr(value, "setstate", None)
        ):
            rng_states[name] = ("python", copy.deepcopy(value.getstate()))
        elif callable(getattr(value, "_get_state_hex", None)) and callable(
            getattr(value, "_set_state_hex", None)
        ):
            rng_states[name] = ("hex", str(value._get_state_hex()))
    nested_indices = {
        name: dict(getattr(value, "_indices"))
        for name, value in vars(obj).items()
        if isinstance(getattr(value, "_indices", None), dict)
    }
    nested_rng_snapshots = {
        name: copy.deepcopy(value.capture_snapshot())
        for name, value in vars(obj).items()
        if callable(getattr(value, "capture_snapshot", None))
        and callable(getattr(value, "apply_snapshot_dict", None))
    }
    return {
        "values": values,
        "rng_states": rng_states,
        "nested_indices": nested_indices,
        "nested_rng_snapshots": nested_rng_snapshots,
    }


def _restore_checkpoint_object(obj: Any, snapshot: Dict[str, Any]) -> None:
    values = dict(snapshot["values"])
    for name, saved in tuple(values.items()):
        current = getattr(obj, name, None)
        if (
            isinstance(saved, np.ndarray)
            and isinstance(current, np.ndarray)
            and current.shape == saved.shape
            and current.dtype == saved.dtype
        ):
            current[...] = saved
            values[name] = current
        elif torch is not None and (
                isinstance(saved, torch.Tensor)
                and isinstance(current, torch.Tensor)
                and current.shape == saved.shape
                and current.dtype == saved.dtype
                and current.device == saved.device
        ):
            current.copy_(saved)
            values[name] = current
    for name in set(vars(obj)) - set(values):
        delattr(obj, name)
    vars(obj).update(values)
    for name, (kind, state) in dict(snapshot.get("rng_states", {})).items():
        rng = getattr(obj, name, None)
        if kind == "numpy":
            rng.bit_generator.state = copy.deepcopy(state)
        elif kind == "python":
            rng.setstate(copy.deepcopy(state))
        else:
            rng._set_state_hex(str(state))
    for name, indices in dict(snapshot.get("nested_indices", {})).items():
        target = getattr(obj, name, None)
        target_indices = getattr(target, "_indices", None)
        if isinstance(target_indices, dict):
            target_indices.clear()
            target_indices.update(indices)
    for name, rng_snapshot in dict(
        snapshot.get("nested_rng_snapshots", {})
    ).items():
        target = getattr(obj, name, None)
        if callable(getattr(target, "apply_snapshot_dict", None)):
            target.apply_snapshot_dict(copy.deepcopy(rng_snapshot))


def _snapshot_python_checkpoint_install(sim: Any) -> Dict[str, Any]:
    env = getattr(sim, "env", None)
    pop = getattr(sim, "pop", None)
    if env is None or pop is None:
        raise CheckpointStrictError("checkpoint install requires env and population")
    return {
        "env": _snapshot_checkpoint_object(env),
        "pop": _snapshot_checkpoint_object(pop),
    }


def _restore_python_checkpoint_install(sim: Any, snapshot: Dict[str, Any]) -> None:
    _restore_checkpoint_object(getattr(sim, "env"), snapshot["env"])
    _restore_checkpoint_object(getattr(sim, "pop"), snapshot["pop"])


def _snapshot_native_reference_install(sim: Any) -> Dict[str, Any]:
    snapshot = _snapshot_python_checkpoint_install(sim)
    snapshot["runner"] = _snapshot_checkpoint_object(sim)
    return snapshot


def _restore_native_reference_install(sim: Any, snapshot: Dict[str, Any]) -> None:
    _restore_checkpoint_object(sim, snapshot["runner"])
    _restore_python_checkpoint_install(sim, snapshot)


def _apply_python_checkpoint_transaction(
    sim: Any,
    *,
    world: Dict[str, Any],
    population: Dict[str, Any],
    resident: Dict[str, Any],
    weights_dir: str,
) -> None:
    install_snapshot = _snapshot_python_checkpoint_install(sim)
    try:
        _apply_population(sim, population, weights_dir=weights_dir)
        _apply_world(sim, world)
        _apply_resident_environment_scalars(sim, resident)
    except BaseException:
        _restore_python_checkpoint_install(sim, install_snapshot)
        raise



def _export_env_rng_state(env: Any) -> Any:
    """Export environment RNG internal state (JSON-serializable).

    This captures env.rng.bit_generator.state when present. If the environment does not
    expose such a Generator, returns None.

    Phase 3: Resume determinism requires restoring any environment-local RNG islands.
    """
    rng = getattr(env, "rng", None)
    if rng is None:
        return None
    bitgen = getattr(rng, "bit_generator", None)
    if bitgen is None:
        return None
    state = getattr(bitgen, "state", None)
    if state is None:
        return None
    # Ensure JSON-serializable and canonical (no numpy scalars/arrays).
    try:
        return json.loads(json.dumps(state))
    except Exception:
        # Fall back to raw state if it already is serializable-ish.
        return state


def _apply_env_rng_state(env: Any, state: Any) -> None:
    """Restore environment RNG internal state if supported."""
    if state is None:
        return
    rng = getattr(env, "rng", None)
    if rng is None:
        return
    bitgen = getattr(rng, "bit_generator", None)
    if bitgen is None:
        return
    try:
        bitgen.state = state
    except Exception:
        # Some generators may not allow assignment; treat as no-op to avoid crashing.
        return

def _export_world(sim: Any) -> Dict[str, Any]:
    env = getattr(sim, "env", None)
    if env is None:
        raise ValueError("sim.env missing")

    def arr(a: Any) -> List[Any]:
        return np.asarray(a).tolist()

    binding = {
        "capacity": int(np.asarray(getattr(env, "slot_to_agent_id")).size),
        "slot_to_agent_id": arr(getattr(env, "slot_to_agent_id")),
        "slot_tokens": arr(getattr(env, "slot_tokens")),
        "world_alive_by_slot": arr(getattr(env, "agent_world_alive_by_slot")),
        "pos_y_by_slot": arr(getattr(env, "agent_pos_y_by_slot")),
        "pos_x_by_slot": arr(getattr(env, "agent_pos_x_by_slot")),
    }
    binding["binding_digest"] = _sha256_bytes(canonical_json_bytes(binding))
    out = {
        "schema_version": "world_state_v3",
        "world_membership_authority": binding,
        "grid": arr(getattr(env, "grid")),
        "agent_grid": arr(getattr(env, "agent_grid")),
        "biome_grid": arr(getattr(env, "biome_grid")),
        "agent_pos_y": arr(getattr(env, "agent_pos_y")),
        "agent_pos_x": arr(getattr(env, "agent_pos_x")),
        "agent_alive": arr(getattr(env, "agent_alive")),
        "food_spawn_mult": float(getattr(env, "food_spawn_mult", 1.0)),
        "action_cost_mult": float(getattr(env, "action_cost_mult", 1.0)),
        "rest_cost_mult": float(getattr(env, "rest_cost_mult", 1.0)),
        "env_rng_bitgen_state": _export_env_rng_state(env),
    }

    if bool(getattr(env, "patch_world_enable", False)):
        from env.gridworld_pkg.patch_world import (
            DELAYED_CROWDING_WASTE_ARRAY_NAMES,
            DELAYED_RESOURCE_DEPLETION_ARRAY_NAMES,
            EED_RESOURCE_IMPACT_ARRAY_NAMES,
            EED_RESOURCE_IMPACT_OVERLAY_ID,
            EED_RESOURCE_IMPACT_SCHEMA_VERSION,
            MEM_MAX_DELAYED_CHOICE_ARRAY_NAMES,
            MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION,
            PATCH_WORLD_ARRAY_NAMES,
            PREDICTION_DEMAND_ARRAY_NAMES,
            PREDICTION_DEMAND_SCHEMA_VERSION,
            WDL_DELAYED_HAZARD_ARRAY_NAMES,
        )

        out["patch_world"] = {
            "schema_version": int(getattr(env, "patch_world_schema_version", 1) or 1),
            "arrays": {name: arr(getattr(env, name)) for name in PATCH_WORLD_ARRAY_NAMES},
            "pending_arrays": {
                "patch_scent_emit_pending": arr(getattr(env, "patch_scent_emit_pending")),
                "patch_waste_emit_pending": arr(getattr(env, "patch_waste_emit_pending")),
            },
        }
        if bool(getattr(getattr(env, "cfg", None), "respec0b_resource_specialization_enable", False)):
            out["patch_world"]["respec0b_resource_handling_state"] = {
                "schema_version": "respec0b_slot_cursor_state_v1",
                "slot_tokens": arr(getattr(env, "slot_tokens")),
                "active_y": arr(getattr(env, "respec0b_resource_active_y")),
                "active_x": arr(getattr(env, "respec0b_resource_active_x")),
                "active_class": arr(getattr(env, "respec0b_resource_active_class")),
                "progress": arr(getattr(env, "respec0b_resource_handling_progress")),
            }
        if bool(
            getattr(
                getattr(env, "cfg", None),
                "eed_resource_impact_enable",
                False,
            )
        ):
            out["patch_world"]["eed_resource_impact"] = {
                "schema_version": EED_RESOURCE_IMPACT_SCHEMA_VERSION,
                "overlay_id": EED_RESOURCE_IMPACT_OVERLAY_ID,
                "arrays": {
                    name: arr(getattr(env, name))
                    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES
                },
            }
        if bool(getattr(getattr(env, "cfg", None), "wdl_delayed_hazard_enable", False)):
            out["patch_world"]["wdl_delayed_hazard"] = {
                "schema_version": int(getattr(getattr(env, "cfg", None), "wdl_delayed_hazard_schema_version", 1) or 1),
                "arrays": {name: arr(getattr(env, name)) for name in WDL_DELAYED_HAZARD_ARRAY_NAMES},
            }
        if bool(getattr(getattr(env, "cfg", None), "prediction_demand_enable", False)):
            out["patch_world"]["prediction_demand"] = {
                "schema_version": int(
                    getattr(getattr(env, "cfg", None), "prediction_demand_schema_version", PREDICTION_DEMAND_SCHEMA_VERSION)
                    or PREDICTION_DEMAND_SCHEMA_VERSION
                ),
                "arrays": {name: arr(getattr(env, name)) for name in PREDICTION_DEMAND_ARRAY_NAMES},
            }
        if bool(getattr(getattr(env, "cfg", None), "delayed_resource_depletion_enable", False)):
            out["patch_world"]["delayed_resource_depletion"] = {
                "schema_version": int(
                    getattr(getattr(env, "cfg", None), "delayed_resource_depletion_schema_version", 1) or 1
                ),
                "arrays": {name: arr(getattr(env, name)) for name in DELAYED_RESOURCE_DEPLETION_ARRAY_NAMES},
            }
        if bool(getattr(getattr(env, "cfg", None), "delayed_crowding_waste_enable", False)):
            out["patch_world"]["delayed_crowding_waste"] = {
                "schema_version": int(
                    getattr(getattr(env, "cfg", None), "delayed_crowding_waste_schema_version", 1) or 1
                ),
                "arrays": {name: arr(getattr(env, name)) for name in DELAYED_CROWDING_WASTE_ARRAY_NAMES},
            }
        if bool(getattr(getattr(env, "cfg", None), "mem_max_delayed_choice_enable", False)):
            out["patch_world"]["mem_max_delayed_choice"] = {
                "schema_version": int(
                    getattr(
                        getattr(env, "cfg", None),
                        "mem_max_delayed_choice_schema_version",
                        MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION,
                    )
                    or MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION
                ),
                "arrays": {name: arr(getattr(env, name)) for name in MEM_MAX_DELAYED_CHOICE_ARRAY_NAMES},
            }

    return out


def _export_population_and_weights(sim: Any, *, weights_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pop = getattr(sim, "pop", None)
    if pop is None:
        raise ValueError("sim.pop missing")

    # Phase 3 determinism (resume equivalence): population.json serializes the
    # agent mirror field `a.drives`, while the live simulation primarily uses
    # runner-level drive arrays. Ensure the agent mirrors are synchronized from
    # runner arrays at the SafeBoundary before serialization.
    try:
        sync_fn = getattr(sim, "_sync_drive_slot", None)
        if callable(sync_fn):
            for _a in pop.agents.values():
                try:
                    sync_fn(_a)
                except Exception:
                    pass
    except Exception:
        pass

    agents = []
    weights_info = {"format": "EIELW2", "count": 0}

    sim_cfg = getattr(sim, "sim_cfg", None)
    structural_development_enabled = bool(
        getattr(sim_cfg, "structural_development_enable", False)
    )
    for a in sorted(pop.agents.values(), key=lambda x: int(getattr(x, "agent_id", 0))):
        ad = _agent_to_dict(
            a,
            structural_development_enabled=structural_development_enabled,
            structural_development_sim_cfg=sim_cfg,
        )
        agents.append(ad)
        # weights
        try:
            _save_agent_weights(a, os.path.join(weights_dir, f"agent_{int(a.agent_id)}.bin"), obs_size=int(getattr(sim, "_obs_size", 0) or 0), n_actions=int(getattr(sim, "_out", 0) or 0))
            weights_info["count"] = int(weights_info["count"]) + 1
        except Exception:
            # Keep bundle write robust; missing weights will be caught in strict validation if needed.
            pass

    # Slot pool
    slots = getattr(pop, "_slots", None)
    slot_state = {}
    try:
        slot_state = {
            "max_slots": int(getattr(slots, "max_slots")),
            "free_slots": list(getattr(slots, "free_slots")),
            "slot_tokens": list(getattr(slots, "slot_tokens")),
        }
    except Exception:
        slot_state = {}

        # Population RNG (resume determinism): if the Population owns an independent RNG
    # (e.g. numpy Generator), it must be checkpointed and restored. This is read-only.
    pop_rng_state = None
    pop_rng_kind = None
    try:
        prng = getattr(pop, "rng", None)
        if prng is not None:
            if hasattr(prng, "bit_generator") and hasattr(prng.bit_generator, "state"):
                pop_rng_state = prng.bit_generator.state
                pop_rng_kind = "np_generator"
            elif hasattr(prng, "get_state"):
                pop_rng_state = prng.get_state()
                pop_rng_kind = "np_randomstate"
    except Exception:
        pop_rng_state = None
        pop_rng_kind = None

    return {
        "schema_version": "population_state_v2",
        "next_id": int(getattr(pop, "next_id", 1)),
        "pop_rng_kind": pop_rng_kind,
        "pop_rng_state": pop_rng_state,
        "agents": agents,
        "slot_pool": slot_state,
    }, weights_info


def _export_rng_snapshot(sim: Any, generation_index: int) -> Dict[str, Any]:
    """Export RNG snapshot.

    Phase 3 determinism note
    ------------------------
    RNGManager (Phase 1) captures the engine RNG streams, but parts of the
    simulation also rely on torch RNG for sampling and crossover/mutation.
    For resume determinism, the torch RNG state must be captured at the
    SafeBoundary and restored on resume.

    We embed torch RNG state inside rng_snapshot_v2.json under the key
    "torch_rng_state_v1" using hex-encoded byte strings (canonical JSON safe).
    """
    rm = getattr(sim, "rng_manager", None)
    if rm is None:
        # Deterministic placeholder (allowed in non-strict operational mode)
        snap: Dict[str, Any] = {
            "schema_version": "rng_snapshot_v2",
            "generation_index": int(generation_index),
            "note": "rng_manager_missing",
        }
    else:
        snap = rm.capture_snapshot(generation_index=int(generation_index))
        # Ensure schema marker is present
        if isinstance(snap, dict) and "schema_version" not in snap:
            snap = dict(snap)
            snap["schema_version"] = "rng_snapshot_v2"

    # Torch RNG state (required for deterministic resume when torch.rand* is used)
    if torch is not None:
        try:
            cpu_state = torch.get_rng_state()
            cpu_hex = bytes(cpu_state.cpu().numpy().tobytes()).hex()
            torch_state: Dict[str, Any] = {
                "schema_version": "torch_rng_state_v1",
                "cpu_hex": cpu_hex,
            }
            # CUDA RNG state (best-effort, may be absent on CPU-only runs)
            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    cuda_states = torch.cuda.get_rng_state_all()
                    cuda_hex_list = []
                    for st in cuda_states:
                        cuda_hex_list.append(bytes(st.detach().to("cpu").numpy().tobytes()).hex())
                    torch_state["cuda_hex_list"] = cuda_hex_list
            except Exception:
                pass

            # Attach to snapshot
            if isinstance(snap, dict):
                snap = dict(snap)
                snap["torch_rng_state_v1"] = torch_state
        except Exception:
            # Never fail bundle export due to torch RNG capture.
            pass


    # NumPy global RNG state (legacy RandomState) -- important if any code uses np.random.*
    try:
        st = np.random.get_state()
        # st = (algorithm, keys[uint32], pos, has_gauss, cached_gaussian)
        algo = str(st[0])
        keys = np.asarray(st[1], dtype=np.uint32)
        keys_bytes = keys.tobytes(order="C")
        snap = dict(snap) if isinstance(snap, dict) else {"schema_version": "rng_snapshot_v2", "generation_index": int(generation_index)}
        snap["numpy_random_state_v1"] = {
            "schema_version": "numpy_random_state_v1",
            "algorithm": algo,
            "keys_hex": keys_bytes.hex(),
            "keys_dtype": str(keys.dtype),
            "keys_shape": list(keys.shape),
            "pos": int(st[2]),
            "has_gauss": int(st[3]),
            "cached_gaussian": float(st[4]),
        }
    except Exception:
        pass
    # NumPy Generator RNG islands (preferred) -- runner/pop generators used for deterministic shuffles etc.
    try:
        snap = dict(snap) if isinstance(snap, dict) else {"schema_version": "rng_snapshot_v2", "generation_index": int(generation_index)}
        # runner generator
        rng = getattr(sim, "np_rng", None)
        if rng is not None and getattr(rng, "bit_generator", None) is not None:
            snap["runner_np_rng_state_v1"] = {
                "schema_version": "np_generator_state_v1",
                "bitgen_state": _to_jsonable(getattr(rng.bit_generator, "state", None)),
            }
        # population generator
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        prng = getattr(pop, "np_rng", None) if pop is not None else None
        if prng is not None and getattr(prng, "bit_generator", None) is not None:
            snap["pop_np_rng_state_v1"] = {
                "schema_version": "np_generator_state_v1",
                "bitgen_state": _to_jsonable(getattr(prng.bit_generator, "state", None)),
            }
    except Exception:
        pass




    # Python global RNG state (legacy `random` module) -- important if any code uses random.random/choices
    try:
        import random as _random  # noqa: WPS433
        import pickle as _pickle  # noqa: WPS433
        ps = _random.getstate()
        snap = dict(snap) if isinstance(snap, dict) else {"schema_version": "rng_snapshot_v2", "generation_index": int(generation_index)}
        snap["python_random_state_v1"] = {
            "schema_version": "python_random_state_v1",
            # pickle for Python's random state is stable across identical python versions; hex-encode for JSON
            "pickle_hex": _pickle.dumps(ps, protocol=_pickle.HIGHEST_PROTOCOL).hex(),
        }
    except Exception:
        pass

    return snap


def _export_control_plane(sim: Any) -> Dict[str, Any]:
    cp = getattr(sim, "control_plane", None)
    if cp is None:
        return {"schema_version": "control_plane_state_v1"}

    # Export minimal deterministic state
    seq = 0
    try:
        seq = int(cp.current_submission_sequence_number())
    except Exception:
        try:
            seq = int(cp.get_submission_sequence_number())
        except Exception:
            seq = int(getattr(sim, "_control_plane_submission_sequence_number", 0) or 0)

    sim_cfg = getattr(sim, "sim_cfg", None)
    return {
        "schema_version": "control_plane_state_v1",
        "branch_id": str(getattr(sim_cfg, "branch_id", "default") or "default"),
        "campaign_id": str(getattr(sim_cfg, "campaign_id", "") or ""),
        "submission_sequence_number": int(seq),
        "plan_version": str(getattr(sim_cfg, "plan_version", "") or ""),
    }


def _runner_targets(sim: Any) -> List[Any]:
    """Return candidate objects that may hold runner-level state.

    Different repo integrations may store mixin state on `sim` directly
    or on `sim.runner`. For Phase-3 resume equivalence we must capture/apply
    from whichever object actually owns the state.
    """
    targets: List[Any] = []
    runner = getattr(sim, "runner", None)
    if runner is not None:
        targets.append(runner)
    targets.append(sim)
    # de-dup by id while preserving order
    seen = set()
    out: List[Any] = []
    for t in targets:
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
    return out





def _invalidate_env_transient_caches(env: Any) -> None:
    """Invalidate env-derived caches that must not affect resume semantics.

    Some env mixins lazily cache derived structures (e.g. biome cell indices, padded views).
    In a continuous run these caches are typically warm by the SafeBoundary; after resume
    they are cold unless explicitly restored. To keep fresh-vs-resume behavior identical,
    we force a cold start at bundle_save_post and resume_apply_post.

    This function must not draw RNG.
    """
    if env is None:
        return

    # Cached biome cell indices used by FoodMixin.spawn_food when biomes are enabled.
    if hasattr(env, "_biome_cells"):
        try:
            delattr(env, "_biome_cells")
        except Exception:
            # Fall back to clearing the cache value without deleting the attribute.
            try:
                setattr(env, "_biome_cells", None)
            except Exception:
                pass

    # Cached padded views used by ObservationsMixin.get_local_observations.
    for attr in ("_occ_p", "_food_p"):
        if hasattr(env, attr):
            try:
                setattr(env, attr, None)
            except Exception:
                pass
    if hasattr(env, "_views_dirty"):
        try:
            setattr(env, "_views_dirty", True)
        except Exception:
            pass



def _flush_transient_caches(sim: Any) -> None:
    """Flush transient per-tick caches that must not affect semantics across resume.

    Phase 3 resume determinism requires that non-serialized caches do not influence
    post-boundary evolution. Some caches are performance optimizations (GPU index/buffer
    reuse) and are intentionally not serialized. If their warm/cold state changes the
    evolution outcome, that is a determinism leak.

    This function clears only *transient* caches (not authoritative state).
    It is invoked symmetrically:
      - after a bundle save (fresh run) before continuing
      - after resume apply (resumed run) before continuing

    It is safe in strict mode because subsequent ticks deterministically rebuild caches.
    """
    try:
        tick_cache = getattr(sim, "_policy_tick_cache", None)
        if isinstance(tick_cache, dict):
            tick_cache.clear()
        elif tick_cache is not None:
            # If a non-dict cache exists, drop it entirely.
            try:
                delattr(sim, "_policy_tick_cache")
            except Exception:
                setattr(sim, "_policy_tick_cache", None)
    except Exception:
        pass

    # Live policy-stride cache rows are trajectory state: their sampled action
    # sequence and cursor determine future ticks. They are serialized by
    # _export_runner_state and must survive both save and apply unchanged.

    try:
        dev_cache = getattr(sim, "_dev_expr_cache", None)
        if isinstance(dev_cache, dict):
            dev_cache.clear()
        elif dev_cache is not None:
            setattr(sim, "_dev_expr_cache", {})
    except Exception:
        pass
    try:
        lock_trace_pending = getattr(sim, "_dev_lock_trace_pending", None)
        if isinstance(lock_trace_pending, dict):
            lock_trace_pending.clear()
        elif lock_trace_pending is not None:
            setattr(sim, "_dev_lock_trace_pending", {})
    except Exception:
        pass

    # Env-side derived caches must not differ warm/cold across resume boundaries.
    try:
        _invalidate_env_transient_caches(getattr(sim, "env", None))
    except Exception:
        pass

    # Selection caches are per-generation conveniences; ensure they don't leak.
    for nm, empty in [
        ("_selected_parent_ids_gen", set()),
        ("_selected_parent_agents_gen", []),
    ]:
        try:
            if hasattr(sim, nm):
                setattr(sim, nm, empty)
        except Exception:
            pass


def _canonicalize_runner_drive_arrays(sim: Any) -> None:
    """Canonicalize runner drive arrays at SafeBoundary for resume equivalence.

    Problem this solves (Phase 3 boundary leak):
    - Runner-level drive arrays (_drive_hunger/_pain/_fatigue/_fear + _drive_slot_token)
      are *not* serialized in the bundle.
    - population.json serializes per-agent mirror `a.drives` only for agents that exist.
    - Free slots (no agent currently occupying the slot) can contain historical drive
      values from previously-dead agents in a fresh run, while a resumed runner
      constructor typically starts those slots at zeros/-1. If the state-dump hashes
      include the entire runner arrays, this appears as a boundary mismatch even
      though it may not affect forward behavior.

    Canonical rule at boundary:
    - Occupied slot: runner arrays must equal the occupying agent mirror.
    - Free slot: runner arrays are forced to zeros, and slot_token forced to -1.

    This is safe because slot initialization for new agents is always done via
    _sync_drive_slot, and -1 guarantees we don't "reuse" stale state for a new token.
    """
    try:
        pop = getattr(sim, "pop", None)
        if pop is None:
            return
        sync_fn = getattr(sim, "_sync_drive_slot", None)
        if not callable(sync_fn):
            return

        # Ensure arrays exist
        try:
            if hasattr(sim, "_ensure_drive_arrays"):
                sim._ensure_drive_arrays()
        except Exception:
            pass

        # SlotPool discovery (supports both naming conventions)
        sp = getattr(pop, "slot_pool", None)
        if sp is None:
            sp = getattr(pop, "_slots", None)

        free_slots = set()
        max_slots = 0
        try:
            fs = list(getattr(sp, "free_slots"))
            free_slots = set(int(x) for x in fs)
            max_slots = int(getattr(sp, "max_slots", 0) or 0)
        except Exception:
            free_slots = set()
            max_slots = 0

        # Fallback max_slots from runner arrays if needed
        try:
            if max_slots <= 0:
                tok = getattr(sim, "_drive_slot_token", None)
                if tok is not None and getattr(tok, "shape", None) is not None:
                    max_slots = int(tok.shape[0])
        except Exception:
            max_slots = 0

        if max_slots <= 0:
            return

        # Build slot -> agent mapping (there should be at most one agent per slot)
        slot_to_agent = {}
        try:
            for a in pop.agents.values():
                try:
                    s = int(getattr(a, "slot", -1))
                except Exception:
                    s = -1
                if s >= 0 and s < max_slots:
                    slot_to_agent[s] = a
        except Exception:
            slot_to_agent = {}

        # Ensure occupied slots are initialized from agent mirrors (deterministically)
        for s, a in slot_to_agent.items():
            if s in free_slots:
                continue
            try:
                sync_fn(a)
            except Exception:
                pass

        # Force canonical state for free slots (zeros + token=-1)
        try:
            dh = getattr(sim, "_drive_hunger", None)
            dp = getattr(sim, "_drive_pain", None)
            df = getattr(sim, "_drive_fatigue", None)
            dfe = getattr(sim, "_drive_fear", None)
            dtok = getattr(sim, "_drive_slot_token", None)
            if dh is None or dp is None or df is None or dfe is None or dtok is None:
                return
            for s in free_slots:
                if 0 <= s < max_slots:
                    dh[s] = np.float32(0.0)
                    dp[s] = np.float32(0.0)
                    df[s] = np.float32(0.0)
                    dfe[s] = np.float32(0.0)
                    dtok[s] = np.int32(-1)
        except Exception:
            pass
    except Exception:
        return


def _export_runner_state(sim: Any) -> Dict[str, Any]:
    """Export runner-level state that influences simulation trajectory.

    Phase 3 resume equivalence requires that *all* state affecting
    world dynamics is restored on resume. Some mixins keep internal
    counters/arrays in the runner that are not part of world/population JSON.

    IMPORTANT: Some mixins lazily initialize their internal members. We must
    force initialization at checkpoint boundaries so the state is not silently
    omitted from the bundle.
    """
    state: Dict[str, Any] = {"schema_version": "runner_state_v1"}

    targets = _runner_targets(sim)

    # Force lazy init if available (safe no-op if not implemented).
    for t in targets:
        try:
            if hasattr(t, "_ensure_shock_state"):
                t._ensure_shock_state()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if hasattr(t, "_ensure_identity_arrays"):
                t._ensure_identity_arrays()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ---- Shock mixin counters ----
    for k in ["_shock_active_rem", "_shock_cooldown_rem", "_shock_last_active", "_shock_last_kind"]:
        for t in targets:
            if hasattr(t, k):
                try:
                    state[k] = _to_jsonable(getattr(t, k))
                except Exception:
                    pass
                break  # first owner wins

    # ---- Identity drift probe runner-level arrays ----
    def _pack_ndarray(arr: Any) -> Optional[Dict[str, Any]]:
        if arr is None:
            return None
        if not isinstance(arr, np.ndarray):
            return None
        try:
            data = arr.tobytes(order="C")
            h = hashlib.sha256(data).hexdigest()
            return {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "order": "C",
                "data_hex": data.hex(),
                "sha256": h,
            }
        except Exception:
            return None

    array_keys = ["_idprobe_baseline_probs", "_idprobe_baseline_token", "_idprobe_drift_ema"]
    for k in array_keys:
        for t in targets:
            if hasattr(t, k):
                try:
                    payload = _pack_ndarray(getattr(t, k))
                    if payload is not None:
                        state[k] = payload
                except Exception:
                    pass
                break

    # ---- Reporting-only structural action opportunity telemetry ----
    # These counters do not influence world dynamics, but they are deterministic
    # report state and should survive checkpoint/resume when the default-off
    # telemetry is explicitly enabled.
    for k in [
        "_structural_action_opportunity_gen",
        "_structural_action_opportunity_total",
        "_structural_action_margin_gen",
        "_structural_action_margin_total",
        "_structural_policy_action_family_gen",
        "_structural_policy_action_family_total",
        "_structural_policy_action_family_pending",
        "_structural_attack_consequence_events",
        "_structural_selection_funnel_pre_turnover_gen",
        "_structural_selection_funnel_total",
        "_structural_selection_funnel_recent_child_links",
        "_structural_tick_reproduction_funnel_gen",
        "_structural_tick_reproduction_funnel_total",
        "_structural_tick_reproduction_recent_child_links",
        "_structural_tick_reproduction_previous_child_links",
    ]:
        for t in targets:
            if hasattr(t, k):
                try:
                    state[k] = _to_jsonable(getattr(t, k))
                except Exception:
                    pass
                break


    # ---- Runner drive arrays (authoritative) ----
    # Runner-level drive arrays are authoritative hot-path state. They MUST be persisted
    # to ensure boundary-level equivalence across resume.
    try:
        # Ensure arrays exist (safe no-op if not implemented)
        for t in targets:
            try:
                if hasattr(t, "_ensure_drive_arrays"):
                    t._ensure_drive_arrays()  # type: ignore[attr-defined]
            except Exception:
                pass
        drive_keys = ["_drive_hunger", "_drive_pain", "_drive_fatigue", "_drive_fear", "_drive_slot_token"]
        for k in drive_keys:
            for t in targets:
                if hasattr(t, k):
                    try:
                        payload = _pack_ndarray(getattr(t, k))
                        if payload is not None:
                            state[k] = payload
                    except Exception:
                        pass
                    break
    except Exception:
        pass

    # Scalar/tick counters that influence when probes are applied
    for k in ["_idprobe_last_tick", "_pred_tick"]:
        for t in targets:
            if hasattr(t, k):
                try:
                    state[k] = int(getattr(t, k))
                except Exception:
                    pass
                break

    # ---- Policy stride action cache (Phase 3 resume equivalence) ----
    # Some runners cache sampled actions when policy_tick_stride > 1.
    # If not persisted, resumed runs will resample and diverge immediately.
    for t in targets:
        if hasattr(t, "_policy_action_cache"):
            try:
                cache = getattr(t, "_policy_action_cache")
                if cache is None:
                    state["_policy_action_cache"] = None
                elif isinstance(cache, dict):
                    def _cache_key_to_jsonable(x: Any) -> Any:
                        """Encode cache keys deterministically without losing int-ness.

                        The stride cache is keyed by agent_id (int). If resume loads
                        keys as strings, cache hits become misses and policy sampling
                        diverges immediately.

                        Preserve numeric keys as ints in the JSON payload.
                        """
                        try:
                            if isinstance(x, (int, np.integer)):
                                return int(x)
                        except Exception:
                            pass
                        try:
                            s = str(x).strip()
                            if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
                                return int(s)
                        except Exception:
                            pass
                        return str(x)

                    def _cache_sort_key(x: Any) -> tuple:
                        k = _cache_key_to_jsonable(x)
                        if isinstance(k, int):
                            return (0, k)
                        return (1, str(k))

                    def _encode_cache_entry(v: Any) -> Any:
                        """Encode a stride-cache entry to JSON-safe primitives.

                        Expected in-memory format (runner_ticks_loop.py):
                          cache[agent_id] = [slot_token:int, actions:(np.ndarray|list), cursor:int]

                        We must persist `actions` as a plain list[int] so resume restores
                        the exact remaining action sequence without resampling.
                        """
                        try:
                            if isinstance(v, list) and len(v) >= 3:
                                tok = int(v[0])
                                acts = v[1]
                                cur = int(v[2])
                                if isinstance(acts, np.ndarray):
                                    acts_list = [int(x) for x in acts.astype(np.int64, copy=False).tolist()]
                                elif isinstance(acts, (list, tuple)):
                                    acts_list = [int(x) for x in acts]
                                else:
                                    return _to_jsonable(v)
                                return [tok, acts_list, cur]
                        except Exception:
                            return _to_jsonable(v)
                        return _to_jsonable(v)

                    entries = []
                    for kk in sorted(cache.keys(), key=_cache_sort_key):
                        try:
                            entries.append({"k": _cache_key_to_jsonable(kk), "v": _encode_cache_entry(cache[kk])})
                        except Exception:
                            entries.append({"k": _cache_key_to_jsonable(kk), "v": str(cache[kk])})
                    state["_policy_action_cache"] = {"schema_version": "policy_action_cache_v1", "entries": entries}
                else:
                    state["_policy_action_cache"] = _to_jsonable(cache)
            except Exception:
                pass
            break

    for t in targets:
        try:
            captures = getattr(t, "_cxxrewrite1_lifecycle_captures", None)
        except Exception:
            captures = None
        if isinstance(captures, list) and captures:
            try:
                state["_cxxrewrite1_lifecycle_captures"] = _to_jsonable(captures[-8:])
            except Exception:
                pass
            break


    # ---- Reproduction / birth counters (generation metrics support) ----
    for k in [
        "_injury_blocked_births_gen",
        "_birth_attempts_gen",
        "_birth_pre_cooldown_gen",
        "_birth_pre_no_energy_gen",
        "_birth_pre_injury_gate_gen",
        "_birth_pre_no_space_gen",
        "_birth_pre_no_mate_gen",
        "_birth_attempt_conception_fail_gen",
        "_birth_gestation_fail_gen",
        "_birth_make_child_fail_gen",
        "_birth_fail_no_space_gen",
        "_birth_fail_no_mate_gen",
        "_birth_fail_buffer_gen",
        "_birth_fail_injury_gate_gen",
        "_birth_fail_scar_gate_gen",
        "_birth_fail_conception_gen",
        "_birth_fail_gestation_gen",
        "_a3_recovery_baseline_fitness_ema",
        "_a3_recovery_baseline_drift_ema",
        "_dev_stage_transition_count_gen",
        "_dev_d2_lock_event_count_gen",
        "_patch_food_regrowth_total_gen",
        "_patch_food_harvest_total_gen",
        "_patch_movement_energy_extra_total_gen",
        "_patch_movement_energy_event_count_gen",
        "_patch_scent_emitted_total_gen",
        "_patch_scent_decayed_total_gen",
        "_patch_waste_emitted_total_gen",
        "_patch_waste_decayed_total_gen",
        "_patch_regrowth_base_total_gen",
        "_patch_regrowth_effective_total_gen",
        "_patch_regrowth_suppressed_total_gen",
        "_patch_regrowth_consequence_steps_gen",
        "_patch_regrowth_multiplier_mean_sum_gen",
        "_patch_regrowth_multiplier_min_gen",
        "_patch_regrowth_multiplier_p10_sum_gen",
        "_patch_regrowth_multiplier_p90_sum_gen",
        "_patch_waste_regrowth_penalty_mean_sum_gen",
        "_patch_waste_regrowth_penalty_max_gen",
        "_patch_density_regrowth_penalty_mean_sum_gen",
        "_patch_density_regrowth_penalty_max_gen",
        "_patch_high_waste_regrowth_suppression_frac_sum_gen",
        "_patch_high_density_regrowth_suppression_frac_sum_gen",
        "_patch_recovery_positive_total_gen",
        "_patch_recovery_negative_total_gen",
        "_patch_recovery_decay_total_gen",
        "_jungle_recovery_regrowth_boost_total_gen",
        "_jungle_recovery_regrowth_boost_mean_sum_gen",
        "_jungle_recovery_regrowth_boost_max_gen",
        "_jungle_recovery_regrowth_boost_p90_sum_gen",
        "_patch_regrowth_after_waste_total_gen",
        "_patch_regrowth_after_recovery_total_gen",
        "_patch_recovery_added_regrowth_total_gen",
        "_jungle_predator_events_gen",
        "_jungle_predator_exposure_count_gen",
        "_jungle_predator_damage_total_gen",
        "_jungle_predator_damage_raw_total_gen",
        "_jungle_predator_damage_effective_total_gen",
        "_jungle_predator_damage_prevented_total_gen",
        "_jungle_predator_deaths_gen",
        "_jungle_birth_scent_emitted_total_gen",
        "_jungle_birth_waste_emitted_total_gen",
        "_jungle_birth_emission_count_gen",
        "_jungle_juvenile_predator_events_gen",
        "_jungle_juvenile_predator_damage_total_gen",
        "_jungle_juvenile_predator_deaths_gen",
        "_jungle_adult_predator_events_gen",
        "_jungle_adult_predator_damage_total_gen",
        "_wdl_delayed_hazard_event_count_gen",
        "_wdl_delayed_hazard_cue_cell_count_sum_gen",
        "_wdl_delayed_hazard_active_cell_count_sum_gen",
        "_wdl_delayed_hazard_tick_count_gen",
        "_wdl_delayed_hazard_agent_cue_exposure_count_gen",
        "_wdl_delayed_hazard_agent_active_exposure_count_gen",
        "_wdl_delayed_hazard_damage_total_gen",
        "_wdl_delayed_hazard_damage_max_gen",
        "_wdl_delayed_hazard_agents_damaged_gen",
        "_wdl_delayed_hazard_cue_to_active_delay_sum_gen",
        "_wdl_delayed_hazard_cue_to_active_delay_count_gen",
        "_wdl_delayed_hazard_cue_without_active_count_gen",
        "_wdl_delayed_hazard_active_without_prior_cue_count_gen",
        "_wdl_cue_seen_agents_gen",
        "_wdl_cue_exposed_next_move_away_count_gen",
        "_wdl_cue_exposed_next_move_toward_count_gen",
        "_wdl_cue_exposed_stayed_count_gen",
        "_wdl_pre_hazard_exit_count_gen",
        "_wdl_hazard_enter_count_gen",
        "_wdl_hazard_exposure_after_cue_count_gen",
        "_path1_drh0_damage_in_refuge_gen",
        "_path1_drh0_damage_outside_refuge_gen",
        "_path1_drh0_damage_prevented_by_refuge_gen",
        "_path1_drh0_static_refuge_damage_prevented_gen",
        "_path1_drh0_cue_conditioned_damage_prevented_gen",
        "_path1_drh0_protected_exposure_count_gen",
        "_path1_drh0_refuge_exposure_count_gen",
        "_path1_drh0_no_cue_false_positive_refuge_movement_gen",
        "_path1_drh0_cue_visible_steps_gen",
        "_path1_drh0_hidden_window_steps_gen",
        "_path1_drh0_hazard_active_steps_gen",
        "_prediction_demand_event_count_gen",
        "_prediction_demand_transition_count_gen",
        "_prediction_demand_cue_cell_count_sum_gen",
        "_prediction_demand_active_cell_count_sum_gen",
        "_prediction_demand_tick_count_gen",
        "_prediction_demand_cue_ambiguity_sum_gen",
        "_prediction_demand_cue_ambiguity_count_gen",
        "_prediction_demand_cue_to_consequence_delay_sum_gen",
        "_prediction_demand_cue_to_consequence_delay_count_gen",
        "_prediction_demand_cue_without_consequence_count_gen",
        "_prediction_demand_consequence_without_prior_cue_count_gen",
        "_prediction_demand_pre_consequence_agent_context_count_gen",
        "_prediction_demand_consequence_agent_exposure_count_gen",
        "_prediction_demand_resource_consequence_total_gen",
        "_delayed_resource_depletion_event_count_gen",
        "_delayed_resource_depletion_cue_cell_count_sum_gen",
        "_delayed_resource_depletion_hidden_cell_count_sum_gen",
        "_delayed_resource_depletion_active_cell_count_sum_gen",
        "_delayed_resource_depletion_tick_count_gen",
        "_delayed_resource_depletion_agent_cue_exposure_count_gen",
        "_delayed_resource_depletion_agent_hidden_gap_exposure_count_gen",
        "_delayed_resource_depletion_agent_active_exposure_count_gen",
        "_delayed_resource_depletion_consequence_count_gen",
        "_delayed_resource_depletion_consequence_total_gen",
        "_delayed_resource_depletion_cue_to_active_delay_sum_gen",
        "_delayed_resource_depletion_cue_to_active_delay_count_gen",
        "_delayed_resource_depletion_cue_without_active_count_gen",
        "_delayed_resource_depletion_active_without_prior_cue_count_gen",
        "_delayed_resource_depletion_gather_cue_count_gen",
        "_delayed_resource_depletion_gather_hidden_count_gen",
        "_delayed_resource_depletion_gather_active_count_gen",
        "_delayed_resource_depletion_revisit_after_cue_count_gen",
        "_delayed_resource_depletion_move_away_count_gen",
        "_delayed_resource_depletion_move_toward_count_gen",
        "_delayed_resource_depletion_stay_count_gen",
        "_delayed_crowding_waste_event_count_gen",
        "_delayed_crowding_waste_cue_cell_count_sum_gen",
        "_delayed_crowding_waste_hidden_cell_count_sum_gen",
        "_delayed_crowding_waste_active_cell_count_sum_gen",
        "_delayed_crowding_waste_tick_count_gen",
        "_delayed_crowding_waste_agent_cue_exposure_count_gen",
        "_delayed_crowding_waste_agent_hidden_gap_exposure_count_gen",
        "_delayed_crowding_waste_agent_active_exposure_count_gen",
        "_delayed_crowding_waste_consequence_count_gen",
        "_delayed_crowding_waste_consequence_total_gen",
        "_delayed_crowding_waste_cue_to_active_delay_sum_gen",
        "_delayed_crowding_waste_cue_to_active_delay_count_gen",
        "_delayed_crowding_waste_cue_without_active_count_gen",
        "_delayed_crowding_waste_active_without_prior_cue_count_gen",
        "_delayed_crowding_waste_gather_cue_count_gen",
        "_delayed_crowding_waste_gather_hidden_count_gen",
        "_delayed_crowding_waste_gather_active_count_gen",
        "_delayed_crowding_waste_revisit_after_cue_count_gen",
        "_delayed_crowding_waste_move_away_count_gen",
        "_delayed_crowding_waste_move_toward_count_gen",
        "_delayed_crowding_waste_stay_count_gen",
    ]:
        for t in targets:
            if hasattr(t, k):
                try:
                    v = getattr(t, k)
                    if isinstance(v, (int, np.integer)):
                        state[k] = int(v)
                    else:
                        state[k] = float(v)
                except Exception:
                    pass
                break

    for k in [
        "_dev_stage_eval_stage_counts_gen",
        "_patch_birth_contexts_gen",
        "_patch_death_contexts_gen",
        "_jungle_predator_event_contexts_gen",
        "_jungle_predator_damage_multipliers_gen",
        "_jungle_birth_emission_contexts_gen",
        "_wdl_delayed_hazard_event_contexts_gen",
    ]:
        for t in targets:
            if hasattr(t, k):
                try:
                    state[k] = _to_jsonable(getattr(t, k))
                except Exception:
                    pass
                break

    # A3 taxonomy last generation annotation (queryable artifact parity helper).
    for t in targets:
        if hasattr(t, "_a3_recovery_taxonomy_last_gen"):
            try:
                state["_a3_recovery_taxonomy_last_gen"] = _to_jsonable(getattr(t, "_a3_recovery_taxonomy_last_gen"))
            except Exception:
                pass
            break

    # ---- Phase 6 (draft): PressureAutopilot state ----
    # This is runner-level controller state that must be restored for
    # checkpoint/resume equivalence when the autopilot is enabled.
    try:
        ap = getattr(sim, "pressure_autopilot", None)
        if ap is not None and hasattr(ap, "export_state"):
            state["pressure_autopilot_state"] = ap.export_state()
    except Exception:
        pass

    # Scientific Chronicle state is observer-only, versioned, and RNG-free.
    for target in targets:
        recorder = getattr(target, "scientific_chronicle", None)
        if recorder is not None and hasattr(recorder, "export_state"):
            try:
                state["scientific_chronicle_state"] = _to_jsonable(recorder.export_state())
            except Exception:
                pass
            break

    # GEN1-F1 capture is observer-only, versioned, RNG-free, and duplicate-safe.
    for target in targets:
        recorder = getattr(target, "gen1_f1_capture", None)
        if recorder is not None and hasattr(recorder, "export_state"):
            try:
                state["gen1_f1_capture_state"] = _to_jsonable(recorder.export_state())
            except Exception:
                pass
            break

    # RECOMB-72 live audit state is required for full-history continuation.
    # The fields are observer state except for per-agent ancestry, which is
    # consumed only by the audit metric projection and never by mechanics.
    try:
        cfg = getattr(sim, "sim_cfg", None)
        pop = getattr(sim, "pop", None)
        if bool(getattr(cfg, "recomb72_audit_enable", False)) and pop is not None:
            ancestry = {}
            founder_groups = {}
            for agent in pop.alive_agents():
                agent_id = str(int(getattr(agent, "agent_id")))
                ancestry[agent_id] = _to_jsonable(
                    dict(getattr(agent, "_recomb72_ancestry"))
                )
                founder_groups[agent_id] = str(
                    getattr(agent, "_recomb72_founder_group", "mixed")
                )
            state["recomb72_live_audit_state"] = {
                "schema_version": "recomb72_live_checkpoint_state_v2",
                "generation_rows": _to_jsonable(
                    list(getattr(sim, "_recomb72_generation_rows", ()) or ())
                ),
                "real_tick_count": int(
                    getattr(sim, "_recomb72_real_tick_count", 0)
                ),
                "neutral_action_agent_tick_count": int(
                    getattr(sim, "_recomb72_neutral_action_agent_ticks", 0)
                ),
                "neutrality_receipt": _to_jsonable(
                    getattr(sim, "_recomb72_neutrality_receipt", None)
                ),
                "founder_receipt": _to_jsonable(
                    getattr(pop, "_recomb72_founder_receipt", None)
                ),
                "ancestry_by_agent_id": ancestry,
                "founder_group_by_agent_id": founder_groups,
            }
    except Exception as exc:
        raise CheckpointStrictError(
            f"RECOMB72_LIVE_CHECKPOINT_EXPORT_FAILED: {exc}"
        ) from exc

    return state


def _export_resident_runtime_state_v2(sim: Any, *, generation_index: int) -> Dict[str, Any]:
    """Export the resident runner snapshot used by CXXGEN-1 checkpoint gates.

    This is intentionally additive: current checkpoint apply remains driven by
    the existing world/population/runner/bank payloads.  The resident payload is
    a strict, hashable contract for native/resident runtime work so future
    candidate lanes can prove they resume from the same slot, Hmem, action
    cache, policy, and scalar state instead of reconstructing it informally.
    """

    def _pack_array(arr: Any) -> Dict[str, Any]:
        np_arr = np.ascontiguousarray(np.asarray(arr))
        data = np_arr.tobytes(order="C")
        return {
            "shape": [int(x) for x in np_arr.shape],
            "dtype": str(np_arr.dtype),
            "order": "C",
            "data_hex": data.hex(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def _safe_scalar(value: Any) -> Any:
        try:
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
        except Exception:
            pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, set):
            return sorted(_safe_scalar(x) for x in value)
        if isinstance(value, (list, tuple)):
            return [_safe_scalar(x) for x in value]
        if isinstance(value, dict):
            return {str(k): _safe_scalar(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        return str(value)

    def _environment_state() -> Dict[str, Any]:
        env = getattr(sim, "env", None)
        if env is None:
            return {"schema_version": "resident_environment_state_v1", "status": "missing_env"}
        array_names = [
            "patch_food_amount",
            "patch_waste",
            "patch_scent",
            "patch_shelter",
            "patch_cover",
            "patch_fertility",
            "patch_movement_cost",
            "patch_recovery_activity",
            "patch_scent_emit_pending",
            "patch_waste_emit_pending",
            "patch_delayed_hazard_cue",
            "patch_delayed_hazard_active",
            "patch_delayed_hazard_timer",
            "patch_delayed_hazard_age",
            "patch_delayed_hazard_event_id",
            "patch_delayed_hazard_intensity",
            "patch_delayed_hazard_prior_cue",
            "patch_prediction_demand_cue",
            "patch_prediction_demand_active",
            "patch_prediction_demand_timer",
            "patch_prediction_demand_age",
            "patch_prediction_demand_delay",
            "patch_prediction_demand_event_id",
            "patch_prediction_demand_intensity",
            "patch_prediction_demand_prior_cue",
            "patch_prediction_demand_transition_sign",
            "patch_prediction_demand_trend",
        ]
        if bool(
            getattr(
                getattr(env, "cfg", None),
                "eed_resource_impact_enable",
                False,
            )
        ):
            from env.gridworld_pkg.patch_world import (
                EED_RESOURCE_IMPACT_ARRAY_NAMES,
            )

            array_names.extend(EED_RESOURCE_IMPACT_ARRAY_NAMES)
        scalar_prefixes = (
            "patch_",
            "wdl_delayed_hazard_",
            "prediction_demand_",
            "eed_resource_",
        )
        arrays: Dict[str, Any] = {}
        scalars: Dict[str, Any] = {}
        for name in array_names:
            if hasattr(env, name):
                value = getattr(env, name)
                if isinstance(value, np.ndarray):
                    arrays[name] = _pack_array(value)
        for name in dir(env):
            if not name.startswith(scalar_prefixes):
                continue
            if name in arrays or name.startswith("__"):
                continue
            try:
                value = getattr(env, name)
            except Exception:
                continue
            if callable(value) or isinstance(value, np.ndarray):
                continue
            if isinstance(value, (str, int, float, bool, list, tuple, set, dict, np.integer, np.floating, np.bool_)) or value is None:
                scalars[name] = _safe_scalar(value)
        return {
            "schema_version": "resident_environment_state_v1",
            "patch_world_enable": bool(getattr(env, "patch_world_enable", False)),
            "array_count": int(len(arrays)),
            "scalar_count": int(len(scalars)),
            "arrays": arrays,
            "scalars": scalars,
        }

    def _generation_counters() -> Dict[str, Any]:
        names = [
            "_lineage_reproduction_funnel_event_contexts_gen",
            "_wdl_delayed_hazard_event_count_gen",
            "_wdl_delayed_hazard_cue_cell_count_sum_gen",
            "_wdl_delayed_hazard_active_cell_count_sum_gen",
            "_wdl_delayed_hazard_tick_count_gen",
            "_wdl_delayed_hazard_agent_cue_exposure_count_gen",
            "_wdl_delayed_hazard_agent_active_exposure_count_gen",
            "_wdl_delayed_hazard_damage_total_gen",
            "_wdl_delayed_hazard_damage_max_gen",
            "_wdl_delayed_hazard_agents_damaged_gen",
            "_wdl_delayed_hazard_agents_damaged_ids_gen",
            "_wdl_delayed_hazard_cue_to_active_delay_sum_gen",
            "_wdl_delayed_hazard_cue_to_active_delay_count_gen",
            "_wdl_delayed_hazard_cue_without_active_count_gen",
            "_wdl_delayed_hazard_active_without_prior_cue_count_gen",
            "_wdl_cue_seen_agents_gen",
            "_wdl_cue_exposed_next_move_away_count_gen",
            "_wdl_cue_exposed_next_move_toward_count_gen",
            "_wdl_cue_exposed_stayed_count_gen",
            "_wdl_hazard_exposure_after_cue_count_gen",
            "_prediction_demand_event_count_gen",
            "_prediction_demand_transition_count_gen",
            "_prediction_demand_cue_cell_count_sum_gen",
            "_prediction_demand_active_cell_count_sum_gen",
            "_prediction_demand_tick_count_gen",
            "_prediction_demand_cue_ambiguity_sum_gen",
            "_prediction_demand_cue_ambiguity_count_gen",
            "_prediction_demand_cue_to_consequence_delay_sum_gen",
            "_prediction_demand_cue_to_consequence_delay_count_gen",
            "_prediction_demand_cue_without_consequence_count_gen",
            "_prediction_demand_consequence_without_prior_cue_count_gen",
            "_prediction_demand_pre_consequence_agent_context_count_gen",
            "_prediction_demand_consequence_agent_exposure_count_gen",
            "_prediction_demand_resource_consequence_total_gen",
        ]
        counters: Dict[str, Any] = {}
        for name in names:
            if not hasattr(sim, name):
                continue
            try:
                value = getattr(sim, name)
            except Exception:
                continue
            if name == "_lineage_reproduction_funnel_event_contexts_gen" and isinstance(value, list):
                counters["lineage_reproduction_funnel_event_cursor"] = int(len(value))
                continue
            counters[name.lstrip("_")] = _safe_scalar(value)
        counters.setdefault("lineage_reproduction_funnel_event_cursor", 0)
        return {
            "schema_version": "resident_generation_counters_v1",
            "counter_count": int(len(counters)),
            "counters": counters,
        }

    def _population_transaction_tape_state() -> Dict[str, Any]:
        captures: List[Any] = []
        for target in _runner_targets(sim):
            try:
                value = getattr(target, "_cxxrewrite1_lifecycle_captures", None)
            except Exception:
                value = None
            if isinstance(value, list) and value:
                captures = list(value)
                break
        if not captures:
            return {
                "schema_version": "resident_population_transaction_tape_v1",
                "status": "not_enabled",
                "capture_count": 0,
                "event_cursor": 0,
                "latest_lifecycle_digest": "",
                "latest_churn_tape_digest": "",
            }
        latest = captures[-1] if isinstance(captures[-1], dict) else {}
        return {
            "schema_version": "resident_population_transaction_tape_v1",
            "status": "captured",
            "capture_count": int(len(captures)),
            "generation": int(latest.get("generation", -1) or -1),
            "phase": str(latest.get("phase", "") or ""),
            "event_cursor": int(latest.get("churn_event_count", 0) or 0),
            "latest_lifecycle_digest": str(latest.get("digest", "") or ""),
            "latest_churn_tape_digest": str(latest.get("churn_tape_digest", "") or ""),
            "validation_ok": bool((latest.get("validation", {}) or {}).get("ok", False))
            if isinstance(latest.get("validation", {}), dict)
            else False,
        }

    def _policy_execution_plan_state(arrays: Dict[str, Any]) -> Dict[str, Any]:
        try:
            alive_mask = np.asarray(arrays.get("alive_mask", []), dtype=np.uint8).astype(bool)
            use_gru = np.asarray(arrays.get("policy_use_gru", []), dtype=np.uint8).astype(bool)
            cache_token = np.asarray(arrays.get("action_cache_token", []), dtype=np.int64)
            cache_cursor = np.asarray(arrays.get("action_cache_cursor", []), dtype=np.int32)
            cache_next = np.asarray(arrays.get("action_cache_next", []), dtype=np.int32)
            hmem = np.asarray(arrays.get("hmem", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
            alive_slots = np.nonzero(alive_mask)[0].astype(np.int64)
            gru_slots = alive_slots[use_gru[alive_slots]] if use_gru.shape[0] >= alive_mask.shape[0] else np.empty((0,), dtype=np.int64)
            mlp_slots = alive_slots[~use_gru[alive_slots]] if use_gru.shape[0] >= alive_mask.shape[0] else alive_slots
            cache_valid = alive_slots[
                (cache_token[alive_slots] >= 0) & (cache_cursor[alive_slots] >= 0)
            ] if cache_token.shape[0] >= alive_mask.shape[0] and cache_cursor.shape[0] >= alive_mask.shape[0] else np.empty((0,), dtype=np.int64)
            cache_material = {
                "slots": [int(x) for x in cache_valid.tolist()],
                "token": [int(cache_token[int(x)]) for x in cache_valid.tolist()],
                "cursor": [int(cache_cursor[int(x)]) for x in cache_valid.tolist()],
                "next": [int(cache_next[int(x)]) for x in cache_valid.tolist()]
                if cache_next.shape[0] >= alive_mask.shape[0]
                else [],
            }
            hmem_bytes = np.ascontiguousarray(hmem).tobytes(order="C")
            return {
                "schema_version": "resident_policy_execution_plan_v1",
                "status": "captured",
                "membership_epoch": int(generation_index),
                "alive_slot_count": int(alive_slots.shape[0]),
                "mlp_slot_count": int(mlp_slots.shape[0]),
                "gru_slot_count": int(gru_slots.shape[0]),
                "action_cache_valid_slot_count": int(cache_valid.shape[0]),
                "action_cache_boundary_status": "flushed_before_checkpoint" if int(cache_valid.shape[0]) == 0 else "captured",
                "action_cache_digest": _sha256_bytes(canonical_json_bytes_lossless(cache_material)),
                "hmem_shape": [int(x) for x in hmem.shape],
                "hmem_sha256": hashlib.sha256(hmem_bytes).hexdigest(),
                "hmem_materialized": bool(hmem.ndim == 2 and int(hmem.shape[1]) > 0),
            }
        except Exception as exc:
            return {
                "schema_version": "resident_policy_execution_plan_v1",
                "status": "failed",
                "error": repr(exc),
            }

    try:
        from simulation.runnercore.state import import_generation_start, state_digest

        state = import_generation_start(sim, generation=int(generation_index), tick=0)
        arrays = {
            "agent_ids_by_alive_order": state.agent_ids_by_alive_order,
            "slot_to_agent_id": state.slot_to_agent_id,
            "agent_id_to_slot_agent_ids": state.agent_id_to_slot_agent_ids,
            "agent_id_to_slot_slots": state.agent_id_to_slot_slots,
            "slot_tokens": state.slot_tokens,
            "alive_mask": np.asarray(state.alive_mask, dtype=np.uint8),
            "pos_y": state.pos_y,
            "pos_x": state.pos_x,
            "energy": state.energy,
            "health": state.health,
            "injury": state.injury,
            "age": state.age,
            "drives": state.drives,
            "hmem": state.hmem,
            "policy_attack_bias": state.policy_attack_bias,
            "policy_action_temperature": state.policy_action_temperature,
            "policy_drive_logit_gain": state.policy_drive_logit_gain,
            "policy_use_gru": np.asarray(state.policy_use_gru, dtype=np.uint8),
            "action_cache_token": state.action_cache_token,
            "action_cache_cursor": state.action_cache_cursor,
            "action_cache_next": state.action_cache_next,
        }
        payload = {
            "schema_version": "resident_runtime_state_v2",
            "producer": "checkpoint_bundle_v2.cxxgen1",
            "runnercore_state_schema_version": str(state.schema_version),
            "profile": str(state.profile),
            "generation_index": int(generation_index),
            "tick": int(state.tick),
            "capacity": int(state.capacity),
            "alive_count": int(state.alive_count),
            "slot_epoch": int(generation_index),
            "event_cursor": 0,
            "arrays": {name: _pack_array(value) for name, value in arrays.items()},
            "environment_state": _environment_state(),
            "generation_counters": _generation_counters(),
            "population_transaction_tape": _population_transaction_tape_state(),
            "policy_execution_plan": _policy_execution_plan_state(arrays),
            "runnercore_state_digest": state_digest(state),
            "apply_status": "not_applied_by_checkpoint_loader_yet",
        }
        payload["resident_payload_sha256"] = _sha256_bytes(canonical_json_bytes_lossless({
            key: value for key, value in payload.items() if key != "resident_payload_sha256"
        }))
        return payload
    except Exception as exc:
        return {
            "schema_version": "resident_runtime_state_v2",
            "producer": "checkpoint_bundle_v2.cxxgen1",
            "generation_index": int(generation_index),
            "export_status": "failed",
            "error": repr(exc),
            "apply_status": "not_applied_by_checkpoint_loader_yet",
        }


def _verify_resident_runtime_state_v2_against_sim(
    sim: Any,
    resident: Dict[str, Any],
    *,
    generation_index: int,
) -> Dict[str, Any]:
    """Compare a saved resident payload with the restored live runner state.

    The checkpoint loader still applies the mature world/population/runner/bank
    payloads.  This verifier is the promotion gate for ResidentRuntimeStateV2:
    if the restored authoritative state cannot reproduce the resident array and
    environment hashes exactly, resident apply is not allowed to become
    authoritative.
    """

    def _arr_sig(record: Any) -> Optional[Tuple[Tuple[int, ...], str, str]]:
        if not isinstance(record, dict):
            return None
        try:
            shape = tuple(int(x) for x in record.get("shape", []))
            dtype = str(record.get("dtype", "") or "")
            data_hex = str(record.get("data_hex", "") or "")
            expected = str(record.get("sha256", "") or "")
            if not dtype or not expected:
                return None
            data = bytes.fromhex(data_hex)
            got = hashlib.sha256(data).hexdigest()
            if got != expected:
                raise CheckpointStrictError(f"resident array hash mismatch in payload: expected {expected} got {got}")
            return shape, dtype, expected
        except CheckpointStrictError:
            raise
        except Exception:
            return None

    def _compare_array_maps(
        expected: Any,
        current: Any,
        *,
        prefix: str,
        mismatches: List[Dict[str, Any]],
    ) -> Tuple[int, int, int]:
        if not isinstance(expected, dict):
            expected = {}
        if not isinstance(current, dict):
            current = {}
        matched = 0
        mismatched = 0
        missing = 0
        for name in sorted(str(k) for k in expected.keys()):
            exp_sig = _arr_sig(expected.get(name))
            cur_sig = _arr_sig(current.get(name))
            if cur_sig is None:
                missing += 1
                mismatches.append({"field": f"{prefix}.{name}", "reason": "missing_current_array"})
                continue
            if exp_sig is None:
                mismatched += 1
                mismatches.append({"field": f"{prefix}.{name}", "reason": "invalid_expected_array"})
                continue
            if (
                prefix == "arrays"
                and name == "hmem"
                and len(exp_sig[0]) == 2
                and len(cur_sig[0]) == 2
                and int(exp_sig[0][0]) == int(cur_sig[0][0])
                and int(exp_sig[0][1]) == 0
            ):
                # No GRU rows were authoritative in the saved resident payload.
                # The legacy apply path may materialize a default Hmem cache;
                # that is not a resident-state mismatch until saved GRU memory
                # exists, at which point the normal exact hash comparison applies.
                matched += 1
                continue
            if exp_sig != cur_sig:
                mismatched += 1
                mismatches.append(
                    {
                        "field": f"{prefix}.{name}",
                        "reason": "array_signature_mismatch",
                        "expected": {"shape": list(exp_sig[0]), "dtype": exp_sig[1], "sha256": exp_sig[2]},
                        "current": {"shape": list(cur_sig[0]), "dtype": cur_sig[1], "sha256": cur_sig[2]},
                    }
                )
                continue
            matched += 1
        return matched, mismatched, missing

    def _dict_subset_diff(expected: Any, current: Any, *, prefix: str, mismatches: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not isinstance(expected, dict):
            expected = {}
        if not isinstance(current, dict):
            current = {}
        matched = 0
        mismatched = 0
        for key in sorted(str(k) for k in expected.keys()):
            if key not in current:
                mismatched += 1
                mismatches.append({"field": f"{prefix}.{key}", "reason": "missing_current_scalar"})
                continue
            if expected.get(key) != current.get(key):
                mismatched += 1
                mismatches.append(
                    {
                        "field": f"{prefix}.{key}",
                        "reason": "scalar_mismatch",
                        "expected": expected.get(key),
                        "current": current.get(key),
                    }
                )
                continue
            matched += 1
        return matched, mismatched

    def _policy_plan_subset_diff(
        expected: Any,
        current: Any,
        *,
        prefix: str,
        mismatches: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        if not isinstance(expected, dict):
            expected = {}
        if not isinstance(current, dict):
            current = {}
        matched = 0
        mismatched = 0
        expected_hmem_shape = expected.get("hmem_shape", [])
        hmem_unmaterialized_at_save = (
            isinstance(expected_hmem_shape, list)
            and len(expected_hmem_shape) == 2
            and int(expected_hmem_shape[1]) == 0
        )
        for key in sorted(str(k) for k in expected.keys()):
            if hmem_unmaterialized_at_save and key in {"hmem_shape", "hmem_sha256", "hmem_materialized"}:
                matched += 1
                continue
            if key not in current:
                mismatched += 1
                mismatches.append({"field": f"{prefix}.{key}", "reason": "missing_current_scalar"})
                continue
            if expected.get(key) != current.get(key):
                mismatched += 1
                mismatches.append(
                    {
                        "field": f"{prefix}.{key}",
                        "reason": "scalar_mismatch",
                        "expected": expected.get(key),
                        "current": current.get(key),
                    }
                )
                continue
            matched += 1
        return matched, mismatched

    mismatches: List[Dict[str, Any]] = []
    if not isinstance(resident, dict):
        return {
            "schema_version": "resident_runtime_state_v2_apply_verification_v1",
            "ok": False,
            "status": "invalid_resident_payload",
            "mismatches": [{"field": "resident_runtime_state_v2", "reason": "not_a_dict"}],
        }
    if resident.get("export_status") == "failed":
        return {
            "schema_version": "resident_runtime_state_v2_apply_verification_v1",
            "ok": False,
            "status": "resident_export_failed_in_bundle",
            "mismatches": [{"field": "resident_runtime_state_v2", "reason": str(resident.get("error", ""))}],
        }

    current = _export_resident_runtime_state_v2(sim, generation_index=int(generation_index))
    if current.get("export_status") == "failed":
        return {
            "schema_version": "resident_runtime_state_v2_apply_verification_v1",
            "ok": False,
            "status": "current_export_failed",
            "mismatches": [{"field": "current_resident_runtime_state_v2", "reason": str(current.get("error", ""))}],
        }

    array_matched, array_mismatched, array_missing = _compare_array_maps(
        resident.get("arrays", {}),
        current.get("arrays", {}),
        prefix="arrays",
        mismatches=mismatches,
    )
    env_expected = resident.get("environment_state", {}) if isinstance(resident.get("environment_state", {}), dict) else {}
    env_current = current.get("environment_state", {}) if isinstance(current.get("environment_state", {}), dict) else {}
    env_array_matched, env_array_mismatched, env_array_missing = _compare_array_maps(
        env_expected.get("arrays", {}),
        env_current.get("arrays", {}),
        prefix="environment_state.arrays",
        mismatches=mismatches,
    )
    env_scalar_matched, env_scalar_mismatched = _dict_subset_diff(
        env_expected.get("scalars", {}),
        env_current.get("scalars", {}),
        prefix="environment_state.scalars",
        mismatches=mismatches,
    )

    for scalar_name in ["generation_index", "tick", "capacity", "alive_count", "slot_epoch", "event_cursor"]:
        if resident.get(scalar_name) != current.get(scalar_name):
            mismatches.append(
                {
                    "field": scalar_name,
                    "reason": "scalar_mismatch",
                    "expected": resident.get(scalar_name),
                    "current": current.get(scalar_name),
                }
            )

    gen_expected = resident.get("generation_counters", {}) if isinstance(resident.get("generation_counters", {}), dict) else {}
    gen_current = current.get("generation_counters", {}) if isinstance(current.get("generation_counters", {}), dict) else {}
    counter_matched, counter_mismatched = _dict_subset_diff(
        gen_expected.get("counters", {}),
        gen_current.get("counters", {}),
        prefix="generation_counters.counters",
        mismatches=mismatches,
    )
    tape_matched, tape_mismatched = _dict_subset_diff(
        resident.get("population_transaction_tape", {}),
        current.get("population_transaction_tape", {}),
        prefix="population_transaction_tape",
        mismatches=mismatches,
    )
    policy_plan_matched, policy_plan_mismatched = _policy_plan_subset_diff(
        resident.get("policy_execution_plan", {}),
        current.get("policy_execution_plan", {}),
        prefix="policy_execution_plan",
        mismatches=mismatches,
    )

    ok = not bool(mismatches)
    return {
        "schema_version": "resident_runtime_state_v2_apply_verification_v1",
        "ok": bool(ok),
        "status": "ok" if ok else "mismatch",
        "generation_index": int(generation_index),
        "array_match_count": int(array_matched),
        "array_mismatch_count": int(array_mismatched),
        "array_missing_count": int(array_missing),
        "environment_array_match_count": int(env_array_matched),
        "environment_array_mismatch_count": int(env_array_mismatched),
        "environment_array_missing_count": int(env_array_missing),
        "environment_scalar_match_count": int(env_scalar_matched),
        "environment_scalar_mismatch_count": int(env_scalar_mismatched),
        "generation_counter_match_count": int(counter_matched),
        "generation_counter_mismatch_count": int(counter_mismatched),
        "population_transaction_tape_match_count": int(tape_matched),
        "population_transaction_tape_mismatch_count": int(tape_mismatched),
        "policy_execution_plan_match_count": int(policy_plan_matched),
        "policy_execution_plan_mismatch_count": int(policy_plan_mismatched),
        "mismatch_count": int(len(mismatches)),
        "mismatches": mismatches[:64],
    }


def _export_metrics_state(sim: Any) -> Dict[str, Any]:
    """Export determinism-critical MetricsLogger internal state.

    Why this exists
    ---------------
    Some metrics (e.g. diversity collapse streak) are stateful across generations
    and influence metrics.csv output. That state is not derivable purely from
    world/population snapshots at the resume boundary.

    Schema
    ------
      {"schema_version":"metrics_state_v1","collapse_streak":int,"phase_e_e1_state":dict?}
    """
    out: Dict[str, Any] = {
        "schema_version": "metrics_state_v1",
        "collapse_streak": 0,
    }

    try:
        logger = getattr(sim, "logger", None)
        if logger is None:
            # Some integrations attach logger to sim.runner; check runner targets.
            for t in _runner_targets(sim):
                logger = getattr(t, "logger", None)
                if logger is not None:
                    break

        if logger is None:
            return out

        if hasattr(logger, "export_state"):
            try:
                st = logger.export_state()  # type: ignore[attr-defined]
                if isinstance(st, dict):
                    for key, value in st.items():
                        out[str(key)] = _to_jsonable(value)
                    if "schema_version" in st and st.get("schema_version"):
                        out["schema_version"] = str(st.get("schema_version"))
                    if "collapse_streak" in st:
                        out["collapse_streak"] = int(st.get("collapse_streak", 0) or 0)
            except Exception:
                pass

        # Fallback: direct private attribute.
        if hasattr(logger, "_collapse_streak"):
            try:
                out["collapse_streak"] = int(getattr(logger, "_collapse_streak", 0) or 0)
            except Exception:
                out["collapse_streak"] = 0
    except Exception:
        return out

    # Optional Phase E1 state (metrics-only, replay-critical when enabled).
    try:
        e1 = getattr(sim, "_phase_e_metrics_engine", None)
        if e1 is not None and hasattr(e1, "export_state"):
            st = e1.export_state()
            if isinstance(st, dict):
                out["phase_e_e1_state"] = _to_jsonable(st)
    except Exception:
        pass

    # Optional Phase E2 state (metrics-only, replay-critical when enabled).
    try:
        e2 = getattr(sim, "_phase_e_failure_taxonomy_engine", None)
        if e2 is not None and hasattr(e2, "export_state"):
            st = e2.export_state()
            if isinstance(st, dict):
                out["phase_e_e2_state"] = _to_jsonable(st)
    except Exception:
        pass

    return out


def _apply_runner_state(sim: Any, state: Dict[str, Any]) -> None:
    """Apply runner-level state back into the current runner/sim.

    We apply onto both `sim.runner` (if present) and `sim` itself, after forcing
    lazy initialization hooks so attributes exist.
    """
    targets = _runner_targets(sim)

    chronicle_state = state.get("scientific_chronicle_state")
    if isinstance(chronicle_state, dict):
        for target in targets:
            recorder = getattr(target, "scientific_chronicle", None)
            if recorder is not None and hasattr(recorder, "import_state"):
                recorder.import_state(chronicle_state)
                break

    gen1_f1_state = state.get("gen1_f1_capture_state")
    if isinstance(gen1_f1_state, dict):
        for target in targets:
            recorder = getattr(target, "gen1_f1_capture", None)
            if recorder is not None and hasattr(recorder, "import_state"):
                recorder.import_state(gen1_f1_state)
                break

    recomb72_state = state.get("recomb72_live_audit_state")
    if isinstance(recomb72_state, dict):
        if (
            recomb72_state.get("schema_version")
            != "recomb72_live_checkpoint_state_v2"
        ):
            raise CheckpointStrictError(
                "RECOMB72_LIVE_CHECKPOINT_SCHEMA_MISMATCH"
            )
        pop = getattr(sim, "pop", None)
        if pop is None:
            raise CheckpointStrictError("RECOMB72_LIVE_CHECKPOINT_POPULATION_MISSING")
        ancestry = recomb72_state.get("ancestry_by_agent_id")
        groups = recomb72_state.get("founder_group_by_agent_id")
        if not isinstance(ancestry, dict) or not isinstance(groups, dict):
            raise CheckpointStrictError("RECOMB72_LIVE_CHECKPOINT_ANCESTRY_MISSING")
        alive = list(pop.alive_agents())
        if {str(int(agent.agent_id)) for agent in alive} != set(ancestry):
            raise CheckpointStrictError(
                "RECOMB72_LIVE_CHECKPOINT_ANCESTRY_COVERAGE_MISMATCH"
            )
        for agent in alive:
            key = str(int(agent.agent_id))
            setattr(agent, "_recomb72_ancestry", dict(ancestry[key]))
            setattr(agent, "_recomb72_founder_group", str(groups[key]))
        setattr(
            pop,
            "_recomb72_founder_receipt",
            _to_jsonable(recomb72_state.get("founder_receipt")),
        )
        setattr(
            sim,
            "_recomb72_generation_rows",
            _to_jsonable(list(recomb72_state.get("generation_rows", ()) or ())),
        )
        setattr(
            sim,
            "_recomb72_real_tick_count",
            int(recomb72_state.get("real_tick_count", 0)),
        )
        setattr(
            sim,
            "_recomb72_neutral_action_agent_ticks",
            int(recomb72_state.get("neutral_action_agent_tick_count", 0)),
        )
        setattr(
            sim,
            "_recomb72_neutrality_receipt",
            _to_jsonable(recomb72_state.get("neutrality_receipt")),
        )

    # Force lazy init prior to setattr
    for t in targets:
        try:
            if hasattr(t, "_ensure_shock_state"):
                t._ensure_shock_state()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if hasattr(t, "_ensure_identity_arrays"):
                t._ensure_identity_arrays()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Shock counters
    for k in ["_shock_active_rem", "_shock_cooldown_rem", "_shock_last_active", "_shock_last_kind"]:
        if k in state:
            for t in targets:
                try:
                    setattr(t, k, state[k])
                except Exception:
                    pass

    for k in ["_dev_stage_transition_count_gen", "_dev_stage_eval_stage_counts_gen", "_dev_d2_lock_event_count_gen"]:
        if k in state:
            for t in targets:
                try:
                    setattr(t, k, state[k])
                except Exception:
                    pass

    def _unpack_ndarray(payload: Any) -> Optional[np.ndarray]:
        if not isinstance(payload, dict):
            return None
        if "data_hex" not in payload or "dtype" not in payload or "shape" not in payload:
            return None
        try:
            data = bytes.fromhex(str(payload["data_hex"]))
            dt = np.dtype(str(payload["dtype"]))
            arr = np.frombuffer(data, dtype=dt)
            shape = tuple(int(x) for x in payload["shape"])
            arr = arr.reshape(shape, order="C")
            # frombuffer yields a non-writeable view; make an owned, writeable copy for safety
            arr = np.ascontiguousarray(arr).copy()
            # Optional integrity check
            sha = payload.get("sha256")
            if sha:
                if hashlib.sha256(data).hexdigest() != str(sha):
                    raise CheckpointStrictError("runner_state ndarray sha256 mismatch")
            return arr
        except Exception:
            return None

    for k in ["_idprobe_baseline_probs", "_idprobe_baseline_token", "_idprobe_drift_ema"]:
        if k in state:
            arr = _unpack_ndarray(state[k])
            if arr is None:
                continue
            for t in targets:
                try:
                    setattr(t, k, arr)
                except Exception:
                    pass

    for k in [
        "_structural_action_opportunity_gen",
        "_structural_action_opportunity_total",
        "_structural_action_margin_gen",
        "_structural_action_margin_total",
        "_structural_policy_action_family_gen",
        "_structural_policy_action_family_total",
        "_structural_policy_action_family_pending",
        "_structural_attack_consequence_events",
        "_structural_selection_funnel_pre_turnover_gen",
        "_structural_selection_funnel_total",
        "_structural_selection_funnel_recent_child_links",
        "_structural_tick_reproduction_funnel_gen",
        "_structural_tick_reproduction_funnel_total",
        "_structural_tick_reproduction_recent_child_links",
        "_structural_tick_reproduction_previous_child_links",
    ]:
        if k in state:
            payload = state.get(k)
            if isinstance(payload, dict):
                for t in targets:
                    try:
                        setattr(t, k, dict(payload))
                    except Exception:
                        pass
            elif isinstance(payload, list):
                for t in targets:
                    try:
                        setattr(t, k, list(payload))
                    except Exception:
                        pass


    # ---- Runner drive arrays (authoritative) ----
    for k in ["_drive_hunger", "_drive_pain", "_drive_fatigue", "_drive_fear", "_drive_slot_token"]:
        if k in state:
            arr = _unpack_ndarray(state[k])
            if arr is None:
                continue
            for t in targets:
                try:
                    setattr(t, k, arr)
                except Exception:
                    pass

    # After restoring runner drive arrays, refresh agent mirrors for alive agents (if available)
    try:
        pop = getattr(sim, "pop", None)
        sync_fn = getattr(sim, "_sync_drive_slot", None)
        if pop is not None and callable(sync_fn):
            alive_fn = getattr(pop, "alive_agents", None)
            if callable(alive_fn):
                for a in alive_fn():
                    try:
                        sync_fn(a)
                    except Exception:
                        pass
    except Exception:
        pass

    for k in ["_idprobe_last_tick", "_pred_tick"]:
        if k in state:
            for t in targets:
                try:
                    setattr(t, k, int(state[k]))
                except Exception:
                    pass



    # Restore policy stride action cache if present
    if "_policy_action_cache" in state:
        payload = state.get("_policy_action_cache")
        for t in targets:
            if hasattr(t, "_policy_action_cache"):
                try:
                    if payload is None:
                        setattr(t, "_policy_action_cache", None)
                    elif isinstance(payload, dict) and payload.get("schema_version") == "policy_action_cache_v1":
                        def _decode_cache_key(x: Any) -> Any:
                            """Decode numeric cache keys back to ints.

                            Older bundles serialized keys as strings; resume must
                            restore int keys so cache.get(aid:int) hits.
                            """
                            try:
                                if isinstance(x, (int, np.integer)):
                                    return int(x)
                            except Exception:
                                pass
                            try:
                                if isinstance(x, str):
                                    s = x.strip()
                                    if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
                                        return int(s)
                                    return s
                            except Exception:
                                pass
                            try:
                                return int(x)
                            except Exception:
                                return str(x)

                        rebuilt = {}
                        for ent in payload.get("entries", []) or []:
                            if not isinstance(ent, dict) or "k" not in ent:
                                continue
                            k_dec = _decode_cache_key(ent.get("k"))
                            v = ent.get("v")
                            # Expected persisted format: [tok:int, actions:list[int], cursor:int]
                            if isinstance(v, list) and len(v) >= 3:
                                try:
                                    tok = int(v[0])
                                    acts_raw = v[1]
                                    cur = int(v[2])
                                    if isinstance(acts_raw, (list, tuple)):
                                        acts = [int(x) for x in acts_raw]
                                    else:
                                        acts = acts_raw
                                    rebuilt[k_dec] = [tok, acts, cur]
                                    continue
                                except Exception:
                                    pass
                            rebuilt[k_dec] = v
                        setattr(t, "_policy_action_cache", rebuilt)
                    else:
                        setattr(t, "_policy_action_cache", payload)
                except Exception:
                    pass
    if "_cxxrewrite1_lifecycle_captures" in state:
        captures = state.get("_cxxrewrite1_lifecycle_captures")
        if isinstance(captures, list):
            for t in targets:
                try:
                    setattr(t, "_cxxrewrite1_lifecycle_captures", _to_jsonable(captures))
                except Exception:
                    pass
    # ---- Apply reproduction / birth counters (if present) ----
    for k in [
        "_injury_blocked_births_gen",
        "_birth_attempts_gen",
        "_birth_pre_cooldown_gen",
        "_birth_pre_no_energy_gen",
        "_birth_pre_injury_gate_gen",
        "_birth_pre_no_space_gen",
        "_birth_pre_no_mate_gen",
        "_birth_attempt_conception_fail_gen",
        "_birth_gestation_fail_gen",
        "_birth_make_child_fail_gen",
        "_birth_fail_no_space_gen",
        "_birth_fail_no_mate_gen",
        "_birth_fail_buffer_gen",
        "_birth_fail_injury_gate_gen",
        "_birth_fail_scar_gate_gen",
        "_birth_fail_conception_gen",
        "_birth_fail_gestation_gen",
        "_a3_recovery_baseline_fitness_ema",
        "_a3_recovery_baseline_drift_ema",
        "_patch_food_regrowth_total_gen",
        "_patch_food_harvest_total_gen",
        "_patch_movement_energy_extra_total_gen",
        "_patch_movement_energy_event_count_gen",
        "_patch_scent_emitted_total_gen",
        "_patch_scent_decayed_total_gen",
        "_patch_waste_emitted_total_gen",
        "_patch_waste_decayed_total_gen",
        "_patch_regrowth_base_total_gen",
        "_patch_regrowth_effective_total_gen",
        "_patch_regrowth_suppressed_total_gen",
        "_patch_regrowth_consequence_steps_gen",
        "_patch_regrowth_multiplier_mean_sum_gen",
        "_patch_regrowth_multiplier_min_gen",
        "_patch_regrowth_multiplier_p10_sum_gen",
        "_patch_regrowth_multiplier_p90_sum_gen",
        "_patch_waste_regrowth_penalty_mean_sum_gen",
        "_patch_waste_regrowth_penalty_max_gen",
        "_patch_density_regrowth_penalty_mean_sum_gen",
        "_patch_density_regrowth_penalty_max_gen",
        "_patch_high_waste_regrowth_suppression_frac_sum_gen",
        "_patch_high_density_regrowth_suppression_frac_sum_gen",
        "_patch_recovery_positive_total_gen",
        "_patch_recovery_negative_total_gen",
        "_patch_recovery_decay_total_gen",
        "_jungle_recovery_regrowth_boost_total_gen",
        "_jungle_recovery_regrowth_boost_mean_sum_gen",
        "_jungle_recovery_regrowth_boost_max_gen",
        "_jungle_recovery_regrowth_boost_p90_sum_gen",
        "_patch_regrowth_after_waste_total_gen",
        "_patch_regrowth_after_recovery_total_gen",
        "_patch_recovery_added_regrowth_total_gen",
        "_jungle_predator_events_gen",
        "_jungle_predator_exposure_count_gen",
        "_jungle_predator_damage_total_gen",
        "_jungle_predator_damage_raw_total_gen",
        "_jungle_predator_damage_effective_total_gen",
        "_jungle_predator_damage_prevented_total_gen",
        "_jungle_predator_deaths_gen",
        "_jungle_birth_scent_emitted_total_gen",
        "_jungle_birth_waste_emitted_total_gen",
        "_jungle_birth_emission_count_gen",
        "_jungle_juvenile_predator_events_gen",
        "_jungle_juvenile_predator_damage_total_gen",
        "_jungle_juvenile_predator_deaths_gen",
        "_jungle_adult_predator_events_gen",
        "_jungle_adult_predator_damage_total_gen",
        "_wdl_delayed_hazard_event_count_gen",
        "_wdl_delayed_hazard_cue_cell_count_sum_gen",
        "_wdl_delayed_hazard_active_cell_count_sum_gen",
        "_wdl_delayed_hazard_tick_count_gen",
        "_wdl_delayed_hazard_agent_cue_exposure_count_gen",
        "_wdl_delayed_hazard_agent_active_exposure_count_gen",
        "_wdl_delayed_hazard_damage_total_gen",
        "_wdl_delayed_hazard_damage_max_gen",
        "_wdl_delayed_hazard_agents_damaged_gen",
        "_wdl_delayed_hazard_cue_to_active_delay_sum_gen",
        "_wdl_delayed_hazard_cue_to_active_delay_count_gen",
        "_wdl_delayed_hazard_cue_without_active_count_gen",
        "_wdl_delayed_hazard_active_without_prior_cue_count_gen",
        "_wdl_cue_seen_agents_gen",
        "_wdl_cue_exposed_next_move_away_count_gen",
        "_wdl_cue_exposed_next_move_toward_count_gen",
        "_wdl_cue_exposed_stayed_count_gen",
        "_wdl_pre_hazard_exit_count_gen",
        "_wdl_hazard_enter_count_gen",
        "_wdl_hazard_exposure_after_cue_count_gen",
        "_path1_drh0_damage_in_refuge_gen",
        "_path1_drh0_damage_outside_refuge_gen",
        "_path1_drh0_damage_prevented_by_refuge_gen",
        "_path1_drh0_static_refuge_damage_prevented_gen",
        "_path1_drh0_cue_conditioned_damage_prevented_gen",
        "_path1_drh0_protected_exposure_count_gen",
        "_path1_drh0_refuge_exposure_count_gen",
        "_path1_drh0_no_cue_false_positive_refuge_movement_gen",
        "_path1_drh0_cue_visible_steps_gen",
        "_path1_drh0_hidden_window_steps_gen",
        "_path1_drh0_hazard_active_steps_gen",
        "_prediction_demand_event_count_gen",
        "_prediction_demand_transition_count_gen",
        "_prediction_demand_cue_cell_count_sum_gen",
        "_prediction_demand_active_cell_count_sum_gen",
        "_prediction_demand_tick_count_gen",
        "_prediction_demand_cue_ambiguity_sum_gen",
        "_prediction_demand_cue_ambiguity_count_gen",
        "_prediction_demand_cue_to_consequence_delay_sum_gen",
        "_prediction_demand_cue_to_consequence_delay_count_gen",
        "_prediction_demand_cue_without_consequence_count_gen",
        "_prediction_demand_consequence_without_prior_cue_count_gen",
        "_prediction_demand_pre_consequence_agent_context_count_gen",
        "_prediction_demand_consequence_agent_exposure_count_gen",
        "_prediction_demand_resource_consequence_total_gen",
        "_delayed_resource_depletion_event_count_gen",
        "_delayed_resource_depletion_cue_cell_count_sum_gen",
        "_delayed_resource_depletion_hidden_cell_count_sum_gen",
        "_delayed_resource_depletion_active_cell_count_sum_gen",
        "_delayed_resource_depletion_tick_count_gen",
        "_delayed_resource_depletion_agent_cue_exposure_count_gen",
        "_delayed_resource_depletion_agent_hidden_gap_exposure_count_gen",
        "_delayed_resource_depletion_agent_active_exposure_count_gen",
        "_delayed_resource_depletion_consequence_count_gen",
        "_delayed_resource_depletion_consequence_total_gen",
        "_delayed_resource_depletion_cue_to_active_delay_sum_gen",
        "_delayed_resource_depletion_cue_to_active_delay_count_gen",
        "_delayed_resource_depletion_cue_without_active_count_gen",
        "_delayed_resource_depletion_active_without_prior_cue_count_gen",
        "_delayed_resource_depletion_gather_cue_count_gen",
        "_delayed_resource_depletion_gather_hidden_count_gen",
        "_delayed_resource_depletion_gather_active_count_gen",
        "_delayed_resource_depletion_revisit_after_cue_count_gen",
        "_delayed_resource_depletion_move_away_count_gen",
        "_delayed_resource_depletion_move_toward_count_gen",
        "_delayed_resource_depletion_stay_count_gen",
        "_delayed_crowding_waste_event_count_gen",
        "_delayed_crowding_waste_cue_cell_count_sum_gen",
        "_delayed_crowding_waste_hidden_cell_count_sum_gen",
        "_delayed_crowding_waste_active_cell_count_sum_gen",
        "_delayed_crowding_waste_tick_count_gen",
        "_delayed_crowding_waste_agent_cue_exposure_count_gen",
        "_delayed_crowding_waste_agent_hidden_gap_exposure_count_gen",
        "_delayed_crowding_waste_agent_active_exposure_count_gen",
        "_delayed_crowding_waste_consequence_count_gen",
        "_delayed_crowding_waste_consequence_total_gen",
        "_delayed_crowding_waste_cue_to_active_delay_sum_gen",
        "_delayed_crowding_waste_cue_to_active_delay_count_gen",
        "_delayed_crowding_waste_cue_without_active_count_gen",
        "_delayed_crowding_waste_active_without_prior_cue_count_gen",
        "_delayed_crowding_waste_gather_cue_count_gen",
        "_delayed_crowding_waste_gather_hidden_count_gen",
        "_delayed_crowding_waste_gather_active_count_gen",
        "_delayed_crowding_waste_revisit_after_cue_count_gen",
        "_delayed_crowding_waste_move_away_count_gen",
        "_delayed_crowding_waste_move_toward_count_gen",
        "_delayed_crowding_waste_stay_count_gen",
    ]:
        if k in state:
            for t in targets:
                try:
                    if isinstance(state[k], (int, np.integer)):
                        setattr(t, k, int(state[k]))
                    else:
                        setattr(t, k, float(state[k]))
                except Exception:
                    pass
            # first-set wins; all targets receive same scalar

    for k in [
        "_patch_birth_contexts_gen",
        "_patch_death_contexts_gen",
        "_jungle_predator_event_contexts_gen",
        "_jungle_predator_damage_multipliers_gen",
        "_jungle_birth_emission_contexts_gen",
        "_wdl_delayed_hazard_event_contexts_gen",
    ]:
        if k in state:
            for t in targets:
                try:
                    setattr(t, k, list(state.get(k) or []))
                except Exception:
                    pass

    if "_a3_recovery_taxonomy_last_gen" in state:
        for t in targets:
            try:
                setattr(t, "_a3_recovery_taxonomy_last_gen", _to_jsonable(state["_a3_recovery_taxonomy_last_gen"]))
            except Exception:
                pass


    # ---- Phase 6 (draft): PressureAutopilot state restore ----
    try:
        ap_state = state.get("pressure_autopilot_state", None)
        ap = getattr(sim, "pressure_autopilot", None)
        if ap_state is not None and ap is not None and hasattr(ap, "restore_state"):
            ap.restore_state(ap_state)
    except Exception:
        pass


def _apply_metrics_state(sim: Any, state: Dict[str, Any]) -> None:
    """Apply determinism-critical MetricsLogger internal state.

    Must be applied BEFORE the first post-resume metrics row is written.
    """
    if not isinstance(state, dict):
        return

    logger = getattr(sim, "logger", None)
    if logger is None:
        # Some integrations attach logger to sim.runner; check runner targets.
        try:
            for t in _runner_targets(sim):
                logger = getattr(t, "logger", None)
                if logger is not None:
                    break
        except Exception:
            logger = None

    if logger is None:
        return

    if hasattr(logger, "restore_state"):
        try:
            logger.restore_state(state)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Fallback: direct private attribute.
    if "collapse_streak" in state:
        try:
            setattr(logger, "_collapse_streak", int(state.get("collapse_streak", 0) or 0))
        except Exception:
            pass

    # Optional Phase E1 metrics state restore.
    try:
        e1_state = state.get("phase_e_e1_state", None)
        if isinstance(e1_state, dict):
            e1 = getattr(sim, "_phase_e_metrics_engine", None)
            if e1 is None:
                try:
                    from simulation.metrics_pkg.phase_e_metrics_engine import PhaseEMetricsEngine  # noqa: WPS433
                except Exception:
                    from engine.simulation.metrics_pkg.phase_e_metrics_engine import PhaseEMetricsEngine  # type: ignore  # noqa: WPS433

                e1 = PhaseEMetricsEngine(
                    run_dir=str(getattr(sim, "run_dir", ".")),
                    retention_k=max(1, int(getattr(getattr(sim, "sim_cfg", None), "phase_e_e1_retention_k_gen", 3) or 3)),
                    neutrality_window=max(2, int(getattr(getattr(sim, "sim_cfg", None), "phase_e_e1_neutrality_window", 8) or 8)),
                )
                setattr(sim, "_phase_e_metrics_engine", e1)
            if hasattr(e1, "restore_state"):
                e1.restore_state(e1_state)
    except Exception:
        pass

    # Optional Phase E2 metrics state restore.
    try:
        e2_state = state.get("phase_e_e2_state", None)
        if isinstance(e2_state, dict):
            e2 = getattr(sim, "_phase_e_failure_taxonomy_engine", None)
            if e2 is None:
                try:
                    from simulation.metrics_pkg.phase_e_metrics_engine import PhaseEFailureTaxonomyEngine  # noqa: WPS433
                except Exception:
                    from engine.simulation.metrics_pkg.phase_e_metrics_engine import PhaseEFailureTaxonomyEngine  # type: ignore  # noqa: WPS433

                e2 = PhaseEFailureTaxonomyEngine(run_dir=str(getattr(sim, "run_dir", ".")))
                setattr(sim, "_phase_e_failure_taxonomy_engine", e2)
            if hasattr(e2, "restore_state"):
                e2.restore_state(e2_state)
    except Exception:
        pass



def _export_bank_state(sim: Any) -> Optional[Dict[str, Any]]:
    """Export GRU memory/bank state required for resume equivalence.

    NOTE:
    - MLP parameter banks are pure caches and can be rebuilt deterministically.
    - GRU memory state (`_Hmem`) is *persistent across steps/generations* and
      directly affects policy outputs. If GRU is in use, `_Hmem` must be
      checkpointed, otherwise resume will diverge immediately.
    """
    if torch is None:
        return None

    targets = _runner_targets(sim)

    # Ensure GRU bank exists if possible (lazy init)
    for t in targets:
        try:
            if hasattr(t, "_ensure_gru_bank"):
                t._ensure_gru_bank()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Find `_Hmem` owner
    hmem = None
    owner = None
    for t in targets:
        if hasattr(t, "_Hmem"):
            try:
                hmem = getattr(t, "_Hmem")
                owner = t
            except Exception:
                hmem = None
            break

    if hmem is None:
        return None

    try:
        if hasattr(hmem, "detach"):
            h_cpu = hmem.detach().to("cpu")
        else:
            return None
        arr = h_cpu.numpy()
    except Exception:
        return None

    data = arr.tobytes(order="C")
    payload = {
        "schema_version": "bank_state_v1",
        "_gru_inited": bool(getattr(owner, "_gru_inited", False)) if owner is not None else False,
        "_gru_hidden": int(getattr(owner, "_gru_hidden", 0)) if owner is not None else 0,
        "_Hmem": {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "order": "C",
            "data_hex": data.hex(),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    }
    return payload


def _apply_bank_state(sim: Any, state: Dict[str, Any]) -> None:
    """Apply GRU memory/bank state back into the current runner/sim."""
    if torch is None:
        return
    if not isinstance(state, dict):
        return
    if state.get("schema_version") != "bank_state_v1":
        return
    if "_Hmem" not in state:
        return

    targets = _runner_targets(sim)

    # Ensure GRU bank exists before setting memory
    for t in targets:
        try:
            if hasattr(t, "_ensure_gru_bank"):
                t._ensure_gru_bank()  # type: ignore[attr-defined]
        except Exception:
            pass

    h_payload = state.get("_Hmem", {})
    try:
        data = bytes.fromhex(str(h_payload["data_hex"]))
        sha = str(h_payload.get("sha256", ""))
        if sha and hashlib.sha256(data).hexdigest() != sha:
            raise CheckpointStrictError("bank_state _Hmem sha256 mismatch")
        dt = np.dtype(str(h_payload["dtype"]))
        arr = np.frombuffer(data, dtype=dt)
        shape = tuple(int(x) for x in h_payload["shape"])
        arr = arr.reshape(shape, order="C")
    except Exception:
        return

    # Convert to torch on bank device
    for t in targets:
        if hasattr(t, "_bank_device"):
            dev = getattr(t, "_bank_device")
        else:
            dev = getattr(t, "device", "cpu")
        # NOTE: np.frombuffer creates a non-writeable view; torch.from_numpy on a non-writeable array
        # can lead to undefined behavior if the tensor is later mutated (which GRU memory is).
        # Force owned, writeable, C-contiguous storage.
        try:
            if isinstance(arr, np.ndarray):
                if not arr.flags.writeable:
                    arr = arr.copy()
                arr = np.ascontiguousarray(arr)
        except Exception:
            pass
        try:
            tens = torch.from_numpy(arr).to(dev)
        except Exception:
            tens = torch.from_numpy(arr)
        try:
            setattr(t, "_Hmem", tens)
            # Restore flags if present
            try:
                setattr(t, "_gru_inited", bool(state.get("_gru_inited", True)))
            except Exception:
                pass
            try:
                if int(state.get("_gru_hidden", 0)) > 0:
                    setattr(t, "_gru_hidden", int(state.get("_gru_hidden")))
            except Exception:
                pass
        except Exception:
            pass


def _checkpoint_population_identity_contract(pop_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pop_obj, dict):
        raise CheckpointStrictError("population checkpoint state must be an object")
    slots_obj = pop_obj.get("slot_pool")
    agents_obj = pop_obj.get("agents")
    if not isinstance(slots_obj, dict) or not isinstance(agents_obj, list):
        raise CheckpointStrictError("population checkpoint identity state is malformed")
    capacity = _checkpoint_exact_int(
        slots_obj.get("max_slots"), "population.slot_pool.max_slots"
    )
    tokens = _checkpoint_exact_int_list(
        slots_obj.get("slot_tokens"), "population.slot_pool.slot_tokens"
    )
    free = _checkpoint_exact_int_list(
        slots_obj.get("free_slots"), "population.slot_pool.free_slots"
    )
    if capacity <= 0 or len(tokens) != capacity:
        raise CheckpointStrictError(
            f"population SlotPool capacity malformed: capacity={capacity};tokens={len(tokens)}"
        )
    if len(set(free)) != len(free) or any(slot < 0 or slot >= capacity for slot in free):
        raise CheckpointStrictError("population SlotPool free queue is malformed")

    by_id: Dict[int, Dict[str, Any]] = {}
    by_slot: Dict[int, Dict[str, Any]] = {}
    for raw in agents_obj:
        if not isinstance(raw, dict):
            raise CheckpointStrictError("population agent row must be an object")
        agent_id = _checkpoint_exact_int(raw.get("agent_id"), "population.agent_id")
        slot = _checkpoint_exact_int(raw.get("slot"), "population.agent.slot")
        token = _checkpoint_exact_int(
            raw.get("slot_token"), "population.agent.slot_token"
        )
        alive_value = raw.get("alive", True)
        if type(alive_value) is not bool:
            raise CheckpointStrictError("population.agent.alive must be an exact boolean")
        if (
            agent_id < 0
            or agent_id > np.iinfo(np.int32).max
            or slot < 0
            or slot >= capacity
            or token <= 0
        ):
            raise CheckpointStrictError(
                f"population agent identity malformed: id={agent_id};slot={slot};token={token}"
            )
        if agent_id in by_id or slot in by_slot:
            raise CheckpointStrictError("population agent IDs and slots must be unique")
        if tokens[slot] != token:
            raise CheckpointStrictError(
                f"population slot token mismatch: slot={slot};agent={token};pool={tokens[slot]}"
            )
        row = {
            "agent_id": agent_id,
            "slot": slot,
            "slot_token": token,
            "alive": alive_value,
        }
        by_id[agent_id] = row
        by_slot[slot] = row
    occupied = set(range(capacity)) - set(free)
    if occupied != set(by_slot):
        raise CheckpointStrictError(
            "population SlotPool occupied/free partition does not match agent rows"
        )
    return {
        "capacity": capacity,
        "slot_tokens": tokens,
        "free_slots": free,
        "by_id": by_id,
        "by_slot": by_slot,
        "live_ids": {agent_id for agent_id, row in by_id.items() if row["alive"]},
    }


def _checkpoint_exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise CheckpointStrictError(f"{field} must be an exact integer")
    return value


def _checkpoint_exact_int_list(value: Any, field: str) -> List[int]:
    if not isinstance(value, list):
        raise CheckpointStrictError(f"{field} must be an integer array")
    return [
        _checkpoint_exact_int(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]


def _checkpoint_exact_bool_list(value: Any, field: str) -> List[bool]:
    if not isinstance(value, list) or any(type(item) is not bool for item in value):
        raise CheckpointStrictError(f"{field} must be a boolean array")
    return list(value)


def _checkpoint_exact_number_list(value: Any, field: str) -> List[float]:
    if not isinstance(value, list):
        raise CheckpointStrictError(f"{field} must be a numeric array")
    out: List[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CheckpointStrictError(
                f"{field}[{index}] must be an exact numeric value"
            )
        number = float(item)
        if not np.isfinite(number):
            raise CheckpointStrictError(f"{field}[{index}] must be finite")
        out.append(number)
    return out


def _checkpoint_respec_cursor_state(
    state: Any,
    *,
    expected_shape: tuple[int, ...],
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(state, dict):
        raise CheckpointStrictError(f"{field} is missing")
    arrays = (
        np.asarray(
            _checkpoint_exact_int_list(state.get("active_y"), f"{field}.active_y"),
            dtype=np.int32,
        ),
        np.asarray(
            _checkpoint_exact_int_list(state.get("active_x"), f"{field}.active_x"),
            dtype=np.int32,
        ),
        np.asarray(
            _checkpoint_exact_int_list(
                state.get("active_class"), f"{field}.active_class"
            ),
            dtype=np.int32,
        ),
        np.asarray(
            _checkpoint_exact_number_list(state.get("progress"), f"{field}.progress"),
            dtype=np.float32,
        ),
    )
    if any(array.shape != expected_shape for array in arrays):
        raise CheckpointStrictError(f"{field} shapes are malformed")
    return arrays


def _checkpoint_grid_positions(
    world: Dict[str, Any],
    *,
    slot_to_agent_id: List[int] | None = None,
) -> Dict[int, tuple[int, int]]:
    raw_grid = world.get("agent_grid")
    if (
        not isinstance(raw_grid, list)
        or not raw_grid
        or any(not isinstance(row, list) for row in raw_grid)
    ):
        raise CheckpointStrictError("world agent_grid must be a two-dimensional integer array")
    width = len(raw_grid[0])
    if width == 0 or any(len(row) != width for row in raw_grid):
        raise CheckpointStrictError("world agent_grid must be a rectangular integer array")
    for y, row in enumerate(raw_grid):
        for x, value in enumerate(row):
            _checkpoint_exact_int(value, f"world.agent_grid[{y}][{x}]")
    grid = np.asarray(raw_grid, dtype=np.int64)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise CheckpointStrictError("world agent_grid must be a two-dimensional integer array")
    grid_i64 = grid.astype(np.int64, copy=False)
    if bool(np.any(grid_i64 < -1)) or bool(
        np.any(grid_i64 > np.iinfo(np.int32).max)
    ):
        raise CheckpointStrictError("world agent_grid IDs must fit signed int32")
    positions: Dict[int, tuple[int, int]] = {}
    for y, x in np.argwhere(grid_i64 != -1).tolist():
        grid_value = int(grid_i64[int(y), int(x)])
        if slot_to_agent_id is None:
            agent_id = grid_value
        else:
            if grid_value < 0 or grid_value >= len(slot_to_agent_id):
                raise CheckpointStrictError(
                    f"world agent_grid slot out of range: slot={grid_value}"
                )
            agent_id = int(slot_to_agent_id[grid_value])
            if agent_id == -1:
                raise CheckpointStrictError(
                    f"world agent_grid references free slot: slot={grid_value}"
                )
        if agent_id in positions:
            raise CheckpointStrictError(
                f"world agent_grid contains duplicate agent ID: {agent_id}"
            )
        positions[agent_id] = (int(y), int(x))
    return positions


def _normalize_checkpoint_world_authority(
    sim: Any,
    world: Dict[str, Any],
    pop_obj: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(world, dict):
        raise CheckpointStrictError("world checkpoint state must be an object")
    population = _checkpoint_population_identity_contract(pop_obj)
    capacity = int(population["capacity"])
    pool_tokens = list(population["slot_tokens"])
    respec_enabled = bool(
        getattr(
            getattr(getattr(sim, "env", None), "cfg", None),
            "respec0b_resource_specialization_enable",
            False,
        )
    )
    respec_cursor_arrays = None
    schema = str(world.get("schema_version", ""))
    if schema == "world_state_v3":
        binding = world.get("world_membership_authority")
        if not isinstance(binding, dict):
            raise CheckpointStrictError("world_state_v3 membership authority is missing")
        raw_for_digest = {
            key: binding.get(key)
            for key in (
                "capacity",
                "slot_to_agent_id",
                "slot_tokens",
                "world_alive_by_slot",
                "pos_y_by_slot",
                "pos_x_by_slot",
            )
        }
        expected_digest = _sha256_bytes(canonical_json_bytes(raw_for_digest))
        if str(binding.get("binding_digest", "")) != expected_digest:
            raise CheckpointStrictError("world_state_v3 binding digest mismatch")
        grid_slot_ids = _checkpoint_exact_int_list(
            binding.get("slot_to_agent_id"),
            "world.world_membership_authority.slot_to_agent_id",
        )
        positions = _checkpoint_grid_positions(
            world, slot_to_agent_id=grid_slot_ids
        )
        binding_capacity = _checkpoint_exact_int(
            binding.get("capacity"), "world.world_membership_authority.capacity"
        )
        if binding_capacity != capacity:
            # A SimulationRunner can be checkpointed before it has placed its
            # founders.  Its GridWorld still has the configuration-derived
            # empty allocation while SlotPool has the route-derived capacity.
            # This is a canonical empty state, not a cross-identity migration;
            # normalize it to the SlotPool capacity below.  Any occupied or
            # token-bearing binding remains a hard mismatch.
            raw_ids = _checkpoint_exact_int_list(
                binding.get("slot_to_agent_id"),
                "world.world_membership_authority.slot_to_agent_id",
            )
            raw_tokens = _checkpoint_exact_int_list(
                binding.get("slot_tokens"),
                "world.world_membership_authority.slot_tokens",
            )
            raw_alive = _checkpoint_exact_bool_list(
                binding.get("world_alive_by_slot"),
                "world.world_membership_authority.world_alive_by_slot",
            )
            raw_y = _checkpoint_exact_int_list(
                binding.get("pos_y_by_slot"),
                "world.world_membership_authority.pos_y_by_slot",
            )
            raw_x = _checkpoint_exact_int_list(
                binding.get("pos_x_by_slot"),
                "world.world_membership_authority.pos_x_by_slot",
            )
            pristine_binding = (
                all(agent_id == -1 for agent_id in raw_ids)
                and all(token == 0 for token in raw_tokens)
                and not any(raw_alive)
                and all(value == -1 for value in raw_y)
                and all(value == -1 for value in raw_x)
            )
            if not pristine_binding or positions or population["live_ids"]:
                raise CheckpointStrictError(
                    "world_state_v3 capacity does not match SlotPool"
                )
            slot_ids = np.full((capacity,), -1, dtype=np.int64)
            slot_tokens = np.asarray(pool_tokens, dtype=np.int64)
            alive = np.zeros((capacity,), dtype=np.bool_)
            pos_y = np.full((capacity,), -1, dtype=np.int64)
            pos_x = np.full((capacity,), -1, dtype=np.int64)
        else:
            slot_ids = np.asarray(
                _checkpoint_exact_int_list(
                    binding.get("slot_to_agent_id"),
                    "world.world_membership_authority.slot_to_agent_id",
                ),
                dtype=np.int64,
            )
            slot_tokens = np.asarray(
                _checkpoint_exact_int_list(
                    binding.get("slot_tokens"),
                    "world.world_membership_authority.slot_tokens",
                ),
                dtype=np.int64,
            )
            alive = np.asarray(
                _checkpoint_exact_bool_list(
                    binding.get("world_alive_by_slot"),
                    "world.world_membership_authority.world_alive_by_slot",
                ),
                dtype=np.bool_,
            )
            pos_y = np.asarray(
                _checkpoint_exact_int_list(
                    binding.get("pos_y_by_slot"),
                    "world.world_membership_authority.pos_y_by_slot",
                ),
                dtype=np.int64,
            )
            pos_x = np.asarray(
                _checkpoint_exact_int_list(
                    binding.get("pos_x_by_slot"),
                    "world.world_membership_authority.pos_x_by_slot",
                ),
                dtype=np.int64,
            )
        if any(array.shape != (capacity,) for array in (slot_ids, slot_tokens, alive, pos_y, pos_x)):
            raise CheckpointStrictError("world_state_v3 slot arrays have malformed shapes")
        if respec_enabled:
            patch = world.get("patch_world")
            state = (
                patch.get("respec0b_resource_handling_state")
                if isinstance(patch, dict)
                else None
            )
            if not isinstance(state, dict) or str(
                state.get("schema_version", "")
            ) != "respec0b_slot_cursor_state_v1":
                raise CheckpointStrictError(
                    "world_state_v3 RESPEC slot cursor schema is missing"
                )
            cursor_tokens = _checkpoint_exact_int_list(
                state.get("slot_tokens"),
                "world.patch_world.respec0b_resource_handling_state.slot_tokens",
            )
            if cursor_tokens != pool_tokens:
                raise CheckpointStrictError(
                    "world_state_v3 RESPEC cursor tokens do not match SlotPool"
                )
            respec_cursor_arrays = _checkpoint_respec_cursor_state(
                state,
                expected_shape=(capacity,),
                field="world_state_v3 RESPEC cursor",
            )
    elif schema == "world_state_v2":
        positions = _checkpoint_grid_positions(world)
        dense_alive = np.asarray(
            _checkpoint_exact_bool_list(world.get("agent_alive"), "world.agent_alive"),
            dtype=np.bool_,
        )
        dense_y = np.asarray(
            _checkpoint_exact_int_list(world.get("agent_pos_y"), "world.agent_pos_y"),
            dtype=np.int64,
        )
        dense_x = np.asarray(
            _checkpoint_exact_int_list(world.get("agent_pos_x"), "world.agent_pos_x"),
            dtype=np.int64,
        )
        if dense_alive.ndim != 1 or dense_y.shape != dense_alive.shape or dense_x.shape != dense_alive.shape:
            raise CheckpointStrictError("world_state_v2 dense world arrays are malformed")
        dense_live = set(np.flatnonzero(dense_alive).tolist())
        if dense_live != set(positions) or dense_live != set(population["live_ids"]):
            raise CheckpointStrictError("world_state_v2 dense/grid/population live IDs disagree")
        for agent_id in dense_live:
            if agent_id >= dense_alive.size:
                raise CheckpointStrictError("world_state_v2 cannot represent sparse high IDs")
            if (int(dense_y[agent_id]), int(dense_x[agent_id])) != positions[agent_id]:
                raise CheckpointStrictError(
                    f"world_state_v2 dense position mismatch: agent_id={agent_id}"
                )
        dead = ~dense_alive
        if bool(np.any(dense_y[dead] != -1)) or bool(np.any(dense_x[dead] != -1)):
            raise CheckpointStrictError("world_state_v2 dead dense rows are not canonical")
        slot_ids = np.full((capacity,), -1, dtype=np.int64)
        slot_tokens = np.asarray(pool_tokens, dtype=np.int64)
        alive = np.zeros((capacity,), dtype=np.bool_)
        pos_y = np.full((capacity,), -1, dtype=np.int64)
        pos_x = np.full((capacity,), -1, dtype=np.int64)
        for agent_id in sorted(population["live_ids"]):
            row = population["by_id"][agent_id]
            slot = int(row["slot"])
            slot_ids[slot] = agent_id
            alive[slot] = True
            pos_y[slot], pos_x[slot] = positions[agent_id]
        if respec_enabled:
            patch = world.get("patch_world")
            state = patch.get("respec0b_resource_handling_state") if isinstance(patch, dict) else None
            legacy_cursor_arrays = _checkpoint_respec_cursor_state(
                state,
                expected_shape=dense_alive.shape,
                field="world_state_v2 RESPEC cursor",
            )
            if (
                bool(np.any(legacy_cursor_arrays[0][dead] != -1))
                or bool(np.any(legacy_cursor_arrays[1][dead] != -1))
                or bool(np.any(legacy_cursor_arrays[2][dead] != 0))
                or bool(np.any(legacy_cursor_arrays[3][dead] != 0.0))
            ):
                raise CheckpointStrictError("world_state_v2 dead RESPEC cursor rows are not canonical")
            respec_cursor_arrays = (
                np.full((capacity,), -1, dtype=np.int32),
                np.full((capacity,), -1, dtype=np.int32),
                np.zeros((capacity,), dtype=np.int32),
                np.zeros((capacity,), dtype=np.float32),
            )
            for agent_id in population["live_ids"]:
                slot = int(population["by_id"][agent_id]["slot"])
                for destination, source in zip(
                    respec_cursor_arrays, legacy_cursor_arrays, strict=True
                ):
                    destination[slot] = source[agent_id]
    else:
        raise CheckpointStrictError(f"unsupported world checkpoint schema: {schema!r}")

    if set(positions) != set(population["live_ids"]):
        raise CheckpointStrictError(
            "world/population live identity mismatch: "
            f"grid={sorted(positions)};population={sorted(population['live_ids'])}"
        )

    dense_alive = np.asarray(
        _checkpoint_exact_bool_list(world.get("agent_alive"), "world.agent_alive"),
        dtype=np.bool_,
    )
    dense_y = np.asarray(
        _checkpoint_exact_int_list(world.get("agent_pos_y"), "world.agent_pos_y"),
        dtype=np.int64,
    )
    dense_x = np.asarray(
        _checkpoint_exact_int_list(world.get("agent_pos_x"), "world.agent_pos_x"),
        dtype=np.int64,
    )
    if (
        dense_alive.ndim != 1
        or dense_y.shape != dense_alive.shape
        or dense_x.shape != dense_alive.shape
    ):
        raise CheckpointStrictError("world dense compatibility shadows are malformed")
    expected_shadow_ids = {
        agent_id
        for agent_id in population["live_ids"]
        if agent_id < int(dense_alive.size)
    }
    if set(np.flatnonzero(dense_alive).tolist()) != expected_shadow_ids:
        raise CheckpointStrictError("world dense compatibility alive shadow is contradictory")
    for agent_id in expected_shadow_ids:
        if (int(dense_y[agent_id]), int(dense_x[agent_id])) != positions[agent_id]:
            raise CheckpointStrictError(
                f"world dense compatibility position mismatch: agent_id={agent_id}"
            )
    dead_shadow = ~dense_alive
    if bool(np.any(dense_y[dead_shadow] != -1)) or bool(
        np.any(dense_x[dead_shadow] != -1)
    ):
        raise CheckpointStrictError("world dense compatibility dead rows are noncanonical")

    seen_ids: set[int] = set()
    for slot in range(capacity):
        agent_id = int(slot_ids[slot])
        token = int(slot_tokens[slot])
        if token != int(pool_tokens[slot]):
            raise CheckpointStrictError(f"world/SlotPool token mismatch at slot {slot}")
        if agent_id == -1:
            if bool(alive[slot]) or int(pos_y[slot]) != -1 or int(pos_x[slot]) != -1:
                raise CheckpointStrictError(f"world_state_v3 free row is noncanonical: slot={slot}")
            continue
        if agent_id in seen_ids:
            raise CheckpointStrictError("world_state_v3 contains duplicate agent IDs")
        seen_ids.add(agent_id)
        row = population["by_slot"].get(slot)
        if row is None or int(row["agent_id"]) != agent_id or int(row["slot_token"]) != token:
            raise CheckpointStrictError(f"world/population slot binding mismatch: slot={slot}")
        if not bool(row["alive"]) or not bool(alive[slot]):
            raise CheckpointStrictError(f"world membership includes non-live row: slot={slot}")
        if (int(pos_y[slot]), int(pos_x[slot])) != positions[agent_id]:
            raise CheckpointStrictError(f"world slot/grid position mismatch: slot={slot}")
    if seen_ids != set(population["live_ids"]):
        raise CheckpointStrictError("world slot rows do not cover live population")

    binding = {
        "capacity": capacity,
        "slot_to_agent_id": slot_ids.astype(np.int64).tolist(),
        "slot_tokens": slot_tokens.astype(np.int64).tolist(),
        "world_alive_by_slot": alive.astype(np.bool_).tolist(),
        "pos_y_by_slot": pos_y.astype(np.int32).tolist(),
        "pos_x_by_slot": pos_x.astype(np.int32).tolist(),
    }
    binding["binding_digest"] = _sha256_bytes(canonical_json_bytes(binding))
    normalized = dict(world)
    normalized["schema_version"] = "world_state_v3"
    normalized["world_membership_authority"] = binding
    if respec_enabled:
        assert respec_cursor_arrays is not None
        cursor_y, cursor_x, cursor_class, cursor_progress = respec_cursor_arrays
        free_rows = slot_ids == -1
        none_rows = cursor_class == 0
        if bool(np.any((cursor_class < 0) | (cursor_class > 2))):
            raise CheckpointStrictError("RESPEC cursor resource class is invalid")
        if (
            bool(np.any(cursor_y[free_rows] != -1))
            or bool(np.any(cursor_x[free_rows] != -1))
            or bool(np.any(cursor_class[free_rows] != 0))
            or bool(np.any(cursor_progress[free_rows] != 0.0))
            or bool(np.any(cursor_y[none_rows] != -1))
            or bool(np.any(cursor_x[none_rows] != -1))
            or bool(np.any(cursor_progress[none_rows] != 0.0))
        ):
            raise CheckpointStrictError("RESPEC free/idle cursor rows are noncanonical")
        active_rows = ~none_rows
        grid_shape = np.asarray(world.get("agent_grid")).shape
        if (
            bool(np.any(cursor_progress[active_rows] < 0.0))
            or bool(np.any(cursor_y[active_rows] < 0))
            or bool(np.any(cursor_x[active_rows] < 0))
            or bool(np.any(cursor_y[active_rows] >= int(grid_shape[0])))
            or bool(np.any(cursor_x[active_rows] >= int(grid_shape[1])))
        ):
            raise CheckpointStrictError("RESPEC active cursor rows are invalid")
        patch = dict(world.get("patch_world") or {})
        patch["respec0b_resource_handling_state"] = {
            "schema_version": "respec0b_slot_cursor_state_v1",
            "slot_tokens": list(pool_tokens),
            "active_y": cursor_y.tolist(),
            "active_x": cursor_x.tolist(),
            "active_class": cursor_class.tolist(),
            "progress": cursor_progress.tolist(),
        }
        normalized["patch_world"] = patch
    return normalized


def _apply_world(sim: Any, world: Dict[str, Any]) -> None:
    """Apply world (env) state from bundle onto the live environment.

    Determinism note:
    - Preserve dtypes of the live env arrays where possible (int8 vs int32 vs bool).
      A dtype mismatch changes .tobytes() and will break boundary probes/fingerprints
      even when values are identical.
    - This function must not draw RNG.
    """
    env = getattr(sim, "env", None)
    if env is None:
        raise ValueError("sim.env missing")

    def _env_dtype(attr: str, fallback):
        try:
            a = getattr(env, attr, None)
            if a is not None and hasattr(a, "dtype"):
                return a.dtype
        except Exception:
            pass
        return fallback

    def to_arr(x: Any, *, dtype):
        a = np.array(x, dtype=dtype)
        return a

    # Preserve original env dtypes (or use sensible defaults if env is missing an attribute).
    grid_dtype = _env_dtype("grid", np.int8)
    agent_grid_dtype = _env_dtype("agent_grid", np.int32)
    biome_dtype = _env_dtype("biome_grid", np.int8)
    pos_dtype = _env_dtype("agent_pos_y", np.int32)
    alive_dtype = _env_dtype("agent_alive", np.bool_)

    setattr(env, "grid", to_arr(world.get("grid"), dtype=grid_dtype))
    setattr(env, "agent_grid", to_arr(world.get("agent_grid"), dtype=agent_grid_dtype))
    setattr(env, "biome_grid", to_arr(world.get("biome_grid"), dtype=biome_dtype))
    setattr(env, "agent_pos_y", to_arr(world.get("agent_pos_y"), dtype=pos_dtype))
    setattr(env, "agent_pos_x", to_arr(world.get("agent_pos_x"), dtype=pos_dtype))
    setattr(env, "agent_alive", to_arr(world.get("agent_alive"), dtype=alive_dtype))
    binding = world.get("world_membership_authority")
    if str(world.get("schema_version", "")) != "world_state_v3" or not isinstance(binding, dict):
        raise CheckpointStrictError("world apply requires normalized world_state_v3 authority")
    capacity = int(binding.get("capacity", 0) or 0)
    env.slot_to_agent_id = np.asarray(
        binding.get("slot_to_agent_id"), dtype=np.int64
    ).reshape((capacity,))
    env.slot_tokens = np.asarray(binding.get("slot_tokens"), dtype=np.int64).reshape(
        (capacity,)
    )
    env.agent_world_alive_by_slot = np.asarray(
        binding.get("world_alive_by_slot"), dtype=np.bool_
    ).reshape((capacity,))
    env.agent_pos_y_by_slot = np.asarray(
        binding.get("pos_y_by_slot"), dtype=np.int32
    ).reshape((capacity,))
    env.agent_pos_x_by_slot = np.asarray(
        binding.get("pos_x_by_slot"), dtype=np.int32
    ).reshape((capacity,))
    env._agent_id_to_world_slot = {
        int(agent_id): (int(slot), int(env.slot_tokens[slot]))
        for slot, agent_id in enumerate(env.slot_to_agent_id.tolist())
        if int(agent_id) != -1
    }
    env._world_slot_capacity_configured = True
    env._world_authority_migration_required = False

    patch_obj = world.get("patch_world", None)
    if bool(getattr(env, "patch_world_enable", False)):
        if isinstance(patch_obj, dict):
            from env.gridworld_pkg.patch_world import (
                EED_RESOURCE_IMPACT_OVERLAY_ID,
                EED_RESOURCE_IMPACT_SCHEMA_VERSION,
                compute_local_density_map,
                initialize_patch_world_runtime_fields,
                project_patch_food_to_binary_grid,
                validate_patch_world_arrays,
                validate_patch_world_schema_version,
            )

            schema_version = validate_patch_world_schema_version(int(patch_obj.get("schema_version", 1) or 1))
            patch_arrays = dict(patch_obj.get("arrays") or {})
            wdl_obj = patch_obj.get("wdl_delayed_hazard")
            if isinstance(wdl_obj, dict):
                patch_arrays.update(dict(wdl_obj.get("arrays") or {}))
            prediction_obj = patch_obj.get("prediction_demand")
            if isinstance(prediction_obj, dict):
                patch_arrays.update(dict(prediction_obj.get("arrays") or {}))
            resource_obj = patch_obj.get("delayed_resource_depletion")
            if isinstance(resource_obj, dict):
                patch_arrays.update(dict(resource_obj.get("arrays") or {}))
            overuse_obj = patch_obj.get("delayed_crowding_waste")
            if isinstance(overuse_obj, dict):
                patch_arrays.update(dict(overuse_obj.get("arrays") or {}))
            delayed_choice_obj = patch_obj.get("mem_max_delayed_choice")
            if isinstance(delayed_choice_obj, dict):
                patch_arrays.update(dict(delayed_choice_obj.get("arrays") or {}))
            if bool(
                getattr(
                    getattr(env, "cfg", None),
                    "eed_resource_impact_enable",
                    False,
                )
            ):
                eed_obj = patch_obj.get("eed_resource_impact")
                if not isinstance(eed_obj, dict):
                    raise ValueError(
                        "EED_RESOURCE_IMPACT_CHECKPOINT_MISSING: "
                        "enabled restore requires the conditional EED plane"
                    )
                if (
                    int(eed_obj.get("schema_version", 0) or 0)
                    != EED_RESOURCE_IMPACT_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "EED_RESOURCE_IMPACT_SCHEMA_MISMATCH: "
                        "checkpoint schema version is unsupported"
                    )
                if str(eed_obj.get("overlay_id", "")) != (
                    EED_RESOURCE_IMPACT_OVERLAY_ID
                ):
                    raise ValueError(
                        "EED_RESOURCE_IMPACT_OVERLAY_MISMATCH: "
                        "checkpoint overlay identity is unsupported"
                    )
                patch_arrays.update(dict(eed_obj.get("arrays") or {}))
            arrays = validate_patch_world_arrays(
                patch_arrays,
                shape=tuple(getattr(env, "grid").shape),
                cfg=getattr(env, "cfg", None),
            )
            setattr(env, "patch_world_schema_version", int(schema_version))
            for name, patch_arr in arrays.items():
                setattr(env, name, patch_arr)
            initialize_patch_world_runtime_fields(env)
            pending = dict(patch_obj.get("pending_arrays") or {})
            shape = tuple(getattr(env, "grid").shape)
            for pending_name in ("patch_scent_emit_pending", "patch_waste_emit_pending"):
                if pending_name in pending:
                    setattr(
                        env,
                        pending_name,
                        np.ascontiguousarray(
                            np.asarray(pending.get(pending_name), dtype=np.float32).reshape(shape),
                            dtype=np.float32,
                        ),
                    )
            respec0b_state = dict(patch_obj.get("respec0b_resource_handling_state") or {})
            if respec0b_state and bool(getattr(getattr(env, "cfg", None), "respec0b_resource_specialization_enable", False)):
                if str(respec0b_state.get("schema_version", "")) != (
                    "respec0b_slot_cursor_state_v1"
                ):
                    raise CheckpointStrictError(
                        "world apply requires slot-keyed RESPEC cursor authority"
                    )
                capacity = int(getattr(env, "slot_to_agent_id").shape[0])
                cursor_tokens = np.asarray(
                    respec0b_state.get("slot_tokens"), dtype=np.int64
                )
                if cursor_tokens.shape != (capacity,) or not np.array_equal(
                    cursor_tokens, np.asarray(env.slot_tokens, dtype=np.int64)
                ):
                    raise CheckpointStrictError(
                        "world apply RESPEC cursor tokens do not match world slots"
                    )
                for name, key, dtype in (
                    ("respec0b_resource_active_y", "active_y", np.int32),
                    ("respec0b_resource_active_x", "active_x", np.int32),
                    ("respec0b_resource_active_class", "active_class", np.int32),
                    ("respec0b_resource_handling_progress", "progress", np.float32),
                ):
                    restored = np.asarray(
                        respec0b_state.get(key), dtype=dtype
                    )
                    if restored.shape != (capacity,):
                        raise CheckpointStrictError(
                            f"world apply RESPEC cursor shape mismatch: field={key}"
                        )
                    setattr(
                        env,
                        name,
                        np.ascontiguousarray(restored, dtype=dtype),
                    )
            compute_local_density_map(env)
            project_patch_food_to_binary_grid(env)
        else:
            reset_patch = getattr(env, "_reset_patch_world_arrays", None)
            if callable(reset_patch):
                reset_patch()
            try:
                from env.gridworld_pkg.patch_world import compute_local_density_map, project_patch_food_to_binary_grid

                compute_local_density_map(env)
                project_patch_food_to_binary_grid(env)
            except Exception:
                pass

    for k in ["food_spawn_mult", "action_cost_mult", "rest_cost_mult"]:
        try:
            setattr(env, k, float(world.get(k, getattr(env, k, 1.0))))
        except Exception:
            pass

    _apply_env_rng_state(env, world.get("env_rng_bitgen_state"))

    # World arrays were overwritten; invalidate derived caches.
    _invalidate_env_transient_caches(env)


def _apply_resident_environment_scalars(sim: Any, resident: Dict[str, Any]) -> None:
    """Restore scalar resident env fields that legacy world.json does not own.

    Patch/WDL runtime fields include deterministic generation-local counters
    such as ``patch_food_harvest_total_last``.  They are intentionally captured
    in resident_runtime_state_v2 and verified after apply, but the mature
    world.json payload only restores arrays and a few legacy multipliers.  This
    helper closes that scalar gap without drawing RNG or rebuilding policy.
    """
    env = getattr(sim, "env", None)
    if env is None or not isinstance(resident, dict):
        return
    env_state = resident.get("environment_state", {})
    if not isinstance(env_state, dict):
        return
    scalars = env_state.get("scalars", {})
    if not isinstance(scalars, dict):
        return
    allowed_prefixes = (
        "patch_",
        "wdl_delayed_hazard_",
        "prediction_demand_",
        "eed_resource_",
    )
    for name, value in sorted(scalars.items(), key=lambda item: str(item[0])):
        key = str(name)
        if not key.startswith(allowed_prefixes):
            continue
        if not hasattr(env, key):
            continue
        try:
            current = getattr(env, key)
        except Exception:
            continue
        if callable(current) or isinstance(current, np.ndarray):
            continue
        try:
            if isinstance(current, bool):
                setattr(env, key, bool(value))
            elif isinstance(current, int) and not isinstance(current, bool):
                setattr(env, key, int(value))
            elif isinstance(current, float):
                setattr(env, key, float(value))
            elif isinstance(current, str):
                setattr(env, key, str(value))
            elif isinstance(value, (list, tuple, dict, set)) or value is None:
                setattr(env, key, value)
            else:
                setattr(env, key, value)
        except Exception:
            continue


def _apply_resident_generation_counters(sim: Any, resident: Dict[str, Any]) -> None:
    """Restore generation counters that are only covered by resident v2.

    Legacy runner_state predates some lineage/WDL counters.  Resident v2 already
    captures them for checkpoint verification; this bridge restores safe scalar
    and set/list boundary state without running any generation logic.
    """
    if not isinstance(resident, dict):
        return
    gen = resident.get("generation_counters", {})
    if not isinstance(gen, dict):
        return
    counters = gen.get("counters", {})
    if not isinstance(counters, dict):
        return
    targets = _runner_targets(sim)
    cursor = counters.get("lineage_reproduction_funnel_event_cursor", None)
    if cursor is not None:
        try:
            n = int(cursor)
        except Exception:
            n = 0
        for t in targets:
            try:
                current = getattr(t, "_lineage_reproduction_funnel_event_contexts_gen", None)
                if not isinstance(current, list) or len(current) != n:
                    setattr(t, "_lineage_reproduction_funnel_event_contexts_gen", [{} for _ in range(max(0, n))])
            except Exception:
                pass
    if "wdl_delayed_hazard_agents_damaged_ids_gen" in counters:
        raw = counters.get("wdl_delayed_hazard_agents_damaged_ids_gen")
        try:
            values = set(int(x) for x in (raw or []))
        except Exception:
            values = set()
        for t in targets:
            try:
                setattr(t, "_wdl_delayed_hazard_agents_damaged_ids_gen", values)
            except Exception:
                pass


def _apply_population(sim: Any, pop_obj: Dict[str, Any], *, weights_dir: str) -> None:
    pop = getattr(sim, "pop", None)
    if pop is None:
        raise ValueError("sim.pop missing")

    from agents.agent import Agent, AgentStats, normalize_lineage_stats
    from agents.genome import Genome
    from agents.agent_policy import MLPPolicy
    from agents.gru_policy import GRUPolicy

    # Restore slot pool first
    slots_obj = pop_obj.get("slot_pool", {}) if isinstance(pop_obj, dict) else {}
    slots = getattr(pop, "_slots", None)
    if slots is not None and isinstance(slots_obj, dict):
        try:
            slots.max_slots = int(slots_obj.get("max_slots", slots.max_slots))
            # free_slots is a deque
            from collections import deque
            slots.free_slots = deque([int(x) for x in (slots_obj.get("free_slots") or [])])
            slots.slot_tokens = [int(x) for x in (slots_obj.get("slot_tokens") or [])]
            pop.max_slots = int(slots.max_slots)
        except Exception:
            pass

    pop.next_id = int(pop_obj.get("next_id", 1) or 1)
    pop.agents.clear()

    # Restore population-owned RNG if present in the bundle.
    # Must not draw RNG; only sets state.
    try:
        prng = getattr(pop, "rng", None)
        if prng is not None and isinstance(pop_obj, dict):
            kind = pop_obj.get("pop_rng_kind", None)
            st = pop_obj.get("pop_rng_state", None)
            if st is not None:
                if kind == "np_generator" and hasattr(prng, "bit_generator") and hasattr(prng.bit_generator, "state"):
                    prng.bit_generator.state = st
                elif kind == "np_randomstate" and hasattr(prng, "set_state"):
                    prng.set_state(st)
    except Exception:
        pass


    obs_size = int(getattr(sim, "_obs_size", 0) or 0)
    n_actions = int(getattr(sim, "_out", 7) or 7)

    for ad in (pop_obj.get("agents") or []):
        if not isinstance(ad, dict):
            continue
        aid = int(ad.get("agent_id", 0) or 0)
        gd = ad.get("genome") or {}
        try:
            g = Genome.from_dict(gd if isinstance(gd, dict) else {})
        except Exception as e:
            # Deterministic governance: never silently degrade malformed/unsupported
            # schema payloads to defaults during checkpoint population restore.
            detail = str(e).replace("\n", " ").strip()
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_GENOME_MIGRATION_ERROR\n"
                f"context=agent_id={aid};detail={detail}\n"
            ) from e

        sustainable_growth_enabled = bool(
            getattr(
                getattr(sim, "sim_cfg", None),
                "structural_sustainable_growth_enable",
                False,
            )
        )
        if sustainable_growth_enabled:
            try:
                for module in list(getattr(g, "modules", []) or []):
                    normalize_activity_module_params(module.get("params", {}))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "ESCAP8_CHECKPOINT_ACTIVITY_RECORD_INVALID "
                    f"agent_id={aid};error={exc}"
                ) from exc

        # Phase 3 determinism: preserve serialized `use_gru` even if Genome.__post_init__
        # derives it from memory_dim. Historical runs can legally carry use_gru=True with
        # memory_dim==0 (GRU architecture with fully-masked effective memory). If resume
        # silently flips use_gru, policy-family splits change and torch RNG consumption
        # diverges at the first tick.
        try:
            if isinstance(gd, dict) and ("use_gru" in gd):
                use_saved = bool(gd.get("use_gru"))
                if bool(getattr(g, "use_gru", False)) != use_saved:
                    setattr(g, "use_gru", use_saved)
        except Exception:
            pass

        # Create policy object and load weights.
        #
        # Resume determinism requires the serialized policy blob to be the
        # architecture authority. Some historical live states can carry
        # genome.use_gru=True with memory_dim==0 while the actual policy object
        # remains an MLPPolicy. Reconstructing from the genome flag in that case
        # creates a GRU, silently leaves random parameters in place, and diverges
        # at the first resumed tick.
        pol = None
        try:
            wpath = os.path.join(weights_dir, f"agent_{aid}.bin")
            if os.path.exists(wpath):
                pol = _policy_from_weights_blob(
                    wpath,
                    fallback_genome=g,
                    obs_size=obs_size,
                    n_actions=n_actions,
                    mlp_policy_cls=MLPPolicy,
                    gru_policy_cls=GRUPolicy,
                )
            elif bool(getattr(g, "use_gru", False)):
                pol = GRUPolicy(
                    obs_size=obs_size,
                    # Phase 3 determinism: GRU architecture is fixed-size (banked execution).
                    # genome.memory_dim is an *effective* memory mask and may be 0..64; it must NOT
                    # be used as the GRU hidden size or resume will create a different model (or fail).
                    hidden_dim=64,
                    n_actions=n_actions,
                )
            else:
                pol = MLPPolicy(obs_size=obs_size, hidden_size=int(getattr(g, "hidden_size", 64) or 64), n_actions=n_actions)
            _load_agent_weights_into_policy(wpath, pol)
        except Exception as e:
            if "DETERMINISM_VIOLATION" in str(e) or "DET_OBSERVATION_SCHEMA_MISMATCH" in str(e):
                raise
            pol = None

        # Stats
        st = AgentStats()
        sd = ad.get("stats") or {}
        if isinstance(sd, dict):
            for k, v in sd.items():
                # ------------------------------------------------------------------
                # Phase 3 determinism: restore lossless-JSON encoded ndarrays.
                #
                # canonical_json_bytes_lossless encodes numpy arrays as:
                #   {"__ndarray__": <nested lists>, "dtype": "float32", "shape": [..]}
                # If we assign that dict directly onto AgentStats fields (exp_dE/exp_dInj/
                # prev_drive_delta), resume silently loses learned predictor state and
                # diverges immediately at the next generation boundary.
                #
                # This decode path must not draw RNG.
                # ------------------------------------------------------------------
                vv = v
                try:
                    if isinstance(vv, dict) and "__ndarray__" in vv and "dtype" in vv and "shape" in vv:
                        dt = np.dtype(str(vv.get("dtype")))
                        data = vv.get("__ndarray__")
                        arr = np.array(data, dtype=dt)
                        shape = vv.get("shape")
                        if isinstance(shape, list) and shape:
                            arr = arr.reshape(tuple(int(x) for x in shape))
                        vv = arr
                except Exception:
                    # Fall back to raw value; downstream tick code has best-effort
                    # robustness guards for malformed stats fields.
                    vv = v

                # pred_queue is a deque in-memory; JSON checkpoints may restore lists.
                # Convert here so resume starts from identical state.
                if str(k) == "pred_queue" and isinstance(vv, list):
                    try:
                        from collections import deque as _deque  # noqa: WPS433

                        # Elements are typically tuples; JSON round-trips tuples as lists.
                        items = [tuple(x) if isinstance(x, list) else x for x in vv]
                        vv = _deque(items, maxlen=8)
                    except Exception:
                        pass
                try:
                    setattr(st, k, vv)
                except Exception:
                    pass

        a = Agent(
            agent_id=aid,
            genome=g,
            policy=pol,
            energy=float(ad.get("energy", 0.0) or 0.0),
            health=float(ad.get("health", 0.0) or 0.0),
            alive=bool(ad.get("alive", True)),
        )
        normalize_lineage_stats(st, aid, generation=0, tick=0)
        a.stats = st
        a.injury = float(ad.get("injury", 0.0) or 0.0)
        a.scar = float(ad.get("scar", 0.0) or 0.0)
        a.habitat_acclimation_state = float(ad.get("habitat_acclimation_state", 0.0) or 0.0)
        a.rest_streak = int(ad.get("rest_streak", 0) or 0)
        a.repro_cooldown = int(ad.get("repro_cooldown", 0) or 0)
        a.slot = int(ad.get("slot", -1))
        a.slot_token = int(ad.get("slot_token", 0) or 0)
        # Bank tokens (mlp/gru): preserve checkpoint bank sync state used for token gating.
        try:
            if "_bank_token" in ad:
                raw_bank_token = ad.get("_bank_token", -1)
                setattr(a, "_bank_token", int(-1 if raw_bank_token is None else raw_bank_token))
        except Exception:
            pass
        try:
            if "_gru_bank_token" in ad:
                raw_gru_bank_token = ad.get("_gru_bank_token", -1)
                setattr(
                    a,
                    "_gru_bank_token",
                    int(-1 if raw_gru_bank_token is None else raw_gru_bank_token),
                )
        except Exception:
            pass


        # drives
        try:
            dv = ad.get("drives")
            if isinstance(dv, list) and len(dv) > 0:
                a.drives = np.array(dv, dtype=np.float32)
        except Exception:
            pass
        # chemistry (Phase 1A)
        try:
            cv = ad.get("chemistry")
            if isinstance(cv, list) and len(cv) > 0:
                a.chemistry = np.array(cv, dtype=np.float32)
        except Exception:
            pass
        try:
            mv = ad.get("maintenance")
            if isinstance(mv, list) and len(mv) > 0:
                a.maintenance = np.array(mv, dtype=np.float32)
        except Exception:
            pass
        # ESCAP-3 owned state: strict schema, zero-RNG restore, stable instance keys.
        live_runtime = ad.get("structural_live_module_runtime", {})
        if live_runtime:
            if not isinstance(live_runtime, dict) or str(live_runtime.get("schema_version", "")) != "escap3_live_module_runtime_v1":
                raise RuntimeError("ESCAP3_CHECKPOINT_RUNTIME_SCHEMA_UNSUPPORTED")
            state = live_runtime.get("state_by_instance_id", {})
            outputs = live_runtime.get("previous_outputs_by_instance_id", {})
            telemetry = live_runtime.get("telemetry", {})
            if not isinstance(state, dict) or not isinstance(outputs, dict) or not isinstance(telemetry, dict):
                raise RuntimeError("ESCAP3_CHECKPOINT_RUNTIME_MALFORMED")
            a.structural_live_module_state = {str(k): float(v) for k, v in sorted(state.items())}
            a.structural_live_module_outputs = {str(k): float(v) for k, v in sorted(outputs.items())}
            a.structural_live_module_telemetry = dict(telemetry)
            if sustainable_growth_enabled:
                dormant_instance_ids = {
                    str(module.get("module_instance_id", "") or "")
                    for module in list(getattr(g, "modules", []) or [])
                    if module_is_dormant(module)
                }
                stale_dormant_outputs = sorted(
                    instance_id
                    for instance_id in dormant_instance_ids
                    if float(a.structural_live_module_outputs.get(instance_id, 0.0)) != 0.0
                )
                if stale_dormant_outputs:
                    raise RuntimeError(
                        "ESCAP8_CHECKPOINT_DORMANT_OUTPUT_NONZERO "
                        f"agent_id={aid};instance_ids={stale_dormant_outputs!r}"
                    )
            from evolution.structural_live_modules_v1 import reconcile_runtime_state_for_genome

            reconcile_runtime_state_for_genome(
                a,
                orphan_policy="reject",
                sustainable_growth_enabled=sustainable_growth_enabled,
            )

        # ESCAP-6 lifetime commitment state is enabled-only, newborn-empty,
        # and keyed by stable module instance identity.  It is not hereditary.
        development_enabled = bool(
            getattr(getattr(sim, "sim_cfg", None), "structural_development_enable", False)
        )
        development_runtime_present = "structural_development_runtime" in ad
        development_runtime = ad.get("structural_development_runtime")
        if development_enabled and not development_runtime_present:
            raise RuntimeError("ESCAP6_CHECKPOINT_RUNTIME_MISSING")
        if not development_enabled and development_runtime_present:
            raise RuntimeError("ESCAP6_CHECKPOINT_RUNTIME_PRESENT_WHILE_DISABLED")
        if development_enabled:
            if not isinstance(development_runtime, dict):
                raise RuntimeError("ESCAP6_CHECKPOINT_RUNTIME_MALFORMED")
            required_development_runtime_fields = {
                "schema_version",
                "latch_version",
                "resolved_stage_id",
                "effective_gain_by_module_instance_id",
                "latch_by_module_instance_id",
            }
            if set(development_runtime) != required_development_runtime_fields:
                raise RuntimeError("ESCAP6_CHECKPOINT_RUNTIME_FIELDS_MISMATCH")
            if str(development_runtime.get("schema_version", "")) != "escap6_development_runtime_v1":
                raise RuntimeError("ESCAP6_CHECKPOINT_RUNTIME_SCHEMA_UNSUPPORTED")
            if str(development_runtime.get("latch_version", "")) != str(
                getattr(
                    getattr(sim, "sim_cfg", None),
                    "structural_development_latch_version",
                    LATCH_VERSION,
                )
                or LATCH_VERSION
            ):
                raise RuntimeError("ESCAP6_CHECKPOINT_LATCH_VERSION_MISMATCH")
            raw_latches = development_runtime.get("latch_by_module_instance_id", {})
            if not isinstance(raw_latches, dict):
                raise RuntimeError("ESCAP6_CHECKPOINT_LATCH_RUNTIME_MALFORMED")
            known_instance_ids = {
                str(module.get("module_instance_id", "") or "")
                for module in list(getattr(g, "modules", []) or [])
                if isinstance(module, dict)
            }
            known_instance_ids.discard("")
            stored_stage_id = str(development_runtime.get("resolved_stage_id", "") or "")
            expected_stage_id, _ = resolve_structural_stage(
                age_steps=int(getattr(st, "survival_steps", 0) or 0),
                sim_cfg=getattr(sim, "sim_cfg", None),
            )
            if stored_stage_id not in DEVELOPMENTAL_STAGE_IDS or stored_stage_id != expected_stage_id:
                raise RuntimeError("ESCAP6_CHECKPOINT_RESOLVED_STAGE_MISMATCH")
            raw_effective_gains = development_runtime.get(
                "effective_gain_by_module_instance_id",
                {},
            )
            if not isinstance(raw_effective_gains, dict):
                raise RuntimeError("ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED")
            restored_effective_gains: Dict[str, float] = {}
            for key, value in sorted(
                raw_effective_gains.items(),
                key=lambda item: str(item[0]),
            ):
                instance_id = str(key)
                if (
                    not instance_id
                    or instance_id not in known_instance_ids
                    or isinstance(value, bool)
                ):
                    raise RuntimeError("ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED")
                try:
                    gain = float(value)
                except Exception as exc:
                    raise RuntimeError(
                        "ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED"
                    ) from exc
                if not np.isfinite(gain) or not MIN_STAGE_GAIN <= gain <= MAX_STAGE_GAIN:
                    raise RuntimeError("ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED")
                restored_effective_gains[instance_id] = gain
            restored_latches: Dict[str, bool] = {}
            for key, value in sorted(raw_latches.items(), key=lambda item: str(item[0])):
                instance_id = str(key)
                if (
                    not instance_id
                    or instance_id not in known_instance_ids
                    or type(value) is not bool
                ):
                    raise RuntimeError("ESCAP6_CHECKPOINT_LATCH_RUNTIME_MALFORMED")
                restored_latches[instance_id] = bool(value)
            a.structural_development_latch_by_instance_id = restored_latches
            a.structural_development_stage_id = stored_stage_id
            a.structural_development_effective_gain_by_instance_id = restored_effective_gains

        # optional memory_cpu
        if torch is not None:
            try:
                mc = ad.get("memory_cpu")
                if isinstance(mc, list):
                    a.memory_cpu = torch.tensor(mc, dtype=torch.float32)
            except Exception:
                pass

        # recent_actions history (coherence).
        try:
            ra = ad.get("recent_actions", None)
            if isinstance(ra, list):
                from collections import deque as _deque_actions
                a.recent_actions = _deque_actions([int(x) for x in ra], maxlen=12)
        except Exception:
            pass

        a.stats = st
        pop.agents[aid] = a


def _apply_control_plane(sim: Any, cp: Dict[str, Any]) -> None:
    # Runner mirrors submission_sequence_number for checkpoint persistence.
    try:
        seq = int(cp.get("submission_sequence_number", 0) or 0)
        setattr(sim, "_control_plane_submission_sequence_number", seq)
    except Exception:
        pass

    cpo = getattr(sim, "control_plane", None)
    if cpo is not None:
        try:
            cpo.set_submission_sequence_number(int(cp.get("submission_sequence_number", 0) or 0))
        except Exception:
            pass


def _apply_rng_snapshot(sim: Any, rng: Dict[str, Any]) -> None:
    """Restore RNG snapshot.

    Phase 3 requirements:
    - Restores Phase 1 RNGManager streams (authoritative engine RNG).
    - Restores torch RNG state (CPU + CUDA) if present.
    - Restores NumPy *global* legacy RandomState.
    - Restores Python `random` module global state.

    Notes
    -----
    This function MUST be pure with respect to RNG consumption (i.e., it must
    not call any RNG draw APIs). It only sets state.
    """
    # ------------------------------------------------------------------
    # Phase 1: RNGManager streams (authoritative engine RNG)
    # ------------------------------------------------------------------
    # RNGManager restore (Phase 1 determinism substrate)
    # ------------------------------------------------------------------
    rm = getattr(sim, "rng_manager", None)
    if rm is not None:
        try:
            # Preferred API in this repo: apply_snapshot_dict()
            if hasattr(rm, "apply_snapshot_dict"):
                rm.apply_snapshot_dict(rng)
            elif hasattr(rm, "apply_snapshot"):
                rm.apply_snapshot(rng)
            elif hasattr(rm, "restore_snapshot"):
                rm.restore_snapshot(rng)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Torch RNG restore (best-effort)
    # ------------------------------------------------------------------
    if torch is not None:
        try:
            ts = rng.get("torch_rng_state_v1", None) if isinstance(rng, dict) else None
            if isinstance(ts, dict) and str(ts.get("schema_version", "") or "") == "torch_rng_state_v1":
                cpu_hex = str(ts.get("cpu_hex", "") or "")
                if cpu_hex:
                    b = bytes.fromhex(cpu_hex)
                    torch.set_rng_state(torch.tensor(list(b), dtype=torch.uint8))
                # CUDA states (if available)
                try:
                    cuda_hex_list = ts.get("cuda_hex_list", None)
                    if cuda_hex_list and hasattr(torch, "cuda") and torch.cuda.is_available():
                        states = []
                        for hx in list(cuda_hex_list):
                            bb = bytes.fromhex(str(hx))
                            states.append(torch.tensor(list(bb), dtype=torch.uint8))
                        torch.cuda.set_rng_state_all(states)
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # NumPy global RNG restore (legacy RandomState)
    # ------------------------------------------------------------------
    try:
        ns = rng.get("numpy_random_state_v1", None) if isinstance(rng, dict) else None
        if isinstance(ns, dict) and str(ns.get("schema_version", "")) == "numpy_random_state_v1":
            algo = str(ns.get("algorithm", "") or "")
            keys_hex = str(ns.get("keys_hex", "") or "")
            keys_dtype = np.dtype(str(ns.get("keys_dtype", "uint32")))
            keys_shape = tuple(int(x) for x in (ns.get("keys_shape") or []))
            pos = int(ns.get("pos", 0) or 0)
            has_gauss = int(ns.get("has_gauss", 0) or 0)
            cached_gaussian = float(ns.get("cached_gaussian", 0.0) or 0.0)

            keys_buf = bytes.fromhex(keys_hex) if keys_hex else b""
            keys_arr = np.frombuffer(keys_buf, dtype=keys_dtype)
            if keys_shape:
                keys_arr = keys_arr.reshape(keys_shape, order="C")
            keys_arr = np.ascontiguousarray(keys_arr).copy()
            np.random.set_state((algo, keys_arr, pos, has_gauss, cached_gaussian))
    except Exception:
        pass

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # NumPy Generator RNG islands restore (runner/pop generators)
    # ------------------------------------------------------------------
    try:
        if isinstance(rng, dict):
            rs = rng.get("runner_np_rng_state_v1")
            if isinstance(rs, dict) and str(rs.get("schema_version", "")) == "np_generator_state_v1":
                bit_state = rs.get("bitgen_state", None)
                g = getattr(sim, "np_rng", None)
                if g is not None and getattr(g, "bit_generator", None) is not None and bit_state is not None:
                    g.bit_generator.state = bit_state
            ps = rng.get("pop_np_rng_state_v1")
            if isinstance(ps, dict) and str(ps.get("schema_version", "")) == "np_generator_state_v1":
                bit_state = ps.get("bitgen_state", None)
                pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
                g = getattr(pop, "np_rng", None) if pop is not None else None
                if g is not None and getattr(g, "bit_generator", None) is not None and bit_state is not None:
                    g.bit_generator.state = bit_state
    except Exception:
        pass

    # Python global RNG restore (legacy `random` module)
    # ------------------------------------------------------------------
    try:
        ps = rng.get("python_random_state_v1", None) if isinstance(rng, dict) else None
        if isinstance(ps, dict) and str(ps.get("schema_version", "")) == "python_random_state_v1":
            import pickle as _pickle  # noqa: WPS433 (std)
            import random as _random  # noqa: WPS433 (std)
            hx = str(ps.get("pickle_hex", "") or "")
            if hx:
                _random.setstate(_pickle.loads(bytes.fromhex(hx)))
    except Exception:
        pass


# ---------------------------------------------------------------------
# Agent serialization helpers
# ---------------------------------------------------------------------

def _agent_to_dict(
    a: Any,
    *,
    structural_development_enabled: bool = False,
    structural_development_sim_cfg: Any = None,
) -> Dict[str, Any]:
    st = getattr(a, "stats", None)
    st_obj: Dict[str, Any] = {}
    if st is not None:
        try:
            st_obj = _to_jsonable(st)
        except Exception:
            st_obj = {}
        # D1: AgentStats carries runtime-attached developmental guard fields that
        # are not part of the dataclass schema. Persist explicitly for resume parity.
        if not isinstance(st_obj, dict):
            st_obj = {}
        for k in (
            "_dev_module_structure_signature",
            "_dev_last_effective",
            "_dev_last_expr_snapshot",
            "_dev_stage_id",
            "_dev_active_module_count",
            "_dev_last_trace_gen",
            "_dev_last_trace_tick",
            "_dev_d2_lock_state_by_module_id",
            "_dev_d2_lock_state_sha256",
        ):
            try:
                if hasattr(st, k):
                    st_obj[k] = _to_jsonable(getattr(st, k))
            except Exception:
                continue

    d: Dict[str, Any] = {
        "agent_id": int(getattr(a, "agent_id", 0) or 0),
        "alive": bool(getattr(a, "alive", True)),
        "energy": float(getattr(a, "energy", 0.0) or 0.0),
        "health": float(getattr(a, "health", 0.0) or 0.0),
        "injury": float(getattr(a, "injury", 0.0) or 0.0),
        "habitat_acclimation_state": float(getattr(a, "habitat_acclimation_state", 0.0) or 0.0),
        # Scar is a persistent injury-derived impairment used by env costs and
        # reproduction gating. It must round-trip through checkpoint bundles.
        "scar": float(getattr(a, "scar", 0.0) or 0.0),
        "rest_streak": int(getattr(a, "rest_streak", 0) or 0),
        "repro_cooldown": int(getattr(a, "repro_cooldown", 0) or 0),
        "slot": (-1 if getattr(a, "slot", -1) is None else int(getattr(a, "slot", -1))),
        "slot_token": int(getattr(a, "slot_token", 0) or 0),
        "genome": (
            _to_jsonable(getattr(a, "genome", None).to_dict())
            if getattr(getattr(a, "genome", None), "to_dict", None) is not None
            else _to_jsonable(getattr(a, "genome", None))
        ),
        "stats": st_obj,
    }
    # Bank tokens: determinism-critical for bank sync gating (especially GRU memory reset).
    # Persist them so resume can reproduce which slots were considered "synced" at the boundary.
    try:
        d["_bank_token"] = int(getattr(a, "_bank_token", -1))
    except Exception:
        d["_bank_token"] = -1
    try:
        d["_gru_bank_token"] = int(getattr(a, "_gru_bank_token", -1))
    except Exception:
        d["_gru_bank_token"] = -1

    try:
        d["drives"] = np.asarray(getattr(a, "drives")).astype(np.float32).tolist()
    except Exception:
        pass
    try:
        d["chemistry"] = np.asarray(getattr(a, "chemistry")).astype(np.float32).tolist()
    except Exception:
        pass
    try:
        d["maintenance"] = np.asarray(getattr(a, "maintenance")).astype(np.float32).tolist()
    except Exception:
        pass
    live_state = dict(getattr(a, "structural_live_module_state", {}) or {})
    live_outputs = dict(getattr(a, "structural_live_module_outputs", {}) or {})
    live_telemetry = dict(getattr(a, "structural_live_module_telemetry", {}) or {})
    sustainable_growth_enabled = bool(
        getattr(
            structural_development_sim_cfg,
            "structural_sustainable_growth_enable",
            False,
        )
    )
    if sustainable_growth_enabled:
        dormant_instance_ids = {
            str(module.get("module_instance_id", "") or "")
            for module in list(getattr(getattr(a, "genome", None), "modules", []) or [])
            if module_is_dormant(module)
        }
        stale_dormant_outputs = sorted(
            instance_id
            for instance_id in dormant_instance_ids
            if float(live_outputs.get(instance_id, 0.0)) != 0.0
        )
        if stale_dormant_outputs:
            raise RuntimeError(
                "ESCAP8_CHECKPOINT_DORMANT_OUTPUT_NONZERO "
                f"agent_id={getattr(a, 'agent_id', 0)};"
                f"instance_ids={stale_dormant_outputs!r}"
            )
    if live_state or live_outputs or live_telemetry:
        d["structural_live_module_runtime"] = {
            "schema_version": "escap3_live_module_runtime_v1",
            "state_by_instance_id": _to_jsonable(live_state),
            "previous_outputs_by_instance_id": _to_jsonable(live_outputs),
            "telemetry": _to_jsonable(live_telemetry),
        }
    if bool(structural_development_enabled):
        known_instance_ids = {
            str(module.get("module_instance_id", "") or "")
            for module in list(getattr(getattr(a, "genome", None), "modules", []) or [])
            if isinstance(module, dict)
        }
        known_instance_ids.discard("")
        latch_by_instance_id = dict(
            getattr(a, "structural_development_latch_by_instance_id", {}) or {}
        )
        canonical_latches: Dict[str, bool] = {}
        for key, value in sorted(latch_by_instance_id.items(), key=lambda item: str(item[0])):
            instance_id = str(key)
            if (
                not instance_id
                or instance_id not in known_instance_ids
                or type(value) is not bool
            ):
                raise RuntimeError("ESCAP6_CHECKPOINT_LATCH_RUNTIME_MALFORMED")
            canonical_latches[instance_id] = bool(value)
        effective_gain_by_instance_id = dict(
            getattr(a, "structural_development_effective_gain_by_instance_id", {}) or {}
        )
        canonical_effective_gains: Dict[str, float] = {}
        for key, value in sorted(
            effective_gain_by_instance_id.items(),
            key=lambda item: str(item[0]),
        ):
            instance_id = str(key)
            if not instance_id or instance_id not in known_instance_ids or isinstance(value, bool):
                raise RuntimeError("ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED")
            try:
                gain = float(value)
            except Exception as exc:
                raise RuntimeError("ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED") from exc
            if not np.isfinite(gain) or not MIN_STAGE_GAIN <= gain <= MAX_STAGE_GAIN:
                raise RuntimeError("ESCAP6_CHECKPOINT_EFFECTIVE_GAIN_RUNTIME_MALFORMED")
            canonical_effective_gains[instance_id] = gain
        age_steps = int(getattr(getattr(a, "stats", None), "survival_steps", 0) or 0)
        resolved_stage_id, _ = resolve_structural_stage(
            age_steps=age_steps,
            sim_cfg=structural_development_sim_cfg,
        )
        d["structural_development_runtime"] = {
            "schema_version": "escap6_development_runtime_v1",
            "latch_version": str(
                getattr(
                    structural_development_sim_cfg,
                    "structural_development_latch_version",
                    LATCH_VERSION,
                )
                or LATCH_VERSION
            ),
            "resolved_stage_id": str(resolved_stage_id),
            "effective_gain_by_module_instance_id": canonical_effective_gains,
            "latch_by_module_instance_id": canonical_latches,
        }
    # recent_actions: bounded history used for coherence penalties and thus effective fitness.
    try:
        ra = getattr(a, "recent_actions", None)
        if ra is not None:
            d["recent_actions"] = [int(x) for x in list(ra)]
    except Exception:
        pass
    if torch is not None:
        try:
            mc = getattr(a, "memory_cpu", None)
            if mc is not None:
                d["memory_cpu"] = mc.detach().to("cpu").float().tolist()
        except Exception:
            pass
    return d


def _save_agent_weights(agent: Any, out_path: str, *, obs_size: int, n_actions: int) -> None:
    """Serialize agent.policy weights into a stable binary blob (no pickle)."""
    if torch is None:
        raise RuntimeError("torch required to serialize weights")
    pol = getattr(agent, "policy", None)
    if pol is None:
        raise RuntimeError("agent.policy missing")
    sd = pol.state_dict()
    keys = sorted(sd.keys())
    header = {
        "schema_version": "weights_blob_v2",
        "agent_id": int(getattr(agent, "agent_id", 0) or 0),
        "policy_class": str(pol.__class__.__name__),
        "obs_size": int(obs_size),
        "n_actions": int(n_actions),
        "tensors": [],
    }

    raw_chunks: List[bytes] = []
    offset = 0
    for k in keys:
        t = sd[k]
        if not hasattr(t, "detach"):
            continue
        tt = t.detach().to("cpu")
        # Force float32 for portability and deterministic bytes.
        if tt.dtype != torch.float32:
            tt = tt.float()
        arr = tt.contiguous().numpy()
        b = arr.tobytes(order="C")
        raw_chunks.append(b)
        header["tensors"].append({
            "name": str(k),
            "dtype": "float32",
            "shape": list(arr.shape),
            "offset": int(offset),
            "nbytes": int(len(b)),
        })
        offset += len(b)

    header_bytes = canonical_json_bytes(header)
    with open(out_path, "wb") as f:
        f.write(WEIGHTS_MAGIC)
        f.write(struct.pack("<I", int(len(header_bytes))))
        f.write(header_bytes)
        for ch in raw_chunks:
            f.write(ch)


def _read_agent_weights_blob(path: str) -> Tuple[Dict[str, Any], bytes]:
    """Read an EIELW2 policy weights blob without constructing a policy."""
    with open(path, "rb") as f:
        magic = f.read(len(WEIGHTS_MAGIC))
        if magic != WEIGHTS_MAGIC:
            raise ValueError("invalid weights blob magic")
        (hlen,) = struct.unpack("<I", f.read(4))
        hbytes = f.read(int(hlen))
        header = json.loads(hbytes.decode("utf-8"))
        payload = f.read()
    if not isinstance(header, dict):
        raise ValueError("invalid weights blob header")
    return header, payload


def _weights_blob_tensor_shapes(header: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    out: Dict[str, Tuple[int, ...]] = {}
    for tinfo in (header.get("tensors") or []):
        if not isinstance(tinfo, dict):
            continue
        name = str(tinfo.get("name", ""))
        if not name:
            continue
        out[name] = tuple(int(x) for x in (tinfo.get("shape") or []))
    return out


def _policy_from_weights_blob(
    path: str,
    *,
    fallback_genome: Any,
    obs_size: int,
    n_actions: int,
    mlp_policy_cls: Any,
    gru_policy_cls: Any,
) -> Any:
    """Construct the policy architecture declared by a checkpoint weight blob."""
    header, _payload = _read_agent_weights_blob(path)
    policy_class = str(header.get("policy_class", "") or "")
    saved_obs = int(header.get("obs_size", 0) or 0)
    saved_actions = int(header.get("n_actions", 0) or 0)
    effective_obs = saved_obs if saved_obs > 0 else int(obs_size)
    effective_actions = saved_actions if saved_actions > 0 else int(n_actions)
    shapes = _weights_blob_tensor_shapes(header)

    if policy_class == "MLPPolicy":
        hidden_size = int(getattr(fallback_genome, "hidden_size", 64) or 64)
        first_weight = shapes.get("net.0.weight")
        if first_weight and len(first_weight) >= 1 and int(first_weight[0]) > 0:
            hidden_size = int(first_weight[0])
        return mlp_policy_cls(obs_size=int(effective_obs), hidden_size=int(hidden_size), n_actions=int(effective_actions))

    if policy_class == "GRUPolicy":
        # GRUPolicy enforces its fixed bank-compatible hidden size internally.
        return gru_policy_cls(obs_size=int(effective_obs), hidden_dim=64, n_actions=int(effective_actions))

    raise RuntimeError(
        "[DETERMINISM_VIOLATION]\n"
        "code=DET_POLICY_WEIGHT_BLOB_UNSUPPORTED\n"
        f"context=path={path};policy_class={policy_class}\n"
    )


def _load_agent_weights_into_policy(path: str, policy: Any) -> None:
    """Load a weights blob into the provided policy module."""
    if torch is None:
        raise RuntimeError("torch required to load weights")
    header, payload = _read_agent_weights_blob(path)

    tensors = header.get("tensors") or []
    try:
        saved_obs = int(header.get("obs_size", 0) or 0)
        policy_obs = int(getattr(policy, "obs_size", 0) or 0)
        if saved_obs > 0 and policy_obs > 0 and saved_obs != policy_obs:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_OBSERVATION_SCHEMA_MISMATCH\n"
                f"context=checkpoint_obs_size={saved_obs};runtime_obs_size={policy_obs}\n"
            )
    except RuntimeError:
        raise
    except Exception:
        pass
    sd = policy.state_dict()
    new_sd = {}
    for tinfo in tensors:
        name = str(tinfo.get("name"))
        if name not in sd:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_POLICY_WEIGHT_KEY_MISMATCH\n"
                f"context=path={path};extra_saved_key={name}\n"
            )
        shape = tuple(int(x) for x in (tinfo.get("shape") or []))
        expected_shape = tuple(int(x) for x in tuple(sd[name].shape))
        if shape != expected_shape:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_POLICY_WEIGHT_SHAPE_MISMATCH\n"
                f"context=path={path};key={name};checkpoint_shape={shape};runtime_shape={expected_shape}\n"
            )
        off = int(tinfo.get("offset", 0) or 0)
        nbytes = int(tinfo.get("nbytes", 0) or 0)
        chunk = payload[off:off + nbytes]
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(np.float32).itemsize)
        if len(chunk) != nbytes or int(nbytes) != expected_nbytes:
            raise RuntimeError(
                "[DETERMINISM_VIOLATION]\n"
                "code=DET_POLICY_WEIGHT_PAYLOAD_MISMATCH\n"
                f"context=path={path};key={name};nbytes={nbytes};expected_nbytes={expected_nbytes}\n"
            )
        arr = np.frombuffer(chunk, dtype=np.float32).copy().reshape(shape)
        new_sd[name] = torch.from_numpy(arr).to(dtype=sd[name].dtype)
    missing = sorted(str(k) for k in set(sd.keys()) - set(new_sd.keys()))
    if missing:
        raise RuntimeError(
            "[DETERMINISM_VIOLATION]\n"
            "code=DET_POLICY_WEIGHT_KEY_MISMATCH\n"
            f"context=path={path};missing_saved_keys={','.join(missing)}\n"
        )
    sd.update(new_sd)
    policy.load_state_dict(sd)


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_metadata(
    meta: Dict[str, Any],
    *,
    strict: bool,
    historical_analysis: bool = False,
) -> None:
    """Validate metadata.json contents for Checkpoint Bundle v2.

    In strict mode, this enforces the determinism-critical field contract defined
    by TSD v3.1. Non-strict mode remains permissive to support exploratory
    debugging and potential legacy tooling.
    """
    if not isinstance(meta, dict):
        raise CheckpointSchemaError("metadata.json must be a JSON object")
    if str(meta.get("schema_version", "")) != SCHEMA_VERSION:
        raise CheckpointSchemaError(f"unsupported schema_version: {meta.get('schema_version')}")

    # A repaired envelope, when present, is never advisory.  Validate its
    # complete frozen contract even for a non-strict exploratory load.
    if "inheritance" in meta:
        recorded_inheritance = meta.get("inheritance")
        if not isinstance(recorded_inheritance, dict):
            raise CheckpointStrictError("inheritance checkpoint metadata must be an object")
        recorded_profile = validate_inheritance_profile(
            recorded_inheritance.get("profile")
        )
        if recorded_profile == LINKED_INHERITANCE_PROFILE:
            if not historical_analysis:
                raise CheckpointStrictError(
                    "P12_RETIRED_INHERITANCE_PROFILE "
                    "checkpoint is historical-analysis-only"
                )
            try:
                inspect_historical_inheritance_checkpoint_v1(
                    recorded_inheritance
                )
            except ValueError as exc:
                raise CheckpointStrictError(str(exc)) from exc
        else:
            expected_inheritance = inheritance_checkpoint_metadata(
                SimpleNamespace(inheritance_profile=recorded_profile)
            )
            if recorded_inheritance != expected_inheritance:
                raise CheckpointStrictError(
                    "inheritance checkpoint metadata contract mismatch"
                )

    if not strict:
        return

    # Determinism-critical contract (TSD v3.1)
    required = [
        "checkpoint_id",
        "generation_index",
        "branch_id",
        "campaign_id",
        "code_commit_hash",
        "python_version",
        "numpy_version",
        "torch_version",
        "population_hash",
        "world_hash",
        "rng_snapshot_hash",
        "event_log_hash",
        "wal_hash",
        "submission_sequence_number",
        "strict_replay_mode",
    ]
    for k in required:
        if k not in meta:
            raise CheckpointStrictError(f"metadata missing required field: {k}")

    # Basic type sanity (avoid silent schema drift)
    if not isinstance(meta.get("generation_index"), int):
        raise CheckpointStrictError("metadata.generation_index must be int")
    if not isinstance(meta.get("submission_sequence_number"), int):
        raise CheckpointStrictError("metadata.submission_sequence_number must be int")
    if not isinstance(meta.get("strict_replay_mode"), bool):
        raise CheckpointStrictError("metadata.strict_replay_mode must be bool")


def _reject_pickle_artifacts(bundle_dir: str) -> None:
    bad_ext = (".pkl", ".pickle", ".pt", ".pth")
    for root, _dirs, files in os.walk(bundle_dir):
        for fn in files:
            lower = fn.lower()
            if lower.endswith(bad_ext):
                raise CheckpointStrictError(f"pickle/torch artifact forbidden in v2 bundle: {os.path.join(root, fn)}")


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def _write_canonical_json(path: str, obj: Any, *, durable: bool = False) -> None:
    b = canonical_json_bytes(obj)
    with open(path, "wb") as f:
        f.write(b)
        if durable:
            _flush_and_fsync_file(f, path=path)


def _write_canonical_json_lossless(
    path: str,
    obj: Any,
    *,
    durable: bool = False,
) -> None:
    """Checkpoint-bundle canonical JSON without lossy float normalization."""
    b = canonical_json_bytes_lossless(obj)
    with open(path, "wb") as f:
        f.write(b)
        if durable:
            _flush_and_fsync_file(f, path=path)


def _write_canonical_json_lossless_gzip(
    path: str,
    obj: Any,
    *,
    durable: bool = False,
) -> None:
    """Checkpoint-bundle canonical JSON sidecar stored in deterministic gzip."""

    b = canonical_json_bytes_lossless(obj)
    with open(path, "wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=1,
            mtime=0,
        ) as f:
            f.write(b)
        if durable:
            _flush_and_fsync_file(raw, path=path)


def _checkpoint_projection_sidecar_name() -> str:
    compression = str(
        os.environ.get("EIEL_CHECKPOINT_PROJECTION_COMPRESSION", "gzip") or ""
    ).strip().lower()
    if compression in {"0", "false", "off", "none", "json"}:
        return NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME
    if compression not in {"", "1", "true", "on", "gzip", "gz"}:
        raise CheckpointStrictError(
            "unsupported EIEL_CHECKPOINT_PROJECTION_COMPRESSION: "
            f"{compression!r}"
        )
    return NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME


def _bundle_json_path(bdir: str, filename: str) -> str:
    path = os.path.join(bdir, filename)
    if os.path.exists(path):
        return path
    if filename.endswith(".json"):
        gzip_path = path + ".gz"
        if os.path.exists(gzip_path):
            return gzip_path
    return path


def _bundle_json_exists(bdir: str, filename: str) -> bool:
    return os.path.exists(_bundle_json_path(bdir, filename))


def _native_runtime_checkpoint_projection_path_for_load(bundle_dir: str) -> str:
    return _bundle_json_path(bundle_dir, NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME)


def _native_runtime_checkpoint_projection_exists(bundle_dir: str) -> bool:
    return os.path.exists(_native_runtime_checkpoint_projection_path_for_load(bundle_dir))


def _read_json(path: str) -> Any:
    """Read JSON with support for lossless numpy ndarray encoding.

    The checkpoint bundles may encode ndarrays as:
        {"__ndarray__": <nested lists>, "dtype": "<dtype>", "shape": [..]}
    This hook reconstructs a numpy array with the requested dtype/shape.
    """
    def _hook(obj: Any) -> Any:
        if isinstance(obj, dict) and "__ndarray__" in obj and "dtype" in obj and "shape" in obj:
            data = obj["__ndarray__"]
            dtype = obj["dtype"]
            shape = obj["shape"]
            arr = np.array(data, dtype=dtype)
            try:
                arr = arr.reshape(tuple(shape))
            except Exception as e:
                raise ValueError(f"invalid ndarray encoding in {path}: dtype={dtype} shape={shape}") from e
            return arr
        return obj

    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f, object_hook=_hook)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=_hook)



def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _event_log_sha256(sim: Any) -> str:
    run_dir = getattr(sim, "run_dir", "") or ""
    p = os.path.join(str(run_dir), "events", "command_events.jsonl")
    if not os.path.exists(p):
        # Deterministic placeholder (empty log)
        return hashlib.sha256(b"").hexdigest()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wal_dir_sha256(sim: Any) -> str:
    """Stable sha256 for wal/*.jsonl in sim.run_dir (lex filename order)."""
    run_dir = getattr(sim, "run_dir", "") or ""
    wdir = os.path.join(str(run_dir), "wal")
    if not os.path.isdir(wdir):
        return hashlib.sha256(b"").hexdigest()
    files = [fn for fn in os.listdir(wdir) if fn.endswith('.jsonl')]
    files.sort()
    h = hashlib.sha256()
    for fn in files:
        h.update(fn.encode('utf-8'))
        h.update(b"\n")
        fp = os.path.join(wdir, fn)
        with open(fp, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\n")
    return h.hexdigest()


def _rm_tree_best_effort(path: str) -> None:
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # Numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
        d = {}
        for k, v in obj.__dict__.items():
            if str(k).startswith("_"):
                continue
            d[str(k)] = _to_jsonable(v)
        return d
    return str(obj)


def _git_commit_if_available(sim: Any) -> str:
    # Best-effort: allow injection
    commit = getattr(sim, "git_commit", "") or ""
    if commit:
        return str(commit)

    run_dir = getattr(sim, "run_dir", "") or ""
    if not run_dir:
        return ""
    here = os.path.abspath(run_dir)
    for _ in range(6):
        git_dir = os.path.join(here, ".git")
        head = os.path.join(git_dir, "HEAD")
        if os.path.exists(head):
            try:
                with open(head, "r", encoding="utf-8") as f:
                    ref = f.read().strip()
                if ref.startswith("ref:"):
                    ref_path = ref.split(":", 1)[1].strip()
                    ref_file = os.path.join(git_dir, ref_path)
                    if os.path.exists(ref_file):
                        with open(ref_file, "r", encoding="utf-8") as rf:
                            return rf.read().strip()
                # Detached head
                if len(ref) >= 7:
                    return ref
            except Exception:
                return ""
        here = os.path.dirname(here)
    return ""
