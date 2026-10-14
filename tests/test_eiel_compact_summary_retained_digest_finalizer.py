from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.native_runtime.checkpoint_digests import (
    SCHEMA_VERSION as CHECKPOINT_LAYER_SCHEMA_VERSION,
    SOURCE_RNG_SCHEMA_VERSION,
    DigestLayer,
)
from engine.native_runtime.compact_summary_cpu import WIRE_SCHEMA_VERSION
from engine.native_runtime.compact_summary_digests import (
    ACTIVE_AGENT_IDS_SCHEMA_VERSION,
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
    RECOVERABLE_ERROR_SCHEMA_VERSION,
    active_agent_ids_digest_from_tensor_sha256,
    behavior_rows_digest_sha256,
    checkpoint_layer_digest_from_tensor_manifest,
    checkpoint_rng_digest_from_tensor_manifest,
    checkpoint_source_projection_digest_from_tensor_manifest,
    lineage_records_digest_sha256,
    policy_mutation_tape_digest_from_tensor_sha256,
    recoverable_error_digest_sha256,
)
from engine.native_runtime.compact_summary_planes import DIGEST_FIELDS
from engine.native_runtime.compact_summary_reporting_projection import (
    REPORTING_AUTHORITY_SCHEMA_VERSION,
    REPORTING_RETAINED_LAYOUT_VERSION,
)
from engine.native_runtime.compact_summary_retained_digest_finalizer import (
    FINALIZER_SOURCE_SCHEMA_VERSION,
    CompactSummaryRetainedDigestFinalizerError,
    finalize_compact_summary_retained_digests,
)
from engine.native_runtime.lifecycle_events import (
    SCHEMA_VERSION as TYPED_LIFECYCLE_SCHEMA_VERSION,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _rng_state(seed: int) -> dict[str, Any]:
    return {
        "bit_generator": "PCG64",
        "state": {"state": seed, "inc": seed * 2 + 1},
        "has_uint32": 0,
        "uinteger": 0,
    }


def _behavior_row(*, generation: int, agent_id: int) -> dict[str, Any]:
    return {
        "generation": generation,
        "agent_id": agent_id,
        "cohort": "start",
        "move_n": agent_id,
        "move_s": 0,
        "move_e": 0,
        "move_w": 0,
        "gather": 1,
        "rest": 0,
        "attack": 0,
        "lifespan_steps": 4,
        "energy_collected": 1.25,
        "offspring": 0,
        "attacks_made": 0,
        "fitness": 2.5,
        "alive_end": 1,
        "eligible_for_selection": 1,
        "kept_next": 1,
        "injury_end": 0.0,
        "injury_delta": 0.0,
        "aggression_bias": 0.125,
    }


def _lineage_record(agent_id: int) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "energy": 10.0,
        "health": 1.0,
        "injury": 0.0,
        "drives": [0.1, 0.2, 0.3, 0.4],
        "y": agent_id,
        "x": 0,
        "stats": {
            "lineage_id": f"L{agent_id:08d}",
            "founder_id": agent_id,
            "parent_ids": [],
            "parent_lineage_ids": [],
            "birth_generation": 0,
            "birth_tick": 0,
            "lineage_depth": 0,
            "offspring_count": 0,
            "survival_steps": 4.0,
            "energy_collected": 1.25,
            "first_repro_step": None,
            "pred_err_count": 0,
            "pred_err_sum": 0.0,
            "episode0_pred_err_mean": 0.0,
            "identity_drift_ema": 0.0,
            "drive_jitter": 0.0,
            "ecological_role": "general_agent",
            "role_source": "default",
            "role_confidence": None,
            "patch_habitat_lineage_id": agent_id,
        },
        "genome": {
            "hidden_size": 8,
            "memory_dim": 4,
            "use_gru": False,
            "aggression_bias": 0.125,
            "exploration_bias": -0.25,
        },
    }


def _fixture(*, generation: int = 0, cell: int = 0, world: str = "flat") -> dict[str, Any]:
    agent_id = cell + 1
    active_hash = _hash(f"active-{generation}-{cell}")
    rng_hash = _hash(f"rng-{generation}-{cell}")
    descriptors = {
        "agent_ids_by_alive_order": {
            "device": "cuda:0",
            "dtype": "int64",
            "shape": [1],
            "sha256": active_hash,
        },
        "rng_manager_mutation_torch_cpu_state": {
            "device": "cuda:0",
            "dtype": "uint8",
            "shape": [4],
            "sha256": rng_hash,
        },
    }
    original_manifest = {
        name: {
            "shape": list(entry["shape"]),
            "dtype": "Long" if entry["dtype"] == "int64" else "Byte",
            "device": "cuda:0",
            "numel": int(np.prod(entry["shape"])),
            "element_size": 8 if entry["dtype"] == "int64" else 1,
            "bytes": int(np.prod(entry["shape"]))
            * (8 if entry["dtype"] == "int64" else 1),
            "is_cuda": True,
            "is_contiguous": True,
        }
        for name, entry in descriptors.items()
    }
    env_state = _rng_state(10 + generation * 3 + cell)
    metadata = {
        "world": world,
        "generation": generation + 1,
        "tick": 0,
        "configured_steps_per_generation": 4,
        "agent_grid_runner_phase": "generation_boundary",
        "agent_grid_authority_epoch": f"generation:{generation + 1}:tick:0",
        "alive_count": 1,
        "native_runtime_numpy_env_state": env_state,
        "native_runtime_numpy_runner_state": _rng_state(20 + cell),
        "native_runtime_numpy_policy_state": copy.deepcopy(env_state),
        "native_runtime_env_policy_numpy_alias": True,
        "native_runtime_numpy_population_state": _rng_state(30 + generation),
        "native_runtime_python_random_state": [3, [1, 2, 3], None],
        "native_runtime_rng_manager_stream_indices": {
            "policy": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
            "mutation": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
            "metrics": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
            "stress": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
        },
    }
    projection_fields = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": metadata,
        "tensor_manifest": original_manifest,
        "restore_count": 0,
        "generation_calls": generation + 1,
        "event_cursor": 3,
        "runtime_event_cursor": 7,
        "last_applied_actions": 1,
        "last_moved": 1,
        "last_gathered": 0,
        "last_policy_refill_calls": 1,
        "last_policy_refill_rows": 1,
        "pending_generation_free_agent_ids": [],
        "pending_generation_free_slots": [],
        "last_world": world,
    }
    trajectory_fields = [
        {
            "field": "agent_ids_by_alive_order",
            "dtype": "int64",
            "shape": [1],
            "sha256": active_hash,
        },
        {
            "field": "rng_manager_mutation_torch_cpu_state",
            "dtype": "uint8",
            "shape": [4],
            "sha256": rng_hash,
        },
    ]
    storage_fields = copy.deepcopy(trajectory_fields)
    policy_shapes = {
        "policy_birth_crossover_mask_tape": [1, 2],
        "policy_birth_delta_tape": [1, 2],
        "policy_birth_mutation_mask_tape": [1, 2],
        "policy_birth_post_rng_state_tape": [1, 4],
        "policy_birth_semantic_draws_tape": [1],
    }
    policy_hashes = {
        name: _hash(f"{name}-{generation}-{cell}") for name in policy_shapes
    }
    behavior_rows = [_behavior_row(generation=generation, agent_id=agent_id)]
    lineage_records = [_lineage_record(agent_id)]
    return {
        "schema_version": FINALIZER_SOURCE_SCHEMA_VERSION,
        "wire_schema_version": WIRE_SCHEMA_VERSION,
        "generation": generation,
        "cell": cell,
        "world": world,
        "checkpoint": {
            "source_version": CHECKPOINT_LAYER_SCHEMA_VERSION,
            "rng_source_version": SOURCE_RNG_SCHEMA_VERSION,
            "projection_schema_version": projection_fields["schema_version"],
            "projection_fields": projection_fields,
            "tensor_manifest": descriptors,
            "trajectory_state_fields": trajectory_fields,
            "canonical_checkpoint_storage_fields": storage_fields,
        },
        "identity": {
            "source_version": ACTIVE_AGENT_IDS_SCHEMA_VERSION,
            "active_agent_count": 1,
            "tensor_sha256": active_hash,
        },
        "policy_mutation_tape": {
            "source_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
            "cursor": 1,
            "consumed_shapes": policy_shapes,
            "tensor_sha256": policy_hashes,
        },
        "reporting": {
            "schema_version": REPORTING_AUTHORITY_SCHEMA_VERSION,
            "retained_layout_version": REPORTING_RETAINED_LAYOUT_VERSION,
            "generation": generation,
            "cell": cell,
            "start_id_ceiling": 100,
            "behavior_rows": behavior_rows,
            "lineage_records": lineage_records,
            "behavior_rows_digest_sha256": behavior_rows_digest_sha256(behavior_rows),
            "lineage_records_digest_sha256": lineage_records_digest_sha256(
                generation, lineage_records
            ),
        },
        "lifecycle": {
            "source_version": TYPED_LIFECYCLE_SCHEMA_VERSION,
            "generation": generation,
            "cell": cell,
            "event_cursor_start": 1,
            "event_cursor_pre_turnover": 2,
            "event_cursor_end": 3,
            "authoritative_event_cursor": 3,
            "event_row_count": 2,
            "active_prefix_digest_sha256": _hash(
                f"typed-lifecycle-{generation}-{cell}"
            ),
        },
        "terminal": {
            "source_version": RECOVERABLE_ERROR_SCHEMA_VERSION,
            "generation": generation,
            "cell": cell,
            "recoverable_error_code": "retained_fixture_notice",
            "recoverable_error_details": {"cell": cell, "generation": generation},
        },
    }


def test_flat_fixture_composes_exact_helpers_into_ten_wire_fields() -> None:
    source = _fixture(generation=2, cell=1)
    checkpoint = source["checkpoint"]
    projection = checkpoint["projection_fields"]
    metadata = projection["metadata"]

    actual = finalize_compact_summary_retained_digests(source)

    assert tuple(actual) == DIGEST_FIELDS
    assert len(actual) == 10
    assert all(value is not None and len(value) == 64 for value in actual.values())
    assert actual == {
        "events.active_prefix_digest_sha256": source["lifecycle"][
            "active_prefix_digest_sha256"
        ],
        "identity.active_agent_ids_digest_sha256": (
            active_agent_ids_digest_from_tensor_sha256(
                source["identity"]["active_agent_count"],
                source["identity"]["tensor_sha256"],
            )
        ),
        "rng.authority_digest_sha256": checkpoint_rng_digest_from_tensor_manifest(
            metadata=metadata,
            tensor_manifest=checkpoint["tensor_manifest"],
        ),
        "rng.policy_mutation_tape_digest_sha256": (
            policy_mutation_tape_digest_from_tensor_sha256(
                source["policy_mutation_tape"]["cursor"],
                source["policy_mutation_tape"]["consumed_shapes"],
                source["policy_mutation_tape"]["tensor_sha256"],
            )
        ),
        "terminal.recoverable_error_digest_sha256": recoverable_error_digest_sha256(
            source["terminal"]["recoverable_error_code"],
            source["terminal"]["recoverable_error_details"],
        ),
        "state_digests.authoritative_state_digest_sha256": (
            checkpoint_layer_digest_from_tensor_manifest(
                layer=DigestLayer.TRAJECTORY_STATE,
                projection_schema_version=checkpoint["projection_schema_version"],
                world=source["world"],
                generation=metadata["generation"],
                tick=metadata["tick"],
                fields=checkpoint["trajectory_state_fields"],
            )
        ),
        "state_digests.canonical_checkpoint_storage_digest_sha256": (
            checkpoint_layer_digest_from_tensor_manifest(
                layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                projection_schema_version=checkpoint["projection_schema_version"],
                world=source["world"],
                generation=metadata["generation"],
                tick=metadata["tick"],
                fields=checkpoint["canonical_checkpoint_storage_fields"],
            )
        ),
        "state_digests.next_generation_start_state_digest_sha256": (
            checkpoint_source_projection_digest_from_tensor_manifest(
                projection_fields=projection,
                tensor_manifest=checkpoint["tensor_manifest"],
            )
        ),
        "behavior_metrics.rows_digest_sha256": source["reporting"][
            "behavior_rows_digest_sha256"
        ],
        "lineage_metrics.records_digest_sha256": source["reporting"][
            "lineage_records_digest_sha256"
        ],
    }


def test_runtime_event_cursor_is_exactly_retained_and_hashed() -> None:
    baseline = _fixture()
    changed = copy.deepcopy(baseline)
    changed["checkpoint"]["projection_fields"]["runtime_event_cursor"] += 1

    baseline_digests = finalize_compact_summary_retained_digests(baseline)
    changed_digests = finalize_compact_summary_retained_digests(changed)

    assert (
        baseline_digests["state_digests.next_generation_start_state_digest_sha256"]
        != changed_digests["state_digests.next_generation_start_state_digest_sha256"]
    )
    assert {
        field
        for field in DIGEST_FIELDS
        if baseline_digests[field] != changed_digests[field]
    } == {"state_digests.next_generation_start_state_digest_sha256"}

    missing = copy.deepcopy(baseline)
    del missing["checkpoint"]["projection_fields"]["runtime_event_cursor"]
    with pytest.raises(ValueError, match="missing=\\['runtime_event_cursor'\\]"):
        finalize_compact_summary_retained_digests(missing)

    malformed = copy.deepcopy(baseline)
    malformed["checkpoint"]["projection_fields"]["runtime_event_cursor"] = True
    with pytest.raises(TypeError, match="runtime_event_cursor must be an int"):
        finalize_compact_summary_retained_digests(malformed)


def test_incomplete_patch_fixture_preserves_source_projection_fail_closed_contract() -> None:
    source = _fixture(world="patch")

    with pytest.raises(ValueError, match="Patch metadata is missing required fields"):
        finalize_compact_summary_retained_digests(source)


def test_g2_b3_authorities_are_generation_cell_isolated() -> None:
    sources = [[_fixture(generation=g, cell=b) for b in range(3)] for g in range(2)]
    baseline = [
        [finalize_compact_summary_retained_digests(source) for source in row]
        for row in sources
    ]
    changed = copy.deepcopy(sources)
    target = changed[1][2]["reporting"]
    target["behavior_rows"][0]["move_n"] += 7
    target["behavior_rows_digest_sha256"] = behavior_rows_digest_sha256(
        target["behavior_rows"]
    )

    actual = [
        [finalize_compact_summary_retained_digests(source) for source in row]
        for row in changed
    ]

    for generation in range(2):
        for cell in range(3):
            if (generation, cell) == (1, 2):
                differing = {
                    field
                    for field in DIGEST_FIELDS
                    if actual[generation][cell][field] != baseline[generation][cell][field]
                }
                assert differing == {"behavior_metrics.rows_digest_sha256"}
            else:
                assert actual[generation][cell] == baseline[generation][cell]


def test_missing_and_extra_authority_keys_fail_closed() -> None:
    missing = _fixture()
    missing.pop("lifecycle")
    with pytest.raises(CompactSummaryRetainedDigestFinalizerError, match="missing=.*lifecycle"):
        finalize_compact_summary_retained_digests(missing)

    extra = _fixture()
    extra["identity"]["default_digest"] = _hash("forbidden")
    with pytest.raises(CompactSummaryRetainedDigestFinalizerError, match="extra=.*default_digest"):
        finalize_compact_summary_retained_digests(extra)

    nested_missing = _fixture()
    nested_missing["checkpoint"].pop("trajectory_state_fields")
    with pytest.raises(
        CompactSummaryRetainedDigestFinalizerError,
        match="missing=.*trajectory_state_fields",
    ):
        finalize_compact_summary_retained_digests(nested_missing)

    descriptor_extra = _fixture()
    descriptor_extra["checkpoint"]["tensor_manifest"][
        "agent_ids_by_alive_order"
    ]["bytes"] = 8
    with pytest.raises(
        CompactSummaryRetainedDigestFinalizerError, match="extra=.*bytes"
    ):
        finalize_compact_summary_retained_digests(descriptor_extra)


def test_source_versions_and_nested_coordinates_are_exact() -> None:
    wrong_wire = _fixture()
    wrong_wire["wire_schema_version"] = "eiel_compact_generation_summary_v0"
    with pytest.raises(ValueError, match="wire_schema_version"):
        finalize_compact_summary_retained_digests(wrong_wire)

    wrong_rng_source = _fixture()
    wrong_rng_source["checkpoint"]["rng_source_version"] = (
        "eiel_checkpoint_source_rng_v0"
    )
    with pytest.raises(ValueError, match="rng_source_version"):
        finalize_compact_summary_retained_digests(wrong_rng_source)

    boolean_cell = _fixture()
    boolean_cell["reporting"]["cell"] = False
    with pytest.raises(TypeError, match="reporting.cell must be an int"):
        finalize_compact_summary_retained_digests(boolean_cell)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda source: source["lifecycle"].update(
                active_prefix_digest_sha256="A" * 64
            ),
            "lowercase SHA-256",
        ),
        (
            lambda source: source["reporting"].update(
                behavior_rows_digest_sha256="0" * 64
            ),
            "contradicts behavior rows",
        ),
        (
            lambda source: source["checkpoint"]["tensor_manifest"][
                "rng_manager_mutation_torch_cpu_state"
            ].update(sha256="bad"),
            "lowercase SHA-256",
        ),
        (
            lambda source: source["policy_mutation_tape"]["tensor_sha256"].update(
                policy_birth_delta_tape="not-a-hash"
            ),
            "lowercase SHA-256",
        ),
    ],
)
def test_malformed_or_contradictory_hashes_fail_closed(mutator: Any, match: str) -> None:
    source = _fixture()
    mutator(source)

    with pytest.raises(ValueError, match=match):
        finalize_compact_summary_retained_digests(source)


def test_missing_or_contradictory_metadata_fails_closed() -> None:
    missing = _fixture()
    missing["checkpoint"]["projection_fields"]["metadata"].pop(
        "native_runtime_numpy_runner_state"
    )
    with pytest.raises(ValueError, match="missing required fields"):
        finalize_compact_summary_retained_digests(missing)

    wrong_generation = _fixture()
    wrong_generation["checkpoint"]["projection_fields"]["metadata"]["generation"] = 9
    with pytest.raises(ValueError, match="must be the next generation"):
        finalize_compact_summary_retained_digests(wrong_generation)

    wrong_phase = _fixture()
    wrong_phase["checkpoint"]["projection_fields"]["metadata"][
        "agent_grid_runner_phase"
    ] = "partial_generation"
    with pytest.raises(ValueError, match="generation_boundary"):
        finalize_compact_summary_retained_digests(wrong_phase)


def test_finalizer_rejects_unhashed_arrays_and_has_no_transfer_surface() -> None:
    source = _fixture()
    source["checkpoint"]["projection_fields"]["metadata"]["raw_state"] = np.zeros(
        1, dtype=np.uint8
    )
    with pytest.raises(ValueError, match="transfer and prehash"):
        finalize_compact_summary_retained_digests(source)

    module = Path(
        "engine/native_runtime/compact_summary_retained_digest_finalizer.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in module
    assert ".cpu(" not in module
    assert ".to(" not in module


def test_finalizer_does_not_mutate_inputs_or_rng_state() -> None:
    source = _fixture(generation=1, cell=2)
    before_source = copy.deepcopy(source)
    random.seed(42)
    np.random.seed(42)
    before_python_rng = random.getstate()
    before_numpy_rng = np.random.get_state()

    finalize_compact_summary_retained_digests(source)

    assert source == before_source
    assert random.getstate() == before_python_rng
    after_numpy_rng = np.random.get_state()
    assert after_numpy_rng[0] == before_numpy_rng[0]
    assert np.array_equal(after_numpy_rng[1], before_numpy_rng[1])
    assert after_numpy_rng[2:] == before_numpy_rng[2:]
