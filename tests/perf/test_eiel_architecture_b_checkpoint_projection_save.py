from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.campaigns.providers import (
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
)
from native_runtime import decode_bundled_native_checkpoint_projection
from native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from simulation import checkpoint_bundle_v2 as bundle_module
from simulation.checkpoint_bundle_v2 import (
    ARCHITECTURE_B_BUNDLE_KIND,
    ARCHITECTURE_B_METRICS_RESTORE_RECEIPT_SCHEMA_VERSION,
    ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION,
    ARCHITECTURE_B_RESUME_ENVELOPE_SCHEMA_VERSION,
    CheckpointStrictError,
    architecture_b_metrics_logger_state_from_payload,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save_projection as _checkpoint_bundle_v2_save_projection,
    decode_architecture_b_resume_envelope,
    restore_architecture_b_metrics_logger_state,
)
from simulation.control_plane_pkg.util_json import canonical_json_bytes_lossless
from simulation.metrics_pkg.lineage import LineageMetricsState
from simulation.metrics_pkg.lineage_forensics import LineageCollapseForensicsState
from simulation.metrics_pkg.metrics_logger import MetricsLogger
from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
    _projection,
)


_DELETE = object()


def _metrics_logger_state() -> dict[str, Any]:
    lineage_state = LineageMetricsState(
        collapse_streak=1,
        population_dominance_streak=2,
        reproduction_dominance_streak=3,
        last_dominant_lineage_id="lineage-a",
        seen_lineages={"lineage-a", "lineage-b"},
        lineage_presence_counts={"lineage-a": 2, "lineage-b": 1},
        lineage_first_seen_generation={"lineage-a": 0, "lineage-b": 1},
    )
    return {
        "schema_version": "metrics_state_v1",
        "collapse_streak": 4,
        "theme5_lineage_state": lineage_state.to_dict(),
        "lineage_collapse_forensics_state": (
            LineageCollapseForensicsState().to_dict()
        ),
        "early_lineage_effective_gen_1": 2.0,
        "early_lineage_effective_gen_2": None,
        "early_lineage_top1_share_gen_1": 0.625,
        "early_lineage_top1_share_gen_2": None,
    }


def checkpoint_bundle_v2_save_projection(
    *args: Any,
    **kwargs: Any,
) -> str:
    kwargs.setdefault("metrics_logger_state", _metrics_logger_state())
    return _checkpoint_bundle_v2_save_projection(*args, **kwargs)


def _source_digest(projection: dict[str, Any]) -> str:
    return str(
        checkpoint_source_projection_authority_manifest(projection)[
            "source_projection_sha256"
        ]
    )


def _projection_on_cuda(projection: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(projection)
    out["tensors"] = {
        str(name): tensor.detach().to("cuda").clone()
        for name, tensor in projection["tensors"].items()
    }
    out["tensor_manifest"] = {
        str(name): {
            **dict(row),
            "device": str(out["tensors"][name].device),
            "is_cuda": True,
        }
        for name, row in dict(projection["tensor_manifest"]).items()
    }
    return out


def _identity(*, generation: int, source_digest: str) -> dict[str, Any]:
    return {
        "run_id": "architecture-b-projection-save",
        "controller_id": "architecture-b-controller-1",
        "owner_epoch": 7,
        "snapshot_id": "architecture-b-controller-1:owner:7:boundary:2",
        "snapshot_sequence": 2,
        "snapshot_digest_sha256": "2" * 64,
        "next_generation": generation,
        "tick": 0,
        "phase": "generation_boundary",
        "source_projection_sha256": source_digest,
    }


def _observer_continuation(*, generation: int) -> dict[str, Any]:
    completed_generation = generation - 1
    sequence = 2
    return {
        "schema_version": "eiel_architecture_b_observer_continuation_v1",
        "checkpoint_next_generation": generation,
        "tick": 0,
        "typed_lifecycle_event_cursor": 0,
        "exact_reporting_boundary_ordinal": sequence,
        "behavior_completed_generation_and_row_key": {
            "completed_generation": completed_generation,
            "last_row_key": f"generation:{completed_generation}:agent:1",
        },
        "metrics_completed_generation": completed_generation,
        "lineage_completed_generation_and_row_key": {
            "completed_generation": completed_generation,
            "last_row_key": f"generation:{completed_generation}:lineage:1",
        },
        "chronicle_observation_ordinal": sequence,
        "progress_sequence": sequence,
        "schedule_phase": {
            "origin_generation": 0,
            "next_generation": generation,
        },
        "deduplication": {
            "acknowledged_record_keys": [
                f"generation:{completed_generation}:exact_reporting"
            ]
        },
        "acknowledged_channel_cursor_ranges": {
            "exact_reporting": [completed_generation, generation],
            "behaviors": [completed_generation, generation],
            "metrics": [completed_generation, generation],
            "lineage": [completed_generation, generation],
            "typed_lifecycle": [0, 0],
            "chronicle": [sequence - 1, sequence],
            "checkpoint_generation": [generation, generation + 1],
        },
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, payload: Any) -> None:
    payload_bytes = canonical_json_bytes_lossless(payload)
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(payload_bytes, mtime=0))
    else:
        path.write_bytes(payload_bytes)


def _read_json(path: Path) -> Any:
    if path.name.endswith(".gz"):
        return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _native_sidecar_path(bundle: Path) -> Path:
    path = bundle / bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME
    if path.exists():
        return path
    return bundle / bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME


def _strip_architecture_b_markers_to_valid_generic_native(bundle: Path) -> None:
    paths = {
        "metadata": bundle / "metadata.json",
        "sidecar": _native_sidecar_path(bundle),
        "coherence": bundle / "native_checkpoint_bundle_coherence_report.json",
        "completion": bundle / "bundle_complete.json",
    }
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    sidecar = _read_json(paths["sidecar"])
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))

    sidecar.pop("source")
    sidecar.pop("projection_input_schema_version")
    for field in (
        "architecture_b_projection_save_schema_version",
        "architecture_b_identity_sha256",
        "observer_continuation_sha256",
    ):
        metadata.pop(field)
    for field in (
        "architecture_b_schema_version",
        "architecture_b_identity",
        "architecture_b_identity_sha256",
        "observer_continuation",
        "observer_continuation_sha256",
        "architecture_b_completion_sha256",
    ):
        completion.pop(field)

    metadata["native_runtime_checkpoint_hash"] = hashlib.sha256(
        canonical_json_bytes_lossless(sidecar)
    ).hexdigest()
    metadata.pop("native_checkpoint_bundle_coherence_hash")
    projection = decode_bundled_native_checkpoint_projection(sidecar)
    resident = json.loads(
        (bundle / "resident_runtime_state_v2.json").read_text(encoding="utf-8")
    )
    compatibility_payloads = {
        "world": json.loads((bundle / "world.json").read_text(encoding="utf-8")),
        "population": json.loads(
            (bundle / "population.json").read_text(encoding="utf-8")
        ),
        "rng_snapshot_v2": json.loads(
            (bundle / "rng_snapshot_v2.json").read_text(encoding="utf-8")
        ),
        "runner_state": json.loads(
            (bundle / "runner_state.json").read_text(encoding="utf-8")
        ),
        "control_plane": json.loads(
            (bundle / "control_plane.json").read_text(encoding="utf-8")
        ),
        "metrics_state": json.loads(
            (bundle / "metrics_state.json").read_text(encoding="utf-8")
        ),
        "weights_info": dict(metadata.get("weights_info") or {}),
    }
    coherence = bundle_module._native_checkpoint_bundle_coherence_report(
        metadata=metadata,
        projection=projection,
        sidecar=sidecar,
        resident=resident,
        compatibility_payloads=compatibility_payloads,
    )
    assert coherence["ok"] is True
    coherence_digest = hashlib.sha256(
        canonical_json_bytes_lossless(coherence)
    ).hexdigest()
    metadata["native_checkpoint_bundle_coherence_hash"] = coherence_digest
    completion["coherence_sha256"] = coherence_digest

    _write_json(paths["sidecar"], sidecar)
    _write_json(paths["metadata"], metadata)
    _write_json(paths["coherence"], coherence)
    _write_json(paths["completion"], completion)


def _mutate_path(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: dict[str, Any] = payload
    for field in path[:-1]:
        target = target[field]
    if value is _DELETE:
        target.pop(path[-1])
    else:
        target[path[-1]] = value


def _assert_no_transaction_paths(bundle: Path) -> None:
    assert not Path(f"{bundle}.tmp").exists()
    assert not Path(f"{bundle}.previous").exists()
    assert not list(bundle.parent.glob(f".{bundle.name}.tmp-*"))


def test_tick_native_absolute_boundary_observer_continuation_roundtrips_strictly(
    tmp_path: Path,
) -> None:
    """An absolute tick frontier must not be treated as a generation ordinal."""

    generation = 7
    absolute_next_tick = 1120
    projection = _projection(generation=generation, tick=0)
    source_digest = _source_digest(projection)
    controller_id = "architecture-b-tick-native-cell-0"
    identity = {
        "run_id": "architecture-b-tick-native-boundary",
        "controller_id": controller_id,
        "owner_epoch": absolute_next_tick,
        "snapshot_id": (
            f"{controller_id}:owner:{absolute_next_tick}:boundary:"
            f"{absolute_next_tick}"
        ),
        "snapshot_sequence": absolute_next_tick,
        "snapshot_digest_sha256": source_digest,
        "next_generation": generation,
        "tick": 0,
        "phase": "generation_boundary",
        "source_projection_sha256": source_digest,
    }
    continuation = ProductionEngineLifecycleV1._observer_continuation(
        generation,
        tick=0,
        event_cursor=0,
        snapshot_sequence=absolute_next_tick,
        absolute_tick_authority=True,
    )
    assert continuation["schedule_phase"] == {
        "origin_generation": generation,
        "next_generation": generation,
    }

    bundle = tmp_path / "tick-native-boundary"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=generation,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )

    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    assert payload["bundle_complete"]["architecture_b_identity"][
        "snapshot_sequence"
    ] == absolute_next_tick
    assert payload["bundle_complete"]["observer_continuation"]["schedule_phase"] == {
        "origin_generation": generation,
        "next_generation": generation,
    }


def test_projection_save_persists_detached_authority_and_completion_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    continuation = _observer_continuation(generation=2)
    expected_continuation = copy.deepcopy(continuation)
    metrics_logger_state = _metrics_logger_state()
    expected_metrics_logger_state = copy.deepcopy(metrics_logger_state)

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("projection save must not export or apply live owner state")

    monkeypatch.setattr(
        bundle_module,
        "_export_native_runtime_checkpoint_projection",
        _forbidden,
    )
    monkeypatch.setattr(
        bundle_module,
        "_apply_native_runtime_checkpoint_projection",
        _forbidden,
    )

    bundle = tmp_path / "bundle"
    publish_calls: list[tuple[str, str]] = []
    original_publish = bundle_module._publish_directory_absent

    def _record_publish(source: str, destination: str) -> None:
        staged = Path(source)
        assert staged.parent == bundle.resolve().parent
        assert staged.name.startswith(f".{bundle.name}.tmp-")
        assert destination == str(bundle.resolve())
        assert not bundle.exists()
        publish_calls.append((source, destination))
        original_publish(source, destination)

    monkeypatch.setattr(bundle_module, "_publish_directory_absent", _record_publish)
    result = checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
        metrics_logger_state=metrics_logger_state,
    )
    assert result == str(bundle.resolve())
    assert len(publish_calls) == 1
    assert publish_calls[0][1] == str(bundle.resolve())

    projection["tensors"]["energy"][0] = -999.0
    identity["run_id"] = "mutated-after-save"
    continuation["typed_lifecycle_event_cursor"] = 999
    metrics_logger_state["early_lineage_effective_gen_1"] = 999.0

    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    completion = payload["bundle_complete"]
    assert completion["architecture_b_schema_version"] == (
        ARCHITECTURE_B_PROJECTION_SAVE_SCHEMA_VERSION
    )
    assert completion["architecture_b_identity"]["run_id"] == (
        "architecture-b-projection-save"
    )
    assert completion["architecture_b_identity"]["next_generation"] == 2
    assert completion["observer_continuation"]["checkpoint_next_generation"] == 2
    assert completion["observer_continuation"]["typed_lifecycle_event_cursor"] == 0
    assert completion["observer_continuation"] == expected_continuation
    assert len(completion["architecture_b_completion_sha256"]) == 64
    restored = decode_bundled_native_checkpoint_projection(
        payload["native_runtime_checkpoint_projection"]
    )
    torch.testing.assert_close(
        restored["tensors"]["energy"][0],
        torch.tensor(13.5, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    resume_envelope = decode_architecture_b_resume_envelope(payload)
    assert architecture_b_metrics_logger_state_from_payload(payload) == (
        expected_metrics_logger_state
    )
    assert resume_envelope["schema_version"] == (
        ARCHITECTURE_B_RESUME_ENVELOPE_SCHEMA_VERSION
    )
    assert resume_envelope["bundle_kind"] == ARCHITECTURE_B_BUNDLE_KIND
    assert resume_envelope["architecture_b_identity"] == completion[
        "architecture_b_identity"
    ]
    assert resume_envelope["observer_continuation"] == expected_continuation
    assert "observer_continuation" not in resume_envelope["checkpoint_projection"][
        "metadata"
    ]
    assert resume_envelope["controller_import_metadata"][
        "observer_continuation"
    ] == expected_continuation
    resume_envelope["controller_import_metadata"]["observer_continuation"][
        "typed_lifecycle_event_cursor"
    ] = 999
    assert restored["metadata"].get("observer_continuation") is None
    with pytest.raises(CheckpointStrictError, match="generic checkpoint_bundle_v2_apply"):
        checkpoint_bundle_v2_apply(object(), payload)
    _assert_no_transaction_paths(bundle)


def test_projection_save_roundtrips_partial_generation_cursor(tmp_path: Path) -> None:
    projection = _projection(generation=2, tick=1)
    source_digest = _source_digest(projection)
    sequence = 321
    identity = {
        "run_id": "architecture-b-partial-save",
        "controller_id": "architecture-b-controller-1",
        "owner_epoch": sequence,
        "snapshot_id": (
            f"architecture-b-controller-1:owner:{sequence}:boundary:{sequence}"
        ),
        "snapshot_sequence": sequence,
        "snapshot_digest_sha256": "2" * 64,
        "next_generation": 2,
        "tick": 1,
        "phase": "partial_generation",
        "source_projection_sha256": source_digest,
    }
    continuation = _observer_continuation(generation=2)
    continuation.update(
        {
            "tick": 1,
            "exact_reporting_boundary_ordinal": sequence,
            "chronicle_observation_ordinal": sequence,
            "progress_sequence": sequence,
            "schedule_phase": {"origin_generation": 2, "next_generation": 2},
        }
    )
    continuation["acknowledged_channel_cursor_ranges"]["chronicle"] = [
        sequence - 1,
        sequence,
    ]

    bundle = tmp_path / "bundle-tick-321"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )
    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    resume = decode_architecture_b_resume_envelope(payload)

    assert resume["architecture_b_identity"]["tick"] == 1
    assert resume["architecture_b_identity"]["phase"] == "partial_generation"
    assert resume["checkpoint_projection"]["metadata"]["generation"] == 2
    assert resume["checkpoint_projection"]["metadata"]["tick"] == 1
    _assert_no_transaction_paths(bundle)


def test_projection_save_defaults_neutral_and_restores_exact_metrics_logger_state(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    compatibility_bundle = tmp_path / "compatibility-bundle"
    bundle = tmp_path / "bundle"
    expected_state = _metrics_logger_state()

    _checkpoint_bundle_v2_save_projection(
        projection,
        str(compatibility_bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=source_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
    )
    compatibility_payload = checkpoint_bundle_v2_load(
        str(compatibility_bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    compatibility_state = architecture_b_metrics_logger_state_from_payload(
        compatibility_payload
    )
    assert compatibility_state["early_lineage_effective_gen_1"] is None
    assert compatibility_state["theme5_lineage_state"]["seen_lineages"] == []

    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=source_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
        metrics_logger_state=expected_state,
    )
    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    logger = MetricsLogger(str(tmp_path / "manual-resume"))
    runner = SimpleNamespace(logger=logger)

    receipt = restore_architecture_b_metrics_logger_state(runner, payload)

    assert receipt["schema_version"] == (
        ARCHITECTURE_B_METRICS_RESTORE_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["status"] == "restored_and_verified"
    assert logger.export_state() == expected_state
    assert logger.export_state()["early_lineage_effective_gen_1"] == 2.0
    assert logger.export_state()["early_lineage_top1_share_gen_1"] == 0.625
    assert logger.export_state()["theme5_lineage_state"][
        "last_dominant_lineage_id"
    ] == "lineage-a"

    payload["metrics_state"]["early_lineage_effective_gen_1"] = 99.0
    with pytest.raises(CheckpointStrictError, match="changed"):
        architecture_b_metrics_logger_state_from_payload(payload)


def test_native_projection_checkpoint_restores_hash_bound_metrics_state(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "native-projection-bundle"
    expected_state = _metrics_logger_state()

    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=source_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
        metrics_logger_state=expected_state,
    )
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)

    assert payload["metadata"]["metrics_state_hash"] == hashlib.sha256(
        bundle_module.canonical_json_bytes(payload["metrics_state"])
    ).hexdigest()
    assert architecture_b_metrics_logger_state_from_payload(payload) == expected_state

    payload["metrics_state"]["early_lineage_effective_gen_1"] = 99.0
    with pytest.raises(CheckpointStrictError, match="metrics_state hash mismatch"):
        architecture_b_metrics_logger_state_from_payload(payload)


def test_projection_save_fails_closed_on_generation_tick_and_digest(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    continuation = _observer_continuation(generation=2)

    cases = [
        (
            "generation",
            projection,
            3,
            source_digest,
            "generation contradiction",
        ),
        (
            "tick",
            _projection(generation=2, tick=1),
            2,
            _source_digest(_projection(generation=2, tick=1)),
            "tick contradiction",
        ),
        (
            "malformed-digest",
            projection,
            2,
            "A" * 64,
            "64 lowercase hexadecimal",
        ),
        (
            "digest-mismatch",
            projection,
            2,
            "0" * 64,
            "digest mismatch",
        ),
    ]
    for name, candidate, generation, expected_digest, error in cases:
        bundle = tmp_path / name
        with pytest.raises(CheckpointStrictError, match=error):
            checkpoint_bundle_v2_save_projection(
                candidate,
                str(bundle),
                generation_index=generation,
                expected_source_projection_sha256=expected_digest,
                architecture_b_identity=identity,
                observer_continuation=continuation,
            )
        assert not bundle.exists()
        _assert_no_transaction_paths(bundle)


def test_resume_decoder_requires_exact_strict_loader_payload_provenance(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=source_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
    )

    def _strict_load() -> dict[str, Any]:
        return checkpoint_bundle_v2_load(
            str(bundle),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )

    payload = _strict_load()
    decoded = decode_architecture_b_resume_envelope(payload)
    copied_payload = dict(payload)
    synthetic_payload = {
        "bundle_kind": payload["bundle_kind"],
        "metadata": copy.deepcopy(payload["metadata"]),
        "bundle_complete": copy.deepcopy(payload["bundle_complete"]),
        "architecture_b_resume_envelope": copy.deepcopy(
            payload["architecture_b_resume_envelope"]
        ),
    }
    assert synthetic_payload["bundle_complete"][
        "architecture_b_completion_sha256"
    ] == decoded["bundle_completion_authority"][
        "architecture_b_completion_sha256"
    ]
    for forged in (copied_payload, synthetic_payload):
        with pytest.raises(CheckpointStrictError, match="exact strict-loader"):
            decode_architecture_b_resume_envelope(forged)

    identity_mutated = _strict_load()
    identity_mutated["bundle_complete"]["architecture_b_identity"][
        "snapshot_digest_sha256"
    ] = "f" * 64
    with pytest.raises(CheckpointStrictError, match="changed after strict loading"):
        decode_architecture_b_resume_envelope(identity_mutated)

    continuation_mutated = _strict_load()
    continuation_mutated["bundle_complete"]["observer_continuation"][
        "progress_sequence"
    ] = 3
    with pytest.raises(CheckpointStrictError, match="changed after strict loading"):
        decode_architecture_b_resume_envelope(continuation_mutated)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_projection_save_authority_digest_survives_cuda_to_cpu_sidecar_decode(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    projection["metadata"]["native_runtime_python_random_state"] = (
        3,
        (1, 2, 3),
        None,
    )
    projection = _projection_on_cuda(projection)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"

    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=source_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
    )

    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    sidecar = payload["native_runtime_checkpoint_projection"]
    restored = decode_bundled_native_checkpoint_projection(sidecar)
    assert _source_digest(restored) == sidecar["source_projection_sha256"]


def test_projection_save_strict_staging_failure_preserves_existing_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    continuation = _observer_continuation(generation=2)
    bundle = tmp_path / "bundle"

    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )
    before = _tree_bytes(bundle)

    def _reject_staging(*_args: Any, **_kwargs: Any) -> None:
        raise CheckpointStrictError("deliberate staging verification failure")

    monkeypatch.setattr(
        bundle_module,
        "_validate_architecture_b_completion_manifest",
        _reject_staging,
    )
    with pytest.raises(CheckpointStrictError, match="staging verification failure"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=identity,
            observer_continuation=continuation,
        )

    assert _tree_bytes(bundle) == before
    _assert_no_transaction_paths(bundle)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", None, "observer_continuation.schema_version"),
        ("schema_version", "wrong", "observer_continuation.schema_version"),
        ("checkpoint_next_generation", 3, "checkpoint_next_generation"),
        ("tick", 1, "observer_continuation.tick"),
        ("typed_lifecycle_event_cursor", 1, "typed_lifecycle_event_cursor"),
        ("exact_reporting_boundary_ordinal", 3, "exact_reporting_boundary_ordinal"),
        ("chronicle_observation_ordinal", 3, "chronicle_observation_ordinal"),
        ("progress_sequence", 3, "progress_sequence"),
    ],
)
def test_projection_save_requires_exact_observer_continuation_contract(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    continuation = _observer_continuation(generation=2)
    if value is None:
        continuation.pop(field)
    else:
        continuation[field] = value

    bundle = tmp_path / f"invalid-{field}"
    expected_error = "missing required fields" if value is None else error
    with pytest.raises(CheckpointStrictError, match=expected_error):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(
                generation=2,
                source_digest=source_digest,
            ),
            observer_continuation=continuation,
        )
    assert not bundle.exists()
    _assert_no_transaction_paths(bundle)


@pytest.mark.parametrize(
    ("name", "path", "value", "error"),
    [
        (
            "behavior-object",
            ("behavior_completed_generation_and_row_key",),
            [],
            "behavior_completed_generation_and_row_key must be an object",
        ),
        (
            "behavior-generation-type",
            ("behavior_completed_generation_and_row_key", "completed_generation"),
            True,
            "behavior_completed_generation_and_row_key.completed_generation",
        ),
        (
            "behavior-generation-relation",
            ("behavior_completed_generation_and_row_key", "completed_generation"),
            0,
            "behavior_completed_generation_and_row_key.completed_generation contradiction",
        ),
        (
            "behavior-row-key-type",
            ("behavior_completed_generation_and_row_key", "last_row_key"),
            1,
            "behavior_completed_generation_and_row_key.last_row_key",
        ),
        (
            "behavior-row-key-relation",
            ("behavior_completed_generation_and_row_key", "last_row_key"),
            "generation:0:agent:1",
            "last_row_key must identify",
        ),
        (
            "metrics-type",
            ("metrics_completed_generation",),
            True,
            "metrics_completed_generation must be an integer",
        ),
        (
            "metrics-relation",
            ("metrics_completed_generation",),
            0,
            "metrics_completed_generation contradiction",
        ),
        (
            "lineage-object",
            ("lineage_completed_generation_and_row_key",),
            [],
            "lineage_completed_generation_and_row_key must be an object",
        ),
        (
            "lineage-generation-type",
            ("lineage_completed_generation_and_row_key", "completed_generation"),
            True,
            "lineage_completed_generation_and_row_key.completed_generation",
        ),
        (
            "lineage-generation-relation",
            ("lineage_completed_generation_and_row_key", "completed_generation"),
            0,
            "lineage_completed_generation_and_row_key.completed_generation contradiction",
        ),
        (
            "lineage-row-key-relation",
            ("lineage_completed_generation_and_row_key", "last_row_key"),
            "generation:1:agent:1",
            "last_row_key must identify",
        ),
        (
            "schedule-object",
            ("schedule_phase",),
            [],
            "schedule_phase must be an object",
        ),
        (
            "schedule-origin-negative",
            ("schedule_phase", "origin_generation"),
            -1,
            "schedule_phase.origin_generation",
        ),
        (
            "schedule-origin-relation",
            ("schedule_phase", "origin_generation"),
            1,
            "origin_generation contradicts",
        ),
        (
            "schedule-next-relation",
            ("schedule_phase", "next_generation"),
            3,
            "schedule_phase.next_generation contradiction",
        ),
        (
            "dedup-object",
            ("deduplication",),
            [],
            "deduplication must be an object",
        ),
        (
            "dedup-list-type",
            ("deduplication", "acknowledged_record_keys"),
            (),
            "acknowledged_record_keys must be an array",
        ),
        (
            "dedup-key-type",
            ("deduplication", "acknowledged_record_keys"),
            [1],
            "acknowledged_record_keys\\[0\\]",
        ),
        (
            "dedup-duplicate",
            ("deduplication", "acknowledged_record_keys"),
            ["generation:1:exact_reporting", "generation:1:exact_reporting"],
            "acknowledged_record_keys must be unique",
        ),
        (
            "dedup-future",
            ("deduplication", "acknowledged_record_keys"),
            ["generation:2:exact_reporting"],
            "outside the acknowledged continuation frontier",
        ),
        (
            "ranges-object",
            ("acknowledged_channel_cursor_ranges",),
            [],
            "acknowledged_channel_cursor_ranges must be an object",
        ),
        (
            "range-type",
            ("acknowledged_channel_cursor_ranges", "behaviors"),
            (1, 2),
            "behaviors must be a two-item array",
        ),
        (
            "range-bound-type",
            ("acknowledged_channel_cursor_ranges", "metrics"),
            [True, 2],
            "metrics\\[0\\] must be an integer",
        ),
        (
            "range-negative",
            ("acknowledged_channel_cursor_ranges", "lineage"),
            [-1, 2],
            "lineage\\[0\\] must be an integer",
        ),
        (
            "range-reversed",
            ("acknowledged_channel_cursor_ranges", "typed_lifecycle"),
            [1, 0],
            "typed_lifecycle must be a non-reversed",
        ),
        (
            "range-lifecycle-end",
            ("acknowledged_channel_cursor_ranges", "typed_lifecycle"),
            [0, 1],
            "typed_lifecycle must end",
        ),
        *[
            (
                f"range-{channel}-relation",
                ("acknowledged_channel_cursor_ranges", channel),
                [0, 0],
                rf"{channel} contradiction",
            )
            for channel in (
                "exact_reporting",
                "behaviors",
                "metrics",
                "lineage",
                "chronicle",
                "checkpoint_generation",
            )
        ],
    ],
)
def test_projection_save_rejects_every_continuation_contradiction_class(
    tmp_path: Path,
    name: str,
    path: tuple[str, ...],
    value: Any,
    error: str,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    continuation = _observer_continuation(generation=2)
    _mutate_path(continuation, path, value)
    bundle = tmp_path / name

    with pytest.raises(CheckpointStrictError, match=error):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(
                generation=2,
                source_digest=source_digest,
            ),
            observer_continuation=continuation,
        )
    assert not bundle.exists()
    _assert_no_transaction_paths(bundle)


@pytest.mark.parametrize(
    ("name", "path", "unknown"),
    [
        ("top", (), "unknown_top_level"),
        ("behavior", ("behavior_completed_generation_and_row_key",), "unknown"),
        ("lineage", ("lineage_completed_generation_and_row_key",), "unknown"),
        ("schedule", ("schedule_phase",), "unknown"),
        ("dedup", ("deduplication",), "unknown"),
        ("ranges", ("acknowledged_channel_cursor_ranges",), "unknown"),
    ],
)
def test_projection_save_rejects_unknown_continuation_fields(
    tmp_path: Path,
    name: str,
    path: tuple[str, ...],
    unknown: str,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    continuation = _observer_continuation(generation=2)
    target = continuation
    for field in path:
        target = target[field]
    target[unknown] = 0
    bundle = tmp_path / f"unknown-{name}"

    with pytest.raises(CheckpointStrictError, match="unknown fields"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(generation=2, source_digest=source_digest),
            observer_continuation=continuation,
        )
    _assert_no_transaction_paths(bundle)


@pytest.mark.parametrize(
    ("name", "path", "field"),
    [
        ("top-behavior", (), "behavior_completed_generation_and_row_key"),
        ("behavior", ("behavior_completed_generation_and_row_key",), "last_row_key"),
        ("lineage", ("lineage_completed_generation_and_row_key",), "last_row_key"),
        ("schedule", ("schedule_phase",), "origin_generation"),
        ("dedup", ("deduplication",), "acknowledged_record_keys"),
        ("ranges", ("acknowledged_channel_cursor_ranges",), "metrics"),
    ],
)
def test_projection_save_rejects_missing_nested_continuation_fields(
    tmp_path: Path,
    name: str,
    path: tuple[str, ...],
    field: str,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    continuation = _observer_continuation(generation=2)
    target = continuation
    for part in path:
        target = target[part]
    target.pop(field)
    bundle = tmp_path / f"missing-{name}"

    with pytest.raises(CheckpointStrictError, match="missing required fields"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(generation=2, source_digest=source_digest),
            observer_continuation=continuation,
        )
    _assert_no_transaction_paths(bundle)


def test_projection_save_requires_exact_snapshot_identity(tmp_path: Path) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    identity["snapshot_id"] = "architecture-b-controller-1:owner:7:boundary:3"
    bundle = tmp_path / "invalid-snapshot-identity"

    with pytest.raises(CheckpointStrictError, match="snapshot_id contradicts"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=identity,
            observer_continuation=_observer_continuation(generation=2),
        )
    assert not bundle.exists()
    _assert_no_transaction_paths(bundle)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_projection_save_requires_closed_snapshot_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    if mutation == "missing":
        identity.pop("run_id")
        expected = "missing required fields"
    else:
        identity["snapshot_digest"] = identity["snapshot_digest_sha256"]
        expected = "unknown fields"

    bundle = tmp_path / f"identity-{mutation}"
    with pytest.raises(CheckpointStrictError, match=expected):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=identity,
            observer_continuation=_observer_continuation(generation=2),
        )
    _assert_no_transaction_paths(bundle)


def test_strict_load_rejects_tampered_observer_continuation(tmp_path: Path) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )

    completion_path = bundle / "bundle_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["observer_continuation"]["typed_lifecycle_event_cursor"] = 1
    continuation_hash = hashlib.sha256(
        canonical_json_bytes_lossless(completion["observer_continuation"])
    ).hexdigest()
    completion["observer_continuation_sha256"] = continuation_hash
    completion["architecture_b_completion_sha256"] = (
        bundle_module._architecture_b_completion_sha256(completion)
    )
    _write_json(completion_path, completion)
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["observer_continuation_sha256"] = continuation_hash
    _write_json(metadata_path, metadata)

    with pytest.raises(CheckpointStrictError, match="typed_lifecycle_event_cursor"):
        checkpoint_bundle_v2_load(
            str(bundle),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )


def test_strict_load_uses_sidecar_provenance_to_reject_stripped_architecture_b_metadata(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )

    sidecar = _read_json(_native_sidecar_path(bundle))
    assert sidecar["source"] == (
        bundle_module.ARCHITECTURE_B_PROJECTION_INPUT_SOURCE
    )

    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in (
        "checkpoint_authority_schema_version",
        "architecture_b_projection_save_schema_version",
        "architecture_b_identity_sha256",
        "observer_continuation_sha256",
    ):
        metadata.pop(field)
    _write_json(metadata_path, metadata)

    completion_path = bundle / "bundle_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for field in (
        "architecture_b_schema_version",
        "architecture_b_identity",
        "architecture_b_identity_sha256",
        "observer_continuation",
        "observer_continuation_sha256",
        "architecture_b_completion_sha256",
    ):
        completion.pop(field)
    _write_json(completion_path, completion)

    with pytest.raises(CheckpointStrictError, match="authority metadata marker is invalid"):
        checkpoint_bundle_v2_load(
            str(bundle),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )


def test_required_architecture_b_kind_prevents_full_marker_stripping_downgrade(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )
    _strip_architecture_b_markers_to_valid_generic_native(bundle)

    generic_payload = checkpoint_bundle_v2_load(str(bundle), strict=True)
    assert "architecture_b_resume_envelope" not in generic_payload
    with pytest.raises(CheckpointStrictError, match="Architecture B"):
        checkpoint_bundle_v2_load(
            str(bundle),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )


def test_strict_load_accepts_legacy_loose_native_projection_sidecar(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )

    compressed_sidecar = (
        bundle / bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME
    )
    legacy_sidecar = (
        bundle / bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_FILENAME
    )
    assert compressed_sidecar.is_file()
    legacy_sidecar.write_bytes(gzip.decompress(compressed_sidecar.read_bytes()))
    compressed_sidecar.unlink()

    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    assert payload["native_runtime_checkpoint_projection"]["status"] == "present"


@pytest.mark.parametrize(
    ("filename", "payload_name"),
    [
        ("world.json", "world"),
        ("population.json", "population"),
        ("rng_snapshot_v2.json", "rng_snapshot_v2"),
        ("control_plane.json", "control_plane"),
        ("runner_state.json", "runner_state"),
        ("metrics_state.json", "metrics_state"),
        ("resident_runtime_state_v2.json", "resident_runtime_state_v2"),
        (
            bundle_module.NATIVE_RUNTIME_CHECKPOINT_PROJECTION_GZIP_FILENAME,
            "native_runtime_checkpoint_projection",
        ),
    ],
)
def test_strict_load_recomputes_every_architecture_b_payload_hash(
    tmp_path: Path,
    filename: str,
    payload_name: str,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / payload_name
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )

    payload_path = bundle / filename
    payload = _read_json(payload_path)
    payload["architecture_b_audit_tamper"] = payload_name
    _write_json(payload_path, payload)

    with pytest.raises(CheckpointStrictError, match=rf"{payload_name} hash mismatch"):
        checkpoint_bundle_v2_load(
            str(bundle),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )


def test_strict_load_recomputes_canonical_checkpoint_storage_manifest(
    tmp_path: Path,
) -> None:
    mutations = (
        (
            "digest",
            "canonical_checkpoint_storage_digest",
            lambda metadata: "0" * 64,
            "canonical storage manifest digest mismatch",
        ),
        (
            "field-count",
            "canonical_checkpoint_storage_field_count",
            lambda metadata: int(metadata["canonical_checkpoint_storage_field_count"]) + 1,
            "canonical storage manifest field count mismatch",
        ),
        (
            "schema",
            "canonical_checkpoint_storage_schema_version",
            lambda metadata: "unsupported",
            "canonical storage manifest schema mismatch",
        ),
    )
    for name, field, replacement, error in mutations:
        projection = _projection(generation=2, tick=0)
        source_digest = _source_digest(projection)
        bundle = tmp_path / name
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(generation=2, source_digest=source_digest),
            observer_continuation=_observer_continuation(generation=2),
        )
        metadata_path = bundle / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[field] = replacement(metadata)
        _write_json(metadata_path, metadata)

        with pytest.raises(CheckpointStrictError, match=error):
            checkpoint_bundle_v2_load(
                str(bundle),
                strict=True,
                required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
            )


def test_projection_save_fsyncs_every_file_and_publication_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    fsynced_files: set[str] = set()
    fsynced_directory_contexts: list[str] = []
    original_file_fsync = bundle_module._flush_and_fsync_file
    original_directory_fsync = bundle_module._fsync_directory

    def _record_file_fsync(handle: Any, *, path: str) -> None:
        fsynced_files.add(Path(path).name)
        original_file_fsync(handle, path=path)

    def _record_directory_fsync(path: str, *, context: str) -> None:
        fsynced_directory_contexts.append(context)
        original_directory_fsync(path, context=context)

    monkeypatch.setattr(bundle_module, "_flush_and_fsync_file", _record_file_fsync)
    monkeypatch.setattr(bundle_module, "_fsync_directory", _record_directory_fsync)
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )

    assert fsynced_files == {path.name for path in bundle.iterdir() if path.is_file()}
    assert "staged child directory" in fsynced_directory_contexts
    assert "staged directory" in fsynced_directory_contexts
    assert "publication parent" in fsynced_directory_contexts


def test_projection_save_reports_complete_post_capture_stage_telemetry(
    tmp_path: Path,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    telemetry: dict[str, float] = {}

    checkpoint_bundle_v2_save_projection(
        projection,
        str(tmp_path / "bundle"),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=source_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
        performance_telemetry=telemetry,
    )

    assert set(telemetry) == {
        "checkpoint_clone_materialization",
        "checkpoint_canonical_pack",
        "checkpoint_sidecar_validation",
        "checkpoint_serialize",
        "checkpoint_hash",
        "checkpoint_write",
        "checkpoint_atomic_replace",
        "checkpoint_digest_coherence",
        "checkpoint_projection_gzip",
        "checkpoint_seal_write",
        "checkpoint_strict_readback",
        "checkpoint_staged_tree_fsync",
        "checkpoint_total",
    }
    assert all(value >= 0.0 for value in telemetry.values())
    assert telemetry["checkpoint_total"] >= telemetry["checkpoint_atomic_replace"]
    assert telemetry["checkpoint_write"] >= telemetry["checkpoint_seal_write"]


def test_provider_writer_reports_end_to_end_post_capture_denominator(
    tmp_path: Path,
) -> None:
    provider = ProductionEngineLifecycleV1()
    request = ResidentBatchRequestV1(
        profile_id="acs-test",
        batch_id="batch-0000",
        cell_ids=("cell-0",),
        generations=2,
        ticks_per_generation=160,
        checkpoint_milestones=(2,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="6" * 64,
        acs_capture_instrumentation=True,
    )
    projection = _projection(generation=2, tick=0)
    expected_digest = _source_digest(projection)

    result = provider._persist_projection_checkpoint_set_v1(
        batch_root=tmp_path,
        request=request,
        source_projections=(projection,),
        generation=2,
        tick=0,
        absolute_next_tick=None,
        segment_ordinal=2,
        expected_source_projection_digest_sha256s={
            "cell-0": expected_digest
        },
    )

    stages = result["cell_stage_telemetry_seconds"]["cell-0"]
    assert set(stages) == {
        "checkpoint_clone_materialization",
        "checkpoint_canonical_pack",
        "checkpoint_sidecar_validation",
        "checkpoint_serialize",
        "checkpoint_hash",
        "checkpoint_write",
        "checkpoint_atomic_replace",
        "checkpoint_digest_coherence",
        "checkpoint_projection_gzip",
        "checkpoint_seal_write",
        "checkpoint_strict_readback",
        "checkpoint_staged_tree_fsync",
        "checkpoint_total",
        "source_digest_seconds",
        "checkpoint_identity_seconds",
        "writer_call_seconds",
        "outer_total_seconds",
        "provider_post_digest_seconds",
    }
    assert result["terminal_cell_digests"] == {"cell-0": expected_digest}
    assert result["checkpoint_bytes"] > 0
    assert result["cell_checkpoint_bytes"] == {
        "cell-0": result["checkpoint_bytes"]
    }
    assert result["checkpoint_bytes_measurement_status"] == "complete"
    checkpoint_bundle_v2_load(
        result["bundle_rows"][0]["path"],
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )


def test_projection_save_file_fsync_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"

    def _fail_file_fsync(_handle: Any, *, path: str) -> None:
        raise CheckpointStrictError(f"injected file fsync failure: {path}")

    monkeypatch.setattr(bundle_module, "_flush_and_fsync_file", _fail_file_fsync)
    with pytest.raises(CheckpointStrictError, match="file fsync failure"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(generation=2, source_digest=source_digest),
            observer_continuation=_observer_continuation(generation=2),
        )
    assert not bundle.exists()
    _assert_no_transaction_paths(bundle)


def test_projection_save_staged_directory_fsync_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    original = bundle_module._fsync_directory

    def _fail_staged_directory(path: str, *, context: str) -> None:
        if context == "staged directory" and Path(path).name.startswith(
            f".{bundle.name}.tmp-"
        ):
            raise CheckpointStrictError("injected staged directory fsync failure")
        original(path, context=context)

    monkeypatch.setattr(bundle_module, "_fsync_directory", _fail_staged_directory)
    with pytest.raises(CheckpointStrictError, match="staged directory fsync failure"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(generation=2, source_digest=source_digest),
            observer_continuation=_observer_continuation(generation=2),
        )
    assert not bundle.exists()
    _assert_no_transaction_paths(bundle)


def test_projection_save_parent_fsync_failure_invalidates_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    original = bundle_module._fsync_directory

    def _fail_parent(path: str, *, context: str) -> None:
        if context == "publication parent":
            raise CheckpointStrictError("injected publication parent fsync failure")
        original(path, context=context)

    monkeypatch.setattr(bundle_module, "_fsync_directory", _fail_parent)
    with pytest.raises(CheckpointStrictError, match="publication parent fsync failure"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=_identity(generation=2, source_digest=source_digest),
            observer_continuation=_observer_continuation(generation=2),
        )
    assert not (bundle / "bundle_complete.json").exists()
    assert not bundle.exists()
    _assert_no_transaction_paths(bundle)


@pytest.mark.parametrize(
    ("crash_stage", "exit_code", "staged_complete"),
    (
        ("before_seal", 81, False),
        ("after_seal_before_publication", 82, True),
    ),
)
def test_projection_save_process_crash_never_supersedes_prior_durable_bundle(
    tmp_path: Path,
    crash_stage: str,
    exit_code: int,
    staged_complete: bool,
) -> None:
    durable_projection = _projection(generation=2, tick=0)
    durable_digest = _source_digest(durable_projection)
    durable_bundle = tmp_path / "durable"
    checkpoint_bundle_v2_save_projection(
        durable_projection,
        str(durable_bundle),
        generation_index=2,
        expected_source_projection_sha256=durable_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=durable_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
    )

    candidate_bundle = tmp_path / "candidate"
    crash_script = r"""
import os
from pathlib import Path
import sys
from typing import Any

from simulation import checkpoint_bundle_v2 as bundle_module
from tests.perf.test_eiel_architecture_b_checkpoint_projection_save import (
    _identity,
    _observer_continuation,
    _projection,
    _source_digest,
    checkpoint_bundle_v2_save_projection,
)

destination = sys.argv[1]
stage = sys.argv[2]
exit_code = int(sys.argv[3])
if stage == "before_seal":
    original_write = bundle_module._write_canonical_json_lossless

    def crash_before_seal(path: str, value: Any, **kwargs: Any) -> None:
        if Path(path).name == "bundle_complete.json":
            os._exit(exit_code)
        original_write(path, value, **kwargs)

    bundle_module._write_canonical_json_lossless = crash_before_seal
else:
    bundle_module._publish_directory_absent = (
        lambda _source, _destination: os._exit(exit_code)
    )
projection = _projection(generation=2, tick=0)
digest = _source_digest(projection)
checkpoint_bundle_v2_save_projection(
    projection,
    destination,
    generation_index=2,
    expected_source_projection_sha256=digest,
    architecture_b_identity=_identity(generation=2, source_digest=digest),
    observer_continuation=_observer_continuation(generation=2),
)
os._exit(99)
"""
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_script,
            str(candidate_bundle),
            crash_stage,
            str(exit_code),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
    )
    assert crashed.returncode == exit_code
    assert not candidate_bundle.exists()
    staged = list(tmp_path.glob(".candidate.tmp-*"))
    assert len(staged) == 1
    assert (staged[0] / "bundle_complete.json").exists() is staged_complete
    if staged_complete:
        checkpoint_bundle_v2_load(
            str(staged[0]),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )
    durable = checkpoint_bundle_v2_load(
        str(durable_bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    assert durable["bundle_complete"]["status"] == "complete"


def test_multicell_process_crash_leaves_partial_set_unadvertised_and_retryable(
    tmp_path: Path,
) -> None:
    """One published cell is an orphan until the whole set is catalogued."""

    durable_projection = _projection(generation=2, tick=0)
    durable_digest = _source_digest(durable_projection)
    durable_bundle = tmp_path / "durable"
    checkpoint_bundle_v2_save_projection(
        durable_projection,
        str(durable_bundle),
        generation_index=2,
        expected_source_projection_sha256=durable_digest,
        architecture_b_identity=_identity(
            generation=2,
            source_digest=durable_digest,
        ),
        observer_continuation=_observer_continuation(generation=2),
    )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"generation": 2, "cell_bundles": [str(durable_bundle)]})
        + "\n",
        encoding="utf-8",
    )

    candidates = (tmp_path / "candidate-cell-0", tmp_path / "candidate-cell-1")
    crash_script = r"""
import os
import sys

from simulation import checkpoint_bundle_v2 as bundle_module
from tests.perf.test_eiel_architecture_b_checkpoint_projection_save import (
    _identity,
    _observer_continuation,
    _projection,
    _source_digest,
    checkpoint_bundle_v2_save_projection,
)

destinations = sys.argv[1:3]
exit_code = int(sys.argv[3])
original_publish = bundle_module._publish_directory_absent
published = 0

def crash_after_first_publication(source, destination):
    global published
    original_publish(source, destination)
    published += 1
    if published == 1:
        os._exit(exit_code)

bundle_module._publish_directory_absent = crash_after_first_publication
projection = _projection(generation=2, tick=0)
digest = _source_digest(projection)
for destination in destinations:
    checkpoint_bundle_v2_save_projection(
        projection,
        destination,
        generation_index=2,
        expected_source_projection_sha256=digest,
        architecture_b_identity=_identity(generation=2, source_digest=digest),
        observer_continuation=_observer_continuation(generation=2),
    )
os._exit(99)
"""
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_script,
            str(candidates[0]),
            str(candidates[1]),
            "93",
        ],
        cwd=Path(__file__).parents[2],
        check=False,
    )

    assert crashed.returncode == 93
    assert candidates[0].is_dir()
    assert not candidates[1].exists()
    assert json.loads(catalog_path.read_text(encoding="utf-8")) == {
        "generation": 2,
        "cell_bundles": [str(durable_bundle)],
    }
    checkpoint_bundle_v2_load(
        str(candidates[0]),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )

    retry_projection = _projection(generation=2, tick=0)
    retry_digest = _source_digest(retry_projection)
    for candidate in candidates:
        checkpoint_bundle_v2_save_projection(
            retry_projection,
            str(candidate),
            generation_index=2,
            expected_source_projection_sha256=retry_digest,
            architecture_b_identity=_identity(
                generation=2,
                source_digest=retry_digest,
            ),
            observer_continuation=_observer_continuation(generation=2),
        )
        checkpoint_bundle_v2_load(
            str(candidate),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )
    catalog_path.write_text(
        json.dumps(
            {
                "generation": 2,
                "publication": "successor-multicell-set",
                "cell_bundles": [str(path) for path in candidates],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert json.loads(catalog_path.read_text(encoding="utf-8"))["publication"] == (
        "successor-multicell-set"
    )


@pytest.mark.parametrize("error_number", [errno.ENOSYS, errno.EOPNOTSUPP])
def test_projection_save_uses_locked_publication_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / f"bundle-{error_number}"
    monkeypatch.setattr(
        bundle_module,
        "_renameat2_noreplace_errno",
        lambda _source, _destination: error_number,
    )

    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )
    checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    assert bundle.exists()
    _assert_no_transaction_paths(bundle)


def test_projection_save_identical_retry_reuses_final_and_conflict_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    continuation = _observer_continuation(generation=2)
    bundle = tmp_path / "bundle"

    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )
    before = _tree_bytes(bundle)
    before_inode = bundle.stat().st_ino

    original_replace = bundle_module.os.replace

    def _forbid_final_replace(source: Any, destination: Any) -> None:
        if Path(source) == bundle or Path(destination) == bundle:
            raise AssertionError("Architecture B save must not move or replace its final")
        original_replace(source, destination)

    monkeypatch.setattr(bundle_module.os, "replace", _forbid_final_replace)
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )
    assert bundle.stat().st_ino == before_inode
    assert _tree_bytes(bundle) == before

    conflicting_identity = copy.deepcopy(identity)
    conflicting_identity["snapshot_digest_sha256"] = "3" * 64
    with pytest.raises(CheckpointStrictError, match="publication conflict"):
        checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=conflicting_identity,
            observer_continuation=continuation,
        )

    assert bundle.stat().st_ino == before_inode
    assert _tree_bytes(bundle) == before
    _assert_no_transaction_paths(bundle)


def test_concurrent_identical_projection_writers_publish_once_and_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    identity = _identity(generation=2, source_digest=source_digest)
    continuation = _observer_continuation(generation=2)
    bundle = tmp_path / "bundle"
    publication_barrier = threading.Barrier(2)
    staged_paths: list[str] = []
    staged_paths_lock = threading.Lock()
    original_publish = bundle_module._publish_directory_absent

    def _synchronized_publish(source: str, destination: str) -> None:
        with staged_paths_lock:
            staged_paths.append(source)
        publication_barrier.wait(timeout=10)
        original_publish(source, destination)

    monkeypatch.setattr(
        bundle_module,
        "_publish_directory_absent",
        _synchronized_publish,
    )

    def _save() -> str:
        return checkpoint_bundle_v2_save_projection(
            projection,
            str(bundle),
            generation_index=2,
            expected_source_projection_sha256=source_digest,
            architecture_b_identity=identity,
            observer_continuation=continuation,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: _save(), range(2)))

    assert results == [str(bundle.resolve()), str(bundle.resolve())]
    assert len(staged_paths) == 2
    assert len(set(staged_paths)) == 2
    checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    _assert_no_transaction_paths(bundle)


def test_concurrent_conflicting_projection_writers_leave_one_immutable_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    continuation = _observer_continuation(generation=2)
    bundle = tmp_path / "bundle"
    publication_barrier = threading.Barrier(2)
    original_publish = bundle_module._publish_directory_absent

    def _synchronized_publish(source: str, destination: str) -> None:
        publication_barrier.wait(timeout=10)
        original_publish(source, destination)

    monkeypatch.setattr(
        bundle_module,
        "_publish_directory_absent",
        _synchronized_publish,
    )

    identities = [
        _identity(generation=2, source_digest=source_digest),
        _identity(generation=2, source_digest=source_digest),
    ]
    identities[1]["snapshot_digest_sha256"] = "3" * 64

    def _save(identity: dict[str, Any]) -> tuple[str, str]:
        try:
            result = checkpoint_bundle_v2_save_projection(
                projection,
                str(bundle),
                generation_index=2,
                expected_source_projection_sha256=source_digest,
                architecture_b_identity=identity,
                observer_continuation=continuation,
            )
            return "published", result
        except CheckpointStrictError as exc:
            return "conflict", str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(_save, identities))

    assert sorted(outcome for outcome, _detail in outcomes) == ["conflict", "published"]
    assert "publication conflict" in next(
        detail for outcome, detail in outcomes if outcome == "conflict"
    )
    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    winner_digest = payload["bundle_complete"]["architecture_b_identity"][
        "snapshot_digest_sha256"
    ]
    assert winner_digest in {"2" * 64, "3" * 64}
    _assert_no_transaction_paths(bundle)


def test_projection_bundle_strict_loads_in_fresh_process(tmp_path: Path) -> None:
    projection = _projection(generation=2, tick=0)
    source_digest = _source_digest(projection)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=_identity(generation=2, source_digest=source_digest),
        observer_continuation=_observer_continuation(generation=2),
    )

    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    python_path = str(project_root / "engine")
    if environment.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = python_path
    script = """
import json
import sys

from native_runtime import decode_bundled_native_checkpoint_projection
from simulation.checkpoint_bundle_v2 import (
    ARCHITECTURE_B_BUNDLE_KIND,
    checkpoint_bundle_v2_load,
    decode_architecture_b_resume_envelope,
)

payload = checkpoint_bundle_v2_load(
    sys.argv[1],
    strict=True,
    required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
)
projection = decode_bundled_native_checkpoint_projection(
    payload["native_runtime_checkpoint_projection"]
)
completion = payload["bundle_complete"]
resume_envelope = decode_architecture_b_resume_envelope(payload)
print(json.dumps({
    "generation": projection["metadata"]["generation"],
    "tick": projection["metadata"]["tick"],
    "energy": float(projection["tensors"]["energy"][0]),
    "source_projection_sha256": completion["source_projection_sha256"],
    "resume_envelope_type": type(resume_envelope).__name__,
    "resume_completion_sha256": resume_envelope[
        "bundle_completion_authority"
    ]["architecture_b_completion_sha256"],
    "typed_lifecycle_event_cursor": completion[
        "observer_continuation"
    ]["typed_lifecycle_event_cursor"],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(bundle)],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    restored = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_completion = json.loads(
        (bundle / "bundle_complete.json").read_text(encoding="utf-8")
    )
    assert restored == {
        "energy": 13.5,
        "generation": 2,
        "source_projection_sha256": expected_completion["source_projection_sha256"],
        "resume_envelope_type": "ValidatedArchitectureBResumeEnvelope",
        "resume_completion_sha256": expected_completion[
            "architecture_b_completion_sha256"
        ],
        "tick": 0,
        "typed_lifecycle_event_cursor": 0,
    }
