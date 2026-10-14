"""Evidence-bound ACS0/ACS4 checkpoint reporting and cadence modelling.

The report consumes completed batch manifests.  It does not infer missing
measurements and it never turns a qualitative master-program decision into an
automatic promotion.  Complete evidence is reported as ready for review;
incomplete evidence fails closed to ACS0-D or an incomplete ACS4 decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import math
from statistics import median
from typing import Any


ACS0_REPORT_SCHEMA_VERSION = "eiel_acs0_checkpoint_report_v1"
ACS4_REPORT_SCHEMA_VERSION = "eiel_acs4_checkpoint_report_v1"
ACS_CADENCE_MODEL_SCHEMA_VERSION = "eiel_acs_cadence_model_v1"
ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION = "eiel_acs_restore_measurement_v1"
ACS_GPU_TELEMETRY_SCHEMA_VERSION = "eiel_acs_gpu_telemetry_receipt_v1"
ACS_CADENCES_TICKS_V1 = (1_000, 2_500, 5_000, 10_000)

_CAPTURE_SCALARS = (
    "checkpoint_projection_time_ns",
    "snapshot_digest_time_ns",
    "host_capture_freeze_time_ns",
    "capture_cpu_time_ns",
    "native_final_stream_synchronization_time_ns",
    "last_useful_kernel_completion_observed_host_time_ns",
    "capture_barrier_time_ns",
    "post_sync_gpu_idle_proxy_time_ns",
    "captured_bytes",
)
_HOST_RESOURCE_SCALARS = ("rss_bytes", "pss_bytes", "pinned_bytes")
_GPU_RESOURCE_SCALARS = (
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "max_memory_allocated_bytes",
)
_TRANSFER_SCALARS = (
    "final_host_synchronizations",
    "final_device_to_host_operations",
    "compact_settlement_d2h_operations",
    "final_transfer_bytes",
)
_POST_CAPTURE_STAGE_SECONDS = (
    "source_digest_seconds",
    "checkpoint_identity_seconds",
    "checkpoint_clone_materialization",
    "checkpoint_canonical_pack",
    "checkpoint_sidecar_validation",
    "checkpoint_serialize",
    "checkpoint_hash",
    "checkpoint_digest_coherence",
    "checkpoint_projection_gzip",
    "checkpoint_write",
    "checkpoint_seal_write",
    "checkpoint_strict_readback",
    "checkpoint_staged_tree_fsync",
    "checkpoint_atomic_replace",
    "checkpoint_total",
    "writer_call_seconds",
    "outer_total_seconds",
    "provider_post_digest_seconds",
)
_SEMANTIC_IDENTITY_DIGEST_FIELDS = (
    "canonical_terminal_state_digest_sha256s",
    "canonical_checkpoint_storage_digest_sha256s",
)
_TERMINAL_PROJECTION_DIGEST_FIELD = "terminal_projection_digest_sha256s"
_MECHANICS_DIGEST_FIELD = "mechanics_checkpoint_digests"


class ACSCheckpointReportError(RuntimeError):
    """The supplied ACS evidence or modelling input is malformed."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ACSCheckpointReportError(f"{field} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < 0 or (positive and rendered <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ACSCheckpointReportError(f"{field} must be finite and {qualifier}")
    return rendered


def _integer(value: Any, *, field: str, positive: bool = False) -> int:
    if type(value) is not int or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ACSCheckpointReportError(f"{field} must be a {qualifier} integer")
    return value


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ACSCheckpointReportError("cannot summarize an empty measurement set")
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
        "total": sum(values),
    }


def _iso8601(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ACSCheckpointReportError(f"{field} must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ACSCheckpointReportError(f"{field} is not valid ISO-8601") from exc


def _wall_seconds(manifest: Mapping[str, Any]) -> float:
    started = _iso8601(manifest.get("created_at"), field="created_at")
    finished = _iso8601(manifest.get("updated_at"), field="updated_at")
    wall = (finished - started).total_seconds()
    if wall <= 0:
        raise ACSCheckpointReportError("batch manifest wall interval must be positive")
    return wall


def _ordered_catalog(result: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    raw = result.get("checkpoint_catalog")
    if not isinstance(raw, Mapping) or not raw:
        return []
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for coordinate, payload in raw.items():
        try:
            rendered_coordinate = int(coordinate)
        except (TypeError, ValueError) as exc:
            raise ACSCheckpointReportError(
                "checkpoint catalog coordinate must be integer-like"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ACSCheckpointReportError(
                "checkpoint catalog payload must be an object"
            )
        rows.append((rendered_coordinate, payload))
    return sorted(rows)


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cell_ids = manifest.get("cell_ids")
    if (
        not isinstance(cell_ids, list)
        or not cell_ids
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids)
    ):
        raise ACSCheckpointReportError(
            "batch manifest cell_ids must be non-empty strings"
        )
    plan_sha256 = manifest.get("plan_sha256")
    route_sha256 = manifest.get("execution_route_sha256")
    for field, value in (
        ("plan_sha256", plan_sha256),
        ("execution_route_sha256", route_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ACSCheckpointReportError(f"batch manifest {field} is not SHA-256")
    result = manifest.get("result")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) and isinstance(result, Mapping):
        catalog = _ordered_catalog(result)
        checkpoint = catalog[-1][1] if catalog else None
    binding = (
        checkpoint.get("checkpoint_binding")
        if isinstance(checkpoint, Mapping)
        else None
    )
    if not isinstance(binding, Mapping):
        raise ACSCheckpointReportError(
            "batch manifest lacks final checkpoint artifact binding"
        )
    artifact_sha256s = binding.get("artifact_sha256s")
    if not isinstance(artifact_sha256s, Mapping) or not artifact_sha256s:
        raise ACSCheckpointReportError(
            "final checkpoint binding lacks artifact SHA-256 identities"
        )
    for name, digest in artifact_sha256s.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ACSCheckpointReportError(
                "final checkpoint artifact binding is malformed"
            )
    return {
        "schema_version": manifest.get("schema_version"),
        "campaign_id": manifest.get("campaign_id"),
        "batch_id": manifest.get("batch_id"),
        "plan_sha256": plan_sha256,
        "execution_route_sha256": route_sha256,
        "provider_id": manifest.get("provider_id"),
        "profile_id": manifest.get("profile_id"),
        "cell_ids": list(cell_ids),
        "artifact_sha256s": dict(sorted(artifact_sha256s.items())),
        "qualification_key_sha256": binding.get("qualification_key_sha256"),
        "manifest_sha256": _canonical_sha256(dict(manifest)),
    }


def _missing_numeric(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    prefix: str,
) -> tuple[list[str], dict[str, float]]:
    missing: list[str] = []
    observed: dict[str, float] = {}
    for field in fields:
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            missing.append(f"{prefix}.{field}")
            continue
        rendered = float(raw)
        if not math.isfinite(rendered) or rendered < 0:
            missing.append(f"{prefix}.{field}")
            continue
        observed[field] = rendered
    return missing, observed


def _extract_measurements_v1(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("status") != "completed":
        raise ACSCheckpointReportError("ACS report requires a completed batch manifest")
    result = manifest.get("result")
    if not isinstance(result, Mapping):
        raise ACSCheckpointReportError("batch manifest result must be an object")
    raw_capture = result.get("acs_capture_measurements")
    captures = raw_capture if isinstance(raw_capture, list) else []
    missing: list[str] = []
    capture_values: dict[str, list[float]] = {field: [] for field in _CAPTURE_SCALARS}
    resource_values: dict[str, list[float]] = {}
    transfer_values: dict[str, list[float]] = {field: [] for field in _TRANSFER_SCALARS}
    limitations: set[str] = set()
    if not captures:
        missing.append("result.acs_capture_measurements")
    for index, raw_capture_row in enumerate(captures):
        prefix = f"result.acs_capture_measurements[{index}]"
        if not isinstance(raw_capture_row, Mapping):
            missing.append(prefix)
            continue
        absent, observed = _missing_numeric(
            raw_capture_row, _CAPTURE_SCALARS, prefix=prefix
        )
        missing.extend(absent)
        for field, value in observed.items():
            capture_values[field].append(value)
        for resource_name, resource_fields in (
            ("host_resources_before", _HOST_RESOURCE_SCALARS),
            ("host_resources_after", _HOST_RESOURCE_SCALARS),
            ("gpu_resources_before", _GPU_RESOURCE_SCALARS),
            ("gpu_resources_after", _GPU_RESOURCE_SCALARS),
        ):
            resource = raw_capture_row.get(resource_name)
            if not isinstance(resource, Mapping):
                missing.append(f"{prefix}.{resource_name}")
                continue
            absent, observed = _missing_numeric(
                resource,
                resource_fields,
                prefix=f"{prefix}.{resource_name}",
            )
            missing.extend(absent)
            for field, value in observed.items():
                resource_values.setdefault(f"{resource_name}.{field}", []).append(value)
        transfer = raw_capture_row.get("native_transfer_receipt")
        if not isinstance(transfer, Mapping):
            missing.append(f"{prefix}.native_transfer_receipt")
        else:
            absent, observed = _missing_numeric(
                transfer,
                _TRANSFER_SCALARS,
                prefix=f"{prefix}.native_transfer_receipt",
            )
            missing.extend(absent)
            for field, value in observed.items():
                transfer_values[field].append(value)
        raw_limitations = raw_capture_row.get("limitations")
        if isinstance(raw_limitations, list):
            limitations.update(
                str(value) for value in raw_limitations if isinstance(value, str)
            )

    stage_values: dict[str, list[float]] = {
        field: [] for field in _POST_CAPTURE_STAGE_SECONDS
    }
    writer_service_seconds: list[float] = []
    checkpoint_bytes: list[float] = []
    persistence_throughput_bytes_per_second: list[float] = []
    catalog = _ordered_catalog(result)
    if not catalog:
        missing.append("result.checkpoint_catalog")
    for coordinate, checkpoint in catalog:
        instrumentation = checkpoint.get("acs_checkpoint_instrumentation")
        if not isinstance(instrumentation, Mapping):
            instrumentation = checkpoint.get("acs_checkpoint_staging")
        prefix = f"result.checkpoint_catalog[{coordinate}].acs_instrumentation"
        if not isinstance(instrumentation, Mapping):
            missing.append(prefix)
            continue
        writer_ns = instrumentation.get("writer_service_time_ns")
        if isinstance(writer_ns, bool) or not isinstance(writer_ns, (int, float)):
            missing.append(f"{prefix}.writer_service_time_ns")
        elif float(writer_ns) < 0:
            missing.append(f"{prefix}.writer_service_time_ns")
        else:
            writer_seconds = float(writer_ns) / 1_000_000_000.0
            writer_service_seconds.append(writer_seconds)
            raw_checkpoint_bytes = instrumentation.get("checkpoint_bytes")
            if (
                isinstance(raw_checkpoint_bytes, bool)
                or not isinstance(raw_checkpoint_bytes, (int, float))
                or float(raw_checkpoint_bytes) <= 0
                or instrumentation.get("checkpoint_bytes_measurement_status")
                != "complete"
            ):
                missing.append(f"{prefix}.checkpoint_bytes")
            else:
                rendered_bytes = float(raw_checkpoint_bytes)
                checkpoint_bytes.append(rendered_bytes)
                if writer_seconds > 0:
                    persistence_throughput_bytes_per_second.append(
                        rendered_bytes / writer_seconds
                    )
        cell_stages = instrumentation.get("cell_stage_telemetry_seconds")
        if not isinstance(cell_stages, Mapping) or not cell_stages:
            missing.append(f"{prefix}.cell_stage_telemetry_seconds")
            continue
        for cell_id, raw_stages in cell_stages.items():
            stage_prefix = f"{prefix}.cell_stage_telemetry_seconds[{cell_id}]"
            if not isinstance(raw_stages, Mapping):
                missing.append(stage_prefix)
                continue
            absent, observed = _missing_numeric(
                raw_stages,
                _POST_CAPTURE_STAGE_SECONDS,
                prefix=stage_prefix,
            )
            missing.extend(absent)
            for field, value in observed.items():
                stage_values[field].append(value)

    publications = result.get("acs_catalog_publication_measurements")
    publication_seconds: list[float] = []
    if not isinstance(publications, list) or not publications:
        missing.append("result.acs_catalog_publication_measurements")
    else:
        for index, publication in enumerate(publications):
            if not isinstance(publication, Mapping):
                missing.append(f"result.acs_catalog_publication_measurements[{index}]")
                continue
            raw_total = publication.get("total_finalization_time_ns")
            if isinstance(raw_total, bool) or not isinstance(raw_total, (int, float)):
                missing.append(
                    "result.acs_catalog_publication_measurements"
                    f"[{index}].total_finalization_time_ns"
                )
            elif float(raw_total) < 0:
                missing.append(
                    "result.acs_catalog_publication_measurements"
                    f"[{index}].total_finalization_time_ns"
                )
            else:
                publication_seconds.append(float(raw_total) / 1_000_000_000.0)

    return {
        "complete": not missing,
        "missing_or_invalid_fields": sorted(set(missing)),
        "capture": {
            field: _summary(values)
            for field, values in capture_values.items()
            if values
        },
        "resources": {
            field: _summary(values)
            for field, values in resource_values.items()
            if values
        },
        "traffic": {
            field: _summary(values)
            for field, values in transfer_values.items()
            if values
        },
        "post_capture_stages_seconds": {
            field: _summary(values) for field, values in stage_values.items() if values
        },
        "writer_service_seconds": (
            _summary(writer_service_seconds) if writer_service_seconds else None
        ),
        "checkpoint_bytes": (_summary(checkpoint_bytes) if checkpoint_bytes else None),
        "effective_persistence_bytes_per_second": (
            _summary(persistence_throughput_bytes_per_second)
            if persistence_throughput_bytes_per_second
            else None
        ),
        "catalog_publication_seconds": (
            _summary(publication_seconds) if publication_seconds else None
        ),
        "limitations": sorted(limitations),
        "checkpoint_count": len(catalog),
    }


def _gpu_idle_boundary_ns(manifest: Mapping[str, Any]) -> int | None:
    result = manifest.get("result")
    captures = (
        result.get("acs_capture_measurements") if isinstance(result, Mapping) else None
    )
    if not isinstance(captures, list) or not captures:
        return None
    boundaries = [
        capture.get("last_useful_kernel_completion_observed_host_time_ns")
        for capture in captures
        if isinstance(capture, Mapping)
    ]
    if not boundaries or any(
        type(value) is not int or value < 0 for value in boundaries
    ):
        return None
    return max(boundaries)


def _gpu_telemetry_status(
    receipt: Mapping[str, Any] | None,
    *,
    manifest_sha256: str,
    post_sync_boundary_ns: int | None,
) -> dict[str, Any]:
    if receipt is None:
        return {"status": "MISSING", "complete": False}
    idle_fraction = receipt.get("gpu_idle_fraction")
    samples = receipt.get("runtime_samples")
    idle_threshold = receipt.get("idle_sm_threshold_percent")
    valid_samples = isinstance(samples, list) and all(
        isinstance(sample, Mapping)
        and type(sample.get("monotonic_time_ns")) is int
        and sample["monotonic_time_ns"] >= 0
        and not isinstance(sample.get("sm_percent"), bool)
        and isinstance(sample.get("sm_percent"), (int, float))
        and math.isfinite(float(sample["sm_percent"]))
        and 0.0 <= float(sample["sm_percent"]) <= 100.0
        for sample in (samples if isinstance(samples, list) else [])
    )
    valid_idle_threshold = (
        not isinstance(idle_threshold, bool)
        and isinstance(idle_threshold, (int, float))
        and math.isfinite(float(idle_threshold))
        and 0.0 <= float(idle_threshold) <= 100.0
    )
    post_sync_samples = (
        [
            sample
            for sample in samples
            if sample["monotonic_time_ns"] >= post_sync_boundary_ns
        ]
        if valid_samples and post_sync_boundary_ns is not None
        else []
    )
    derived_idle_fraction = (
        sum(
            float(sample["sm_percent"]) <= float(idle_threshold)
            for sample in post_sync_samples
        )
        / len(post_sync_samples)
        if post_sync_samples and valid_idle_threshold
        else None
    )
    complete = (
        receipt.get("schema_version") == ACS_GPU_TELEMETRY_SCHEMA_VERSION
        and receipt.get("status") == "passed"
        and receipt.get("baseline_admitted") is True
        and receipt.get("launched") is True
        and receipt.get("command_exit_code") == 0
        and receipt.get("no_competing_gpu_workloads") is True
        and (
            receipt.get("process_attributed") is True
            or receipt.get("exclusive_gpu_window") is True
        )
        and receipt.get("manifest_sha256") == manifest_sha256
        and receipt.get("post_sync_boundary_monotonic_ns") == post_sync_boundary_ns
        and valid_samples
        and type(receipt.get("sample_count")) is int
        and receipt["sample_count"] == len(samples)
        and receipt["sample_count"] > 0
        and type(receipt.get("post_sync_sample_count")) is int
        and receipt["post_sync_sample_count"] == len(post_sync_samples)
        and receipt["post_sync_sample_count"] > 0
        and not isinstance(idle_fraction, bool)
        and isinstance(idle_fraction, (int, float))
        and math.isfinite(float(idle_fraction))
        and 0.0 <= float(idle_fraction) <= 1.0
        and derived_idle_fraction is not None
        and math.isclose(float(idle_fraction), derived_idle_fraction)
    )
    return {
        "status": "PASSED" if complete else "INVALID",
        "complete": complete,
        "receipt": dict(receipt),
    }


def build_acs0_report_v1(
    manifest: Mapping[str, Any],
    *,
    gpu_telemetry_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and summarize one synchronous ACS0 instrumentation manifest."""

    identity = _manifest_identity(manifest)
    measurements = _extract_measurements_v1(manifest)
    gpu_telemetry = _gpu_telemetry_status(
        gpu_telemetry_receipt,
        manifest_sha256=identity["manifest_sha256"],
        post_sync_boundary_ns=_gpu_idle_boundary_ns(manifest),
    )
    complete = measurements["complete"] and gpu_telemetry["complete"]
    core = {
        "schema_version": ACS0_REPORT_SCHEMA_VERSION,
        "source_identity": identity,
        "wall_seconds": _wall_seconds(manifest),
        "measurement_status": "COMPLETE" if complete else "INCOMPLETE",
        "decision": "UNSET_REQUIRES_EVIDENCE_REVIEW" if complete else "ACS0-D",
        "admissible_decisions": (
            ["ACS0-A", "ACS0-B", "ACS0-C"] if complete else ["ACS0-D"]
        ),
        "measurements": measurements,
        "external_gpu_telemetry": gpu_telemetry,
        "claim_boundary": (
            "instrumentation completeness and measured values only; qualitative "
            "ACS0 A/B/C selection is not automated"
        ),
    }
    return {**core, "report_sha256": _canonical_sha256(core)}


def model_acs_cadence_v1(
    *,
    cadence_ticks: int,
    horizon_ticks: int,
    compute_wall_seconds_per_tick: float,
    capture_barrier_seconds: float,
    persistence_service_seconds: float,
    catalog_publication_seconds: float,
    slots: int,
) -> dict[str, Any]:
    """Model the provider's reserve-before-compute bounded FIFO schedule."""

    cadence = _integer(cadence_ticks, field="cadence_ticks", positive=True)
    horizon = _integer(horizon_ticks, field="horizon_ticks", positive=True)
    if type(slots) is not int or slots not in {1, 2, 3}:
        raise ACSCheckpointReportError("slots must be exactly one, two, or three")
    compute_per_tick = _number(
        compute_wall_seconds_per_tick,
        field="compute_wall_seconds_per_tick",
        positive=True,
    )
    capture = _number(capture_barrier_seconds, field="capture_barrier_seconds")
    service = _number(
        persistence_service_seconds,
        field="persistence_service_seconds",
        positive=True,
    )
    publication = _number(
        catalog_publication_seconds,
        field="catalog_publication_seconds",
    )
    checkpoint_count = math.ceil(horizon / cadence)
    producer_time = 0.0
    writer_available = 0.0
    departures: list[float] = []
    blocked_seconds = 0.0
    first_backpressure_checkpoint: int | None = None
    first_backpressure_seconds: float | None = None
    first_backpressure_ticks: int | None = None
    high_water = 0
    completed_ticks = 0

    def finalize_ready() -> None:
        nonlocal producer_time
        while departures and departures[0] <= producer_time:
            departures.pop(0)
            producer_time += publication

    for checkpoint_index in range(1, checkpoint_count + 1):
        finalize_ready()
        if len(departures) >= slots:
            if first_backpressure_checkpoint is None:
                first_backpressure_checkpoint = checkpoint_index
                first_backpressure_seconds = producer_time
                first_backpressure_ticks = completed_ticks
            wait = max(0.0, departures[0] - producer_time)
            blocked_seconds += wait
            producer_time += wait
            finalize_ready()
        high_water = max(high_water, len(departures) + 1)
        span = min(cadence, horizon - completed_ticks)
        completed_ticks += span
        producer_time += span * compute_per_tick + capture
        writer_available = max(writer_available, producer_time) + service
        departures.append(writer_available)

    while departures:
        producer_time = max(producer_time, departures[0])
        finalize_ready()

    interval_seconds = cadence * compute_per_tick + capture
    utilization = service / interval_seconds
    sustainable_cadence = max(
        1,
        math.ceil(max(0.0, service - capture - publication) / compute_per_tick),
    )
    synchronous_wall = horizon * compute_per_tick + checkpoint_count * (
        capture + service + publication
    )
    core = {
        "schema_version": ACS_CADENCE_MODEL_SCHEMA_VERSION,
        "cadence_ticks": cadence,
        "horizon_ticks": horizon,
        "checkpoint_count": checkpoint_count,
        "slots": slots,
        "synchronous_tax_seconds": checkpoint_count * (capture + service + publication),
        "synchronous_expected_total_wall_seconds": synchronous_wall,
        "async_writer_utilization": utilization,
        "time_to_backpressure_seconds": first_backpressure_seconds,
        "time_to_backpressure_ticks": first_backpressure_ticks,
        "first_backpressure_checkpoint": first_backpressure_checkpoint,
        "safe_sustainable_minimum_cadence_ticks": sustainable_cadence,
        "safe_sustainable_checkpoint_rate_per_second": 1.0 / service,
        "expected_total_wall_seconds": producer_time,
        "producer_blocked_seconds": blocked_seconds,
        "buffer_high_water": high_water,
        "service_rate_sustainable": utilization <= 1.0,
        "model_scope": (
            "deterministic one-writer reserve-before-compute FIFO using measured "
            "central capture, persistence and publication service times"
        ),
    }
    return core


def _exact_cell_digest_map(
    checkpoint: Mapping[str, Any],
    *,
    field: str,
    expected_cell_ids: Sequence[str],
) -> dict[str, str]:
    value = checkpoint.get(field)
    if not isinstance(value, Mapping) or set(value) != set(expected_cell_ids):
        raise ACSCheckpointReportError(
            f"final checkpoint {field} must cover the exact manifest cell set"
        )
    for cell_id, digest in value.items():
        if (
            not isinstance(cell_id, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ACSCheckpointReportError(
                f"final checkpoint {field} contains a malformed digest"
            )
    return dict(sorted(value.items()))


def _final_identity(
    result: Mapping[str, Any],
    *,
    expected_cell_ids: Sequence[str],
) -> dict[str, Any] | None:
    catalog = _ordered_catalog(result)
    if not catalog:
        return None
    coordinate, checkpoint = catalog[-1]
    digests: dict[str, Any] = {"coordinate": coordinate}
    for field in (
        *_SEMANTIC_IDENTITY_DIGEST_FIELDS,
        _TERMINAL_PROJECTION_DIGEST_FIELD,
        _MECHANICS_DIGEST_FIELD,
    ):
        digests[field] = _exact_cell_digest_map(
            checkpoint,
            field=field,
            expected_cell_ids=expected_cell_ids,
        )
    if (
        digests[_MECHANICS_DIGEST_FIELD]
        != digests["canonical_terminal_state_digest_sha256s"]
    ):
        raise ACSCheckpointReportError(
            "final checkpoint mechanics digests disagree with canonical terminal state"
        )
    identity_contract = checkpoint.get("checkpoint_identity_contract")
    if (
        not isinstance(identity_contract, Mapping)
        or identity_contract.get("q5_terminal_equality_authority")
        != "canonical_terminal_state_digest_sha256s"
        or identity_contract.get("route_provenance_may_differ_from_terminal_state")
        is not True
    ):
        raise ACSCheckpointReportError(
            "final checkpoint identity contract lacks canonical terminal authority"
        )
    digests["identity_contract"] = {
        "q5_terminal_equality_authority": (
            identity_contract["q5_terminal_equality_authority"]
        ),
        "route_provenance_may_differ_from_terminal_state": True,
    }
    return digests


def _restore_status(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    if receipt is None:
        return {"status": "MISSING", "complete": False}
    complete = (
        receipt.get("schema_version") == ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION
        and receipt.get("status") == "passed"
        and receipt.get("semantic_digest_identity") is True
        and not isinstance(receipt.get("restore_wall_seconds"), bool)
        and isinstance(receipt.get("restore_wall_seconds"), (int, float))
        and float(receipt["restore_wall_seconds"]) >= 0
    )
    return {
        "status": "PASSED" if complete else "INVALID",
        "complete": complete,
        "receipt": dict(receipt),
    }


def _terminal_tick_horizon(
    result: Mapping[str, Any],
    *,
    arm: str,
    expected_cell_ids: Sequence[str],
) -> dict[str, Any]:
    if "terminal_next_tick" in result:
        terminal_ticks = _integer(
            result.get("terminal_next_tick"),
            field=f"{arm} result.terminal_next_tick",
            positive=True,
        )
        return {
            "authority": "terminal_next_tick",
            "terminal_ticks": terminal_ticks,
        }

    terminal_generation = _integer(
        result.get("terminal_generation"),
        field=f"{arm} result.terminal_generation",
        positive=True,
    )
    runner_receipts = result.get("runner_receipts")
    if not isinstance(runner_receipts, list) or not runner_receipts:
        raise ACSCheckpointReportError(
            f"{arm} result.runner_receipts must be a non-empty list"
        )
    ticks_per_generation: list[int] = []
    receipt_cell_ids: set[str] = set()
    for index, receipt in enumerate(runner_receipts):
        if not isinstance(receipt, Mapping):
            raise ACSCheckpointReportError(
                f"{arm} result.runner_receipts[{index}] must be an object"
            )
        dimensions = receipt.get("dimensions")
        cell_id = receipt.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise ACSCheckpointReportError(
                f"{arm} result.runner_receipts[{index}].cell_id must be non-empty"
            )
        receipt_cell_ids.add(cell_id)
        if not isinstance(dimensions, Mapping):
            raise ACSCheckpointReportError(
                f"{arm} result.runner_receipts[{index}].dimensions must be an object"
            )
        ticks_per_generation.append(
            _integer(
                dimensions.get("ticks_per_generation"),
                field=(
                    f"{arm} result.runner_receipts[{index}]"
                    ".dimensions.ticks_per_generation"
                ),
                positive=True,
            )
        )
    if receipt_cell_ids != set(expected_cell_ids):
        raise ACSCheckpointReportError(
            f"{arm} runner receipts do not cover the exact manifest cell set"
        )
    if len(set(ticks_per_generation)) != 1:
        raise ACSCheckpointReportError(
            f"{arm} runner receipts disagree on ticks_per_generation"
        )
    exact_ticks_per_generation = ticks_per_generation[0]
    return {
        "authority": "terminal_generation",
        "terminal_generation": terminal_generation,
        "ticks_per_generation": exact_ticks_per_generation,
        "terminal_ticks": terminal_generation * exact_ticks_per_generation,
    }


def build_acs4_report_v1(
    synchronous_manifest: Mapping[str, Any],
    acs_manifest: Mapping[str, Any],
    *,
    restore_receipt: Mapping[str, Any] | None = None,
    synchronous_gpu_telemetry_receipt: Mapping[str, Any] | None = None,
    acs_gpu_telemetry_receipt: Mapping[str, Any] | None = None,
    horizon_ticks: int = 50_000,
    cadences: Sequence[int] = ACS_CADENCES_TICKS_V1,
) -> dict[str, Any]:
    """Compare synchronous/ACS manifests and model required checkpoint cadences."""

    sync_identity = _manifest_identity(synchronous_manifest)
    acs_identity = _manifest_identity(acs_manifest)
    sync_metrics = _extract_measurements_v1(synchronous_manifest)
    acs_metrics = _extract_measurements_v1(acs_manifest)
    sync_result = synchronous_manifest.get("result")
    acs_result = acs_manifest.get("result")
    assert isinstance(sync_result, Mapping) and isinstance(acs_result, Mapping)
    sync_wall = _wall_seconds(synchronous_manifest)
    acs_wall = _wall_seconds(acs_manifest)
    cell_count = len(sync_identity["cell_ids"])
    same_cells = sync_identity["cell_ids"] == acs_identity["cell_ids"]
    if not same_cells:
        raise ACSCheckpointReportError(
            "synchronous and ACS manifests must have identical ordered cell arms"
        )
    sync_horizon = _terminal_tick_horizon(
        sync_result,
        arm="synchronous",
        expected_cell_ids=sync_identity["cell_ids"],
    )
    acs_horizon = _terminal_tick_horizon(
        acs_result,
        arm="ACS",
        expected_cell_ids=acs_identity["cell_ids"],
    )
    if sync_horizon["authority"] != acs_horizon["authority"]:
        raise ACSCheckpointReportError(
            "synchronous and ACS manifests use mixed terminal time authorities"
        )
    if sync_horizon["authority"] == "terminal_generation" and (
        sync_horizon["terminal_generation"] != acs_horizon["terminal_generation"]
        or sync_horizon["ticks_per_generation"] != acs_horizon["ticks_per_generation"]
    ):
        raise ACSCheckpointReportError(
            "synchronous and ACS generation-route terminal horizons disagree"
        )
    sync_ticks = sync_horizon["terminal_ticks"]
    acs_ticks = acs_horizon["terminal_ticks"]
    sync_final = _final_identity(
        sync_result,
        expected_cell_ids=sync_identity["cell_ids"],
    )
    acs_final = _final_identity(
        acs_result,
        expected_cell_ids=acs_identity["cell_ids"],
    )
    semantic_identity = (
        sync_final is not None
        and acs_final is not None
        and all(
            sync_final[field] == acs_final[field]
            for field in ("coordinate", *_SEMANTIC_IDENTITY_DIGEST_FIELDS)
        )
    )
    terminal_projection_identity = (
        sync_final is not None
        and acs_final is not None
        and sync_final["coordinate"] == acs_final["coordinate"]
        and sync_final[_TERMINAL_PROJECTION_DIGEST_FIELD]
        == acs_final[_TERMINAL_PROJECTION_DIGEST_FIELD]
    )
    staging = acs_result.get("acs_checkpoint_staging")
    slots = staging.get("slots") if isinstance(staging, Mapping) else None
    if type(slots) is not int or slots not in {1, 2, 3}:
        raise ACSCheckpointReportError(
            "ACS4 candidate manifest lacks a bounded one/two/three-slot staging result"
        )
    required_summaries = (
        sync_metrics.get("writer_service_seconds"),
        sync_metrics.get("catalog_publication_seconds"),
        sync_metrics.get("capture", {}).get("capture_barrier_time_ns"),
    )
    if not all(isinstance(value, Mapping) for value in required_summaries):
        raise ACSCheckpointReportError(
            "synchronous manifest lacks central checkpoint timing summaries"
        )
    sync_service = float(required_summaries[0]["median"])
    sync_publication = float(required_summaries[1]["median"])
    sync_capture = float(required_summaries[2]["median"]) / 1_000_000_000.0
    checkpoint_count = _integer(
        sync_metrics["checkpoint_count"],
        field="synchronous checkpoint count",
        positive=True,
    )
    if sync_ticks != acs_ticks:
        raise ACSCheckpointReportError(
            "synchronous and ACS manifests must cover the same positive tick horizon"
        )
    synchronous_checkpoint_seconds = checkpoint_count * (
        sync_capture + sync_service + sync_publication
    )
    compute_wall = sync_wall - synchronous_checkpoint_seconds
    if compute_wall <= 0:
        raise ACSCheckpointReportError(
            "measured synchronous checkpoint denominator consumes the full batch wall"
        )
    compute_per_tick = compute_wall / int(sync_ticks)
    acs_capture_summary = acs_metrics.get("capture", {}).get("capture_barrier_time_ns")
    acs_service_summary = acs_metrics.get("writer_service_seconds")
    acs_publication_summary = acs_metrics.get("catalog_publication_seconds")
    if not all(
        isinstance(value, Mapping)
        for value in (
            acs_capture_summary,
            acs_service_summary,
            acs_publication_summary,
        )
    ):
        raise ACSCheckpointReportError("ACS manifest lacks central timing summaries")
    models = [
        model_acs_cadence_v1(
            cadence_ticks=_integer(cadence, field="cadence", positive=True),
            horizon_ticks=horizon_ticks,
            compute_wall_seconds_per_tick=compute_per_tick,
            capture_barrier_seconds=(
                float(acs_capture_summary["median"]) / 1_000_000_000.0
            ),
            persistence_service_seconds=float(acs_service_summary["median"]),
            catalog_publication_seconds=float(acs_publication_summary["median"]),
            slots=slots,
        )
        for cadence in cadences
    ]
    restore = _restore_status(restore_receipt)
    synchronous_gpu_telemetry = _gpu_telemetry_status(
        synchronous_gpu_telemetry_receipt,
        manifest_sha256=sync_identity["manifest_sha256"],
        post_sync_boundary_ns=_gpu_idle_boundary_ns(synchronous_manifest),
    )
    acs_gpu_telemetry = _gpu_telemetry_status(
        acs_gpu_telemetry_receipt,
        manifest_sha256=acs_identity["manifest_sha256"],
        post_sync_boundary_ns=_gpu_idle_boundary_ns(acs_manifest),
    )
    complete = (
        sync_metrics["complete"]
        and acs_metrics["complete"]
        and same_cells
        and semantic_identity
        and restore["complete"]
        and synchronous_gpu_telemetry["complete"]
        and acs_gpu_telemetry["complete"]
    )
    useful_world_ticks = cell_count * int(sync_ticks)
    core = {
        "schema_version": ACS4_REPORT_SCHEMA_VERSION,
        "synchronous_source_identity": sync_identity,
        "acs_source_identity": acs_identity,
        "measurement_status": "COMPLETE" if complete else "INCOMPLETE",
        "decision": "UNSET_REQUIRES_EVIDENCE_REVIEW" if complete else "ACS-D",
        "admissible_decisions": (
            ["ACS-A", "ACS-B", "ACS-C", "ACS-D", "ACS-E"]
            if complete
            else ["ACS-D", "ACS-E"]
        ),
        "measured_comparison": {
            "cell_count": cell_count,
            "terminal_ticks": sync_ticks,
            "terminal_time_authority": sync_horizon["authority"],
            "terminal_generation": sync_horizon.get("terminal_generation"),
            "ticks_per_generation": sync_horizon.get("ticks_per_generation"),
            "useful_world_ticks": useful_world_ticks,
            "synchronous_wall_seconds": sync_wall,
            "acs_wall_seconds": acs_wall,
            "synchronous_useful_world_ticks_per_wall_second": (
                useful_world_ticks / sync_wall
            ),
            "acs_useful_world_ticks_per_wall_second": useful_world_ticks / acs_wall,
            "acs_to_synchronous_rate_ratio": sync_wall / acs_wall,
            "same_cells": same_cells,
            "semantic_digest_identity": semantic_identity,
            "terminal_projection_digest_identity": terminal_projection_identity,
            "terminal_projection_digest_scope": (
                "complete checkpoint projection including route provenance; "
                "not the cross-route semantic equality authority"
            ),
            "synchronous_final_identity": sync_final,
            "acs_final_identity": acs_final,
            "acs_slots": slots,
        },
        "synchronous_measurements": sync_metrics,
        "acs_measurements": acs_metrics,
        "restore_measurement": restore,
        "synchronous_external_gpu_telemetry": synchronous_gpu_telemetry,
        "acs_external_gpu_telemetry": acs_gpu_telemetry,
        "cadence_models": models,
        "model_inputs": {
            "horizon_ticks": horizon_ticks,
            "compute_wall_seconds_per_tick": compute_per_tick,
            "method": (
                "synchronous batch wall minus measured synchronous capture, "
                "persistence and publication denominator"
            ),
        },
        "claim_boundary": (
            "measured manifest comparison and deterministic cadence projection; "
            "ACS A/B/C/D/E selection remains an explicit evidence review"
        ),
    }
    return {**core, "report_sha256": _canonical_sha256(core)}


__all__ = [
    "ACS0_REPORT_SCHEMA_VERSION",
    "ACS4_REPORT_SCHEMA_VERSION",
    "ACS_CADENCES_TICKS_V1",
    "ACS_CADENCE_MODEL_SCHEMA_VERSION",
    "ACS_GPU_TELEMETRY_SCHEMA_VERSION",
    "ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION",
    "ACSCheckpointReportError",
    "build_acs0_report_v1",
    "build_acs4_report_v1",
    "model_acs_cadence_v1",
]
