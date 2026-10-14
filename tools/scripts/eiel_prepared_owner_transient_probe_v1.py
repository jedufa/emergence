#!/usr/bin/env python3
"""Trace transient CUDA allocation during one prepared-owner measurement."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import pretty_json  # noqa: E402
from engine.campaigns.compiler import (  # noqa: E402
    CampaignError,
    compile_campaign,
    load_campaign_spec,
    resolve_campaign_profile,
)
from engine.campaigns.preflight import preflight_campaign  # noqa: E402
from engine.campaigns.production_engine_provider import (  # noqa: E402
    EielProductionExecutionProviderV1,
)
from engine.campaigns.providers import (  # noqa: E402
    ExecutionProviderRegistryV1,
    default_execution_provider_registry,
)
from engine.campaigns.resident_prepared_owner_resource_audit_v1 import (  # noqa: E402
    PREPARE_ONLY_STAGES,
    cuda_memory_stage_sampler_v1,
)
from tools.scripts.eiel_campaign import (  # noqa: E402
    _runtime_root,
    _selected_batch_measurement_plan,
    _write_plan,
)


TRACE_SCHEMA_VERSION = "eiel_prepared_owner_transient_probe_v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(payload), encoding="utf-8")
    temporary.replace(path)


def _as_bool(value: Any) -> bool:
    return bool(isinstance(value, Mapping) and value.get("ok") is True)


def summarize_trace_v1(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize probe samples without changing canonical audit semantics."""

    rows = [dict(sample) for sample in samples]
    if not rows:
        raise ValueError("transient probe requires at least one sample")
    baseline = rows[0]
    required = {
        "stage",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "free_bytes",
        "total_bytes",
        "elapsed_ns",
    }
    missing = sorted(required - set(baseline))
    if missing:
        raise ValueError(f"baseline sample is missing fields: {missing}")
    base_allocated = int(baseline["allocated_bytes"])
    base_reserved = int(baseline["reserved_bytes"])
    base_peak_allocated = int(baseline["peak_allocated_bytes"])
    base_peak_reserved = int(baseline["peak_reserved_bytes"])
    base_free = int(baseline["free_bytes"])
    enriched: list[dict[str, Any]] = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"probe sample is missing fields: {missing}")
        enriched.append(
            {
                **row,
                "allocated_delta_bytes": int(row["allocated_bytes"])
                - base_allocated,
                "reserved_delta_bytes": int(row["reserved_bytes"]) - base_reserved,
                "peak_allocated_delta_bytes": int(row["peak_allocated_bytes"])
                - base_peak_allocated,
                "peak_reserved_delta_bytes": int(row["peak_reserved_bytes"])
                - base_peak_reserved,
                "free_delta_bytes": int(row["free_bytes"]) - base_free,
            }
        )
    max_peak_reserved = max(int(row["peak_reserved_delta_bytes"]) for row in enriched)
    max_peak_allocated = max(
        int(row["peak_allocated_delta_bytes"]) for row in enriched
    )
    max_retained_reserved = max(
        int(row["reserved_delta_bytes"]) for row in enriched
    )
    peak_reserved = next(
        row
        for row in enriched
        if int(row["peak_reserved_delta_bytes"]) == max_peak_reserved
    )
    peak_allocated = next(
        row
        for row in enriched
        if int(row["peak_allocated_delta_bytes"]) == max_peak_allocated
    )
    retained_reserved = next(
        row
        for row in enriched
        if int(row["reserved_delta_bytes"]) == max_retained_reserved
    )
    return {
        "schema_version": "eiel_prepared_owner_transient_probe_summary_v1",
        "sample_count": len(enriched),
        "peak_reserved_stage": str(peak_reserved["stage"]),
        "peak_reserved_delta_bytes": int(
            peak_reserved["peak_reserved_delta_bytes"]
        ),
        "peak_allocated_stage": str(peak_allocated["stage"]),
        "peak_allocated_delta_bytes": int(
            peak_allocated["peak_allocated_delta_bytes"]
        ),
        "max_retained_reserved_stage": str(retained_reserved["stage"]),
        "max_retained_reserved_delta_bytes": int(
            retained_reserved["reserved_delta_bytes"]
        ),
        "samples": enriched,
    }


def _compact_decision_v1(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        return {"ok": False}
    keys = (
        "ok",
        "decision",
        "route",
        "route_profile",
        "prepared_owner_resource_projection_ok",
        "measured_owner_peak_bytes",
        "measured_cuda_retained_bytes",
        "declared_budget_bytes",
        "budget_headroom_bytes",
    )
    return {key: decision[key] for key in keys if key in decision}


class TransientProbeRecorder:
    """Capture CUDA memory samples at diagnostic-only internal boundaries."""

    def __init__(self, torch_module: Any, hardware_identity: Mapping[str, Any]) -> None:
        self._torch = torch_module
        self._sample = cuda_memory_stage_sampler_v1(
            torch_module,
            expected_hardware_identity=hardware_identity,
        )
        self._started_ns = time.perf_counter_ns()
        self.samples: list[dict[str, Any]] = []

    def capture(self, stage: str) -> Any:
        if stage in PREPARE_ONLY_STAGES:
            sample = self._sample(stage)
            self.samples.append(sample.payload())
            return sample
        cuda = self._torch.cuda
        cuda.synchronize()
        free_bytes, total_bytes = cuda.mem_get_info()
        row = {
            "schema_version": "eiel_prepared_owner_transient_probe_sample_v1",
            "stage": stage,
            "elapsed_ns": max(0, time.perf_counter_ns() - self._started_ns),
            "allocated_bytes": int(cuda.memory_allocated()),
            "reserved_bytes": int(cuda.memory_reserved()),
            "peak_allocated_bytes": int(cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(cuda.max_memory_reserved()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
        self.samples.append(row)
        return row


def _selected_measurement(
    spec_path: Path,
    batch_id: str | None,
) -> tuple[dict[str, Any], str, Mapping[str, Any], Any]:
    resolved = spec_path.expanduser().resolve()
    spec = load_campaign_spec(resolved)
    plan = compile_campaign(spec).payload()
    _write_plan(plan)
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or not batches:
        raise CampaignError("compiled plan has no physical batch")
    known = [
        str(row["batch_id"])
        for row in batches
        if isinstance(row, Mapping) and isinstance(row.get("batch_id"), str)
    ]
    selected = batch_id or (known[0] if len(known) == 1 else None)
    if selected is None:
        raise CampaignError(
            "--batch-id is required when the plan has multiple physical batches"
        )
    if selected not in known:
        raise CampaignError(f"unknown physical batch: {selected!r}")
    measurement_plan = _selected_batch_measurement_plan(plan, selected)
    default_registry = default_execution_provider_registry()
    profile = resolve_campaign_profile(spec)
    preflight = preflight_campaign(
        measurement_plan,
        profile=profile,
        provider_registry=default_registry,
    )
    if preflight.get("ready_for_prepared_owner_measurement") is not True:
        reasons = [
            *preflight.get("reasons", []),
            *preflight.get("prepared_owner_resource_audit", {}).get(
                "measurement_authority_reasons", []
            ),
        ]
        raise CampaignError(
            "prepared-owner transient probe is not authorized for the exact plan: "
            + "; ".join(
                str(row.get("message", row)) if isinstance(row, Mapping) else str(row)
                for row in reasons
            )
        )
    prepared_owner = preflight.get("prepared_owner_resource_audit")
    authorization = (
        prepared_owner.get("measurement_authorization")
        if isinstance(prepared_owner, Mapping)
        else None
    )
    if not isinstance(authorization, Mapping):
        raise CampaignError("preflight did not produce measurement authorization")
    provider = default_registry.get(str(measurement_plan.get("resolved_profile", "")))
    if provider is None:
        raise CampaignError("resolved profile has no default provider")
    request = provider.translate_batch(measurement_plan, selected)
    return measurement_plan, selected, authorization, request


class TracedEielProductionExecutionProviderV1(EielProductionExecutionProviderV1):
    """Production provider with diagnostic-only adapter boundary sampling."""

    def __init__(self, recorder: TransientProbeRecorder) -> None:
        super().__init__(bind_committed_defaults=True)
        self._transient_probe_recorder = recorder

    def _prepare_generation_zero_sources_v2(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Mapping[str, Any], ...]:
        self._transient_probe_recorder.capture("probe:source_preparer:before")
        try:
            return super()._prepare_generation_zero_sources_v2(*args, **kwargs)
        finally:
            self._transient_probe_recorder.capture("probe:source_preparer:after")

    def _resolved_execution_adapters(
        self, plan: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        adapters = dict(super()._resolved_execution_adapters(plan))

        def wrap_indexed(
            name: str,
            stage_prefix: str,
        ) -> None:
            target = adapters.get(name)
            if not callable(target):
                return
            counter = {"count": 0}

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                counter["count"] += 1
                index = counter["count"]
                self._transient_probe_recorder.capture(
                    f"probe:{stage_prefix}:{index}:before"
                )
                try:
                    return target(*args, **kwargs)
                finally:
                    self._transient_probe_recorder.capture(
                        f"probe:{stage_prefix}:{index}:after"
                    )

            adapters[name] = wrapped

        def wrap_single(name: str, stage_prefix: str) -> None:
            target = adapters.get(name)
            if not callable(target):
                return

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                self._transient_probe_recorder.capture(f"probe:{stage_prefix}:before")
                try:
                    return target(*args, **kwargs)
                finally:
                    self._transient_probe_recorder.capture(
                        f"probe:{stage_prefix}:after"
                    )

            adapters[name] = wrapped

        wrap_indexed("runner_builder", "runner_builder")
        wrap_indexed("request_builder", "request_builder")
        wrap_single("preparer", "batched_preparer")
        return adapters


def _traced_provider(recorder: TransientProbeRecorder) -> EielProductionExecutionProviderV1:
    return TracedEielProductionExecutionProviderV1(recorder)


def run_probe_v1(
    spec_path: Path,
    *,
    batch_id: str | None,
    output_root: Path | None,
) -> dict[str, Any]:
    import torch

    measurement_plan, selected, authorization, request = _selected_measurement(
        spec_path,
        batch_id,
    )
    bindings = authorization.get("request_bindings")
    if not isinstance(bindings, Mapping):
        raise CampaignError("measurement authorization lacks request bindings")
    hardware_identity = bindings.get("hardware_identity")
    if not isinstance(hardware_identity, Mapping):
        raise CampaignError("measurement authorization lacks hardware identity")
    recorder = TransientProbeRecorder(torch, hardware_identity)
    provider = _traced_provider(recorder)
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    audit_id = f"{selected}-{str(measurement_plan['plan_sha256'])[:12]}-transient-probe"
    root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else _runtime_root(measurement_plan)
        / "prepared_owner_transient_probes"
        / audit_id
    )
    if root.exists():
        raise CampaignError(f"prepared-owner transient probe root exists: {root}")
    result = provider.measure_prepared_owner_resources_v1(
        measurement_plan,
        request,
        audit_root=root,
        audit_id=audit_id,
        measurement_authorization=authorization,
        stage_sampler=recorder.capture,
    )
    if not isinstance(result, Mapping):
        raise CampaignError("transient probe measurement returned no receipt")
    summary = summarize_trace_v1(recorder.samples)
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "authority": "diagnostic_sidecar_not_promotable",
        "receipt_path": result.get("receipt_path"),
        "projection_path": result.get("projection_path"),
        "decision_ok": _as_bool(result.get("decision")),
        "measurement_ok": bool(
            isinstance(result.get("decision"), Mapping)
            and result["decision"].get("ok") is True
        ),
        "summary": summary,
    }
    trace_path = root / "prepared_owner_transient_probe_trace.json"
    _atomic_json(trace_path, trace)
    return {
        "ok": bool(
            isinstance(result.get("decision"), Mapping)
            and result["decision"].get("ok") is True
        ),
        "schema_version": "eiel_prepared_owner_transient_probe_cli_result_v1",
        "command": "probe-prepared-owner-transients",
        "spec_path": str(spec_path.expanduser().resolve()),
        "receipt_path": result.get("receipt_path"),
        "projection_path": result.get("projection_path"),
        "decision": _compact_decision_v1(result.get("decision")),
        "trace_path": str(trace_path),
        "trace_summary": {
            key: value
            for key, value in summary.items()
            if key != "samples"
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace transient CUDA allocation during prepared-owner setup."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument(
        "--batch-id",
        help="physical batch ID; optional only when exactly one batch exists",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="new diagnostic audit directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            result = run_probe_v1(
                args.spec,
                batch_id=args.batch_id,
                output_root=args.output_root,
            )
    except Exception as exc:  # noqa: BLE001 - machine-readable fail-closed CLI.
        result = {
            "ok": False,
            "schema_version": "eiel_prepared_owner_transient_probe_failure_v1",
            "command": "probe-prepared-owner-transients",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
