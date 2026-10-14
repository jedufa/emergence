from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


COSTSURFACE_SOA_DIAGNOSTIC_SCHEMA_VERSION = "costsurface_soa_diagnostic_summary_v1"

COSTSURFACE_SOA_FLOAT_FIELDS: tuple[str, ...] = (
    "sensor_channel_cost_per_tick",
    "sensor_radius_cost_hypothetical",
    "prediction_cost_charged",
    "developmental_cost_charged",
    "maturation_cost_charged",
    "introspection_cost_charged",
    "self_regulation_cost_charged",
    "self_modification_cost_charged",
)


def costsurface_soa_row(surface: Any) -> np.ndarray:
    """Pack one CostChargeSurface-like object into the COSTSOA-1 row contract."""
    return np.asarray([float(getattr(surface, field_name)) for field_name in COSTSURFACE_SOA_FLOAT_FIELDS], dtype=np.float64)


def costsurface_soa_scalar_from_row(
    row: Any,
    *,
    enable_cognitive_budget_ledger: bool = True,
    charge_channels: bool = False,
    charge_radius: bool = False,
) -> float:
    if not bool(enable_cognitive_budget_ledger):
        return 0.0
    matrix_row = np.asarray(row, dtype=np.float64)
    perception = (
        (float(matrix_row[0]) if bool(charge_channels) else 0.0)
        + (float(matrix_row[1]) if bool(charge_radius) else 0.0)
    )
    return float(
        float(perception)
        + float(matrix_row[2])
        + float(matrix_row[3])
        + float(matrix_row[4])
        + float(matrix_row[5])
        + float(matrix_row[6])
        + float(matrix_row[7])
    )


@dataclass
class CostSurfaceSoADiagnostic:
    run_dir: str
    enabled: bool = False
    field_names: tuple[str, ...] = COSTSURFACE_SOA_FLOAT_FIELDS
    rows: list[np.ndarray] = field(default_factory=list)
    rows_produced: int = 0
    active_agents_covered: int = 0
    missing_fallback_rows: int = 0
    estimated_object_reads: int = 0
    production_s: float = 0.0
    current_scalar_extraction_s: float = 0.0
    max_field_abs_diff: dict[str, float] = field(default_factory=dict)
    field_mismatch_counts: dict[str, int] = field(default_factory=dict)
    scalar_mismatch_count: int = 0
    max_scalar_abs_diff: float = 0.0
    skipped_reason_counts: dict[str, int] = field(default_factory=dict)

    def record_surface(
        self,
        surface: Any,
        *,
        current_scalar: float,
        current_scalar_s: float = 0.0,
        sim_cfg: Any = None,
    ) -> None:
        if not bool(self.enabled):
            return
        started = time.perf_counter()
        row = costsurface_soa_row(surface)
        self.rows.append(row)
        self.rows_produced += 1
        self.active_agents_covered += 1
        self.estimated_object_reads += len(self.field_names)
        for index, field_name in enumerate(self.field_names):
            diff = abs(float(row[index]) - float(getattr(surface, field_name)))
            self.max_field_abs_diff[field_name] = max(float(self.max_field_abs_diff.get(field_name, 0.0)), float(diff))
            if diff > 1.0e-12:
                self.field_mismatch_counts[field_name] = int(self.field_mismatch_counts.get(field_name, 0)) + 1
        row_scalar = costsurface_soa_scalar_from_row(
            row,
            enable_cognitive_budget_ledger=bool(getattr(sim_cfg, "enable_cognitive_budget_ledger", True)),
            charge_channels=bool(getattr(sim_cfg, "enable_experimental_perception_cost_charging", False)),
            charge_radius=bool(getattr(sim_cfg, "enable_experimental_sensor_radius_cost_charging", False)),
        )
        scalar_diff = abs(float(row_scalar) - float(current_scalar))
        self.max_scalar_abs_diff = max(float(self.max_scalar_abs_diff), float(scalar_diff))
        if scalar_diff > 1.0e-12:
            self.scalar_mismatch_count += 1
        self.current_scalar_extraction_s += max(0.0, float(current_scalar_s))
        self.production_s += float(time.perf_counter() - started)

    def record_fallback(self, reason: str) -> None:
        if not bool(self.enabled):
            return
        key = str(reason or "unknown")
        self.missing_fallback_rows += 1
        self.skipped_reason_counts[key] = int(self.skipped_reason_counts.get(key, 0)) + 1

    def summary_payload(self) -> dict[str, Any]:
        if self.rows:
            matrix = np.vstack(self.rows).astype(np.float64, copy=False)
        else:
            matrix = np.empty((0, len(self.field_names)), dtype=np.float64)
        return {
            "schema_version": COSTSURFACE_SOA_DIAGNOSTIC_SCHEMA_VERSION,
            "field_names": list(self.field_names),
            "dtype": str(matrix.dtype),
            "array_shape": [int(v) for v in matrix.shape],
            "array_memory_bytes": int(matrix.nbytes),
            "rows_produced": int(self.rows_produced),
            "active_agents_covered": int(self.active_agents_covered),
            "missing_fallback_rows": int(self.missing_fallback_rows),
            "field_mismatch_counts": {
                field_name: int(self.field_mismatch_counts.get(field_name, 0)) for field_name in self.field_names
            },
            "max_numeric_diff_by_field": {
                field_name: float(self.max_field_abs_diff.get(field_name, 0.0)) for field_name in self.field_names
            },
            "scalar_mismatch_count": int(self.scalar_mismatch_count),
            "max_scalar_abs_diff": float(self.max_scalar_abs_diff),
            "production_s": float(self.production_s),
            "current_scalar_extraction_s": float(self.current_scalar_extraction_s),
            "estimated_avoidable_object_reads": int(self.estimated_object_reads),
            "skipped_reason_counts": dict(sorted(self.skipped_reason_counts.items())),
            "authoritative_runtime_behavior": "current CostChargeSurface path remains authoritative; this is diagnostics only",
        }

    def write_summary(self) -> str:
        metrics_dir = os.path.join(str(self.run_dir), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        out_path = os.path.join(metrics_dir, "costsurface_soa_diagnostic_summary.json")
        payload = self.summary_payload()
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, out_path)
        return out_path
