#!/usr/bin/env python3
from __future__ import annotations

"""Build MEM-EVO-2 memory evolution observability summaries from metrics CSVs."""

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

try:
    from simulation.metrics_pkg.rows import MetricsRow
except Exception:  # pragma: no cover - import failure is reported in the output.
    rows_path = ENGINE_ROOT / "simulation" / "metrics_pkg" / "rows.py"
    spec = importlib.util.spec_from_file_location("mem_evo_2_rows", rows_path)
    if spec is not None and spec.loader is not None:
        rows_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = rows_module
        spec.loader.exec_module(rows_module)
        MetricsRow = rows_module.MetricsRow  # type: ignore[assignment]
    else:
        MetricsRow = None  # type: ignore[assignment]


MEMORY_BUCKETS = (0, 8, 16, 32, 64)
REQUIRED_MEM_EVO_FIELDS = (
    "mem_evo_population_memory_dim_histogram",
    "mem_evo_selected_parent_memory_dim_histogram",
    "mem_evo_descendant_memory_dim_histogram",
    "mem_evo_lineage_memory_dim_summary_json",
    "mem_evo_effective_memory_dim_histogram",
    "mem_evo_use_gru_share",
    "mem_evo_memory_cost_mean",
    "mem_evo_memory_cost_total",
    "mem_evo_memory_dim_selected_minus_population_mean",
    "mem_evo_memory_dim_descendant_minus_initial_mean",
    "mem_evo_memory_dim_wdl_minus_flat_delta",
    "mem_evo_memory_dim_shift_interpretable",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_jsonish(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"{}", "[]"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _generation(row: Mapping[str, Any], index: int) -> int:
    for key in ("generation", "gen", "final_generation"):
        if str(row.get(key, "")).strip() != "":
            try:
                return int(float(str(row[key])))
            except Exception:
                pass
    return int(index)


def _expand_final_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    payload = _load_jsonish(row.get("final_metrics"))
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.setdefault(str(key), value)
    return out


def discover_csvs(inputs: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in inputs:
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*.csv") if p.is_file()))
        elif path.is_file() and path.suffix.lower() == ".csv":
            out.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _context(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "world_context": str(row.get("world_context", row.get("context", "")) or ""),
        "memory_arm": str(row.get("memory_arm", row.get("trait_arm", "")) or ""),
        "init_arm": str(row.get("init_arm", row.get("initialization", "")) or ""),
        "seed": str(row.get("seed", "") or ""),
        "run_id": str(row.get("run_id", "") or ""),
    }


def _hist_from_json_field(row: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    payload = _load_jsonish(row.get(key))
    if isinstance(payload, dict):
        return payload
    return None


def _hist_from_buckets(row: Mapping[str, Any], prefix: str) -> dict[str, Any] | None:
    fractions: dict[str, float] = {}
    found = False
    for bucket in MEMORY_BUCKETS:
        for name in (f"{prefix}{bucket}_fraction", f"{prefix}{bucket}_frac", f"mem{bucket}_frac_alive"):
            if name in row and str(row.get(name, "")).strip() != "":
                fractions[str(bucket)] = _f(row, name)
                found = True
                break
        else:
            fractions[str(bucket)] = 0.0
    if not found:
        return None
    return {
        "buckets": {},
        "count": 0,
        "fractions": fractions,
        "mean": sum(float(bucket) * float(fractions[str(bucket)]) for bucket in MEMORY_BUCKETS),
        "source": "bucket_fraction_fallback",
    }


def population_mean(row: Mapping[str, Any]) -> float:
    hist = _hist_from_json_field(row, "mem_evo_population_memory_dim_histogram") or _hist_from_buckets(row, "memory_bucket_")
    if hist:
        return float(hist.get("mean", 0.0) or 0.0)
    for key in ("memory_dim_mean", "mean_memory_dim_alive"):
        if key in row and str(row.get(key, "")).strip() != "":
            return _f(row, key)
    return 0.0


def selected_parent_mean(row: Mapping[str, Any]) -> float:
    hist = _hist_from_json_field(row, "mem_evo_selected_parent_memory_dim_histogram")
    if hist:
        return float(hist.get("mean", 0.0) or 0.0)
    for key in ("parent_memory_dim_mean", "mean_memory_dim_parent"):
        if key in row and str(row.get(key, "")).strip() != "":
            return _f(row, key)
    return 0.0


def relevant_columns(row: Mapping[str, Any]) -> set[str]:
    needles = ("memory", "mem_evo", "gru", "lineage", "wdl", "final_metrics")
    return {key for key in row if any(needle in key for needle in needles)}


def build_summaries(csvs: list[Path]) -> dict[str, Any]:
    field_counts: dict[str, int] = defaultdict(int)
    inputs: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in csvs:
        try:
            raw_rows = read_csv(path)
        except Exception as exc:
            inputs.append({"path": str(path), "status": "read_error", "error": str(exc)})
            continue
        extracted = 0
        for index, raw in enumerate(raw_rows):
            row = _expand_final_metrics(raw)
            if not relevant_columns(row):
                continue
            extracted += 1
            for key in relevant_columns(row):
                if str(row.get(key, "")).strip() != "":
                    field_counts[key] += 1
            ctx = _context(row)
            generation = _generation(row, index)
            pop_hist = (
                _hist_from_json_field(row, "mem_evo_population_memory_dim_histogram")
                or _hist_from_buckets(row, "memory_bucket_")
                or _hist_from_buckets(row, "mem")
            )
            parent_hist = _hist_from_json_field(row, "mem_evo_selected_parent_memory_dim_histogram")
            pop_mean = population_mean(row)
            parent_mean = selected_parent_mean(row)
            source = str(path)
            has_population_observation = bool(
                pop_hist or any(key in row and str(row.get(key, "")).strip() != "" for key in ("memory_dim_mean", "mean_memory_dim_alive"))
            )
            if has_population_observation:
                rows_by_source[source].append(
                    {
                        "generation": generation,
                        "population_mean": pop_mean,
                        "row": row,
                        "source": source,
                        **ctx,
                    }
                )
                population_rows.append(
                    {
                        "source": source,
                        "generation": generation,
                        **ctx,
                        "population_memory_dim_mean": pop_mean,
                        "population_memory_dim_p90": _f(row, "memory_dim_p90", _f(row, "p90_memory_dim_alive")),
                        "population_memory_dim_max": _f(row, "memory_dim_max", _f(row, "max_memory_dim_alive")),
                        "population_histogram_json": json.dumps(pop_hist or {}, sort_keys=True),
                        "effective_memory_dim_mean": _f(row, "effective_memory_mean"),
                        "use_gru_share": _f(row, "mem_evo_use_gru_share", _f(row, "gru_enabled_fraction", _f(row, "gru_alive_frac"))),
                        "memory_cost_mean": _f(row, "mem_evo_memory_cost_mean", _f(row, "memory_cost_mean")),
                        "memory_cost_total": _f(row, "mem_evo_memory_cost_total", _f(row, "memory_cost_total")),
                    }
                )
            if parent_hist or any(key in row for key in ("mean_memory_dim_parent", "parent_memory_dim_mean")):
                parent_rows.append(
                    {
                        "source": source,
                        "generation": generation,
                        **ctx,
                        "selected_parent_memory_dim_mean": parent_mean,
                        "population_memory_dim_mean": pop_mean,
                        "selected_minus_population_mean": _f(
                            row,
                            "mem_evo_memory_dim_selected_minus_population_mean",
                            parent_mean - pop_mean,
                        ),
                        "selected_parent_memory_dim_p90": _f(row, "p90_memory_dim_parent"),
                        "selected_parent_histogram_json": json.dumps(parent_hist or {}, sort_keys=True),
                    }
                )
            lineage_payload = _load_jsonish(row.get("mem_evo_lineage_memory_dim_summary_json"))
            if isinstance(lineage_payload, dict):
                lineage_rows.append(
                    {
                        "source": source,
                        "generation": generation,
                        **ctx,
                        "lineage_count": lineage_payload.get("lineage_count", 0),
                        "mean_lineage_memory_dim_mean": lineage_payload.get("mean_lineage_memory_dim_mean", 0.0),
                        "lineage_memory_dim_summary_json": json.dumps(lineage_payload, sort_keys=True),
                    }
                )
            elif "lineage_memory_mean" in row or "top_lineage_memory_dim" in row:
                lineage_rows.append(
                    {
                        "source": source,
                        "generation": generation,
                        **ctx,
                        "lineage_count": "",
                        "mean_lineage_memory_dim_mean": _f(row, "lineage_memory_mean"),
                        "lineage_memory_dim_summary_json": "{}",
                    }
                )
        inputs.append({"path": str(path), "row_count": len(raw_rows), "extracted_row_count": extracted, "status": "ok"})

    descendant_rows: list[dict[str, Any]] = []
    for source, rows in sorted(rows_by_source.items()):
        if not rows:
            continue
        ordered = sorted(rows, key=lambda item: int(item["generation"]))
        first = ordered[0]
        last = ordered[-1]
        descendant_rows.append(
            {
                "source": source,
                "run_id": last.get("run_id", ""),
                "world_context": last.get("world_context", ""),
                "memory_arm": last.get("memory_arm", ""),
                "init_arm": last.get("init_arm", ""),
                "seed": last.get("seed", ""),
                "initial_generation": first["generation"],
                "final_generation": last["generation"],
                "initial_population_memory_dim_mean": first["population_mean"],
                "final_descendant_memory_dim_mean": last["population_mean"],
                "descendant_minus_initial_mean": float(last["population_mean"]) - float(first["population_mean"]),
            }
        )

    schema_fields = set(getattr(MetricsRow, "__annotations__", {}).keys()) if MetricsRow is not None else set()
    schema_present = {field: field in schema_fields for field in REQUIRED_MEM_EVO_FIELDS}
    extracted_present = {field: field_counts.get(field, 0) for field in REQUIRED_MEM_EVO_FIELDS}
    current_schema_complete = all(schema_present.values())
    existing_shift_interpretable = False
    if parent_rows and descendant_rows:
        existing_shift_interpretable = any(abs(float(row["selected_minus_population_mean"])) > 0.0 for row in parent_rows) and any(
            abs(float(row["descendant_minus_initial_mean"])) > 0.0 for row in descendant_rows
        )

    decision = "MEM-EVO-2B" if current_schema_complete else "MEM-EVO-2C"
    if existing_shift_interpretable and current_schema_complete:
        decision = "MEM-EVO-2A"

    gap_table = [
        {
            "surface": "full population memory_dim distribution by generation",
            "schema_status": "implemented" if schema_present["mem_evo_population_memory_dim_histogram"] else "missing",
            "existing_evidence_status": "extractable" if population_rows else "not_found",
        },
        {
            "surface": "selected-parent memory_dim distribution by generation",
            "schema_status": "implemented" if schema_present["mem_evo_selected_parent_memory_dim_histogram"] else "missing",
            "existing_evidence_status": "extractable" if parent_rows else "not_found",
        },
        {
            "surface": "offspring/descendant memory_dim distribution after reproduction/cull/refill",
            "schema_status": "implemented" if schema_present["mem_evo_descendant_memory_dim_histogram"] else "missing",
            "existing_evidence_status": "extractable" if descendant_rows else "not_found",
        },
        {
            "surface": "lineage-level memory_dim summaries",
            "schema_status": "implemented" if schema_present["mem_evo_lineage_memory_dim_summary_json"] else "missing",
            "existing_evidence_status": "extractable" if lineage_rows else "not_found",
        },
        {
            "surface": "effective memory, use_gru, and memory cost distributions",
            "schema_status": "implemented"
            if all(schema_present[k] for k in ("mem_evo_effective_memory_dim_histogram", "mem_evo_use_gru_share", "mem_evo_memory_cost_mean", "mem_evo_memory_cost_total"))
            else "missing",
            "existing_evidence_status": "extractable" if population_rows else "not_found",
        },
        {
            "surface": "flat/WDL and memory_normal/memory_ablated grouping",
            "schema_status": "extractor_grouping",
            "existing_evidence_status": "partial" if any(row.get("world_context") or row.get("memory_arm") for row in population_rows) else "not_found",
        },
    ]

    return {
        "inputs": inputs,
        "field_inventory": {
            "schema_present": schema_present,
            "extracted_nonempty_counts": dict(sorted(field_counts.items())),
            "required_extracted_nonempty_counts": extracted_present,
        },
        "gap_table": gap_table,
        "population_rows": population_rows,
        "parent_rows": parent_rows,
        "descendant_rows": descendant_rows,
        "lineage_rows": lineage_rows,
        "decision": {
            "decision": decision,
            "current_schema_complete": bool(current_schema_complete),
            "existing_shift_interpretable": bool(existing_shift_interpretable),
            "population_row_count": len(population_rows),
            "selected_parent_row_count": len(parent_rows),
            "descendant_row_count": len(descendant_rows),
            "lineage_row_count": len(lineage_rows),
            "recommendation": "Run a narrow MEM-EVO selection-shift follow-up with the new reporting fields."
            if decision == "MEM-EVO-2B"
            else "Repair reporting before interpreting memory evolution.",
        },
    }


def write_outputs(out_dir: Path, summary: Mapping[str, Any]) -> None:
    write_json(out_dir / "memory_evolution_field_inventory.json", summary["field_inventory"])
    write_json(out_dir / "memory_evolution_gap_table.json", summary["gap_table"])
    write_json(out_dir / "memory_dim_distribution_summary.json", summary["population_rows"])
    write_json(out_dir / "selected_parent_memory_dim_summary.json", summary["parent_rows"])
    write_json(out_dir / "descendant_memory_dim_summary.json", summary["descendant_rows"])
    write_json(out_dir / "lineage_memory_dim_summary.json", summary["lineage_rows"])
    write_json(out_dir / "mem_evo_2_decision.json", summary["decision"])
    common = ["source", "generation", "run_id", "world_context", "memory_arm", "init_arm", "seed"]
    write_csv(
        out_dir / "memory_dim_distribution_summary.csv",
        summary["population_rows"],
        common
        + [
            "population_memory_dim_mean",
            "population_memory_dim_p90",
            "population_memory_dim_max",
            "effective_memory_dim_mean",
            "use_gru_share",
            "memory_cost_mean",
            "memory_cost_total",
            "population_histogram_json",
        ],
    )
    write_csv(
        out_dir / "selected_parent_memory_dim_summary.csv",
        summary["parent_rows"],
        common
        + [
            "selected_parent_memory_dim_mean",
            "population_memory_dim_mean",
            "selected_minus_population_mean",
            "selected_parent_memory_dim_p90",
            "selected_parent_histogram_json",
        ],
    )
    write_csv(
        out_dir / "descendant_memory_dim_summary.csv",
        summary["descendant_rows"],
        [
            "source",
            "run_id",
            "world_context",
            "memory_arm",
            "init_arm",
            "seed",
            "initial_generation",
            "final_generation",
            "initial_population_memory_dim_mean",
            "final_descendant_memory_dim_mean",
            "descendant_minus_initial_mean",
        ],
    )
    write_csv(
        out_dir / "lineage_memory_dim_summary.csv",
        summary["lineage_rows"],
        common + ["lineage_count", "mean_lineage_memory_dim_mean", "lineage_memory_dim_summary_json"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=[], help="CSV file or directory to scan; may repeat.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = args.input or [PROJECT_ROOT / "docs" / "40_eiel_design"]
    out_dir = args.out_dir.resolve()
    csvs = [path for path in discover_csvs(inputs) if not path.resolve().is_relative_to(out_dir)]
    summary = build_summaries(csvs)
    write_outputs(args.out_dir, summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
