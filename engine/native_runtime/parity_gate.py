from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


CXX19_GATE_SCHEMA_VERSION = "eiel_cxxbackbone19_formal_benchmark_gate_v1"
CXX19_DECISION_INVALID_NONPARITY = "CXXBACKBONE19-G/INVALID-NONPARITY-OR-NONISOLATED-BENCHMARK-CLOSEOUT"

REQUIRED_CHURN_FIELDS = (
    "successful_births",
    "failed_births",
    "deaths",
    "slot_reuse",
    "mutation_events",
    "lineage_changes",
    "event_rows",
    "public_output_rows",
)

REQUIRED_DIGEST_FIELDS = (
    "population_digest",
    "stable_id_slot_digest",
    "spawn_position_digest",
    "child_hash_digest",
    "rng_index_digest",
    "death_cull_refill_digest",
    "lineage_digest",
    "event_tape_digest",
    "public_output_digest",
    "checkpoint_digest",
    "final_state_digest",
)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric(row: dict[str, Any], canonical: str) -> int | None:
    aliases = {
        "successful_births": ("successful_births", "births", "final_successful_births"),
        "failed_births": ("failed_births", "final_failed_births"),
        "deaths": ("deaths", "final_deaths"),
        "slot_reuse": ("slot_reuse", "slot_reuse_rows"),
        "mutation_events": ("mutation_events",),
        "lineage_changes": ("lineage_changes",),
        "event_rows": ("event_rows", "typed_event_rows"),
        "public_output_rows": ("public_output_rows",),
    }
    for key in aliases[canonical]:
        parsed = _as_int(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _rows_by_arm(summary: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    rows = summary.get(arm)
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _parity_gate(row: dict[str, Any]) -> dict[str, Any]:
    gate = row.get("parity_gate")
    return dict(gate) if isinstance(gate, dict) else {}


def _seed(row: dict[str, Any]) -> int | None:
    return _as_int(row.get("seed"))


def _repeat(row: dict[str, Any]) -> int | None:
    return _as_int(row.get("repeat"))


def _same_seed_pairs(native_rows: Iterable[dict[str, Any]], reference_rows: Iterable[dict[str, Any]]) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    reference_by_seed: dict[int, dict[str, Any]] = {}
    for row in reference_rows:
        seed = _seed(row)
        repeat = _repeat(row)
        if seed is not None and (repeat is None or repeat == 0):
            reference_by_seed.setdefault(seed, row)
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for row in native_rows:
        seed = _seed(row)
        repeat = _repeat(row)
        if seed is not None and (repeat is None or repeat == 0) and seed in reference_by_seed:
            pairs.append((seed, row, reference_by_seed[seed]))
    return pairs


def _protocol_limitations(summary: dict[str, Any]) -> list[str]:
    raw = summary.get("protocol_limitations")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def evaluate_cxx19_formal_benchmark_gate(
    single_cell_summary: dict[str, Any],
    *,
    campaign_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return whether timing evidence satisfies the CXX19 production gate.

    The gate is deliberately stricter than a timing parser.  It rejects any row
    that lacks exact production parity evidence, real churn, three isolated
    repeats per seed/arm, and campaign evidence.
    """

    native_rows = _rows_by_arm(single_cell_summary, "native")
    reference_rows = _rows_by_arm(single_cell_summary, "reference")
    rejection_reasons: list[str] = []

    if not native_rows:
        rejection_reasons.append("missing_native_timing_rows")
    if not reference_rows:
        rejection_reasons.append("missing_reference_timing_rows")

    for limitation in _protocol_limitations(single_cell_summary):
        lowered = limitation.lower()
        if "concurrent" in lowered:
            rejection_reasons.append("timing_protocol_concurrent_runs")
        if "one timed repeat" in lowered or "not three" in lowered:
            rejection_reasons.append("timing_protocol_fewer_than_three_repeats")
        if "no formal w6" in lowered or "no campaign" in lowered:
            rejection_reasons.append("missing_formal_w6_campaign_matrix")

    for arm, rows in (("native", native_rows), ("reference", reference_rows)):
        repeats = Counter((seed, _repeat(row)) for row in rows if (seed := _seed(row)) is not None)
        seeds = sorted({seed for seed, _unused in repeats})
        for seed in seeds:
            repeat_count = sum(1 for row_seed, repeat in repeats if row_seed == seed and repeat is not None)
            if repeat_count and repeat_count < 3:
                rejection_reasons.append(f"{arm}_seed_{seed}_fewer_than_three_timed_repeats")
            elif repeat_count == 0 and len([row for row in rows if _seed(row) == seed]) < 3:
                rejection_reasons.append(f"{arm}_seed_{seed}_fewer_than_three_timed_repeats")

        for index, row in enumerate(rows):
            gate = _parity_gate(row)
            if gate.get("ok") is not True:
                rejection_reasons.append(f"{arm}_row_{index}_missing_or_failed_parity_gate")
            missing_digests = [field for field in REQUIRED_DIGEST_FIELDS if not gate.get(field) and not row.get(field)]
            if missing_digests:
                rejection_reasons.append(f"{arm}_row_{index}_missing_exact_digests")
            for field in REQUIRED_CHURN_FIELDS:
                value = _metric(row, field)
                if value is None:
                    rejection_reasons.append(f"{arm}_row_{index}_missing_{field}")
                elif value <= 0:
                    rejection_reasons.append(f"{arm}_row_{index}_nonpositive_{field}")

    for seed, native, reference in _same_seed_pairs(native_rows, reference_rows):
        for field in ("successful_births", "failed_births", "deaths"):
            native_value = _metric(native, field)
            reference_value = _metric(reference, field)
            if native_value is not None and reference_value is not None and native_value != reference_value:
                rejection_reasons.append(f"seed_{seed}_{field}_mismatch_native_{native_value}_reference_{reference_value}")

    if campaign_summary is None:
        rejection_reasons.append("missing_formal_w6_campaign_matrix")
    else:
        campaign_gate = campaign_summary.get("parity_gate") if isinstance(campaign_summary, dict) else None
        if not isinstance(campaign_gate, dict) or campaign_gate.get("ok") is not True:
            rejection_reasons.append("campaign_missing_or_failed_parity_gate")
        repeats = _as_int(campaign_summary.get("campaign_repeats")) if isinstance(campaign_summary, dict) else None
        if repeats is None or repeats < 3:
            rejection_reasons.append("campaign_fewer_than_three_repeats")

    unique_reasons = sorted(set(rejection_reasons))
    accepted = not unique_reasons
    return {
        "schema_version": CXX19_GATE_SCHEMA_VERSION,
        "accepted_for_production_timing": accepted,
        "decision_if_closed_now": (
            "CXXBACKBONE19-A/CPU-NATIVE-FOUNDATION-MERGE-READY"
            if accepted
            else CXX19_DECISION_INVALID_NONPARITY
        ),
        "native_rows": len(native_rows),
        "reference_rows": len(reference_rows),
        "required_churn_fields": list(REQUIRED_CHURN_FIELDS),
        "required_digest_fields": list(REQUIRED_DIGEST_FIELDS),
        "rejection_reasons": unique_reasons,
    }


def exact_cpu_merge_envelope_supported(capability: dict[str, Any]) -> bool:
    required = (
        "supports_exact_production_asexual_child_construction",
        "supports_exact_production_spawn_rng_order",
        "supports_exact_production_mutation_rng_budget",
        "supports_exact_death_cull_slot_reuse_lineage",
        "supports_exact_typed_event_tape",
        "supports_exact_public_output_serialization",
    )
    return all(bool(capability.get(key)) for key in required)
