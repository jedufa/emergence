#!/usr/bin/env python3
"""Collect hereditary-capacity evidence from exact Genome V2 checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Final, Mapping, Sequence


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.authoritative_state.schema import content_sha256, file_sha256  # noqa: E402
from engine.evolution.genome_upgrade_v2 import (  # noqa: E402
    PARAMETER_COUNT,
    PRODUCT_COUNT,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    PREREGISTRATION_SHA256,
    REPEAT_INDICES,
    validate_authority_v1,
)
from engine.native_runtime import (  # noqa: E402
    decode_bundled_native_checkpoint_projection,
)


SOURCE_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_hereditary_capacity_repeat_source_v1"
)
COLLECTOR_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_hereditary_capacity_collector_v1"
)
EXECUTION_SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_hereditary_capacity_execution_receipt_v1"
)
LOCUS_COUNT: Final[int] = PRODUCT_COUNT * (PARAMETER_COUNT + 1)


class HereditaryCapacityCollectorError(RuntimeError):
    """Exact hereditary-capacity evidence cannot be established."""


def _fail(message: str) -> None:
    raise HereditaryCapacityCollectorError(message)


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HereditaryCapacityCollectorError(
            f"{label} is not readable JSON: {path}"
        ) from exc


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return dict(value)


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _checkpoint_parts(payload: Any) -> tuple[dict[str, Any], list[bool]]:
    root = _object(payload, label="checkpoint")
    try:
        projection = decode_bundled_native_checkpoint_projection(root)
    except (TypeError, ValueError) as exc:
        raise HereditaryCapacityCollectorError(
            "checkpoint has no exact Genome V2 resident tensors; compact or "
            "terminal summaries cannot establish longitudinal retention"
        ) from exc
    bank = projection.get("genome_upgrade_v2_checkpoint_projection")
    if bank is None:
        _fail(
            "checkpoint has no exact Genome V2 resident tensors; compact or "
            "terminal summaries cannot establish longitudinal retention"
        )
    bank = _object(bank, label="Genome V2 checkpoint projection")
    tensors = _object(bank.get("tensors"), label="Genome V2 resident tensors")
    required = {
        "genome_v2_alleles_q10",
        "genome_v2_modifiers_q10",
        "genome_v2_products_q10",
        "genome_v2_phenotypes_q10",
        "genome_v2_expression_cost_q10",
        "genome_v2_saturated_product_count",
    }
    if not required.issubset(tensors):
        _fail("Genome V2 resident tensor set is incomplete")
    outer = _object(
        projection.get("tensors"), label="native resident checkpoint tensors"
    )
    alive_tensor = outer.get("alive_mask")
    if not hasattr(alive_tensor, "detach"):
        _fail("checkpoint alive_mask is absent")
    alive_raw = alive_tensor.detach().cpu().tolist()
    if alive_raw and isinstance(alive_raw[0], list):
        if len(alive_raw) != 1:
            _fail("hereditary repeat checkpoint must contain exactly one cell")
        alive_raw = alive_raw[0]
    if not alive_raw or any(type(value) not in (bool, int) for value in alive_raw):
        _fail("checkpoint alive_mask is malformed")
    alive = [bool(value) for value in alive_raw]
    if not any(alive):
        _fail("checkpoint has no alive resident population")
    return tensors, alive


def _cell_tensor(
    tensors: Mapping[str, Any], name: str, *, capacity: int
) -> Any:
    value = tensors[name]
    if not isinstance(value, list) or len(value) != 1:
        _fail(f"{name} must contain exactly one cell")
    cell = value[0]

    def _last_width(item: Any) -> int | None:
        while isinstance(item, list) and item:
            if all(not isinstance(child, list) for child in item):
                return len(item)
            item = item[0]
        return None

    if _last_width(cell) != capacity:
        _fail(f"{name} capacity differs from alive_mask")
    return cell


def _snapshot(payload: Any, generation: int) -> dict[str, Any]:
    tensors, alive = _checkpoint_parts(payload)
    capacity = len(alive)
    alleles = _cell_tensor(
        tensors, "genome_v2_alleles_q10", capacity=capacity
    )
    modifiers = _cell_tensor(
        tensors, "genome_v2_modifiers_q10", capacity=capacity
    )
    products = _cell_tensor(
        tensors, "genome_v2_products_q10", capacity=capacity
    )
    phenotypes = _cell_tensor(
        tensors, "genome_v2_phenotypes_q10", capacity=capacity
    )
    costs = _cell_tensor(
        tensors, "genome_v2_expression_cost_q10", capacity=capacity
    )
    saturated = _cell_tensor(
        tensors, "genome_v2_saturated_product_count", capacity=capacity
    )
    if (
        len(alleles) != 2
        or any(len(homolog) != PRODUCT_COUNT for homolog in alleles)
        or any(
            len(parameters) != PARAMETER_COUNT
            for homolog in alleles
            for parameters in homolog
        )
        or len(modifiers) != 2
        or any(len(homolog) != PRODUCT_COUNT for homolog in modifiers)
        or len(products) != PRODUCT_COUNT
    ):
        _fail("Genome V2 resident tensor dimensions differ from contract")
    slots = [index for index, is_alive in enumerate(alive) if is_alive]
    genotypes: list[tuple[int, ...]] = []
    homologs: list[tuple[int, ...]] = []
    locus_sets = [set() for _ in range(LOCUS_COUNT)]
    hidden: set[tuple[int, int]] = set()
    heterozygous = 0
    for slot in slots:
        genotype: list[int] = []
        per_homolog: list[list[int]] = [[], []]
        locus = 0
        for product in range(PRODUCT_COUNT):
            for parameter in range(PARAMETER_COUNT):
                pair = (
                    int(alleles[0][product][parameter][slot]),
                    int(alleles[1][product][parameter][slot]),
                )
                for homolog, value in enumerate(pair):
                    genotype.append(value)
                    per_homolog[homolog].append(value)
                    locus_sets[locus].add(value)
                if pair[0] != pair[1]:
                    heterozygous += 1
                    hidden.update((locus, value) for value in pair)
                locus += 1
        for product in range(PRODUCT_COUNT):
            pair = (
                int(modifiers[0][product][slot]),
                int(modifiers[1][product][slot]),
            )
            for homolog, value in enumerate(pair):
                genotype.append(value)
                per_homolog[homolog].append(value)
                locus_sets[locus].add(value)
            if pair[0] != pair[1]:
                heterozygous += 1
                hidden.update((locus, value) for value in pair)
            locus += 1
        genotypes.append(tuple(genotype))
        homologs.extend(tuple(row) for row in per_homolog)
    product_rows = {
        tuple(int(products[index][slot]) for index in range(len(products)))
        for slot in slots
    }
    phenotype_rows = {
        tuple(int(phenotypes[index][slot]) for index in range(len(phenotypes)))
        for slot in slots
    }
    return {
        "generation": generation,
        "population": len(slots),
        "genotypes": set(genotypes),
        "homologs": homologs,
        "locus_sets": locus_sets,
        "hidden": hidden,
        "heterozygous": heterozygous,
        "products": product_rows,
        "phenotypes": phenotype_rows,
        "expression_cost_q10": sum(int(costs[slot]) for slot in slots),
        "all_saturated": all(
            int(saturated[slot]) >= PRODUCT_COUNT for slot in slots
        ),
    }


def _linkage_disequilibrium_q10(snapshot: Mapping[str, Any]) -> int:
    homologs = list(snapshot["homologs"])
    if not homologs:
        _fail("linkage disequilibrium has no homolog observations")
    terms: list[float] = []
    # Products are inherited in two-product linkage blocks. Compare matching
    # parameter and modifier loci across each block using exact allele indicators.
    for block in range(PRODUCT_COUNT // 2):
        left_product = 2 * block
        right_product = left_product + 1
        for parameter in range(PARAMETER_COUNT + 1):
            if parameter < PARAMETER_COUNT:
                left = left_product * PARAMETER_COUNT + parameter
                right = right_product * PARAMETER_COUNT + parameter
            else:
                left = PRODUCT_COUNT * PARAMETER_COUNT + left_product
                right = PRODUCT_COUNT * PARAMETER_COUNT + right_product
            pairs = [(row[left], row[right]) for row in homologs]
            left_counts = Counter(value for value, _ in pairs)
            right_counts = Counter(value for _, value in pairs)
            pair_counts = Counter(pairs)
            n = len(pairs)
            for left_value in left_counts:
                for right_value in right_counts:
                    terms.append(
                        abs(
                            pair_counts[(left_value, right_value)] / n
                            - (left_counts[left_value] / n)
                            * (right_counts[right_value] / n)
                        )
                    )
    return 0 if not terms else min(1024, round(1024 * sum(terms) / len(terms)))


def _execution_flags(
    receipt: Any, *, authority: Mapping[str, str]
) -> tuple[bool, bool]:
    row = _object(receipt, label="execution receipt")
    supplied = row.pop("receipt_sha256", None)
    if (
        row.get("schema_version") != EXECUTION_SCHEMA_VERSION
        or supplied != content_sha256(row)
    ):
        _fail("execution receipt schema or hash differs")
    expected = {
        "schema_version",
        "plan_sha256",
        "batch_manifest_sha256",
        "batch_id",
        "cell_id",
        "artifact_sha256",
        "execution_route_sha256",
        "execution_device",
        "explicit_native_no_fallback",
        "hidden_fallback",
        "fallback_used",
        "runtime_jit",
    }
    if set(row) != expected or any(
        type(row.get(name)) is not bool
        for name in (
            "explicit_native_no_fallback",
            "hidden_fallback",
            "fallback_used",
            "runtime_jit",
        )
    ):
        _fail("execution receipt fields differ from contract")
    for name in (
        "plan_sha256",
        "batch_manifest_sha256",
        "artifact_sha256",
        "execution_route_sha256",
    ):
        _digest(row.get(name), label=f"execution receipt {name}")
    if (
        not isinstance(row.get("batch_id"), str)
        or not row["batch_id"]
        or not isinstance(row.get("cell_id"), str)
        or not row["cell_id"]
    ):
        _fail("execution receipt batch or cell identity is invalid")
    if (
        row["artifact_sha256"] != authority["artifact_sha256"]
        or row["execution_route_sha256"]
        != authority["execution_route_sha256"]
    ):
        _fail("execution receipt differs from hereditary authority")
    if row["execution_device"] != "cuda":
        _fail("hereditary-capacity production evidence must use CUDA")
    return (
        row["explicit_native_no_fallback"]
        and not row["hidden_fallback"]
        and not row["fallback_used"],
        not row["runtime_jit"],
    )


def collect_repeat_v1(
    source_path: Path, *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    source = _object(_read_json(source_path, label="repeat source"), label="repeat source")
    if set(source) != {
        "schema_version",
        "repeat_index",
        "authority",
        "execution_receipt",
        "observations",
    } or source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        _fail("repeat source fields or schema differ from contract")
    canonical_authority = validate_authority_v1(authority)
    if validate_authority_v1(source.get("authority")) != canonical_authority:
        _fail("repeat source authority differs")
    repeat = source.get("repeat_index")
    if type(repeat) is not int or repeat not in REPEAT_INDICES:
        _fail("repeat index is outside the exact preregistered set")
    no_fallback, no_jit = _execution_flags(
        source.get("execution_receipt"), authority=canonical_authority
    )
    observations = source.get("observations")
    if not isinstance(observations, list) or len(observations) < 2:
        _fail("longitudinal evidence requires at least two checkpoints")
    snapshots: list[dict[str, Any]] = []
    generations: list[int] = []
    for index, raw in enumerate(observations):
        observation = _object(raw, label=f"observation {index}")
        if set(observation) != {
            "generation",
            "checkpoint_path",
            "checkpoint_sha256",
        }:
            _fail(f"observation {index} fields differ from contract")
        generation = observation.get("generation")
        if type(generation) is not int or generation < 0:
            _fail(f"observation {index} generation is invalid")
        checkpoint_path = Path(str(observation.get("checkpoint_path")))
        if not checkpoint_path.is_absolute():
            checkpoint_path = source_path.parent / checkpoint_path
        expected_hash = _digest(
            observation.get("checkpoint_sha256"),
            label=f"observation {index} checkpoint hash",
        )
        if file_sha256(checkpoint_path) != expected_hash:
            _fail(f"observation {index} checkpoint hash differs")
        snapshots.append(
            _snapshot(
                _read_json(checkpoint_path, label=f"checkpoint {index}"),
                generation,
            )
        )
        generations.append(generation)
    if generations[0] != 0 or any(
        right <= left for left, right in zip(generations, generations[1:])
    ):
        _fail("checkpoint generations must start at zero and strictly increase")
    baseline, terminal = snapshots[0], snapshots[-1]
    baseline_alleles = {
        (locus, value)
        for locus, values in enumerate(baseline["locus_sets"])
        for value in values
    }
    terminal_alleles = {
        (locus, value)
        for locus, values in enumerate(terminal["locus_sets"])
        for value in values
    }
    hidden = set(baseline["hidden"])
    novel = terminal_alleles - baseline_alleles
    homogenization_generation: int | None = None
    if len(terminal["genotypes"]) == 1:
        if any(
            right != left + 1
            for left, right in zip(generations, generations[1:])
        ):
            _fail(
                "sparse checkpoints cannot establish the exact first "
                "homogenization generation"
            )
        homogenized = [
            row for row in snapshots[1:] if len(row["genotypes"]) == 1
        ]
        if homogenized:
            homogenization_generation = int(homogenized[0]["generation"])
    metrics = {
        "allele_count": len(terminal_alleles),
        "genotype_count": len(terminal["genotypes"]),
        "heterozygosity_q10": round(
            1024
            * int(terminal["heterozygous"])
            / (int(terminal["population"]) * LOCUS_COUNT)
        ),
        "hidden_allele_count": len(hidden),
        "retained_hidden_allele_count": len(hidden & terminal_alleles),
        "fixed_allele_count": sum(
            len(baseline["locus_sets"][index]) > 1
            and len(terminal["locus_sets"][index]) == 1
            for index in range(LOCUS_COUNT)
        ),
        "lost_allele_count": len(baseline_alleles - terminal_alleles),
        "retained_package_count": len(
            set(baseline["genotypes"]) & set(terminal["genotypes"])
        ),
        "linkage_disequilibrium_abs_q10": _linkage_disequilibrium_q10(terminal),
        "homogenization_generation": homogenization_generation,
        "expression_cost_q10": int(terminal["expression_cost_q10"]),
        "runtime_cost_q10": 0,
    }
    causal = bool(novel) and (
        baseline["products"] != terminal["products"]
        or baseline["phenotypes"] != terminal["phenotypes"]
    )
    value = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": canonical_authority,
        "repeat_index": repeat,
        "metrics": metrics,
        "causal_effect_observed": causal,
        "cost_tradeoff_observed": causal
        and metrics["expression_cost_q10"] > 0,
        "mutation_reachable": bool(novel),
        "immediate_cap_saturation": bool(baseline["all_saturated"]),
        "no_fallback": no_fallback,
        "no_jit": no_jit,
    }
    return {**value, "evidence_sha256": content_sha256(value)}


def materialize_v1(
    *,
    authority: Mapping[str, Any],
    repeat_sources: Sequence[Path],
    output_path: Path,
) -> list[dict[str, Any]]:
    canonical_authority = validate_authority_v1(authority)
    rows = [
        collect_repeat_v1(path, authority=canonical_authority)
        for path in repeat_sources
    ]
    if sorted(row["repeat_index"] for row in rows) != list(REPEAT_INDICES):
        _fail("sources must select exactly repeats 0 and 1")
    rows.sort(key=lambda row: row["repeat_index"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise HereditaryCapacityCollectorError(
            "evidence output already exists; immutable overwrite refused"
        ) from exc
    return rows


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument(
        "--repeat-source", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        authority = validate_authority_v1(
            _read_json(args.authority, label="authority")
        )
        rows = materialize_v1(
            authority=authority,
            repeat_sources=args.repeat_source,
            output_path=args.output,
        )
    except (HereditaryCapacityCollectorError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": COLLECTOR_SCHEMA_VERSION,
                "output_path": str(args.output),
                "repeat_count": len(rows),
                "evidence_sha256": content_sha256(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
