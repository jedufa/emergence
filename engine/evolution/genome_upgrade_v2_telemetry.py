"""Bounded lineage/population telemetry for GENOME-UPGRADE-V2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import copy
import json
from typing import Any

import torch

from engine.authoritative_state.schema import content_sha256
from evolution.genome_upgrade_v2 import (
    ALLELE_PARAMETERS,
    CONTEXT_COUNT,
    PHENOTYPE_CHANNELS,
    POLYGENIC_WEIGHTS_Q10,
    PRODUCT_CHANNELS,
    Q10_SCALE,
    GenomeUpgradeV2Error,
    GenomeUpgradeV2V1,
    express_reference_v1,
)
from .genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)
from .genome_upgrade_v2_bd_capabilities_v1 import (
    polygenic_decomposition_v1,
)

COMPACT_SUMMARY_SCHEMA_VERSION = "genome_upgrade_v2_compact_generation_summary_v2"
CHRONICLE_SCHEMA_VERSION = "genome_v2_chronicle_v2"
REPORTING_SCHEMA_VERSION = "genome_v2_reporting_v2"
HEREDITARY_CAPACITY_SCHEMA_VERSION = "genome_v2_hereditary_capacity_summary_v1"
RECOMB72_LONGITUDINAL_SCHEMA_VERSION = "genome_v2_recomb72_longitudinal_summary_v1"
E1_METRIC_ATTRIBUTION_SCHEMA_VERSION = "genome_v2_e1_metric_attribution_v1"
OBSERVABILITY_STORAGE_BUDGET_BYTES = 256 * 1024
SELECTED_EVENT_LIMIT = 16
HEREDITARY_LOCUS_COUNT = len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS) + len(
    PRODUCT_CHANNELS
)


def reporting_schema_v1() -> dict[str, Any]:
    return {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "capability_contract_sha256": (GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256),
        "source_schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
        "population_selection": "authoritative_alive_mask",
        "per_cell_outputs": [
            "product_mean_std_min_max_q10",
            "product_saturation",
            "expression_cost_q10",
            "phenotype_mean_std_min_max_q10",
            "phenotype_saturation",
            "polygenic_contribution_decomposition",
            "context_exposure",
            "consequence_delta_by_surface",
            "total_applied_delta",
            "cap_binding_by_surface",
            "bounded_selected_causal_events",
            "bounded_hereditary_capacity_summary",
        ],
        "selected_event_limit": SELECTED_EVENT_LIMIT,
        "context_measurement": ("terminal_generation_boundary_population_snapshot"),
        "selected_event_semantics": (
            "bounded_terminal_snapshot_not_event_time_evidence"
        ),
        "raw_tick_arrays_retained": False,
        "hereditary_capacity": {
            "schema_version": HEREDITARY_CAPACITY_SCHEMA_VERSION,
            "maximum_locus_summaries": HEREDITARY_LOCUS_COUNT,
            "measurement": (
                "terminal_alive_population_exact_resident_tensor_reduction"
            ),
            "retention_semantics": (
                "terminal_presence_only_not_longitudinal_persistence"
            ),
            "genotype_identity": "ordered_homolog_q10_vector_v1",
            "phenotype_identity": "exact_phenotype_q10_vector_v1",
            "raw_genotypes_retained": False,
        },
        "recomb72_gv2_longitudinal": {
            "schema_version": RECOMB72_LONGITUDINAL_SCHEMA_VERSION,
            "measurement": (
                "generation_boundary_checkpoint_alive_population_exact_"
                "resident_tensor_reduction"
            ),
            "milestones": [0, 5, 10, 20, 30],
            "all_locus_exact_counts": True,
            "ordered_homolog_package_classification": True,
            "authoritative_genotype_identity": ("genome_upgrade_v2_v1.genotype_sha256"),
            "association_source": (
                "exact_phased_2x2_counts_and_individual_b_dosage_moments"
            ),
            "changed_loci_counter_required_for_science": True,
            "raw_agent_rows_retained": False,
        },
        "compatibility": {
            "base_v2_contract_fields_stable": True,
            "optional_typed_cell_extensions": {
                "recomb72_gv2": RECOMB72_LONGITUDINAL_SCHEMA_VERSION,
            },
            "version_policy": (
                "typed_optional_profile_extensions_do_not_change_base_v2_"
                "reader_semantics"
            ),
        },
        "authority": "derived_read_only_not_restore_authority",
    }


def chronicle_schema_v1() -> dict[str, Any]:
    reporting = reporting_schema_v1()
    return {
        "schema_version": CHRONICLE_SCHEMA_VERSION,
        "capability_contract_sha256": (GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256),
        "reporting_schema_sha256": content_sha256(reporting),
        "source": (
            "terminal_checkpoint.genome_upgrade_v2_checkpoint_projection."
            "compact_generation_summary"
        ),
        "optional_extension_source": (
            "generation_boundary_checkpoint.genome_upgrade_v2_checkpoint_"
            "projection.compact_generation_summary.recomb72_gv2"
        ),
        "projection_phase": "after_authoritative_settlement",
        "trajectory_influence": "forbidden",
        "rng_access": "forbidden",
        "resident_mutation": "forbidden",
        "retained_fields": [
            "cell_id",
            "generation",
            "tick",
            "sidecar_payload_sha256",
            "summary_sha256",
            "summary",
        ],
    }


def observability_storage_budget_v1() -> dict[str, Any]:
    return {
        "schema_version": "genome_v2_observability_storage_budget_v1",
        "maximum_serialized_bytes_per_cell_generation": (
            OBSERVABILITY_STORAGE_BUDGET_BYTES
        ),
        "selected_event_limit": SELECTED_EVENT_LIMIT,
        "raw_resident_arrays": "forbidden",
        "raw_tick_arrays": "forbidden",
        "raw_genotype_rows": "forbidden",
        "unbounded_event_jsonl": "forbidden",
        "hot_loop_device_to_host_copy": "forbidden",
        "measurement_phase": "terminal_settlement_only",
        "optional_profile_extension_measurement_phase": (
            "generation_boundary_checkpoint_settlement"
        ),
        "maximum_hereditary_locus_summaries": HEREDITARY_LOCUS_COUNT,
    }


def project_genome_upgrade_v2_checkpoint_summary_v1(
    checkpoint_projection: Any,
    *,
    expected_cell_id: str,
) -> dict[str, Any] | None:
    """Copy one bounded terminal summary without touching simulation authority."""

    if not isinstance(checkpoint_projection, dict):
        raise GenomeUpgradeV2Error("checkpoint reporting projection is malformed")
    sidecar = checkpoint_projection.get("genome_upgrade_v2_checkpoint_projection")
    if sidecar is None:
        return None
    if not isinstance(sidecar, dict):
        raise GenomeUpgradeV2Error("Genome V2 reporting sidecar is malformed")
    identity = sidecar.get("identity")
    summary = sidecar.get("compact_generation_summary")
    if not isinstance(identity, dict) or not isinstance(summary, dict):
        raise GenomeUpgradeV2Error("Genome V2 reporting sidecar is incomplete")
    if identity.get("cell_id") != expected_cell_id:
        raise GenomeUpgradeV2Error("Genome V2 reporting sidecar cell identity differs")
    for name in ("generation", "tick"):
        if type(identity.get(name)) is not int or identity[name] < 0:
            raise GenomeUpgradeV2Error(f"Genome V2 reporting sidecar {name} is invalid")
    payload_sha256 = identity.get("payload_sha256")
    if (
        not isinstance(payload_sha256, str)
        or len(payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in payload_sha256)
    ):
        raise GenomeUpgradeV2Error(
            "Genome V2 reporting sidecar payload hash is invalid"
        )
    if (
        summary.get("schema_version") != COMPACT_SUMMARY_SCHEMA_VERSION
        or summary.get("raw_tick_arrays_retained") is not False
        or summary.get("bounded_selected_event_limit") != SELECTED_EVENT_LIMIT
        or summary.get("population_selection") != "authoritative_alive_mask"
    ):
        raise GenomeUpgradeV2Error("Genome V2 compact reporting contract differs")
    cells = summary.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise GenomeUpgradeV2Error(
            "Genome V2 reporting sidecar must contain one physical cell"
        )
    selected = cells[0].get("selected_causal_events")
    required_cell_fields = {
        "cell",
        "population_count",
        "products_q10",
        "phenotypes_q10",
        "saturated_product_count",
        "active_product_count",
        "expression_cost_q10",
        "expression_cost_total_q10",
        "active_phenotype_count",
        "genotype_phenotype_variance_ratio",
        "hereditary_capacity",
        "polygenic_decomposition",
        "contexts",
        "consequences",
        "selected_causal_events",
    }
    optional_cell_fields = {"recomb72_gv2"}
    if (
        not required_cell_fields <= set(cells[0])
        or set(cells[0]) - required_cell_fields - optional_cell_fields
    ):
        raise GenomeUpgradeV2Error("Genome V2 compact reporting cell fields differ")
    if set(cells[0]["products_q10"]) != set(PRODUCT_CHANNELS):
        raise GenomeUpgradeV2Error("Genome V2 product reporting channels differ")
    if set(cells[0]["phenotypes_q10"]) != set(PHENOTYPE_CHANNELS):
        raise GenomeUpgradeV2Error("Genome V2 phenotype reporting channels differ")
    hereditary = cells[0]["hereditary_capacity"]
    if (
        not isinstance(hereditary, Mapping)
        or hereditary.get("schema_version") != HEREDITARY_CAPACITY_SCHEMA_VERSION
        or hereditary.get("locus_summary_count") != HEREDITARY_LOCUS_COUNT
        or not isinstance(hereditary.get("loci"), list)
        or len(hereditary["loci"]) != HEREDITARY_LOCUS_COUNT
        or hereditary.get("raw_genotype_rows_retained") is not False
        or hereditary.get("genotype_identity") != "ordered_homolog_q10_vector_v1"
        or hereditary.get("phenotype_identity") != "exact_phenotype_q10_vector_v1"
    ):
        raise GenomeUpgradeV2Error(
            "Genome V2 hereditary-capacity reporting contract differs"
        )
    if not isinstance(selected, list) or len(selected) > SELECTED_EVENT_LIMIT:
        raise GenomeUpgradeV2Error(
            "Genome V2 selected causal event sample is unbounded"
        )
    recomb72 = cells[0].get("recomb72_gv2")
    if recomb72 is not None:
        _validate_recomb72_longitudinal_summary_v1(
            recomb72,
            founder_profile=sidecar.get("founder_profile"),
        )
    bounded = copy.deepcopy(summary)
    projected = {
        "schema_version": CHRONICLE_SCHEMA_VERSION,
        "cell_id": expected_cell_id,
        "generation": identity["generation"],
        "tick": identity["tick"],
        "sidecar_payload_sha256": payload_sha256,
        "summary_sha256": content_sha256(bounded),
        "summary": bounded,
    }
    serialized = json.dumps(
        projected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(serialized) > OBSERVABILITY_STORAGE_BUDGET_BYTES:
        raise GenomeUpgradeV2Error(
            "Genome V2 compact reporting projection exceeds its storage budget"
        )
    return projected


def project_genome_upgrade_v2_e1_metric_attribution_v1(
    checkpoint_projection: Any,
    *,
    expected_cell_id: str,
) -> dict[str, Any] | None:
    """Reduce one settled sidecar to bounded, reporting-only E1 attribution."""

    projected = project_genome_upgrade_v2_checkpoint_summary_v1(
        checkpoint_projection,
        expected_cell_id=expected_cell_id,
    )
    if projected is None:
        return None
    summary = projected["summary"]
    cell = summary["cells"][0]
    population_count = cell["population_count"]
    contexts = cell["contexts"]
    consequences = cell["consequences"]
    expression_cost = cell["expression_cost_q10"]
    if (
        type(population_count) is not int
        or population_count < 0
        or not isinstance(contexts, Mapping)
        or not isinstance(consequences, Mapping)
        or not isinstance(expression_cost, Mapping)
    ):
        raise GenomeUpgradeV2Error("Genome V2 E1 attribution source is malformed")
    missing_context = contexts.get("missing_count")
    context_available = contexts.get("available")
    consequence_count = consequences.get("observed_count")
    cost_invalid_count = expression_cost.get("invalid_count")
    if (
        type(missing_context) is not int
        or missing_context < 0
        or missing_context > population_count
        or type(context_available) is not bool
        or type(consequence_count) is not int
        or consequence_count < 0
        or type(cost_invalid_count) is not int
        or cost_invalid_count < 0
    ):
        raise GenomeUpgradeV2Error("Genome V2 E1 observation counts are malformed")
    cap_bindings = consequences.get("cap_binding_count_by_surface")
    applied = consequences.get("applied_delta_by_surface")
    if not isinstance(cap_bindings, Mapping) or not isinstance(applied, Mapping):
        raise GenomeUpgradeV2Error("Genome V2 E1 consequence attribution is malformed")
    if any(
        type(value) is not int or value < 0 for value in cap_bindings.values()
    ) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in applied.values()
    ):
        raise GenomeUpgradeV2Error("Genome V2 E1 consequence values are malformed")
    context_observations = (
        population_count - missing_context if context_available else 0
    )
    cost_observations = max(0, population_count - cost_invalid_count)
    metrics = {
        "genome_v2.expression_observation_count": population_count,
        "genome_v2.context_observation_count": context_observations,
        "genome_v2.cost_observation_count": cost_observations,
        "genome_v2.cap_observation_count": consequence_count,
    }
    attribution = {
        "schema_version": E1_METRIC_ATTRIBUTION_SCHEMA_VERSION,
        "cell_id": expected_cell_id,
        "generation": projected["generation"],
        "summary_sha256": projected["summary_sha256"],
        "metrics": metrics,
        "expression_cost_total_q10": cell["expression_cost_total_q10"],
        "cap_binding_count_total": sum(cap_bindings.values()),
        "applied_delta_by_surface": {
            str(name): float(value) for name, value in sorted(applied.items())
        },
        "raw_tick_arrays_retained": False,
        "trajectory_influence": "forbidden",
    }
    serialized = json.dumps(
        attribution,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(serialized) > OBSERVABILITY_STORAGE_BUDGET_BYTES:
        raise GenomeUpgradeV2Error(
            "Genome V2 E1 attribution exceeds its storage budget"
        )
    return attribution


def summarize_genome_upgrade_v2_population(
    genomes: Iterable[GenomeUpgradeV2V1],
    *,
    context_q10: Sequence[int] | None = None,
) -> dict[str, Any]:
    rows = list(genomes)
    context = tuple(context_q10 or (0,) * CONTEXT_COUNT)
    if len(context) != CONTEXT_COUNT:
        raise GenomeUpgradeV2Error("telemetry context has the wrong length")
    expressions = [express_reference_v1(genome, context) for genome in rows]
    count = len(rows)
    allele_parameter_count = count * 2 * len(PRODUCT_CHANNELS) * 4
    heterozygous = sum(
        genome.alleles[0][product][parameter] != genome.alleles[1][product][parameter]
        for genome in rows
        for product in range(len(PRODUCT_CHANNELS))
        for parameter in range(4)
    )
    modifier_count = count * 2 * len(PRODUCT_CHANNELS)
    modifier_heterozygous = sum(
        genome.modifiers[0][product] != genome.modifiers[1][product]
        for genome in rows
        for product in range(len(PRODUCT_CHANNELS))
    )
    hidden_variation = sum(
        any(
            genome.alleles[0][product] != genome.alleles[1][product]
            or genome.modifiers[0][product] != genome.modifiers[1][product]
            for product in range(len(PRODUCT_CHANNELS))
        )
        and all(value == 0 for value in expression.phenotypes_q10)
        for genome, expression in zip(rows, expressions)
    )

    def mean(values: list[int]) -> float | None:
        return sum(values) / len(values) if values else None

    product_means = {
        name: mean([expression.products_q10[index] for expression in expressions])
        for index, name in enumerate(PRODUCT_CHANNELS)
    }
    phenotype_means = {
        name: mean([expression.phenotypes_q10[index] for expression in expressions])
        for index, name in enumerate(PHENOTYPE_CHANNELS)
    }
    pleiotropic_output_counts = {
        name: sum(weights[product_index] != 0 for weights in POLYGENIC_WEIGHTS_Q10)
        for product_index, name in enumerate(PRODUCT_CHANNELS)
    }
    return {
        "schema_version": "genome_upgrade_v2_population_telemetry_v1",
        "population_count": count,
        "allele_parameter_copy_count": allele_parameter_count,
        "allele_parameter_heterozygous_count": heterozygous,
        "allele_parameter_heterozygosity": (
            heterozygous / (allele_parameter_count // 2)
            if allele_parameter_count
            else None
        ),
        "modifier_copy_count": modifier_count,
        "modifier_heterozygous_count": modifier_heterozygous,
        "modifier_heterozygosity": (
            modifier_heterozygous / (modifier_count // 2) if modifier_count else None
        ),
        "hidden_variation_count": hidden_variation,
        "product_means_q10": product_means,
        "phenotype_means_q10": phenotype_means,
        "expression_cost_mean_q10": mean(
            [expression.expression_cost_q10 for expression in expressions]
        ),
        "active_product_count_mean": mean(
            [expression.active_product_count for expression in expressions]
        ),
        "saturated_product_count_mean": mean(
            [expression.saturated_product_count for expression in expressions]
        ),
        "pleiotropic_output_counts": pleiotropic_output_counts,
    }


def _bounded_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    if values.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "invalid_count": 0,
        }
    exact = values.to(dtype=torch.int64)
    floating = exact.to(dtype=torch.float64)
    return {
        "mean": float(floating.mean().item()),
        "std": float(floating.std(correction=0).item()),
        "min": int(exact.min().item()),
        "max": int(exact.max().item()),
        "invalid_count": 0,
    }


def _locus_summary_v1(
    *,
    locus_id: str,
    locus_kind: str,
    values: torch.Tensor,
    neutral_q10: int,
) -> dict[str, Any]:
    if values.ndim != 2 or int(values.shape[0]) != 2:
        raise GenomeUpgradeV2Error(
            "Genome V2 hereditary locus must have two homolog rows"
        )
    active_count = int(values.shape[1])
    copies = values.detach().to(device="cpu", dtype=torch.int64)
    flat = [int(value) for value in copies.reshape(-1).tolist()]
    distinct = sorted(set(flat))
    heterozygous = int((copies[0] != copies[1]).sum().item()) if active_count else 0
    fixed = active_count > 0 and len(distinct) == 1
    non_neutral = [value for value in flat if value != neutral_q10]
    return {
        "locus_id": locus_id,
        "locus_kind": locus_kind,
        "neutral_q10": neutral_q10,
        "observed_agent_count": active_count,
        "observed_copy_count": len(flat),
        "distinct_allele_count": len(distinct),
        "minimum_q10": min(distinct) if distinct else None,
        "maximum_q10": max(distinct) if distinct else None,
        "heterozygous_agent_count": heterozygous,
        "heterozygosity": (heterozygous / active_count if active_count else None),
        "non_neutral_copy_count": len(non_neutral),
        "non_neutral_allele_count": len(set(non_neutral)),
        "non_neutral_variant_retained": bool(non_neutral),
        "variation_retained": len(distinct) > 1,
        "fixed": fixed,
        "neutral_fixed": fixed and distinct[0] == neutral_q10,
        "non_neutral_fixed": fixed and distinct[0] != neutral_q10,
        "fixed_value_q10": distinct[0] if fixed else None,
    }


def _hereditary_capacity_summary_v1(
    *,
    alleles: torch.Tensor,
    modifiers: torch.Tensor,
    phenotypes: torch.Tensor,
) -> dict[str, Any]:
    active_count = int(alleles.shape[-1])
    alleles = alleles.detach().to(device="cpu", dtype=torch.int64)
    modifiers = modifiers.detach().to(device="cpu", dtype=torch.int64)
    phenotypes = phenotypes.detach().to(device="cpu", dtype=torch.int64)
    loci = [
        _locus_summary_v1(
            locus_id=f"allele.{product_name}.{parameter_name}",
            locus_kind="allele_parameter",
            values=alleles[:, product_index, parameter_index, :],
            neutral_q10=(
                0 if parameter_name in {"activity", "sensitivity"} else Q10_SCALE
            ),
        )
        for product_index, product_name in enumerate(PRODUCT_CHANNELS)
        for parameter_index, parameter_name in enumerate(ALLELE_PARAMETERS)
    ]
    loci.extend(
        _locus_summary_v1(
            locus_id=f"modifier.{product_name}",
            locus_kind="modifier",
            values=modifiers[:, product_index, :],
            neutral_q10=0,
        )
        for product_index, product_name in enumerate(PRODUCT_CHANNELS)
    )

    allele_heterozygous = sum(
        row["heterozygous_agent_count"]
        for row in loci
        if row["locus_kind"] == "allele_parameter"
    )
    modifier_heterozygous = sum(
        row["heterozygous_agent_count"]
        for row in loci
        if row["locus_kind"] == "modifier"
    )
    genotype_rows = torch.cat(
        (
            alleles.permute(3, 0, 1, 2).reshape(
                active_count,
                2 * len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS),
            ),
            modifiers.permute(2, 0, 1).reshape(
                active_count,
                2 * len(PRODUCT_CHANNELS),
            ),
        ),
        dim=1,
    )
    phenotype_rows = phenotypes.transpose(0, 1)
    genotypes = [tuple(int(value) for value in row) for row in genotype_rows.tolist()]
    phenotype_values = [
        tuple(int(value) for value in row) for row in phenotype_rows.tolist()
    ]
    phenotype_genotypes: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for genotype, phenotype in zip(genotypes, phenotype_values):
        phenotype_genotypes.setdefault(phenotype, set()).add(genotype)
    equivalent = {
        phenotype: values
        for phenotype, values in phenotype_genotypes.items()
        if len(values) > 1
    }
    equivalent_agents = sum(phenotype in equivalent for phenotype in phenotype_values)
    retained_agents = (
        int(
            (
                (alleles[0] != alleles[1]).reshape(-1, active_count).any(dim=0)
                | (modifiers[0] != modifiers[1]).reshape(-1, active_count).any(dim=0)
            )
            .sum()
            .item()
        )
        if active_count
        else 0
    )
    allele_denominator = active_count * len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS)
    modifier_denominator = active_count * len(PRODUCT_CHANNELS)
    return {
        "schema_version": HEREDITARY_CAPACITY_SCHEMA_VERSION,
        "measurement": ("terminal_alive_population_exact_resident_tensor_reduction"),
        "retention_semantics": ("terminal_presence_only_not_longitudinal_persistence"),
        "genotype_identity": "ordered_homolog_q10_vector_v1",
        "phenotype_identity": "exact_phenotype_q10_vector_v1",
        "population_count": active_count,
        "allele_parameter_locus_count": (
            len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS)
        ),
        "modifier_locus_count": len(PRODUCT_CHANNELS),
        "locus_summary_count": len(loci),
        "allele_parameter_heterozygous_count": allele_heterozygous,
        "allele_parameter_heterozygosity": (
            allele_heterozygous / allele_denominator if allele_denominator else None
        ),
        "modifier_heterozygous_count": modifier_heterozygous,
        "modifier_heterozygosity": (
            modifier_heterozygous / modifier_denominator
            if modifier_denominator
            else None
        ),
        "distinct_genotype_count": len(set(genotypes)),
        "distinct_phenotype_count": len(set(phenotype_values)),
        "hidden_genotype_retained_agent_count": retained_agents,
        "hidden_genotype_retention_fraction": (
            retained_agents / active_count if active_count else None
        ),
        "phenotype_equivalent_genotype_group_count": len(equivalent),
        "phenotype_equivalent_distinct_genotype_count": sum(
            len(values) for values in equivalent.values()
        ),
        "phenotype_equivalent_agent_count": equivalent_agents,
        "raw_genotype_rows_retained": False,
        "loci": loci,
    }


def _recomb72_longitudinal_summary_v1(
    *,
    founder_profile_identity: Mapping[str, Any],
    alleles: torch.Tensor,
    modifiers: torch.Tensor,
    phenotypes: torch.Tensor,
) -> dict[str, Any]:
    from .genome_upgrade_v2_recomb72_founders_v1 import (
        FOUNDER_PROFILE_PREFIX,
    )
    from perf.cuda_genome_upgrade_v2_state_bank import (
        resolve_genome_v2_founder_profile_v1,
    )

    profile_id = founder_profile_identity.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.startswith(
        f"{FOUNDER_PROFILE_PREFIX}."
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 founder profile identity differs")
    admitted_profile = resolve_genome_v2_founder_profile_v1(profile_id)
    if (
        founder_profile_identity.get("payload_sha256")
        != admitted_profile["payload_sha256"]
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 founder profile hash differs")
    profile = admitted_profile.get("founder_population")
    if not isinstance(profile, Mapping):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 founder population contract is absent"
        )

    active_count = int(alleles.shape[-1])
    allele_rows = alleles.detach().to(device="cpu", dtype=torch.int64)
    modifier_rows = modifiers.detach().to(device="cpu", dtype=torch.int64)
    phenotype_rows = phenotypes.detach().to(device="cpu", dtype=torch.int64)
    homolog_rows = torch.cat(
        (
            allele_rows.permute(3, 0, 1, 2).reshape(active_count, 2, -1),
            modifier_rows.permute(2, 0, 1).reshape(active_count, 2, -1),
        ),
        dim=2,
    )
    endpoint_a = torch.tensor(profile["haplotype_a_q10"], dtype=torch.int64)
    endpoint_b = torch.tensor(profile["haplotype_b_q10"], dtype=torch.int64)
    if int(homolog_rows.shape[-1]) != int(endpoint_a.numel()):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 resident haplotype width differs")

    intact_a = (homolog_rows == endpoint_a).all(dim=2)
    intact_b = (homolog_rows == endpoint_b).all(dim=2)
    endpoint_valid = ((homolog_rows == endpoint_a) | (homolog_rows == endpoint_b)).all(
        dim=2
    )
    recombinant = endpoint_valid & ~intact_a & ~intact_b
    neither = ~endpoint_valid

    genotype_masks = {
        "aa": intact_a[:, 0] & intact_a[:, 1],
        "ab": intact_a[:, 0] & intact_b[:, 1],
        "ba": intact_b[:, 0] & intact_a[:, 1],
        "bb": intact_b[:, 0] & intact_b[:, 1],
    }
    genotype_counts = {
        name: int(mask.sum().item()) for name, mask in genotype_masks.items()
    }
    genotype_counts["other"] = active_count - sum(genotype_counts.values())

    partial = ((intact_a | intact_b).sum(dim=1) == 1) & (recombinant.sum(dim=1) == 1)
    recombinant_individual = recombinant[:, 0] & recombinant[:, 1]
    package_counts = {
        "intact_aa": genotype_counts["aa"],
        "intact_ab": genotype_counts["ab"],
        "intact_ba": genotype_counts["ba"],
        "intact_bb": genotype_counts["bb"],
        "partial": int(partial.sum().item()),
        "recombinant": int(recombinant_individual.sum().item()),
    }
    package_counts["neither"] = active_count - sum(package_counts.values())

    all_locus_definitions = [
        {
            "locus_id": f"allele.{product_name}.{parameter_name}",
            "locus_kind": "allele_parameter",
            "flat_haplotype_index": (
                product_index * len(ALLELE_PARAMETERS) + parameter_index
            ),
            "product_index": product_index,
            "parameter_index": parameter_index,
        }
        for product_index, product_name in enumerate(PRODUCT_CHANNELS)
        for parameter_index, parameter_name in enumerate(ALLELE_PARAMETERS)
    ]
    all_locus_definitions.extend(
        {
            "locus_id": f"modifier.{product_name}",
            "locus_kind": "modifier",
            "flat_haplotype_index": (
                len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS) + product_index
            ),
            "product_index": product_index,
            "parameter_index": None,
        }
        for product_index, product_name in enumerate(PRODUCT_CHANNELS)
    )
    all_locus_rows: list[dict[str, Any]] = []
    unexpected_genome_copy_count = 0
    for locus in all_locus_definitions:
        flat_index = int(locus["flat_haplotype_index"])
        values = homolog_rows[:, :, flat_index]
        value_a = int(endpoint_a[flat_index].item())
        value_b = int(endpoint_b[flat_index].item())
        distinct_values, distinct_counts = torch.unique(
            values,
            sorted=True,
            return_counts=True,
        )
        value_counts = [
            {"value_q10": int(value), "copy_count": int(count)}
            for value, count in zip(
                distinct_values.tolist(),
                distinct_counts.tolist(),
                strict=True,
            )
        ]
        expected_values = {value_a, value_b}
        unexpected = sum(
            row["copy_count"]
            for row in value_counts
            if row["value_q10"] not in expected_values
        )
        unexpected_genome_copy_count += unexpected
        all_locus_rows.append(
            {
                **locus,
                "endpoint_a_q10": value_a,
                "endpoint_b_q10": value_b,
                "observed_copy_count": 2 * active_count,
                "distinct_value_count": len(value_counts),
                "q10_value_counts": value_counts,
                "heterozygous_agent_count": int(
                    (values[:, 0] != values[:, 1]).sum().item()
                ),
                "unexpected_copy_count": unexpected,
            }
        )

    audited_loci = list(profile["audited_loci"])
    audited_index_by_id = {
        str(row["locus_id"]): int(row["flat_haplotype_index"])
        for row in all_locus_definitions
    }
    locus_rows: list[dict[str, Any]] = []
    locus_copy_classes: dict[str, torch.Tensor] = {}
    for locus in audited_loci:
        locus_id = str(locus["locus_id"])
        values = homolog_rows[:, :, audited_index_by_id[locus_id]]
        value_a = int(locus["endpoint_a_q10"])
        value_b = int(locus["endpoint_b_q10"])
        classes = torch.full(values.shape, 2, dtype=torch.int64)
        classes[values == value_a] = 0
        classes[values == value_b] = 1
        locus_copy_classes[locus_id] = classes
        a_count = int((classes == 0).sum().item())
        b_count = int((classes == 1).sum().item())
        other_count = int((classes == 2).sum().item())
        copy_count = 2 * active_count
        observed_heterozygous = int(
            (
                (classes[:, 0] != classes[:, 1])
                & (classes[:, 0] < 2)
                & (classes[:, 1] < 2)
            )
            .sum()
            .item()
        )
        b_frequency = b_count / copy_count if copy_count else None
        locus_rows.append(
            {
                **dict(locus),
                "endpoint_copy_counts": {
                    "a": a_count,
                    "b": b_count,
                    "other": other_count,
                },
                "endpoint_copy_count": copy_count,
                "endpoint_b_frequency": b_frequency,
                "observed_heterozygous_count": observed_heterozygous,
                "observed_heterozygosity": (
                    observed_heterozygous / active_count if active_count else None
                ),
                "expected_heterozygosity": (
                    2.0 * b_frequency * (1.0 - b_frequency)
                    if b_frequency is not None
                    else None
                ),
            }
        )

    association_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(audited_loci):
        for right in audited_loci[left_index + 1 :]:
            left_id = str(left["locus_id"])
            right_id = str(right["locus_id"])
            left_classes = locus_copy_classes[left_id]
            right_classes = locus_copy_classes[right_id]
            valid_copies = (left_classes < 2) & (right_classes < 2)
            phased = {
                f"{left_endpoint}{right_endpoint}": int(
                    (
                        valid_copies
                        & (left_classes == left_value)
                        & (right_classes == right_value)
                    )
                    .sum()
                    .item()
                )
                for left_endpoint, left_value in (("a", 0), ("b", 1))
                for right_endpoint, right_value in (("a", 0), ("b", 1))
            }
            valid_individuals = valid_copies.all(dim=1)
            left_dosage = (left_classes == 1).sum(dim=1)
            right_dosage = (right_classes == 1).sum(dim=1)
            dosage = {
                f"{left_value},{right_value}": int(
                    (
                        valid_individuals
                        & (left_dosage == left_value)
                        & (right_dosage == right_value)
                    )
                    .sum()
                    .item()
                )
                for left_value in range(3)
                for right_value in range(3)
            }
            association_rows.append(
                {
                    "left_locus_id": left_id,
                    "right_locus_id": right_id,
                    "same_linkage_block": (
                        left["linkage_block"] == right["linkage_block"]
                    ),
                    "phased_endpoint_copy_counts": phased,
                    "excluded_other_copy_count": int((~valid_copies).sum().item()),
                    "individual_b_dosage_counts": dosage,
                    "dosage_product_moments": {
                        "valid_individual_count": int(valid_individuals.sum().item()),
                        "sum_left_b_dosage": int(
                            left_dosage[valid_individuals].sum().item()
                        ),
                        "sum_right_b_dosage": int(
                            right_dosage[valid_individuals].sum().item()
                        ),
                        "sum_left_b_dosage_squared": int(
                            (
                                left_dosage[valid_individuals]
                                * left_dosage[valid_individuals]
                            )
                            .sum()
                            .item()
                        ),
                        "sum_right_b_dosage_squared": int(
                            (
                                right_dosage[valid_individuals]
                                * right_dosage[valid_individuals]
                            )
                            .sum()
                            .item()
                        ),
                        "sum_b_dosage_product": int(
                            (
                                left_dosage[valid_individuals]
                                * right_dosage[valid_individuals]
                            )
                            .sum()
                            .item()
                        ),
                    },
                    "phased_ld_d_rational": {
                        "defined": bool(sum(phased.values())),
                        "numerator": (
                            phased["bb"] * sum(phased.values())
                            - (phased["ba"] + phased["bb"])
                            * (phased["ab"] + phased["bb"])
                        ),
                        "denominator": sum(phased.values()) ** 2,
                    },
                    "excluded_other_individual_count": int(
                        (~valid_individuals).sum().item()
                    ),
                }
            )

    genotype_distribution: dict[str, int] = {}
    phenotype_distribution: dict[str, int] = {}
    phenotype_vectors: dict[str, list[int]] = {}
    phenotype_genotype_counts: dict[str, dict[str, int]] = {}
    for genotype, phenotype in zip(
        homolog_rows.tolist(),
        phenotype_rows.transpose(0, 1).tolist(),
        strict=True,
    ):
        genome = GenomeUpgradeV2V1(
            alleles=tuple(
                tuple(
                    tuple(
                        genotype[homolog][
                            product * len(ALLELE_PARAMETERS) : (product + 1)
                            * len(ALLELE_PARAMETERS)
                        ]
                    )
                    for product in range(len(PRODUCT_CHANNELS))
                )
                for homolog in range(2)
            ),
            modifiers=tuple(
                tuple(
                    genotype[homolog][len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS) :]
                )
                for homolog in range(2)
            ),
        )
        genotype_sha256 = genome.to_payload()["genotype_sha256"]
        phenotype_sha256 = content_sha256(phenotype)
        genotype_distribution[genotype_sha256] = (
            genotype_distribution.get(genotype_sha256, 0) + 1
        )
        phenotype_distribution[phenotype_sha256] = (
            phenotype_distribution.get(phenotype_sha256, 0) + 1
        )
        phenotype_vectors[phenotype_sha256] = list(phenotype)
        joint = phenotype_genotype_counts.setdefault(phenotype_sha256, {})
        joint[genotype_sha256] = joint.get(genotype_sha256, 0) + 1
    hidden_groups = [
        {
            "phenotype_sha256": phenotype_sha256,
            "phenotype_q10": phenotype_vectors[phenotype_sha256],
            "genotype_sha256_counts": dict(sorted(genotype_counts.items())),
            "genotype_count": len(genotype_counts),
            "agent_count": sum(genotype_counts.values()),
        }
        for phenotype_sha256, genotype_counts in sorted(
            phenotype_genotype_counts.items()
        )
        if len(genotype_counts) > 1
    ]
    novel_endpoint_copy_count = sum(
        row["endpoint_copy_counts"]["other"] for row in locus_rows
    )
    return {
        "schema_version": RECOMB72_LONGITUDINAL_SCHEMA_VERSION,
        "profile_id": profile_id,
        "profile_sha256": admitted_profile["payload_sha256"],
        "founder_population_sha256": profile["payload_sha256"],
        "pattern_id": profile["pattern_id"],
        "state_id": profile["state_id"],
        "population_count": active_count,
        "haplotype_endpoint_sha256": {
            "a": profile["haplotype_a_sha256"],
            "b": profile["haplotype_b_sha256"],
        },
        "ordered_genotype_counts": genotype_counts,
        "homolog_package_counts": {
            "intact_a": int(intact_a.sum().item()),
            "intact_b": int(intact_b.sum().item()),
            "recombinant": int(recombinant.sum().item()),
            "neither": int(neither.sum().item()),
        },
        "individual_package_counts": package_counts,
        "cross_recombinant_observed": bool(recombinant.any().item()),
        "all_loci": all_locus_rows,
        "audited_loci": locus_rows,
        "audited_locus_pair_associations": association_rows,
        "genotype_distribution_sha256_counts": dict(
            sorted(genotype_distribution.items())
        ),
        "phenotype_distribution_sha256_counts": dict(
            sorted(phenotype_distribution.items())
        ),
        "phenotype_genotype_sha256_counts": {
            phenotype_sha256: dict(sorted(genotype_counts.items()))
            for phenotype_sha256, genotype_counts in sorted(
                phenotype_genotype_counts.items()
            )
        },
        "phenotype_distribution": [
            {
                "phenotype_sha256": phenotype_sha256,
                "phenotype_q10": phenotype_vectors[phenotype_sha256],
                "agent_count": count,
            }
            for phenotype_sha256, count in sorted(phenotype_distribution.items())
        ],
        "phenotype_equivalent_hidden_genotype_groups": hidden_groups,
        "novel_endpoint_copy_count": novel_endpoint_copy_count,
        "unexpected_genome_copy_count": unexpected_genome_copy_count,
        "mutation_observation": (
            "no_unexpected_genome_values_observed"
            if unexpected_genome_copy_count == 0
            else "unexpected_genome_values_observed"
        ),
        "changed_loci_counter_available": False,
        "association_formula_contract": {
            "phased_ld_d": (
                "numerator_over_denominator_where_numerator_is_"
                "n_bb*n_valid-n_left_b*n_right_b"
            ),
            "dosage_correlation": (
                "pearson_from_exact_n_sum_x_sum_y_sum_x2_sum_y2_sum_xy"
            ),
        },
        "raw_agent_rows_retained": False,
    }


def _validate_recomb72_longitudinal_summary_v1(
    value: Any,
    *,
    founder_profile: Any,
) -> None:
    from perf.cuda_genome_upgrade_v2_state_bank import (
        resolve_genome_v2_founder_profile_v1,
    )

    required_fields = {
        "schema_version",
        "profile_id",
        "profile_sha256",
        "founder_population_sha256",
        "pattern_id",
        "state_id",
        "population_count",
        "haplotype_endpoint_sha256",
        "ordered_genotype_counts",
        "homolog_package_counts",
        "individual_package_counts",
        "cross_recombinant_observed",
        "all_loci",
        "audited_loci",
        "audited_locus_pair_associations",
        "genotype_distribution_sha256_counts",
        "phenotype_distribution_sha256_counts",
        "phenotype_genotype_sha256_counts",
        "phenotype_distribution",
        "phenotype_equivalent_hidden_genotype_groups",
        "novel_endpoint_copy_count",
        "unexpected_genome_copy_count",
        "mutation_observation",
        "changed_loci_counter_available",
        "association_formula_contract",
        "raw_agent_rows_retained",
    }
    if not isinstance(value, Mapping) or set(value) != required_fields:
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 longitudinal reporting fields differ")
    if (
        value.get("schema_version") != RECOMB72_LONGITUDINAL_SCHEMA_VERSION
        or not isinstance(founder_profile, Mapping)
        or set(founder_profile) != {"schema_version", "profile_id", "payload_sha256"}
        or value.get("profile_id") != founder_profile.get("profile_id")
        or value.get("profile_sha256") != founder_profile.get("payload_sha256")
        or value.get("raw_agent_rows_retained") is not False
        or value.get("changed_loci_counter_available") is not False
    ):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 longitudinal reporting contract differs"
        )
    admitted = resolve_genome_v2_founder_profile_v1(str(founder_profile["profile_id"]))
    profile = admitted.get("founder_population")
    if (
        admitted["payload_sha256"] != value["profile_sha256"]
        or not isinstance(profile, Mapping)
        or profile.get("payload_sha256") != value["founder_population_sha256"]
        or profile.get("pattern_id") != value["pattern_id"]
        or profile.get("state_id") != value["state_id"]
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 nested founder authority differs")
    if value.get("haplotype_endpoint_sha256") != {
        "a": profile["haplotype_a_sha256"],
        "b": profile["haplotype_b_sha256"],
    }:
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 haplotype endpoint identity differs")
    population_count = value.get("population_count")
    if type(population_count) is not int or population_count < 0:
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 population count is invalid")

    def counted_partition(
        payload: Any,
        *,
        fields: set[str],
        total: int,
        name: str,
    ) -> Mapping[str, int]:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != fields
            or any(type(item) is not int or item < 0 for item in payload.values())
            or sum(payload.values()) != total
        ):
            raise GenomeUpgradeV2Error(f"RECOMB-72-GV2 {name} partition differs")
        return payload

    ordered_genotypes = counted_partition(
        value["ordered_genotype_counts"],
        fields={"aa", "ab", "ba", "bb", "other"},
        total=population_count,
        name="ordered genotype",
    )
    homolog_packages = counted_partition(
        value["homolog_package_counts"],
        fields={"intact_a", "intact_b", "recombinant", "neither"},
        total=2 * population_count,
        name="homolog package",
    )
    individual_packages = counted_partition(
        value["individual_package_counts"],
        fields={
            "intact_aa",
            "intact_ab",
            "intact_ba",
            "intact_bb",
            "partial",
            "recombinant",
            "neither",
        },
        total=population_count,
        name="individual package",
    )
    if any(
        ordered_genotypes[name] != individual_packages[f"intact_{name}"]
        for name in ("aa", "ab", "ba", "bb")
    ):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 intact genotype/package counts differ"
        )
    if value["cross_recombinant_observed"] is not (homolog_packages["recombinant"] > 0):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 recombinant indicator differs")

    endpoint_a = list(profile["haplotype_a_q10"])
    endpoint_b = list(profile["haplotype_b_q10"])
    expected_all_loci = [
        {
            "locus_id": f"allele.{product_name}.{parameter_name}",
            "locus_kind": "allele_parameter",
            "flat_haplotype_index": (
                product_index * len(ALLELE_PARAMETERS) + parameter_index
            ),
            "product_index": product_index,
            "parameter_index": parameter_index,
        }
        for product_index, product_name in enumerate(PRODUCT_CHANNELS)
        for parameter_index, parameter_name in enumerate(ALLELE_PARAMETERS)
    ]
    expected_all_loci.extend(
        {
            "locus_id": f"modifier.{product_name}",
            "locus_kind": "modifier",
            "flat_haplotype_index": (
                len(PRODUCT_CHANNELS) * len(ALLELE_PARAMETERS) + product_index
            ),
            "product_index": product_index,
            "parameter_index": None,
        }
        for product_index, product_name in enumerate(PRODUCT_CHANNELS)
    )
    expected_all_loci = [
        {
            **row,
            "endpoint_a_q10": endpoint_a[row["flat_haplotype_index"]],
            "endpoint_b_q10": endpoint_b[row["flat_haplotype_index"]],
        }
        for row in expected_all_loci
    ]
    all_loci = value["all_loci"]
    if (
        not isinstance(all_loci, list)
        or any(not isinstance(row, Mapping) for row in all_loci)
        or [row.get("locus_id") for row in all_loci]
        != [row["locus_id"] for row in expected_all_loci]
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 all-locus identity or order differs")
    unexpected_total = 0
    all_locus_fields = {
        "locus_id",
        "locus_kind",
        "flat_haplotype_index",
        "product_index",
        "parameter_index",
        "endpoint_a_q10",
        "endpoint_b_q10",
        "observed_copy_count",
        "distinct_value_count",
        "q10_value_counts",
        "heterozygous_agent_count",
        "unexpected_copy_count",
    }
    for row, expected in zip(all_loci, expected_all_loci, strict=True):
        counts = row.get("q10_value_counts")
        if (
            set(row) != all_locus_fields
            or any(row.get(name) != expected[name] for name in expected)
            or not isinstance(counts, list)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"value_q10", "copy_count"}
                or type(item["value_q10"]) is not int
                or type(item["copy_count"]) is not int
                or item["copy_count"] <= 0
                for item in counts
            )
            or [
                item["value_q10"]
                for item in counts
                if isinstance(item, Mapping) and "value_q10" in item
            ]
            != sorted(
                {
                    item["value_q10"]
                    for item in counts
                    if isinstance(item, Mapping) and "value_q10" in item
                }
            )
            or row.get("observed_copy_count") != 2 * population_count
            or sum(item["copy_count"] for item in counts) != 2 * population_count
            or row.get("distinct_value_count") != len(counts)
            or type(row.get("heterozygous_agent_count")) is not int
            or not 0 <= row["heterozygous_agent_count"] <= population_count
        ):
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 all-locus counts differ")
        expected_values = {expected["endpoint_a_q10"], expected["endpoint_b_q10"]}
        unexpected = sum(
            item["copy_count"]
            for item in counts
            if item["value_q10"] not in expected_values
        )
        if row.get("unexpected_copy_count") != unexpected:
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 unexpected locus count differs")
        unexpected_total += unexpected
    if value["unexpected_genome_copy_count"] != unexpected_total:
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 unexpected genome count differs")

    audited = value["audited_loci"]
    expected_audited = list(profile["audited_loci"])
    if (
        not isinstance(audited, list)
        or any(not isinstance(row, Mapping) for row in audited)
        or [row.get("locus_id") for row in audited]
        != [row["locus_id"] for row in expected_audited]
    ):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 audited-locus identity or order differs"
        )
    audited_other_total = 0
    for row, expected in zip(audited, expected_audited, strict=True):
        if any(row.get(name) != expected[name] for name in expected):
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 audited-locus authority differs")
        counts = counted_partition(
            row.get("endpoint_copy_counts"),
            fields={"a", "b", "other"},
            total=2 * population_count,
            name="audited endpoint",
        )
        if (
            row.get("endpoint_copy_count") != 2 * population_count
            or type(row.get("observed_heterozygous_count")) is not int
            or not 0 <= row["observed_heterozygous_count"] <= population_count
            or row.get("endpoint_b_frequency")
            != (counts["b"] / (2 * population_count) if population_count else None)
            or row.get("observed_heterozygosity")
            != (
                row["observed_heterozygous_count"] / population_count
                if population_count
                else None
            )
            or row.get("expected_heterozygosity")
            != (
                2.0 * row["endpoint_b_frequency"] * (1.0 - row["endpoint_b_frequency"])
                if population_count
                else None
            )
        ):
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 audited-locus counts differ")
        audited_other_total += counts["other"]
    if value["novel_endpoint_copy_count"] != audited_other_total:
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 novel endpoint count differs")

    associations = value["audited_locus_pair_associations"]
    expected_pairs = [
        (left["locus_id"], right["locus_id"])
        for index, left in enumerate(expected_audited)
        for right in expected_audited[index + 1 :]
    ]
    if (
        not isinstance(associations, list)
        or any(not isinstance(row, Mapping) for row in associations)
        or [
            (row.get("left_locus_id"), row.get("right_locus_id"))
            for row in associations
        ]
        != expected_pairs
    ):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 association identity or order differs"
        )
    for row in associations:
        expected_association_fields = {
            "left_locus_id",
            "right_locus_id",
            "same_linkage_block",
            "phased_endpoint_copy_counts",
            "excluded_other_copy_count",
            "individual_b_dosage_counts",
            "dosage_product_moments",
            "phased_ld_d_rational",
            "excluded_other_individual_count",
        }
        phased = row.get("phased_endpoint_copy_counts")
        dosage = row.get("individual_b_dosage_counts")
        if (
            set(row) != expected_association_fields
            or not isinstance(phased, Mapping)
            or set(phased) != {"aa", "ab", "ba", "bb"}
            or any(type(item) is not int or item < 0 for item in phased.values())
            or type(row.get("excluded_other_copy_count")) is not int
            or row["excluded_other_copy_count"] < 0
            or sum(phased.values()) + row["excluded_other_copy_count"]
            != 2 * population_count
            or not isinstance(dosage, Mapping)
            or set(dosage)
            != {f"{left},{right}" for left in range(3) for right in range(3)}
            or any(type(item) is not int or item < 0 for item in dosage.values())
            or type(row.get("excluded_other_individual_count")) is not int
            or row["excluded_other_individual_count"] < 0
            or sum(dosage.values()) + row["excluded_other_individual_count"]
            != population_count
        ):
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 association counts differ")
        left_locus = next(
            item
            for item in expected_audited
            if item["locus_id"] == row["left_locus_id"]
        )
        right_locus = next(
            item
            for item in expected_audited
            if item["locus_id"] == row["right_locus_id"]
        )
        if row["same_linkage_block"] is not (
            left_locus["linkage_block"] == right_locus["linkage_block"]
        ):
            raise GenomeUpgradeV2Error(
                "RECOMB-72-GV2 association linkage identity differs"
            )
        dosage_moments = {
            "valid_individual_count": sum(dosage.values()),
            "sum_left_b_dosage": sum(
                int(key.split(",", maxsplit=1)[0]) * count
                for key, count in dosage.items()
            ),
            "sum_right_b_dosage": sum(
                int(key.split(",", maxsplit=1)[1]) * count
                for key, count in dosage.items()
            ),
            "sum_left_b_dosage_squared": sum(
                int(key.split(",", maxsplit=1)[0]) ** 2 * count
                for key, count in dosage.items()
            ),
            "sum_right_b_dosage_squared": sum(
                int(key.split(",", maxsplit=1)[1]) ** 2 * count
                for key, count in dosage.items()
            ),
            "sum_b_dosage_product": sum(
                int(key.split(",", maxsplit=1)[0])
                * int(key.split(",", maxsplit=1)[1])
                * count
                for key, count in dosage.items()
            ),
        }
        if row.get("dosage_product_moments") != dosage_moments:
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 dosage moments differ")
        valid = sum(phased.values())
        expected_ld = {
            "defined": valid > 0,
            "numerator": (
                phased["bb"] * valid
                - (phased["ba"] + phased["bb"]) * (phased["ab"] + phased["bb"])
            ),
            "denominator": valid**2,
        }
        if row.get("phased_ld_d_rational") != expected_ld:
            raise GenomeUpgradeV2Error("RECOMB-72-GV2 phased LD derivation differs")

    def digest_counts(payload: Any, *, name: str) -> Mapping[str, int]:
        if (
            not isinstance(payload, Mapping)
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(count) is not int
                or count <= 0
                for digest, count in payload.items()
            )
            or sum(payload.values()) != population_count
        ):
            raise GenomeUpgradeV2Error(f"RECOMB-72-GV2 {name} distribution differs")
        return payload

    genotype_counts = digest_counts(
        value["genotype_distribution_sha256_counts"],
        name="genotype",
    )
    phenotype_counts = digest_counts(
        value["phenotype_distribution_sha256_counts"],
        name="phenotype",
    )
    phenotype_rows = value["phenotype_distribution"]
    if (
        not isinstance(phenotype_rows, list)
        or any(not isinstance(row, Mapping) for row in phenotype_rows)
        or len(phenotype_rows) != len(phenotype_counts)
        or {row.get("phenotype_sha256") for row in phenotype_rows}
        != set(phenotype_counts)
        or any(
            set(row) != {"phenotype_sha256", "phenotype_q10", "agent_count"}
            or not isinstance(row.get("phenotype_q10"), list)
            or len(row["phenotype_q10"]) != len(PHENOTYPE_CHANNELS)
            or any(type(item) is not int for item in row["phenotype_q10"])
            or row.get("agent_count") != phenotype_counts[row["phenotype_sha256"]]
            or content_sha256(row.get("phenotype_q10")) != row["phenotype_sha256"]
            for row in phenotype_rows
        )
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 exact phenotype distribution differs")
    phenotype_by_digest = {
        row["phenotype_sha256"]: row["phenotype_q10"] for row in phenotype_rows
    }
    joint_counts = value["phenotype_genotype_sha256_counts"]
    if (
        not isinstance(joint_counts, Mapping)
        or set(joint_counts) != set(phenotype_counts)
        or any(not isinstance(counts, Mapping) for counts in joint_counts.values())
    ):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 phenotype-genotype distribution differs"
        )
    aggregate_genotypes = {digest: 0 for digest in genotype_counts}
    for phenotype_sha256, counts in joint_counts.items():
        if (
            not counts
            or any(
                digest not in genotype_counts
                or type(count) is not int
                or count <= 0
                for digest, count in counts.items()
            )
            or sum(counts.values()) != phenotype_counts[phenotype_sha256]
        ):
            raise GenomeUpgradeV2Error(
                "RECOMB-72-GV2 phenotype-genotype counts differ"
            )
        for digest, count in counts.items():
            aggregate_genotypes[digest] += count
    if aggregate_genotypes != dict(genotype_counts):
        raise GenomeUpgradeV2Error(
            "RECOMB-72-GV2 phenotype-genotype conservation differs"
        )
    hidden_groups = value["phenotype_equivalent_hidden_genotype_groups"]
    if not isinstance(hidden_groups, list) or any(
        not isinstance(row, Mapping) for row in hidden_groups
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 hidden-genotype groups are malformed")
    expected_hidden_groups = [
        {
            "phenotype_sha256": phenotype_sha256,
            "phenotype_q10": phenotype_by_digest[phenotype_sha256],
            "genotype_sha256_counts": dict(sorted(counts.items())),
            "genotype_count": len(counts),
            "agent_count": sum(counts.values()),
        }
        for phenotype_sha256, counts in sorted(joint_counts.items())
        if len(counts) > 1
    ]
    if hidden_groups != expected_hidden_groups or any(
        set(row)
        != {
                "phenotype_sha256",
                "phenotype_q10",
                "genotype_sha256_counts",
                "genotype_count",
                "agent_count",
            }
        for row in hidden_groups
    ):
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 hidden-genotype group differs")

    expected_mutation_observation = (
        "no_unexpected_genome_values_observed"
        if unexpected_total == 0
        else "unexpected_genome_values_observed"
    )
    if value["mutation_observation"] != expected_mutation_observation or value[
        "association_formula_contract"
    ] != {
        "phased_ld_d": (
            "numerator_over_denominator_where_numerator_is_"
            "n_bb*n_valid-n_left_b*n_right_b"
        ),
        "dosage_correlation": ("pearson_from_exact_n_sum_x_sum_y_sum_x2_sum_y2_sum_xy"),
    }:
        raise GenomeUpgradeV2Error("RECOMB-72-GV2 analysis formula contract differs")


def summarize_genome_upgrade_v2_resident_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    alive_mask: torch.Tensor,
    context_q10: torch.Tensor | None = None,
    consequence_rows: Sequence[dict[str, Any]] = (),
    selected_events: Sequence[dict[str, Any]] = (),
    selected_event_limit: int = SELECTED_EVENT_LIMIT,
    population_selection: str = "authoritative_alive_mask",
    founder_profiles: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reduce settled resident tensors without retaining per-tick raw arrays."""

    if selected_event_limit < 0 or selected_event_limit > 64:
        raise GenomeUpgradeV2Error("selected event limit must be in [0, 64]")
    required = {
        "genome_v2_alleles_q10",
        "genome_v2_modifiers_q10",
        "genome_v2_products_q10",
        "genome_v2_phenotypes_q10",
        "genome_v2_expression_cost_q10",
        "genome_v2_active_product_count",
        "genome_v2_saturated_product_count",
    }
    if set(tensors) != required:
        raise GenomeUpgradeV2Error("resident telemetry tensor set differs")
    alleles = tensors["genome_v2_alleles_q10"]
    cells = int(alleles.shape[0])
    capacity = int(alleles.shape[-1])
    if alive_mask.dtype != torch.bool or tuple(alive_mask.shape) != (
        cells,
        capacity,
    ):
        raise GenomeUpgradeV2Error("resident telemetry alive mask differs")
    if alive_mask.device != alleles.device:
        raise GenomeUpgradeV2Error("resident telemetry alive mask device differs")
    if context_q10 is not None and (
        context_q10.dtype != torch.int64
        or context_q10.device != alleles.device
        or tuple(context_q10.shape) != (cells, capacity, CONTEXT_COUNT)
    ):
        raise GenomeUpgradeV2Error("resident telemetry context tensor differs")
    if founder_profiles is not None and (
        len(founder_profiles) != cells
        or any(not isinstance(profile, Mapping) for profile in founder_profiles)
    ):
        raise GenomeUpgradeV2Error(
            "resident telemetry founder profile cardinality differs"
        )

    rows: list[dict[str, Any]] = []
    for cell in range(cells):
        active = alive_mask[cell]
        active_count = int(active.sum().item())
        product_rows = tensors["genome_v2_products_q10"][cell, :, active]
        phenotype_rows = tensors["genome_v2_phenotypes_q10"][cell, :, active]
        allele_rows = alleles[cell, ..., active]
        modifier_rows = tensors["genome_v2_modifiers_q10"][cell, ..., active]
        product_stats = {
            name: {
                **_bounded_stats(product_rows[index]),
                "saturation_frequency": (
                    float(
                        (product_rows[index] == 1024)
                        .to(dtype=torch.float64)
                        .mean()
                        .item()
                    )
                    if active_count
                    else None
                ),
            }
            for index, name in enumerate(PRODUCT_CHANNELS)
        }
        phenotype_stats = {
            name: {
                **_bounded_stats(phenotype_rows[index]),
                "saturation_frequency": (
                    float(
                        (phenotype_rows[index].abs() == 1024)
                        .to(dtype=torch.float64)
                        .mean()
                        .item()
                    )
                    if active_count
                    else None
                ),
            }
            for index, name in enumerate(PHENOTYPE_CHANNELS)
        }
        product_means_q10 = [
            (
                (abs(int(product_rows[index].sum().item())) // active_count)
                * (-1 if int(product_rows[index].sum().item()) < 0 else 1)
                if active_count
                else 0
            )
            for index in range(len(PRODUCT_CHANNELS))
        ]
        hereditary_values = torch.cat(
            (
                alleles[cell, ..., active].reshape(-1),
                tensors["genome_v2_modifiers_q10"][cell, ..., active].reshape(-1),
            )
        ).to(dtype=torch.float64)
        phenotype_values = phenotype_rows.reshape(-1).to(dtype=torch.float64)
        hereditary_variance = (
            float(hereditary_values.var(correction=0).item())
            if hereditary_values.numel()
            else 0.0
        )
        phenotype_variance = (
            float(phenotype_values.var(correction=0).item())
            if phenotype_values.numel()
            else 0.0
        )
        context_summary: dict[str, Any] = {
            "available": context_q10 is not None,
            "missing_count": 0 if context_q10 is not None else active_count,
            "invalid_count": 0,
            "channels": {},
        }
        if context_q10 is not None:
            context_rows = context_q10[cell, active, :]
            context_summary["channels"] = {
                str(index): {
                    **_bounded_stats(context_rows[:, index]),
                    "exposure_frequency": (
                        float(
                            (context_rows[:, index] != 0)
                            .to(dtype=torch.float64)
                            .mean()
                            .item()
                        )
                        if active_count
                        else None
                    ),
                }
                for index in range(CONTEXT_COUNT)
            }
        cell_consequences = [
            row for row in consequence_rows if int(row.get("cell", -1)) == cell
        ]
        consequence_totals: dict[str, float] = {}
        cap_bindings: dict[str, int] = {}
        for consequence in cell_consequences:
            surface = str(consequence["surface"])
            consequence_totals[surface] = consequence_totals.get(surface, 0.0) + float(
                consequence["applied_delta"]
            )
            cap_bindings[surface] = cap_bindings.get(surface, 0) + int(
                bool(consequence.get("cap_bound", False))
            )
        cell_events = [
            {name: value for name, value in row.items() if name != "cell"}
            for row in selected_events
            if int(row.get("cell", -1)) == cell
        ][:selected_event_limit]
        recomb72_profile = None
        if founder_profiles is not None:
            profile_id = founder_profiles[cell].get("profile_id")
            if isinstance(profile_id, str) and profile_id.startswith(
                "recomb72_gv2_v1."
            ):
                recomb72_profile = founder_profiles[cell]
        hereditary_alleles = allele_rows
        hereditary_modifiers = modifier_rows
        hereditary_phenotypes = phenotype_rows
        if recomb72_profile is not None:
            hereditary_alleles = allele_rows.detach().to(
                device="cpu", dtype=torch.int64
            )
            hereditary_modifiers = modifier_rows.detach().to(
                device="cpu", dtype=torch.int64
            )
            hereditary_phenotypes = phenotype_rows.detach().to(
                device="cpu", dtype=torch.int64
            )
        row = {
            "cell": cell,
            "population_count": active_count,
            "products_q10": product_stats,
            "phenotypes_q10": phenotype_stats,
            "saturated_product_count": _bounded_stats(
                tensors["genome_v2_saturated_product_count"][cell, active]
            ),
            "active_product_count": _bounded_stats(
                tensors["genome_v2_active_product_count"][cell, active]
            ),
            "expression_cost_q10": _bounded_stats(
                tensors["genome_v2_expression_cost_q10"][cell, active]
            ),
            "expression_cost_total_q10": int(
                tensors["genome_v2_expression_cost_q10"][cell, active]
                .to(dtype=torch.int64)
                .sum()
                .item()
            ),
            "active_phenotype_count": int((phenotype_rows != 0).sum().item()),
            "genotype_phenotype_variance_ratio": (
                hereditary_variance / phenotype_variance
                if phenotype_variance > 0.0
                else None
            ),
            "hereditary_capacity": _hereditary_capacity_summary_v1(
                alleles=hereditary_alleles,
                modifiers=hereditary_modifiers,
                phenotypes=hereditary_phenotypes,
            ),
            "polygenic_decomposition": polygenic_decomposition_v1(product_means_q10),
            "contexts": context_summary,
            "consequences": {
                "observed_count": len(cell_consequences),
                "applied_delta_by_surface": consequence_totals,
                "cap_binding_count_by_surface": cap_bindings,
                "total_applied_delta": sum(consequence_totals.values()),
            },
            "selected_causal_events": cell_events,
        }
        if recomb72_profile is not None:
            row["recomb72_gv2"] = _recomb72_longitudinal_summary_v1(
                founder_profile_identity=recomb72_profile,
                alleles=hereditary_alleles,
                modifiers=hereditary_modifiers,
                phenotypes=hereditary_phenotypes,
            )
        rows.append(row)
    return {
        "schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
        "bounded_selected_event_limit": selected_event_limit,
        "raw_tick_arrays_retained": False,
        "population_selection": population_selection,
        "cells": rows,
    }


__all__ = [
    "CHRONICLE_SCHEMA_VERSION",
    "COMPACT_SUMMARY_SCHEMA_VERSION",
    "E1_METRIC_ATTRIBUTION_SCHEMA_VERSION",
    "HEREDITARY_CAPACITY_SCHEMA_VERSION",
    "HEREDITARY_LOCUS_COUNT",
    "OBSERVABILITY_STORAGE_BUDGET_BYTES",
    "RECOMB72_LONGITUDINAL_SCHEMA_VERSION",
    "REPORTING_SCHEMA_VERSION",
    "SELECTED_EVENT_LIMIT",
    "chronicle_schema_v1",
    "observability_storage_budget_v1",
    "project_genome_upgrade_v2_checkpoint_summary_v1",
    "project_genome_upgrade_v2_e1_metric_attribution_v1",
    "reporting_schema_v1",
    "summarize_genome_upgrade_v2_population",
    "summarize_genome_upgrade_v2_resident_tensors",
]
