from __future__ import annotations

"""Emit the reviewable GENOME-UPGRADE-V2 mutable-state authority matrices."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evolution.genome_upgrade_v2 import CONTRACT_SHA256, contract_payload
from evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
    GenomeV2P10CapabilityContract,
)
from evolution.genome_upgrade_v2_package_v1 import (
    PRE_ROUTE_PACKAGE_OBLIGATIONS,
    executable_package_contract_v1,
    invalidation_rules_v1,
    obligation_dictionary_v1,
)
from evolution.genome_upgrade_v2_build_admission_v1 import (
    build_admission_policy_v1,
)
from evolution.genome_upgrade_v2_telemetry import (
    chronicle_schema_v1,
    observability_storage_budget_v1,
    reporting_schema_v1,
)
from evolution.genome_upgrade_v2_promotion_contracts_v1 import (
    promotion_contract_artifacts_v1,
    promotion_matrix_rows_v1,
)
from engine.authoritative_state.schema import content_sha256


SCHEMA_VERSION = "genome_v2_authoritative_state_inventory_v1"
EXECUTABLE_CONTRACT_SCHEMA_VERSION = (
    "genome_upgrade_v2_executable_contract_v1"
)
CUT_POINTS = (
    "pre_tick",
    "generation_terminal",
    "cancellation_boundary",
    "sticky_failure_boundary",
)
TRANSACTION_ONLY_CUT_POINTS = (
    "pre_ordinary_birth_commit",
    "post_ordinary_birth_commit",
    "pre_turnover_commit",
    "post_turnover_commit",
)


def _state(
    name: str,
    *,
    category: str,
    owner: str,
    type_shape: str,
    mutation_point: str,
    reset_rule: str,
    checkpoint_inclusion: str,
    restore_rule: str,
    rollback_rule: str,
    equality_rule: str,
    route_support: str,
    evidence_field: str,
    legal_cut_points: tuple[str, ...] = CUT_POINTS,
) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "owner": owner,
        "type_shape": type_shape,
        "mutation_point": mutation_point,
        "reset_rule": reset_rule,
        "checkpoint_inclusion": checkpoint_inclusion,
        "restore_rule": restore_rule,
        "rollback_rule": rollback_rule,
        "equality_rule": equality_rule,
        "route_support": route_support,
        "evidence_field": evidence_field,
        "legal_cut_points": list(legal_cut_points),
    }


def inventory_rows() -> list[dict[str, Any]]:
    resident_owner = "CudaGenomeUpgradeV2StateBank"
    exact_sidecar = "required in identified per-cell genome-v2 sidecar"
    exact_restore = "fresh fixed-shape tensor; exact dtype, shape, and value"
    shadow_rollback = "resident shadow restored before failure becomes sticky"
    cuda_only = "unified native sm75 production; Python/CPU reference only"
    rows = [
        _state(
            "genome_v2_enabled",
            category="configuration",
            owner="Authoritative State Schema ordinary_birth_genome_v2_139",
            type_shape="bool[cells]",
            mutation_point="source allocation only",
            reset_rule="fresh configuration",
            checkpoint_inclusion="bound by base projection and sidecar identity",
            restore_rule="must equal admitted configuration",
            rollback_rule="immutable during run",
            equality_rule="bit exact",
            route_support=cuda_only,
            evidence_field="identity.genome_v2_enabled",
        ),
        _state(
            "genome_v2_mutation_parameters",
            category="configuration",
            owner="Authoritative State Schema ordinary_birth_genome_v2_139",
            type_shape="int64[cells,2] rate_q10,step_q10",
            mutation_point="source allocation only",
            reset_rule="fresh configuration",
            checkpoint_inclusion="bound by canonical base projection",
            restore_rule="must equal admitted configuration and [0,1024] bounds",
            rollback_rule="immutable during run",
            equality_rule="integer exact",
            route_support=cuda_only,
            evidence_field="identity.mutation_contract",
        ),
    ]
    for name, shape, mutation in (
        (
            "genome_v2_alleles_q10",
            "int64[cells,2,12,4,capacity]",
            "ordinary-birth or turnover atomic commit",
        ),
        (
            "genome_v2_modifiers_q10",
            "int64[cells,2,12,capacity]",
            "ordinary-birth or turnover atomic commit",
        ),
        (
            "genome_v2_products_q10",
            "int64[cells,12,capacity]",
            "per-tick expression and newborn/turnover commit",
        ),
        (
            "genome_v2_phenotypes_q10",
            "int64[cells,10,capacity]",
            "per-tick expression and newborn/turnover commit",
        ),
        (
            "genome_v2_expression_cost_q10",
            "int64[cells,capacity]",
            "per-tick expression and newborn/turnover commit",
        ),
        (
            "genome_v2_active_product_count",
            "int64[cells,capacity]",
            "per-tick expression and newborn/turnover commit",
        ),
        (
            "genome_v2_saturated_product_count",
            "int64[cells,capacity]",
            "per-tick expression and newborn/turnover commit",
        ),
    ):
        rows.append(
            _state(
                name,
                category=(
                    "hereditary"
                    if "alleles" in name or "modifiers" in name
                    else "expression_cache"
                ),
                owner=resident_owner,
                type_shape=shape,
                mutation_point=mutation,
                reset_rule="neutral fixed-envelope initialization; dead slot sanitized",
                checkpoint_inclusion=exact_sidecar,
                restore_rule=exact_restore,
                rollback_rule=shadow_rollback,
                equality_rule="integer and byte exact",
                route_support=cuda_only,
                evidence_field=f"sidecar.tensors.{name}",
            )
        )
    rows.extend(
        [
            _state(
                "genome_v2_context_q10",
                category="ephemeral_context",
                owner="flattened_controller_tick_core_and_birth_plan_kernel_v1",
                type_shape="int64[8] thread-local snapshot per live slot",
                mutation_point="rebuilt once before observation/policy each tick",
                reset_rule="stack lifetime; neutral context for newborn commit",
                checkpoint_inclusion="not stored; legal checkpoints are before next sample",
                restore_rule="recomputed exactly from restored authoritative world state",
                rollback_rule="discarded with uncommitted kernel work",
                equality_rule="integer exact against reference normalizer",
                route_support=cuda_only,
                evidence_field="probe.context_q10",
            ),
            _state(
                "genome_v2_runtime_projection",
                category="ephemeral_projection",
                owner="genome_upgrade_v2_q10_v1",
                type_shape="RuntimeProjectionV1 thread-local per live slot",
                mutation_point="after expression cache update each tick",
                reset_rule="stack lifetime",
                checkpoint_inclusion="not stored; deterministic from checkpointed caches",
                restore_rule="recomputed exactly before consequence consumption",
                rollback_rule="discarded with uncommitted kernel work",
                equality_rule="integer exact; fixed float consequence formula",
                route_support=cuda_only,
                evidence_field="probe.runtime_projection",
            ),
            _state(
                "genome_v2_regulatory_projection",
                category="ephemeral_regulation",
                owner="genome_upgrade_v2_fixed_regulatory_network_v1",
                type_shape=(
                    "int64[4] regulator nodes plus int64[12] epistatic "
                    "deltas and bounded cost counters per live slot"
                ),
                mutation_point=(
                    "recomputed once before phenotype projection each tick"
                ),
                reset_rule=(
                    "stack lifetime; fixed topology has no mutable hidden state"
                ),
                checkpoint_inclusion=(
                    "inputs and regulated product outputs stored; ephemeral "
                    "nodes exactly recomputed"
                ),
                restore_rule=(
                    "deterministic recomputation from genotype and context"
                ),
                rollback_rule="discarded with uncommitted kernel work",
                equality_rule="integer exact across reference/native/CUDA",
                route_support=cuda_only,
                evidence_field="probe.regulatory_projection",
            ),
            _state(
                "genome_v2_staged_children",
                category="transaction_scratch",
                owner="FlatBirthScratchArenaV2",
                type_shape="StagedChildV1[cells,max_births_per_tick]",
                mutation_point="birth prepare after canonical parent ordering",
                reset_rule="overwritten on every prepare",
                checkpoint_inclusion="absent at legal checkpoint; transactions are atomic",
                restore_rule="never reconstructed from partial staging",
                rollback_rule="discard scratch; resident arrays remain unchanged",
                equality_rule="genotype, semantic draws, changed loci exact",
                route_support=cuda_only,
                evidence_field="birth_receipt.staged_child_digest",
                legal_cut_points=TRANSACTION_ONLY_CUT_POINTS,
            ),
            _state(
                "genome_v2_turnover_staging_and_shadows",
                category="transaction_scratch",
                owner="flattened_turnover_v1 and resident source allocator",
                type_shape="fixed tensors matching all seven resident planes",
                mutation_point="turnover stage; shadow before commit",
                reset_rule="overwritten per turnover transaction",
                checkpoint_inclusion="absent at legal checkpoint; no partial turnover",
                restore_rule="resident tensors restored, scratch freshly allocated",
                rollback_rule="all seven resident planes restored together",
                equality_rule="tensor byte exact",
                route_support=cuda_only,
                evidence_field="turnover_receipt.genome_v2_shadow_equal",
                legal_cut_points=TRANSACTION_ONLY_CUT_POINTS,
            ),
            _state(
                "canonical_hereditary_digest",
                category="identity",
                owner="CudaDiploidStateBankV1",
                type_shape="uint8[cells,capacity,32]",
                mutation_point="birth or turnover hereditary commit",
                reset_rule="canonical neutral digest on source allocation",
                checkpoint_inclusion="required diploid sidecar plus base digest binding",
                restore_rule="exact sidecar value; recomputation is validation only",
                rollback_rule="diploid shadow restored atomically",
                equality_rule="byte exact SHA-256 digest",
                route_support=cuda_only,
                evidence_field="diploid_sidecar.canonical_hereditary_digest",
            ),
            _state(
                "linkage_and_inheritance_rng",
                category="inheritance",
                owner="diploid and genome-v2 birth prepare",
                type_shape="6 linkage blocks; Python RNG words/index/gauss tapes/counters",
                mutation_point="birth prepare with commit-only cursor publication",
                reset_rule="seeded source allocation",
                checkpoint_inclusion="required in authoritative base/diploid checkpoint",
                restore_rule="exact words, cursors, buffers, and semantic draw counters",
                rollback_rule="trial RNG discarded; published RNG unchanged on failure",
                equality_rule="byte exact plus semantic draw count exact",
                route_support=cuda_only,
                evidence_field="birth_receipt.inheritance_draws",
            ),
            _state(
                "ordinary_birth_identity_lineage_and_reservations",
                category="transaction",
                owner="runtime birth prepare/commit v2",
                type_shape="fixed parent,mate,child,slot,lineage,event rows",
                mutation_point="prepare scratch then single commit kernel",
                reset_rule="scratch overwritten; outputs sentinel-filled",
                checkpoint_inclusion="only committed IDs/lineage/events in base checkpoint",
                restore_rule="exact committed state; no reservation resurrection",
                rollback_rule="no resident mutation before commit; failure is sticky",
                equality_rule="integer and event sequence exact",
                route_support=cuda_only,
                evidence_field="birth_receipt.identity_and_lineage",
                legal_cut_points=TRANSACTION_ONLY_CUT_POINTS,
            ),
            _state(
                "energy_health_reproduction_and_memory_targets",
                category="ecological_consequence",
                owner="unified flattened controller shared runtime state",
                type_shape="energy,health,injury,repro_threshold,policy_mem_dim per slot",
                mutation_point="one authoritative consequence boundary per surface",
                reset_rule="existing lifecycle initialization and dead-slot sanitizer",
                checkpoint_inclusion="required in canonical base projection",
                restore_rule="base projection is authoritative; genome sidecar cannot override",
                rollback_rule="existing tick/birth/turnover transaction semantics",
                equality_rule="frozen float contract; discrete memory integer exact",
                route_support=cuda_only,
                evidence_field="consequence_receipt.applied_delta",
            ),
            _state(
                "observation_and_policy_consequence_targets",
                category="ecological_consequence",
                owner="unified flattened controller observation/policy layer",
                type_shape="environment observations, attack bias, action cost inputs",
                mutation_point="after context/expression and before policy/action",
                reset_rule="rebuilt from hereditary baseline each tick",
                checkpoint_inclusion="authoritative baselines checkpointed; transient view absent",
                restore_rule="recomputed before first resumed policy decision",
                rollback_rule="discarded before any failed transaction publishes",
                equality_rule="frozen float contract",
                route_support=cuda_only,
                evidence_field="consequence_receipt.policy_input_delta",
            ),
            _state(
                "inheritance_sidecar_identity",
                category="checkpoint_identity",
                owner="inheritance_sidecar_identity_v1",
                type_shape="canonical JSON identity plus payload SHA-256",
                mutation_point="terminal checkpoint settlement only",
                reset_rule="new identity for each cell checkpoint",
                checkpoint_inclusion="wrapper for every diploid/genome-v2 sidecar",
                restore_rule="complete unique cell-ID map; reject mixed/extra/stale/corrupt",
                rollback_rule="invalid sidecar is never applied",
                equality_rule="canonical JSON and payload SHA-256 exact",
                route_support=cuda_only,
                evidence_field="sidecar.identity",
            ),
            _state(
                "checkpoint_cursors_and_cut_point",
                category="checkpoint_control",
                owner="canonical direct-resident checkpoint owner",
                type_shape="generation,tick,cursors,route/schema/artifact identities",
                mutation_point="legal terminal/cancellation boundary only",
                reset_rule="fresh run starts from admitted configuration",
                checkpoint_inclusion="required canonical projection",
                restore_rule="fail closed unless all identities and cursors agree",
                rollback_rule="valid complete checkpoint or absent checkpoint",
                equality_rule="integer/string/hash exact",
                route_support=cuda_only,
                evidence_field="checkpoint.identity_and_cursors",
            ),
            _state(
                "cancellation_and_sticky_failure",
                category="control",
                owner="resident run control and controller status arrays",
                type_shape="per-run cancellation plus per-cell terminal status",
                mutation_point="batch boundary or first failed authoritative operation",
                reset_rule="fresh owner only",
                checkpoint_inclusion="terminal/cancel receipt and base checkpoint status",
                restore_rule="failed owner cannot silently resume; fresh owner required",
                rollback_rule="no post-terminal draw, mutation, or partial archive",
                equality_rule="status code and terminal tick exact",
                route_support=cuda_only,
                evidence_field="continuation_receipt.terminal_status",
            ),
            _state(
                "compact_generation_summary",
                category="observability",
                owner="terminal genome-v2 summary reducer and Chronicle projection",
                type_shape="bounded per-generation/cell aggregates and selected-event sample",
                mutation_point="generation terminal only, after authoritative state settles",
                reset_rule="new bounded row per generation/cell",
                checkpoint_inclusion="derived report projection; never restores authority",
                restore_rule="regenerated from restored authority for equality check",
                rollback_rule="summary failure cannot mutate simulation state",
                equality_rule="exact counts/min/max; frozen float aggregate tolerance",
                route_support=cuda_only,
                evidence_field="chronicle.genome_upgrade_v2",
            ),
        ]
    )
    return rows


def checkpoint_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "state": row["name"],
            "checkpoint_inclusion": row["checkpoint_inclusion"],
            "restore_rule": row["restore_rule"],
            "rollback_rule": row["rollback_rule"],
            "equality_rule": row["equality_rule"],
            "legal_cut_points": "|".join(row["legal_cut_points"]),
        }
        for row in rows
    ]


def write_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = inventory_rows()
    promotion_contracts = promotion_contract_artifacts_v1()
    promotion_matrices = promotion_matrix_rows_v1()
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "production_route": "unified_native_sm75_cuda",
        "reference_routes": ["python", "cpu_cxx", "torch_cuda"],
        "legal_checkpoint_policy": (
            "no partial ordinary-birth or turnover transaction is checkpointable"
        ),
        "states": rows,
    }
    (output_dir / "genome_v2_authoritative_state_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    executable_contract = {
        "schema_version": EXECUTABLE_CONTRACT_SCHEMA_VERSION,
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "mechanic_contract_sha256": CONTRACT_SHA256,
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
        "mechanic_contract": contract_payload(),
        "required_obligations": list(PRE_ROUTE_PACKAGE_OBLIGATIONS),
        "state_inventory": "genome_v2_authoritative_state_inventory.json",
        "mutable_state_ownership": "genome_v2_mutable_state_ownership.csv",
        "checkpoint_surface_matrix": "genome_v2_checkpoint_surface_matrix.csv",
        "promotion_test_contract_sha256": {
            filename: content_sha256(payload)
            for filename, payload in sorted(promotion_contracts.items())
        },
        "promotion_test_matrix_sha256": {
            filename: content_sha256(matrix_rows)
            for filename, matrix_rows in sorted(promotion_matrices.items())
        },
    }
    (output_dir / "genome_v2_executable_contract.json").write_text(
        json.dumps(executable_contract, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    generated_json = {
        "genome_v2_p10_capability_contract.json": (
            GenomeV2P10CapabilityContract.v1().to_dict()
        ),
        "genome_v2_executable_package_contract_v1.json": (
            executable_package_contract_v1()
        ),
        "genome_v2_package_obligation_dictionary.json": (
            obligation_dictionary_v1()
        ),
        "genome_v2_package_invalidation_rules.json": (
            invalidation_rules_v1()
        ),
        "genome_v2_build_admission_policy.json": (
            build_admission_policy_v1()
        ),
        "genome_v2_chronicle_schema.json": chronicle_schema_v1(),
        "genome_v2_reporting_schema.json": reporting_schema_v1(),
        "genome_v2_observability_storage_budget.json": (
            observability_storage_budget_v1()
        ),
    }
    for filename, payload in generated_json.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    for filename, payload in promotion_contracts.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    for filename, matrix_rows in promotion_matrices.items():
        if not matrix_rows:
            raise ValueError(f"promotion matrix is empty: {filename}")
        with (output_dir / filename).open(
            "w", encoding="ascii", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=tuple(matrix_rows[0])
            )
            writer.writeheader()
            writer.writerows(matrix_rows)
    ownership_columns = (
        "name",
        "category",
        "owner",
        "type_shape",
        "mutation_point",
        "reset_rule",
        "route_support",
        "evidence_field",
    )
    with (
        output_dir / "genome_v2_mutable_state_ownership.csv"
    ).open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ownership_columns)
        writer.writeheader()
        writer.writerows(
            {column: row[column] for column in ownership_columns}
            for row in rows
        )
    checkpoint_columns = (
        "state",
        "checkpoint_inclusion",
        "restore_rule",
        "rollback_rule",
        "equality_rule",
        "legal_cut_points",
    )
    with (
        output_dir / "genome_v2_checkpoint_surface_matrix.csv"
    ).open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=checkpoint_columns)
        writer.writeheader()
        writer.writerows(checkpoint_rows(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_artifacts(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
