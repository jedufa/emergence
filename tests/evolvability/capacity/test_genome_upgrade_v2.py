from __future__ import annotations

from copy import deepcopy

import pytest

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from evolution.genome_upgrade_v2 import (
    CONTEXT_COUNT,
    CONTRACT_SHA256,
    HEREDITARY_LOCUS_COUNT,
    MEIOSIS_DRAW_COUNT,
    MUTATION_SEMANTIC_DRAW_COUNT,
    PHENOTYPE_CHANNELS,
    PRODUCT_CHANNELS,
    Q10_SCALE,
    GenomeUpgradeV2Error,
    GenomeUpgradeV2V1,
    assemble_child_from_tape_v1,
    attach_genome_upgrade_v2,
    build_runtime_context_q10_v1,
    contract_payload,
    dominance_fixture_v1,
    express_reference_v1,
    express_tensor_v1,
    genome_upgrade_v2_from_genome,
    project_runtime_traits_v1,
)
from evolution.genome_upgrade_v2_telemetry import (
    summarize_genome_upgrade_v2_population,
)
from evolution.diploid_inheritance_v1 import DIPLOID_INHERITANCE_PROFILE
from evolution.reproduction import crossover_genome
from simulation.rng_manager import RNGManager


def _genome() -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.0,
        aggression_bias=0.0,
        reproduction_threshold=30.0,
        mutation_rate=0.02,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.8,
        drive_noise=0.01,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _with_product(
    *,
    product_index: int,
    homolog_0: tuple[int, int, int, int],
    homolog_1: tuple[int, int, int, int],
    modifier_0: int = 0,
    modifier_1: int = 0,
) -> GenomeUpgradeV2V1:
    value = GenomeUpgradeV2V1.neutral()
    alleles = [
        [list(parameters) for parameters in homolog]
        for homolog in value.alleles
    ]
    modifiers = [list(homolog) for homolog in value.modifiers]
    alleles[0][product_index] = list(homolog_0)
    alleles[1][product_index] = list(homolog_1)
    modifiers[0][product_index] = modifier_0
    modifiers[1][product_index] = modifier_1
    return GenomeUpgradeV2V1(
        alleles=tuple(
            tuple(tuple(parameters) for parameters in homolog)
            for homolog in alleles
        ),
        modifiers=tuple(tuple(homolog) for homolog in modifiers),
    )


def _zero_context() -> tuple[int, ...]:
    return (0,) * CONTEXT_COUNT


def test_contract_is_fixed_shaped_polygenic_and_pleiotropic() -> None:
    payload = contract_payload()
    assert len(CONTRACT_SHA256) == 64
    assert payload["hidden_rng_draws"] == 0
    assert payload["runtime_projection"]["memory_dimension_ladder"] == [
        0,
        8,
        16,
        32,
        64,
    ]
    weights = payload["polygenic_weights_q10"]
    assert all(sum(value != 0 for value in row) >= 3 for row in weights)
    contribution_counts = [
        sum(row[index] != 0 for row in weights)
        for index in range(len(PRODUCT_CHANNELS))
    ]
    assert min(contribution_counts) >= 2


def test_neutral_expression_has_no_output_or_cost() -> None:
    result = express_reference_v1(GenomeUpgradeV2V1.neutral(), _zero_context())
    assert result.products_q10 == (0,) * len(PRODUCT_CHANNELS)
    assert result.phenotypes_q10 == (0,) * len(PHENOTYPE_CHANNELS)
    assert result.expression_cost_q10 == 0
    assert result.active_product_count == 0
    assert result.saturated_product_count == 0


def test_dosage_saturation_derives_dominance_without_a_dominance_field() -> None:
    value = _with_product(
        product_index=0,
        homolog_0=(Q10_SCALE, Q10_SCALE, Q10_SCALE, 0),
        homolog_1=(Q10_SCALE, Q10_SCALE, Q10_SCALE, 0),
    )
    result = express_reference_v1(value, _zero_context())
    assert result.allele_products_q10[0][0] == Q10_SCALE
    assert result.allele_products_q10[1][0] == Q10_SCALE
    assert result.products_q10[0] == Q10_SCALE
    assert result.saturated_product_count == 1
    fixture = dominance_fixture_v1(
        value,
        _zero_context(),
        product_index=0,
        threshold_q10=768,
    )
    assert fixture["dominance_is_derived"] is True
    assert fixture["homolog_0_expressed"] is True
    assert fixture["homolog_1_expressed"] is True
    assert fixture["heterozygote_expressed"] is True
    assert "dominance_strength" not in value.to_payload()


def test_modifier_and_environment_change_expression_contextually() -> None:
    base = _with_product(
        product_index=5,
        homolog_0=(512, 1024, 1024, 512),
        homolog_1=(0, 1024, 1024, 512),
        modifier_0=256,
        modifier_1=256,
    )
    context = list(_zero_context())
    context[3] = 512
    expressed = express_reference_v1(base, context)
    control = express_reference_v1(base, _zero_context())
    assert control.products_q10[5] == 576
    assert expressed.products_q10[5] == 832
    assert expressed.phenotype_dict()["sensory_fidelity"] > (
        control.phenotype_dict()["sensory_fidelity"]
    )


def test_runtime_context_is_normalized_bounded_and_developmental() -> None:
    assert build_runtime_context_q10_v1(
        energy=25.0,
        max_energy=100.0,
        injury=12.5,
        start_health=50.0,
        crowding=0.5,
        resource_a=1.5,
        resource_b=-0.25,
        hazard=0.125,
        age=7,
        maturity_age=8,
    ) == (768, 256, 512, 1024, 0, 128, 1024, 0)
    assert build_runtime_context_q10_v1(
        energy=100.0,
        max_energy=100.0,
        injury=0.0,
        start_health=50.0,
        crowding=0.0,
        resource_a=0.0,
        resource_b=0.0,
        hazard=0.0,
        age=32,
        maturity_age=8,
    )[-2:] == (0, 1024)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad"])
def test_runtime_context_rejects_invalid_numeric_inputs(value: object) -> None:
    with pytest.raises(
        GenomeUpgradeV2Error,
        match="runtime context normalization is invalid",
    ):
        build_runtime_context_q10_v1(
            energy=value,  # type: ignore[arg-type]
            max_energy=100.0,
            injury=0.0,
            start_health=50.0,
            crowding=0.0,
            resource_a=0.0,
            resource_b=0.0,
            hazard=0.0,
            age=1,
            maturity_age=8,
        )


def test_pleiotropy_has_measurable_cost_and_tradeoff() -> None:
    value = _with_product(
        product_index=0,
        homolog_0=(512, 1024, 1024, 0),
        homolog_1=(512, 1024, 1024, 0),
    )
    result = express_reference_v1(value, _zero_context())
    phenotype = result.phenotype_dict()
    assert result.expression_cost_q10 > 0
    assert phenotype["body_size"] > 0
    assert phenotype["metabolic_efficiency"] > 0
    assert phenotype["movement_capacity"] < 0
    assert phenotype["reproductive_investment"] < 0


def test_runtime_projection_is_bounded_and_uses_legal_memory_ladder() -> None:
    value = _with_product(
        product_index=10,
        homolog_0=(1024, 1024, 1024, 0),
        homolog_1=(1024, 1024, 1024, 0),
    )
    expression = express_reference_v1(value, _zero_context())
    projection = project_runtime_traits_v1(expression)
    factors = (
        projection.body_size_factor_q10,
        projection.movement_capacity_factor_q10,
        projection.damage_received_factor_q10,
        projection.metabolic_efficiency_factor_q10,
        projection.recovery_factor_q10,
        projection.sensory_fidelity_factor_q10,
        projection.reproduction_threshold_factor_q10,
        projection.offspring_energy_factor_q10,
    )
    assert all(768 <= value <= 1280 for value in factors)
    assert projection.memory_dimension in {0, 8, 16, 32, 64}
    assert projection.expression_energy_debit_q10 == (
        expression.expression_cost_q10
    )


def test_population_telemetry_measures_hidden_variation_and_pleiotropy() -> None:
    hidden = _with_product(
        product_index=0,
        homolog_0=(512, 1024, 1024, 0),
        homolog_1=(-512, 1024, 1024, 0),
    )
    telemetry = summarize_genome_upgrade_v2_population(
        [GenomeUpgradeV2V1.neutral(), hidden]
    )
    assert telemetry["population_count"] == 2
    assert telemetry["allele_parameter_heterozygosity"] > 0
    assert telemetry["hidden_variation_count"] == 1
    assert min(telemetry["pleiotropic_output_counts"].values()) >= 2
    empty = summarize_genome_upgrade_v2_population([])
    assert empty["allele_parameter_heterozygosity"] is None
    assert empty["expression_cost_mean_q10"] is None


def test_checkpoint_payload_round_trip_and_tamper_rejection() -> None:
    value = _with_product(
        product_index=3,
        homolog_0=(600, 700, 800, -200),
        homolog_1=(-300, 500, 900, 100),
        modifier_0=75,
        modifier_1=-125,
    )
    payload = value.to_payload()
    assert GenomeUpgradeV2V1.from_payload(payload) == value
    tampered = deepcopy(payload)
    tampered["alleles"][0][3][0] += 1
    with pytest.raises(GenomeUpgradeV2Error, match="genotype hash mismatch"):
        GenomeUpgradeV2V1.from_payload(tampered)


def test_explicit_tape_inheritance_has_fixed_draw_contract() -> None:
    parent_a = _with_product(
        product_index=0,
        homolog_0=(100, 1024, 1024, 0),
        homolog_1=(200, 1024, 1024, 0),
    )
    parent_b = _with_product(
        product_index=0,
        homolog_0=(-100, 1024, 1024, 0),
        homolog_1=(-200, 1024, 1024, 0),
    )
    mutation_homologs = [0] * HEREDITARY_LOCUS_COUNT
    mutation_enabled = [0] * HEREDITARY_LOCUS_COUNT
    mutation_deltas = [0] * HEREDITARY_LOCUS_COUNT
    mutation_enabled[0] = 1
    mutation_deltas[0] = 25
    child, receipt = assemble_child_from_tape_v1(
        parent_a,
        parent_b,
        donor_homologs=[1] * MEIOSIS_DRAW_COUNT,
        mutation_homologs=mutation_homologs,
        mutation_enabled=mutation_enabled,
        mutation_delta_q10=mutation_deltas,
    )
    assert child.alleles[0][0][0] == 225
    assert child.alleles[1][0][0] == -200
    assert receipt["meiosis_semantic_draws"] == MEIOSIS_DRAW_COUNT
    assert receipt["mutation_semantic_draws"] == MUTATION_SEMANTIC_DRAW_COUNT
    assert receipt["hidden_rng_draws"] == 0
    assert receipt["changed_loci"] == 1


def test_genome_canonical_round_trip_preserves_optional_upgrade_bundle() -> None:
    genome = _genome()
    original_hash = genome.canonical_hash()
    assert genome_upgrade_v2_from_genome(genome) is None
    value = _with_product(
        product_index=7,
        homolog_0=(400, 800, 900, 0),
        homolog_1=(200, 700, 800, 0),
    )
    attach_genome_upgrade_v2(genome, value)
    payload = genome.to_dict()
    assert payload["genome_upgrade_v2_v1"]["contract_sha256"] == CONTRACT_SHA256
    restored = Genome.from_dict(payload)
    assert genome_upgrade_v2_from_genome(restored) == value
    assert restored.canonical_hash() == genome.canonical_hash()
    assert restored.canonical_hash() != original_hash


def test_live_reference_birth_uses_authoritative_fixed_rng_tape() -> None:
    parent_a = _genome()
    parent_b = _genome()
    parent_b.aggression_bias = 0.1
    parent_b.refresh_canonical_payload_cache()
    attach_genome_upgrade_v2(
        parent_a,
        _with_product(
            product_index=1,
            homolog_0=(300, 800, 900, 0),
            homolog_1=(500, 700, 800, 0),
        ),
    )
    attach_genome_upgrade_v2(
        parent_b,
        _with_product(
            product_index=1,
            homolog_0=(-300, 800, 900, 0),
            homolog_1=(-500, 700, 800, 0),
        ),
    )
    manager = RNGManager(global_seed=7712)
    cfg = type(
        "Cfg",
        (),
        {
            "inheritance_profile": DIPLOID_INHERITANCE_PROFILE,
            "genome_c1_enable": False,
            "genome_c2_enable": False,
            "resource_handling_representation_enable": False,
            "diploid_mutation_enable": False,
            "genome_upgrade_v2_enable": True,
            "genome_upgrade_v2_mutation_rate_q10": 0,
            "genome_upgrade_v2_mutation_step_q10": 64,
        },
    )()
    child = crossover_genome(
        parent_a,
        parent_b,
        manager.mutation_py,
        rng_manager=manager,
        sim_cfg=cfg,
    )
    value = genome_upgrade_v2_from_genome(child)
    assert value is not None
    receipt = child._genome_upgrade_v2_inheritance_receipt_v1
    assert receipt["total_semantic_draws"] == (
        MEIOSIS_DRAW_COUNT + MUTATION_SEMANTIC_DRAW_COUNT
    )
    assert receipt["hidden_rng_draws"] == 0
    assert receipt["changed_loci"] == 0
    assert child.to_dict()["genome_upgrade_v2_v1"] == value.to_payload()
    assert "rng_semantics_version" not in receipt


def test_live_reference_birth_dispatches_rngv2_without_host_tape() -> None:
    parent_a = _genome()
    parent_b = _genome()
    attach_genome_upgrade_v2(parent_a, GenomeUpgradeV2V1.neutral())
    attach_genome_upgrade_v2(parent_b, GenomeUpgradeV2V1.neutral())
    manager = RNGManager(global_seed=7712)
    cfg = type(
        "Cfg",
        (),
        {
            "inheritance_profile": DIPLOID_INHERITANCE_PROFILE,
            "genome_c1_enable": False,
            "genome_c2_enable": False,
            "resource_handling_representation_enable": False,
            "diploid_mutation_enable": False,
            "genome_upgrade_v2_enable": True,
            "genome_upgrade_v2_mutation_rate_q10": 20,
            "genome_upgrade_v2_mutation_step_q10": 64,
            "rng_semantics_version": 2,
            "world_uid": 1,
        },
    )()
    child = crossover_genome(
        parent_a,
        parent_b,
        manager.mutation_py,
        rng_manager=manager,
        sim_cfg=cfg,
        structural_identity_context={"tick": 3, "child_agent_id": 8},
    )
    receipt = child._genome_upgrade_v2_inheritance_receipt_v1
    assert receipt["rng_semantics_version"] == 2
    assert receipt["host_tape"] is False
    assert receipt["total_semantic_draws"] == (
        MEIOSIS_DRAW_COUNT + MUTATION_SEMANTIC_DRAW_COUNT
    )


def test_live_enable_fails_closed_outside_diploid_profile() -> None:
    manager = RNGManager(global_seed=7713)
    before = manager.capture_snapshot()
    cfg = type(
        "Cfg",
        (),
        {
            "inheritance_profile": "legacy_midpoint_v1",
            "genome_upgrade_v2_enable": True,
        },
    )()
    with pytest.raises(
        RuntimeError, match="GENOME_UPGRADE_V2_REQUIRES_DIPLOID_INHERITANCE"
    ):
        crossover_genome(
            _genome(),
            _genome(),
            manager.mutation_py,
            rng_manager=manager,
            sim_cfg=cfg,
        )
    assert manager.capture_snapshot() == before


def test_tensor_cpu_matches_scalar_reference_exactly() -> None:
    torch = pytest.importorskip("torch")
    values = [
        GenomeUpgradeV2V1.neutral(),
        _with_product(
            product_index=2,
            homolog_0=(733, 811, 967, -211),
            homolog_1=(-409, 613, 887, 319),
            modifier_0=177,
            modifier_1=-83,
        ),
    ]
    contexts = [
        _zero_context(),
        (111, -222, 333, -444, 555, -666, 777, -888),
    ]
    alleles = torch.tensor([value.alleles for value in values], dtype=torch.int64)
    modifiers = torch.tensor(
        [value.modifiers for value in values], dtype=torch.int64
    )
    context = torch.tensor(contexts, dtype=torch.int64)
    result = express_tensor_v1(alleles, modifiers, context)
    for index, (value, context_row) in enumerate(zip(values, contexts)):
        expected = express_reference_v1(value, context_row)
        assert tuple(result["products_q10"][index].tolist()) == expected.products_q10
        assert tuple(result["phenotypes_q10"][index].tolist()) == (
            expected.phenotypes_q10
        )
        assert int(result["expression_cost_q10"][index]) == (
            expected.expression_cost_q10
        )


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_matches_cpu_tensor_contract_bit_exact() -> None:
    import torch

    value = _with_product(
        product_index=9,
        homolog_0=(701, 823, 991, -317),
        homolog_1=(-257, 719, 877, 401),
        modifier_0=211,
        modifier_1=-99,
    )
    context = (113, -227, 331, -449, 557, -661, 773, -881)
    alleles = torch.tensor([value.alleles], dtype=torch.int64)
    modifiers = torch.tensor([value.modifiers], dtype=torch.int64)
    contexts = torch.tensor([context], dtype=torch.int64)
    cpu = express_tensor_v1(alleles, modifiers, contexts)
    cuda = express_tensor_v1(
        alleles.cuda(),
        modifiers.cuda(),
        contexts.cuda(),
    )
    for key in cpu:
        assert torch.equal(cpu[key], cuda[key].cpu()), key
