from __future__ import annotations

from pathlib import Path
import random
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _source() -> str:
    state = random.Random(0).getstate()[1]
    words = ", ".join(f"{int(word)}U" for word in state[:624])
    return textwrap.dedent(
        rf"""
        #include <cassert>
        #include <cmath>
        #include <cstdint>
        #include <cstring>

        #define __device__
        #define __forceinline__ inline
        #define isfinite std::isfinite

        #include "engine/perf/cuda_architecture_b_persistent_turnover_scalar_genome.cuh"

        namespace scalar =
            eiel::cuda_architecture_b::persistent_turnover_scalar_genome;
        namespace inheritance = eiel::inheritance_linkage_v1;

        constexpr int64_t kSlots = 4;
        constexpr int64_t kGenotypes = 4;
        constexpr int64_t kGauss = 256;
        constexpr uint32_t kInitialWords[624] = {{{words}}};

        struct Fixture {{
            int64_t ids[kSlots] = {{10, 20, 30, 40}};
            bool alive[kSlots] = {{true, true, true, true}};
            int32_t ints[8][kSlots] = {{}};
            int64_t policy_mem_dim[kSlots] = {{}};
            bool policy_use_gru[kSlots] = {{}};
            float floats[24][kSlots] = {{}};

            int64_t parent_pool[1] = {{0}};
            int64_t parent_pool_count = 2;
            int64_t parent_pool_storage[3] = {{0, 1, 3}};
            int64_t child_slots[1] = {{2}};
            int64_t planned_count = 1;

            uint32_t words[624] = {{}};
            int64_t index = 624;
            bool has_gauss = false;
            double gauss_next = 0.0;
            double gauss_primary[kGauss] = {{}};
            double gauss_secondary[kGauss] = {{}};
            int64_t raw_draws = 0;
            int64_t semantic_draws = 0;
            uint32_t trial_words[624] = {{}};

            int64_t selected_parent[1] = {{-1}};
            int64_t selected_mate[1] = {{-1}};
            int64_t committed = -1;
            int64_t scalar_draws = -1;
            int32_t scalar_status = -1;

            uint8_t donors[7] = {{}};
            uint8_t donor_statuses[7] = {{}};
            int64_t donor_draws = -1;
            int32_t inheritance_status = -1;
            int64_t genotype_codes[kSlots] = {{2, 0, 3, 1}};
            int64_t digest_rank_by_code[kGenotypes] = {{2, 0, 3, 1}};
            int64_t canonical_parent_0_out[1] = {{-1}};
            int64_t canonical_parent_1_out[1] = {{-1}};

            Fixture() {{
                std::memcpy(words, kInitialWords, sizeof(words));
                for (int row = 0; row < 8; ++row) {{
                    ints[row][0] = 1 + row;
                    ints[row][1] = 11 + row;
                    ints[row][2] = -1;
                }}
                ints[0][0] = 64;
                ints[0][1] = 128;
                ints[1][0] = 8;
                ints[1][1] = 16;
                policy_mem_dim[0] = 8;
                policy_mem_dim[1] = 16;
                policy_mem_dim[3] = 24;
                policy_use_gru[0] = false;
                policy_use_gru[1] = true;
                policy_use_gru[3] = false;
                for (int row = 0; row < 24; ++row) {{
                    floats[row][0] = 0.01f * static_cast<float>(row + 1);
                    floats[row][1] = 0.02f * static_cast<float>(row + 1);
                    floats[row][2] = -99.0f;
                    floats[row][3] = 0.03f * static_cast<float>(row + 1);
                }}
                floats[0][0] = -0.75f;
                floats[0][1] = 0.75f;
                floats[1][0] = -1.25f;
                floats[1][1] = 1.25f;
                floats[2][0] = 0.001f;
                floats[2][1] = 0.002f;
                floats[3][0] = 12.0f;
                floats[3][1] = 72.0f;
                floats[4][0] = 0.5f;
                floats[4][1] = 2.0f;
                floats[5][0] = 0.80f;
                floats[5][1] = 0.95f;
                floats[6][0] = 0.01f;
                floats[6][1] = 0.09f;
                floats[7][0] = 0.5f;
                floats[7][1] = 2.5f;
                floats[8][0] = 0.05f;
                floats[8][1] = 0.75f;
                floats[9][0] = 0.20f;
                floats[9][1] = 0.80f;
                floats[10][0] = 0.10f;
                floats[10][1] = 0.70f;
                floats[11][0] = 0.30f;
                floats[11][1] = 0.60f;
                floats[12][0] = 0.40f;
                floats[12][1] = 0.90f;
                floats[13][0] = 0.25f;
                floats[13][1] = 0.75f;
                floats[14][0] = 0.04f;
                floats[14][1] = 0.40f;
                floats[15][0] = 0.02f;
                floats[15][1] = 0.18f;
                floats[16][0] = 12.0f;
                floats[16][1] = 72.0f;
                floats[17][0] = 0.2f;
                floats[17][1] = 0.7f;
                for (int row = 0; row < 8; ++row) {{
                    ints[row][3] = 21 + row;
                }}
                ints[0][3] = 96;
                ints[1][3] = 24;
                floats[0][3] = 0.25f;
                floats[1][3] = 0.50f;
                floats[2][3] = 0.003f;
                floats[3][3] = 36.0f;
            }}
        }};

        scalar::FlatFixedMlpScalarGenomeState genome(Fixture& f) {{
            return {{
                f.ids,
                f.alive,
                f.ints[0],
                f.ints[1],
                f.ints[2],
                f.ints[3],
                f.ints[4],
                f.ints[5],
                f.ints[6],
                f.ints[7],
                f.policy_mem_dim,
                f.policy_use_gru,
                f.floats[0],
                f.floats[1],
                f.floats[2],
                f.floats[3],
                f.floats[4],
                f.floats[5],
                f.floats[6],
                f.floats[7],
                f.floats[8],
                f.floats[9],
                f.floats[10],
                f.floats[11],
                f.floats[12],
                f.floats[13],
                f.floats[14],
                f.floats[15],
                f.floats[16],
                f.floats[17],
                f.floats[18],
                f.floats[19],
                f.floats[20],
                f.floats[21],
                f.floats[22],
                f.floats[23],
            }};
        }}

        scalar::FlatTurnoverScalarGenomePlan plan(Fixture& f) {{
            return {{
                f.parent_pool_storage,
                &f.parent_pool_count,
                f.child_slots,
                &f.planned_count,
            }};
        }}

        scalar::FlatCpythonRandomState rng(Fixture& f) {{
            return {{
                f.words,
                &f.index,
                &f.has_gauss,
                &f.gauss_next,
                f.gauss_primary,
                f.gauss_secondary,
                &f.raw_draws,
                &f.semantic_draws,
            }};
        }}

        scalar::FlatTurnoverScalarGenomeOutputs outputs(Fixture& f) {{
            return {{
                f.selected_parent,
                f.selected_mate,
                &f.committed,
                &f.scalar_draws,
                &f.scalar_status,
            }};
        }}

        scalar::FlatTurnoverInheritanceOutputsV1 inheritance_outputs(
            Fixture& f
        ) {{
            return {{
                f.donors,
                f.donor_statuses,
                &f.donor_draws,
                &f.inheritance_status,
            }};
        }}

        scalar::TurnoverScalarGenomeConfig config() {{
            return {{1, kSlots, kGauss, 16}};
        }}

        void run_legacy(Fixture& f) {{
            scalar::commit_turnover_fixed_mlp_child_scalar_genome_serial(
                genome(f),
                plan(f),
                rng(f),
                outputs(f),
                config(),
                f.trial_words
            );
        }}

        void run_policy(
            Fixture& f,
            const scalar::TurnoverInheritanceConfigV1 policy,
            const bool provide_outputs = true
        ) {{
            scalar::
                commit_turnover_fixed_mlp_child_scalar_genome_with_inheritance_v1_serial(
                    genome(f),
                    plan(f),
                    rng(f),
                    outputs(f),
                    config(),
                    f.trial_words,
                    policy,
                    provide_outputs
                        ? inheritance_outputs(f)
                        : scalar::FlatTurnoverInheritanceOutputsV1{{
                            nullptr, nullptr, nullptr, nullptr
                        }}
                );
        }}

        scalar::TurnoverCanonicalParentRankAuthorityV1 rank_authority(
            Fixture& f
        ) {{
            return {{
                f.genotype_codes,
                f.digest_rank_by_code,
                kGenotypes,
                f.canonical_parent_0_out,
                f.canonical_parent_1_out,
            }};
        }}

        void run_ranked(
            Fixture& f,
            scalar::TurnoverCanonicalParentRankAuthorityV1 authority
        ) {{
            scalar::
                commit_turnover_fixed_mlp_child_scalar_genome_with_ranked_inheritance_v2_serial(
                    genome(f),
                    plan(f),
                    rng(f),
                    outputs(f),
                    config(),
                    f.trial_words,
                    {{
                        inheritance::InheritanceModeV1::linked_parent_copy_v1,
                        7,
                        false,
                        authority,
                    }},
                    inheritance_outputs(f)
                );
        }}

        void assert_rng_pristine(const Fixture& f) {{
            Fixture pristine;
            assert(std::memcmp(f.words, kInitialWords, sizeof(f.words)) == 0);
            assert(f.index == 624);
            assert(f.raw_draws == 0);
            assert(f.semantic_draws == 0);
            assert(
                std::memcmp(f.ints, pristine.ints, sizeof(f.ints)) == 0
            );
            assert(
                std::memcmp(
                    f.policy_mem_dim,
                    pristine.policy_mem_dim,
                    sizeof(f.policy_mem_dim)
                ) == 0
            );
            assert(
                std::memcmp(
                    f.policy_use_gru,
                    pristine.policy_use_gru,
                    sizeof(f.policy_use_gru)
                ) == 0
            );
            assert(
                std::memcmp(f.floats, pristine.floats, sizeof(f.floats)) == 0
            );
        }}

        void test_fail_closed_contracts() {{
            int64_t canonical_0[1] = {{0}};
            int64_t canonical_1[1] = {{1}};

            Fixture invalid_mode;
            run_policy(
                invalid_mode,
                {{
                    static_cast<inheritance::InheritanceModeV1>(99),
                    0,
                    true,
                    canonical_0,
                    canonical_1,
                }}
            );
            assert(
                invalid_mode.scalar_status ==
                scalar::kStatusInvalidInheritanceMode
            );
            assert_rng_pristine(invalid_mode);

            Fixture invalid_budget;
            run_policy(
                invalid_budget,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    6,
                    true,
                    canonical_0,
                    canonical_1,
                }}
            );
            assert(
                invalid_budget.scalar_status ==
                scalar::kStatusInvalidInheritanceDonorBudget
            );
            assert_rng_pristine(invalid_budget);

            Fixture missing_sidecar;
            run_policy(
                missing_sidecar,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    7,
                    true,
                    nullptr,
                    nullptr,
                }}
            );
            assert(
                missing_sidecar.scalar_status ==
                scalar::kStatusMissingInheritanceSidecar
            );
            assert_rng_pristine(missing_sidecar);

            Fixture missing_outputs;
            run_policy(
                missing_outputs,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    7,
                    true,
                    canonical_0,
                    canonical_1,
                }},
                false
            );
            assert(
                missing_outputs.scalar_status ==
                scalar::kStatusMissingInheritanceOutputs
            );
            assert_rng_pristine(missing_outputs);

            int64_t wrong_1[1] = {{2}};
            Fixture wrong_order;
            run_policy(
                wrong_order,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    7,
                    true,
                    canonical_0,
                    wrong_1,
                }}
            );
            assert(
                wrong_order.scalar_status ==
                scalar::kStatusInvalidCanonicalParentOrder
            );
            assert(
                wrong_order.inheritance_status ==
                scalar::kStatusInvalidCanonicalParentOrder
            );
            assert_rng_pristine(wrong_order);
        }}

        void test_legacy_wrapper_is_exact() {{
            Fixture wrapper;
            Fixture typed;
            run_legacy(wrapper);
            run_policy(
                typed,
                {{
                    inheritance::InheritanceModeV1::legacy_midpoint_v1,
                    0,
                    true,
                    nullptr,
                    nullptr,
                }}
            );
            assert(wrapper.scalar_status == scalar::kStatusOk);
            assert(typed.scalar_status == scalar::kStatusOk);
            assert(
                std::memcmp(wrapper.ints, typed.ints, sizeof(wrapper.ints)) == 0
            );
            assert(
                std::memcmp(
                    wrapper.policy_mem_dim,
                    typed.policy_mem_dim,
                    sizeof(wrapper.policy_mem_dim)
                ) == 0
            );
            assert(
                std::memcmp(
                    wrapper.policy_use_gru,
                    typed.policy_use_gru,
                    sizeof(wrapper.policy_use_gru)
                ) == 0
            );
            assert(
                std::memcmp(
                    wrapper.floats,
                    typed.floats,
                    sizeof(wrapper.floats)
                ) == 0
            );
            assert(
                std::memcmp(
                    wrapper.words,
                    typed.words,
                    sizeof(wrapper.words)
                ) == 0
            );
            assert(wrapper.index == typed.index);
            assert(wrapper.raw_draws == typed.raw_draws);
            assert(wrapper.semantic_draws == typed.semantic_draws);
            assert(wrapper.scalar_draws == typed.scalar_draws);
            assert(wrapper.selected_parent[0] == 0);
            assert(wrapper.selected_mate[0] == 1);
            assert(wrapper.ints[0][2] == 128);
            assert(wrapper.floats[0][2] == 0.0f);
            assert(wrapper.floats[1][2] == 0.0f);
            assert(wrapper.floats[2][2] == 0.0015f);
            assert(wrapper.floats[3][2] == 42.0f);
            assert(wrapper.floats[4][2] == 1.25f);
            assert(wrapper.scalar_draws == 16);
            assert(typed.donor_draws == 0);
            for (int group = 0; group < 7; ++group) {{
                assert(
                    typed.donors[group] == inheritance::kNoDonorOrdinal
                );
            }}
        }}

        void test_linked_blocks_and_telemetry() {{
            int64_t canonical_0[1] = {{0}};
            int64_t canonical_1[1] = {{1}};
            Fixture linked;
            run_policy(
                linked,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    7,
                    true,
                    canonical_0,
                    canonical_1,
                }}
            );
            assert(linked.scalar_status == scalar::kStatusOk);
            assert(linked.inheritance_status == scalar::kStatusOk);
            assert(linked.committed == 1);
            assert(linked.selected_parent[0] == 0);
            assert(linked.selected_mate[0] == 1);
            constexpr uint8_t expected[7] = {{1, 0, 1, 1, 1, 0, 1}};
            for (int group = 0; group < 7; ++group) {{
                assert(linked.donors[group] == expected[group]);
                assert(
                    linked.donor_statuses[group] ==
                    static_cast<uint8_t>(
                        inheritance::SelectionStatusV1::ok
                    )
                );
            }}
            assert(linked.donor_draws == 7);

            // Behavioral block: canonical parent 1.
            assert(linked.floats[0][2] == linked.floats[0][1]);
            assert(linked.floats[1][2] == linked.floats[1][1]);
            assert(linked.floats[4][2] == linked.floats[4][1]);
            // Regulation/prediction block: canonical parent 0.
            for (int row = 5; row <= 15; ++row) {{
                assert(linked.floats[row][2] == linked.floats[row][0]);
            }}
            // Life-history and mutation-rate blocks: canonical parent 1.
            assert(linked.floats[3][2] == linked.floats[3][1]);
            assert(linked.floats[2][2] == linked.floats[2][1]);
            // Represented memory/controller block: canonical parent 1.
            assert(linked.ints[0][2] == linked.ints[0][1]);
            assert(linked.ints[1][2] == linked.ints[1][1]);
            assert(linked.policy_mem_dim[2] == linked.policy_mem_dim[1]);
            assert(linked.policy_use_gru[2] == linked.policy_use_gru[1]);
            assert(linked.semantic_draws == linked.scalar_draws);
            assert(linked.semantic_draws >= 19);
        }}

        void test_mutation_disabled_preserves_only_inheritance_draws() {{
            int64_t canonical_0[1] = {{0}};
            int64_t canonical_1[1] = {{1}};

            Fixture legacy;
            run_policy(
                legacy,
                {{
                    inheritance::InheritanceModeV1::legacy_midpoint_v1,
                    0,
                    false,
                    nullptr,
                    nullptr,
                }}
            );
            assert(legacy.scalar_status == scalar::kStatusOk);
            assert(legacy.scalar_draws >= 3);
            assert(legacy.floats[0][2] == 0.0f);
            assert(legacy.floats[1][2] == 0.0f);
            assert(legacy.floats[2][2] == 0.0015f);
            assert(legacy.floats[3][2] == 42.0f);

            Fixture linked;
            run_policy(
                linked,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    7,
                    false,
                    canonical_0,
                    canonical_1,
                }}
            );
            assert(linked.scalar_status == scalar::kStatusOk);
            assert(linked.inheritance_status == scalar::kStatusOk);
            assert(linked.donor_draws == 7);
            assert(linked.scalar_draws == legacy.scalar_draws + 6);
            assert(linked.floats[0][2] == linked.floats[0][1]);
            assert(linked.floats[1][2] == linked.floats[1][1]);
            assert(linked.floats[2][2] == linked.floats[2][1]);
            assert(linked.floats[3][2] == linked.floats[3][1]);
            assert(linked.semantic_draws == linked.scalar_draws);
        }}

        void test_ranked_authority_binds_the_actual_redrawn_pair() {{
            Fixture ranked;
            ranked.parent_pool_count = 3;
            run_ranked(ranked, rank_authority(ranked));

            assert(ranked.scalar_status == scalar::kStatusOk);
            assert(ranked.inheritance_status == scalar::kStatusOk);
            assert(ranked.committed == 1);
            assert(ranked.selected_parent[0] != ranked.selected_mate[0]);
            const int64_t selected_0 = ranked.selected_parent[0];
            const int64_t selected_1 = ranked.selected_mate[0];
            assert(
                (ranked.canonical_parent_0_out[0] == selected_0
                 && ranked.canonical_parent_1_out[0] == selected_1)
                || (ranked.canonical_parent_0_out[0] == selected_1
                    && ranked.canonical_parent_1_out[0] == selected_0)
            );
            const int64_t rank_0 = ranked.digest_rank_by_code[
                ranked.genotype_codes[ranked.canonical_parent_0_out[0]]
            ];
            const int64_t rank_1 = ranked.digest_rank_by_code[
                ranked.genotype_codes[ranked.canonical_parent_1_out[0]]
            ];
            assert(rank_0 <= rank_1);
            assert(ranked.donor_draws == 7);
            assert(ranked.scalar_draws >= 11);
        }}

        void test_ranked_v2_matches_v1_when_authority_agrees() {{
            Fixture ranked;
            ranked.parent_pool_count = 2;
            run_ranked(ranked, rank_authority(ranked));
            assert(ranked.scalar_status == scalar::kStatusOk);

            Fixture v1;
            int64_t canonical_0[1] = {{
                ranked.canonical_parent_0_out[0]
            }};
            int64_t canonical_1[1] = {{
                ranked.canonical_parent_1_out[0]
            }};
            run_policy(
                v1,
                {{
                    inheritance::InheritanceModeV1::linked_parent_copy_v1,
                    7,
                    false,
                    canonical_0,
                    canonical_1,
                }}
            );
            assert(v1.scalar_status == scalar::kStatusOk);
            assert(
                std::memcmp(ranked.ints, v1.ints, sizeof(ranked.ints)) == 0
            );
            assert(
                std::memcmp(ranked.floats, v1.floats, sizeof(ranked.floats)) == 0
            );
            assert(
                std::memcmp(ranked.words, v1.words, sizeof(ranked.words)) == 0
            );
            assert(ranked.index == v1.index);
            assert(ranked.raw_draws == v1.raw_draws);
            assert(ranked.semantic_draws == v1.semantic_draws);
            assert(ranked.donor_draws == v1.donor_draws);
            assert(
                std::memcmp(ranked.donors, v1.donors, sizeof(ranked.donors)) == 0
            );
        }}

        void test_ranked_authority_failures_are_pristine() {{
            Fixture null_rank;
            auto missing = rank_authority(null_rank);
            missing.digest_rank_by_code = nullptr;
            run_ranked(null_rank, missing);
            assert(
                null_rank.scalar_status ==
                scalar::kStatusInvalidCanonicalParentRankAuthority
            );
            assert(
                null_rank.inheritance_status ==
                scalar::kStatusInvalidCanonicalParentRankAuthority
            );
            assert_rng_pristine(null_rank);
            assert(null_rank.canonical_parent_0_out[0] == -1);
            assert(null_rank.canonical_parent_1_out[0] == -1);

            Fixture out_of_range;
            for (int64_t slot = 0; slot < kSlots; ++slot) {{
                out_of_range.genotype_codes[slot] = kGenotypes;
            }}
            run_ranked(out_of_range, rank_authority(out_of_range));
            assert(
                out_of_range.scalar_status ==
                scalar::kStatusInvalidCanonicalParentRankAuthority
            );
            assert(
                out_of_range.inheritance_status ==
                scalar::kStatusInvalidCanonicalParentRankAuthority
            );
            assert_rng_pristine(out_of_range);
            assert(out_of_range.canonical_parent_0_out[0] == -1);
            assert(out_of_range.canonical_parent_1_out[0] == -1);
        }}

        int main() {{
            test_fail_closed_contracts();
            test_legacy_wrapper_is_exact();
            test_linked_blocks_and_telemetry();
            test_mutation_disabled_preserves_only_inheritance_draws();
            test_ranked_authority_binds_the_actual_redrawn_pair();
            test_ranked_v2_matches_v1_when_authority_agrees();
            test_ranked_authority_failures_are_pristine();
            return 0;
        }}
        """
    )


def test_turnover_linked_inheritance_contract_cpu_cpp(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    include = tmp_path / "include"
    include.mkdir()
    (include / "cuda_runtime.h").write_text("#pragma once\n", encoding="utf-8")
    source = tmp_path / "turnover_linked_v1.cpp"
    source.write_text(_source(), encoding="utf-8")
    executable = tmp_path / "turnover_linked_v1"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-I",
            str(include),
            "-I",
            str(ROOT),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_turnover_linked_inheritance_cuda_api_compiles(tmp_path: Path) -> None:
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc is unavailable")
    source = (
        ROOT
        / "tests"
        / "perf"
        / "test_cuda_architecture_b_persistent_turnover_scalar_genome_compile.cu"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT),
            "-c",
            str(source),
            "-o",
            str(tmp_path / "turnover_linked_v1.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
