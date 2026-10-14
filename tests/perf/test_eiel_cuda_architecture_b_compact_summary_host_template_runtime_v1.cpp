#include "cuda_architecture_b_compact_summary_host_template_runtime_v1.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime =
    eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1;
namespace renderer =
    eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1;

namespace {

template <typename T>
std::vector<T> read_binary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize byte_count = stream.tellg();
    if (byte_count < 0
        || byte_count % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid element width in " + path);
    }
    stream.seekg(0);
    std::vector<T> values(
        static_cast<std::size_t>(byte_count) / sizeof(T)
    );
    if (byte_count != 0
        && !stream.read(
            reinterpret_cast<char*>(values.data()),
            byte_count
        )) {
        throw std::runtime_error("cannot read " + path);
    }
    return values;
}

runtime::HostTemplateWireTableV1 table(
    const std::vector<std::uint64_t>& values,
    const std::uint32_t columns
) {
    if (values.size() % columns != 0U) {
        throw std::runtime_error("table has a partial row");
    }
    return {
        values.empty() ? nullptr : values.data(),
        values.size() / columns,
        columns,
        0,
    };
}

int report_prepare_failure(
    const runtime::HostTemplateRuntimeStatusV1 status,
    const runtime::HostTemplateRuntimeV1& prepared
) {
    const runtime::HostTemplateRuntimeFailureV1 failure =
        runtime::host_template_runtime_failure_v1(&prepared);
    std::cout
        << "status=" << runtime::host_template_runtime_status_name_v1(status)
        << " component=" << failure.component
        << " row=" << failure.row
        << " column=" << failure.column
        << " renderer_status=" << failure.renderer_status
        << " renderer_component=" << failure.renderer_component
        << " renderer_index=" << failure.renderer_index
        << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected one wire directory argument");
        }
        const std::string root = argv[1];
        const std::vector<std::uint64_t> programs = read_binary<std::uint64_t>(
            root + "/programs.bin"
        );
        const std::vector<std::uint64_t> tokens = read_binary<std::uint64_t>(
            root + "/tokens.bin"
        );
        const std::vector<std::uint64_t> parts = read_binary<std::uint64_t>(
            root + "/parts.bin"
        );
        const std::vector<std::uint64_t> constraints =
            read_binary<std::uint64_t>(root + "/constraints.bin");
        const std::vector<std::uint8_t> byte_pool = read_binary<std::uint8_t>(
            root + "/byte_pool.bin"
        );

        const runtime::HostTemplateWireViewV1 wire = {
            runtime::kHostTemplateWireAbiVersionV1,
            0,
            2,
            1,
            table(programs, runtime::kProgramWireColumnCountV1),
            table(tokens, runtime::kTokenWireColumnCountV1),
            table(parts, runtime::kPartWireColumnCountV1),
            table(constraints, runtime::kConstraintWireColumnCountV1),
            {
                byte_pool.empty() ? nullptr : byte_pool.data(),
                byte_pool.size(),
                0,
            },
            0,
        };
        runtime::HostTemplateRuntimeV1 prepared;
        const runtime::HostTemplateRuntimeStatusV1 status =
            runtime::prepare_host_template_runtime_v1(wire, &prepared);
        if (status != runtime::HostTemplateRuntimeStatusV1::kSuccess) {
            return report_prepare_failure(status, prepared);
        }
        if (runtime::host_template_program_count_v1(&prepared) != 12U) {
            throw std::runtime_error("g2/b1 did not prepare exactly 12 programs");
        }

        const runtime::HostTemplateProgramBindingV1 first_before =
            runtime::lookup_host_template_program_v1(&prepared, 0, 0, 1);
        if (first_before.program == nullptr) {
            throw std::runtime_error("first program lookup failed");
        }
        std::uint32_t nullable_count = 0U;
        for (std::uint32_t generation = 0; generation < 2U; ++generation) {
            for (std::uint32_t surface = 1; surface <= 6U; ++surface) {
                const runtime::HostTemplateProgramBindingV1 binding =
                    runtime::lookup_host_template_program_v1(
                        &prepared,
                        generation,
                        0,
                        surface
                    );
                if (binding.program == nullptr
                    || binding.token_value_count != binding.program->token_count) {
                    throw std::runtime_error("program binding is incomplete");
                }
                for (std::uint32_t token = 0;
                     token < binding.token_value_count;
                     ++token) {
                    runtime::HostTemplateTokenStorageV1 storage =
                        runtime::lookup_host_template_token_storage_v1(
                            &prepared,
                            generation,
                            0,
                            surface,
                            token
                        );
                    if (storage.value == nullptr) {
                        throw std::runtime_error("token storage lookup failed");
                    }
                    if (storage.value->token_type == static_cast<std::uint32_t>(
                            renderer::HostTemplateTokenTypeV1::kNullableFloat64
                        )) {
                        double representative = 0.0;
                        std::memcpy(
                            &representative,
                            &storage.value->scalar_low,
                            sizeof(representative)
                        );
                        if (storage.value->scalar_high != 1U
                            || representative
                                != -std::numeric_limits<double>::max()) {
                            throw std::runtime_error(
                                "nullable representative initialization drift"
                            );
                        }
                        ++nullable_count;
                    }
                    storage.value->scalar_low = 0U;
                    storage.value->scalar_high = 0U;
                    if (storage.value->token_type == static_cast<std::uint32_t>(
                            renderer::HostTemplateTokenTypeV1::kTensorShape
                        )) {
                        if ((storage.shape_dimension_capacity != 0U)
                            != (storage.shape_dimensions != nullptr)) {
                            throw std::runtime_error("shape backing is incomplete");
                        }
                        std::fill_n(
                            storage.shape_dimensions,
                            storage.shape_dimension_capacity,
                            0U
                        );
                    } else if (storage.value->token_type
                        == static_cast<std::uint32_t>(
                            renderer::HostTemplateTokenTypeV1::kSha256Hex
                        )) {
                        if (storage.sha256_hex == nullptr
                            || storage.sha256_hex_capacity != 64U) {
                            throw std::runtime_error("SHA backing is incomplete");
                        }
                        std::fill_n(
                            storage.sha256_hex,
                            storage.sha256_hex_capacity,
                            static_cast<std::uint8_t>('0')
                        );
                    }
                }
            }
        }
        if (nullable_count == 0U) {
            throw std::runtime_error("nullable runtime token was not covered");
        }

        const runtime::HostTemplateProgramBindingV1 first_after =
            runtime::lookup_host_template_program_v1(&prepared, 0, 0, 1);
        if (first_after.program != first_before.program
            || first_after.token_values != first_before.token_values) {
            throw std::runtime_error("prepare backing pointers moved");
        }
        if (runtime::lookup_host_template_program_v1(&prepared, 2, 0, 1).program
            != nullptr) {
            throw std::runtime_error("out-of-range lookup did not fail closed");
        }

        std::uint64_t rendered_bytes = 0;
        for (std::uint32_t generation = 0; generation < 2U; ++generation) {
            for (std::uint32_t surface = 1; surface <= 6U; ++surface) {
                const runtime::HostTemplateProgramBindingV1 binding =
                    runtime::lookup_host_template_program_v1(
                        &prepared,
                        generation,
                        0,
                        surface
                    );
                std::vector<std::uint8_t> output(
                    static_cast<std::size_t>(
                        binding.program->maximum_rendered_utf8_bytes
                    )
                );
                renderer::HostTemplateResultV1 result{};
                renderer::HostTemplateRequestV1 request{};
                request.abi_version = renderer::kHostTemplateRendererAbiVersionV1;
                request.mode = static_cast<std::uint32_t>(
                    renderer::HostTemplateRenderModeV1::kRenderAndHash
                );
                request.program = binding.program;
                request.token_values = binding.token_values;
                request.token_value_count = binding.token_value_count;
                request.output_utf8 = output.data();
                request.output_capacity = output.size();
                request.result = &result;
                renderer::render_and_hash_host_template_callback_v1(&request);
                if (result.status != static_cast<std::uint32_t>(
                        renderer::HostTemplateStatusV1::kSuccess
                    )) {
                    throw std::runtime_error("callback renderer rejected prepared data");
                }
                rendered_bytes += result.rendered_utf8_bytes;
            }
        }
        std::cout
            << "status=success programs=12 callbacks=12 rendered_bytes="
            << rendered_bytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
