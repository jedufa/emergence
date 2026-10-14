#ifndef EIEL_CUDA_CANONICAL_HEREDITARY_DIGEST_V1_CUH_
#define EIEL_CUDA_CANONICAL_HEREDITARY_DIGEST_V1_CUH_

#include "cuda_architecture_b_sha256_v1.cuh"

#include <cstdint>

#if !defined(__CUDACC__)
#error "cuda_canonical_hereditary_digest_v1.cuh requires nvcc"
#endif

namespace eiel::cuda_architecture_b::canonical_hereditary_digest_v1 {

namespace sha = eiel::cuda_architecture_b::sha256_v1;

inline constexpr int32_t kLocusCount = 17;
inline constexpr int32_t kGeneFieldCount = 25;
inline constexpr uint8_t kTokenPrefix = 0x1e;
inline constexpr uint8_t kTokenSuffix = 0x1f;

struct DynamicGenomeV1 {
    const float* allele_0;
    const float* allele_1;
    const float* phenotype;
    int32_t hidden_size;
    int32_t memory_dim;
    int32_t sensor_channel_mask;
    int32_t sensor_radius;
    int32_t introspection_channel_mask;
    int32_t self_regulation_control_mask;
    int32_t self_modification_target_mask;
    int32_t directed_inheritance_channel_mask;
    bool use_gru;
};

__device__ __forceinline__ void update_byte(
    sha::Sha256ContextV1* context,
    const uint8_t value
) {
    sha::sha256_update_v1(context, &value, 1);
}

__device__ __forceinline__ void update_literal(
    sha::Sha256ContextV1* context,
    const char* value,
    const int32_t size
) {
    sha::sha256_update_v1(
        context,
        reinterpret_cast<const uint8_t*>(value),
        static_cast<uint64_t>(size)
    );
}

__device__ __forceinline__ void update_unsigned_decimal(
    sha::Sha256ContextV1* context,
    uint64_t value
) {
    char buffer[24];
    int32_t cursor = 24;
    do {
        buffer[--cursor] = static_cast<char>('0' + value % 10);
        value /= 10;
    } while (value != 0);
    update_literal(context, buffer + cursor, 24 - cursor);
}

__device__ __forceinline__ void update_signed_decimal(
    sha::Sha256ContextV1* context,
    const int64_t value
) {
    if (value < 0) {
        update_byte(context, static_cast<uint8_t>('-'));
        update_unsigned_decimal(
            context,
            static_cast<uint64_t>(-(value + 1)) + 1
        );
        return;
    }
    update_unsigned_decimal(context, static_cast<uint64_t>(value));
}

__device__ __forceinline__ int32_t decimal_digits(uint64_t value) {
    int32_t digits = 1;
    while (value >= 10) {
        value /= 10;
        digits += 1;
    }
    return digits;
}

__device__ __forceinline__ void update_canonical_float(
    sha::Sha256ContextV1* context,
    const float raw
) {
    const bool negative = ::signbit(raw);
    const double magnitude =
        negative ? -static_cast<double>(raw) : static_cast<double>(raw);
    const uint64_t scaled = static_cast<uint64_t>(
        __double2ll_rn(magnitude * 100000000.0)
    );
    if (negative) {
        update_byte(context, static_cast<uint8_t>('-'));
    }
    if (scaled == 0) {
        update_literal(context, "0.0", 3);
        return;
    }
    if (scaled < 10000) {
        const int32_t digits = decimal_digits(scaled);
        char raw_digits[8];
        uint64_t remaining = scaled;
        for (int32_t index = digits - 1; index >= 0; --index) {
            raw_digits[index] = static_cast<char>('0' + remaining % 10);
            remaining /= 10;
        }
        update_byte(context, static_cast<uint8_t>(raw_digits[0]));
        int32_t tail = digits;
        while (tail > 1 && raw_digits[tail - 1] == '0') {
            tail -= 1;
        }
        if (tail > 1) {
            update_byte(context, static_cast<uint8_t>('.'));
            update_literal(context, raw_digits + 1, tail - 1);
        }
        update_literal(context, "e-", 2);
        const int32_t exponent = 9 - digits;
        if (exponent < 10) {
            update_byte(context, static_cast<uint8_t>('0'));
        }
        update_unsigned_decimal(
            context,
            static_cast<uint64_t>(exponent)
        );
        return;
    }
    const uint64_t integer = scaled / 100000000ULL;
    uint64_t fraction = scaled % 100000000ULL;
    update_unsigned_decimal(context, integer);
    update_byte(context, static_cast<uint8_t>('.'));
    if (fraction == 0) {
        update_byte(context, static_cast<uint8_t>('0'));
        return;
    }
    char fraction_digits[8];
    for (int32_t index = 7; index >= 0; --index) {
        fraction_digits[index] =
            static_cast<char>('0' + fraction % 10);
        fraction /= 10;
    }
    int32_t tail = 8;
    while (tail > 1 && fraction_digits[tail - 1] == '0') {
        tail -= 1;
    }
    update_literal(context, fraction_digits, tail);
}

__device__ __forceinline__ void update_digest_json_string(
    sha::Sha256ContextV1* context,
    const uint8_t digest[32]
) {
    static constexpr char kHex[] = "0123456789abcdef";
    update_byte(context, static_cast<uint8_t>('"'));
    char encoded[64];
#pragma unroll
    for (int32_t index = 0; index < 32; ++index) {
        encoded[2 * index] = kHex[digest[index] >> 4];
        encoded[2 * index + 1] = kHex[digest[index] & 15];
    }
    update_literal(context, encoded, 64);
    update_byte(context, static_cast<uint8_t>('"'));
}

template <typename TokenWriter>
__device__ __forceinline__ void render_template(
    sha::Sha256ContextV1* context,
    const uint8_t* template_bytes,
    const int64_t template_size,
    TokenWriter write_token
) {
    int64_t literal_start = 0;
    int64_t cursor = 0;
    while (cursor < template_size) {
        if (
            cursor + 2 < template_size
            && template_bytes[cursor] == kTokenPrefix
            && template_bytes[cursor + 2] == kTokenSuffix
        ) {
            sha::sha256_update_v1(
                context,
                template_bytes + literal_start,
                static_cast<uint64_t>(cursor - literal_start)
            );
            write_token(context, template_bytes[cursor + 1]);
            cursor += 3;
            literal_start = cursor;
            continue;
        }
        cursor += 1;
    }
    sha::sha256_update_v1(
        context,
        template_bytes + literal_start,
        static_cast<uint64_t>(template_size - literal_start)
    );
}

__device__ __forceinline__ float gene_float(
    const DynamicGenomeV1& genome,
    const uint8_t token
) {
    switch (token) {
        case 1: return genome.phenotype[2];
        case 2: return genome.phenotype[1];
        case 4: return genome.phenotype[13];
        case 5: return genome.phenotype[3];
        case 6: return genome.phenotype[5];
        case 7: return genome.phenotype[4];
        case 8: return genome.phenotype[9];
        case 9: return genome.phenotype[10];
        case 10: return genome.phenotype[7];
        case 11: return genome.phenotype[8];
        case 12: return genome.phenotype[12];
        case 13: return genome.phenotype[11];
        case 14: return genome.phenotype[0];
        case 18: return genome.phenotype[15];
        case 19: return genome.phenotype[6];
        case 20: return genome.phenotype[14];
        default: return 0.0f;
    }
}

__device__ __forceinline__ void compute_v1(
    const uint8_t* canonical_template,
    const int64_t canonical_template_bytes,
    const uint8_t* diploid_payload_template,
    const int64_t diploid_payload_template_bytes,
    const uint8_t* genotype_template,
    const int64_t genotype_template_bytes,
    const DynamicGenomeV1& genome,
    uint8_t output[32]
) {
    uint8_t genotype_digest[32];
    sha::Sha256ContextV1 genotype_context;
    sha::sha256_init_v1(&genotype_context);
    render_template(
        &genotype_context,
        genotype_template,
        genotype_template_bytes,
        [&](sha::Sha256ContextV1* context, const uint8_t token) {
            const float value = token < kLocusCount
                ? genome.allele_0[token]
                : genome.allele_1[token - kLocusCount];
            update_canonical_float(context, value);
        }
    );
    sha::sha256_final_v1(&genotype_context, genotype_digest);

    auto write_diploid = [&](sha::Sha256ContextV1* context) {
        render_template(
            context,
            diploid_payload_template,
            diploid_payload_template_bytes,
            [&](sha::Sha256ContextV1* nested, const uint8_t token) {
                if (token < 2 * kLocusCount) {
                    const float value = token < kLocusCount
                        ? genome.allele_0[token]
                        : genome.allele_1[token - kLocusCount];
                    update_canonical_float(nested, value);
                } else {
                    update_digest_json_string(nested, genotype_digest);
                }
            }
        );
    };

    sha::Sha256ContextV1 canonical_context;
    sha::sha256_init_v1(&canonical_context);
    render_template(
        &canonical_context,
        canonical_template,
        canonical_template_bytes,
        [&](sha::Sha256ContextV1* context, const uint8_t token) {
            if (token == 0) {
                write_diploid(context);
            } else if (
                token == 3 || token == 15 || token == 16 || token == 17
                || (token >= 21 && token <= 24)
            ) {
                int32_t value = 0;
                switch (token) {
                    case 3:
                        value = genome.directed_inheritance_channel_mask;
                        break;
                    case 15: value = genome.hidden_size; break;
                    case 16:
                        value = genome.introspection_channel_mask;
                        break;
                    case 17: value = genome.memory_dim; break;
                    case 21:
                        value = genome.self_modification_target_mask;
                        break;
                    case 22:
                        value = genome.self_regulation_control_mask;
                        break;
                    case 23: value = genome.sensor_channel_mask; break;
                    case 24: value = genome.sensor_radius; break;
                }
                update_signed_decimal(context, value);
            } else if (token == 25) {
                update_literal(
                    context,
                    genome.use_gru ? "true" : "false",
                    genome.use_gru ? 4 : 5
                );
            } else {
                update_canonical_float(context, gene_float(genome, token));
            }
        }
    );
    sha::sha256_final_v1(&canonical_context, output);
}

}  // namespace eiel::cuda_architecture_b::canonical_hereditary_digest_v1

#endif
