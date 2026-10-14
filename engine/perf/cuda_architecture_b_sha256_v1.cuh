#ifndef EIEL_CUDA_ARCHITECTURE_B_SHA256_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_SHA256_V1_CUH_

#include <cstdint>

#if !defined(__CUDACC__)
#error "cuda_architecture_b_sha256_v1.cuh must be compiled by nvcc"
#endif

#define EIEL_SHA256_V1_DEVICE_INLINE __device__ __forceinline__

namespace eiel::cuda_architecture_b::sha256_v1 {

inline constexpr std::uint32_t kBlockBytesV1 = 64;
inline constexpr std::uint32_t kDigestBytesV1 = 32;

struct Sha256ContextV1 {
    std::uint64_t total_bytes;
    std::uint32_t state[8];
    std::uint32_t buffered_bytes;
    std::uint8_t buffer[kBlockBytesV1];
};

namespace detail {

// Internal linkage keeps this header self-contained when several CUDA
// translation units instantiate the primitive.
static __device__ __constant__ const std::uint32_t kRoundConstantsV1[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

EIEL_SHA256_V1_DEVICE_INLINE std::uint32_t rotate_right(
    const std::uint32_t value,
    const std::uint32_t distance
) {
    return (value >> distance) | (value << (32U - distance));
}

EIEL_SHA256_V1_DEVICE_INLINE std::uint32_t load_big_endian_u32(
    const std::uint8_t* bytes
) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U)
        | (static_cast<std::uint32_t>(bytes[1]) << 16U)
        | (static_cast<std::uint32_t>(bytes[2]) << 8U)
        | static_cast<std::uint32_t>(bytes[3]);
}

EIEL_SHA256_V1_DEVICE_INLINE void store_big_endian_u32(
    std::uint8_t* bytes,
    const std::uint32_t value
) {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

EIEL_SHA256_V1_DEVICE_INLINE void transform_block(
    Sha256ContextV1* context,
    const std::uint8_t* block
) {
    std::uint32_t words[16];
#pragma unroll
    for (std::uint32_t index = 0; index < 16; ++index) {
        words[index] = load_big_endian_u32(block + 4U * index);
    }

    std::uint32_t a = context->state[0];
    std::uint32_t b = context->state[1];
    std::uint32_t c = context->state[2];
    std::uint32_t d = context->state[3];
    std::uint32_t e = context->state[4];
    std::uint32_t f = context->state[5];
    std::uint32_t g = context->state[6];
    std::uint32_t h = context->state[7];

#pragma unroll
    for (std::uint32_t round = 0; round < 64; ++round) {
        const std::uint32_t word_index = round & 15U;
        if (round >= 16U) {
            const std::uint32_t word_minus_15 = words[(round + 1U) & 15U];
            const std::uint32_t word_minus_2 = words[(round + 14U) & 15U];
            const std::uint32_t sigma0 = rotate_right(word_minus_15, 7U)
                ^ rotate_right(word_minus_15, 18U)
                ^ (word_minus_15 >> 3U);
            const std::uint32_t sigma1 = rotate_right(word_minus_2, 17U)
                ^ rotate_right(word_minus_2, 19U)
                ^ (word_minus_2 >> 10U);
            words[word_index] = words[word_index]
                + sigma0
                + words[(round + 9U) & 15U]
                + sigma1;
        }

        const std::uint32_t sum1 = rotate_right(e, 6U)
            ^ rotate_right(e, 11U)
            ^ rotate_right(e, 25U);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temporary1 = h
            + sum1
            + choose
            + kRoundConstantsV1[round]
            + words[word_index];
        const std::uint32_t sum0 = rotate_right(a, 2U)
            ^ rotate_right(a, 13U)
            ^ rotate_right(a, 22U);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

}  // namespace detail

EIEL_SHA256_V1_DEVICE_INLINE void sha256_init_v1(Sha256ContextV1* context) {
    context->total_bytes = 0;
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
    context->buffered_bytes = 0;
}

EIEL_SHA256_V1_DEVICE_INLINE void sha256_update_v1(
    Sha256ContextV1* context,
    const std::uint8_t* input,
    std::uint64_t input_bytes
) {
    context->total_bytes += input_bytes;
    if (input_bytes == 0) {
        return;
    }

    if (context->buffered_bytes != 0U) {
        const std::uint32_t available = kBlockBytesV1 - context->buffered_bytes;
        const std::uint32_t copied = input_bytes < available
            ? static_cast<std::uint32_t>(input_bytes)
            : available;
        for (std::uint32_t index = 0; index < copied; ++index) {
            context->buffer[context->buffered_bytes + index] = input[index];
        }
        context->buffered_bytes += copied;
        input += copied;
        input_bytes -= copied;
        if (context->buffered_bytes == kBlockBytesV1) {
            detail::transform_block(context, context->buffer);
            context->buffered_bytes = 0;
        }
        if (input_bytes == 0) {
            return;
        }
    }

    while (input_bytes >= kBlockBytesV1) {
        detail::transform_block(context, input);
        input += kBlockBytesV1;
        input_bytes -= kBlockBytesV1;
    }

    const std::uint32_t tail_bytes = static_cast<std::uint32_t>(input_bytes);
    for (std::uint32_t index = 0; index < tail_bytes; ++index) {
        context->buffer[index] = input[index];
    }
    context->buffered_bytes = tail_bytes;
}

EIEL_SHA256_V1_DEVICE_INLINE void sha256_final_v1(
    Sha256ContextV1* context,
    std::uint8_t output[kDigestBytesV1]
) {
    const std::uint64_t message_bits = context->total_bytes * 8U;
    std::uint32_t cursor = context->buffered_bytes;
    context->buffer[cursor++] = 0x80U;

    if (cursor > 56U) {
        while (cursor < kBlockBytesV1) {
            context->buffer[cursor++] = 0;
        }
        detail::transform_block(context, context->buffer);
        cursor = 0;
    }
    while (cursor < 56U) {
        context->buffer[cursor++] = 0;
    }
    for (std::uint32_t index = 0; index < 8; ++index) {
        context->buffer[56U + index] = static_cast<std::uint8_t>(
            message_bits >> (56U - 8U * index)
        );
    }
    detail::transform_block(context, context->buffer);
    context->buffered_bytes = 0;

#pragma unroll
    for (std::uint32_t index = 0; index < 8; ++index) {
        detail::store_big_endian_u32(output + 4U * index, context->state[index]);
    }
}

EIEL_SHA256_V1_DEVICE_INLINE void sha256_digest_v1(
    const std::uint8_t* input,
    const std::uint64_t input_bytes,
    std::uint8_t output[kDigestBytesV1]
) {
    Sha256ContextV1 context;
    sha256_init_v1(&context);
    sha256_update_v1(&context, input, input_bytes);
    sha256_final_v1(&context, output);
}

}  // namespace eiel::cuda_architecture_b::sha256_v1

#undef EIEL_SHA256_V1_DEVICE_INLINE

#endif
