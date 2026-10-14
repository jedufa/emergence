#include "tw_rngv2.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

__global__ void generate_kernel(
    const uint32_t* world,
    const uint32_t* domain,
    const uint32_t* epoch,
    const uint64_t* subject,
    const uint64_t* tick,
    const uint32_t* operation,
    const uint64_t* n_values,
    uint64_t namespace_key,
    int64_t count,
    uint32_t* w0,
    uint32_t* w1,
    uint32_t* w2,
    uint32_t* w3,
    uint32_t* bounded,
    int32_t* accepted,
    uint32_t* u32_bits,
    uint64_t* u64_bits,
    uint64_t* z0_bits,
    uint64_t* z1_bits,
    uint64_t* mix,
    int32_t* status
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const uint32_t world_uid = world[index];
    const uint32_t domain_id = domain[index];
    const uint32_t semantic_epoch = epoch[index];
    const uint64_t subject_uid = subject[index];
    const uint64_t absolute_tick = tick[index];
    const uint32_t operation_id = operation[index];
    int32_t field_status = tw_rngv2::validate_fields(
        world_uid,
        domain_id,
        semantic_epoch,
        subject_uid,
        absolute_tick,
        operation_id,
        0
    );
    if (field_status != tw_rngv2::RNG_OK) {
        w0[index] = 0;
        w1[index] = 0;
        w2[index] = 0;
        w3[index] = 0;
        bounded[index] = 0;
        accepted[index] = 0;
        u32_bits[index] = 0;
        u64_bits[index] = 0;
        z0_bits[index] = 0;
        z1_bits[index] = 0;
        mix[index] = 0;
        status[index] = field_status;
        return;
    }
    uint32_t k0 = 0;
    uint32_t k1 = 0;
    uint32_t c0 = 0;
    uint32_t c1 = 0;
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    tw_rngv2::encode_key(namespace_key, world_uid, domain_id, semantic_epoch, &k0, &k1);
    tw_rngv2::encode_counter(subject_uid, absolute_tick, operation_id, 0, &c0, &c1, &c2, &c3);
    const tw_rngv2::PhiloxWords words = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
    w0[index] = words.w0;
    w1[index] = words.w1;
    w2[index] = words.w2;
    w3[index] = words.w3;
    uint32_t bounded_value = 0;
    int32_t accepted_ordinal = 0;
    int32_t bounded_status = tw_rngv2::RNG_OK;
    tw_rngv2::bounded_uint32(
        n_values[index],
        namespace_key,
        world_uid,
        domain_id,
        semantic_epoch,
        subject_uid,
        absolute_tick,
        operation_id,
        &bounded_value,
        &accepted_ordinal,
        &bounded_status
    );
    bounded[index] = bounded_value;
    accepted[index] = accepted_ordinal;
    const float u32 = tw_rngv2::u32_to_f32_half_open(words.w0);
    u32_bits[index] = __float_as_uint(u32);
    const uint64_t packed_u64 = static_cast<uint64_t>(words.w0) | (static_cast<uint64_t>(words.w1) << 32);
    u64_bits[index] = tw_rngv2::f64_bits(tw_rngv2::u64_to_f64_half_open(packed_u64));
    uint64_t z0 = 0;
    uint64_t z1 = 0;
    int32_t normal_status = tw_rngv2::RNG_OK;
    tw_rngv2::box_muller(words, &z0, &z1, &normal_status);
    z0_bits[index] = z0;
    z1_bits[index] = z1;
    mix[index] = tw_rngv2::mix64(z0 ^ (z1 << 1) ^ packed_u64 ^ bounded_value);
    status[index] = (bounded_status != tw_rngv2::RNG_OK) ? bounded_status : normal_status;
}

}  // namespace

std::vector<torch::Tensor> tw_rngv2_generate_cuda(
    torch::Tensor world,
    torch::Tensor domain,
    torch::Tensor epoch,
    torch::Tensor subject,
    torch::Tensor tick,
    torch::Tensor operation,
    torch::Tensor n_values,
    uint64_t namespace_key
) {
    const int64_t count = world.size(0);
    auto opts32 = world.options().dtype(torch::kUInt32);
    auto opts64 = world.options().dtype(torch::kUInt64);
    auto optsi = world.options().dtype(torch::kInt32);
    auto w0 = torch::empty({count}, opts32);
    auto w1 = torch::empty({count}, opts32);
    auto w2 = torch::empty({count}, opts32);
    auto w3 = torch::empty({count}, opts32);
    auto bounded = torch::empty({count}, opts32);
    auto accepted = torch::empty({count}, optsi);
    auto u32_bits = torch::empty({count}, opts32);
    auto u64_bits = torch::empty({count}, opts64);
    auto z0 = torch::empty({count}, opts64);
    auto z1 = torch::empty({count}, opts64);
    auto mix = torch::empty({count}, opts64);
    auto status = torch::empty({count}, optsi);
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (count > 0) {
        generate_kernel<<<blocks, threads>>>(
            world.data_ptr<uint32_t>(),
            domain.data_ptr<uint32_t>(),
            epoch.data_ptr<uint32_t>(),
            subject.data_ptr<uint64_t>(),
            tick.data_ptr<uint64_t>(),
            operation.data_ptr<uint32_t>(),
            n_values.data_ptr<uint64_t>(),
            namespace_key,
            count,
            w0.data_ptr<uint32_t>(),
            w1.data_ptr<uint32_t>(),
            w2.data_ptr<uint32_t>(),
            w3.data_ptr<uint32_t>(),
            bounded.data_ptr<uint32_t>(),
            accepted.data_ptr<int32_t>(),
            u32_bits.data_ptr<uint32_t>(),
            u64_bits.data_ptr<uint64_t>(),
            z0.data_ptr<uint64_t>(),
            z1.data_ptr<uint64_t>(),
            mix.data_ptr<uint64_t>(),
            status.data_ptr<int32_t>()
        );
    }
    return {w0, w1, w2, w3, bounded, accepted, u32_bits, u64_bits, z0, z1, mix, status};
}

__global__ void philox_kernel(
    const uint32_t* counters,
    const uint32_t* keys,
    int64_t rounds,
    int64_t count,
    uint32_t* output
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const auto words = tw_rngv2::philox4x32_rounds(
        counters[index * 4 + 0],
        counters[index * 4 + 1],
        counters[index * 4 + 2],
        counters[index * 4 + 3],
        keys[index * 2 + 0],
        keys[index * 2 + 1],
        static_cast<int>(rounds)
    );
    output[index * 4 + 0] = words.w0;
    output[index * 4 + 1] = words.w1;
    output[index * 4 + 2] = words.w2;
    output[index * 4 + 3] = words.w3;
}

torch::Tensor tw_rngv2_philox_cuda(
    torch::Tensor counters,
    torch::Tensor keys,
    int64_t rounds
) {
    auto output = torch::empty_like(counters);
    const int64_t count = counters.size(0);
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (count > 0) {
        philox_kernel<<<blocks, threads>>>(
            counters.data_ptr<uint32_t>(),
            keys.data_ptr<uint32_t>(),
            rounds,
            count,
            output.data_ptr<uint32_t>()
        );
    }
    return output;
}
