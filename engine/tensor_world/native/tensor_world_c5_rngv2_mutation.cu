#include "../../tw_rngv2/native/tw_rngv2.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kBernoulli = 0x0330u;
constexpr uint32_t kNormal = 0x0331u;
constexpr uint32_t kDiscrete = 0x0332u;

__device__ __forceinline__ double clamp_d(double value, double lo, double hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__global__ void mutate_kernel(
    const uint32_t* world,
    const uint64_t* tick,
    const uint64_t* subject,
    const double* parent_genes,
    const double* mate_genes,
    const double* parent_rate,
    const double* mate_rate,
    uint64_t namespace_key,
    uint32_t semantic_epoch,
    int64_t count,
    double* child_genes,
    double* child_rate,
    int32_t* retries,
    int32_t* status
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const double specs[8][3] = {
        {0.10, -1.0, 1.0}, {0.15, -2.0, 2.0}, {2.0, 2.0, 90.0},
        {0.10, 0.25, 2.5}, {0.05, 0.0, 0.999}, {0.01, 0.0, 0.10},
        {0.20, 0.0, 3.0}, {0.03, 0.0, 1.0},
    };
    const uint32_t world_uid = world[index];
    const uint64_t absolute_tick = tick[index];
    const uint64_t subject_uid = subject[index];
    const double rate = clamp_d(0.5 * (parent_rate[index] + mate_rate[index]), 0.0, 1.0);
    for (int field = 0; field < 8; ++field) {
        const double mid = 0.5 * (parent_genes[index * 8 + field] + mate_genes[index * 8 + field]);
        int32_t field_status = tw_rngv2::validate_fields(
            world_uid, kBernoulli, semantic_epoch, subject_uid, absolute_tick,
            static_cast<uint32_t>(field), 0
        );
        if (field_status != tw_rngv2::RNG_OK) {
            status[index] = field_status;
            return;
        }
        uint32_t k0 = 0;
        uint32_t k1 = 0;
        uint32_t c0 = 0;
        uint32_t c1 = 0;
        uint32_t c2 = 0;
        uint32_t c3 = 0;
        tw_rngv2::encode_key(namespace_key, world_uid, kBernoulli, semantic_epoch, &k0, &k1);
        tw_rngv2::encode_counter(
            subject_uid, absolute_tick, static_cast<uint32_t>(field), 0, &c0, &c1, &c2, &c3
        );
        const tw_rngv2::PhiloxWords bernoulli = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
        const uint64_t u_bits = static_cast<uint64_t>(bernoulli.w0) | (static_cast<uint64_t>(bernoulli.w1) << 32);
        const double uniform = tw_rngv2::u64_to_f64_half_open(u_bits);
        double value = mid;
        if (uniform < rate) {
            tw_rngv2::encode_key(namespace_key, world_uid, kNormal, semantic_epoch, &k0, &k1);
            tw_rngv2::encode_counter(
                subject_uid, absolute_tick, static_cast<uint32_t>(field), 0, &c0, &c1, &c2, &c3
            );
            const tw_rngv2::PhiloxWords normal = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
            uint64_t z0_bits = 0;
            uint64_t z1_bits = 0;
            int32_t nstatus = tw_rngv2::RNG_OK;
            tw_rngv2::box_muller(normal, &z0_bits, &z1_bits, &nstatus);
            if (nstatus != tw_rngv2::RNG_OK) {
                status[index] = nstatus;
                return;
            }
            value = mid + specs[field][0] * tw_rngv2::f64_from_bits(z0_bits);
        }
        child_genes[index * 8 + field] = clamp_d(value, specs[field][1], specs[field][2]);
        uint32_t bounded = 0;
        int32_t accepted = 0;
        int32_t bstatus = tw_rngv2::RNG_OK;
        tw_rngv2::bounded_uint32(
            7ull, namespace_key, world_uid, kDiscrete, semantic_epoch,
            subject_uid, absolute_tick, static_cast<uint32_t>(field),
            &bounded, &accepted, &bstatus
        );
        if (bstatus != tw_rngv2::RNG_OK) {
            status[index] = bstatus;
            return;
        }
        retries[index * 8 + field] = accepted;
    }
    uint32_t k0 = 0;
    uint32_t k1 = 0;
    uint32_t c0 = 0;
    uint32_t c1 = 0;
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    tw_rngv2::encode_key(namespace_key, world_uid, kBernoulli, semantic_epoch, &k0, &k1);
    tw_rngv2::encode_counter(subject_uid, absolute_tick, 8u, 0, &c0, &c1, &c2, &c3);
    const tw_rngv2::PhiloxWords rate_words = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
    const uint64_t rate_bits = static_cast<uint64_t>(rate_words.w0) | (static_cast<uint64_t>(rate_words.w1) << 32);
    double out_rate = rate;
    if (tw_rngv2::u64_to_f64_half_open(rate_bits) < 0.2) {
        tw_rngv2::encode_key(namespace_key, world_uid, kNormal, semantic_epoch, &k0, &k1);
        tw_rngv2::encode_counter(subject_uid, absolute_tick, 8u, 0, &c0, &c1, &c2, &c3);
        const tw_rngv2::PhiloxWords normal = tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
        uint64_t z0_bits = 0;
        uint64_t z1_bits = 0;
        int32_t nstatus = tw_rngv2::RNG_OK;
        tw_rngv2::box_muller(normal, &z0_bits, &z1_bits, &nstatus);
        if (nstatus != tw_rngv2::RNG_OK) {
            status[index] = nstatus;
            return;
        }
        out_rate = clamp_d(rate + 0.005 * tw_rngv2::f64_from_bits(z0_bits), 0.0, 1.0);
    }
    child_rate[index] = out_rate;
    status[index] = tw_rngv2::RNG_OK;
}

}  // namespace

std::vector<torch::Tensor> tensor_world_c5_rngv2_mutate_cuda(
    torch::Tensor world,
    torch::Tensor tick,
    torch::Tensor subject,
    torch::Tensor parent_genes,
    torch::Tensor mate_genes,
    torch::Tensor parent_rate,
    torch::Tensor mate_rate,
    uint64_t namespace_key,
    int64_t semantic_epoch
) {
    const int64_t count = world.size(0);
    auto child_genes = torch::empty({count, 8}, parent_genes.options());
    auto child_rate = torch::empty({count}, parent_rate.options());
    auto retries = torch::empty({count, 8}, world.options().dtype(torch::kInt32));
    auto status = torch::empty({count}, world.options().dtype(torch::kInt32));
    const int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (count > 0) {
        mutate_kernel<<<blocks, threads>>>(
            world.data_ptr<uint32_t>(),
            tick.data_ptr<uint64_t>(),
            subject.data_ptr<uint64_t>(),
            parent_genes.data_ptr<double>(),
            mate_genes.data_ptr<double>(),
            parent_rate.data_ptr<double>(),
            mate_rate.data_ptr<double>(),
            namespace_key,
            static_cast<uint32_t>(semantic_epoch),
            count,
            child_genes.data_ptr<double>(),
            child_rate.data_ptr<double>(),
            retries.data_ptr<int32_t>(),
            status.data_ptr<int32_t>()
        );
    }
    return {child_genes, child_rate, retries, status};
}
