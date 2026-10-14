#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "glibc_2_43_fma_math.cuh"


namespace {

constexpr int32_t kStateWords = 624;
constexpr int32_t kM = 397;
constexpr uint32_t kMatrixA = 0x9908b0dfU;
constexpr uint32_t kUpperMask = 0x80000000U;
constexpr uint32_t kLowerMask = 0x7fffffffU;
constexpr double kTwopi = 0x1.921fb54442d18p+2;

constexpr int32_t kRaw32 = 1;
constexpr int32_t kRandom = 2;
constexpr int32_t kGetrandbits = 3;
constexpr int32_t kRandbelow = 4;
constexpr int32_t kGauss = 5;
constexpr int32_t kPredicateRandomLt = 1;

constexpr int32_t kStatusOk = 0;
constexpr int32_t kStatusInvalidOpcode = 1;
constexpr int32_t kStatusInvalidGetrandbits = 2;
constexpr int32_t kStatusInvalidRandbelow = 3;
constexpr int32_t kStatusGaussDisabled = 4;
constexpr int32_t kStatusInvalidGaussianStrategy = 5;
constexpr int32_t kStatusCounterOverflow = 6;
constexpr int32_t kStatusInvalidPredicate = 7;
constexpr int32_t kStatusInvalidPredicateSource = 8;
constexpr int32_t kStatusTapeExhausted = 9;

constexpr int32_t kCounterRawWords = 0;
constexpr int32_t kCounterSemanticDraws = 1;
constexpr int32_t kCounterRandomCalls = 2;
constexpr int32_t kCounterGetrandbitsCalls = 3;
constexpr int32_t kCounterRandbelowRejections = 4;
constexpr int32_t kCounterGaussCalls = 5;
constexpr int32_t kCounterGaussCacheHits = 6;
constexpr int32_t kCounterGaussCacheMisses = 7;
constexpr int32_t kCounterTwists = 8;
constexpr int32_t kCounterCount = 9;


__device__ __forceinline__ bool increment_counter(
    int64_t* counters,
    const int32_t index,
    const int64_t amount = 1
) {
    if (amount < 0 || counters[index] > INT64_MAX - amount) {
        return false;
    }
    counters[index] += amount;
    return true;
}


struct CpythonMt19937 {
    uint32_t* words;
    int64_t index;
    int64_t* counters;
    int32_t status = kStatusOk;

    __device__ __forceinline__ void twist() {
        for (int32_t offset = 0; offset < kStateWords - kM; ++offset) {
            const uint32_t mixed =
                (words[offset] & kUpperMask)
                | (words[offset + 1] & kLowerMask);
            words[offset] = words[offset + kM]
                ^ (mixed >> 1U)
                ^ ((mixed & 1U) != 0U ? kMatrixA : 0U);
        }
        for (int32_t offset = kStateWords - kM; offset < kStateWords - 1; ++offset) {
            const uint32_t mixed =
                (words[offset] & kUpperMask)
                | (words[offset + 1] & kLowerMask);
            words[offset] = words[offset + (kM - kStateWords)]
                ^ (mixed >> 1U)
                ^ ((mixed & 1U) != 0U ? kMatrixA : 0U);
        }
        const uint32_t mixed =
            (words[kStateWords - 1] & kUpperMask)
            | (words[0] & kLowerMask);
        words[kStateWords - 1] = words[kM - 1]
            ^ (mixed >> 1U)
            ^ ((mixed & 1U) != 0U ? kMatrixA : 0U);
        index = 0;
        if (!increment_counter(counters, kCounterTwists)) {
            status = kStatusCounterOverflow;
        }
    }

    __device__ __forceinline__ uint32_t raw32() {
        if (index >= kStateWords) {
            twist();
        }
        if (status != kStatusOk) {
            return 0;
        }
        uint32_t value = words[index++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        value ^= value >> 18U;
        if (!increment_counter(counters, kCounterRawWords)) {
            status = kStatusCounterOverflow;
            return 0;
        }
        return value;
    }

    __device__ __forceinline__ double random_double() {
        const uint32_t first = raw32() >> 5U;
        const uint32_t second = raw32() >> 6U;
        if (!increment_counter(counters, kCounterRandomCalls)) {
            status = kStatusCounterOverflow;
            return 0.0;
        }
        const double numerator =
            static_cast<double>(first) * 67108864.0
            + static_cast<double>(second);
        return numerator * (1.0 / 9007199254740992.0);
    }

    __device__ __forceinline__ uint64_t getrandbits(const int64_t bit_count) {
        if (!increment_counter(counters, kCounterGetrandbitsCalls)) {
            status = kStatusCounterOverflow;
            return 0;
        }
        if (bit_count == 0) {
            return 0;
        }
        if (bit_count < 0 || bit_count > 64) {
            status = kStatusInvalidGetrandbits;
            return 0;
        }
        if (bit_count <= 32) {
            return static_cast<uint64_t>(raw32() >> (32 - bit_count));
        }
        const uint64_t low = static_cast<uint64_t>(raw32());
        const uint64_t high = static_cast<uint64_t>(
            raw32() >> (64 - bit_count)
        );
        return low | (high << 32U);
    }

    __device__ __forceinline__ uint64_t randbelow(const int64_t upper) {
        if (upper <= 0) {
            status = kStatusInvalidRandbelow;
            return 0;
        }
        int64_t bit_count = 0;
        for (uint64_t value = static_cast<uint64_t>(upper); value != 0; value >>= 1U) {
            ++bit_count;
        }
        int64_t rejected = 0;
        uint64_t value = getrandbits(bit_count);
        while (status == kStatusOk && value >= static_cast<uint64_t>(upper)) {
            ++rejected;
            value = getrandbits(bit_count);
        }
        if (
            rejected > 0
            && !increment_counter(counters, kCounterRandbelowRejections, rejected)
        ) {
            status = kStatusCounterOverflow;
            return 0;
        }
        return value;
    }
};


__device__ __forceinline__ uint64_t double_bits(const double value) {
    return __double_as_longlong(value);
}


__device__ __forceinline__ double bits_double(const uint64_t value) {
    return __longlong_as_double(value);
}


__device__ __forceinline__ double gauss_libdevice(
    CpythonMt19937* engine,
    bool* has_gauss,
    double* gauss_next,
    const double mu,
    const double sigma
) {
    if (!increment_counter(engine->counters, kCounterGaussCalls)) {
        engine->status = kStatusCounterOverflow;
        return 0.0;
    }
    double z;
    if (*has_gauss) {
        if (!increment_counter(engine->counters, kCounterGaussCacheHits)) {
            engine->status = kStatusCounterOverflow;
            return 0.0;
        }
        z = *gauss_next;
        *has_gauss = false;
        *gauss_next = 0.0;
    } else {
        if (!increment_counter(engine->counters, kCounterGaussCacheMisses)) {
            engine->status = kStatusCounterOverflow;
            return 0.0;
        }
        const double x2pi = engine->random_double() * kTwopi;
        const double uniform = engine->random_double();
        const double g2rad = ::sqrt(-2.0 * ::log(1.0 - uniform));
        z = ::cos(x2pi) * g2rad;
        *gauss_next = ::sin(x2pi) * g2rad;
        *has_gauss = true;
    }
    return mu + z * sigma;
}


__device__ __forceinline__ double gauss_glibc_fma_software(
    CpythonMt19937* engine,
    bool* has_gauss,
    double* gauss_next,
    const double mu,
    const double sigma
) {
    if (!increment_counter(engine->counters, kCounterGaussCalls)) {
        engine->status = kStatusCounterOverflow;
        return 0.0;
    }
    double z;
    if (*has_gauss) {
        if (!increment_counter(engine->counters, kCounterGaussCacheHits)) {
            engine->status = kStatusCounterOverflow;
            return 0.0;
        }
        z = *gauss_next;
        *has_gauss = false;
        *gauss_next = 0.0;
    } else {
        if (!increment_counter(engine->counters, kCounterGaussCacheMisses)) {
            engine->status = kStatusCounterOverflow;
            return 0.0;
        }
        const double x2pi = engine->random_double() * kTwopi;
        const double uniform = engine->random_double();
        const double logarithm = option_d_glibc_log_v1(1.0 - uniform);
        const double g2rad = __dsqrt_rn(-2.0 * logarithm);
        z = option_d_glibc_cos_v1(x2pi) * g2rad;
        *gauss_next = option_d_glibc_sin_v1(x2pi) * g2rad;
        *has_gauss = true;
    }
    return mu + z * sigma;
}


__device__ __forceinline__ double gauss_host_tape_baseline(
    CpythonMt19937* engine,
    bool* has_gauss,
    double* gauss_next,
    const double mu,
    const double sigma,
    const double* primary,
    const double* secondary,
    const int64_t tape_base_raw_words,
    const int64_t tape_capacity
) {
    if (!increment_counter(engine->counters, kCounterGaussCalls)) {
        engine->status = kStatusCounterOverflow;
        return 0.0;
    }
    double z;
    if (*has_gauss) {
        if (!increment_counter(engine->counters, kCounterGaussCacheHits)) {
            engine->status = kStatusCounterOverflow;
            return 0.0;
        }
        z = *gauss_next;
        *has_gauss = false;
        *gauss_next = 0.0;
    } else {
        if (!increment_counter(engine->counters, kCounterGaussCacheMisses)) {
            engine->status = kStatusCounterOverflow;
            return 0.0;
        }
        const int64_t tape_index =
            engine->counters[kCounterRawWords] - tape_base_raw_words;
        if (tape_index < 0 || tape_index >= tape_capacity) {
            engine->status = kStatusTapeExhausted;
            return 0.0;
        }
        // The current C5 tape is indexed before the two random() calls that
        // advance MT state for one CPython Gaussian cache miss.
        engine->random_double();
        engine->random_double();
        if (engine->status != kStatusOk) {
            return 0.0;
        }
        z = primary[tape_index];
        *gauss_next = secondary[tape_index];
        *has_gauss = true;
    }
    return mu + z * sigma;
}


__global__ void execute_kernel(
    uint32_t* words,
    int64_t* indices,
    bool* has_gauss,
    double* gauss_next,
    int64_t* counters,
    const int32_t* opcodes,
    const int64_t* arguments,
    const double* mus,
    const double* sigmas,
    const int64_t* predicate_sources,
    const int32_t* predicate_kinds,
    const double* predicate_thresholds,
    const int64_t* operation_counts,
    const int64_t world_count,
    const int64_t operation_capacity,
    const int64_t gaussian_strategy,
    const double* tape_primary,
    const double* tape_secondary,
    const int64_t* tape_base_raw_words,
    const int64_t tape_capacity,
    uint64_t* outputs,
    bool* executed,
    int32_t* statuses,
    int64_t* error_operations
) {
    const int64_t world = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (world >= world_count) {
        return;
    }
    CpythonMt19937 engine{
        words + world * kStateWords,
        indices[world],
        counters + world * kCounterCount,
    };
    bool local_has_gauss = has_gauss[world];
    double local_gauss_next = gauss_next[world];
    const int64_t operation_count = operation_counts[world];
    for (int64_t operation = 0; operation < operation_count; ++operation) {
        const int64_t offset = world * operation_capacity + operation;
        const int32_t opcode = opcodes[offset];
        const int64_t predicate_source = predicate_sources[offset];
        if (predicate_source >= 0) {
            if (
                predicate_kinds[offset] != kPredicateRandomLt
                || predicate_source >= operation
            ) {
                statuses[world] = kStatusInvalidPredicate;
                error_operations[world] = operation;
                return;
            }
            const int64_t source_offset =
                world * operation_capacity + predicate_source;
            if (!executed[source_offset] || opcodes[source_offset] != kRandom) {
                statuses[world] = kStatusInvalidPredicateSource;
                error_operations[world] = operation;
                return;
            }
            if (!(
                bits_double(outputs[source_offset])
                < predicate_thresholds[offset]
            )) {
                outputs[offset] = 0;
                executed[offset] = false;
                continue;
            }
        }
        uint64_t output = 0;
        if (opcode == kRaw32) {
            output = static_cast<uint64_t>(engine.raw32());
        } else if (opcode == kRandom) {
            output = double_bits(engine.random_double());
        } else if (opcode == kGetrandbits) {
            output = engine.getrandbits(arguments[offset]);
        } else if (opcode == kRandbelow) {
            output = engine.randbelow(arguments[offset]);
        } else if (opcode == kGauss) {
            if (gaussian_strategy == 0) {
                engine.status = kStatusGaussDisabled;
            } else if (gaussian_strategy == 1) {
                output = double_bits(gauss_libdevice(
                    &engine,
                    &local_has_gauss,
                    &local_gauss_next,
                    mus[offset],
                    sigmas[offset]
                ));
            } else if (gaussian_strategy == 2) {
                output = double_bits(gauss_glibc_fma_software(
                    &engine,
                    &local_has_gauss,
                    &local_gauss_next,
                    mus[offset],
                    sigmas[offset]
                ));
            } else if (gaussian_strategy == 3) {
                output = double_bits(gauss_host_tape_baseline(
                    &engine,
                    &local_has_gauss,
                    &local_gauss_next,
                    mus[offset],
                    sigmas[offset],
                    tape_primary + world * tape_capacity,
                    tape_secondary + world * tape_capacity,
                    tape_base_raw_words[world],
                    tape_capacity
                ));
            } else {
                engine.status = kStatusInvalidGaussianStrategy;
            }
        } else {
            engine.status = kStatusInvalidOpcode;
        }
        if (engine.status != kStatusOk) {
            statuses[world] = engine.status;
            error_operations[world] = operation;
            return;
        }
        if (!increment_counter(engine.counters, kCounterSemanticDraws)) {
            statuses[world] = kStatusCounterOverflow;
            error_operations[world] = operation;
            return;
        }
        outputs[offset] = output;
        executed[offset] = true;
    }
    indices[world] = engine.index;
    has_gauss[world] = local_has_gauss;
    gauss_next[world] = local_gauss_next;
}


__global__ void math_probe_kernel(
    const double* inputs,
    const int64_t count,
    const int64_t gaussian_strategy,
    double* outputs
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const double angle = inputs[index * 2];
    const double logarithm_input = inputs[index * 2 + 1];
    if (gaussian_strategy == 1) {
        outputs[index * 3] = ::sin(angle);
        outputs[index * 3 + 1] = ::cos(angle);
        outputs[index * 3 + 2] = ::log(logarithm_input);
    } else {
        outputs[index * 3] = option_d_glibc_sin_v1(angle);
        outputs[index * 3 + 1] = option_d_glibc_cos_v1(angle);
        outputs[index * 3 + 2] = option_d_glibc_log_v1(logarithm_input);
    }
}


void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace


static std::vector<torch::Tensor> option_d_execute_impl(
    const torch::Tensor& words,
    const torch::Tensor& indices,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    const torch::Tensor& counters,
    const torch::Tensor& opcodes,
    const torch::Tensor& arguments,
    const torch::Tensor& mus,
    const torch::Tensor& sigmas,
    const torch::Tensor& predicate_sources,
    const torch::Tensor& predicate_kinds,
    const torch::Tensor& predicate_thresholds,
    const torch::Tensor& operation_counts,
    const int64_t gaussian_strategy,
    const torch::Tensor* tape_primary,
    const torch::Tensor* tape_secondary,
    const torch::Tensor* tape_base_raw_words
) {
    require_cuda_contiguous(words, "words");
    require_cuda_contiguous(indices, "indices");
    require_cuda_contiguous(has_gauss, "has_gauss");
    require_cuda_contiguous(gauss_next, "gauss_next");
    require_cuda_contiguous(counters, "counters");
    require_cuda_contiguous(opcodes, "opcodes");
    require_cuda_contiguous(arguments, "arguments");
    require_cuda_contiguous(mus, "mus");
    require_cuda_contiguous(sigmas, "sigmas");
    require_cuda_contiguous(predicate_sources, "predicate_sources");
    require_cuda_contiguous(predicate_kinds, "predicate_kinds");
    require_cuda_contiguous(predicate_thresholds, "predicate_thresholds");
    require_cuda_contiguous(operation_counts, "operation_counts");
    const auto device = words.device();
    for (const auto& tensor : {
        indices, has_gauss, gauss_next, counters, opcodes,
        arguments, mus, sigmas, predicate_sources, predicate_kinds,
        predicate_thresholds, operation_counts
    }) {
        TORCH_CHECK(tensor.device() == device, "all Option D tensors must share one CUDA device");
    }
    TORCH_CHECK(words.scalar_type() == torch::kUInt32, "words must be uint32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(has_gauss.scalar_type() == torch::kBool, "has_gauss must be bool");
    TORCH_CHECK(gauss_next.scalar_type() == torch::kFloat64, "gauss_next must be float64");
    TORCH_CHECK(counters.scalar_type() == torch::kInt64, "counters must be int64");
    TORCH_CHECK(opcodes.scalar_type() == torch::kInt32, "opcodes must be int32");
    TORCH_CHECK(arguments.scalar_type() == torch::kInt64, "arguments must be int64");
    TORCH_CHECK(mus.scalar_type() == torch::kFloat64, "mus must be float64");
    TORCH_CHECK(sigmas.scalar_type() == torch::kFloat64, "sigmas must be float64");
    TORCH_CHECK(predicate_sources.scalar_type() == torch::kInt64, "predicate_sources must be int64");
    TORCH_CHECK(predicate_kinds.scalar_type() == torch::kInt32, "predicate_kinds must be int32");
    TORCH_CHECK(predicate_thresholds.scalar_type() == torch::kFloat64, "predicate_thresholds must be float64");
    TORCH_CHECK(operation_counts.scalar_type() == torch::kInt64, "operation_counts must be int64");
    TORCH_CHECK(words.dim() == 2 && words.size(1) == kStateWords, "words must have shape [B,624]");
    const int64_t worlds = words.size(0);
    TORCH_CHECK(indices.dim() == 1 && indices.size(0) == worlds, "indices must have shape [B]");
    TORCH_CHECK(has_gauss.dim() == 1 && has_gauss.size(0) == worlds, "has_gauss must have shape [B]");
    TORCH_CHECK(gauss_next.dim() == 1 && gauss_next.size(0) == worlds, "gauss_next must have shape [B]");
    TORCH_CHECK(
        counters.dim() == 2
            && counters.size(0) == worlds
            && counters.size(1) == kCounterCount,
        "counters must have shape [B,9]"
    );
    TORCH_CHECK(opcodes.dim() == 2 && opcodes.size(0) == worlds, "opcodes must have shape [B,M]");
    const int64_t capacity = opcodes.size(1);
    TORCH_CHECK(arguments.sizes() == opcodes.sizes(), "arguments must match opcodes");
    TORCH_CHECK(mus.sizes() == opcodes.sizes(), "mus must match opcodes");
    TORCH_CHECK(sigmas.sizes() == opcodes.sizes(), "sigmas must match opcodes");
    TORCH_CHECK(predicate_sources.sizes() == opcodes.sizes(), "predicate_sources must match opcodes");
    TORCH_CHECK(predicate_kinds.sizes() == opcodes.sizes(), "predicate_kinds must match opcodes");
    TORCH_CHECK(predicate_thresholds.sizes() == opcodes.sizes(), "predicate_thresholds must match opcodes");
    TORCH_CHECK(
        operation_counts.dim() == 1 && operation_counts.size(0) == worlds,
        "operation_counts must have shape [B]"
    );
    TORCH_CHECK(gaussian_strategy >= 0 && gaussian_strategy <= 3, "gaussian strategy must be in [0,3]");
    int64_t tape_capacity = 0;
    if (gaussian_strategy == 3) {
        TORCH_CHECK(
            tape_primary != nullptr
                && tape_secondary != nullptr
                && tape_base_raw_words != nullptr,
            "host-tape baseline requires primary, secondary, and base tensors"
        );
        require_cuda_contiguous(*tape_primary, "tape_primary");
        require_cuda_contiguous(*tape_secondary, "tape_secondary");
        require_cuda_contiguous(*tape_base_raw_words, "tape_base_raw_words");
        TORCH_CHECK(
            tape_primary->device() == device
                && tape_secondary->device() == device
                && tape_base_raw_words->device() == device,
            "host-tape tensors must share the Option D device"
        );
        TORCH_CHECK(
            tape_primary->scalar_type() == torch::kFloat64
                && tape_secondary->scalar_type() == torch::kFloat64,
            "host-tape primary and secondary tensors must be float64"
        );
        TORCH_CHECK(
            tape_base_raw_words->scalar_type() == torch::kInt64,
            "host-tape base tensor must be int64"
        );
        TORCH_CHECK(
            tape_primary->dim() == 2
                && tape_primary->size(0) == worlds
                && tape_primary->size(1) > 0,
            "host-tape primary tensor must have shape [B,T] with T > 0"
        );
        TORCH_CHECK(
            tape_secondary->sizes() == tape_primary->sizes(),
            "host-tape secondary tensor must match primary"
        );
        TORCH_CHECK(
            tape_base_raw_words->dim() == 1
                && tape_base_raw_words->size(0) == worlds,
            "host-tape base tensor must have shape [B]"
        );
        tape_capacity = tape_primary->size(1);
    }

    c10::cuda::CUDAGuard guard(device);
    auto successor_words = words.clone();
    auto successor_indices = indices.clone();
    auto successor_has_gauss = has_gauss.clone();
    auto successor_gauss_next = gauss_next.clone();
    auto successor_counters = counters.clone();
    auto outputs = torch::zeros(
        {worlds, capacity},
        words.options().dtype(torch::kUInt64)
    );
    auto executed = torch::zeros(
        {worlds, capacity},
        words.options().dtype(torch::kBool)
    );
    auto statuses = torch::zeros(
        {worlds},
        words.options().dtype(torch::kInt32)
    );
    auto error_operations = torch::full(
        {worlds},
        -1,
        words.options().dtype(torch::kInt64)
    );
    if (worlds > 0) {
        constexpr int32_t threads = 128;
        const int32_t blocks = static_cast<int32_t>((worlds + threads - 1) / threads);
        execute_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            successor_words.data_ptr<uint32_t>(),
            successor_indices.data_ptr<int64_t>(),
            successor_has_gauss.data_ptr<bool>(),
            successor_gauss_next.data_ptr<double>(),
            successor_counters.data_ptr<int64_t>(),
            opcodes.data_ptr<int32_t>(),
            arguments.data_ptr<int64_t>(),
            mus.data_ptr<double>(),
            sigmas.data_ptr<double>(),
            predicate_sources.data_ptr<int64_t>(),
            predicate_kinds.data_ptr<int32_t>(),
            predicate_thresholds.data_ptr<double>(),
            operation_counts.data_ptr<int64_t>(),
            worlds,
            capacity,
            gaussian_strategy,
            tape_primary == nullptr ? nullptr : tape_primary->data_ptr<double>(),
            tape_secondary == nullptr ? nullptr : tape_secondary->data_ptr<double>(),
            tape_base_raw_words == nullptr
                ? nullptr
                : tape_base_raw_words->data_ptr<int64_t>(),
            tape_capacity,
            outputs.data_ptr<uint64_t>(),
            executed.data_ptr<bool>(),
            statuses.data_ptr<int32_t>(),
            error_operations.data_ptr<int64_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {
        successor_words,
        successor_indices,
        successor_has_gauss,
        successor_gauss_next,
        successor_counters,
        outputs,
        executed,
        statuses,
        error_operations,
    };
}


std::vector<torch::Tensor> option_d_execute_cuda(
    const torch::Tensor& words,
    const torch::Tensor& indices,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    const torch::Tensor& counters,
    const torch::Tensor& opcodes,
    const torch::Tensor& arguments,
    const torch::Tensor& mus,
    const torch::Tensor& sigmas,
    const torch::Tensor& predicate_sources,
    const torch::Tensor& predicate_kinds,
    const torch::Tensor& predicate_thresholds,
    const torch::Tensor& operation_counts,
    const int64_t gaussian_strategy
) {
    TORCH_CHECK(
        gaussian_strategy >= 0 && gaussian_strategy <= 2,
        "standalone Gaussian strategy must be 0, 1, or 2"
    );
    return option_d_execute_impl(
        words,
        indices,
        has_gauss,
        gauss_next,
        counters,
        opcodes,
        arguments,
        mus,
        sigmas,
        predicate_sources,
        predicate_kinds,
        predicate_thresholds,
        operation_counts,
        gaussian_strategy,
        nullptr,
        nullptr,
        nullptr
    );
}


std::vector<torch::Tensor> option_d_execute_tape_baseline_cuda(
    const torch::Tensor& words,
    const torch::Tensor& indices,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    const torch::Tensor& counters,
    const torch::Tensor& opcodes,
    const torch::Tensor& arguments,
    const torch::Tensor& mus,
    const torch::Tensor& sigmas,
    const torch::Tensor& predicate_sources,
    const torch::Tensor& predicate_kinds,
    const torch::Tensor& predicate_thresholds,
    const torch::Tensor& operation_counts,
    const torch::Tensor& tape_primary,
    const torch::Tensor& tape_secondary,
    const torch::Tensor& tape_base_raw_words
) {
    return option_d_execute_impl(
        words,
        indices,
        has_gauss,
        gauss_next,
        counters,
        opcodes,
        arguments,
        mus,
        sigmas,
        predicate_sources,
        predicate_kinds,
        predicate_thresholds,
        operation_counts,
        3,
        &tape_primary,
        &tape_secondary,
        &tape_base_raw_words
    );
}


torch::Tensor option_d_math_probe_cuda(
    const torch::Tensor& inputs,
    const int64_t gaussian_strategy
) {
    require_cuda_contiguous(inputs, "math probe inputs");
    TORCH_CHECK(inputs.scalar_type() == torch::kFloat64, "math probe inputs must be float64");
    TORCH_CHECK(inputs.dim() == 2 && inputs.size(1) == 2, "math probe inputs must have shape [N,2]");
    TORCH_CHECK(gaussian_strategy == 1 || gaussian_strategy == 2, "math probe strategy must be 1 or 2");
    c10::cuda::CUDAGuard guard(inputs.device());
    const int64_t count = inputs.size(0);
    auto outputs = torch::empty({count, 3}, inputs.options());
    if (count > 0) {
        constexpr int32_t threads = 128;
        const int32_t blocks = static_cast<int32_t>((count + threads - 1) / threads);
        math_probe_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            inputs.data_ptr<double>(),
            count,
            gaussian_strategy,
            outputs.data_ptr<double>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return outputs;
}
