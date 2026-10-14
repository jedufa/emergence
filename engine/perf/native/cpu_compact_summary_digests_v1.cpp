#include "cpu_compact_summary_digests_v1.hpp"
#include "../cuda_architecture_b_torch_rng_state_pack_v1.cuh"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace eiel::native::cpu_compact_summary_digests_v1 {
namespace {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;

class Sha256V1 {
  public:
    Sha256V1() noexcept
        : state_{
              0x6a09e667U,
              0xbb67ae85U,
              0x3c6ef372U,
              0xa54ff53aU,
              0x510e527fU,
              0x9b05688cU,
              0x1f83d9abU,
              0x5be0cd19U,
          } {}

    bool update(const void* const data, const std::uint64_t size) noexcept {
        if (!valid_ || size > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
            valid_ = false;
            return false;
        }
        if (size == 0U) {
            return true;
        }
        if (data == nullptr) {
            valid_ = false;
            return false;
        }

        const auto* input = static_cast<const std::uint8_t*>(data);
        std::uint64_t remaining = size;
        total_bytes_ += size;

        if (buffer_size_ != 0U) {
            const std::size_t available = buffer_.size() - buffer_size_;
            const std::size_t copied = remaining < available
                ? static_cast<std::size_t>(remaining)
                : available;
            std::memcpy(buffer_.data() + buffer_size_, input, copied);
            buffer_size_ += copied;
            input += copied;
            remaining -= copied;
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0U;
            }
        }

        while (remaining >= buffer_.size()) {
            transform(input);
            input += buffer_.size();
            remaining -= buffer_.size();
        }
        if (remaining != 0U) {
            const std::size_t tail = static_cast<std::size_t>(remaining);
            std::memcpy(buffer_.data(), input, tail);
            buffer_size_ = tail;
        }
        return true;
    }

    bool final(DigestV1& output) noexcept {
        if (!valid_ || total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
            return false;
        }
        const std::uint64_t total_bits = total_bytes_ * 8U;
        buffer_[buffer_size_++] = 0x80U;
        if (buffer_size_ > 56U) {
            while (buffer_size_ < buffer_.size()) {
                buffer_[buffer_size_++] = 0U;
            }
            transform(buffer_.data());
            buffer_size_ = 0U;
        }
        while (buffer_size_ < 56U) {
            buffer_[buffer_size_++] = 0U;
        }
        for (std::size_t index = 0U; index < 8U; ++index) {
            buffer_[56U + index] = static_cast<std::uint8_t>(
                total_bits >> (56U - index * 8U)
            );
        }
        transform(buffer_.data());
        for (std::size_t index = 0U; index < state_.size(); ++index) {
            output.bytes[index * 4U] = static_cast<std::uint8_t>(state_[index] >> 24U);
            output.bytes[index * 4U + 1U] = static_cast<std::uint8_t>(state_[index] >> 16U);
            output.bytes[index * 4U + 2U] = static_cast<std::uint8_t>(state_[index] >> 8U);
            output.bytes[index * 4U + 3U] = static_cast<std::uint8_t>(state_[index]);
        }
        return true;
    }

  private:
    static std::uint32_t rotate_right(
        const std::uint32_t value,
        const std::uint32_t distance
    ) noexcept {
        return (value >> distance) | (value << (32U - distance));
    }

    void transform(const std::uint8_t* const block) noexcept {
        static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
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
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0U; index < 16U; ++index) {
            const std::size_t offset = index * 4U;
            words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U)
                | (static_cast<std::uint32_t>(block[offset + 1U]) << 16U)
                | (static_cast<std::uint32_t>(block[offset + 2U]) << 8U)
                | static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t sigma0 = rotate_right(words[index - 15U], 7U)
                ^ rotate_right(words[index - 15U], 18U)
                ^ (words[index - 15U] >> 3U);
            const std::uint32_t sigma1 = rotate_right(words[index - 2U], 17U)
                ^ rotate_right(words[index - 2U], 19U)
                ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U]
                + sigma0
                + words[index - 7U]
                + sigma1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0U; index < words.size(); ++index) {
            const std::uint32_t sum1 = rotate_right(e, 6U)
                ^ rotate_right(e, 11U)
                ^ rotate_right(e, 25U);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 = h
                + sum1
                + choose
                + kRoundConstants[index]
                + words[index];
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
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0U;
    std::uint64_t total_bytes_ = 0U;
    bool valid_ = true;
};

template <std::size_t Size>
bool update_literal(Sha256V1& digest, const char (&value)[Size]) noexcept {
    return digest.update(value, static_cast<std::uint64_t>(Size - 1U));
}

bool little_endian_host() noexcept {
    constexpr std::uint16_t value = 1U;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1U;
}

std::uint64_t scalar_size(const ScalarTypeV1 dtype) noexcept {
    switch (dtype) {
        case ScalarTypeV1::kBool:
        case ScalarTypeV1::kInt8:
        case ScalarTypeV1::kUInt8:
            return 1U;
        case ScalarTypeV1::kInt32:
        case ScalarTypeV1::kFloat32:
            return 4U;
        case ScalarTypeV1::kInt64:
        case ScalarTypeV1::kFloat64:
            return 8U;
        case ScalarTypeV1::kUnknown:
            return 0U;
    }
    return 0U;
}

DigestStatusV1 validated_tensor_bytes(
    const TensorViewV1& view,
    const ScalarTypeV1 expected_dtype,
    const std::int32_t expected_rank,
    std::uint64_t& byte_count
) noexcept {
    byte_count = 0U;
    if (view.dtype != expected_dtype) {
        return DigestStatusV1::kDtypeMismatch;
    }
    if (view.rank != expected_rank) {
        return DigestStatusV1::kRankMismatch;
    }
    if (view.extent0 < 0 || view.extent1 < 0) {
        return DigestStatusV1::kNegativeExtent;
    }
    if ((expected_rank == 1 && view.extent1 != 0) || expected_rank < 1 || expected_rank > 2) {
        return DigestStatusV1::kRankMismatch;
    }
    if (!view.c_contiguous) {
        return DigestStatusV1::kNonContiguous;
    }

    const std::uint64_t extent0 = static_cast<std::uint64_t>(view.extent0);
    const std::uint64_t extent1 = expected_rank == 1
        ? 1U
        : static_cast<std::uint64_t>(view.extent1);
    if (extent1 != 0U && extent0 > std::numeric_limits<std::uint64_t>::max() / extent1) {
        return DigestStatusV1::kSizeOverflow;
    }
    const std::uint64_t elements = extent0 * extent1;
    const std::uint64_t item_size = scalar_size(expected_dtype);
    if (item_size == 0U || elements > std::numeric_limits<std::uint64_t>::max() / item_size) {
        return DigestStatusV1::kSizeOverflow;
    }
    byte_count = elements * item_size;
    if (view.storage_bytes != byte_count) {
        return DigestStatusV1::kStorageSizeMismatch;
    }
    if (byte_count != 0U && view.data == nullptr) {
        return DigestStatusV1::kNullData;
    }
    return DigestStatusV1::kSuccess;
}

template <typename T>
const T* tensor_data(const TensorViewV1& view) noexcept {
    return static_cast<const T*>(view.data);
}

std::uint32_t decimal_bytes(
    const std::uint64_t value,
    std::uint8_t output[20]
) noexcept {
    std::uint8_t reversed[20];
    std::uint32_t count = 0U;
    std::uint64_t remaining = value;
    do {
        reversed[count++] = static_cast<std::uint8_t>('0' + remaining % 10U);
        remaining /= 10U;
    } while (remaining != 0U);
    for (std::uint32_t index = 0U; index < count; ++index) {
        output[index] = reversed[count - index - 1U];
    }
    return count;
}

bool update_decimal(Sha256V1& digest, const std::uint64_t value) noexcept {
    std::uint8_t bytes[20];
    const std::uint32_t count = decimal_bytes(value, bytes);
    return digest.update(bytes, count);
}

void digest_to_hex(const DigestV1& digest, std::uint8_t output[64]) noexcept {
    constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
        output[index * 2U] = static_cast<std::uint8_t>(kHex[digest.bytes[index] >> 4U]);
        output[index * 2U + 1U] = static_cast<std::uint8_t>(kHex[digest.bytes[index] & 0x0FU]);
    }
}

bool raw_sha256(
    const void* const data,
    const std::uint64_t byte_count,
    DigestV1& output
) noexcept {
    Sha256V1 digest;
    return digest.update(data, byte_count) && digest.final(output);
}

bool canonical_sha256_from_packed_torch_rng_rows(
    const TensorViewV1& packed_view,
    const std::uint64_t cursor,
    DigestV1& output
) noexcept {
    if (
        packed_view.data == nullptr
        && cursor != 0U
    ) {
        return false;
    }
    const auto* const packed =
        static_cast<const std::uint8_t*>(packed_view.data);
    Sha256V1 digest;
    std::array<std::uint8_t, torch_rng_pack::kFullStateBytes> decoded_row{};
    for (std::uint64_t row = 0U; row < cursor; ++row) {
        const std::uint8_t* const packed_row =
            packed + row * torch_rng_pack::kPackedStateBytes;
        for (std::int64_t byte = 0; byte < torch_rng_pack::kFullStateBytes; ++byte) {
            decoded_row[static_cast<std::size_t>(byte)] =
                torch_rng_pack::decoded_byte_v1(packed_row, byte);
        }
        if (!digest.update(decoded_row.data(), decoded_row.size())) {
            return false;
        }
    }
    return digest.final(output);
}

bool update_zero_bytes(Sha256V1& digest, std::uint64_t byte_count) noexcept {
    constexpr std::array<std::uint8_t, 64> kZeroBytes{};
    while (byte_count != 0U) {
        const std::uint64_t chunk = byte_count < kZeroBytes.size()
            ? byte_count
            : static_cast<std::uint64_t>(kZeroBytes.size());
        if (!digest.update(kZeroBytes.data(), chunk)) {
            return false;
        }
        byte_count -= chunk;
    }
    return true;
}

DigestStatusV1 validate_checkpoint_tensor_descriptor(
    const CheckpointTensorDigestDescriptorV1& descriptor,
    std::uint64_t& tensor_bytes,
    std::uint64_t& row_bytes
) noexcept {
    tensor_bytes = 0U;
    row_bytes = 0U;
    if (descriptor.tensor.data == nullptr) {
        return DigestStatusV1::kNullData;
    }
    if (scalar_size(descriptor.tensor.dtype) == 0U) {
        return DigestStatusV1::kDtypeMismatch;
    }
    const auto mode = static_cast<std::int32_t>(descriptor.mode);
    if (
        mode < static_cast<std::int32_t>(
            TensorCanonicalizationModeV1::kExactStateRawBytes
        )
        || mode > static_cast<std::int32_t>(
            TensorCanonicalizationModeV1::kActivePrefix
        )
    ) {
        return DigestStatusV1::kInvalidMode;
    }
    if (descriptor.tensor.rank < 1 || descriptor.tensor.rank > 2) {
        return DigestStatusV1::kRankMismatch;
    }
    const DigestStatusV1 tensor_status = validated_tensor_bytes(
        descriptor.tensor,
        descriptor.tensor.dtype,
        descriptor.tensor.rank,
        tensor_bytes
    );
    if (tensor_status != DigestStatusV1::kSuccess) {
        return tensor_status;
    }
    if (
        descriptor.tensor.extent0 == 0
        || (descriptor.tensor.rank == 2 && descriptor.tensor.extent1 == 0)
    ) {
        return DigestStatusV1::kInvalidRowLayout;
    }

    const std::uint64_t item_size = scalar_size(descriptor.tensor.dtype);
    const std::uint64_t columns = descriptor.tensor.rank == 1
        ? 1U
        : static_cast<std::uint64_t>(descriptor.tensor.extent1);
    if (columns > std::numeric_limits<std::uint64_t>::max() / item_size) {
        return DigestStatusV1::kSizeOverflow;
    }
    row_bytes = columns * item_size;
    const std::uint64_t row_capacity = static_cast<std::uint64_t>(
        descriptor.tensor.extent0
    );
    if (
        row_capacity > std::numeric_limits<std::uint64_t>::max() / row_bytes
        || row_capacity * row_bytes != tensor_bytes
    ) {
        return DigestStatusV1::kInvalidRowLayout;
    }

    if (descriptor.mode == TensorCanonicalizationModeV1::kLiveRowsDeadZero) {
        if (descriptor.alive_mask == nullptr || descriptor.alive_mask->data == nullptr) {
            return DigestStatusV1::kNullActiveSource;
        }
        std::uint64_t mask_bytes = 0U;
        const DigestStatusV1 mask_status = validated_tensor_bytes(
            *descriptor.alive_mask,
            ScalarTypeV1::kBool,
            1,
            mask_bytes
        );
        if (mask_status != DigestStatusV1::kSuccess) {
            return mask_status;
        }
        if (
            descriptor.alive_mask->extent0 != descriptor.tensor.extent0
            || mask_bytes != row_capacity
        ) {
            return DigestStatusV1::kInvalidRowLayout;
        }
        const auto* const mask = tensor_data<std::uint8_t>(*descriptor.alive_mask);
        for (std::uint64_t row = 0U; row < row_capacity; ++row) {
            if (mask[row] > 1U) {
                return DigestStatusV1::kBooleanValueInvalid;
            }
        }
    }
    if (
        descriptor.mode == TensorCanonicalizationModeV1::kActivePrefix
        && (
            descriptor.active_length < 0
            || static_cast<std::uint64_t>(descriptor.active_length) > row_capacity
        )
    ) {
        return DigestStatusV1::kActiveLengthOutOfRange;
    }
    return DigestStatusV1::kSuccess;
}

DigestStatusV1 checkpoint_result_failure(
    CheckpointTensorDigestResultV1& output,
    const DigestStatusV1 status
) noexcept {
    output.status = status;
    return status;
}

bool update_tensor_manifest_entry(
    Sha256V1& digest,
    const char* const dtype,
    const std::size_t dtype_size,
    const char* const field,
    const std::size_t field_size,
    const DigestV1& raw_digest,
    const std::uint64_t extent0,
    const bool rank_two,
    const std::uint64_t extent1
) noexcept {
    std::uint8_t digest_hex[64];
    digest_to_hex(raw_digest, digest_hex);
    return update_literal(digest, "{\"dtype\":\"")
        && digest.update(dtype, dtype_size)
        && update_literal(digest, "\",\"field\":\"")
        && digest.update(field, field_size)
        && update_literal(digest, "\",\"sha256\":\"")
        && digest.update(digest_hex, sizeof(digest_hex))
        && update_literal(digest, "\",\"shape\":[")
        && update_decimal(digest, extent0)
        && (!rank_two
            || (update_literal(digest, ",") && update_decimal(digest, extent1)))
        && update_literal(digest, "]}");
}

template <typename T>
bool update_lifecycle_field(
    Sha256V1& digest,
    const char* const field,
    const std::size_t field_size,
    const char* const dtype,
    const std::size_t dtype_size,
    const TensorViewV1& view,
    const std::uint64_t active_count
) noexcept {
    std::uint8_t shape[22];
    shape[0] = static_cast<std::uint8_t>('[');
    const std::uint32_t digits = decimal_bytes(active_count, shape + 1U);
    shape[digits + 1U] = static_cast<std::uint8_t>(']');
    const std::uint64_t raw_bytes = active_count * sizeof(T);
    return digest.update(field, field_size)
        && digest.update(dtype, dtype_size)
        && digest.update(shape, digits + 2U)
        && digest.update(view.data, raw_bytes);
}

DigestStatusV1 validate_lifecycle(
    const TypedLifecycleEventsViewV1& events
) noexcept {
    struct FieldValidation {
        const TensorViewV1* view;
        ScalarTypeV1 dtype;
    };
    const std::array<FieldValidation, 16> fields = {{
        {&events.event_type, ScalarTypeV1::kInt32},
        {&events.generation, ScalarTypeV1::kInt64},
        {&events.tick, ScalarTypeV1::kInt64},
        {&events.sequence, ScalarTypeV1::kInt64},
        {&events.actor_agent_id, ScalarTypeV1::kInt64},
        {&events.target_agent_id, ScalarTypeV1::kInt64},
        {&events.actor_slot, ScalarTypeV1::kInt64},
        {&events.target_slot, ScalarTypeV1::kInt64},
        {&events.parent_agent_id, ScalarTypeV1::kInt64},
        {&events.child_agent_id, ScalarTypeV1::kInt64},
        {&events.lineage_founder_id, ScalarTypeV1::kInt64},
        {&events.lineage_depth, ScalarTypeV1::kInt32},
        {&events.subtype_code, ScalarTypeV1::kInt32},
        {&events.reason_code, ScalarTypeV1::kInt32},
        {&events.success, ScalarTypeV1::kInt8},
        {&events.value, ScalarTypeV1::kFloat32},
    }};
    std::int64_t capacity = -1;
    for (const FieldValidation& field : fields) {
        std::uint64_t ignored_bytes = 0U;
        const DigestStatusV1 status = validated_tensor_bytes(
            *field.view, field.dtype, 1, ignored_bytes
        );
        if (status != DigestStatusV1::kSuccess) {
            return status;
        }
        if (capacity < 0) {
            capacity = field.view->extent0;
        } else if (field.view->extent0 != capacity) {
            return DigestStatusV1::kColumnLengthMismatch;
        }
    }
    if (events.active_count < 0 || events.active_count > capacity) {
        return DigestStatusV1::kCursorOutOfRange;
    }

    const auto* const generation = tensor_data<std::int64_t>(events.generation);
    const auto* const tick = tensor_data<std::int64_t>(events.tick);
    const auto* const sequence = tensor_data<std::int64_t>(events.sequence);
    const auto* const value = tensor_data<float>(events.value);
    std::int64_t previous_generation = 0;
    std::int64_t previous_tick = 0;
    std::int64_t previous_sequence = -1;
    for (std::int64_t row = 0; row < events.active_count; ++row) {
        if (sequence[row] != row) {
            return DigestStatusV1::kSequenceMismatch;
        }
        if (!std::isfinite(value[row])) {
            return DigestStatusV1::kNonFiniteValue;
        }
        if (row != 0
            && (generation[row] < previous_generation
                || (generation[row] == previous_generation && tick[row] < previous_tick)
                || (generation[row] == previous_generation
                    && tick[row] == previous_tick
                    && sequence[row] <= previous_sequence))) {
            return DigestStatusV1::kRowOrderInvalid;
        }
        previous_generation = generation[row];
        previous_tick = tick[row];
        previous_sequence = sequence[row];
    }
    return DigestStatusV1::kSuccess;
}

DigestStatusV1 validate_policy_tape(
    const PolicyMutationTapeViewV1& tape
) noexcept {
    struct FieldValidation {
        const TensorViewV1* view;
        ScalarTypeV1 dtype;
        std::int32_t rank;
    };
    const std::array<FieldValidation, 5> fields = {{
        {&tape.crossover_mask, ScalarTypeV1::kBool, 2},
        {&tape.delta, ScalarTypeV1::kFloat32, 2},
        {&tape.mutation_mask, ScalarTypeV1::kBool, 2},
        {&tape.post_rng_state, ScalarTypeV1::kUInt8, 2},
        {&tape.semantic_draws, ScalarTypeV1::kInt64, 1},
    }};
    std::int64_t capacity = -1;
    for (const FieldValidation& field : fields) {
        std::uint64_t ignored_bytes = 0U;
        const DigestStatusV1 status = validated_tensor_bytes(
            *field.view, field.dtype, field.rank, ignored_bytes
        );
        if (status != DigestStatusV1::kSuccess) {
            return status;
        }
        if (capacity < 0) {
            capacity = field.view->extent0;
        } else if (field.view->extent0 != capacity) {
            return DigestStatusV1::kColumnLengthMismatch;
        }
    }
    if (tape.cursor < 0 || tape.cursor > capacity) {
        return DigestStatusV1::kCursorOutOfRange;
    }
    if (tape.crossover_mask.extent1 != tape.delta.extent1
        || tape.mutation_mask.extent1 != tape.delta.extent1) {
        return DigestStatusV1::kPolicyMaskShapeMismatch;
    }
    if (
        tape.post_rng_state_is_packed
        && tape.post_rng_state.extent1 != torch_rng_pack::kPackedStateBytes
    ) {
        return DigestStatusV1::kInvalidRowLayout;
    }

    const auto consumed_rows = static_cast<std::uint64_t>(tape.cursor);
    const auto delta_columns = static_cast<std::uint64_t>(tape.delta.extent1);
    const std::uint64_t delta_elements = consumed_rows * delta_columns;
    const auto* const delta = tensor_data<float>(tape.delta);
    for (std::uint64_t index = 0U; index < delta_elements; ++index) {
        if (!std::isfinite(delta[index])) {
            return DigestStatusV1::kNonFiniteValue;
        }
    }
    const auto validate_mask = [consumed_rows](const TensorViewV1& mask) noexcept {
        const std::uint64_t elements = consumed_rows
            * static_cast<std::uint64_t>(mask.extent1);
        const auto* const bytes = tensor_data<std::uint8_t>(mask);
        for (std::uint64_t index = 0U; index < elements; ++index) {
            if (bytes[index] > 1U) {
                return false;
            }
        }
        return true;
    };
    if (!validate_mask(tape.crossover_mask) || !validate_mask(tape.mutation_mask)) {
        return DigestStatusV1::kBooleanValueInvalid;
    }
    return DigestStatusV1::kSuccess;
}

}  // namespace

DigestStatusV1 checkpoint_tensor_descriptor_digests(
    const CheckpointTensorDigestDescriptorV1& descriptor,
    CheckpointTensorDigestResultV1& output
) noexcept {
    output.raw.bytes.fill(0U);
    output.canonical.bytes.fill(0U);
    output.status = DigestStatusV1::kSuccess;
    output.valid = 0U;
    if (!little_endian_host()) {
        return checkpoint_result_failure(
            output, DigestStatusV1::kNonLittleEndianHost
        );
    }

    std::uint64_t tensor_bytes = 0U;
    std::uint64_t row_bytes = 0U;
    const DigestStatusV1 validation = validate_checkpoint_tensor_descriptor(
        descriptor, tensor_bytes, row_bytes
    );
    if (validation != DigestStatusV1::kSuccess) {
        return checkpoint_result_failure(output, validation);
    }

    DigestV1 raw_candidate{};
    if (!raw_sha256(descriptor.tensor.data, tensor_bytes, raw_candidate)) {
        return checkpoint_result_failure(output, DigestStatusV1::kSizeOverflow);
    }

    DigestV1 canonical_candidate{};
    if (descriptor.mode == TensorCanonicalizationModeV1::kExactStateRawBytes) {
        canonical_candidate = raw_candidate;
    } else if (descriptor.mode == TensorCanonicalizationModeV1::kActivePrefix) {
        const std::uint64_t active_bytes = static_cast<std::uint64_t>(
            descriptor.active_length
        ) * row_bytes;
        if (!raw_sha256(descriptor.tensor.data, active_bytes, canonical_candidate)) {
            return checkpoint_result_failure(output, DigestStatusV1::kSizeOverflow);
        }
    } else {
        Sha256V1 canonical_digest;
        const auto* const source = static_cast<const std::uint8_t*>(
            descriptor.tensor.data
        );
        const auto* const mask = tensor_data<std::uint8_t>(*descriptor.alive_mask);
        const std::uint64_t row_capacity = static_cast<std::uint64_t>(
            descriptor.tensor.extent0
        );
        bool updated = true;
        for (std::uint64_t row = 0U; updated && row < row_capacity; ++row) {
            updated = mask[row] != 0U
                ? canonical_digest.update(source + row * row_bytes, row_bytes)
                : update_zero_bytes(canonical_digest, row_bytes);
        }
        if (!updated || !canonical_digest.final(canonical_candidate)) {
            return checkpoint_result_failure(output, DigestStatusV1::kSizeOverflow);
        }
    }

    output.raw = raw_candidate;
    output.canonical = canonical_candidate;
    output.status = DigestStatusV1::kSuccess;
    output.valid = 1U;
    return DigestStatusV1::kSuccess;
}

DigestStatusV1 typed_lifecycle_active_prefix_digest(
    const TypedLifecycleEventsViewV1& events,
    DigestV1& output
) noexcept {
    output.bytes.fill(0U);
    if (!little_endian_host()) {
        return DigestStatusV1::kNonLittleEndianHost;
    }
    const DigestStatusV1 validation = validate_lifecycle(events);
    if (validation != DigestStatusV1::kSuccess) {
        return validation;
    }

    const std::uint64_t count = static_cast<std::uint64_t>(events.active_count);
    Sha256V1 digest;
    bool valid = update_literal(digest, "eiel_typed_lifecycle_event_v1");
    // FIELD_DTYPES lexicographic name order from lifecycle_events.py.
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "actor_agent_id", 14U, "int64", 5U, events.actor_agent_id, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "actor_slot", 10U, "int64", 5U, events.actor_slot, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "child_agent_id", 14U, "int64", 5U, events.child_agent_id, count
    );
    valid = valid && update_lifecycle_field<std::int32_t>(
        digest, "event_type", 10U, "int32", 5U, events.event_type, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "generation", 10U, "int64", 5U, events.generation, count
    );
    valid = valid && update_lifecycle_field<std::int32_t>(
        digest, "lineage_depth", 13U, "int32", 5U, events.lineage_depth, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "lineage_founder_id", 18U, "int64", 5U,
        events.lineage_founder_id, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "parent_agent_id", 15U, "int64", 5U, events.parent_agent_id, count
    );
    valid = valid && update_lifecycle_field<std::int32_t>(
        digest, "reason_code", 11U, "int32", 5U, events.reason_code, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "sequence", 8U, "int64", 5U, events.sequence, count
    );
    valid = valid && update_lifecycle_field<std::int32_t>(
        digest, "subtype_code", 12U, "int32", 5U, events.subtype_code, count
    );
    valid = valid && update_lifecycle_field<std::int8_t>(
        digest, "success", 7U, "int8", 4U, events.success, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "target_agent_id", 15U, "int64", 5U, events.target_agent_id, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "target_slot", 11U, "int64", 5U, events.target_slot, count
    );
    valid = valid && update_lifecycle_field<std::int64_t>(
        digest, "tick", 4U, "int64", 5U, events.tick, count
    );
    valid = valid && update_lifecycle_field<float>(
        digest, "value", 5U, "float32", 7U, events.value, count
    );
    DigestV1 candidate{};
    if (!valid || !digest.final(candidate)) {
        return DigestStatusV1::kSizeOverflow;
    }
    output = candidate;
    return DigestStatusV1::kSuccess;
}

DigestStatusV1 active_agent_ids_digest(
    const TensorViewV1& active_agent_ids,
    DigestV1& output
) noexcept {
    output.bytes.fill(0U);
    if (!little_endian_host()) {
        return DigestStatusV1::kNonLittleEndianHost;
    }
    std::uint64_t raw_bytes = 0U;
    const DigestStatusV1 validation = validated_tensor_bytes(
        active_agent_ids, ScalarTypeV1::kInt64, 1, raw_bytes
    );
    if (validation != DigestStatusV1::kSuccess) {
        return validation;
    }
    const auto* const ids = tensor_data<std::int64_t>(active_agent_ids);
    for (std::int64_t index = 0; index < active_agent_ids.extent0; ++index) {
        if (ids[index] < 0 || (index != 0 && ids[index] <= ids[index - 1])) {
            return DigestStatusV1::kActiveAgentIdInvalid;
        }
    }

    DigestV1 raw_digest{};
    if (!raw_sha256(active_agent_ids.data, raw_bytes, raw_digest)) {
        return DigestStatusV1::kSizeOverflow;
    }
    Sha256V1 digest;
    const std::uint64_t count = static_cast<std::uint64_t>(active_agent_ids.extent0);
    bool valid = update_literal(
        digest,
        "{\"fields\":["
    );
    valid = valid && update_tensor_manifest_entry(
        digest,
        "int64", 5U,
        "agent_ids_by_alive_order", 24U,
        raw_digest,
        count,
        false,
        0U
    );
    valid = valid && update_literal(
        digest,
        "],\"schema_version\":\"eiel_active_agent_ids_digest_v1\"}"
    );
    DigestV1 candidate{};
    if (!valid || !digest.final(candidate)) {
        return DigestStatusV1::kSizeOverflow;
    }
    output = candidate;
    return DigestStatusV1::kSuccess;
}

DigestStatusV1 policy_mutation_tape_consumed_prefix_digest(
    const PolicyMutationTapeViewV1& tape,
    DigestV1& output
) noexcept {
    output.bytes.fill(0U);
    if (!little_endian_host()) {
        return DigestStatusV1::kNonLittleEndianHost;
    }
    const DigestStatusV1 validation = validate_policy_tape(tape);
    if (validation != DigestStatusV1::kSuccess) {
        return validation;
    }

    const std::uint64_t cursor = static_cast<std::uint64_t>(tape.cursor);
    Sha256V1 digest;
    bool valid = update_literal(digest, "{\"cursor\":");
    valid = valid && update_decimal(digest, cursor);
    valid = valid && update_literal(digest, ",\"fields\":[");

    struct ManifestField {
        const TensorViewV1* view;
        const char* dtype;
        std::size_t dtype_size;
        const char* name;
        std::size_t name_size;
        bool rank_two;
        std::uint64_t item_size;
    };
    const std::array<ManifestField, 5> fields = {{
        {&tape.crossover_mask, "bool", 4U,
         "policy_birth_crossover_mask_tape", 32U, true, 1U},
        {&tape.delta, "float32", 7U,
         "policy_birth_delta_tape", 23U, true, 4U},
        {&tape.mutation_mask, "bool", 4U,
         "policy_birth_mutation_mask_tape", 31U, true, 1U},
        {&tape.post_rng_state, "uint8", 5U,
         "policy_birth_post_rng_state_tape", 32U, true, 1U},
        {&tape.semantic_draws, "int64", 5U,
         "policy_birth_semantic_draws_tape", 32U, false, 8U},
    }};
    for (std::size_t index = 0U; valid && index < fields.size(); ++index) {
        const ManifestField& field = fields[index];
        if (index != 0U) {
            valid = update_literal(digest, ",");
        }
        std::uint64_t extent1 = field.rank_two
            ? static_cast<std::uint64_t>(field.view->extent1)
            : 0U;
        std::uint64_t raw_bytes = cursor
            * (field.rank_two ? extent1 : 1U)
            * field.item_size;
        DigestV1 raw_digest{};
        if (field.view == &tape.post_rng_state && tape.post_rng_state_is_packed) {
            extent1 = static_cast<std::uint64_t>(torch_rng_pack::kFullStateBytes);
            raw_bytes = cursor * extent1;
            valid = valid && canonical_sha256_from_packed_torch_rng_rows(
                *field.view,
                cursor,
                raw_digest
            );
        } else {
            valid = valid && raw_sha256(field.view->data, raw_bytes, raw_digest);
        }
        valid = valid && update_tensor_manifest_entry(
            digest,
            field.dtype,
            field.dtype_size,
            field.name,
            field.name_size,
            raw_digest,
            cursor,
            field.rank_two,
            extent1
        );
    }
    valid = valid && update_literal(
        digest,
        "],\"schema_version\":\"eiel_policy_mutation_tape_consumed_prefix_v1\"}"
    );
    DigestV1 candidate{};
    if (!valid || !digest.final(candidate)) {
        return DigestStatusV1::kSizeOverflow;
    }
    output = candidate;
    return DigestStatusV1::kSuccess;
}

const char* digest_status_name(const DigestStatusV1 status) noexcept {
    switch (status) {
        case DigestStatusV1::kSuccess:
            return "success";
        case DigestStatusV1::kNonLittleEndianHost:
            return "non_little_endian_host";
        case DigestStatusV1::kDtypeMismatch:
            return "dtype_mismatch";
        case DigestStatusV1::kRankMismatch:
            return "rank_mismatch";
        case DigestStatusV1::kNegativeExtent:
            return "negative_extent";
        case DigestStatusV1::kNonContiguous:
            return "non_contiguous";
        case DigestStatusV1::kStorageSizeMismatch:
            return "storage_size_mismatch";
        case DigestStatusV1::kNullData:
            return "null_data";
        case DigestStatusV1::kSizeOverflow:
            return "size_overflow";
        case DigestStatusV1::kColumnLengthMismatch:
            return "column_length_mismatch";
        case DigestStatusV1::kCursorOutOfRange:
            return "cursor_out_of_range";
        case DigestStatusV1::kSequenceMismatch:
            return "sequence_mismatch";
        case DigestStatusV1::kRowOrderInvalid:
            return "row_order_invalid";
        case DigestStatusV1::kNonFiniteValue:
            return "non_finite_value";
        case DigestStatusV1::kActiveAgentIdInvalid:
            return "active_agent_id_invalid";
        case DigestStatusV1::kPolicyMaskShapeMismatch:
            return "policy_mask_shape_mismatch";
        case DigestStatusV1::kBooleanValueInvalid:
            return "boolean_value_invalid";
        case DigestStatusV1::kInvalidMode:
            return "invalid_mode";
        case DigestStatusV1::kInvalidRowLayout:
            return "invalid_row_layout";
        case DigestStatusV1::kNullActiveSource:
            return "null_active_source";
        case DigestStatusV1::kActiveLengthOutOfRange:
            return "active_length_out_of_range";
    }
    return "unknown";
}

}  // namespace eiel::native::cpu_compact_summary_digests_v1
