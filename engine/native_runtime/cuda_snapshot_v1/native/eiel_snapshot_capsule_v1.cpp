#include <torch/extension.h>

#include "eiel_snapshot_projection_v1.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <openssl/sha.h>
#include <zlib.h>

#include <sys/stat.h>
#include <unistd.h>

namespace py = pybind11;

namespace eiel::snapshot::capsule_v1 {

namespace {

constexpr std::array<std::uint8_t, 16> kMagic{
    'E', 'I', 'E', 'L', 'S', 'N', 'A', 'P',
    'C', 'A', 'P', 'S', 'U', 'L', 'E', 0,
};
constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kHeaderBytes = 384;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kDescriptorBytes = 256;
constexpr std::uint32_t kAlignment = 64;
constexpr std::uint64_t kGlobalDigestBytes = 32;
constexpr std::uint16_t kMaxRank = 8;
constexpr std::uint64_t kMaxSignedInt64 = (std::uint64_t{1} << 63U) - 1U;
constexpr const char* kCompactStageProfile = "ACTIVE_COLUMNAR_STAGE_V1";
constexpr const char* kCompactStageProfileIdentity =
    "EIEL_SNAPSHOT_CAPSULE_V1_ACTIVE_COLUMNAR_STAGE_V1";
constexpr const char* kCompactStageEncodingPolicy =
    "EXACT_CAPSULE_V1_EXCLUSIVE_FSYNC_CLOSE";
constexpr const char* kCompatibilityStageEncodingPolicy =
    "EXACT_V1_SINGLE_MEMBER_SEQUENTIAL";
constexpr const char* kCompatibilitySnapshotSchema =
    "eiel_cuda_v2_compact_science_snapshot_artifact_v1";
constexpr const char* kCompatibilityFieldSetVersion =
    "cuda_full_evolution_v2_state_projection_v1";
constexpr const char* kCompatibilitySnapshotReason =
    "native_cuda_v2_science_capture_boundary";
constexpr const char* kStateCaptureTransportSchema =
    "eiel_native_snapshot_state_capture_transport_v1";
constexpr const char* kStateCaptureCanonicalPrefix =
    "canonical_checkpoint_prefix_v1";
constexpr const char* kStateCaptureFullResident =
    "full_resident_row_v1";

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kKindCanonicalPlan = 1;
constexpr std::uint32_t kKindStateField = 2;
constexpr std::uint32_t kKindMetricsField = 3;
constexpr std::uint32_t kKindWorldOffsets = 4;
constexpr std::uint32_t kKindActiveSlots = 5;
constexpr std::uint32_t kKindIdentity = 6;
constexpr std::uint32_t kKindOpaqueTyped = 255;

struct DTypeSpec final {
    std::uint16_t id;
    std::uint64_t width;
    const char* name;
};

constexpr std::array<DTypeSpec, 11> kDTypes{{
    {1, 1, "u8"},
    {2, 1, "i8"},
    {3, 2, "u16"},
    {4, 2, "i16"},
    {5, 4, "u32"},
    {6, 4, "i32"},
    {7, 8, "u64"},
    {8, 8, "i64"},
    {9, 4, "f32"},
    {10, 8, "f64"},
    {11, 1, "bool"},
}};

constexpr bool host_is_little_endian() {
    union {
        std::uint16_t word;
        std::uint8_t bytes[2];
    } value{0x0001};
    return value.bytes[0] == 1;
}

std::uint64_t checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label
) {
    TORCH_CHECK(
        left <= std::numeric_limits<std::uint64_t>::max() - right,
        label,
        " overflows uint64"
    );
    return left + right;
}

std::uint64_t checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    const char* label
) {
    TORCH_CHECK(
        left == 0 || right <= std::numeric_limits<std::uint64_t>::max() / left,
        label,
        " overflows uint64"
    );
    return left * right;
}

std::uint64_t align_up(const std::uint64_t value) {
    return checked_add(value, kAlignment - 1, "capsule alignment")
        & ~(static_cast<std::uint64_t>(kAlignment) - 1U);
}

std::uint64_t elapsed_ns(const Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
            .count()
    );
}

std::size_t checked_host_size(const std::uint64_t value, const char* label) {
    TORCH_CHECK(
        value <= std::numeric_limits<std::size_t>::max(),
        label,
        " exceeds host address space"
    );
    return static_cast<std::size_t>(value);
}

void require_range(
    const std::uint64_t offset,
    const std::uint64_t length,
    const std::uint64_t total,
    const char* label
) {
    TORCH_CHECK(offset <= total && length <= total - offset, label, " is out of bounds");
}

void put_u16(std::vector<std::uint8_t>& out, const std::size_t offset, std::uint16_t value) {
    out[offset] = static_cast<std::uint8_t>(value);
    out[offset + 1] = static_cast<std::uint8_t>(value >> 8U);
}

void put_u32(std::vector<std::uint8_t>& out, const std::size_t offset, std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        out[offset + index] = static_cast<std::uint8_t>(value >> (8U * index));
    }
}

void put_u64(std::vector<std::uint8_t>& out, const std::size_t offset, std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        out[offset + index] = static_cast<std::uint8_t>(value >> (8U * index));
    }
}

void append_u16(std::vector<std::uint8_t>& out, const std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value));
    out.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void append_u32(std::vector<std::uint8_t>& out, const std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        out.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    }
}

void append_u64(std::vector<std::uint8_t>& out, const std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        out.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    }
}

void append_digest(
    std::vector<std::uint8_t>& out,
    const std::array<std::uint8_t, 32>& value
) {
    out.insert(out.end(), value.begin(), value.end());
}

std::uint16_t get_u16(const std::uint8_t* data, const std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset])
        | static_cast<std::uint16_t>(data[offset + 1]) << 8U;
}

std::int16_t get_i16(const std::uint8_t* data, const std::size_t offset) {
    return static_cast<std::int16_t>(get_u16(data, offset));
}

std::uint32_t get_u32(const std::uint8_t* data, const std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(data[offset + index]) << (8U * index);
    }
    return value;
}

std::uint64_t get_u64(const std::uint8_t* data, const std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(data[offset + index]) << (8U * index);
    }
    return value;
}

std::int64_t get_i64(const std::uint8_t* data, const std::size_t offset) {
    const auto encoded = get_u64(data, offset);
    std::int64_t value = 0;
    static_assert(sizeof(value) == sizeof(encoded));
    std::memcpy(&value, &encoded, sizeof(value));
    return value;
}

std::string bytes_to_hex(const std::uint8_t* data, const std::size_t size) {
    constexpr char alphabet[] = "0123456789abcdef";
    std::string out(size * 2, '0');
    for (std::size_t index = 0; index < size; ++index) {
        out[2 * index] = alphabet[data[index] >> 4U];
        out[2 * index + 1] = alphabet[data[index] & 0x0FU];
    }
    return out;
}

void append_hex_byte(std::string& out, const std::uint8_t value) {
    constexpr char alphabet[] = "0123456789abcdef";
    out.push_back(alphabet[value >> 4U]);
    out.push_back(alphabet[value & 0x0FU]);
}

std::uint8_t hex_nibble(const char value) {
    if (value >= '0' && value <= '9') {
        return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<std::uint8_t>(value - 'a' + 10);
    }
    TORCH_CHECK(false, "SHA-256 identity must use lowercase hexadecimal");
}

std::array<std::uint8_t, 32> hex_to_digest(const std::string& value) {
    TORCH_CHECK(value.size() == 64, "SHA-256 identity must contain 64 hex characters");
    std::array<std::uint8_t, 32> out{};
    for (std::size_t index = 0; index < out.size(); ++index) {
        out[index] = static_cast<std::uint8_t>(
            (hex_nibble(value[2 * index]) << 4U) | hex_nibble(value[2 * index + 1])
        );
    }
    return out;
}

class Sha256 final {
public:
    Sha256() { reset(); }

    void update(const std::uint8_t* data, std::size_t size) {
        TORCH_CHECK(!finalized_, "SHA-256 context is already finalized");
        total_bytes_ += size;
        while (size > 0) {
            const auto available = block_.size() - block_bytes_;
            const auto take = std::min(size, available);
            std::memcpy(block_.data() + block_bytes_, data, take);
            block_bytes_ += take;
            data += take;
            size -= take;
            if (block_bytes_ == block_.size()) {
                transform(block_.data());
                block_bytes_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> final() {
        TORCH_CHECK(!finalized_, "SHA-256 context is already finalized");
        const auto bit_count = static_cast<std::uint64_t>(total_bytes_) * 8U;
        block_[block_bytes_++] = 0x80U;
        if (block_bytes_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_bytes_), block_.end(), 0);
            transform(block_.data());
            block_bytes_ = 0;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_bytes_),
            block_.begin() + 56,
            0
        );
        for (std::size_t index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(bit_count >> (8U * index));
        }
        transform(block_.data());
        finalized_ = true;
        std::array<std::uint8_t, 32> out{};
        for (std::size_t word = 0; word < state_.size(); ++word) {
            out[word * 4] = static_cast<std::uint8_t>(state_[word] >> 24U);
            out[word * 4 + 1] = static_cast<std::uint8_t>(state_[word] >> 16U);
            out[word * 4 + 2] = static_cast<std::uint8_t>(state_[word] >> 8U);
            out[word * 4 + 3] = static_cast<std::uint8_t>(state_[word]);
        }
        return out;
    }

private:
    static constexpr std::array<std::uint32_t, 64> kRound{{
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
    }};

    static std::uint32_t rotate_right(std::uint32_t value, std::uint32_t count) {
        return (value >> count) | (value << (32U - count));
    }

    void reset() {
        state_ = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
        };
        block_.fill(0);
        block_bytes_ = 0;
        total_bytes_ = 0;
        finalized_ = false;
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> schedule{};
        for (std::size_t index = 0; index < 16; ++index) {
            schedule[index] =
                static_cast<std::uint32_t>(block[index * 4]) << 24U
                | static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U
                | static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U
                | static_cast<std::uint32_t>(block[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < schedule.size(); ++index) {
            const auto s0 = rotate_right(schedule[index - 15], 7)
                ^ rotate_right(schedule[index - 15], 18)
                ^ (schedule[index - 15] >> 3U);
            const auto s1 = rotate_right(schedule[index - 2], 17)
                ^ rotate_right(schedule[index - 2], 19)
                ^ (schedule[index - 2] >> 10U);
            schedule[index] = schedule[index - 16] + s0
                + schedule[index - 7] + s1;
        }
        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < schedule.size(); ++index) {
            const auto upper_e = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto temporary1 = h + upper_e + choice + kRound[index] + schedule[index];
            const auto upper_a = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = upper_a + majority;
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
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_bytes_ = 0;
    std::size_t total_bytes_ = 0;
    bool finalized_ = false;
};

std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t size) {
    std::array<std::uint8_t, SHA256_DIGEST_LENGTH> digest{};
    const auto* result = SHA256(data, size, digest.data());
    TORCH_CHECK(result != nullptr, "OpenSSL SHA-256 failed");
    return digest;
}

std::array<std::uint8_t, 32> sha256(const std::string& value) {
    return sha256(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
}

std::string sha256_hex(const std::uint8_t* data, const std::size_t size) {
    const auto digest = sha256(data, size);
    return bytes_to_hex(digest.data(), digest.size());
}

std::string sha256_hex(const std::string_view value) {
    return sha256_hex(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size()
    );
}

std::string sha256_prefixed(const std::string_view value) {
    return std::string("sha256:") + sha256_hex(value);
}

void append_json_u64(std::string& output, const std::uint64_t value) {
    std::array<char, 32> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value
    );
    TORCH_CHECK(converted.ec == std::errc(), "uint64 JSON formatting failed");
    output.append(buffer.data(), converted.ptr);
}

void append_json_i64(std::string& output, const std::int64_t value) {
    std::array<char, 32> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value
    );
    TORCH_CHECK(converted.ec == std::errc(), "int64 JSON formatting failed");
    output.append(buffer.data(), converted.ptr);
}

void append_python_float(std::string& output, const double value) {
    std::array<char, 64> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general
    );
    TORCH_CHECK(converted.ec == std::errc(), "float JSON formatting failed");
    const std::string_view encoded(
        buffer.data(), static_cast<std::size_t>(converted.ptr - buffer.data())
    );
    output.append(encoded);
    if (encoded.find('.') == std::string_view::npos
        && encoded.find('e') == std::string_view::npos
        && encoded.find('E') == std::string_view::npos) {
        output.append(".0");
    }
}

bool safe_metric_name(const std::string_view value) {
    return !value.empty()
        && std::all_of(value.begin(), value.end(), [](const char child) {
            return (child >= 'a' && child <= 'z')
                || (child >= '0' && child <= '9') || child == '_';
        });
}

std::vector<std::uint8_t> deterministic_gzip_level_9(
    const std::uint8_t* input,
    const std::size_t input_bytes
) {
    // Exact Python GzipFile(compresslevel=9, mtime=0) single-member envelope:
    // DEFLATE, no optional fields, XFL=2, OS=255.
    std::vector<std::uint8_t> output{
        0x1f, 0x8b, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x02, 0xff,
    };
    z_stream stream{};
    const int initialized = deflateInit2(
        &stream,
        Z_BEST_COMPRESSION,
        Z_DEFLATED,
        -MAX_WBITS,
        8,
        Z_DEFAULT_STRATEGY
    );
    TORCH_CHECK(
        initialized == Z_OK,
        "snapshot compatibility zlib initialization failed"
    );
    try {
        std::array<std::uint8_t, 128U * 1024U> buffer{};
        std::size_t consumed = 0;
        while (consumed < input_bytes) {
            const auto chunk = static_cast<uInt>(std::min<std::size_t>(
                input_bytes - consumed,
                std::numeric_limits<uInt>::max()
            ));
            stream.next_in = const_cast<Bytef*>(
                reinterpret_cast<const Bytef*>(input + consumed)
            );
            stream.avail_in = chunk;
            do {
                stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
                stream.avail_out = static_cast<uInt>(buffer.size());
                const int status = deflate(&stream, Z_NO_FLUSH);
                TORCH_CHECK(
                    status == Z_OK,
                    "snapshot compatibility zlib compression failed"
                );
                output.insert(
                    output.end(), buffer.begin(),
                    buffer.begin() + static_cast<std::ptrdiff_t>(
                        buffer.size() - stream.avail_out
                    )
                );
            } while (stream.avail_in != 0);
            consumed += chunk;
        }
        int status = Z_OK;
        while (status != Z_STREAM_END) {
            stream.next_out = reinterpret_cast<Bytef*>(buffer.data());
            stream.avail_out = static_cast<uInt>(buffer.size());
            status = deflate(&stream, Z_FINISH);
            TORCH_CHECK(
                status == Z_OK || status == Z_STREAM_END,
                "snapshot compatibility zlib finalization failed"
            );
            output.insert(
                output.end(), buffer.begin(),
                buffer.begin() + static_cast<std::ptrdiff_t>(
                    buffer.size() - stream.avail_out
                )
            );
        }
    } catch (...) {
        deflateEnd(&stream);
        throw;
    }
    TORCH_CHECK(
        deflateEnd(&stream) == Z_OK,
        "snapshot compatibility zlib teardown failed"
    );

    uLong crc = crc32(0L, Z_NULL, 0);
    std::size_t checksummed = 0;
    while (checksummed < input_bytes) {
        const auto chunk = static_cast<uInt>(std::min<std::size_t>(
            input_bytes - checksummed,
            std::numeric_limits<uInt>::max()
        ));
        crc = crc32(crc, input + checksummed, chunk);
        checksummed += chunk;
    }
    append_u32(output, static_cast<std::uint32_t>(crc));
    append_u32(
        output, static_cast<std::uint32_t>(input_bytes & 0xffffffffU)
    );
    return output;
}

const DTypeSpec& dtype_by_id(const std::uint16_t id) {
    const auto found = std::find_if(
        kDTypes.begin(), kDTypes.end(), [id](const DTypeSpec& item) { return item.id == id; }
    );
    TORCH_CHECK(found != kDTypes.end(), "capsule section dtype is unsupported");
    return *found;
}

const DTypeSpec& dtype_by_name(const std::string& name) {
    const auto found = std::find_if(
        kDTypes.begin(), kDTypes.end(), [&name](const DTypeSpec& item) { return name == item.name; }
    );
    TORCH_CHECK(found != kDTypes.end(), "snapshot input dtype is unsupported: ", name);
    return *found;
}

bool valid_kind(const std::uint32_t kind) {
    return kind == kKindCanonicalPlan || kind == kKindStateField
        || kind == kKindMetricsField || kind == kKindWorldOffsets
        || kind == kKindActiveSlots || kind == kKindIdentity
        || kind == kKindOpaqueTyped;
}

bool valid_utf8(const std::string_view value) {
    std::size_t index = 0;
    while (index < value.size()) {
        const auto byte = static_cast<std::uint8_t>(value[index]);
        if (byte < 0x80U) {
            if (byte == 0) {
                return false;
            }
            ++index;
            continue;
        }
        std::size_t continuation = 0;
        std::uint32_t code = 0;
        if ((byte & 0xE0U) == 0xC0U) {
            continuation = 1;
            code = byte & 0x1FU;
            if (code < 2) {
                return false;
            }
        } else if ((byte & 0xF0U) == 0xE0U) {
            continuation = 2;
            code = byte & 0x0FU;
        } else if ((byte & 0xF8U) == 0xF0U) {
            continuation = 3;
            code = byte & 0x07U;
        } else {
            return false;
        }
        if (index + continuation >= value.size()) {
            return false;
        }
        for (std::size_t step = 1; step <= continuation; ++step) {
            const auto next = static_cast<std::uint8_t>(value[index + step]);
            if ((next & 0xC0U) != 0x80U) {
                return false;
            }
            code = (code << 6U) | (next & 0x3FU);
        }
        if ((continuation == 2 && code < 0x800U)
            || (continuation == 3 && code < 0x10000U)
            || (code >= 0xD800U && code <= 0xDFFFU)
            || code > 0x10FFFFU) {
            return false;
        }
        index += continuation + 1;
    }
    return !value.empty();
}

std::uint64_t exact_uint(const py::handle value, const char* label) {
    TORCH_CHECK(
        PyLong_Check(value.ptr()) && !PyBool_Check(value.ptr()),
        label,
        " must be an integer"
    );
    try {
        return py::cast<std::uint64_t>(value);
    } catch (const py::cast_error&) {
        TORCH_CHECK(false, label, " is outside uint64");
    }
    return 0;
}

std::vector<std::uint64_t> exact_shape(const py::handle value, const char* label) {
    TORCH_CHECK(
        py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value),
        label,
        " must be an array"
    );
    auto sequence = py::reinterpret_borrow<py::sequence>(value);
    TORCH_CHECK(sequence.size() > 0 && sequence.size() <= kMaxRank, label, " rank differs");
    std::vector<std::uint64_t> shape;
    shape.reserve(sequence.size());
    for (const auto item : sequence) {
        const auto dimension = exact_uint(item, label);
        TORCH_CHECK(dimension > 0, label, " dimensions must be positive");
        shape.push_back(dimension);
    }
    return shape;
}

at::ScalarType scalar_type_for_dtype(const std::string& dtype) {
    if (dtype == "u8") return at::ScalarType::Byte;
    if (dtype == "i8") return at::ScalarType::Char;
    if (dtype == "u16") return at::ScalarType::UInt16;
    if (dtype == "i16") return at::ScalarType::Short;
    if (dtype == "u32") return at::ScalarType::UInt32;
    if (dtype == "i32") return at::ScalarType::Int;
    if (dtype == "u64") return at::ScalarType::UInt64;
    if (dtype == "i64") return at::ScalarType::Long;
    if (dtype == "f32") return at::ScalarType::Float;
    if (dtype == "f64") return at::ScalarType::Double;
    if (dtype == "bool") return at::ScalarType::Bool;
    TORCH_CHECK(false, "snapshot tensor dtype is unsupported: ", dtype);
}

struct InputSpec final {
    std::string name;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t bytes = 0;
    std::uint64_t world_stride_bytes = 0;
};

struct StateCaptureModeSpec final {
    std::string name;
    std::uint64_t row_bytes = 0;
    std::uint64_t total_bytes = 0;
};

struct SectionSpec final {
    std::string name;
    std::uint32_t kind = 0;
    std::uint16_t dtype = 0;
    std::uint64_t element_width = 0;
    bool state_source = false;
    std::uint64_t source_offset_bytes = 0;
    std::uint64_t source_world_stride_bytes = 0;
    std::vector<std::uint64_t> element_shape;
    std::uint64_t elements_per_world = 0;
    std::uint64_t payload_bytes_per_world = 0;
};

struct MaterializedSection final {
    std::string name;
    std::uint32_t kind = 0;
    std::uint16_t dtype = 0;
    std::uint64_t element_width = 0;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> payload;
};

struct ParsedSection final {
    std::string name;
    std::uint32_t kind = 0;
    std::uint16_t dtype = 0;
    std::uint64_t element_width = 0;
    std::uint64_t element_count = 0;
    std::vector<std::uint64_t> shape;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_length = 0;
    std::array<std::uint8_t, 32> payload_sha{};
};

struct ParsedCapsule final {
    std::uint64_t absolute_tick = 0;
    std::uint64_t world_count = 0;
    std::uint64_t row_count = 0;
    std::uint64_t directory_offset = 0;
    std::uint64_t directory_bytes = 0;
    std::uint64_t names_offset = 0;
    std::uint64_t names_bytes = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t trailer_offset = 0;
    std::uint64_t capsule_bytes = 0;
    std::uint64_t canonical_plan_index_plus_one = 0;
    std::array<std::uint8_t, 32> plan_sha{};
    std::array<std::uint8_t, 32> schema_sha{};
    std::array<std::uint8_t, 32> layout_sha{};
    std::array<std::uint8_t, 32> control_sha{};
    std::array<std::uint8_t, 32> integrity_sha{};
    std::array<std::uint8_t, 32> capsule_sha{};
    std::vector<ParsedSection> sections;
};

struct ActiveSlot final {
    std::uint64_t agent_id = 0;
    std::uint32_t slot = 0;
    std::array<std::uint8_t, 32> genome_sha{};
    std::array<std::uint8_t, 32> policy_sha{};
};

struct CapturedWorld final {
    std::uint64_t world_index = 0;
    std::uint64_t world_uid = 0;
    std::uint64_t active_count = 0;
    std::array<std::uint8_t, 32> canonical_state_sha{};
};

struct CompactBuildResult final {
    std::vector<std::uint8_t> capsule;
    std::vector<CapturedWorld> captured_worlds;
    std::uint64_t active_count = 0;
    std::uint64_t catalog_count = 0;
    std::uint64_t row_count = 0;
    std::uint64_t section_payload_bytes = 0;
    std::uint64_t scan_scratch_bytes_upper_bound = 0;
    std::uint64_t dense_oracle_payload_bytes = 0;
    std::uint64_t projection_ns = 0;
    std::uint64_t capsule_encoding_ns = 0;
    std::array<std::uint8_t, 32> active_metrics_sha{};
};

struct StageWriteResult final {
    std::uint64_t open_ns = 0;
    std::uint64_t write_ns = 0;
    std::uint64_t fsync_fstat_ns = 0;
    std::uint64_t close_ns = 0;
    std::uint64_t file_device = 0;
    std::uint64_t file_inode = 0;
    std::uint64_t parent_device = 0;
    std::uint64_t parent_inode = 0;
    bool path_revalidated_after_close = false;
};

struct StageReadResult final {
    std::string bytes;
    std::uint64_t open_fstat_ns = 0;
    std::uint64_t read_ns = 0;
    std::uint64_t close_revalidate_ns = 0;
    std::uint64_t file_device = 0;
    std::uint64_t file_inode = 0;
    std::uint64_t parent_device = 0;
    std::uint64_t parent_inode = 0;
    bool path_revalidated_after_close = false;
};

struct CompatibilityConversionResult final {
    StageReadResult input;
    StageWriteResult output;
    std::string canonical_jsonl;
    std::vector<std::uint8_t> gzip;
    std::array<std::uint8_t, 32> uncompressed_sha{};
    std::array<std::uint8_t, 32> compressed_sha{};
    std::array<std::uint8_t, 32> capsule_sha{};
    std::array<std::uint8_t, 32> active_metrics_sha{};
    std::uint64_t absolute_tick = 0;
    std::uint64_t row_count = 0;
    std::uint64_t selected_world_count = 0;
    std::uint64_t active_agent_count = 0;
    std::uint64_t catalog_row_count = 0;
    bool genome_catalog_included = false;
    std::uint64_t capsule_authentication_ns = 0;
    std::uint64_t canonical_jsonl_encoding_ns = 0;
    std::uint64_t gzip_level_9_ns = 0;
    std::uint64_t content_hashing_ns = 0;
    std::uint64_t native_total_ns = 0;
};

class UniqueFd final {
public:
    explicit UniqueFd(const int value = -1) : value_(value) {}
    ~UniqueFd() {
        if (value_ >= 0) {
            ::close(value_);
        }
    }
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    UniqueFd(UniqueFd&& other) noexcept : value_(other.release()) {}
    UniqueFd& operator=(UniqueFd&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }
    int get() const { return value_; }
    int release() {
        const int value = value_;
        value_ = -1;
        return value;
    }
    void reset(const int value = -1) {
        if (value_ >= 0) {
            ::close(value_);
        }
        value_ = value;
    }

private:
    int value_ = -1;
};

struct StageParentHandle final {
    UniqueFd descriptor;
    std::string filename;
    std::uint64_t device = 0;
    std::uint64_t inode = 0;
};

std::uint64_t stat_device(const struct stat& status) {
    return static_cast<std::uint64_t>(status.st_dev);
}

std::uint64_t stat_inode(const struct stat& status) {
    return static_cast<std::uint64_t>(status.st_ino);
}

bool same_file_identity(
    const struct stat& status,
    const std::uint64_t device,
    const std::uint64_t inode
) {
    return stat_device(status) == device && stat_inode(status) == inode;
}

StageParentHandle open_canonical_no_symlink_parent(
    const std::string& stage_path
) {
    const std::filesystem::path requested(stage_path);
    const auto normalized = requested.lexically_normal();
    TORCH_CHECK(
        requested.is_absolute()
            && requested.native() == normalized.native()
            && requested.root_name().empty()
            && requested.root_directory() == std::filesystem::path("/")
            && requested.has_filename(),
        "compact capsule stage path must be lexical-canonical and absolute"
    );
    const auto filename = requested.filename().string();
    TORCH_CHECK(
        !filename.empty() && filename != "." && filename != ".."
            && filename.find('/') == std::string::npos,
        "compact capsule stage filename is invalid"
    );

    int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
    directory_flags |= O_DIRECTORY;
#else
    TORCH_CHECK(false, "compact capsule stage requires O_DIRECTORY support");
#endif
#ifdef O_NOFOLLOW
    directory_flags |= O_NOFOLLOW;
#else
    TORCH_CHECK(false, "compact capsule stage requires O_NOFOLLOW support");
#endif
#ifdef O_CLOEXEC
    directory_flags |= O_CLOEXEC;
#endif
    UniqueFd current(::open("/", directory_flags));
    const int root_error = errno;
    TORCH_CHECK(
        current.get() >= 0,
        "compact capsule stage root directory open failed: ",
        std::strerror(root_error)
    );
    for (const auto& raw_component : requested.parent_path().relative_path()) {
        const auto component = raw_component.string();
        TORCH_CHECK(
            !component.empty() && component != "." && component != ".."
                && component.find('/') == std::string::npos,
            "compact capsule stage parent path is not canonical"
        );
        const int next = ::openat(current.get(), component.c_str(), directory_flags);
        const int component_error = errno;
        TORCH_CHECK(
            next >= 0,
            "compact capsule stage parent component open failed: ",
            component,
            ": ",
            std::strerror(component_error)
        );
        current.reset(next);
    }
    struct stat parent_status {};
    const int parent_stat = ::fstat(current.get(), &parent_status);
    const int parent_stat_error = errno;
    TORCH_CHECK(
        parent_stat == 0 && S_ISDIR(parent_status.st_mode),
        "compact capsule stage parent fstat failed: ",
        std::strerror(parent_stat_error)
    );
    return {
        std::move(current),
        filename,
        stat_device(parent_status),
        stat_inode(parent_status),
    };
}

std::string preserve_owned_stage_entry(
    const StageParentHandle& parent,
    const std::uint64_t owned_device,
    const std::uint64_t owned_inode
) {
    struct stat observed {};
    if (::fstatat(
            parent.descriptor.get(), parent.filename.c_str(), &observed,
            AT_SYMLINK_NOFOLLOW
        ) != 0) {
        const int check_error = errno;
        if (check_error == ENOENT) {
            return "OWNED_ENTRY_ABSENT_NO_CLEANUP";
        }
        return std::string("STAGE_PRESERVATION_IDENTITY_CHECK_FAILED:")
            + std::strerror(check_error);
    }
    if (!same_file_identity(observed, owned_device, owned_inode)) {
        return "FOREIGN_REPLACEMENT_PRESERVED";
    }
    // Never unlink a named stage from this failure path.  fstatat() followed
    // by unlinkat() is not an atomic identity check: a concurrent rename and
    // replacement can occur between the two syscalls and turn cleanup into a
    // destructive operation against a foreign inode.  Failed stages are
    // intentionally preserved for bounded, identity-aware recovery.
    return "OWNED_ENTRY_PRESERVED_FOR_RECOVERY";
}

void verify_owned_stage_entry(
    const StageParentHandle& parent,
    const std::uint64_t owned_device,
    const std::uint64_t owned_inode,
    const std::uint64_t expected_bytes,
    const char* label
) {
    struct stat observed {};
    const int check_status = ::fstatat(
        parent.descriptor.get(), parent.filename.c_str(), &observed,
        AT_SYMLINK_NOFOLLOW
    );
    const int check_error = errno;
    TORCH_CHECK(
        check_status == 0,
        label,
        " is absent or unreadable: ",
        std::strerror(check_error)
    );
    TORCH_CHECK(
        same_file_identity(observed, owned_device, owned_inode),
        label,
        " device/inode identity differs; foreign replacement preserved"
    );
    TORCH_CHECK(
        S_ISREG(observed.st_mode) && observed.st_size >= 0
            && static_cast<std::uint64_t>(observed.st_size) == expected_bytes,
        label,
        " regular-file extent differs"
    );
}

bool same_stable_file_state(
    const struct stat& before,
    const struct stat& after
) {
    return same_file_identity(
            after, stat_device(before), stat_inode(before)
        )
        && after.st_mode == before.st_mode
        && after.st_size == before.st_size
        && after.st_mtim.tv_sec == before.st_mtim.tv_sec
        && after.st_mtim.tv_nsec == before.st_mtim.tv_nsec
        && after.st_ctim.tv_sec == before.st_ctim.tv_sec
        && after.st_ctim.tv_nsec == before.st_ctim.tv_nsec;
}

StageReadResult read_authenticated_stage_candidate(
    const std::string& stage_path,
    const std::uint64_t maximum_bytes
) {
    StageReadResult result;
    const auto open_started = Clock::now();
    auto parent = open_canonical_no_symlink_parent(stage_path);
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#else
    TORCH_CHECK(false, "snapshot compatibility input requires O_NOFOLLOW support");
#endif
    UniqueFd descriptor(::openat(
        parent.descriptor.get(), parent.filename.c_str(), flags
    ));
    const int open_error = errno;
    TORCH_CHECK(
        descriptor.get() >= 0,
        "snapshot compatibility capsule stage open failed: ",
        std::strerror(open_error)
    );
    struct stat before {};
    const int stat_status = ::fstat(descriptor.get(), &before);
    const int stat_error = errno;
    TORCH_CHECK(
        stat_status == 0,
        "snapshot compatibility capsule stage fstat failed: ",
        std::strerror(stat_error)
    );
    TORCH_CHECK(
        S_ISREG(before.st_mode) && before.st_size >= 0,
        "snapshot compatibility capsule stage is not a regular file"
    );
    const auto input_bytes = static_cast<std::uint64_t>(before.st_size);
    TORCH_CHECK(
        input_bytes >= kHeaderBytes + kGlobalDigestBytes
            && input_bytes <= maximum_bytes,
        "snapshot compatibility capsule stage exceeds its plan-bound extent"
    );
    result.file_device = stat_device(before);
    result.file_inode = stat_inode(before);
    result.parent_device = parent.device;
    result.parent_inode = parent.inode;
    result.open_fstat_ns = elapsed_ns(open_started);

    const auto read_started = Clock::now();
    result.bytes.resize(checked_host_size(
        input_bytes, "snapshot compatibility capsule input bytes"
    ));
    std::size_t offset = 0;
    constexpr std::size_t kReadChunk = 1024U * 1024U;
    while (offset < result.bytes.size()) {
        const auto requested = std::min(kReadChunk, result.bytes.size() - offset);
        const auto received = ::pread(
            descriptor.get(), result.bytes.data() + offset, requested,
            static_cast<off_t>(offset)
        );
        if (received < 0 && errno == EINTR) {
            continue;
        }
        const int read_error = errno;
        TORCH_CHECK(
            received > 0,
            "snapshot compatibility capsule stage read failed or truncated: ",
            std::strerror(read_error)
        );
        offset += static_cast<std::size_t>(received);
    }
    struct stat after {};
    const int after_status = ::fstat(descriptor.get(), &after);
    const int after_error = errno;
    TORCH_CHECK(
        after_status == 0 && same_stable_file_state(before, after),
        "snapshot compatibility capsule stage changed during read: ",
        std::strerror(after_error)
    );
    result.read_ns = elapsed_ns(read_started);

    const auto close_started = Clock::now();
    const int closing_descriptor = descriptor.release();
    const int close_status = ::close(closing_descriptor);
    const int close_error = errno;
    TORCH_CHECK(
        close_status == 0,
        "snapshot compatibility capsule stage close failed: ",
        std::strerror(close_error)
    );
    verify_owned_stage_entry(
        parent, result.file_device, result.file_inode, input_bytes,
        "snapshot compatibility capsule path after close"
    );
    auto resolved_parent = open_canonical_no_symlink_parent(stage_path);
    TORCH_CHECK(
        resolved_parent.device == result.parent_device
            && resolved_parent.inode == result.parent_inode,
        "snapshot compatibility capsule absolute parent path identity differs"
    );
    verify_owned_stage_entry(
        resolved_parent, result.file_device, result.file_inode, input_bytes,
        "snapshot compatibility absolute capsule path after close"
    );
    result.path_revalidated_after_close = true;
    result.close_revalidate_ns = elapsed_ns(close_started);
    return result;
}

bool all_zero(const std::uint8_t* begin, const std::uint8_t* end) {
    return std::all_of(begin, end, [](const std::uint8_t value) { return value == 0; });
}

py::dict capsule_format_manifest() {
    py::dict kinds;
    kinds["canonical_plan_json"] = kKindCanonicalPlan;
    kinds["state_field"] = kKindStateField;
    kinds["metrics_field"] = kKindMetricsField;
    kinds["world_offsets"] = kKindWorldOffsets;
    kinds["active_slots"] = kKindActiveSlots;
    kinds["identity"] = kKindIdentity;
    kinds["opaque_typed"] = kKindOpaqueTyped;
    py::dict dtypes;
    for (const auto& dtype : kDTypes) {
        py::dict row;
        row["id"] = dtype.id;
        row["width"] = dtype.width;
        dtypes[dtype.name] = std::move(row);
    }
    py::dict result;
    result["schema_version"] = "eiel_snapshot_capsule_format_manifest_v1";
    result["capsule_name"] = "EIEL_SNAPSHOT_CAPSULE_V1";
    result["magic_hex"] = bytes_to_hex(kMagic.data(), kMagic.size());
    result["version"] = kVersion;
    result["header_bytes"] = kHeaderBytes;
    result["descriptor_bytes"] = kDescriptorBytes;
    result["alignment_bytes"] = kAlignment;
    result["endian_marker"] = kEndianMarker;
    result["digest"] = "sha256";
    result["digest_bytes"] = kGlobalDigestBytes;
    result["max_rank"] = kMaxRank;
    result["projection_backend"] = "HOST_CONTIGUOUS_SYNC_V1";
    result["projection_backend_id"] = static_cast<std::uint32_t>(
        projection_v1::kImplementedBackend
    );
    result["supports_cuda_projection"] = projection_v1::kSupportsCudaProjection;
    result["canonical_plan_embedded"] = false;
    result["compact_canonical_plan_embedded"] = true;
    result["compact_canonical_plan_header_index"] =
        "ONE_BASED_SECTION_INDEX_V1";
    result["plan_identity_payload_bytes"] = 128;
    result["materialization_profile"] = "DENSE_SELECTED_WORLD_PARITY_ORACLE_V1";
    result["active_row_compaction"] = false;
    result["production_compaction_ready"] = false;
    result["copy_memory_contract"] =
        "PARITY_ONLY_SECTION_VECTORS_PLUS_CAPSULE_PLUS_PYTHON_BYTES_V1";
    result["compact_stage_profile"] = kCompactStageProfile;
    result["compact_stage_encoding_policy"] = kCompactStageEncodingPolicy;
    result["compact_zero_extent_sections"] = true;
    result["compact_stage_returns_payload_bytes"] = false;
    result["compact_stage_exclusive_create_content_fsync_close"] = true;
    result["compact_stage_parent_directory_fsync"] = false;
    result["compact_stage_path_identity"] = "POSIX_ST_DEV_ST_INO_V1";
    result["compact_active_row_compaction"] = true;
    result["compact_production_publication_ready"] = false;
    result["compatibility_converter"] =
        "NATIVE_COMPACT_CAPSULE_TO_EXACT_CANONICAL_V1_V1";
    result["compatibility_stage_encoding_policy"] =
        kCompatibilityStageEncodingPolicy;
    result["compatibility_stage_returns_payload_bytes"] = false;
    result["compatibility_stage_exclusive_create_content_fsync_close"] = true;
    result["compatibility_stage_parent_directory_fsync"] = false;
    result["compatibility_gzip_level"] = 9;
    result["compatibility_gzip_mtime"] = 0;
    result["section_kinds"] = std::move(kinds);
    result["dtypes"] = std::move(dtypes);
    return result;
}

InputSpec parse_input(const py::dict& plan, const char* key, const std::uint64_t world_count) {
    TORCH_CHECK(plan.contains(key), "snapshot plan is missing ", key);
    auto raw = py::cast<py::dict>(plan[key]);
    InputSpec out;
    out.name = key;
    out.dtype = py::cast<std::string>(raw["dtype"]);
    const auto& dtype = dtype_by_name(out.dtype);
    out.shape = exact_shape(raw["shape"], key);
    TORCH_CHECK(out.shape.front() == world_count, key, " leading dimension differs");
    std::uint64_t elements = 1;
    for (const auto dimension : out.shape) {
        elements = checked_multiply(elements, dimension, "snapshot input elements");
    }
    out.bytes = checked_multiply(elements, dtype.width, "snapshot input bytes");
    TORCH_CHECK(out.bytes % world_count == 0, key, " bytes are not world-major");
    out.world_stride_bytes = out.bytes / world_count;
    TORCH_CHECK(
        exact_uint(raw["bytes"], key) == out.bytes
            && exact_uint(raw["world_stride_bytes"], key) == out.world_stride_bytes,
        key,
        " normalized byte geometry differs"
    );
    return out;
}

void validate_tensor(const torch::Tensor& tensor, const InputSpec& spec) {
    TORCH_CHECK(tensor.defined(), spec.name, " tensor is undefined");
    TORCH_CHECK(tensor.device().is_cpu(), spec.name, " must be a CPU tensor in G2");
    TORCH_CHECK(tensor.is_contiguous(), spec.name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == scalar_type_for_dtype(spec.dtype), spec.name, " dtype differs");
    TORCH_CHECK(
        static_cast<std::size_t>(tensor.dim()) == spec.shape.size(),
        spec.name,
        " rank differs"
    );
    for (std::size_t index = 0; index < spec.shape.size(); ++index) {
        TORCH_CHECK(
            static_cast<std::uint64_t>(tensor.size(static_cast<std::int64_t>(index)))
                == spec.shape[index],
            spec.name,
            " shape differs"
        );
    }
    TORCH_CHECK(
        checked_multiply(
            static_cast<std::uint64_t>(tensor.numel()),
            static_cast<std::uint64_t>(tensor.element_size()),
            "snapshot tensor bytes"
        ) == spec.bytes,
        spec.name,
        " byte count differs"
    );
}

ParsedCapsule parse_capsule(const std::string& capsule) {
    const auto total = static_cast<std::uint64_t>(capsule.size());
    TORCH_CHECK(total >= kHeaderBytes + kGlobalDigestBytes, "snapshot capsule is truncated");
    const auto* data = reinterpret_cast<const std::uint8_t*>(capsule.data());
    TORCH_CHECK(std::equal(kMagic.begin(), kMagic.end(), data), "snapshot capsule magic differs");
    TORCH_CHECK(get_u16(data, 16) == kVersion, "snapshot capsule version differs");
    TORCH_CHECK(get_u16(data, 18) == kHeaderBytes, "snapshot capsule header size differs");
    TORCH_CHECK(get_u32(data, 20) == kEndianMarker, "snapshot capsule endian marker differs");
    TORCH_CHECK(get_u32(data, 24) == kDescriptorBytes, "snapshot capsule descriptor size differs");
    TORCH_CHECK(get_u32(data, 28) == kAlignment, "snapshot capsule alignment differs");
    TORCH_CHECK(get_u64(data, 32) == 0, "snapshot capsule flags are unsupported");
    TORCH_CHECK(
        get_u32(data, 264) == static_cast<std::uint32_t>(projection_v1::kImplementedBackend),
        "snapshot capsule projection backend differs"
    );
    TORCH_CHECK(get_u32(data, 268) == 0, "snapshot capsule reserved header word is nonzero");
    TORCH_CHECK(all_zero(data + 280, data + kHeaderBytes), "snapshot capsule reserved header bytes are nonzero");

    ParsedCapsule parsed;
    parsed.absolute_tick = get_u64(data, 40);
    parsed.world_count = get_u64(data, 48);
    parsed.row_count = get_u64(data, 56);
    const auto section_count = get_u64(data, 64);
    parsed.directory_offset = get_u64(data, 72);
    parsed.directory_bytes = get_u64(data, 80);
    parsed.names_offset = get_u64(data, 88);
    parsed.names_bytes = get_u64(data, 96);
    parsed.payload_offset = get_u64(data, 104);
    parsed.payload_bytes = get_u64(data, 112);
    parsed.trailer_offset = get_u64(data, 120);
    parsed.capsule_bytes = get_u64(data, 128);
    parsed.canonical_plan_index_plus_one = get_u64(data, 272);
    std::copy_n(data + 136, 32, parsed.plan_sha.begin());
    std::copy_n(data + 168, 32, parsed.schema_sha.begin());
    std::copy_n(data + 200, 32, parsed.layout_sha.begin());
    std::copy_n(data + 232, 32, parsed.control_sha.begin());
    TORCH_CHECK(section_count >= 2, "snapshot capsule omits mandatory sections");
    TORCH_CHECK(
        parsed.absolute_tick <= kMaxSignedInt64,
        "snapshot capsule absolute tick exceeds signed int64"
    );
    TORCH_CHECK(parsed.world_count > 0, "snapshot capsule world count is empty");
    TORCH_CHECK(parsed.row_count >= parsed.world_count, "snapshot capsule row count is invalid");
    TORCH_CHECK(parsed.directory_offset == kHeaderBytes, "snapshot capsule directory offset differs");
    TORCH_CHECK(
        parsed.directory_bytes == checked_multiply(section_count, kDescriptorBytes, "directory bytes"),
        "snapshot capsule directory extent differs"
    );
    TORCH_CHECK(
        parsed.names_offset == checked_add(parsed.directory_offset, parsed.directory_bytes, "names offset"),
        "snapshot capsule names offset differs"
    );
    const auto names_end = checked_add(parsed.names_offset, parsed.names_bytes, "names extent");
    TORCH_CHECK(parsed.payload_offset == align_up(names_end), "snapshot capsule payload offset differs");
    TORCH_CHECK(parsed.payload_offset % kAlignment == 0, "snapshot capsule payload is unaligned");
    TORCH_CHECK(parsed.trailer_offset % kAlignment == 0, "snapshot capsule trailer is unaligned");
    TORCH_CHECK(
        parsed.trailer_offset >= parsed.payload_offset
            && parsed.payload_bytes == parsed.trailer_offset - parsed.payload_offset,
        "snapshot capsule payload extent differs"
    );
    TORCH_CHECK(
        parsed.capsule_bytes == checked_add(parsed.trailer_offset, kGlobalDigestBytes, "capsule bytes")
            && parsed.capsule_bytes == total,
        "snapshot capsule total byte count differs"
    );
    require_range(parsed.directory_offset, parsed.directory_bytes, total, "snapshot capsule directory");
    require_range(parsed.names_offset, parsed.names_bytes, total, "snapshot capsule name table");
    require_range(parsed.payload_offset, parsed.payload_bytes, total, "snapshot capsule payload");
    require_range(parsed.trailer_offset, kGlobalDigestBytes, total, "snapshot capsule trailer");
    TORCH_CHECK(all_zero(data + names_end, data + parsed.payload_offset), "snapshot capsule name padding is nonzero");
    const auto observed_global = sha256(data, static_cast<std::size_t>(parsed.trailer_offset));
    TORCH_CHECK(
        std::equal(observed_global.begin(), observed_global.end(), data + parsed.trailer_offset),
        "snapshot capsule global SHA-256 differs"
    );
    parsed.integrity_sha = observed_global;
    parsed.capsule_sha = sha256(data, capsule.size());

    std::set<std::string> names;
    std::uint64_t expected_name_offset = parsed.names_offset;
    std::uint64_t expected_payload_end = parsed.payload_offset;
    parsed.sections.reserve(checked_host_size(section_count, "capsule section count"));
    for (std::uint64_t index = 0; index < section_count; ++index) {
        const auto descriptor_offset = checked_add(
            parsed.directory_offset,
            checked_multiply(index, kDescriptorBytes, "descriptor offset"),
            "descriptor offset"
        );
        const auto* descriptor = data + descriptor_offset;
        ParsedSection section;
        section.kind = get_u32(descriptor, 0);
        section.dtype = get_u16(descriptor, 4);
        const auto rank = get_u16(descriptor, 6);
        TORCH_CHECK(valid_kind(section.kind), "snapshot capsule section kind differs");
        TORCH_CHECK(rank > 0 && rank <= kMaxRank, "snapshot capsule section rank differs");
        TORCH_CHECK(get_u32(descriptor, 8) == 0 && get_u32(descriptor, 12) == 0, "snapshot capsule section flags differ");
        const auto& dtype = dtype_by_id(section.dtype);
        section.element_width = get_u64(descriptor, 16);
        section.element_count = get_u64(descriptor, 24);
        TORCH_CHECK(section.element_width == dtype.width, "snapshot capsule element width differs");
        const auto name_offset = get_u64(descriptor, 32);
        const auto name_length = get_u64(descriptor, 40);
        section.payload_offset = get_u64(descriptor, 48);
        section.payload_length = get_u64(descriptor, 56);
        TORCH_CHECK(get_u64(descriptor, 64) == section.payload_length, "snapshot capsule logical bytes differ");
        TORCH_CHECK(name_offset == expected_name_offset && name_length > 0, "snapshot capsule name extent differs");
        require_range(name_offset, name_length, names_end, "snapshot capsule section name");
        section.name.assign(
            reinterpret_cast<const char*>(data + name_offset),
            static_cast<std::size_t>(name_length)
        );
        TORCH_CHECK(valid_utf8(section.name), "snapshot capsule section name is not valid UTF-8");
        TORCH_CHECK(names.insert(section.name).second, "snapshot capsule section name is duplicated");
        expected_name_offset = checked_add(name_offset, name_length, "section name extent");
        std::uint64_t elements = 1;
        bool zero_extent = false;
        section.shape.reserve(rank);
        for (std::uint16_t dimension = 0; dimension < kMaxRank; ++dimension) {
            const auto size = get_u64(descriptor, 72 + 8 * dimension);
            if (dimension < rank) {
                section.shape.push_back(size);
                if (size == 0) {
                    zero_extent = true;
                    elements = 0;
                } else if (!zero_extent) {
                    elements = checked_multiply(elements, size, "section elements");
                }
            } else {
                TORCH_CHECK(size == 0, "snapshot capsule unused shape dimension is nonzero");
            }
        }
        TORCH_CHECK(elements == section.element_count, "snapshot capsule element count differs");
        TORCH_CHECK(
            section.payload_length == checked_multiply(elements, dtype.width, "section payload bytes"),
            "snapshot capsule payload length differs"
        );
        const auto aligned_payload_offset = align_up(expected_payload_end);
        TORCH_CHECK(
            all_zero(data + expected_payload_end, data + aligned_payload_offset),
            "snapshot capsule payload padding differs"
        );
        TORCH_CHECK(section.payload_offset == aligned_payload_offset, "snapshot capsule section offset differs");
        require_range(section.payload_offset, section.payload_length, parsed.trailer_offset, "snapshot capsule section payload");
        std::copy_n(descriptor + 136, 32, section.payload_sha.begin());
        TORCH_CHECK(all_zero(descriptor + 168, descriptor + kDescriptorBytes), "snapshot capsule descriptor reserved bytes are nonzero");
        const auto observed = sha256(
            data + section.payload_offset,
            static_cast<std::size_t>(section.payload_length)
        );
        TORCH_CHECK(observed == section.payload_sha, "snapshot capsule section SHA-256 differs");
        expected_payload_end = checked_add(
            section.payload_offset, section.payload_length, "section payload extent"
        );
        parsed.sections.push_back(std::move(section));
    }
    TORCH_CHECK(expected_name_offset == names_end, "snapshot capsule name table has trailing bytes");
    TORCH_CHECK(align_up(expected_payload_end) == parsed.trailer_offset, "snapshot capsule payload has trailing bytes");
    TORCH_CHECK(all_zero(data + expected_payload_end, data + parsed.trailer_offset), "snapshot capsule final padding is nonzero");
    const auto& plan = parsed.sections.front();
    TORCH_CHECK(
        plan.name == "plan_identity"
            && plan.kind == kKindIdentity
            && plan.dtype == 1
            && plan.shape.size() == 1
            && plan.shape[0] == 128
            && plan.payload_length == 128,
        "snapshot capsule plan identity section differs"
    );
    TORCH_CHECK(
        std::equal(parsed.plan_sha.begin(), parsed.plan_sha.end(), data + plan.payload_offset)
            && std::equal(parsed.schema_sha.begin(), parsed.schema_sha.end(), data + plan.payload_offset + 32)
            && std::equal(parsed.layout_sha.begin(), parsed.layout_sha.end(), data + plan.payload_offset + 64)
            && std::equal(parsed.control_sha.begin(), parsed.control_sha.end(), data + plan.payload_offset + 96),
        "snapshot capsule plan identity payload differs from header"
    );
    const auto& selection = parsed.sections[1];
    TORCH_CHECK(
        selection.name == "selected_world_indices"
            && selection.kind == kKindIdentity
            && selection.dtype == 7
            && selection.shape.size() == 1
            && selection.shape[0] == parsed.world_count,
        "snapshot capsule selected-world section differs"
    );
    const auto* selected_data = data + selection.payload_offset;
    std::uint64_t previous_selected = 0;
    bool first_selected = true;
    for (std::uint64_t index = 0; index < parsed.world_count; ++index) {
        const auto selected = get_u64(selected_data, static_cast<std::size_t>(index * 8));
        TORCH_CHECK(
            first_selected || selected > previous_selected,
            "snapshot capsule selected-world order differs"
        );
        previous_selected = selected;
        first_selected = false;
    }
    const auto canonical_count = static_cast<std::size_t>(std::count_if(
        parsed.sections.begin(), parsed.sections.end(), [](const ParsedSection& section) {
            return section.kind == kKindCanonicalPlan;
        }
    ));
    if (parsed.canonical_plan_index_plus_one == 0) {
        TORCH_CHECK(
            canonical_count == 0,
            "snapshot capsule canonical plan section is unindexed"
        );
    } else {
        TORCH_CHECK(
            parsed.canonical_plan_index_plus_one <= parsed.sections.size()
                && canonical_count == 1,
            "snapshot capsule canonical plan index differs"
        );
        const auto& canonical = parsed.sections[checked_host_size(
            parsed.canonical_plan_index_plus_one - 1U,
            "canonical plan section index"
        )];
        TORCH_CHECK(
            canonical.name == "canonical_plan_json"
                && canonical.kind == kKindCanonicalPlan
                && canonical.dtype == 1
                && canonical.shape.size() == 1
                && canonical.payload_sha == parsed.plan_sha,
            "snapshot capsule canonical plan section differs"
        );
    }
    return parsed;
}

py::dict inspect_parsed(const ParsedCapsule& parsed) {
    py::list sections;
    for (const auto& section : parsed.sections) {
        py::dict row;
        row["name"] = section.name;
        row["kind"] = section.kind;
        row["dtype"] = dtype_by_id(section.dtype).name;
        row["dtype_id"] = section.dtype;
        row["element_width"] = section.element_width;
        row["element_count"] = section.element_count;
        row["shape"] = section.shape;
        row["payload_offset"] = section.payload_offset;
        row["payload_length"] = section.payload_length;
        row["payload_sha256"] = bytes_to_hex(section.payload_sha.data(), section.payload_sha.size());
        sections.append(std::move(row));
    }
    py::dict result;
    result["schema_version"] = "eiel_snapshot_capsule_inspection_v1";
    result["capsule_name"] = "EIEL_SNAPSHOT_CAPSULE_V1";
    result["version"] = kVersion;
    result["projection_backend"] = "HOST_CONTIGUOUS_SYNC_V1";
    result["absolute_tick"] = parsed.absolute_tick;
    result["world_count"] = parsed.world_count;
    result["row_count"] = parsed.row_count;
    result["section_count"] = parsed.sections.size();
    result["directory_offset"] = parsed.directory_offset;
    result["directory_bytes"] = parsed.directory_bytes;
    result["names_offset"] = parsed.names_offset;
    result["names_bytes"] = parsed.names_bytes;
    result["payload_offset"] = parsed.payload_offset;
    result["payload_bytes"] = parsed.payload_bytes;
    result["trailer_offset"] = parsed.trailer_offset;
    result["capsule_bytes"] = parsed.capsule_bytes;
    result["canonical_plan_index_plus_one"] =
        parsed.canonical_plan_index_plus_one;
    result["plan_sha256"] = bytes_to_hex(parsed.plan_sha.data(), parsed.plan_sha.size());
    result["schema_sha256"] = bytes_to_hex(parsed.schema_sha.data(), parsed.schema_sha.size());
    result["layout_sha256"] = bytes_to_hex(parsed.layout_sha.data(), parsed.layout_sha.size());
    result["control_sha256"] = bytes_to_hex(parsed.control_sha.data(), parsed.control_sha.size());
    result["integrity_trailer_sha256"] = bytes_to_hex(
        parsed.integrity_sha.data(), parsed.integrity_sha.size()
    );
    result["capsule_sha256"] = bytes_to_hex(parsed.capsule_sha.data(), parsed.capsule_sha.size());
    result["sections"] = std::move(sections);
    return result;
}

py::bytes extract_section(const std::string& capsule, const std::string& name) {
    const auto parsed = parse_capsule(capsule);
    const auto found = std::find_if(
        parsed.sections.begin(), parsed.sections.end(),
        [&name](const ParsedSection& section) { return section.name == name; }
    );
    TORCH_CHECK(found != parsed.sections.end(), "snapshot capsule section is absent: ", name);
    return py::bytes(
        capsule.data() + static_cast<std::ptrdiff_t>(found->payload_offset),
        static_cast<py::ssize_t>(found->payload_length)
    );
}

}  // namespace

class NativeSnapshotPlanV1 final {
public:
    NativeSnapshotPlanV1(std::string canonical_plan, const py::dict& normalized)
        : canonical_plan_(std::move(canonical_plan)) {
        TORCH_CHECK(host_is_little_endian(), "snapshot capsule V1 requires a little-endian host");
        TORCH_CHECK(!canonical_plan_.empty(), "canonical snapshot plan is empty");
        TORCH_CHECK(
            py::cast<std::string>(normalized["schema_version"])
                == "eiel_native_snapshot_plan_v1",
            "snapshot plan schema version differs"
        );
        snapshot_schema_id_ = py::cast<std::string>(normalized["snapshot_schema_id"]);
        layout_id_ = py::cast<std::string>(normalized["layout_id"]);
        world_count_ = exact_uint(normalized["world_count"], "snapshot world_count");
        TORCH_CHECK(world_count_ > 0, "snapshot world_count must be positive");
        plan_sha_ = sha256(canonical_plan_);
        schema_sha_ = sha256(snapshot_schema_id_);
        layout_sha_ = hex_to_digest(py::cast<std::string>(normalized["layout_sha256"]));
        control_sha_ = hex_to_digest(
            py::cast<std::string>(normalized["native_world_control_set_sha256"])
        );
        execution_publication_context_sha256_ = py::cast<std::string>(
            normalized["execution_publication_context_sha256"]
        );
        static_cast<void>(hex_to_digest(execution_publication_context_sha256_));
        state_input_ = parse_input(normalized, "state_input", world_count_);
        metrics_input_ = parse_input(normalized, "metrics_input", world_count_);
        parse_world_uids(normalized);
        parse_boundaries(normalized);
        parse_sections(normalized);
        parse_catalog(normalized);
        validate_v1_row_accounting_sections();
        parse_compact_geometry(normalized);
        parse_state_capture_transport(normalized);
        parse_compatibility_metadata(normalized);
        compute_compatibility_bounds();
    }

    py::dict identities() const {
        py::dict out;
        out["plan_sha256"] = bytes_to_hex(plan_sha_.data(), plan_sha_.size());
        out["schema_sha256"] = bytes_to_hex(schema_sha_.data(), schema_sha_.size());
        out["layout_sha256"] = bytes_to_hex(layout_sha_.data(), layout_sha_.size());
        out["control_sha256"] = bytes_to_hex(control_sha_.data(), control_sha_.size());
        return out;
    }

    py::dict plan_manifest() const {
        py::dict result;
        result["schema_version"] = "eiel_native_snapshot_compiled_plan_receipt_v1";
        result["snapshot_schema_id"] = snapshot_schema_id_;
        result["layout_id"] = layout_id_;
        result["world_count"] = world_count_;
        result["boundary_count"] = boundaries_.size();
        result["declared_section_count"] = sections_.size();
        result["capsule_section_count"] = sections_.size() + 2;
        result["projection_backend"] = "HOST_CONTIGUOUS_SYNC_V1";
        result["supports_cuda_projection"] = false;
        result["canonical_plan_embedded"] = false;
        result["capsule_plan_identity_bytes"] = 128;
        result["canonical_plan_bytes"] = canonical_plan_.size();
        result["materialization_profile"] = "DENSE_SELECTED_WORLD_PARITY_ORACLE_V1";
        result["active_row_compaction"] = false;
        result["production_compaction_ready"] = false;
        result["copy_memory_contract"] =
            "PARITY_ONLY_SECTION_VECTORS_PLUS_CAPSULE_PLUS_PYTHON_BYTES_V1";
        result["compact_stage_profile"] = kCompactStageProfile;
        result["compact_stage_encoding_policy"] = kCompactStageEncodingPolicy;
        py::list accepted_row_bytes;
        py::list accepted_modes;
        for (const auto& mode : state_capture_modes_) {
            accepted_row_bytes.append(mode.row_bytes);
            accepted_modes.append(mode.name);
        }
        result["compact_state_row_bytes_accepted"] = std::move(accepted_row_bytes);
        result["compact_state_capture_modes"] = std::move(accepted_modes);
        result["compact_state_capture_preferred_mode"] =
            state_capture_preferred_mode_;
        result["compact_state_capture_transport_schema_version"] =
            kStateCaptureTransportSchema;
        result["compact_stage_returns_payload_bytes"] = false;
        result["compact_stage_durability"] =
            "EXCLUSIVE_CREATE_CONTENT_FSYNC_CLOSE_DIRECTORY_NOT_FSYNCED_V1";
        result["compact_stage_parent_directory_fsync"] = false;
        result["compact_stage_path_identity"] = "POSIX_ST_DEV_ST_INO_V1";
        result["compact_canonical_plan_embedded"] = true;
        result["compact_reconstruction_authority"] =
            "EMBEDDED_EXACT_CANONICAL_PLAN_PLUS_TYPED_COLUMNS_V1";
        result["compact_active_row_compaction"] = true;
        result["compact_production_publication_ready"] = false;
        result["execution_publication_context_sha256"] =
            execution_publication_context_sha256_;
        py::dict compaction_kinds;
        compaction_kinds["world_offsets"] = kKindWorldOffsets;
        compaction_kinds["active_slots"] = kKindActiveSlots;
        result["g3_compaction_section_kinds"] = std::move(compaction_kinds);
        result["identities"] = identities();
        return result;
    }

    py::bytes canonical_plan_bytes() const {
        return py::bytes(canonical_plan_);
    }

    py::bytes pack_capsule(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        const py::handle absolute_tick_value
    ) const {
        const auto absolute_tick = exact_uint(
            absolute_tick_value, "snapshot absolute tick"
        );
        TORCH_CHECK(
            absolute_tick <= kMaxSignedInt64,
            "snapshot absolute tick exceeds signed int64"
        );
        validate_tensor(state, state_input_);
        validate_tensor(metrics, metrics_input_);
        const auto boundary = boundaries_.find(absolute_tick);
        TORCH_CHECK(boundary != boundaries_.end(), "snapshot tick is absent from compiled plan");
        const auto& selected = boundary->second;
        std::vector<MaterializedSection> materialized;
        materialized.reserve(sections_.size() + 2);
        MaterializedSection plan;
        plan.name = "plan_identity";
        plan.kind = kKindIdentity;
        plan.dtype = 1;
        plan.element_width = 1;
        plan.shape = {128};
        plan.payload.reserve(128);
        plan.payload.insert(plan.payload.end(), plan_sha_.begin(), plan_sha_.end());
        plan.payload.insert(plan.payload.end(), schema_sha_.begin(), schema_sha_.end());
        plan.payload.insert(plan.payload.end(), layout_sha_.begin(), layout_sha_.end());
        plan.payload.insert(plan.payload.end(), control_sha_.begin(), control_sha_.end());
        materialized.push_back(std::move(plan));
        MaterializedSection selection;
        selection.name = "selected_world_indices";
        selection.kind = kKindIdentity;
        selection.dtype = 7;
        selection.element_width = 8;
        selection.shape = {static_cast<std::uint64_t>(selected.size())};
        const auto selection_bytes = checked_multiply(
            static_cast<std::uint64_t>(selected.size()), 8U, "selected-world bytes"
        );
        selection.payload.resize(
            checked_host_size(selection_bytes, "selected-world bytes")
        );
        for (std::size_t index = 0; index < selected.size(); ++index) {
            for (std::size_t byte = 0; byte < 8; ++byte) {
                selection.payload[index * 8 + byte] = static_cast<std::uint8_t>(
                    selected[index] >> (8U * byte)
                );
            }
        }
        materialized.push_back(std::move(selection));
        const auto* state_data = static_cast<const std::uint8_t*>(state.data_ptr());
        const auto* metrics_data = static_cast<const std::uint8_t*>(metrics.data_ptr());
        validate_selected_state_identity(state_data, selected, absolute_tick);
        std::uint64_t row_count = 0;
        {
            // This synchronous parity oracle reads tensors still owned by its
            // Python caller.  Keep the GIL for the complete read/copy/hash
            // interval; the later native service must instead take an
            // explicit immutable buffer lease before it can run asynchronously.
            row_count = derive_exact_v1_row_count(
                state_data, selected, absolute_tick
            );
            for (const auto& spec : sections_) {
                MaterializedSection section;
                section.name = spec.name;
                section.kind = spec.kind;
                section.dtype = spec.dtype;
                section.element_width = spec.element_width;
                section.shape.reserve(spec.element_shape.size() + 1);
                section.shape.push_back(selected.size());
                section.shape.insert(
                    section.shape.end(), spec.element_shape.begin(), spec.element_shape.end()
                );
                const auto payload_bytes = checked_multiply(
                    selected.size(), spec.payload_bytes_per_world, "section selected bytes"
                );
                section.payload.resize(
                    checked_host_size(payload_bytes, "section selected bytes")
                );
                const auto* source = spec.state_source ? state_data : metrics_data;
                for (std::size_t selected_index = 0; selected_index < selected.size(); ++selected_index) {
                    const auto source_offset = checked_add(
                        checked_multiply(
                            selected[selected_index],
                            spec.source_world_stride_bytes,
                            "section source world offset"
                        ),
                        spec.source_offset_bytes,
                        "section source offset"
                    );
                    std::memcpy(
                        section.payload.data() + selected_index * spec.payload_bytes_per_world,
                        source + source_offset,
                        checked_host_size(
                            spec.payload_bytes_per_world, "section bytes per world"
                        )
                    );
                }
                materialized.push_back(std::move(section));
            }
        }
        return encode_capsule(materialized, absolute_tick, selected.size(), row_count);
    }

    py::dict write_owned_compact_capsule_stage_file(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        const py::handle absolute_tick_value,
        const std::string& stage_path,
        const std::string& encoding_policy
    ) const {
        const auto total_started = Clock::now();
        const auto absolute_tick = exact_uint(
            absolute_tick_value, "snapshot absolute tick"
        );
        TORCH_CHECK(
            absolute_tick <= kMaxSignedInt64,
            "snapshot absolute tick exceeds signed int64"
        );
        const auto boundary = boundaries_.find(absolute_tick);
        TORCH_CHECK(
            boundary != boundaries_.end(),
            "snapshot tick is absent from compiled plan"
        );
        TORCH_CHECK(
            encoding_policy == kCompactStageEncodingPolicy,
            "compact capsule stage encoding policy is unsupported"
        );
        TORCH_CHECK(
            !stage_path.empty()
                && stage_path.find('\0') == std::string::npos
                && std::filesystem::path(stage_path).is_absolute()
                && std::filesystem::path(stage_path).has_filename(),
            "compact capsule stage path must be one absolute file path"
        );
        const auto& state_capture_mode = validate_compact_tensors(state, metrics);
        const auto state_row_bytes = static_cast<std::uint64_t>(state.size(1));
        const auto& state_counter = state.unsafeGetTensorImpl()->version_counter();
        const auto& metrics_counter = metrics.unsafeGetTensorImpl()->version_counter();
        TORCH_CHECK(
            state_counter.enabled() && metrics_counter.enabled(),
            "compact capsule owned tensors require enabled version guards"
        );
        const auto state_input_version = state_counter.current_version();
        const auto metrics_input_version = metrics_counter.current_version();

        CompactBuildResult result;
        {
            py::gil_scoped_release release;
            result = build_compact_capsule(
                static_cast<const std::uint8_t*>(state.data_ptr()),
                state_row_bytes,
                static_cast<const std::int64_t*>(metrics.data_ptr()),
                boundary->second,
                absolute_tick
            );
        }
        const auto state_projection_version = state_counter.current_version();
        const auto metrics_projection_version = metrics_counter.current_version();
        TORCH_CHECK(
            state_projection_version == state_input_version,
            "compact capsule state changed during native projection"
        );
        TORCH_CHECK(
            metrics_projection_version == metrics_input_version,
            "compact capsule metrics changed during native projection"
        );

        StageWriteResult stage;
        {
            py::gil_scoped_release release;
            stage = write_stage_file(result.capsule, stage_path);
        }
        const auto capsule_sha = sha256(result.capsule.data(), result.capsule.size());
        TORCH_CHECK(
            result.capsule.size() >= kGlobalDigestBytes,
            "compact capsule output is truncated"
        );
        const auto trailer_offset = result.capsule.size() - kGlobalDigestBytes;

        py::list captured_worlds;
        for (const auto& world : result.captured_worlds) {
            py::dict row;
            row["world_index"] = world.world_index;
            row["world_uid"] = world.world_uid;
            row["active_agent_count"] = world.active_count;
            row["canonical_state_sha256"] = bytes_to_hex(
                world.canonical_state_sha.data(), world.canonical_state_sha.size()
            );
            captured_worlds.append(std::move(row));
        }
        const auto state_input_bytes = checked_multiply(
            world_count_, state_row_bytes, "compact state input bytes"
        );
        const auto metrics_input_bytes = checked_multiply(
            static_cast<std::uint64_t>(metrics.numel()), 8U,
            "compact metrics input bytes"
        );
        const auto selected_canonical_bytes = checked_multiply(
            boundary->second.size(), canonical_state_bytes_per_world_,
            "compact selected canonical bytes"
        );
        const auto selected_metric_bytes = checked_multiply(
            result.active_count,
            checked_multiply(metric_width_, 8U, "compact metric row bytes"),
            "compact selected metric bytes"
        );
        const auto peak_native_owned = checked_add(
            checked_add(
                result.section_payload_bytes,
                result.capsule.size(),
                "compact native owned bytes"
            ),
            result.scan_scratch_bytes_upper_bound,
            "compact native owned bytes"
        );
        py::dict memory;
        memory["borrowed_state_input_bytes"] = state_input_bytes;
        memory["borrowed_state_row_bytes"] = state_row_bytes;
        memory["borrowed_metrics_input_bytes"] = metrics_input_bytes;
        memory["selected_canonical_state_bytes_read"] = selected_canonical_bytes;
        memory["selected_active_metrics_bytes_read"] = selected_metric_bytes;
        memory["materialized_section_payload_bytes"] = result.section_payload_bytes;
        memory["capsule_vector_bytes"] = result.capsule.size();
        memory["scan_scratch_bytes_upper_bound"] = result.scan_scratch_bytes_upper_bound;
        memory["native_heap_peak_bytes_upper_bound"] = peak_native_owned;
        memory["horizon_dependent_bytes"] = 0;

        py::dict timing;
        timing["native_projection_compaction_ns"] = result.projection_ns;
        timing["native_capsule_encoding_hashing_ns"] = result.capsule_encoding_ns;
        timing["stage_open_ns"] = stage.open_ns;
        timing["stage_write_ns"] = stage.write_ns;
        timing["stage_fsync_fstat_ns"] = stage.fsync_fstat_ns;
        timing["stage_close_ns"] = stage.close_ns;
        timing["native_total_ns"] = elapsed_ns(total_started);

        py::dict receipt;
        receipt["schema_version"] = "eiel_native_compact_snapshot_capsule_stage_receipt_v1";
        receipt["capsule_name"] = "EIEL_SNAPSHOT_CAPSULE_V1";
        receipt["materialization_profile"] = kCompactStageProfile;
        receipt["stage_file_state"] =
            "CONTENT_FSYNCED_CLOSED_STAGE_ONLY_NOT_PUBLISHED";
        receipt["stage_durability_scope"] =
            "FILE_CONTENT_ONLY_DIRECTORY_ENTRY_NOT_FSYNCED_V1";
        receipt["stage_parent_directory_fsync_performed"] = false;
        receipt["stage_path"] = stage_path;
        receipt["stage_path_identity_scheme"] = "POSIX_ST_DEV_ST_INO_V1";
        receipt["stage_file_device"] = stage.file_device;
        receipt["stage_file_inode"] = stage.file_inode;
        receipt["stage_parent_directory_device"] = stage.parent_device;
        receipt["stage_parent_directory_inode"] = stage.parent_inode;
        receipt["stage_path_revalidated_after_close"] =
            stage.path_revalidated_after_close;
        receipt["stage_encoding_policy"] = encoding_policy;
        receipt["state_capture_transport_schema_version"] =
            kStateCaptureTransportSchema;
        receipt["state_capture_mode"] = state_capture_mode.name;
        receipt["state_capture_preferred_mode"] =
            state_capture_preferred_mode_;
        receipt["state_capture_mode_is_preferred"] =
            state_capture_mode.name == state_capture_preferred_mode_;
        receipt["absolute_tick"] = absolute_tick;
        receipt["world_count"] = boundary->second.size();
        receipt["active_agent_count"] = result.active_count;
        receipt["catalog_row_count"] = result.catalog_count;
        receipt["active_row_payload_bytes_per_agent"] = checked_add(
            108U,
            checked_multiply(metric_width_, 8U, "compact active row metric bytes"),
            "compact active row bytes"
        );
        receipt["catalog_row_payload_bytes"] = checked_add(
            32U,
            checked_add(
                policy_loci_,
                checked_multiply(trait_loci_, 2U, "compact catalog trait bytes"),
                "compact catalog allele bytes"
            ),
            "compact catalog row bytes"
        );
        receipt["row_count"] = result.row_count;
        receipt["capsule_bytes"] = result.capsule.size();
        receipt["compact_payload_bytes"] = result.section_payload_bytes;
        receipt["embedded_canonical_plan_bytes"] = canonical_plan_.size();
        receipt["self_describing_reconstruction"] = true;
        receipt["dense_oracle_payload_bytes"] = result.dense_oracle_payload_bytes;
        receipt["capsule_sha256"] = bytes_to_hex(capsule_sha.data(), capsule_sha.size());
        receipt["integrity_trailer_sha256"] = bytes_to_hex(
            result.capsule.data() + trailer_offset, kGlobalDigestBytes
        );
        receipt["plan_sha256"] = bytes_to_hex(plan_sha_.data(), plan_sha_.size());
        receipt["execution_publication_context_sha256"] =
            execution_publication_context_sha256_;
        receipt["active_metrics_content_sha256"] = bytes_to_hex(
            result.active_metrics_sha.data(), result.active_metrics_sha.size()
        );
        receipt["active_metrics_authentication"] =
            "NATIVE_COMPACT_SECTION_PAYLOAD_SHA256_V1";
        receipt["captured_worlds"] = std::move(captured_worlds);
        receipt["state_version_guard_enabled"] = true;
        receipt["metrics_version_guard_enabled"] = true;
        receipt["state_input_version"] = state_input_version;
        receipt["state_projection_version"] = state_projection_version;
        receipt["metrics_input_version"] = metrics_input_version;
        receipt["metrics_projection_version"] = metrics_projection_version;
        receipt["gil_released_during_projection_and_stage_io"] = true;
        receipt["input_ownership_contract"] =
            "caller_owned_immutable_until_projection_version_check_v1";
        receipt["payload_bytes_returned_to_python"] = 0;
        receipt["memory"] = std::move(memory);
        receipt["timing"] = std::move(timing);
        return receipt;
    }

    py::dict convert_compact_capsule_stage_file_to_v1_gzip_stage_file(
        const std::string& capsule_stage_path,
        const std::string& gzip_stage_path,
        const std::string& encoding_policy
    ) const {
        TORCH_CHECK(
            encoding_policy == kCompatibilityStageEncodingPolicy,
            "snapshot compatibility stage encoding policy is unsupported"
        );
        TORCH_CHECK(
            !capsule_stage_path.empty() && !gzip_stage_path.empty()
                && capsule_stage_path.find('\0') == std::string::npos
                && gzip_stage_path.find('\0') == std::string::npos
                && capsule_stage_path != gzip_stage_path,
            "snapshot compatibility input and output paths must be distinct"
        );

        CompatibilityConversionResult result;
        {
            py::gil_scoped_release release;
            const auto total_started = Clock::now();
            result.input = read_authenticated_stage_candidate(
                capsule_stage_path, maximum_compact_capsule_bytes_
            );
            const auto authentication_started = Clock::now();
            const auto parsed = parse_capsule(result.input.bytes);
            validate_compact_expected(result.input.bytes, parsed);
            result.absolute_tick = parsed.absolute_tick;
            result.row_count = parsed.row_count;
            result.selected_world_count = parsed.world_count;
            result.capsule_sha = parsed.capsule_sha;
            result.active_metrics_sha = parsed.sections[21].payload_sha;
            result.capsule_authentication_ns = elapsed_ns(
                authentication_started
            );

            const auto encoding_started = Clock::now();
            result.canonical_jsonl = encode_compatibility_jsonl(
                result.input.bytes, parsed, result
            );
            TORCH_CHECK(
                result.canonical_jsonl.size() <= maximum_canonical_jsonl_bytes_,
                "snapshot compatibility canonical JSONL exceeds its plan-bound extent"
            );
            result.canonical_jsonl_encoding_ns = elapsed_ns(encoding_started);

            const auto compression_started = Clock::now();
            result.gzip = deterministic_gzip_level_9(
                reinterpret_cast<const std::uint8_t*>(
                    result.canonical_jsonl.data()
                ),
                result.canonical_jsonl.size()
            );
            result.gzip_level_9_ns = elapsed_ns(compression_started);

            const auto hashing_started = Clock::now();
            result.uncompressed_sha = sha256(result.canonical_jsonl);
            result.compressed_sha = sha256(
                result.gzip.data(), result.gzip.size()
            );
            result.content_hashing_ns = elapsed_ns(hashing_started);
            result.output = write_stage_file(result.gzip, gzip_stage_path);
            result.native_total_ns = elapsed_ns(total_started);
        }

        py::dict input_identity;
        input_identity["scheme"] = "POSIX_ST_DEV_ST_INO_V1";
        input_identity["device"] = result.input.file_device;
        input_identity["inode"] = result.input.file_inode;
        input_identity["parent_device"] = result.input.parent_device;
        input_identity["parent_inode"] = result.input.parent_inode;

        py::dict output_identity;
        output_identity["scheme"] = "POSIX_ST_DEV_ST_INO_V1";
        output_identity["device"] = result.output.file_device;
        output_identity["inode"] = result.output.file_inode;
        output_identity["parent_device"] = result.output.parent_device;
        output_identity["parent_inode"] = result.output.parent_inode;

        const auto native_owned_peak = checked_add(
            checked_add(
                result.input.bytes.size(), result.canonical_jsonl.size(),
                "snapshot compatibility native owned peak"
            ),
            result.gzip.size(),
            "snapshot compatibility native owned peak"
        );
        py::dict memory;
        memory["input_capsule_bytes"] = result.input.bytes.size();
        memory["canonical_jsonl_bytes"] = result.canonical_jsonl.size();
        memory["gzip_bytes"] = result.gzip.size();
        memory["native_owned_payload_bytes_accounted"] = native_owned_peak;
        memory["native_owned_accounting_scope"] =
            "input_capsule_plus_canonical_jsonl_plus_gzip_payloads; "
            "allocator capacity and transient parser/zlib scratch excluded";
        memory["native_zlib_work_buffer_bytes"] = 128U * 1024U;
        memory["maximum_plan_bound_capsule_bytes"] =
            maximum_compact_capsule_bytes_;
        memory["maximum_plan_bound_canonical_jsonl_bytes"] =
            maximum_canonical_jsonl_bytes_;
        memory["horizon_dependent_bytes"] = 0;

        py::dict timing;
        timing["clock"] = "std::chrono::steady_clock";
        timing["input_stage_open_fstat_ns"] = result.input.open_fstat_ns;
        timing["input_stage_read_ns"] = result.input.read_ns;
        timing["input_stage_close_revalidate_ns"] =
            result.input.close_revalidate_ns;
        timing["capsule_authentication_ns"] =
            result.capsule_authentication_ns;
        timing["canonical_jsonl_encoding_ns"] =
            result.canonical_jsonl_encoding_ns;
        timing["gzip_level_9_ns"] = result.gzip_level_9_ns;
        timing["content_hashing_ns"] = result.content_hashing_ns;
        timing["output_stage_open_ns"] = result.output.open_ns;
        timing["output_stage_write_ns"] = result.output.write_ns;
        timing["output_stage_fsync_fstat_ns"] =
            result.output.fsync_fstat_ns;
        timing["output_stage_close_ns"] = result.output.close_ns;
        timing["native_total_ns"] = result.native_total_ns;

        py::dict receipt;
        receipt["schema_version"] =
            "eiel_native_compact_snapshot_v1_compatibility_stage_receipt_v1";
        receipt["implementation"] =
            "NATIVE_COMPACT_CAPSULE_TO_EXACT_CANONICAL_V1_V1";
        receipt["claim_class"] = "TECHNICAL_ONLY";
        receipt["encoding_policy"] = encoding_policy;
        receipt["capsule_stage_path"] = capsule_stage_path;
        receipt["gzip_stage_path"] = gzip_stage_path;
        receipt["input_path_identity"] = std::move(input_identity);
        receipt["output_path_identity"] = std::move(output_identity);
        receipt["input_path_revalidated_after_close"] =
            result.input.path_revalidated_after_close;
        receipt["output_path_revalidated_after_close"] =
            result.output.path_revalidated_after_close;
        receipt["output_stage_file_state"] =
            "CONTENT_FSYNCED_CLOSED_STAGE_ONLY_NOT_PUBLISHED";
        receipt["output_stage_durability_scope"] =
            "FILE_CONTENT_ONLY_DIRECTORY_ENTRY_NOT_FSYNCED_V1";
        receipt["output_parent_directory_fsync_performed"] = false;
        receipt["authenticated_self_describing_capsule"] = true;
        receipt["embedded_canonical_plan_authenticated"] = true;
        receipt["plan_sha256"] = bytes_to_hex(
            plan_sha_.data(), plan_sha_.size()
        );
        receipt["native_world_control_set_sha256"] = bytes_to_hex(
            control_sha_.data(), control_sha_.size()
        );
        receipt["execution_publication_context_sha256"] =
            execution_publication_context_sha256_;
        receipt["absolute_tick"] = result.absolute_tick;
        receipt["row_count"] = result.row_count;
        receipt["captured_world_count"] = result.selected_world_count;
        receipt["active_agent_count"] = result.active_agent_count;
        receipt["catalog_row_count"] = result.catalog_row_count;
        receipt["genome_catalog_included"] = result.genome_catalog_included;
        receipt["capsule_bytes"] = result.input.bytes.size();
        receipt["capsule_sha256"] = bytes_to_hex(
            result.capsule_sha.data(), result.capsule_sha.size()
        );
        receipt["uncompressed_bytes"] = result.canonical_jsonl.size();
        receipt["uncompressed_content_sha256"] = bytes_to_hex(
            result.uncompressed_sha.data(), result.uncompressed_sha.size()
        );
        receipt["compressed_bytes"] = result.gzip.size();
        receipt["compressed_file_sha256"] = bytes_to_hex(
            result.compressed_sha.data(), result.compressed_sha.size()
        );
        receipt["active_metrics_content_sha256"] = bytes_to_hex(
            result.active_metrics_sha.data(), result.active_metrics_sha.size()
        );
        receipt["active_metrics_authentication"] =
            "NATIVE_COMPACT_SECTION_PAYLOAD_SHA256_V1";
        receipt["gzip_level"] = 9;
        receipt["gzip_mtime"] = 0;
        receipt["gzip_single_member"] = true;
        receipt["gzip_header_os"] = 255;
        receipt["zlib_compile_version"] = std::string(ZLIB_VERSION);
        receipt["zlib_runtime_version"] = std::string(zlibVersion());
        receipt["payload_bytes_returned_to_python"] = 0;
        receipt["python_row_materialization"] = false;
        receipt["gil_released_during_read_encode_compress_and_stage_io"] = true;
        receipt["memory"] = std::move(memory);
        receipt["timing"] = std::move(timing);
        return receipt;
    }

    py::dict write_owned_compact_capsule_to_v1_gzip_stage_file(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        const py::handle absolute_tick_value,
        const std::string& stage_path,
        const std::string& encoding_policy
    ) const {
        const auto total_started = Clock::now();
        const auto absolute_tick = exact_uint(
            absolute_tick_value, "snapshot absolute tick"
        );
        TORCH_CHECK(
            absolute_tick <= kMaxSignedInt64,
            "snapshot absolute tick exceeds signed int64"
        );
        const auto boundary = boundaries_.find(absolute_tick);
        TORCH_CHECK(
            boundary != boundaries_.end(),
            "snapshot tick is absent from compiled plan"
        );
        TORCH_CHECK(
            encoding_policy == kCompatibilityStageEncodingPolicy,
            "snapshot compatibility stage encoding policy is unsupported"
        );
        TORCH_CHECK(
            !stage_path.empty()
                && stage_path.find('\0') == std::string::npos
                && std::filesystem::path(stage_path).is_absolute()
                && std::filesystem::path(stage_path).has_filename(),
            "snapshot compatibility stage path must be one absolute file path"
        );
        const auto& state_capture_mode = validate_compact_tensors(state, metrics);
        const auto state_row_bytes = static_cast<std::uint64_t>(state.size(1));
        const auto& state_counter = state.unsafeGetTensorImpl()->version_counter();
        const auto& metrics_counter = metrics.unsafeGetTensorImpl()->version_counter();
        TORCH_CHECK(
            state_counter.enabled() && metrics_counter.enabled(),
            "compact compatibility tensors require enabled version guards"
        );
        const auto state_input_version = state_counter.current_version();
        const auto metrics_input_version = metrics_counter.current_version();

        CompactBuildResult compact;
        {
            py::gil_scoped_release release;
            compact = build_compact_capsule(
                static_cast<const std::uint8_t*>(state.data_ptr()),
                state_row_bytes,
                static_cast<const std::int64_t*>(metrics.data_ptr()),
                boundary->second,
                absolute_tick
            );
        }
        const auto state_projection_version = state_counter.current_version();
        const auto metrics_projection_version = metrics_counter.current_version();
        TORCH_CHECK(
            state_projection_version == state_input_version
                && metrics_projection_version == metrics_input_version,
            "compact compatibility input changed during native projection"
        );

        CompatibilityConversionResult converted;
        {
            py::gil_scoped_release release;
            const auto conversion_started = Clock::now();
            const std::string capsule(
                reinterpret_cast<const char*>(compact.capsule.data()),
                compact.capsule.size()
            );
            const auto authentication_started = Clock::now();
            const auto parsed = parse_capsule(capsule);
            validate_compact_expected(capsule, parsed);
            converted.absolute_tick = parsed.absolute_tick;
            converted.row_count = parsed.row_count;
            converted.selected_world_count = parsed.world_count;
            converted.capsule_sha = parsed.capsule_sha;
            converted.active_metrics_sha = parsed.sections[21].payload_sha;
            converted.capsule_authentication_ns = elapsed_ns(
                authentication_started
            );
            const auto encoding_started = Clock::now();
            converted.canonical_jsonl = encode_compatibility_jsonl(
                capsule, parsed, converted
            );
            TORCH_CHECK(
                converted.canonical_jsonl.size() <= maximum_canonical_jsonl_bytes_,
                "snapshot compatibility canonical JSONL exceeds its plan-bound extent"
            );
            converted.canonical_jsonl_encoding_ns = elapsed_ns(encoding_started);
            const auto compression_started = Clock::now();
            converted.gzip = deterministic_gzip_level_9(
                reinterpret_cast<const std::uint8_t*>(
                    converted.canonical_jsonl.data()
                ),
                converted.canonical_jsonl.size()
            );
            converted.gzip_level_9_ns = elapsed_ns(compression_started);
            const auto hashing_started = Clock::now();
            converted.uncompressed_sha = sha256(converted.canonical_jsonl);
            converted.compressed_sha = sha256(
                converted.gzip.data(), converted.gzip.size()
            );
            converted.content_hashing_ns = elapsed_ns(hashing_started);
            converted.output = write_stage_file(converted.gzip, stage_path);
            converted.native_total_ns = elapsed_ns(conversion_started);
        }
        const auto state_completion_version = state_counter.current_version();
        const auto metrics_completion_version = metrics_counter.current_version();
        TORCH_CHECK(
            state_completion_version == state_input_version
                && metrics_completion_version == metrics_input_version,
            "compact compatibility input changed during native encoding"
        );

        py::list captured_worlds;
        for (const auto& world : compact.captured_worlds) {
            py::dict row;
            row["world_index"] = world.world_index;
            row["world_uid"] = world.world_uid;
            row["active_agent_count"] = world.active_count;
            row["canonical_state_sha256"] = bytes_to_hex(
                world.canonical_state_sha.data(), world.canonical_state_sha.size()
            );
            captured_worlds.append(std::move(row));
        }
        py::list selected_indices;
        py::list selected_uids;
        for (const auto world_index : boundary->second) {
            selected_indices.append(world_index);
            selected_uids.append(world_uids_[checked_host_size(
                world_index, "selected world index"
            )]);
        }
        const auto capsule_bytes = compact.capsule.size();
        const auto state_input_bytes = checked_multiply(
            world_count_, state_row_bytes, "compact compatibility state input bytes"
        );
        const auto metrics_input_bytes = checked_multiply(
            static_cast<std::uint64_t>(metrics.numel()), 8U,
            "compact compatibility metrics input bytes"
        );
        const auto selected_canonical_bytes = checked_multiply(
            boundary->second.size(), canonical_state_bytes_per_world_,
            "compact compatibility selected canonical bytes"
        );
        const auto selected_metric_bytes = checked_multiply(
            compact.active_count,
            checked_multiply(metric_width_, 8U, "compact compatibility metric row bytes"),
            "compact compatibility selected metric bytes"
        );
        const auto native_owned_peak = checked_add(
            checked_add(
                checked_add(
                    compact.section_payload_bytes, capsule_bytes,
                    "compact compatibility native owned bytes"
                ),
                compact.scan_scratch_bytes_upper_bound,
                "compact compatibility native owned bytes"
            ),
            checked_add(
                converted.canonical_jsonl.size(), converted.gzip.size(),
                "compact compatibility native owned bytes"
            ),
            "compact compatibility native owned bytes"
        );

        py::dict input_memory;
        input_memory["borrowed_state_input_bytes"] = state_input_bytes;
        input_memory["borrowed_state_row_bytes"] = state_row_bytes;
        input_memory["borrowed_metrics_input_bytes"] = metrics_input_bytes;
        input_memory["selected_canonical_state_bytes_read"] = selected_canonical_bytes;
        input_memory["selected_active_metrics_bytes_read"] = selected_metric_bytes;
        input_memory["compact_payload_bytes"] = compact.section_payload_bytes;
        input_memory["capsule_bytes"] = capsule_bytes;
        input_memory["canonical_jsonl_bytes"] = converted.canonical_jsonl.size();
        input_memory["gzip_bytes"] = converted.gzip.size();
        input_memory["native_owned_payload_bytes_accounted"] = native_owned_peak;
        input_memory["native_owned_accounting_scope"] =
            "compact_sections_plus_capsule_plus_canonical_jsonl_plus_gzip; "
            "allocator capacity and transient parser/zlib scratch excluded";
        input_memory["native_zlib_work_buffer_bytes"] = 128U * 1024U;
        input_memory["horizon_dependent_bytes"] = 0;

        py::dict timing;
        timing["clock"] = "std::chrono::steady_clock";
        timing["native_projection_compaction_ns"] = compact.projection_ns;
        timing["native_capsule_encoding_hashing_ns"] = compact.capsule_encoding_ns;
        timing["capsule_authentication_ns"] = converted.capsule_authentication_ns;
        timing["canonical_jsonl_encoding_ns"] = converted.canonical_jsonl_encoding_ns;
        timing["gzip_level_9_ns"] = converted.gzip_level_9_ns;
        timing["content_hashing_ns"] = converted.content_hashing_ns;
        timing["native_stage_file_open_ns"] = converted.output.open_ns;
        timing["native_stage_file_write_ns"] = converted.output.write_ns;
        timing["native_stage_file_fsync_and_fstat_ns"] = converted.output.fsync_fstat_ns;
        timing["native_stage_file_close_ns"] = converted.output.close_ns;
        timing["native_encode_write_fsync_close_ns"] = elapsed_ns(total_started);
        timing["native_fused_capsule_to_v1_ns"] = converted.native_total_ns;

        py::dict receipt;
        receipt["schema_version"] =
            "eiel_native_snapshot_v1_stage_file_receipt_v1";
        receipt["implementation"] =
            "NATIVE_COMPACT_CAPSULE_FUSED_TO_EXACT_CANONICAL_V1_V1";
        receipt["claim_class"] = "TECHNICAL_ONLY";
        receipt["stage_path"] = stage_path;
        receipt["stage_encoding_policy"] = encoding_policy;
        receipt["stage_file_state"] = "FSYNCED_STAGE_ONLY_NOT_PUBLISHED";
        receipt["stage_file_fsync_completed"] = true;
        receipt["stage_file_close_completed"] = true;
        receipt["atomic_publication_performed"] = false;
        receipt["directory_fsync_performed"] = false;
        receipt["stage_bytes_written"] = converted.gzip.size();
        receipt["stage_path_identity_scheme"] = "POSIX_ST_DEV_ST_INO_V1";
        receipt["stage_file_device"] = converted.output.file_device;
        receipt["stage_file_inode"] = converted.output.file_inode;
        receipt["stage_parent_directory_device"] = converted.output.parent_device;
        receipt["stage_parent_directory_inode"] = converted.output.parent_inode;
        receipt["stage_path_revalidated_after_close"] =
            converted.output.path_revalidated_after_close;
        receipt["absolute_tick"] = converted.absolute_tick;
        receipt["selected_world_indices"] = std::move(selected_indices);
        receipt["selected_world_uids"] = std::move(selected_uids);
        receipt["row_count"] = converted.row_count;
        receipt["captured_world_count"] = converted.selected_world_count;
        receipt["active_agent_count"] = converted.active_agent_count;
        receipt["catalog_row_count"] = converted.catalog_row_count;
        receipt["genome_catalog_included"] = converted.genome_catalog_included;
        receipt["captured_worlds"] = std::move(captured_worlds);
        py::list metric_fields;
        for (const auto& field : metric_fields_) metric_fields.append(field);
        receipt["metric_fields"] = std::move(metric_fields);
        receipt["plan_sha256"] = bytes_to_hex(plan_sha_.data(), plan_sha_.size());
        receipt["native_world_control_set_sha256"] =
            bytes_to_hex(control_sha_.data(), control_sha_.size());
        receipt["execution_publication_context_sha256"] =
            execution_publication_context_sha256_;
        receipt["uncompressed_bytes"] = converted.canonical_jsonl.size();
        receipt["uncompressed_content_sha256"] = bytes_to_hex(
            converted.uncompressed_sha.data(), converted.uncompressed_sha.size()
        );
        receipt["compressed_bytes"] = converted.gzip.size();
        receipt["compressed_file_sha256"] = bytes_to_hex(
            converted.compressed_sha.data(), converted.compressed_sha.size()
        );
        receipt["capsule_bytes"] = capsule_bytes;
        receipt["capsule_sha256"] = bytes_to_hex(
            converted.capsule_sha.data(), converted.capsule_sha.size()
        );
        receipt["active_metrics_content_sha256"] = bytes_to_hex(
            converted.active_metrics_sha.data(), converted.active_metrics_sha.size()
        );
        receipt["active_metrics_authentication"] =
            "NATIVE_COMPACT_SECTION_PAYLOAD_SHA256_V1";
        receipt["state_capture_transport_schema_version"] =
            kStateCaptureTransportSchema;
        receipt["state_capture_mode"] = state_capture_mode.name;
        receipt["state_capture_preferred_mode"] = state_capture_preferred_mode_;
        receipt["state_capture_mode_is_preferred"] =
            state_capture_mode.name == state_capture_preferred_mode_;
        receipt["state_input_row_bytes"] = state_row_bytes;
        receipt["state_input_bytes"] = state_input_bytes;
        receipt["canonical_state_bytes_selected"] = selected_canonical_bytes;
        receipt["resident_state_row_bytes"] = state_capture_modes_.back().row_bytes;
        receipt["state_version_guard_enabled"] = true;
        receipt["metrics_version_guard_enabled"] = true;
        receipt["state_input_version"] = state_input_version;
        receipt["state_projection_version"] = state_projection_version;
        receipt["state_completion_version"] = state_completion_version;
        receipt["metrics_input_version"] = metrics_input_version;
        receipt["metrics_projection_version"] = metrics_projection_version;
        receipt["metrics_completion_version"] = metrics_completion_version;
        receipt["input_ownership_contract"] =
            "native_service_exclusive_tensor_lease_until_completion_v1";
        receipt["gil_held_for_full_synchronous_call"] = false;
        receipt["payload_bytes_returned_to_python"] = 0;
        receipt["memory"] = std::move(input_memory);
        receipt["timing"] = std::move(timing);
        receipt["python_row_materialization"] = false;
        receipt["gzip_level"] = 9;
        receipt["gzip_mtime"] = 0;
        receipt["gzip_single_member"] = true;
        receipt["gzip_header_os"] = 255;
        receipt["zlib_compile_version"] = std::string(ZLIB_VERSION);
        receipt["zlib_runtime_version"] = std::string(zlibVersion());
        receipt["fused_capsule_stage_io"] = true;
        receipt["capsule_stage_written"] = false;
        return receipt;
    }

    py::dict inspect_capsule(const py::bytes& value) const {
        const std::string capsule = value;
        const auto parsed = parse_capsule(capsule);
        validate_expected(capsule, parsed);
        return inspect_parsed(parsed);
    }

    py::bytes extract_section(const py::bytes& value, const std::string& name) const {
        const std::string capsule = value;
        const auto parsed = parse_capsule(capsule);
        validate_expected(capsule, parsed);
        const auto found = std::find_if(
            parsed.sections.begin(), parsed.sections.end(),
            [&name](const ParsedSection& section) { return section.name == name; }
        );
        TORCH_CHECK(found != parsed.sections.end(), "snapshot capsule section is absent: ", name);
        return py::bytes(
            capsule.data() + static_cast<std::ptrdiff_t>(found->payload_offset),
            static_cast<py::ssize_t>(found->payload_length)
        );
    }

    py::dict inspect_compact_capsule(const py::bytes& value) const {
        const std::string capsule = value;
        const auto parsed = parse_capsule(capsule);
        validate_compact_expected(capsule, parsed);
        auto result = inspect_parsed(parsed);
        result["materialization_profile"] = kCompactStageProfile;
        return result;
    }

    py::bytes extract_compact_section(
        const py::bytes& value,
        const std::string& name
    ) const {
        const std::string capsule = value;
        const auto parsed = parse_capsule(capsule);
        validate_compact_expected(capsule, parsed);
        const auto found = std::find_if(
            parsed.sections.begin(), parsed.sections.end(),
            [&name](const ParsedSection& section) { return section.name == name; }
        );
        TORCH_CHECK(
            found != parsed.sections.end(),
            "compact snapshot capsule section is absent: ", name
        );
        return py::bytes(
            capsule.data() + static_cast<std::ptrdiff_t>(found->payload_offset),
            static_cast<py::ssize_t>(found->payload_length)
        );
    }

private:
    void parse_world_uids(const py::dict& normalized) {
        TORCH_CHECK(normalized.contains("worlds"), "snapshot plan omits worlds");
        const auto worlds = py::cast<py::sequence>(normalized["worlds"]);
        TORCH_CHECK(
            static_cast<std::uint64_t>(worlds.size()) == world_count_,
            "snapshot plan world identities differ"
        );
        world_uids_.reserve(checked_host_size(world_count_, "snapshot world count"));
        world_cell_id_json_.reserve(
            checked_host_size(world_count_, "snapshot world count")
        );
        std::set<std::uint64_t> seen;
        for (std::uint64_t index = 0; index < world_count_; ++index) {
            const auto world = py::cast<py::dict>(worlds[static_cast<py::ssize_t>(index)]);
            TORCH_CHECK(
                exact_uint(world["world_index"], "snapshot world index") == index,
                "snapshot plan world order differs"
            );
            const auto uid = exact_uint(world["world_uid"], "snapshot world UID");
            TORCH_CHECK(
                uid > 0 && seen.insert(uid).second,
                "snapshot plan world UID differs"
            );
            const auto cell_id_json = py::cast<std::string>(
                world["cell_id_json"]
            );
            TORCH_CHECK(
                cell_id_json.size() >= 2
                    && cell_id_json.front() == '"'
                    && cell_id_json.back() == '"'
                    && valid_utf8(cell_id_json)
                    && cell_id_json.find('\0') == std::string::npos,
                "snapshot plan cell identity JSON differs"
            );
            world_uids_.push_back(uid);
            world_cell_id_json_.push_back(cell_id_json);
        }
    }

    void parse_boundaries(const py::dict& normalized) {
        TORCH_CHECK(normalized.contains("snapshot_boundaries"), "snapshot plan omits boundaries");
        auto rows = py::cast<py::sequence>(normalized["snapshot_boundaries"]);
        TORCH_CHECK(rows.size() > 0, "snapshot plan boundaries are empty");
        std::uint64_t previous_tick = 0;
        bool first_tick = true;
        for (const auto raw_row : rows) {
            auto row = py::cast<py::dict>(raw_row);
            const auto tick = exact_uint(row["absolute_tick"], "snapshot boundary tick");
            TORCH_CHECK(
                tick <= kMaxSignedInt64,
                "snapshot boundary tick exceeds signed int64"
            );
            TORCH_CHECK(first_tick || tick > previous_tick, "snapshot boundary ticks are not ordered");
            auto raw_indices = py::cast<py::sequence>(row["world_indices"]);
            TORCH_CHECK(raw_indices.size() > 0, "snapshot boundary selects no worlds");
            std::vector<std::uint64_t> indices;
            indices.reserve(raw_indices.size());
            std::uint64_t previous_world = 0;
            bool first_world = true;
            for (const auto raw_world : raw_indices) {
                const auto world = exact_uint(raw_world, "snapshot world index");
                TORCH_CHECK(world < world_count_, "snapshot world index is outside the plan");
                TORCH_CHECK(first_world || world > previous_world, "snapshot worlds are not in plan order");
                indices.push_back(world);
                previous_world = world;
                first_world = false;
            }
            TORCH_CHECK(boundaries_.emplace(tick, std::move(indices)).second, "snapshot tick is duplicated");
            previous_tick = tick;
            first_tick = false;
        }
    }

    void parse_sections(const py::dict& normalized) {
        auto rows = py::cast<py::sequence>(normalized["sections"]);
        TORCH_CHECK(rows.size() > 0, "snapshot plan sections are empty");
        std::set<std::string> names{"plan_identity", "selected_world_indices"};
        for (const auto raw_row : rows) {
            auto row = py::cast<py::dict>(raw_row);
            SectionSpec section;
            section.name = py::cast<std::string>(row["name"]);
            TORCH_CHECK(valid_utf8(section.name), "snapshot section name is invalid");
            TORCH_CHECK(names.insert(section.name).second, "snapshot section name is duplicated");
            section.kind = static_cast<std::uint32_t>(exact_uint(row["kind"], "section kind"));
            TORCH_CHECK(valid_kind(section.kind) && section.kind != kKindCanonicalPlan, "section kind differs");
            const auto dtype_name = py::cast<std::string>(row["dtype"]);
            const auto& dtype = dtype_by_name(dtype_name);
            section.dtype = dtype.id;
            section.element_width = dtype.width;
            TORCH_CHECK(
                exact_uint(row["dtype_id"], "section dtype") == dtype.id
                    && exact_uint(row["element_width"], "section element width") == dtype.width,
                "section normalized dtype differs"
            );
            const auto source = py::cast<std::string>(row["source"]);
            TORCH_CHECK(source == "state" || source == "metrics", "section source differs");
            section.state_source = source == "state";
            const auto& input = section.state_source ? state_input_ : metrics_input_;
            section.source_offset_bytes = exact_uint(row["source_offset_bytes"], "section source offset");
            section.source_world_stride_bytes = exact_uint(row["source_world_stride_bytes"], "section source stride");
            TORCH_CHECK(section.source_world_stride_bytes == input.world_stride_bytes, "section source stride differs");
            section.element_shape = exact_shape(row["element_shape"], "section element shape");
            TORCH_CHECK(section.element_shape.size() < kMaxRank, "section element rank is too large");
            section.elements_per_world = 1;
            for (const auto dimension : section.element_shape) {
                section.elements_per_world = checked_multiply(
                    section.elements_per_world, dimension, "section elements per world"
                );
            }
            section.payload_bytes_per_world = checked_multiply(
                section.elements_per_world, dtype.width, "section bytes per world"
            );
            TORCH_CHECK(
                section.source_offset_bytes <= input.world_stride_bytes
                    && section.payload_bytes_per_world
                        <= input.world_stride_bytes - section.source_offset_bytes,
                "section exceeds one source world row"
            );
            TORCH_CHECK(
                exact_uint(row["elements_per_world"], "section element count")
                        == section.elements_per_world
                    && exact_uint(row["payload_bytes_per_world"], "section bytes")
                        == section.payload_bytes_per_world,
                "section normalized geometry differs"
            );
            sections_.push_back(std::move(section));
        }
    }

    void parse_catalog(const py::dict& normalized) {
        TORCH_CHECK(normalized.contains("catalog"), "snapshot plan omits catalog law");
        const auto catalog = py::cast<py::dict>(normalized["catalog"]);
        TORCH_CHECK(
            catalog.contains("absolute_tick"),
            "snapshot plan catalog tick is absent"
        );
        const auto raw_tick = catalog["absolute_tick"];
        if (raw_tick.is_none()) {
            catalog_enabled_ = false;
            return;
        }
        catalog_absolute_tick_ = exact_uint(raw_tick, "snapshot catalog tick");
        TORCH_CHECK(
            catalog_absolute_tick_ <= kMaxSignedInt64,
            "snapshot catalog tick exceeds signed int64"
        );
        TORCH_CHECK(
            boundaries_.find(catalog_absolute_tick_) != boundaries_.end(),
            "snapshot catalog tick is absent from boundaries"
        );
        catalog_enabled_ = true;
    }

    const SectionSpec& required_section(const char* name) const {
        const auto found = std::find_if(
            sections_.begin(), sections_.end(),
            [name](const SectionSpec& section) { return section.name == name; }
        );
        TORCH_CHECK(found != sections_.end(), "snapshot plan omits required section ", name);
        return *found;
    }

    void validate_v1_row_accounting_sections() const {
        const auto& header = required_section("scalar_header_counters");
        const auto& alive = required_section("alive_words");
        const auto& agents = required_section("agent_id");
        const auto& policy = required_section("policy_alleles");
        const auto& trait = required_section("trait_alleles");
        TORCH_CHECK(
            header.state_source && header.dtype == 7
                && header.element_shape.size() == 1
                && header.element_shape.front() >= 2,
            "snapshot scalar-header identity geometry differs"
        );
        TORCH_CHECK(
            alive.state_source && alive.dtype == 5 && alive.element_shape.size() == 1,
            "snapshot alive-word row-accounting geometry differs"
        );
        TORCH_CHECK(
            agents.state_source && agents.dtype == 7 && agents.element_shape.size() == 1,
            "snapshot agent-id row-accounting geometry differs"
        );
        const auto capacity = agents.element_shape.front();
        TORCH_CHECK(
            capacity > 0
                && alive.elements_per_world <= std::numeric_limits<std::uint64_t>::max() / 32U
                && alive.elements_per_world * 32U == capacity,
            "snapshot alive-word capacity differs"
        );
        TORCH_CHECK(
            policy.state_source && policy.dtype == 2
                && policy.element_shape.size() == 3
                && policy.element_shape.back() == capacity
                && policy.elements_per_world % capacity == 0,
            "snapshot policy row-accounting geometry differs"
        );
        TORCH_CHECK(
            trait.state_source && trait.dtype == 4
                && trait.element_shape.size() == 3
                && trait.element_shape.back() == capacity
                && trait.elements_per_world % capacity == 0,
            "snapshot trait row-accounting geometry differs"
        );
    }

    void parse_compact_geometry(const py::dict& normalized) {
        TORCH_CHECK(normalized.contains("layout"), "snapshot plan omits layout");
        const auto layout = py::cast<py::dict>(normalized["layout"]);
        TORCH_CHECK(
            py::cast<std::string>(layout["byte_order"]) == "little",
            "compact capsule requires little-endian state"
        );
        canonical_state_bytes_per_world_ = exact_uint(
            layout["canonical_state_bytes_per_world"],
            "canonical state bytes per world"
        );
        capacity_ = exact_uint(layout["agent_capacity"], "snapshot agent capacity");
        resource_count_ = exact_uint(
            layout["resource_cell_count"], "snapshot resource cell count"
        );
        TORCH_CHECK(
            canonical_state_bytes_per_world_ > 0
                && canonical_state_bytes_per_world_ <= state_input_.world_stride_bytes
                && capacity_ > 0
                && capacity_ <= std::numeric_limits<std::uint32_t>::max()
                && resource_count_ > 0,
            "compact capsule layout geometry differs"
        );
        TORCH_CHECK(
            state_input_.dtype == "u8"
                && state_input_.shape.size() == 2,
            "compact capsule state input geometry differs"
        );
        TORCH_CHECK(
            metrics_input_.dtype == "i64"
                && metrics_input_.shape.size() == 3
                && metrics_input_.shape[1] == capacity_,
            "compact capsule metrics input geometry differs"
        );
        metric_width_ = metrics_input_.shape[2];
        TORCH_CHECK(metric_width_ > 0, "compact capsule metric width is empty");
        const auto raw_metric_fields = py::cast<py::sequence>(normalized["metric_fields"]);
        TORCH_CHECK(
            static_cast<std::uint64_t>(raw_metric_fields.size()) == metric_width_,
            "compact capsule metric registry extent differs"
        );

        const auto require_vector = [this](
            const char* name,
            const std::uint16_t dtype,
            const std::uint64_t extent
        ) {
            const auto& field = required_section(name);
            TORCH_CHECK(
                field.state_source && field.dtype == dtype
                    && field.element_shape == std::vector<std::uint64_t>{extent},
                "compact capsule field geometry differs: ", name
            );
        };
        require_vector("scalar_header_counters", 7, 24);
        require_vector("resource", 3, resource_count_);
        require_vector("position", 3, capacity_);
        require_vector("energy", 4, capacity_);
        require_vector("health", 4, capacity_);
        require_vector("age", 3, capacity_);
        require_vector("absolute_birth_tick", 7, capacity_);
        require_vector("agent_id", 7, capacity_);
        require_vector("parent_a_id", 7, capacity_);
        require_vector("parent_b_id", 7, capacity_);
        const auto& policy = required_section("policy_alleles");
        const auto& trait = required_section("trait_alleles");
        policy_loci_ = policy.elements_per_world / capacity_;
        trait_loci_ = trait.elements_per_world / capacity_;
        TORCH_CHECK(
            policy_loci_ > 0 && trait_loci_ > 0,
            "compact capsule allele geometry is empty"
        );
        for (const auto& field : sections_) {
            if (field.state_source) {
                TORCH_CHECK(
                    field.source_offset_bytes <= canonical_state_bytes_per_world_
                        && field.payload_bytes_per_world
                            <= canonical_state_bytes_per_world_
                                - field.source_offset_bytes,
                    "compact capsule required field exceeds canonical state prefix: ",
                    field.name
                );
            }
        }
    }

    void parse_state_capture_transport(const py::dict& normalized) {
        TORCH_CHECK(
            normalized.contains("state_capture_transport"),
            "snapshot plan omits state capture transport"
        );
        const auto transport = py::cast<py::dict>(
            normalized["state_capture_transport"]
        );
        for (const auto* key : {
                 "schema_version",
                 "resident_source_contract",
                 "prefix_semantics",
                 "preferred_mode",
                 "accepted_modes",
             }) {
            TORCH_CHECK(
                transport.contains(key),
                "snapshot state capture transport field is absent: ", key
            );
        }
        TORCH_CHECK(
            transport.size() == 5
                && py::cast<std::string>(transport["schema_version"])
                    == kStateCaptureTransportSchema
                && py::cast<std::string>(
                    transport["resident_source_contract"]
                ) == "state_input"
                && py::cast<std::string>(transport["prefix_semantics"])
                    == "leading_bytes_of_each_resident_world_row_v1",
            "snapshot state capture transport contract differs"
        );
        state_capture_preferred_mode_ = py::cast<std::string>(
            transport["preferred_mode"]
        );
        TORCH_CHECK(
            state_capture_preferred_mode_ == kStateCaptureCanonicalPrefix,
            "snapshot preferred state capture mode differs"
        );
        const auto raw_modes = py::cast<py::sequence>(transport["accepted_modes"]);
        TORCH_CHECK(
            raw_modes.size() == 2,
            "snapshot state capture mode count differs"
        );
        const std::array<std::pair<const char*, std::uint64_t>, 2>
            expected_modes{{
                {kStateCaptureCanonicalPrefix,
                 canonical_state_bytes_per_world_},
                {kStateCaptureFullResident, state_input_.world_stride_bytes},
            }};
        std::set<std::string> names;
        std::set<std::uint64_t> row_extents;
        for (std::size_t index = 0; index < expected_modes.size(); ++index) {
            const auto mode = py::cast<py::dict>(raw_modes[index]);
            for (const auto* key : {
                     "mode",
                     "dtype",
                     "shape",
                     "contiguity",
                     "world_stride_bytes",
                     "total_bytes",
                 }) {
                TORCH_CHECK(
                    mode.contains(key),
                    "snapshot state capture mode field is absent: ", key
                );
            }
            const auto& [expected_name, expected_row_bytes] =
                expected_modes[index];
            const auto name = py::cast<std::string>(mode["mode"]);
            const auto shape = exact_shape(
                mode["shape"], "snapshot state capture mode shape"
            );
            const auto row_bytes = exact_uint(
                mode["world_stride_bytes"],
                "snapshot state capture mode row bytes"
            );
            const auto total_bytes = checked_multiply(
                world_count_, row_bytes,
                "snapshot state capture transport bytes"
            );
            const std::vector<std::uint64_t> expected_shape{
                world_count_, row_bytes
            };
            TORCH_CHECK(
                mode.size() == 6
                    && name == expected_name
                    && py::cast<std::string>(mode["dtype"]) == state_input_.dtype
                    && py::cast<std::string>(mode["contiguity"]) == "C"
                    && shape == expected_shape
                    && row_bytes == expected_row_bytes
                    && exact_uint(
                        mode["total_bytes"],
                        "snapshot state capture mode total bytes"
                    ) == total_bytes,
                "snapshot state capture mode geometry differs: ", expected_name
            );
            TORCH_CHECK(
                names.insert(name).second
                    && row_extents.insert(row_bytes).second,
                "snapshot state capture modes are duplicated or ambiguous"
            );
            state_capture_modes_.push_back({name, row_bytes, total_bytes});
        }
    }

    void parse_compatibility_metadata(const py::dict& normalized) {
        TORCH_CHECK(
            snapshot_schema_id_ == kCompatibilitySnapshotSchema,
            "snapshot compatibility schema differs"
        );
        const auto raw_fields = py::cast<py::sequence>(
            normalized["metric_fields"]
        );
        TORCH_CHECK(
            static_cast<std::uint64_t>(raw_fields.size()) == metric_width_,
            "snapshot compatibility metric registry extent differs"
        );
        metric_fields_.reserve(raw_fields.size());
        sorted_metric_fields_.reserve(raw_fields.size());
        for (std::size_t index = 0; index < raw_fields.size(); ++index) {
            const auto name = py::cast<std::string>(
                raw_fields[static_cast<py::ssize_t>(index)]
            );
            TORCH_CHECK(
                safe_metric_name(name),
                "snapshot compatibility metric name is unsafe"
            );
            metric_name_bytes_ = checked_add(
                metric_name_bytes_, name.size(),
                "snapshot compatibility metric name bytes"
            );
            metric_fields_.push_back(name);
            sorted_metric_fields_.emplace_back(name, index);
        }
        TORCH_CHECK(
            !metric_fields_.empty(),
            "snapshot compatibility metric registry is empty"
        );
        std::sort(sorted_metric_fields_.begin(), sorted_metric_fields_.end());
        for (std::size_t index = 1; index < sorted_metric_fields_.size(); ++index) {
            TORCH_CHECK(
                sorted_metric_fields_[index - 1].first
                    != sorted_metric_fields_[index].first,
                "snapshot compatibility metric registry contains a duplicate"
            );
        }
    }

    void compute_compatibility_bounds() {
        const auto maximum_agents = checked_multiply(
            world_count_, capacity_,
            "snapshot compatibility maximum active agents"
        );
        const auto active_row_bytes = checked_add(
            108U,
            checked_multiply(
                metric_width_, 8U,
                "snapshot compatibility active metric bytes"
            ),
            "snapshot compatibility active row bytes"
        );
        const auto catalog_row_bytes = checked_add(
            32U,
            checked_add(
                policy_loci_,
                checked_multiply(
                    trait_loci_, 2U,
                    "snapshot compatibility catalog trait bytes"
                ),
                "snapshot compatibility catalog allele bytes"
            ),
            "snapshot compatibility catalog row bytes"
        );
        maximum_compact_capsule_bytes_ = checked_add(
            checked_add(
                canonical_plan_.size(), 2U * 1024U * 1024U,
                "snapshot compatibility capsule fixed bound"
            ),
            checked_multiply(
                maximum_agents,
                checked_add(
                    active_row_bytes, catalog_row_bytes,
                    "snapshot compatibility maximum row bytes"
                ),
                "snapshot compatibility maximum capsule row bytes"
            ),
            "snapshot compatibility maximum capsule bytes"
        );

        maximum_canonical_jsonl_bytes_ = 2U * 1024U * 1024U;
        for (const auto& cell_id_json : world_cell_id_json_) {
            maximum_canonical_jsonl_bytes_ = checked_add(
                maximum_canonical_jsonl_bytes_,
                checked_add(
                    cell_id_json.size(), 4096U,
                    "snapshot compatibility world JSON bound"
                ),
                "snapshot compatibility JSON bound"
            );
            const auto agent_json_bound = checked_add(
                cell_id_json.size(),
                checked_add(
                    metric_name_bytes_, 4096U,
                    "snapshot compatibility agent JSON bound"
                ),
                "snapshot compatibility agent JSON bound"
            );
            maximum_canonical_jsonl_bytes_ = checked_add(
                maximum_canonical_jsonl_bytes_,
                checked_multiply(
                    capacity_, agent_json_bound,
                    "snapshot compatibility agent JSON bound"
                ),
                "snapshot compatibility JSON bound"
            );
            const auto catalog_json_bound = checked_add(
                cell_id_json.size(),
                checked_add(
                    checked_multiply(
                        policy_loci_, 5U,
                        "snapshot compatibility policy JSON bound"
                    ),
                    checked_add(
                        checked_multiply(
                            trait_loci_, 7U,
                            "snapshot compatibility trait JSON bound"
                        ),
                        2048U,
                        "snapshot compatibility catalog JSON bound"
                    ),
                    "snapshot compatibility catalog JSON bound"
                ),
                "snapshot compatibility catalog JSON bound"
            );
            maximum_canonical_jsonl_bytes_ = checked_add(
                maximum_canonical_jsonl_bytes_,
                checked_multiply(
                    capacity_, catalog_json_bound,
                    "snapshot compatibility catalog JSON bound"
                ),
                "snapshot compatibility JSON bound"
            );
        }
        TORCH_CHECK(
            maximum_compact_capsule_bytes_
                    <= static_cast<std::uint64_t>(
                        std::numeric_limits<off_t>::max()
                    )
                && maximum_canonical_jsonl_bytes_
                    <= std::numeric_limits<std::size_t>::max(),
            "snapshot compatibility bounds exceed the host address space"
        );
    }

    const StateCaptureModeSpec& validate_compact_tensors(
        const torch::Tensor& state,
        const torch::Tensor& metrics
    ) const {
        TORCH_CHECK(
            state.defined() && state.device().is_cpu()
                && state.scalar_type() == at::ScalarType::Byte
                && state.is_contiguous() && state.dim() == 2
                && static_cast<std::uint64_t>(state.size(0)) == world_count_,
            "compact capsule state must be CPU-contiguous uint8 [world,row_bytes]"
        );
        const auto state_row_bytes = static_cast<std::uint64_t>(state.size(1));
        const auto found = std::find_if(
            state_capture_modes_.begin(), state_capture_modes_.end(),
            [state_row_bytes](const StateCaptureModeSpec& mode) {
                return mode.row_bytes == state_row_bytes;
            }
        );
        TORCH_CHECK(
            found != state_capture_modes_.end(),
            "compact capsule state must match one plan-declared "
            "CPU-contiguous uint8 capture transport"
        );
        TORCH_CHECK(
            checked_multiply(
                world_count_, state_row_bytes,
                "compact capsule state transport bytes"
            ) == found->total_bytes,
            "compact capsule state transport byte extent differs"
        );
        validate_tensor(metrics, metrics_input_);
        return *found;
    }

    void validate_selected_state_identity(
        const std::uint8_t* state_data,
        const std::vector<std::uint64_t>& selected,
        const std::uint64_t absolute_tick
    ) const {
        const auto& header = required_section("scalar_header_counters");
        for (const auto world_index : selected) {
            const auto source_offset = checked_add(
                checked_multiply(
                    world_index,
                    header.source_world_stride_bytes,
                    "snapshot identity world offset"
                ),
                header.source_offset_bytes,
                "snapshot identity source offset"
            );
            const auto* header_data = state_data + checked_host_size(
                source_offset, "snapshot identity source offset"
            );
            const auto observed_world_uid = get_u64(header_data, 0);
            TORCH_CHECK(
                observed_world_uid
                    == world_uids_[checked_host_size(
                        world_index, "snapshot identity world index"
                    )],
                "snapshot state world identity differs at plan index ",
                world_index
            );
            const auto observed_next_absolute_tick = get_u64(header_data, 8);
            TORCH_CHECK(
                observed_next_absolute_tick == absolute_tick,
                "snapshot state next_absolute_tick differs at plan index ",
                world_index
            );
        }
    }

    std::array<std::uint8_t, 32> genome_state_digest_v1(
        const std::uint8_t* world,
        const std::uint64_t slot,
        const std::uint64_t capacity
    ) const {
        const auto& policy = required_section("policy_alleles");
        const auto& trait = required_section("trait_alleles");
        const auto policy_loci = policy.elements_per_world / capacity;
        const auto trait_loci = trait.elements_per_world / capacity;
        std::string canonical{"{\"policy\":\""};
        const auto canonical_bytes = checked_add(
            24U,
            checked_add(
                checked_multiply(2U, policy_loci, "policy genome hex bytes"),
                checked_multiply(4U, trait_loci, "trait genome hex bytes"),
                "genome hex bytes"
            ),
            "genome canonical bytes"
        );
        canonical.reserve(checked_host_size(canonical_bytes, "genome canonical bytes"));
        for (std::uint64_t locus = 0; locus < policy_loci; ++locus) {
            const auto offset = checked_add(
                policy.source_offset_bytes,
                checked_add(
                    checked_multiply(locus, capacity, "policy locus offset"),
                    slot,
                    "policy slot offset"
                ),
                "policy source offset"
            );
            append_hex_byte(canonical, world[static_cast<std::size_t>(offset)]);
        }
        canonical.append("\",\"trait\":\"");
        for (std::uint64_t locus = 0; locus < trait_loci; ++locus) {
            const auto element = checked_add(
                checked_multiply(locus, capacity, "trait locus offset"),
                slot,
                "trait slot offset"
            );
            const auto offset = checked_add(
                trait.source_offset_bytes,
                checked_multiply(element, 2U, "trait byte offset"),
                "trait source offset"
            );
            append_hex_byte(canonical, world[static_cast<std::size_t>(offset)]);
            append_hex_byte(canonical, world[static_cast<std::size_t>(offset + 1U)]);
        }
        canonical.append("\"}");
        return sha256(canonical);
    }

    std::uint64_t derive_exact_v1_row_count(
        const std::uint8_t* state_data,
        const std::vector<std::uint64_t>& selected,
        const std::uint64_t absolute_tick
    ) const {
        const auto& alive = required_section("alive_words");
        const auto& agents = required_section("agent_id");
        const auto capacity = agents.element_shape.front();
        const bool include_catalog =
            catalog_enabled_ && catalog_absolute_tick_ == absolute_tick;
        auto rows = static_cast<std::uint64_t>(selected.size());
        for (const auto world_index : selected) {
            const auto world_offset = checked_multiply(
                world_index, state_input_.world_stride_bytes, "row-accounting world offset"
            );
            const auto* world = state_data + checked_host_size(
                world_offset, "row-accounting world offset"
            );
            const auto* alive_data = world + checked_host_size(
                alive.source_offset_bytes, "alive-word source offset"
            );
            const auto* agent_data = world + checked_host_size(
                agents.source_offset_bytes, "agent-id source offset"
            );
            std::unordered_set<std::uint64_t> seen_agent_ids;
            std::set<std::array<std::uint8_t, 32>> catalog_genomes;
            for (std::uint64_t slot = 0; slot < capacity; ++slot) {
                const auto agent_id = get_u64(
                    agent_data, static_cast<std::size_t>(slot * 8U)
                );
                if (agent_id == 0) {
                    continue;
                }
                TORCH_CHECK(
                    seen_agent_ids.insert(agent_id).second,
                    "native state contains duplicate agent identity ", agent_id
                );
                const auto alive_word = get_u32(
                    alive_data, static_cast<std::size_t>((slot / 32U) * 4U)
                );
                if ((alive_word & (std::uint32_t{1} << (slot % 32U))) == 0) {
                    continue;
                }
                rows = checked_add(rows, 1U, "snapshot V1 row count");
                if (include_catalog) {
                    catalog_genomes.insert(
                        genome_state_digest_v1(world, slot, capacity)
                    );
                }
            }
            if (include_catalog) {
                rows = checked_add(
                    rows,
                    static_cast<std::uint64_t>(catalog_genomes.size()),
                    "snapshot V1 catalog row count"
                );
            }
        }
        return rows;
    }

    std::array<std::uint8_t, 32> policy_digest_v1(
        const std::uint8_t* world,
        const ActiveSlot& active
    ) const {
        const auto& policy = required_section("policy_alleles");
        std::vector<std::uint8_t> policy_bytes;
        policy_bytes.reserve(checked_host_size(policy_loci_, "policy loci"));
        for (std::uint64_t locus = 0; locus < policy_loci_; ++locus) {
            const auto offset = checked_add(
                policy.source_offset_bytes,
                checked_add(
                    checked_multiply(locus, capacity_, "policy locus offset"),
                    active.slot,
                    "policy slot offset"
                ),
                "policy source offset"
            );
            policy_bytes.push_back(world[checked_host_size(offset, "policy source offset")]);
        }
        return sha256(policy_bytes.data(), policy_bytes.size());
    }

    CompactBuildResult build_compact_capsule(
        const std::uint8_t* state_data,
        const std::uint64_t state_row_bytes,
        const std::int64_t* metrics_data,
        const std::vector<std::uint64_t>& selected,
        const std::uint64_t absolute_tick
    ) const {
        const auto projection_started = Clock::now();
        const bool include_catalog =
            catalog_enabled_ && catalog_absolute_tick_ == absolute_tick;
        std::vector<std::vector<ActiveSlot>> active_by_world;
        std::vector<std::map<std::array<std::uint8_t, 32>, std::uint32_t>>
            catalog_by_world;
        active_by_world.reserve(selected.size());
        catalog_by_world.reserve(selected.size());
        CompactBuildResult result;
        result.captured_worlds.reserve(selected.size());

        const auto& header = required_section("scalar_header_counters");
        const auto& alive = required_section("alive_words");
        const auto& agents = required_section("agent_id");
        for (const auto world_index : selected) {
            const auto world_offset = checked_multiply(
                world_index, state_row_bytes, "compact world source offset"
            );
            const auto* world = state_data + checked_host_size(
                world_offset, "compact world source offset"
            );
            const auto observed_world_uid = get_u64(
                world, checked_host_size(header.source_offset_bytes, "header offset")
            );
            TORCH_CHECK(
                observed_world_uid == world_uids_[checked_host_size(
                    world_index, "compact world index"
                )],
                "compact snapshot state world identity differs at plan index ",
                world_index
            );
            TORCH_CHECK(
                get_u64(
                    world,
                    checked_host_size(header.source_offset_bytes + 8U, "header tick offset")
                ) == absolute_tick,
                "compact snapshot state next_absolute_tick differs at plan index ",
                world_index
            );
            const auto* alive_data = world + checked_host_size(
                alive.source_offset_bytes, "alive offset"
            );
            const auto* agent_data = world + checked_host_size(
                agents.source_offset_bytes, "agent offset"
            );
            std::unordered_set<std::uint64_t> seen_agent_ids;
            std::vector<ActiveSlot> active;
            active.reserve(checked_host_size(capacity_, "agent capacity"));
            for (std::uint64_t slot = 0; slot < capacity_; ++slot) {
                const auto agent_id = get_u64(
                    agent_data, checked_host_size(slot * 8U, "agent-id offset")
                );
                if (agent_id == 0) {
                    continue;
                }
                TORCH_CHECK(
                    seen_agent_ids.insert(agent_id).second,
                    "native state contains duplicate agent identity ", agent_id
                );
                const auto alive_word = get_u32(
                    alive_data,
                    checked_host_size((slot / 32U) * 4U, "alive-word offset")
                );
                if ((alive_word & (std::uint32_t{1} << (slot % 32U))) == 0) {
                    continue;
                }
                ActiveSlot row;
                row.agent_id = agent_id;
                row.slot = static_cast<std::uint32_t>(slot);
                row.genome_sha = genome_state_digest_v1(world, slot, capacity_);
                row.policy_sha = policy_digest_v1(world, row);
                active.push_back(row);
            }
            const auto by_agent_id = [](const auto& left, const auto& right) {
                return left.agent_id < right.agent_id;
            };
            if (!std::is_sorted(active.begin(), active.end(), by_agent_id)) {
                std::sort(active.begin(), active.end(), by_agent_id);
            }
            std::map<std::array<std::uint8_t, 32>, std::uint32_t> catalog;
            if (include_catalog) {
                for (const auto& row : active) {
                    catalog.emplace(row.genome_sha, row.slot);
                }
            }
            result.active_count = checked_add(
                result.active_count, active.size(), "compact active count"
            );
            result.catalog_count = checked_add(
                result.catalog_count, catalog.size(), "compact catalog count"
            );
            const auto canonical_sha = sha256(
                world, checked_host_size(
                    canonical_state_bytes_per_world_, "canonical state bytes"
                )
            );
            result.captured_worlds.push_back({
                world_index,
                observed_world_uid,
                static_cast<std::uint64_t>(active.size()),
                canonical_sha,
            });
            active_by_world.push_back(std::move(active));
            catalog_by_world.push_back(std::move(catalog));
        }
        result.row_count = checked_add(
            selected.size(),
            checked_add(
                result.active_count, result.catalog_count, "compact row count"
            ),
            "compact row count"
        );
        result.scan_scratch_bytes_upper_bound = checked_add(
            checked_multiply(
                selected.size(),
                checked_multiply(
                    capacity_, sizeof(ActiveSlot) + 64U,
                    "compact scan scratch"
                ),
                "compact scan scratch"
            ),
            checked_multiply(
                result.catalog_count, 96U, "compact catalog scratch"
            ),
            "compact scan scratch"
        );

        std::vector<MaterializedSection> sections;
        sections.reserve(include_catalog ? 26 : 22);
        const auto add_section = [&sections](
            const char* name,
            const std::uint32_t kind,
            const std::uint16_t dtype,
            const std::uint64_t width,
            std::vector<std::uint64_t> shape
        ) -> std::size_t {
            MaterializedSection section;
            section.name = name;
            section.kind = kind;
            section.dtype = dtype;
            section.element_width = width;
            section.shape = std::move(shape);
            std::uint64_t elements = 1;
            for (const auto dimension : section.shape) {
                elements = checked_multiply(elements, dimension, "compact section elements");
            }
            section.payload.reserve(checked_host_size(
                checked_multiply(elements, width, "compact section bytes"),
                "compact section bytes"
            ));
            sections.push_back(std::move(section));
            return sections.size() - 1;
        };
        const auto world_extent = static_cast<std::uint64_t>(selected.size());
        const auto plan_index = add_section(
            "plan_identity", kKindIdentity, 1, 1, {128}
        );
        auto& plan = sections[plan_index].payload;
        append_digest(plan, plan_sha_);
        append_digest(plan, schema_sha_);
        append_digest(plan, layout_sha_);
        append_digest(plan, control_sha_);
        const auto selection_index = add_section(
            "selected_world_indices", kKindIdentity, 7, 8, {world_extent}
        );
        for (const auto index : selected) append_u64(sections[selection_index].payload, index);
        const auto profile_index = add_section(
            "compact_profile_identity", kKindIdentity, 1, 1, {32}
        );
        append_digest(sections[profile_index].payload, sha256(std::string{kCompactStageProfileIdentity}));
        const auto reconstruction_index = add_section(
            "canonical_plan_json", kKindCanonicalPlan, 1, 1,
            {static_cast<std::uint64_t>(canonical_plan_.size())}
        );
        sections[reconstruction_index].payload.insert(
            sections[reconstruction_index].payload.end(),
            canonical_plan_.begin(),
            canonical_plan_.end()
        );
        const auto offsets_index = add_section(
            "world_offsets", kKindWorldOffsets, 7, 8, {world_extent + 1U}
        );
        append_u64(sections[offsets_index].payload, 0);
        std::uint64_t active_offset = 0;
        for (const auto& active : active_by_world) {
            active_offset = checked_add(active_offset, active.size(), "world active offset");
            append_u64(sections[offsets_index].payload, active_offset);
        }
        const auto slots_index = add_section(
            "active_slots", kKindActiveSlots, 5, 4, {result.active_count}
        );
        const auto world_uid_index = add_section(
            "world_uid", kKindStateField, 7, 8, {world_extent}
        );
        const auto canonical_sha_index = add_section(
            "canonical_state_sha256", kKindOpaqueTyped, 1, 1, {world_extent, 32}
        );
        const auto resource_sha_index = add_section(
            "resource_state_sha256", kKindOpaqueTyped, 1, 1, {world_extent, 32}
        );
        const auto header_index = add_section(
            "world_header_counts", kKindStateField, 7, 8, {world_extent, 6}
        );
        const auto resource_summary_index = add_section(
            "resource_summary", kKindStateField, 7, 8, {world_extent, 4}
        );
        const auto agent_index = add_section(
            "agent_id", kKindStateField, 7, 8, {result.active_count}
        );
        const auto position_index = add_section(
            "position", kKindStateField, 3, 2, {result.active_count}
        );
        const auto energy_index = add_section(
            "energy", kKindStateField, 4, 2, {result.active_count}
        );
        const auto health_index = add_section(
            "health", kKindStateField, 4, 2, {result.active_count}
        );
        const auto age_index = add_section(
            "age_ticks", kKindStateField, 3, 2, {result.active_count}
        );
        const auto birth_index = add_section(
            "absolute_birth_tick", kKindStateField, 7, 8, {result.active_count}
        );
        const auto parent_a_index = add_section(
            "parent_a_id", kKindStateField, 7, 8, {result.active_count}
        );
        const auto parent_b_index = add_section(
            "parent_b_id", kKindStateField, 7, 8, {result.active_count}
        );
        const auto genome_sha_index = add_section(
            "genome_state_sha256", kKindOpaqueTyped, 1, 1, {result.active_count, 32}
        );
        const auto policy_sha_index = add_section(
            "policy_sha256", kKindOpaqueTyped, 1, 1, {result.active_count, 32}
        );
        const auto metrics_index = add_section(
            "lifetime_metrics", kKindMetricsField, 8, 8,
            {result.active_count, metric_width_}
        );

        std::size_t catalog_offsets_index = 0;
        std::size_t catalog_sha_index = 0;
        std::size_t catalog_policy_index = 0;
        std::size_t catalog_trait_index = 0;
        if (include_catalog) {
            catalog_offsets_index = add_section(
                "catalog_world_offsets", kKindWorldOffsets, 7, 8, {world_extent + 1U}
            );
            catalog_sha_index = add_section(
                "catalog_genome_sha256", kKindOpaqueTyped, 1, 1,
                {result.catalog_count, 32}
            );
            catalog_policy_index = add_section(
                "catalog_policy", kKindStateField, 2, 1,
                {result.catalog_count, policy_loci_}
            );
            catalog_trait_index = add_section(
                "catalog_trait", kKindStateField, 4, 2,
                {result.catalog_count, trait_loci_}
            );
            append_u64(sections[catalog_offsets_index].payload, 0);
            std::uint64_t offset = 0;
            for (const auto& catalog : catalog_by_world) {
                offset = checked_add(offset, catalog.size(), "catalog world offset");
                append_u64(sections[catalog_offsets_index].payload, offset);
            }
        }

        const auto& resource = required_section("resource");
        const auto& position = required_section("position");
        const auto& energy = required_section("energy");
        const auto& health = required_section("health");
        const auto& age = required_section("age");
        const auto& birth = required_section("absolute_birth_tick");
        const auto& parent_a = required_section("parent_a_id");
        const auto& parent_b = required_section("parent_b_id");
        const auto& policy = required_section("policy_alleles");
        const auto& trait = required_section("trait_alleles");
        for (std::size_t selection = 0; selection < selected.size(); ++selection) {
            const auto world_index = selected[selection];
            const auto* world = state_data + checked_host_size(
                checked_multiply(world_index, state_row_bytes, "compact world offset"),
                "compact world offset"
            );
            const auto world_uid = world_uids_[checked_host_size(world_index, "world index")];
            append_u64(sections[world_uid_index].payload, world_uid);
            const auto& captured = result.captured_worlds[selection];
            append_digest(sections[canonical_sha_index].payload, captured.canonical_state_sha);
            const auto* header_data = world + header.source_offset_bytes;
            for (const auto header_word : {1U, 3U, 15U, 16U, 17U, 18U}) {
                append_u64(
                    sections[header_index].payload,
                    get_u64(header_data, static_cast<std::size_t>(header_word) * 8U)
                );
            }
            const auto* resource_data = world + resource.source_offset_bytes;
            append_digest(
                sections[resource_sha_index].payload,
                sha256(resource_data, checked_host_size(resource.payload_bytes_per_world, "resource bytes"))
            );
            std::uint64_t resource_total = 0;
            std::uint16_t resource_min = std::numeric_limits<std::uint16_t>::max();
            std::uint16_t resource_max = 0;
            std::uint64_t resource_nonzero = 0;
            for (std::uint64_t index = 0; index < resource_count_; ++index) {
                const auto value = get_u16(resource_data, checked_host_size(index * 2U, "resource offset"));
                resource_total = checked_add(resource_total, value, "resource total");
                resource_min = std::min(resource_min, value);
                resource_max = std::max(resource_max, value);
                resource_nonzero += value != 0;
            }
            append_u64(sections[resource_summary_index].payload, resource_total);
            append_u64(sections[resource_summary_index].payload, resource_min);
            append_u64(sections[resource_summary_index].payload, resource_max);
            append_u64(sections[resource_summary_index].payload, resource_nonzero);

            const auto* world_metrics = metrics_data
                + checked_host_size(
                    checked_multiply(
                        checked_multiply(world_index, capacity_, "metrics world offset"),
                        metric_width_, "metrics world offset"
                    ),
                    "metrics world offset"
                );
            for (const auto& active : active_by_world[selection]) {
                append_u32(sections[slots_index].payload, active.slot);
                append_u64(sections[agent_index].payload, active.agent_id);
                append_u16(
                    sections[position_index].payload,
                    get_u16(world + position.source_offset_bytes, active.slot * 2U)
                );
                append_u16(
                    sections[energy_index].payload,
                    static_cast<std::uint16_t>(get_i16(
                        world + energy.source_offset_bytes, active.slot * 2U
                    ))
                );
                append_u16(
                    sections[health_index].payload,
                    static_cast<std::uint16_t>(get_i16(
                        world + health.source_offset_bytes, active.slot * 2U
                    ))
                );
                append_u16(
                    sections[age_index].payload,
                    get_u16(world + age.source_offset_bytes, active.slot * 2U)
                );
                append_u64(
                    sections[birth_index].payload,
                    get_u64(world + birth.source_offset_bytes, active.slot * 8U)
                );
                append_u64(
                    sections[parent_a_index].payload,
                    get_u64(world + parent_a.source_offset_bytes, active.slot * 8U)
                );
                append_u64(
                    sections[parent_b_index].payload,
                    get_u64(world + parent_b.source_offset_bytes, active.slot * 8U)
                );
                append_digest(sections[genome_sha_index].payload, active.genome_sha);
                append_digest(sections[policy_sha_index].payload, active.policy_sha);
                const auto* metric_row = world_metrics
                    + static_cast<std::size_t>(active.slot) * metric_width_;
                for (std::uint64_t metric = 0; metric < metric_width_; ++metric) {
                    append_u64(
                        sections[metrics_index].payload,
                        static_cast<std::uint64_t>(metric_row[metric])
                    );
                }
            }
            if (include_catalog) {
                for (const auto& [digest, slot] : catalog_by_world[selection]) {
                    append_digest(sections[catalog_sha_index].payload, digest);
                    for (std::uint64_t locus = 0; locus < policy_loci_; ++locus) {
                        const auto offset = checked_add(
                            policy.source_offset_bytes,
                            checked_add(
                                checked_multiply(locus, capacity_, "catalog policy locus"),
                                slot,
                                "catalog policy slot"
                            ),
                            "catalog policy offset"
                        );
                        sections[catalog_policy_index].payload.push_back(
                            world[checked_host_size(offset, "catalog policy offset")]
                        );
                    }
                    for (std::uint64_t locus = 0; locus < trait_loci_; ++locus) {
                        const auto element = checked_add(
                            checked_multiply(locus, capacity_, "catalog trait locus"),
                            slot,
                            "catalog trait slot"
                        );
                        const auto offset = checked_add(
                            trait.source_offset_bytes,
                            checked_multiply(element, 2U, "catalog trait byte offset"),
                            "catalog trait offset"
                        );
                        sections[catalog_trait_index].payload.push_back(
                            world[checked_host_size(offset, "catalog trait offset")]
                        );
                        sections[catalog_trait_index].payload.push_back(
                            world[checked_host_size(offset + 1U, "catalog trait offset")]
                        );
                    }
                }
            }
        }
        result.projection_ns = elapsed_ns(projection_started);
        for (const auto& section : sections) {
            result.section_payload_bytes = checked_add(
                result.section_payload_bytes, section.payload.size(),
                "compact payload accounting"
            );
        }
        result.active_metrics_sha = sha256(
            sections[metrics_index].payload.data(),
            sections[metrics_index].payload.size()
        );
        result.dense_oracle_payload_bytes = checked_add(
            128U,
            checked_multiply(selected.size(), 8U, "dense selected indices"),
            "dense oracle payload"
        );
        for (const auto& section : sections_) {
            result.dense_oracle_payload_bytes = checked_add(
                result.dense_oracle_payload_bytes,
                checked_multiply(
                    selected.size(), section.payload_bytes_per_world,
                    "dense oracle payload"
                ),
                "dense oracle payload"
            );
        }
        const auto encoding_started = Clock::now();
        result.capsule = encode_capsule_bytes(
            sections, absolute_tick, selected.size(), result.row_count
        );
        result.capsule_encoding_ns = elapsed_ns(encoding_started);
        return result;
    }

    static StageWriteResult write_stage_file(
        const std::vector<std::uint8_t>& capsule,
        const std::string& stage_path
    ) {
        StageWriteResult result;
        const auto open_started = Clock::now();
        auto parent = open_canonical_no_symlink_parent(stage_path);
        int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#else
        TORCH_CHECK(false, "compact capsule stage requires O_NOFOLLOW support");
#endif
        const int raw_descriptor = ::openat(
            parent.descriptor.get(), parent.filename.c_str(), flags, 0600
        );
        const int open_error = errno;
        TORCH_CHECK(
            raw_descriptor >= 0,
            "compact capsule stage exclusive create failed: ",
            std::strerror(open_error)
        );
        UniqueFd descriptor(raw_descriptor);
        struct stat created_status {};
        const int created_stat = ::fstat(descriptor.get(), &created_status);
        const int created_stat_error = errno;
        if (created_stat != 0) {
            descriptor.reset();
            TORCH_CHECK(
                false,
                "compact capsule created stage fstat failed; pathname preserved "
                "because owned inode identity is unavailable: ",
                std::strerror(created_stat_error)
            );
        }
        result.file_device = stat_device(created_status);
        result.file_inode = stat_inode(created_status);
        result.parent_device = parent.device;
        result.parent_inode = parent.inode;
        result.open_ns = elapsed_ns(open_started);
        try {
            TORCH_CHECK(
                S_ISREG(created_status.st_mode),
                "compact capsule created stage is not a regular file"
            );
            const auto write_started = Clock::now();
            std::size_t offset = 0;
            constexpr std::size_t kWriteChunk = 1024U * 1024U;
            while (offset < capsule.size()) {
                const auto requested = std::min(kWriteChunk, capsule.size() - offset);
                const auto written = ::write(
                    descriptor.get(), capsule.data() + offset, requested
                );
                if (written < 0 && errno == EINTR) {
                    continue;
                }
                const int write_error = errno;
                TORCH_CHECK(
                    written > 0,
                    "compact capsule stage write failed: ",
                    std::strerror(write_error)
                );
                offset += static_cast<std::size_t>(written);
            }
            result.write_ns = elapsed_ns(write_started);
            const auto fsync_started = Clock::now();
            const int fsync_status = ::fsync(descriptor.get());
            const int fsync_error = errno;
            TORCH_CHECK(
                fsync_status == 0,
                "compact capsule stage fsync failed: ",
                std::strerror(fsync_error)
            );
            struct stat status {};
            const int stat_status = ::fstat(descriptor.get(), &status);
            const int stat_error = errno;
            TORCH_CHECK(
                stat_status == 0,
                "compact capsule stage fstat failed: ",
                std::strerror(stat_error)
            );
            TORCH_CHECK(
                same_file_identity(
                    status, result.file_device, result.file_inode
                )
                    && S_ISREG(status.st_mode) && status.st_size >= 0
                    && static_cast<std::uint64_t>(status.st_size) == capsule.size(),
                "compact capsule stage content identity or extent differs"
            );
            result.fsync_fstat_ns = elapsed_ns(fsync_started);
            const auto close_started = Clock::now();
            const int closing_descriptor = descriptor.release();
            const int close_status = ::close(closing_descriptor);
            const int close_error = errno;
            result.close_ns = elapsed_ns(close_started);
            TORCH_CHECK(
                close_status == 0,
                "compact capsule stage close failed: ",
                std::strerror(close_error)
            );
            verify_owned_stage_entry(
                parent, result.file_device, result.file_inode, capsule.size(),
                "compact capsule stage path after close"
            );
            auto resolved_parent = open_canonical_no_symlink_parent(stage_path);
            TORCH_CHECK(
                resolved_parent.device == result.parent_device
                    && resolved_parent.inode == result.parent_inode,
                "compact capsule stage absolute parent path identity differs "
                "after close; owned entry cleanup is restricted to its original dirfd"
            );
            verify_owned_stage_entry(
                resolved_parent, result.file_device, result.file_inode,
                capsule.size(),
                "compact capsule absolute stage path after close"
            );
            result.path_revalidated_after_close = true;
        } catch (const std::exception& error) {
            descriptor.reset();
            const auto cleanup = preserve_owned_stage_entry(
                parent, result.file_device, result.file_inode
            );
            throw std::runtime_error(
                std::string(error.what()) + "; owned-stage disposition=" + cleanup
            );
        } catch (...) {
            descriptor.reset();
            const auto cleanup = preserve_owned_stage_entry(
                parent, result.file_device, result.file_inode
            );
            throw std::runtime_error(
                "compact capsule stage failed with unknown native error; "
                "owned-stage disposition=" + cleanup
            );
        }
        return result;
    }

    std::vector<std::uint8_t> encode_capsule_bytes(
        const std::vector<MaterializedSection>& sections,
        const std::uint64_t absolute_tick,
        const std::uint64_t selected_world_count,
        const std::uint64_t row_count
    ) const {
        const auto section_count = static_cast<std::uint64_t>(sections.size());
        const auto directory_offset = static_cast<std::uint64_t>(kHeaderBytes);
        const auto directory_bytes = checked_multiply(section_count, kDescriptorBytes, "directory bytes");
        const auto names_offset = checked_add(directory_offset, directory_bytes, "names offset");
        std::uint64_t names_bytes = 0;
        for (const auto& section : sections) {
            names_bytes = checked_add(names_bytes, section.name.size(), "name table bytes");
        }
        const auto payload_offset = align_up(checked_add(names_offset, names_bytes, "payload offset"));
        std::vector<std::uint64_t> payload_offsets;
        payload_offsets.reserve(sections.size());
        auto cursor = payload_offset;
        for (const auto& section : sections) {
            cursor = align_up(cursor);
            payload_offsets.push_back(cursor);
            cursor = checked_add(cursor, section.payload.size(), "section extent");
        }
        const auto trailer_offset = align_up(cursor);
        const auto payload_bytes = trailer_offset - payload_offset;
        const auto capsule_bytes = checked_add(trailer_offset, kGlobalDigestBytes, "capsule bytes");
        TORCH_CHECK(capsule_bytes <= std::numeric_limits<std::size_t>::max(), "capsule exceeds host address space");
        std::vector<std::uint8_t> out(static_cast<std::size_t>(capsule_bytes), 0);
        std::copy(kMagic.begin(), kMagic.end(), out.begin());
        put_u16(out, 16, kVersion);
        put_u16(out, 18, kHeaderBytes);
        put_u32(out, 20, kEndianMarker);
        put_u32(out, 24, kDescriptorBytes);
        put_u32(out, 28, kAlignment);
        put_u64(out, 32, 0);
        put_u64(out, 40, absolute_tick);
        put_u64(out, 48, selected_world_count);
        put_u64(out, 56, row_count);
        put_u64(out, 64, section_count);
        put_u64(out, 72, directory_offset);
        put_u64(out, 80, directory_bytes);
        put_u64(out, 88, names_offset);
        put_u64(out, 96, names_bytes);
        put_u64(out, 104, payload_offset);
        put_u64(out, 112, payload_bytes);
        put_u64(out, 120, trailer_offset);
        put_u64(out, 128, capsule_bytes);
        std::copy(plan_sha_.begin(), plan_sha_.end(), out.begin() + 136);
        std::copy(schema_sha_.begin(), schema_sha_.end(), out.begin() + 168);
        std::copy(layout_sha_.begin(), layout_sha_.end(), out.begin() + 200);
        std::copy(control_sha_.begin(), control_sha_.end(), out.begin() + 232);
        put_u32(
            out,
            264,
            static_cast<std::uint32_t>(projection_v1::kImplementedBackend)
        );
        std::uint64_t canonical_plan_index_plus_one = 0;
        for (std::size_t index = 0; index < sections.size(); ++index) {
            if (sections[index].kind == kKindCanonicalPlan) {
                TORCH_CHECK(
                    canonical_plan_index_plus_one == 0
                        && sections[index].name == "canonical_plan_json",
                    "snapshot capsule canonical plan section differs"
                );
                canonical_plan_index_plus_one = static_cast<std::uint64_t>(index) + 1U;
            }
        }
        put_u64(out, 272, canonical_plan_index_plus_one);
        auto name_cursor = names_offset;
        for (std::size_t index = 0; index < sections.size(); ++index) {
            const auto& section = sections[index];
            TORCH_CHECK(
                !section.shape.empty() && section.shape.size() <= kMaxRank,
                "snapshot capsule materialized section rank differs"
            );
            const auto descriptor = directory_offset + index * kDescriptorBytes;
            put_u32(out, descriptor, section.kind);
            put_u16(out, descriptor + 4, section.dtype);
            put_u16(out, descriptor + 6, static_cast<std::uint16_t>(section.shape.size()));
            put_u64(out, descriptor + 16, section.element_width);
            std::uint64_t elements = 1;
            for (const auto dimension : section.shape) {
                elements = checked_multiply(elements, dimension, "section elements");
            }
            TORCH_CHECK(
                checked_multiply(
                    elements, section.element_width, "materialized section bytes"
                ) == section.payload.size(),
                "snapshot capsule materialized section payload differs: ",
                section.name
            );
            put_u64(out, descriptor + 24, elements);
            put_u64(out, descriptor + 32, name_cursor);
            put_u64(out, descriptor + 40, section.name.size());
            put_u64(out, descriptor + 48, payload_offsets[index]);
            put_u64(out, descriptor + 56, section.payload.size());
            put_u64(out, descriptor + 64, section.payload.size());
            for (std::size_t dimension = 0; dimension < section.shape.size(); ++dimension) {
                put_u64(out, descriptor + 72 + dimension * 8, section.shape[dimension]);
            }
            const auto section_sha = sha256(section.payload.data(), section.payload.size());
            std::copy(section_sha.begin(), section_sha.end(), out.begin() + descriptor + 136);
            std::copy(section.name.begin(), section.name.end(), out.begin() + name_cursor);
            name_cursor += section.name.size();
            std::copy(
                section.payload.begin(), section.payload.end(),
                out.begin() + static_cast<std::ptrdiff_t>(payload_offsets[index])
            );
        }
        const auto global_sha = sha256(out.data(), static_cast<std::size_t>(trailer_offset));
        std::copy(global_sha.begin(), global_sha.end(), out.begin() + trailer_offset);
        return out;
    }

    py::bytes encode_capsule(
        const std::vector<MaterializedSection>& sections,
        const std::uint64_t absolute_tick,
        const std::uint64_t selected_world_count,
        const std::uint64_t row_count
    ) const {
        const auto out = encode_capsule_bytes(
            sections, absolute_tick, selected_world_count, row_count
        );
        return py::bytes(
            reinterpret_cast<const char*>(out.data()),
            static_cast<py::ssize_t>(out.size())
        );
    }

    std::string encode_compatibility_jsonl(
        const std::string& capsule,
        const ParsedCapsule& parsed,
        CompatibilityConversionResult& result
    ) const {
        const auto* data = reinterpret_cast<const std::uint8_t*>(capsule.data());
        const auto section_data = [&parsed, data](const std::size_t index) {
            return data + checked_host_size(
                parsed.sections[index].payload_offset,
                "snapshot compatibility section offset"
            );
        };
        const auto* selected = section_data(1);
        const auto* world_offsets = section_data(4);
        const auto* world_uids = section_data(6);
        const auto* canonical_digests = section_data(7);
        const auto* resource_digests = section_data(8);
        const auto* headers = section_data(9);
        const auto* resources = section_data(10);
        const auto* agent_ids = section_data(11);
        const auto* positions = section_data(12);
        const auto* energies = section_data(13);
        const auto* healths = section_data(14);
        const auto* ages = section_data(15);
        const auto* birth_ticks = section_data(16);
        const auto* parent_as = section_data(17);
        const auto* parent_bs = section_data(18);
        const auto* genome_digests = section_data(19);
        const auto* policy_digests = section_data(20);
        const auto* metrics = section_data(21);

        result.active_agent_count = get_u64(
            world_offsets,
            checked_host_size(
                result.selected_world_count * 8U,
                "snapshot compatibility final world offset"
            )
        );
        result.genome_catalog_included = parsed.sections.size() == 26U;
        const std::uint8_t* catalog_offsets = nullptr;
        const std::uint8_t* catalog_digests = nullptr;
        const std::uint8_t* catalog_policy = nullptr;
        const std::uint8_t* catalog_trait = nullptr;
        if (result.genome_catalog_included) {
            catalog_offsets = section_data(22);
            catalog_digests = section_data(23);
            catalog_policy = section_data(24);
            catalog_trait = section_data(25);
            result.catalog_row_count = get_u64(
                catalog_offsets,
                checked_host_size(
                    result.selected_world_count * 8U,
                    "snapshot compatibility final catalog offset"
                )
            );
        }

        std::string output;
        const auto initial_reserve = std::min(
            maximum_canonical_jsonl_bytes_,
            checked_multiply(
                result.row_count, 1024U,
                "snapshot compatibility initial JSON reserve"
            )
        );
        output.reserve(checked_host_size(
            initial_reserve, "snapshot compatibility initial JSON reserve"
        ));
        const auto tick = static_cast<std::int64_t>(result.absolute_tick);

        for (std::uint64_t selection_index = 0;
             selection_index < result.selected_world_count;
             ++selection_index) {
            const auto world_index = get_u64(
                selected,
                checked_host_size(
                    selection_index * 8U,
                    "snapshot compatibility selected world offset"
                )
            );
            const auto world_uid = get_u64(
                world_uids,
                checked_host_size(
                    selection_index * 8U,
                    "snapshot compatibility world UID offset"
                )
            );
            const auto& cell_id_json = world_cell_id_json_[checked_host_size(
                world_index, "snapshot compatibility world index"
            )];
            const auto canonical_state_sha256 = bytes_to_hex(
                canonical_digests + checked_host_size(
                    selection_index * 32U,
                    "snapshot compatibility canonical digest offset"
                ),
                32U
            );
            const auto resource_state_sha256 = bytes_to_hex(
                resource_digests + checked_host_size(
                    selection_index * 32U,
                    "snapshot compatibility resource digest offset"
                ),
                32U
            );
            const auto header = [headers, selection_index](
                const std::uint64_t index
            ) {
                return get_u64(
                    headers,
                    checked_host_size(
                        (selection_index * 6U + index) * 8U,
                        "snapshot compatibility header offset"
                    )
                );
            };
            const auto resource = [resources, selection_index](
                const std::uint64_t index
            ) {
                return get_u64(
                    resources,
                    checked_host_size(
                        (selection_index * 4U + index) * 8U,
                        "snapshot compatibility resource summary offset"
                    )
                );
            };

            std::string world_identity;
            world_identity.reserve(180);
            world_identity.append("{\"canonical_state_sha256\":\"");
            world_identity.append(canonical_state_sha256);
            world_identity.append("\",\"tick\":");
            append_json_i64(world_identity, tick);
            world_identity.append(",\"world_uid\":");
            append_json_u64(world_identity, world_uid);
            world_identity.push_back('}');
            const auto world_state_id = sha256_prefixed(world_identity);

            output.append("{\"absolute_tick\":");
            append_json_i64(output, tick);
            output.append(",\"cell_id\":");
            output.append(cell_id_json);
            output.append(",\"field_set_version\":\"");
            output.append(kCompatibilityFieldSetVersion);
            output.append("\",\"qualified_field_values\":{\"absolute_tick\":");
            append_json_i64(output, tick);
            output.append(",\"active_count\":");
            append_json_u64(output, header(1));
            output.append(",\"birth_count\":");
            append_json_u64(output, header(2));
            output.append(",\"death_count\":");
            append_json_u64(output, header(3));
            output.append(",\"mutation_count\":");
            append_json_u64(output, header(4));
            output.append(",\"next_absolute_tick\":");
            append_json_u64(output, header(0));
            output.append(",\"population_count\":");
            append_json_u64(output, header(1));
            output.append(",\"recombination_count\":");
            append_json_u64(output, header(5));
            output.append(",\"resource_max\":");
            append_json_u64(output, resource(2));
            output.append(",\"resource_mean\":");
            append_python_float(
                output,
                static_cast<double>(resource(0))
                    / static_cast<double>(resource_count_)
            );
            output.append(",\"resource_min\":");
            append_json_u64(output, resource(1));
            output.append(",\"resource_nonzero_count\":");
            append_json_u64(output, resource(3));
            output.append(",\"resource_state_sha256\":{\"algorithm\":\"sha256\",\"hex\":\"");
            output.append(resource_state_sha256);
            output.append("\"},\"resource_total\":");
            append_json_u64(output, resource(0));
            output.append(",\"world_uid\":");
            append_json_u64(output, world_uid);
            output.append("},\"row_kind\":\"world_state\",\"snapshot_reason\":\"");
            output.append(kCompatibilitySnapshotReason);
            output.append("\",\"world_state_id\":\"");
            output.append(world_state_id);
            output.append("\",\"world_uid\":");
            append_json_u64(output, world_uid);
            output.append("}\n");

            const auto active_begin = get_u64(
                world_offsets,
                checked_host_size(
                    selection_index * 8U,
                    "snapshot compatibility active begin offset"
                )
            );
            const auto active_end = get_u64(
                world_offsets,
                checked_host_size(
                    (selection_index + 1U) * 8U,
                    "snapshot compatibility active end offset"
                )
            );
            for (auto row = active_begin; row < active_end; ++row) {
                const auto agent_id = get_u64(
                    agent_ids,
                    checked_host_size(
                        row * 8U,
                        "snapshot compatibility agent ID offset"
                    )
                );
                const auto position = get_u16(
                    positions,
                    checked_host_size(
                        row * 2U,
                        "snapshot compatibility position offset"
                    )
                );
                const auto energy = get_i16(
                    energies,
                    checked_host_size(
                        row * 2U,
                        "snapshot compatibility energy offset"
                    )
                );
                const auto health = get_i16(
                    healths,
                    checked_host_size(
                        row * 2U,
                        "snapshot compatibility health offset"
                    )
                );
                const auto age = get_u16(
                    ages,
                    checked_host_size(
                        row * 2U,
                        "snapshot compatibility age offset"
                    )
                );
                const auto birth_tick = get_u64(
                    birth_ticks,
                    checked_host_size(
                        row * 8U,
                        "snapshot compatibility birth tick offset"
                    )
                );
                const auto parent_a = get_u64(
                    parent_as,
                    checked_host_size(
                        row * 8U,
                        "snapshot compatibility parent A offset"
                    )
                );
                const auto parent_b = get_u64(
                    parent_bs,
                    checked_host_size(
                        row * 8U,
                        "snapshot compatibility parent B offset"
                    )
                );
                const auto genome_state_id = std::string("sha256:")
                    + bytes_to_hex(
                        genome_digests + checked_host_size(
                            row * 32U,
                            "snapshot compatibility genome digest offset"
                        ),
                        32U
                    );
                const auto policy_id = std::string("sha256:")
                    + bytes_to_hex(
                        policy_digests + checked_host_size(
                            row * 32U,
                            "snapshot compatibility policy digest offset"
                        ),
                        32U
                    );

                std::string lineage_identity;
                lineage_identity.reserve(160);
                lineage_identity.append("{\"agent_id\":");
                append_json_u64(lineage_identity, agent_id);
                lineage_identity.append(",\"parent_a\":");
                append_json_u64(lineage_identity, parent_a);
                lineage_identity.append(",\"parent_b\":");
                append_json_u64(lineage_identity, parent_b);
                lineage_identity.append(",\"world_uid\":");
                append_json_u64(lineage_identity, world_uid);
                lineage_identity.push_back('}');
                const auto lineage_id = sha256_prefixed(lineage_identity);

                std::string founder_identity;
                founder_identity.reserve(160);
                founder_identity.append("{\"root_parent_a\":");
                append_json_u64(
                    founder_identity, parent_a == 0 ? agent_id : parent_a
                );
                founder_identity.append(",\"root_parent_b\":");
                append_json_u64(founder_identity, parent_b);
                founder_identity.append(",\"world_uid\":");
                append_json_u64(founder_identity, world_uid);
                founder_identity.push_back('}');
                const auto founder_id = sha256_prefixed(founder_identity);

                std::string agent_identity;
                agent_identity.reserve(320);
                agent_identity.append("{\"age_ticks\":");
                append_json_u64(agent_identity, age);
                agent_identity.append(",\"agent_id\":");
                append_json_u64(agent_identity, agent_id);
                agent_identity.append(",\"alive\":true,\"energy\":");
                append_json_i64(agent_identity, energy);
                agent_identity.append(",\"genome_state_id\":\"");
                agent_identity.append(genome_state_id);
                agent_identity.append("\",\"health\":");
                append_json_i64(agent_identity, health);
                agent_identity.append(",\"position\":");
                append_json_u64(agent_identity, position);
                agent_identity.append(",\"tick\":");
                append_json_i64(agent_identity, tick);
                agent_identity.append(",\"world_uid\":");
                append_json_u64(agent_identity, world_uid);
                agent_identity.push_back('}');
                const auto agent_state_id = sha256_prefixed(agent_identity);

                output.append("{\"absolute_birth_tick\":");
                append_json_u64(output, birth_tick);
                output.append(",\"absolute_tick\":");
                append_json_i64(output, tick);
                output.append(",\"age_ticks\":");
                append_json_u64(output, age);
                output.append(",\"agent_id\":");
                append_json_u64(output, agent_id);
                output.append(",\"agent_state_id\":\"");
                output.append(agent_state_id);
                output.append("\",\"alive\":true,\"cell_id\":");
                output.append(cell_id_json);
                output.append(",\"energy\":");
                append_json_i64(output, energy);
                output.append(",\"founder_id\":\"");
                output.append(founder_id);
                output.append("\",\"genome_state_id\":\"");
                output.append(genome_state_id);
                output.append("\",\"health\":");
                append_json_i64(output, health);
                output.append(",\"hereditary_package_state_ids\":[],\"injury\":0,\"lifetime_metrics\":{");
                for (std::size_t metric = 0;
                     metric < sorted_metric_fields_.size();
                     ++metric) {
                    if (metric != 0) {
                        output.push_back(',');
                    }
                    output.push_back('"');
                    output.append(sorted_metric_fields_[metric].first);
                    output.append("\":");
                    const auto element = checked_add(
                        checked_multiply(
                            row, metric_width_,
                            "snapshot compatibility metric row offset"
                        ),
                        sorted_metric_fields_[metric].second,
                        "snapshot compatibility metric offset"
                    );
                    append_json_i64(
                        output,
                        get_i64(
                            metrics,
                            checked_host_size(
                                element * 8U,
                                "snapshot compatibility metric byte offset"
                            )
                        )
                    );
                }
                output.append("},\"lineage_depth\":");
                append_json_u64(
                    output, parent_a == 0 && parent_b == 0 ? 0U : 1U
                );
                output.append(",\"lineage_id\":\"");
                output.append(lineage_id);
                output.append("\",\"linkage_state_id\":\"");
                output.append(genome_state_id);
                output.append("\",\"parent_a_id\":");
                append_json_u64(output, parent_a);
                output.append(",\"parent_b_id\":");
                append_json_u64(output, parent_b);
                output.append(",\"policy_id\":\"");
                output.append(policy_id);
                output.append("\",\"position\":");
                append_json_u64(output, position);
                output.append(",\"row_kind\":\"active_agent_state\",\"world_uid\":");
                append_json_u64(output, world_uid);
                output.append("}\n");
            }

            if (result.genome_catalog_included) {
                const auto catalog_begin = get_u64(
                    catalog_offsets,
                    checked_host_size(
                        selection_index * 8U,
                        "snapshot compatibility catalog begin offset"
                    )
                );
                const auto catalog_end = get_u64(
                    catalog_offsets,
                    checked_host_size(
                        (selection_index + 1U) * 8U,
                        "snapshot compatibility catalog end offset"
                    )
                );
                for (auto row = catalog_begin; row < catalog_end; ++row) {
                    const auto genome_state_id = std::string("sha256:")
                        + bytes_to_hex(
                            catalog_digests + checked_host_size(
                                row * 32U,
                                "snapshot compatibility catalog digest offset"
                            ),
                            32U
                        );
                    output.append("{\"absolute_tick\":");
                    append_json_i64(output, tick);
                    output.append(",\"cell_id\":");
                    output.append(cell_id_json);
                    output.append(",\"genome_state_id\":\"");
                    output.append(genome_state_id);
                    output.append("\",\"policy\":[");
                    for (std::uint64_t locus = 0; locus < policy_loci_; ++locus) {
                        if (locus != 0) {
                            output.push_back(',');
                        }
                        const auto element = checked_add(
                            checked_multiply(
                                row, policy_loci_,
                                "snapshot compatibility catalog policy row"
                            ),
                            locus,
                            "snapshot compatibility catalog policy offset"
                        );
                        append_json_i64(
                            output,
                            static_cast<std::int8_t>(catalog_policy[
                                checked_host_size(
                                    element,
                                    "snapshot compatibility catalog policy byte"
                                )
                            ])
                        );
                    }
                    output.append("],\"row_kind\":\"genome_catalog\",\"trait\":[");
                    for (std::uint64_t locus = 0; locus < trait_loci_; ++locus) {
                        if (locus != 0) {
                            output.push_back(',');
                        }
                        const auto element = checked_add(
                            checked_multiply(
                                row, trait_loci_,
                                "snapshot compatibility catalog trait row"
                            ),
                            locus,
                            "snapshot compatibility catalog trait offset"
                        );
                        append_json_i64(
                            output,
                            get_i16(
                                catalog_trait,
                                checked_host_size(
                                    element * 2U,
                                    "snapshot compatibility catalog trait byte"
                                )
                            )
                        );
                    }
                    output.append("],\"world_uid\":");
                    append_json_u64(output, world_uid);
                    output.append("}\n");
                }
            }
        }
        TORCH_CHECK(
            result.row_count == checked_add(
                result.selected_world_count,
                checked_add(
                    result.active_agent_count, result.catalog_row_count,
                    "snapshot compatibility row count"
                ),
                "snapshot compatibility row count"
            ),
            "snapshot compatibility encoded row count differs"
        );
        return output;
    }

    void validate_expected(const std::string& capsule, const ParsedCapsule& parsed) const {
        TORCH_CHECK(
            parsed.plan_sha == plan_sha_
                && parsed.schema_sha == schema_sha_
                && parsed.layout_sha == layout_sha_
                && parsed.control_sha == control_sha_,
            "snapshot capsule identities differ from compiled plan"
        );
        TORCH_CHECK(parsed.sections.size() == sections_.size() + 2, "snapshot capsule section count differs from plan");
        const auto* data = reinterpret_cast<const std::uint8_t*>(capsule.data());
        const auto boundary = boundaries_.find(parsed.absolute_tick);
        TORCH_CHECK(boundary != boundaries_.end(), "snapshot capsule tick is absent from plan");
        const auto& selected = boundary->second;
        TORCH_CHECK(parsed.world_count == selected.size(), "snapshot capsule selected world count differs");
        const auto& selection = parsed.sections[1];
        TORCH_CHECK(selection.payload_length == selected.size() * 8, "snapshot selected-world payload differs");
        for (std::size_t index = 0; index < selected.size(); ++index) {
            TORCH_CHECK(
                get_u64(data + selection.payload_offset, index * 8) == selected[index],
                "snapshot selected-world order differs"
            );
        }
        for (std::size_t index = 0; index < sections_.size(); ++index) {
            const auto& expected = sections_[index];
            const auto& observed = parsed.sections[index + 2];
            TORCH_CHECK(
                observed.name == expected.name
                    && observed.kind == expected.kind
                    && observed.dtype == expected.dtype
                    && observed.element_width == expected.element_width
                    && observed.shape.size() == expected.element_shape.size() + 1
                    && observed.shape.front() == selected.size()
                    && std::equal(
                        expected.element_shape.begin(), expected.element_shape.end(),
                        observed.shape.begin() + 1
                    )
                    && observed.payload_length
                        == checked_multiply(
                            selected.size(), expected.payload_bytes_per_world,
                            "expected section payload"
                        ),
                "snapshot capsule section differs from compiled plan: ",
                expected.name
            );
        }
    }

    void validate_compact_expected(
        const std::string& capsule,
        const ParsedCapsule& parsed
    ) const {
        TORCH_CHECK(
            parsed.plan_sha == plan_sha_
                && parsed.schema_sha == schema_sha_
                && parsed.layout_sha == layout_sha_
                && parsed.control_sha == control_sha_,
            "compact snapshot capsule identities differ from compiled plan"
        );
        const auto boundary = boundaries_.find(parsed.absolute_tick);
        TORCH_CHECK(
            boundary != boundaries_.end(),
            "compact snapshot capsule tick is absent from plan"
        );
        const auto& selected = boundary->second;
        TORCH_CHECK(
            parsed.world_count == selected.size(),
            "compact snapshot selected world count differs"
        );
        const bool include_catalog =
            catalog_enabled_ && catalog_absolute_tick_ == parsed.absolute_tick;
        TORCH_CHECK(
            parsed.sections.size() == (include_catalog ? 26U : 22U),
            "compact snapshot capsule section count differs"
        );
        const auto* data = reinterpret_cast<const std::uint8_t*>(capsule.data());
        const auto& selection = parsed.sections[1];
        TORCH_CHECK(
            selection.payload_length == selected.size() * 8U,
            "compact snapshot selected-world payload differs"
        );
        for (std::size_t index = 0; index < selected.size(); ++index) {
            TORCH_CHECK(
                get_u64(data + selection.payload_offset, index * 8U) == selected[index],
                "compact snapshot selected-world order differs"
            );
        }
        const auto profile_sha = sha256(std::string{kCompactStageProfileIdentity});
        const auto& profile = parsed.sections[2];
        TORCH_CHECK(
            profile.name == "compact_profile_identity"
                && profile.kind == kKindIdentity
                && profile.dtype == 1
                && profile.shape == std::vector<std::uint64_t>{32}
                && std::equal(
                    profile_sha.begin(), profile_sha.end(),
                    data + profile.payload_offset
                ),
            "compact snapshot materialization profile differs"
        );

        const auto& canonical_plan = parsed.sections[3];
        TORCH_CHECK(
            parsed.canonical_plan_index_plus_one == 4U
                && canonical_plan.name == "canonical_plan_json"
                && canonical_plan.kind == kKindCanonicalPlan
                && canonical_plan.dtype == 1
                && canonical_plan.shape == std::vector<std::uint64_t>{
                    static_cast<std::uint64_t>(canonical_plan_.size())
                }
                && canonical_plan.payload_length == canonical_plan_.size()
                && std::equal(
                    canonical_plan_.begin(), canonical_plan_.end(),
                    reinterpret_cast<const char*>(data + canonical_plan.payload_offset)
                ),
            "compact snapshot canonical reconstruction plan differs"
        );

        const auto& offsets = parsed.sections[4];
        TORCH_CHECK(
            offsets.name == "world_offsets"
                && offsets.kind == kKindWorldOffsets
                && offsets.dtype == 7
                && offsets.shape == std::vector<std::uint64_t>{
                    static_cast<std::uint64_t>(selected.size()) + 1U
                },
            "compact snapshot world offsets differ"
        );
        const auto* offsets_data = data + offsets.payload_offset;
        TORCH_CHECK(get_u64(offsets_data, 0) == 0, "compact snapshot world offsets do not start at zero");
        std::vector<std::uint64_t> active_offsets;
        active_offsets.reserve(selected.size() + 1U);
        for (std::size_t index = 0; index <= selected.size(); ++index) {
            const auto offset = get_u64(offsets_data, index * 8U);
            TORCH_CHECK(
                index == 0 || offset >= active_offsets.back(),
                "compact snapshot world offsets are not monotonic"
            );
            active_offsets.push_back(offset);
        }
        const auto active_count = active_offsets.back();

        std::uint64_t catalog_count = 0;
        std::vector<std::uint64_t> catalog_offsets;
        if (include_catalog) {
            const auto& value = parsed.sections[22];
            TORCH_CHECK(
                value.name == "catalog_world_offsets"
                    && value.kind == kKindWorldOffsets
                    && value.dtype == 7
                    && value.shape == std::vector<std::uint64_t>{
                        static_cast<std::uint64_t>(selected.size()) + 1U
                    },
                "compact snapshot catalog offsets differ"
            );
            const auto* value_data = data + value.payload_offset;
            catalog_offsets.reserve(selected.size() + 1U);
            for (std::size_t index = 0; index <= selected.size(); ++index) {
                const auto offset = get_u64(value_data, index * 8U);
                TORCH_CHECK(
                    (index != 0 || offset == 0)
                        && (index == 0 || offset >= catalog_offsets.back()),
                    "compact snapshot catalog offsets are not monotonic"
                );
                catalog_offsets.push_back(offset);
            }
            catalog_count = catalog_offsets.back();
        }
        TORCH_CHECK(
            parsed.row_count == checked_add(
                selected.size(),
                checked_add(active_count, catalog_count, "compact expected rows"),
                "compact expected rows"
            ),
            "compact snapshot row count differs"
        );

        struct ExpectedSection final {
            const char* name;
            std::uint32_t kind;
            std::uint16_t dtype;
            std::uint64_t width;
            std::vector<std::uint64_t> shape;
        };
        const auto worlds = static_cast<std::uint64_t>(selected.size());
        std::vector<ExpectedSection> expected{
            {"plan_identity", kKindIdentity, 1, 1, {128}},
            {"selected_world_indices", kKindIdentity, 7, 8, {worlds}},
            {"compact_profile_identity", kKindIdentity, 1, 1, {32}},
            {"canonical_plan_json", kKindCanonicalPlan, 1, 1,
             {static_cast<std::uint64_t>(canonical_plan_.size())}},
            {"world_offsets", kKindWorldOffsets, 7, 8, {worlds + 1U}},
            {"active_slots", kKindActiveSlots, 5, 4, {active_count}},
            {"world_uid", kKindStateField, 7, 8, {worlds}},
            {"canonical_state_sha256", kKindOpaqueTyped, 1, 1, {worlds, 32}},
            {"resource_state_sha256", kKindOpaqueTyped, 1, 1, {worlds, 32}},
            {"world_header_counts", kKindStateField, 7, 8, {worlds, 6}},
            {"resource_summary", kKindStateField, 7, 8, {worlds, 4}},
            {"agent_id", kKindStateField, 7, 8, {active_count}},
            {"position", kKindStateField, 3, 2, {active_count}},
            {"energy", kKindStateField, 4, 2, {active_count}},
            {"health", kKindStateField, 4, 2, {active_count}},
            {"age_ticks", kKindStateField, 3, 2, {active_count}},
            {"absolute_birth_tick", kKindStateField, 7, 8, {active_count}},
            {"parent_a_id", kKindStateField, 7, 8, {active_count}},
            {"parent_b_id", kKindStateField, 7, 8, {active_count}},
            {"genome_state_sha256", kKindOpaqueTyped, 1, 1, {active_count, 32}},
            {"policy_sha256", kKindOpaqueTyped, 1, 1, {active_count, 32}},
            {"lifetime_metrics", kKindMetricsField, 8, 8, {active_count, metric_width_}},
        };
        if (include_catalog) {
            expected.push_back({
                "catalog_world_offsets", kKindWorldOffsets, 7, 8, {worlds + 1U}
            });
            expected.push_back({
                "catalog_genome_sha256", kKindOpaqueTyped, 1, 1, {catalog_count, 32}
            });
            expected.push_back({
                "catalog_policy", kKindStateField, 2, 1, {catalog_count, policy_loci_}
            });
            expected.push_back({
                "catalog_trait", kKindStateField, 4, 2, {catalog_count, trait_loci_}
            });
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto& wanted = expected[index];
            const auto& observed = parsed.sections[index];
            TORCH_CHECK(
                observed.name == wanted.name
                    && observed.kind == wanted.kind
                    && observed.dtype == wanted.dtype
                    && observed.element_width == wanted.width
                    && observed.shape == wanted.shape,
                "compact snapshot capsule section differs: ", wanted.name
            );
        }

        const auto& world_uid_section = parsed.sections[6];
        const auto& header_section = parsed.sections[9];
        for (std::size_t selection_index = 0; selection_index < selected.size(); ++selection_index) {
            const auto world_index = selected[selection_index];
            TORCH_CHECK(
                get_u64(
                    data + world_uid_section.payload_offset,
                    selection_index * 8U
                ) == world_uids_[checked_host_size(world_index, "world index")],
                "compact snapshot world identity column differs"
            );
            TORCH_CHECK(
                get_u64(
                    data + header_section.payload_offset,
                    (selection_index * 6U) * 8U
                ) == parsed.absolute_tick,
                "compact snapshot header frontier differs"
            );
        }
        const auto* slots = data + parsed.sections[5].payload_offset;
        const auto* agent_ids = data + parsed.sections[11].payload_offset;
        for (std::size_t world = 0; world < selected.size(); ++world) {
            std::set<std::uint32_t> seen_slots;
            std::uint64_t previous_agent = 0;
            bool first_agent = true;
            for (auto row = active_offsets[world]; row < active_offsets[world + 1U]; ++row) {
                const auto slot = get_u32(slots, checked_host_size(row * 4U, "active slot offset"));
                const auto agent_id = get_u64(agent_ids, checked_host_size(row * 8U, "active agent offset"));
                TORCH_CHECK(
                    slot < capacity_ && seen_slots.insert(slot).second,
                    "compact snapshot active slot differs"
                );
                TORCH_CHECK(
                    agent_id > 0 && (first_agent || agent_id > previous_agent),
                    "compact snapshot active-agent order differs"
                );
                previous_agent = agent_id;
                first_agent = false;
            }
        }
        if (include_catalog) {
            const auto* digests = data + parsed.sections[23].payload_offset;
            for (std::size_t world = 0; world < selected.size(); ++world) {
                const std::uint8_t* previous = nullptr;
                for (auto row = catalog_offsets[world]; row < catalog_offsets[world + 1U]; ++row) {
                    const auto* current = digests + checked_host_size(row * 32U, "catalog digest offset");
                    TORCH_CHECK(
                        previous == nullptr
                            || std::lexicographical_compare(
                                previous, previous + 32, current, current + 32
                            ),
                        "compact snapshot catalog order differs"
                    );
                    previous = current;
                }
            }
        }
    }

    std::string canonical_plan_;
    std::string snapshot_schema_id_;
    std::string layout_id_;
    std::string execution_publication_context_sha256_;
    std::uint64_t world_count_ = 0;
    std::uint64_t canonical_state_bytes_per_world_ = 0;
    std::uint64_t capacity_ = 0;
    std::uint64_t resource_count_ = 0;
    std::uint64_t policy_loci_ = 0;
    std::uint64_t trait_loci_ = 0;
    std::uint64_t metric_width_ = 0;
    std::uint64_t metric_name_bytes_ = 0;
    std::uint64_t maximum_compact_capsule_bytes_ = 0;
    std::uint64_t maximum_canonical_jsonl_bytes_ = 0;
    bool catalog_enabled_ = false;
    std::uint64_t catalog_absolute_tick_ = 0;
    InputSpec state_input_;
    InputSpec metrics_input_;
    std::string state_capture_preferred_mode_;
    std::vector<StateCaptureModeSpec> state_capture_modes_;
    std::vector<std::uint64_t> world_uids_;
    std::vector<std::string> world_cell_id_json_;
    std::vector<std::string> metric_fields_;
    std::vector<std::pair<std::string, std::size_t>> sorted_metric_fields_;
    std::map<std::uint64_t, std::vector<std::uint64_t>> boundaries_;
    std::vector<SectionSpec> sections_;
    std::array<std::uint8_t, 32> plan_sha_{};
    std::array<std::uint8_t, 32> schema_sha_{};
    std::array<std::uint8_t, 32> layout_sha_{};
    std::array<std::uint8_t, 32> control_sha_{};
};

std::shared_ptr<NativeSnapshotPlanV1> compile_snapshot_plan(
    const py::bytes& canonical_plan,
    const py::dict& normalized
) {
    return std::make_shared<NativeSnapshotPlanV1>(
        static_cast<std::string>(canonical_plan), normalized
    );
}

py::dict inspect_capsule_entry(const py::bytes& value) {
    const std::string capsule = value;
    return inspect_parsed(parse_capsule(capsule));
}

py::bytes extract_capsule_entry(const py::bytes& value, const std::string& name) {
    return extract_section(static_cast<std::string>(value), name);
}

}  // namespace eiel::snapshot::capsule_v1

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    using eiel::snapshot::capsule_v1::NativeSnapshotPlanV1;
    module.doc() = "EIEL deterministic native snapshot capsule G2 host core";
    module.def("abi_version", []() {
        return std::string("eiel_native_snapshot_capsule_abi_v1");
    });
    module.def("extension_version", []() {
        return std::string("eiel_native_snapshot_extension_v1");
    });
    module.def("zlib_compile_version", []() {
        return std::string(ZLIB_VERSION);
    });
    module.def("zlib_runtime_version", []() {
        return std::string(zlibVersion());
    });
    module.def(
        "capsule_format_manifest_v1",
        &eiel::snapshot::capsule_v1::capsule_format_manifest
    );
    module.def("supports_cuda_projection_v1", []() { return false; });
    py::class_<NativeSnapshotPlanV1, std::shared_ptr<NativeSnapshotPlanV1>>(
        module, "NativeSnapshotPlanV1", py::module_local()
    )
        .def("identities", &NativeSnapshotPlanV1::identities)
        .def("plan_manifest", &NativeSnapshotPlanV1::plan_manifest)
        .def("canonical_plan_bytes", &NativeSnapshotPlanV1::canonical_plan_bytes)
        .def(
            "pack_capsule",
            &NativeSnapshotPlanV1::pack_capsule,
            py::arg("state"),
            py::arg("metrics"),
            py::arg("absolute_tick")
        )
        .def(
            "write_owned_compact_capsule_stage_file",
            &NativeSnapshotPlanV1::write_owned_compact_capsule_stage_file,
            py::arg("state"),
            py::arg("metrics"),
            py::arg("absolute_tick"),
            py::arg("stage_path"),
            py::arg("encoding_policy")
        )
        .def(
            "convert_compact_capsule_stage_file_to_v1_gzip_stage_file",
            &NativeSnapshotPlanV1::convert_compact_capsule_stage_file_to_v1_gzip_stage_file,
            py::arg("capsule_stage_path"),
            py::arg("gzip_stage_path"),
            py::arg("encoding_policy")
        )
        .def(
            "write_owned_compact_capsule_to_v1_gzip_stage_file",
            &NativeSnapshotPlanV1::write_owned_compact_capsule_to_v1_gzip_stage_file,
            py::arg("state"),
            py::arg("metrics"),
            py::arg("absolute_tick"),
            py::arg("stage_path"),
            py::arg("encoding_policy")
        )
        .def("inspect_capsule", &NativeSnapshotPlanV1::inspect_capsule)
        .def("extract_section", &NativeSnapshotPlanV1::extract_section)
        .def(
            "inspect_compact_capsule",
            &NativeSnapshotPlanV1::inspect_compact_capsule
        )
        .def(
            "extract_compact_section",
            &NativeSnapshotPlanV1::extract_compact_section
        )
        .def_property_readonly(
            "projection_backend",
            [](const NativeSnapshotPlanV1&) { return "HOST_CONTIGUOUS_SYNC_V1"; }
        )
        .def_property_readonly(
            "supports_cuda_projection",
            [](const NativeSnapshotPlanV1&) { return false; }
        );
    module.def(
        "compile_snapshot_plan_v1",
        &eiel::snapshot::capsule_v1::compile_snapshot_plan,
        py::arg("canonical_plan_bytes"),
        py::arg("normalized_plan")
    );
    module.def(
        "inspect_snapshot_capsule_v1",
        &eiel::snapshot::capsule_v1::inspect_capsule_entry,
        py::arg("capsule")
    );
    module.def(
        "extract_snapshot_capsule_section_v1",
        &eiel::snapshot::capsule_v1::extract_capsule_entry,
        py::arg("capsule"),
        py::arg("name")
    );
}
