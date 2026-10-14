#include <torch/extension.h>

#include <openssl/sha.h>
#include <zlib.h>

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
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace py = pybind11;

namespace eiel::snapshot::encoder_v1 {
namespace {

constexpr const char* kAbi =
    "eiel_native_snapshot_v1_compatibility_encoder_abi_v1";
constexpr const char* kExtension =
    "eiel_native_snapshot_v1_compatibility_encoder_extension_v1";
constexpr const char* kReceiptSchema =
    "eiel_native_snapshot_v1_compatibility_encoding_receipt_v1";
constexpr const char* kStageReceiptSchema =
    "eiel_native_snapshot_v1_stage_file_receipt_v1";
constexpr const char* kExactStageEncodingPolicy =
    "EXACT_V1_SINGLE_MEMBER_SEQUENTIAL";
constexpr const char* kSnapshotSchema =
    "eiel_cuda_v2_compact_science_snapshot_artifact_v1";
constexpr const char* kFieldSetVersion =
    "cuda_full_evolution_v2_state_projection_v1";
constexpr const char* kSnapshotReason =
    "native_cuda_v2_science_capture_boundary";
constexpr const char* kStateCaptureTransportSchema =
    "eiel_native_snapshot_state_capture_transport_v1";
constexpr const char* kStateCaptureCanonicalPrefix =
    "canonical_checkpoint_prefix_v1";
constexpr const char* kStateCaptureFullResident =
    "full_resident_row_v1";

using Clock = std::chrono::steady_clock;

constexpr bool host_is_little_endian() {
    union {
        std::uint16_t word;
        std::uint8_t bytes[2];
    } value{0x0001};
    return value.bytes[0] == 1;
}

std::uint64_t elapsed_ns(const Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
            .count()
    );
}

std::string sha256_hex(const std::uint8_t* data, const std::size_t size) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(data, size, digest.data());
    constexpr char alphabet[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        result[2 * index] = alphabet[digest[index] >> 4U];
        result[2 * index + 1] = alphabet[digest[index] & 0x0FU];
    }
    return result;
}

std::string sha256_hex(const std::string_view value) {
    return sha256_hex(
        reinterpret_cast<const std::uint8_t*>(value.data()), value.size()
    );
}

std::string sha256_prefixed(const std::string_view value) {
    return std::string("sha256:") + sha256_hex(value);
}

void append_u64(std::string& output, const std::uint64_t value) {
    std::array<char, 32> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    TORCH_CHECK(converted.ec == std::errc(), "uint64 JSON formatting failed");
    output.append(buffer.data(), converted.ptr);
}

void append_i64(std::string& output, const std::int64_t value) {
    std::array<char, 32> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
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
    const std::string_view encoded(buffer.data(), converted.ptr - buffer.data());
    output.append(encoded);
    if (encoded.find('.') == std::string_view::npos
        && encoded.find('e') == std::string_view::npos
        && encoded.find('E') == std::string_view::npos) {
        output.append(".0");
    }
}

void append_hex_byte(std::string& output, const std::uint8_t value) {
    constexpr char alphabet[] = "0123456789abcdef";
    output.push_back(alphabet[value >> 4U]);
    output.push_back(alphabet[value & 0x0FU]);
}

bool lowercase_sha256(const std::string_view value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](const char child) {
        return (child >= '0' && child <= '9') || (child >= 'a' && child <= 'f');
    });
}

bool safe_metric_name(const std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](const char child) {
        return (child >= 'a' && child <= 'z')
            || (child >= '0' && child <= '9') || child == '_';
    });
}

std::uint16_t read_u16(const std::uint8_t* data, const std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset])
        | (static_cast<std::uint16_t>(data[offset + 1]) << 8U);
}

std::int16_t read_i16(const std::uint8_t* data, const std::size_t offset) {
    return static_cast<std::int16_t>(read_u16(data, offset));
}

std::uint32_t read_u32(const std::uint8_t* data, const std::size_t offset) {
    std::uint32_t result = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        result |= static_cast<std::uint32_t>(data[offset + index]) << (index * 8U);
    }
    return result;
}

std::uint64_t read_u64(const std::uint8_t* data, const std::size_t offset) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(data[offset + index]) << (index * 8U);
    }
    return result;
}

void put_u32(std::vector<std::uint8_t>& output, const std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
    }
}

std::vector<std::uint8_t> deterministic_gzip_level_9(
    const std::uint8_t* input,
    const std::size_t input_bytes
) {
    // This is the exact gzip envelope produced by Python GzipFile with
    // compresslevel=9 and mtime=0: DEFLATE, no optional fields, XFL=2, OS=255.
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
    TORCH_CHECK(initialized == Z_OK, "native snapshot zlib initialization failed");
    try {
        std::array<std::uint8_t, 128 * 1024> buffer{};
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
                TORCH_CHECK(status == Z_OK, "native snapshot zlib compression failed");
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
                "native snapshot zlib finalization failed"
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
    TORCH_CHECK(deflateEnd(&stream) == Z_OK, "native snapshot zlib teardown failed");

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
    put_u32(output, static_cast<std::uint32_t>(crc));
    put_u32(output, static_cast<std::uint32_t>(input_bytes & 0xffffffffU));
    return output;
}

struct FieldExtent final {
    std::size_t offset{};
    std::size_t bytes{};
    std::string dtype;
    std::vector<std::size_t> shape;
};

struct WorldPlan final {
    std::size_t world_index{};
    std::uint64_t world_uid{};
    std::string cell_id_json;
};

struct BoundaryPlan final {
    std::int64_t absolute_tick{};
    std::vector<std::size_t> world_indices;
    bool include_genome_catalog{};
};

struct StateCaptureMode final {
    std::string name;
    std::size_t row_bytes{};
    std::size_t total_bytes{};
};

struct AgentSlot final {
    std::uint64_t agent_id{};
    std::size_t slot{};
};

struct CapturedWorld final {
    std::size_t world_index{};
    std::uint64_t world_uid{};
    std::size_t active_agent_count{};
    std::string canonical_state_sha256;
};

struct EncodingResult final {
    std::int64_t absolute_tick{};
    bool include_genome_catalog{};
    std::size_t row_count{};
    std::size_t active_agent_count{};
    std::size_t state_input_row_bytes{};
    std::size_t state_input_bytes{};
    std::size_t canonical_state_bytes_selected{};
    std::size_t resident_state_row_bytes{};
    std::string state_capture_mode;
    std::string state_capture_preferred_mode;
    std::string canonical_jsonl;
    std::vector<std::uint8_t> gzip;
    std::string uncompressed_sha256;
    std::string compressed_sha256;
    std::string active_metrics_content_sha256;
    std::string active_metrics_authentication;
    std::vector<CapturedWorld> captured_worlds;
    std::vector<std::size_t> selected_world_indices;
    std::vector<std::uint64_t> selected_world_uids;
    std::uint64_t validation_ns{};
    std::uint64_t selection_encoding_ns{};
    std::uint64_t compression_ns{};
    std::uint64_t content_hashing_ns{};
    std::uint64_t active_metrics_content_hashing_ns{};
    std::uint64_t core_encode_compress_hash_ns{};
    bool state_version_guard_enabled{};
    bool metrics_version_guard_enabled{};
    std::uint32_t state_input_version{};
    std::uint32_t metrics_input_version{};
    std::uint32_t state_completion_version{};
    std::uint32_t metrics_completion_version{};
    std::string input_ownership_contract{
        "caller_immutable_for_full_synchronous_call_v1"
    };
    bool gil_held_for_full_synchronous_call{true};
    bool stage_file_complete{};
    std::string stage_path;
    std::string stage_encoding_policy;
    std::size_t stage_bytes_written{};
    std::uint64_t stage_open_ns{};
    std::uint64_t stage_write_ns{};
    std::uint64_t stage_fsync_ns{};
    std::uint64_t stage_close_ns{};
    std::uint64_t native_encode_write_fsync_close_ns{};
};

class NativeSnapshotEncoderPlanV1 final {
public:
    NativeSnapshotEncoderPlanV1(
        const py::bytes& canonical_plan_bytes,
        const py::dict& plan
    ) {
        const auto canonical = canonical_plan_bytes.cast<std::string>();
        TORCH_CHECK(!canonical.empty(), "native snapshot encoder plan is empty");
        const auto canonical_mapping = py::module_::import("json")
            .attr("dumps")(
                plan,
                py::arg("sort_keys") = true,
                py::arg("separators") = py::make_tuple(",", ":"),
                py::arg("ensure_ascii") = true,
                py::arg("allow_nan") = false
            )
            .cast<std::string>();
        TORCH_CHECK(
            canonical == canonical_mapping,
            "native snapshot canonical plan bytes differ from the supplied mapping"
        );
        TORCH_CHECK(
            host_is_little_endian(),
            "native snapshot encoder requires a little-endian host"
        );
        plan_sha256_ = sha256_hex(canonical);
        TORCH_CHECK(
            py::cast<std::string>(plan["schema_version"])
                == "eiel_native_snapshot_plan_v1",
            "native snapshot encoder plan schema differs"
        );
        TORCH_CHECK(
            py::cast<std::string>(plan["snapshot_schema_id"]) == kSnapshotSchema,
            "native snapshot compatibility schema differs"
        );
        native_world_control_set_sha256_ =
            py::cast<std::string>(plan["native_world_control_set_sha256"]);
        TORCH_CHECK(
            lowercase_sha256(native_world_control_set_sha256_),
            "native world-control set identity is invalid"
        );
        execution_publication_context_sha256_ =
            py::cast<std::string>(plan["execution_publication_context_sha256"]);
        TORCH_CHECK(
            lowercase_sha256(execution_publication_context_sha256_),
            "execution/publication context identity is invalid"
        );
        world_count_ = py::cast<std::size_t>(plan["world_count"]);
        TORCH_CHECK(world_count_ > 0, "native snapshot encoder world count is zero");

        const auto layout = py::cast<py::dict>(plan["layout"]);
        TORCH_CHECK(
            py::cast<std::string>(layout["byte_order"]) == "little",
            "native snapshot encoder requires little-endian state"
        );
        state_bytes_per_world_ =
            py::cast<std::size_t>(layout["state_bytes_per_world"]);
        canonical_state_bytes_per_world_ =
            py::cast<std::size_t>(layout["canonical_state_bytes_per_world"]);
        capacity_ = py::cast<std::size_t>(layout["agent_capacity"]);
        alive_word_bits_ = py::cast<std::size_t>(layout["alive_word_bits"]);
        resource_count_ = py::cast<std::size_t>(layout["resource_cell_count"]);
        TORCH_CHECK(
            state_bytes_per_world_ > 0
                && canonical_state_bytes_per_world_ > 0
                && canonical_state_bytes_per_world_ < state_bytes_per_world_
                && capacity_ > 0 && alive_word_bits_ == 32
                && resource_count_ > 0,
            "native snapshot encoder layout bounds differ"
        );

        const auto capture_transport =
            py::cast<py::dict>(plan["state_capture_transport"]);
        for (const auto* key : {
                 "schema_version",
                 "resident_source_contract",
                 "prefix_semantics",
                 "preferred_mode",
                 "accepted_modes",
             }) {
            TORCH_CHECK(
                capture_transport.contains(key),
                "native snapshot state capture transport field is absent: ", key
            );
        }
        TORCH_CHECK(
            capture_transport.size() == 5
                && py::cast<std::string>(capture_transport["schema_version"])
                    == kStateCaptureTransportSchema
                && py::cast<std::string>(
                    capture_transport["resident_source_contract"]
                ) == "state_input"
                && py::cast<std::string>(capture_transport["prefix_semantics"])
                    == "leading_bytes_of_each_resident_world_row_v1",
            "native snapshot state capture transport contract differs"
        );
        state_capture_preferred_mode_ = py::cast<std::string>(
            capture_transport["preferred_mode"]
        );
        TORCH_CHECK(
            state_capture_preferred_mode_ == kStateCaptureCanonicalPrefix,
            "native snapshot preferred state capture mode differs"
        );
        const auto raw_capture_modes =
            py::cast<py::list>(capture_transport["accepted_modes"]);
        TORCH_CHECK(
            raw_capture_modes.size() == 2,
            "native snapshot state capture mode count differs"
        );
        const std::array<std::pair<const char*, std::size_t>, 2>
            expected_capture_modes{{
                {kStateCaptureCanonicalPrefix,
                 canonical_state_bytes_per_world_},
                {kStateCaptureFullResident, state_bytes_per_world_},
            }};
        for (std::size_t index = 0; index < expected_capture_modes.size(); ++index) {
            const auto mode = py::cast<py::dict>(raw_capture_modes[index]);
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
                    "native snapshot state capture mode field is absent: ", key
                );
            }
            const auto& [expected_name, expected_row_bytes] =
                expected_capture_modes[index];
            const auto shape = py::cast<py::list>(mode["shape"]);
            TORCH_CHECK(
                mode.size() == 6
                    && py::cast<std::string>(mode["mode"]) == expected_name
                    && py::cast<std::string>(mode["dtype"]) == "u8"
                    && py::cast<std::string>(mode["contiguity"]) == "C"
                    && shape.size() == 2
                    && py::cast<std::size_t>(shape[0]) == world_count_
                    && py::cast<std::size_t>(shape[1]) == expected_row_bytes
                    && py::cast<std::size_t>(mode["world_stride_bytes"])
                        == expected_row_bytes,
                "native snapshot state capture mode geometry differs: ",
                expected_name
            );
            TORCH_CHECK(
                expected_row_bytes == 0
                    || world_count_
                        <= std::numeric_limits<std::size_t>::max()
                            / expected_row_bytes,
                "native snapshot state capture byte extent overflows"
            );
            const auto expected_total = world_count_ * expected_row_bytes;
            TORCH_CHECK(
                py::cast<std::size_t>(mode["total_bytes"]) == expected_total,
                "native snapshot state capture total bytes differ: ",
                expected_name
            );
            state_capture_modes_.push_back({
                expected_name,
                expected_row_bytes,
                expected_total,
            });
        }
        TORCH_CHECK(
            state_capture_modes_[0].row_bytes
                != state_capture_modes_[1].row_bytes,
            "native snapshot state capture modes are ambiguous"
        );
        const auto raw_fields = py::cast<py::list>(layout["fields"]);
        for (const auto& item : raw_fields) {
            const auto field = py::cast<py::dict>(item);
            FieldExtent extent;
            const auto name = py::cast<std::string>(field["name"]);
            extent.offset = py::cast<std::size_t>(field["offset_bytes"]);
            extent.bytes = py::cast<std::size_t>(field["bytes"]);
            extent.dtype = py::cast<std::string>(field["dtype"]);
            for (const auto& dimension : py::cast<py::list>(field["shape"])) {
                extent.shape.push_back(py::cast<std::size_t>(dimension));
            }
            TORCH_CHECK(
                extent.offset <= state_bytes_per_world_
                    && extent.bytes <= state_bytes_per_world_ - extent.offset,
                "native snapshot encoder field exceeds one world"
            );
            fields_.emplace(name, std::move(extent));
        }
        validate_required_fields();

        const auto raw_metric_fields = py::cast<py::list>(plan["metric_fields"]);
        for (std::size_t index = 0; index < raw_metric_fields.size(); ++index) {
            const auto name = py::cast<std::string>(raw_metric_fields[index]);
            TORCH_CHECK(safe_metric_name(name), "native snapshot metric name is unsafe");
            metric_fields_.push_back(name);
            sorted_metric_fields_.emplace_back(name, index);
        }
        TORCH_CHECK(!metric_fields_.empty(), "native snapshot metric registry is empty");
        std::sort(sorted_metric_fields_.begin(), sorted_metric_fields_.end());
        for (std::size_t index = 1; index < sorted_metric_fields_.size(); ++index) {
            TORCH_CHECK(
                sorted_metric_fields_[index - 1].first
                    != sorted_metric_fields_[index].first,
                "native snapshot metric registry contains a duplicate"
            );
        }

        const auto raw_worlds = py::cast<py::list>(plan["worlds"]);
        TORCH_CHECK(
            raw_worlds.size() == world_count_,
            "native snapshot encoder world plan count differs"
        );
        std::unordered_set<std::uint64_t> world_uids;
        for (std::size_t index = 0; index < raw_worlds.size(); ++index) {
            const auto raw_world = py::cast<py::dict>(raw_worlds[index]);
            WorldPlan world;
            world.world_index = py::cast<std::size_t>(raw_world["world_index"]);
            world.world_uid = py::cast<std::uint64_t>(raw_world["world_uid"]);
            world.cell_id_json = py::cast<std::string>(raw_world["cell_id_json"]);
            TORCH_CHECK(
                world.world_index == index && world.world_uid > 0
                    && world_uids.insert(world.world_uid).second,
                "native snapshot encoder world ordering differs"
            );
            TORCH_CHECK(
                world.cell_id_json.size() >= 2
                    && world.cell_id_json.front() == '"'
                    && world.cell_id_json.back() == '"',
                "native snapshot encoder cell fragment differs"
            );
            worlds_.push_back(std::move(world));
        }

        const auto raw_boundaries = py::cast<py::list>(plan["snapshot_boundaries"]);
        std::int64_t previous_tick = -1;
        for (const auto& item : raw_boundaries) {
            const auto raw_boundary = py::cast<py::dict>(item);
            BoundaryPlan boundary;
            boundary.absolute_tick =
                py::cast<std::int64_t>(raw_boundary["absolute_tick"]);
            boundary.include_genome_catalog =
                py::cast<bool>(raw_boundary["include_genome_catalog"]);
            TORCH_CHECK(
                boundary.absolute_tick >= 0 && boundary.absolute_tick > previous_tick,
                "native snapshot encoder boundary order differs"
            );
            previous_tick = boundary.absolute_tick;
            std::size_t previous_world = 0;
            bool first = true;
            for (const auto& raw_index : py::cast<py::list>(raw_boundary["world_indices"])) {
                const auto index = py::cast<std::size_t>(raw_index);
                TORCH_CHECK(
                    index < world_count_ && (first || index > previous_world),
                    "native snapshot encoder selected world order differs"
                );
                first = false;
                previous_world = index;
                boundary.world_indices.push_back(index);
            }
            TORCH_CHECK(
                !boundary.world_indices.empty(),
                "native snapshot encoder boundary selects no worlds"
            );
            boundaries_.push_back(std::move(boundary));
        }
        TORCH_CHECK(!boundaries_.empty(), "native snapshot encoder has no boundaries");
    }

    const std::string& plan_sha256() const noexcept { return plan_sha256_; }
    const std::string& native_world_control_set_sha256() const noexcept {
        return native_world_control_set_sha256_;
    }
    const std::string& execution_publication_context_sha256() const noexcept {
        return execution_publication_context_sha256_;
    }
    const std::vector<std::string>& metric_fields() const noexcept {
        return metric_fields_;
    }

    EncodingResult encode_projection(
        const torch::Tensor& state,
        const py::dict& projection,
        const py::object& raw_absolute_tick
    ) const {
        TORCH_CHECK(
            PyLong_CheckExact(raw_absolute_tick.ptr()),
            "snapshot absolute tick must be an exact integer"
        );
        const auto absolute_tick = py::cast<std::int64_t>(raw_absolute_tick);
        for (const auto* key : {
                 "schema_version",
                 "next_absolute_tick",
                 "world_uids",
                 "metric_fields",
                 "shape",
                 "content_sha256",
                 "tensor",
             }) {
            TORCH_CHECK(
                projection.contains(key),
                "active metrics projection field is absent: ", key
            );
        }
        TORCH_CHECK(
            py::cast<std::string>(projection["schema_version"])
                == "eiel_full_evolution_cuda_active_science_metrics_v1",
            "active metrics projection schema differs"
        );
        const auto raw_frontier = projection["next_absolute_tick"];
        TORCH_CHECK(
            PyLong_CheckExact(raw_frontier.ptr())
                && py::cast<std::int64_t>(raw_frontier) == absolute_tick,
            "active metrics projection frontier differs from snapshot tick"
        );
        const auto raw_world_uids = py::cast<py::list>(projection["world_uids"]);
        TORCH_CHECK(
            raw_world_uids.size() == worlds_.size(),
            "active metrics projection world count differs"
        );
        for (std::size_t index = 0; index < worlds_.size(); ++index) {
            const auto raw_uid = raw_world_uids[index];
            TORCH_CHECK(
                PyLong_CheckExact(raw_uid.ptr())
                    && py::cast<std::uint64_t>(raw_uid) == worlds_[index].world_uid,
                "active metrics projection world order differs"
            );
        }
        const auto raw_metric_fields =
            py::cast<py::list>(projection["metric_fields"]);
        TORCH_CHECK(
            raw_metric_fields.size() == metric_fields_.size(),
            "active metrics projection registry size differs"
        );
        for (std::size_t index = 0; index < metric_fields_.size(); ++index) {
            TORCH_CHECK(
                py::cast<std::string>(raw_metric_fields[index])
                    == metric_fields_[index],
                "active metrics projection registry order differs"
            );
        }
        const auto raw_shape = py::cast<py::list>(projection["shape"]);
        const std::array<std::size_t, 3> expected_shape{
            world_count_, capacity_, metric_fields_.size(),
        };
        TORCH_CHECK(
            raw_shape.size() == expected_shape.size(),
            "active metrics projection declared shape differs"
        );
        for (std::size_t index = 0; index < expected_shape.size(); ++index) {
            const auto raw_dimension = raw_shape[index];
            TORCH_CHECK(
                PyLong_CheckExact(raw_dimension.ptr())
                    && py::cast<std::size_t>(raw_dimension) == expected_shape[index],
                "active metrics projection declared shape differs"
            );
        }
        const auto content_sha256 =
            py::cast<std::string>(projection["content_sha256"]);
        TORCH_CHECK(
            lowercase_sha256(content_sha256),
            "active metrics projection content digest is invalid"
        );
        const auto metrics = py::cast<torch::Tensor>(projection["tensor"]);
        auto result = encode_core(
            state,
            metrics,
            absolute_tick,
            std::optional<std::string>{content_sha256}
        );
        result.active_metrics_authentication =
            "validated_external_projection_descriptor_v1";
        return result;
    }

    EncodingResult encode_projection_to_stage_file(
        const torch::Tensor& state,
        const py::dict& projection,
        const py::object& raw_absolute_tick,
        const std::string& stage_path,
        const std::string& encoding_policy,
        const std::optional<std::size_t> fault_after_bytes
    ) const {
        validate_stage_request(stage_path, encoding_policy);
        const auto stage_total_started = Clock::now();
        auto result = encode_projection(state, projection, raw_absolute_tick);
        return write_stage_file(
            std::move(result),
            stage_path,
            encoding_policy,
            fault_after_bytes,
            stage_total_started
        );
    }

    EncodingResult encode_owned_tensors_to_stage_file(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        const std::int64_t absolute_tick,
        const std::string& stage_path,
        const std::string& encoding_policy,
        const std::optional<std::size_t> fault_after_bytes
    ) const {
        validate_stage_request(stage_path, encoding_policy);
        const auto stage_total_started = Clock::now();
        auto result = encode_core(
            state, metrics, absolute_tick, std::nullopt
        );
        result.active_metrics_authentication =
            "computed_native_from_plan_bound_resident_tensor_v1";
        result.input_ownership_contract =
            "native_service_exclusive_tensor_lease_until_completion_v1";
        result.gil_held_for_full_synchronous_call = false;
        result = write_stage_file(
            std::move(result),
            stage_path,
            encoding_policy,
            fault_after_bytes,
            stage_total_started
        );
        verify_input_versions_unchanged(state, metrics, result);
        return result;
    }

    EncodingResult encode_core(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        const std::int64_t absolute_tick,
        const std::optional<std::string>& expected_metrics_sha256
    ) const {
        const auto total_started = Clock::now();
        const auto validation_started = Clock::now();
        TORCH_CHECK(absolute_tick >= 0, "snapshot absolute tick is invalid");
        TORCH_CHECK(
            state.device().is_cpu() && state.scalar_type() == torch::kUInt8
                && state.is_contiguous() && state.dim() == 2
                && static_cast<std::size_t>(state.size(0)) == world_count_,
            "snapshot state must be one plan-declared CPU-contiguous uint8 "
            "capture transport"
        );
        const auto state_input_row_bytes =
            static_cast<std::size_t>(state.size(1));
        const auto& state_capture_mode =
            capture_mode_for_row_bytes(state_input_row_bytes);
        TORCH_CHECK(
            metrics.device().is_cpu() && metrics.scalar_type() == torch::kInt64
                && metrics.is_contiguous() && metrics.dim() == 3
                && static_cast<std::size_t>(metrics.size(0)) == world_count_
                && static_cast<std::size_t>(metrics.size(1)) == capacity_
                && static_cast<std::size_t>(metrics.size(2)) == metric_fields_.size(),
            "snapshot metrics must be one exact CPU-contiguous int64 projection"
        );
        const auto& state_version_counter =
            state.unsafeGetTensorImpl()->version_counter();
        const auto& metrics_version_counter =
            metrics.unsafeGetTensorImpl()->version_counter();
        const bool state_version_enabled = state_version_counter.enabled();
        const bool metrics_version_enabled = metrics_version_counter.enabled();
        const auto state_version = state_version_enabled
            ? state_version_counter.current_version()
            : 0;
        const auto metrics_version = metrics_version_enabled
            ? metrics_version_counter.current_version()
            : 0;
        const auto boundary = std::lower_bound(
            boundaries_.begin(), boundaries_.end(), absolute_tick,
            [](const BoundaryPlan& child, const std::int64_t tick) {
                return child.absolute_tick < tick;
            }
        );
        TORCH_CHECK(
            boundary != boundaries_.end() && boundary->absolute_tick == absolute_tick,
            "snapshot absolute tick is not declared by the bound plan"
        );
        EncodingResult result;
        result.absolute_tick = absolute_tick;
        result.include_genome_catalog = boundary->include_genome_catalog;
        result.selected_world_indices = boundary->world_indices;
        result.state_input_row_bytes = state_input_row_bytes;
        result.state_input_bytes = static_cast<std::size_t>(state.numel());
        TORCH_CHECK(
            result.state_input_bytes == state_capture_mode.total_bytes,
            "snapshot state capture tensor bytes differ from the bound mode"
        );
        TORCH_CHECK(
            canonical_state_bytes_per_world_ == 0
                || boundary->world_indices.size()
                    <= std::numeric_limits<std::size_t>::max()
                        / canonical_state_bytes_per_world_,
            "snapshot selected canonical state byte extent overflows"
        );
        result.canonical_state_bytes_selected =
            boundary->world_indices.size() * canonical_state_bytes_per_world_;
        result.resident_state_row_bytes = state_bytes_per_world_;
        result.state_capture_mode = state_capture_mode.name;
        result.state_capture_preferred_mode = state_capture_preferred_mode_;
        result.state_version_guard_enabled = state_version_enabled;
        result.metrics_version_guard_enabled = metrics_version_enabled;
        result.state_input_version = state_version;
        result.metrics_input_version = metrics_version;
        result.validation_ns = elapsed_ns(validation_started);

        const auto metrics_hashing_started = Clock::now();
        result.active_metrics_content_sha256 = sha256_hex(
            reinterpret_cast<const std::uint8_t*>(metrics.data_ptr<std::int64_t>()),
            static_cast<std::size_t>(metrics.numel()) * sizeof(std::int64_t)
        );
        result.active_metrics_content_hashing_ns =
            elapsed_ns(metrics_hashing_started);
        TORCH_CHECK(
            !expected_metrics_sha256.has_value()
                || result.active_metrics_content_sha256
                    == *expected_metrics_sha256,
            "active metrics projection content digest differs"
        );

        const auto* state_data = state.data_ptr<std::uint8_t>();
        const auto* metric_data = metrics.data_ptr<std::int64_t>();
        const auto state_row_bytes = result.state_input_row_bytes;

        const auto encoding_started = Clock::now();
        std::size_t estimated_active = 0;
        std::vector<std::vector<AgentSlot>> active_by_world;
        active_by_world.reserve(boundary->world_indices.size());
        for (const auto world_index : boundary->world_indices) {
            const auto* world = state_data + world_index * state_row_bytes;
            const auto& header = field("scalar_header_counters");
            TORCH_CHECK(
                read_u64(world, header.offset) == worlds_[world_index].world_uid,
                "selected native state header world identity differs from plan"
            );
            TORCH_CHECK(
                read_u64(world, header.offset + 8) ==
                    static_cast<std::uint64_t>(absolute_tick),
                "selected native state header frontier differs from snapshot tick"
            );
            std::unordered_set<std::uint64_t> seen;
            seen.reserve(capacity_);
            std::vector<AgentSlot> active;
            active.reserve(capacity_);
            for (std::size_t slot = 0; slot < capacity_; ++slot) {
                const auto agent_id = read_u64(
                    world, field("agent_id").offset + slot * 8
                );
                if (agent_id == 0) {
                    continue;
                }
                TORCH_CHECK(
                    seen.insert(agent_id).second,
                    "native state contains duplicate agent identity ", agent_id
                );
                const auto alive_word = read_u32(
                    world,
                    field("alive_words").offset + (slot / alive_word_bits_) * 4
                );
                if ((alive_word & (1U << (slot % alive_word_bits_))) != 0) {
                    active.push_back({agent_id, slot});
                }
            }
            std::sort(active.begin(), active.end(), [](const auto& left, const auto& right) {
                return left.agent_id < right.agent_id;
            });
            estimated_active += active.size();
            active_by_world.push_back(std::move(active));
        }
        result.canonical_jsonl.reserve(
            boundary->world_indices.size() * 1024 + estimated_active * 1024
        );

        for (std::size_t selection = 0; selection < boundary->world_indices.size(); ++selection) {
            const auto world_index = boundary->world_indices[selection];
            const auto& world_plan = worlds_[world_index];
            const auto* world = state_data + world_index * state_row_bytes;
            const auto* world_metrics = metric_data
                + world_index * capacity_ * metric_fields_.size();
            const auto& active = active_by_world[selection];
            const auto canonical_state_sha256 = sha256_hex(
                world, canonical_state_bytes_per_world_
            );
            append_world_row(
                result.canonical_jsonl,
                world,
                world_plan,
                absolute_tick,
                canonical_state_sha256
            );
            ++result.row_count;
            result.active_agent_count += active.size();
            result.selected_world_uids.push_back(world_plan.world_uid);
            result.captured_worlds.push_back({
                world_index,
                world_plan.world_uid,
                active.size(),
                canonical_state_sha256,
            });

            std::map<std::string, std::size_t> catalog_slots;
            for (const auto agent : active) {
                const auto genome_state_id = append_agent_row(
                    result.canonical_jsonl,
                    world,
                    world_metrics,
                    world_plan,
                    absolute_tick,
                    agent
                );
                ++result.row_count;
                if (boundary->include_genome_catalog) {
                    catalog_slots.emplace(genome_state_id, agent.slot);
                }
            }
            if (boundary->include_genome_catalog) {
                for (const auto& [genome_state_id, slot] : catalog_slots) {
                    append_catalog_row(
                        result.canonical_jsonl,
                        world,
                        world_plan,
                        absolute_tick,
                        genome_state_id,
                        slot
                    );
                    ++result.row_count;
                }
            }
        }
        result.selection_encoding_ns = elapsed_ns(encoding_started);

        const auto compression_started = Clock::now();
        result.gzip = deterministic_gzip_level_9(
            reinterpret_cast<const std::uint8_t*>(result.canonical_jsonl.data()),
            result.canonical_jsonl.size()
        );
        result.compression_ns = elapsed_ns(compression_started);

        const auto hashing_started = Clock::now();
        result.uncompressed_sha256 = sha256_hex(result.canonical_jsonl);
        result.compressed_sha256 = sha256_hex(result.gzip.data(), result.gzip.size());
        result.content_hashing_ns = elapsed_ns(hashing_started);
        verify_input_versions_unchanged(state, metrics, result);
        result.core_encode_compress_hash_ns = elapsed_ns(total_started);
        return result;
    }

    void verify_owned_tensor_versions_at_python_boundary(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        EncodingResult& result
    ) const {
        verify_input_versions_unchanged(state, metrics, result);
    }

private:
    const StateCaptureMode& capture_mode_for_row_bytes(
        const std::size_t row_bytes
    ) const {
        const auto found = std::find_if(
            state_capture_modes_.begin(),
            state_capture_modes_.end(),
            [row_bytes](const StateCaptureMode& mode) {
                return mode.row_bytes == row_bytes;
            }
        );
        TORCH_CHECK(
            found != state_capture_modes_.end(),
            "snapshot state must be one plan-declared CPU-contiguous uint8 "
            "capture transport"
        );
        return *found;
    }

    static void validate_stage_request(
        const std::string& stage_path,
        const std::string& encoding_policy
    ) {
        TORCH_CHECK(
            encoding_policy == kExactStageEncodingPolicy,
            "snapshot stage encoding policy is unsupported"
        );
        TORCH_CHECK(
            !stage_path.empty()
                && stage_path.find('\0') == std::string::npos
                && std::filesystem::path(stage_path).is_absolute()
                && std::filesystem::path(stage_path).has_filename(),
            "snapshot stage path must be one absolute file path"
        );
    }

    static void verify_input_versions_unchanged(
        const torch::Tensor& state,
        const torch::Tensor& metrics,
        EncodingResult& result
    ) {
        const auto& state_counter =
            state.unsafeGetTensorImpl()->version_counter();
        const auto& metrics_counter =
            metrics.unsafeGetTensorImpl()->version_counter();
        result.state_completion_version = result.state_version_guard_enabled
            ? state_counter.current_version()
            : 0;
        result.metrics_completion_version = result.metrics_version_guard_enabled
            ? metrics_counter.current_version()
            : 0;
        TORCH_CHECK(
            !result.state_version_guard_enabled
                || result.state_completion_version == result.state_input_version,
            "snapshot state changed during synchronous native encoding"
        );
        TORCH_CHECK(
            !result.metrics_version_guard_enabled
                || result.metrics_completion_version
                    == result.metrics_input_version,
            "snapshot metrics changed during synchronous native encoding"
        );
    }

    static EncodingResult write_stage_file(
        EncodingResult result,
        const std::string& stage_path,
        const std::string& encoding_policy,
        const std::optional<std::size_t> fault_after_bytes,
        const Clock::time_point stage_total_started
    ) {
        TORCH_CHECK(
            !fault_after_bytes.has_value()
                || *fault_after_bytes < result.gzip.size(),
            "snapshot stage fault injection must precede durable completion"
        );

        int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        const auto open_started = Clock::now();
        const int descriptor = ::open(stage_path.c_str(), flags, 0600);
        const int open_error = errno;
        result.stage_open_ns = elapsed_ns(open_started);
        TORCH_CHECK(
            descriptor >= 0,
            "native snapshot stage exclusive create failed: ",
            std::strerror(open_error)
        );
        bool descriptor_open = true;
        try {
            const auto write_started = Clock::now();
            std::size_t offset = 0;
            constexpr std::size_t kWriteChunkBytes = 1024 * 1024;
            while (offset < result.gzip.size()) {
                if (fault_after_bytes.has_value() && offset == *fault_after_bytes) {
                    TORCH_CHECK(
                        false,
                        "injected native snapshot stage write failure after ",
                        offset,
                        " bytes"
                    );
                }
                auto requested = std::min(
                    kWriteChunkBytes, result.gzip.size() - offset
                );
                if (fault_after_bytes.has_value()) {
                    requested = std::min(requested, *fault_after_bytes - offset);
                }
                TORCH_CHECK(
                    requested > 0,
                    "injected native snapshot stage write failure before completion"
                );
                const auto written = ::write(
                    descriptor, result.gzip.data() + offset, requested
                );
                if (written < 0 && errno == EINTR) {
                    continue;
                }
                const int write_error = errno;
                TORCH_CHECK(
                    written > 0,
                    "native snapshot stage write failed: ",
                    std::strerror(write_error)
                );
                offset += static_cast<std::size_t>(written);
            }
            result.stage_write_ns = elapsed_ns(write_started);
            result.stage_bytes_written = offset;

            const auto fsync_started = Clock::now();
            const int fsync_status = ::fsync(descriptor);
            const int fsync_error = errno;
            TORCH_CHECK(
                fsync_status == 0,
                "native snapshot stage fsync failed: ",
                std::strerror(fsync_error)
            );
            struct stat status {};
            const int stat_status = ::fstat(descriptor, &status);
            const int stat_error = errno;
            TORCH_CHECK(
                stat_status == 0,
                "native snapshot stage fstat failed: ",
                std::strerror(stat_error)
            );
            TORCH_CHECK(
                S_ISREG(status.st_mode)
                    && status.st_size >= 0
                    && static_cast<std::size_t>(status.st_size)
                        == result.gzip.size(),
                "native snapshot stage durable size differs"
            );
            result.stage_fsync_ns = elapsed_ns(fsync_started);

            const auto close_started = Clock::now();
            const int close_status = ::close(descriptor);
            const int close_error = errno;
            descriptor_open = false;
            result.stage_close_ns = elapsed_ns(close_started);
            TORCH_CHECK(
                close_status == 0,
                "native snapshot stage close failed: ",
                std::strerror(close_error)
            );
        } catch (...) {
            if (descriptor_open) {
                ::close(descriptor);
            }
            throw;
        }
        result.stage_path = stage_path;
        result.stage_encoding_policy = encoding_policy;
        result.stage_file_complete = true;
        result.native_encode_write_fsync_close_ns =
            elapsed_ns(stage_total_started);
        return result;
    }

    const FieldExtent& field(const std::string& name) const {
        const auto found = fields_.find(name);
        TORCH_CHECK(found != fields_.end(), "native snapshot field is absent: ", name);
        return found->second;
    }

    void require_field(
        const std::string& name,
        const std::string& dtype,
        const std::vector<std::size_t>& shape
    ) const {
        const auto& value = field(name);
        TORCH_CHECK(
            value.dtype == dtype && value.shape == shape,
            "native snapshot field geometry differs: ", name
        );
    }

    void validate_required_fields() {
        require_field("scalar_header_counters", "u64", {24});
        require_field("resource", "u16", {resource_count_});
        require_field("position", "u16", {capacity_});
        require_field("energy", "i16", {capacity_});
        require_field("health", "i16", {capacity_});
        require_field("age", "u16", {capacity_});
        require_field("absolute_birth_tick", "u64", {capacity_});
        require_field("alive_words", "u32", {capacity_ / alive_word_bits_});
        require_field("agent_id", "u64", {capacity_});
        require_field("parent_a_id", "u64", {capacity_});
        require_field("parent_b_id", "u64", {capacity_});
        const auto& policy = field("policy_alleles");
        const auto& trait = field("trait_alleles");
        TORCH_CHECK(
            policy.dtype == "i8" && policy.shape.size() == 3
                && policy.shape.back() == capacity_,
            "native snapshot policy geometry differs"
        );
        TORCH_CHECK(
            trait.dtype == "i16" && trait.shape.size() == 3
                && trait.shape.back() == capacity_,
            "native snapshot trait geometry differs"
        );
        policy_loci_ = policy.shape[0] * policy.shape[1];
        trait_loci_ = trait.shape[0] * trait.shape[1];
        TORCH_CHECK(
            policy.bytes == policy_loci_ * capacity_
                && trait.bytes == trait_loci_ * capacity_ * 2,
            "native snapshot allele byte geometry differs"
        );
        for (const auto* name : {
                 "scalar_header_counters",
                 "resource",
                 "position",
                 "energy",
                 "health",
                 "age",
                 "absolute_birth_tick",
                 "alive_words",
                 "agent_id",
                 "parent_a_id",
                 "parent_b_id",
                 "policy_alleles",
                 "trait_alleles",
             }) {
            const auto& extent = field(name);
            TORCH_CHECK(
                extent.offset <= canonical_state_bytes_per_world_
                    && extent.bytes
                        <= canonical_state_bytes_per_world_ - extent.offset,
                "native snapshot field exceeds the canonical state prefix: ",
                name
            );
        }
    }

    void append_world_row(
        std::string& output,
        const std::uint8_t* world,
        const WorldPlan& plan,
        const std::int64_t tick,
        const std::string& canonical_state_sha256
    ) const {
        const auto& header_field = field("scalar_header_counters");
        const auto header = [&](const std::size_t index) {
            return read_u64(world, header_field.offset + index * 8);
        };
        const auto& resource_field = field("resource");
        const auto* resource = world + resource_field.offset;
        std::uint64_t resource_total = 0;
        std::uint16_t resource_min = std::numeric_limits<std::uint16_t>::max();
        std::uint16_t resource_max = 0;
        std::size_t resource_nonzero = 0;
        for (std::size_t index = 0; index < resource_count_; ++index) {
            const auto value = read_u16(resource, index * 2);
            resource_total += value;
            resource_min = std::min(resource_min, value);
            resource_max = std::max(resource_max, value);
            resource_nonzero += value != 0;
        }
        const auto resource_sha256 = sha256_hex(resource, resource_field.bytes);
        std::string world_identity;
        world_identity.reserve(180);
        world_identity.append("{\"canonical_state_sha256\":\"");
        world_identity.append(canonical_state_sha256);
        world_identity.append("\",\"tick\":");
        append_i64(world_identity, tick);
        world_identity.append(",\"world_uid\":");
        append_u64(world_identity, plan.world_uid);
        world_identity.push_back('}');
        const auto world_state_id = sha256_prefixed(world_identity);

        output.append("{\"absolute_tick\":");
        append_i64(output, tick);
        output.append(",\"cell_id\":");
        output.append(plan.cell_id_json);
        output.append(",\"field_set_version\":\"");
        output.append(kFieldSetVersion);
        output.append("\",\"qualified_field_values\":{\"absolute_tick\":");
        append_i64(output, tick);
        output.append(",\"active_count\":");
        append_u64(output, header(3));
        output.append(",\"birth_count\":");
        append_u64(output, header(15));
        output.append(",\"death_count\":");
        append_u64(output, header(16));
        output.append(",\"mutation_count\":");
        append_u64(output, header(17));
        output.append(",\"next_absolute_tick\":");
        append_u64(output, header(1));
        output.append(",\"population_count\":");
        append_u64(output, header(3));
        output.append(",\"recombination_count\":");
        append_u64(output, header(18));
        output.append(",\"resource_max\":");
        append_u64(output, resource_max);
        output.append(",\"resource_mean\":");
        append_python_float(
            output,
            static_cast<double>(resource_total) / static_cast<double>(resource_count_)
        );
        output.append(",\"resource_min\":");
        append_u64(output, resource_min);
        output.append(",\"resource_nonzero_count\":");
        append_u64(output, resource_nonzero);
        output.append(",\"resource_state_sha256\":{\"algorithm\":\"sha256\",\"hex\":\"");
        output.append(resource_sha256);
        output.append("\"},\"resource_total\":");
        append_u64(output, resource_total);
        output.append(",\"world_uid\":");
        append_u64(output, plan.world_uid);
        output.append("},\"row_kind\":\"world_state\",\"snapshot_reason\":\"");
        output.append(kSnapshotReason);
        output.append("\",\"world_state_id\":\"");
        output.append(world_state_id);
        output.append("\",\"world_uid\":");
        append_u64(output, plan.world_uid);
        output.append("}\n");
    }

    std::string append_agent_row(
        std::string& output,
        const std::uint8_t* world,
        const std::int64_t* metrics,
        const WorldPlan& plan,
        const std::int64_t tick,
        const AgentSlot agent
    ) const {
        const auto slot = agent.slot;
        const auto parent_a = read_u64(world, field("parent_a_id").offset + slot * 8);
        const auto parent_b = read_u64(world, field("parent_b_id").offset + slot * 8);
        const auto position = read_u16(world, field("position").offset + slot * 2);
        const auto energy = read_i16(world, field("energy").offset + slot * 2);
        const auto health = read_i16(world, field("health").offset + slot * 2);
        const auto age = read_u16(world, field("age").offset + slot * 2);
        const auto birth_tick = read_u64(
            world, field("absolute_birth_tick").offset + slot * 8
        );

        std::string policy_hex;
        policy_hex.reserve(policy_loci_ * 2);
        std::string trait_hex;
        trait_hex.reserve(trait_loci_ * 4);
        const auto* policy = world + field("policy_alleles").offset;
        const auto* trait = world + field("trait_alleles").offset;
        std::vector<std::uint8_t> policy_bytes;
        policy_bytes.reserve(policy_loci_);
        for (std::size_t locus = 0; locus < policy_loci_; ++locus) {
            const auto raw = policy[locus * capacity_ + slot];
            policy_bytes.push_back(raw);
            append_hex_byte(policy_hex, raw);
        }
        for (std::size_t locus = 0; locus < trait_loci_; ++locus) {
            const auto offset = 2 * (locus * capacity_ + slot);
            append_hex_byte(trait_hex, trait[offset]);
            append_hex_byte(trait_hex, trait[offset + 1]);
        }
        std::string genome_identity;
        genome_identity.reserve(policy_hex.size() + trait_hex.size() + 32);
        genome_identity.append("{\"policy\":\"");
        genome_identity.append(policy_hex);
        genome_identity.append("\",\"trait\":\"");
        genome_identity.append(trait_hex);
        genome_identity.append("\"}");
        const auto genome_state_id = sha256_prefixed(genome_identity);
        const auto policy_id = std::string("sha256:")
            + sha256_hex(policy_bytes.data(), policy_bytes.size());

        std::string lineage_identity;
        lineage_identity.reserve(160);
        lineage_identity.append("{\"agent_id\":");
        append_u64(lineage_identity, agent.agent_id);
        lineage_identity.append(",\"parent_a\":");
        append_u64(lineage_identity, parent_a);
        lineage_identity.append(",\"parent_b\":");
        append_u64(lineage_identity, parent_b);
        lineage_identity.append(",\"world_uid\":");
        append_u64(lineage_identity, plan.world_uid);
        lineage_identity.push_back('}');
        const auto lineage_id = sha256_prefixed(lineage_identity);

        std::string founder_identity;
        founder_identity.reserve(160);
        founder_identity.append("{\"root_parent_a\":");
        append_u64(founder_identity, parent_a == 0 ? agent.agent_id : parent_a);
        founder_identity.append(",\"root_parent_b\":");
        append_u64(founder_identity, parent_b);
        founder_identity.append(",\"world_uid\":");
        append_u64(founder_identity, plan.world_uid);
        founder_identity.push_back('}');
        const auto founder_id = sha256_prefixed(founder_identity);

        std::string agent_identity;
        agent_identity.reserve(320);
        agent_identity.append("{\"age_ticks\":");
        append_u64(agent_identity, age);
        agent_identity.append(",\"agent_id\":");
        append_u64(agent_identity, agent.agent_id);
        agent_identity.append(",\"alive\":true,\"energy\":");
        append_i64(agent_identity, energy);
        agent_identity.append(",\"genome_state_id\":\"");
        agent_identity.append(genome_state_id);
        agent_identity.append("\",\"health\":");
        append_i64(agent_identity, health);
        agent_identity.append(",\"position\":");
        append_u64(agent_identity, position);
        agent_identity.append(",\"tick\":");
        append_i64(agent_identity, tick);
        agent_identity.append(",\"world_uid\":");
        append_u64(agent_identity, plan.world_uid);
        agent_identity.push_back('}');
        const auto agent_state_id = sha256_prefixed(agent_identity);

        output.append("{\"absolute_birth_tick\":");
        append_u64(output, birth_tick);
        output.append(",\"absolute_tick\":");
        append_i64(output, tick);
        output.append(",\"age_ticks\":");
        append_u64(output, age);
        output.append(",\"agent_id\":");
        append_u64(output, agent.agent_id);
        output.append(",\"agent_state_id\":\"");
        output.append(agent_state_id);
        output.append("\",\"alive\":true,\"cell_id\":");
        output.append(plan.cell_id_json);
        output.append(",\"energy\":");
        append_i64(output, energy);
        output.append(",\"founder_id\":\"");
        output.append(founder_id);
        output.append("\",\"genome_state_id\":\"");
        output.append(genome_state_id);
        output.append("\",\"health\":");
        append_i64(output, health);
        output.append(",\"hereditary_package_state_ids\":[],\"injury\":0,\"lifetime_metrics\":{");
        for (std::size_t metric = 0; metric < sorted_metric_fields_.size(); ++metric) {
            if (metric != 0) {
                output.push_back(',');
            }
            output.push_back('"');
            output.append(sorted_metric_fields_[metric].first);
            output.append("\":");
            append_i64(
                output,
                metrics[slot * metric_fields_.size() + sorted_metric_fields_[metric].second]
            );
        }
        output.append("},\"lineage_depth\":");
        append_u64(output, parent_a == 0 && parent_b == 0 ? 0 : 1);
        output.append(",\"lineage_id\":\"");
        output.append(lineage_id);
        output.append("\",\"linkage_state_id\":\"");
        output.append(genome_state_id);
        output.append("\",\"parent_a_id\":");
        append_u64(output, parent_a);
        output.append(",\"parent_b_id\":");
        append_u64(output, parent_b);
        output.append(",\"policy_id\":\"");
        output.append(policy_id);
        output.append("\",\"position\":");
        append_u64(output, position);
        output.append(",\"row_kind\":\"active_agent_state\",\"world_uid\":");
        append_u64(output, plan.world_uid);
        output.append("}\n");
        return genome_state_id;
    }

    void append_catalog_row(
        std::string& output,
        const std::uint8_t* world,
        const WorldPlan& plan,
        const std::int64_t tick,
        const std::string& genome_state_id,
        const std::size_t slot
    ) const {
        const auto* policy = world + field("policy_alleles").offset;
        const auto* trait = world + field("trait_alleles").offset;
        output.append("{\"absolute_tick\":");
        append_i64(output, tick);
        output.append(",\"cell_id\":");
        output.append(plan.cell_id_json);
        output.append(",\"genome_state_id\":\"");
        output.append(genome_state_id);
        output.append("\",\"policy\":[");
        for (std::size_t locus = 0; locus < policy_loci_; ++locus) {
            if (locus != 0) {
                output.push_back(',');
            }
            const auto value = policy[locus * capacity_ + slot];
            append_i64(output, static_cast<std::int8_t>(value));
        }
        output.append("],\"row_kind\":\"genome_catalog\",\"trait\":[");
        for (std::size_t locus = 0; locus < trait_loci_; ++locus) {
            if (locus != 0) {
                output.push_back(',');
            }
            append_i64(
                output,
                read_i16(trait, 2 * (locus * capacity_ + slot))
            );
        }
        output.append("],\"world_uid\":");
        append_u64(output, plan.world_uid);
        output.append("}\n");
    }

    std::string plan_sha256_;
    std::string native_world_control_set_sha256_;
    std::string execution_publication_context_sha256_;
    std::size_t world_count_{};
    std::size_t state_bytes_per_world_{};
    std::size_t canonical_state_bytes_per_world_{};
    std::size_t capacity_{};
    std::size_t alive_word_bits_{};
    std::size_t resource_count_{};
    std::size_t policy_loci_{};
    std::size_t trait_loci_{};
    std::string state_capture_preferred_mode_;
    std::vector<StateCaptureMode> state_capture_modes_;
    std::map<std::string, FieldExtent> fields_;
    std::vector<std::string> metric_fields_;
    std::vector<std::pair<std::string, std::size_t>> sorted_metric_fields_;
    std::vector<WorldPlan> worlds_;
    std::vector<BoundaryPlan> boundaries_;
};

py::dict result_to_python(
    const NativeSnapshotEncoderPlanV1& plan,
    EncodingResult&& result,
    const bool include_payload_bytes
) {
    py::dict timing;
    timing["clock"] = "std::chrono::steady_clock";
    timing["input_validation_ns"] = result.validation_ns;
    timing["selection_materialization_and_json_encoding_ns"] =
        result.selection_encoding_ns;
    timing["gzip_level_9_ns"] = result.compression_ns;
    timing["content_hashing_ns"] = result.content_hashing_ns;
    timing["active_metrics_content_hashing_ns"] =
        result.active_metrics_content_hashing_ns;
    timing["core_encode_compress_hash_ns"] =
        result.core_encode_compress_hash_ns;
    if (result.stage_file_complete) {
        timing["native_stage_file_open_ns"] = result.stage_open_ns;
        timing["native_stage_file_write_ns"] = result.stage_write_ns;
        timing["native_stage_file_fsync_and_fstat_ns"] = result.stage_fsync_ns;
        timing["native_stage_file_close_ns"] = result.stage_close_ns;
        timing["native_encode_write_fsync_close_ns"] =
            result.native_encode_write_fsync_close_ns;
    }

    py::list indices;
    py::list uids;
    py::list captured;
    py::list metric_fields;
    for (const auto index : result.selected_world_indices) {
        indices.append(index);
    }
    for (const auto uid : result.selected_world_uids) {
        uids.append(uid);
    }
    for (const auto& name : plan.metric_fields()) {
        metric_fields.append(name);
    }
    for (const auto& world : result.captured_worlds) {
        py::dict child;
        child["world_index"] = world.world_index;
        child["world_uid"] = world.world_uid;
        child["active_agent_count"] = world.active_agent_count;
        child["canonical_state_sha256"] = world.canonical_state_sha256;
        captured.append(std::move(child));
    }

    py::dict descriptor;
    descriptor["schema_version"] = include_payload_bytes
        ? kReceiptSchema
        : kStageReceiptSchema;
    descriptor["implementation"] = "native_host_contiguous_synchronous_v1";
    descriptor["claim_class"] = "TECHNICAL_ONLY";
    descriptor["plan_sha256"] = plan.plan_sha256();
    descriptor["native_world_control_set_sha256"] =
        plan.native_world_control_set_sha256();
    descriptor["execution_publication_context_sha256"] =
        plan.execution_publication_context_sha256();
    descriptor["absolute_tick"] = result.absolute_tick;
    descriptor["row_count"] = result.row_count;
    descriptor["captured_world_count"] = result.captured_worlds.size();
    descriptor["active_agent_count"] = result.active_agent_count;
    descriptor["state_input_row_bytes"] = result.state_input_row_bytes;
    descriptor["state_input_bytes"] = result.state_input_bytes;
    descriptor["canonical_state_bytes_selected"] =
        result.canonical_state_bytes_selected;
    descriptor["resident_state_row_bytes"] = result.resident_state_row_bytes;
    descriptor["state_capture_transport_schema_version"] =
        kStateCaptureTransportSchema;
    descriptor["state_capture_mode"] = result.state_capture_mode;
    descriptor["state_capture_preferred_mode"] =
        result.state_capture_preferred_mode;
    descriptor["state_capture_mode_is_preferred"] =
        result.state_capture_mode == result.state_capture_preferred_mode;
    descriptor["selected_world_indices"] = std::move(indices);
    descriptor["selected_world_uids"] = std::move(uids);
    descriptor["captured_worlds"] = std::move(captured);
    descriptor["metric_fields"] = std::move(metric_fields);
    descriptor["genome_catalog_included"] = result.include_genome_catalog;
    descriptor["uncompressed_content_sha256"] = result.uncompressed_sha256;
    descriptor["compressed_file_sha256"] = result.compressed_sha256;
    descriptor["active_metrics_content_sha256"] =
        result.active_metrics_content_sha256;
    descriptor["active_metrics_authentication"] =
        result.active_metrics_authentication;
    descriptor["uncompressed_bytes"] = result.canonical_jsonl.size();
    descriptor["compressed_bytes"] = result.gzip.size();
    descriptor["zlib_compile_version"] = std::string(ZLIB_VERSION);
    descriptor["zlib_runtime_version"] = std::string(zlibVersion());
    descriptor["host_only"] = true;
    descriptor["synchronous"] = true;
    descriptor["input_ownership_contract"] = result.input_ownership_contract;
    descriptor["gil_held_for_full_synchronous_call"] =
        result.gil_held_for_full_synchronous_call;
    descriptor["state_version_guard_enabled"] =
        result.state_version_guard_enabled;
    descriptor["metrics_version_guard_enabled"] =
        result.metrics_version_guard_enabled;
    descriptor["state_input_version"] = result.state_input_version;
    descriptor["metrics_input_version"] = result.metrics_input_version;
    descriptor["state_completion_version"] =
        result.state_completion_version;
    descriptor["metrics_completion_version"] =
        result.metrics_completion_version;
    descriptor["input_versions_unchanged"] = true;
    descriptor["filesystem_publication_performed"] = false;
    descriptor["timing"] = std::move(timing);
    if (include_payload_bytes) {
        descriptor["canonical_jsonl"] = py::bytes(
            result.canonical_jsonl.data(), result.canonical_jsonl.size()
        );
        descriptor["deterministic_gzip_level_9"] = py::bytes(
            reinterpret_cast<const char*>(result.gzip.data()), result.gzip.size()
        );
    } else {
        TORCH_CHECK(
            result.stage_file_complete,
            "native snapshot stage result is not durably complete"
        );
        descriptor["stage_path"] = result.stage_path;
        descriptor["stage_encoding_policy"] = result.stage_encoding_policy;
        descriptor["stage_bytes_written"] = result.stage_bytes_written;
        descriptor["stage_file_state"] = "FSYNCED_STAGE_ONLY_NOT_PUBLISHED";
        descriptor["stage_file_fsync_completed"] = true;
        descriptor["stage_file_close_completed"] = true;
        descriptor["directory_fsync_performed"] = false;
        descriptor["atomic_publication_performed"] = false;
        descriptor["partial_stage_cleanup_owner"] = "python_lifecycle_authority";
    }
    return descriptor;
}

}  // namespace
}  // namespace eiel::snapshot::encoder_v1

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    using eiel::snapshot::encoder_v1::NativeSnapshotEncoderPlanV1;
    module.def("abi_version", []() { return eiel::snapshot::encoder_v1::kAbi; });
    module.def("extension_version", []() {
        return eiel::snapshot::encoder_v1::kExtension;
    });
    module.def("zlib_compile_version", []() { return std::string(ZLIB_VERSION); });
    module.def("zlib_runtime_version", []() {
        return std::string(zlibVersion());
    });
    py::class_<NativeSnapshotEncoderPlanV1>(module, "NativeSnapshotEncoderPlanV1")
        .def_property_readonly(
            "plan_sha256", &NativeSnapshotEncoderPlanV1::plan_sha256
        )
        .def(
            "encode",
            [](const NativeSnapshotEncoderPlanV1& plan,
               const torch::Tensor& state,
               const py::dict& active_metrics_projection,
               const py::object& absolute_tick) {
                TORCH_CHECK(
                    PyGILState_Check() != 0,
                    "synchronous snapshot encoder requires exclusive Python ownership"
                );
                auto result = plan.encode_projection(
                    state, active_metrics_projection, absolute_tick
                );
                return eiel::snapshot::encoder_v1::result_to_python(
                    plan, std::move(result), true
                );
            },
            py::arg("state"),
            py::arg("active_metrics_projection"),
            py::arg("absolute_tick")
        )
        .def(
            "encode_to_stage_file",
            [](const NativeSnapshotEncoderPlanV1& plan,
               const torch::Tensor& state,
               const py::dict& active_metrics_projection,
               const py::object& absolute_tick,
               const std::string& stage_path,
               const std::string& encoding_policy,
               const py::object& fault_after_bytes) {
                TORCH_CHECK(
                    PyGILState_Check() != 0,
                    "synchronous snapshot stage encoder requires exclusive Python ownership"
                );
                std::optional<std::size_t> fault;
                if (!fault_after_bytes.is_none()) {
                    TORCH_CHECK(
                        PyLong_CheckExact(fault_after_bytes.ptr()),
                        "snapshot stage fault injection must be an exact integer"
                    );
                    fault = py::cast<std::size_t>(fault_after_bytes);
                }
                auto result = plan.encode_projection_to_stage_file(
                    state,
                    active_metrics_projection,
                    absolute_tick,
                    stage_path,
                    encoding_policy,
                    fault
                );
                return eiel::snapshot::encoder_v1::result_to_python(
                    plan, std::move(result), false
                );
            },
            py::arg("state"),
            py::arg("active_metrics_projection"),
            py::arg("absolute_tick"),
            py::arg("stage_path"),
            py::arg("encoding_policy"),
            py::arg("fault_after_bytes") = py::none()
        )
        .def(
            "encode_owned_tensors_to_stage_file",
            [](const NativeSnapshotEncoderPlanV1& plan,
               const torch::Tensor& state,
               const torch::Tensor& metrics,
               const py::object& absolute_tick,
               const std::string& stage_path,
               const std::string& encoding_policy,
               const py::object& fault_after_bytes) {
                TORCH_CHECK(
                    PyGILState_Check() != 0,
                    "owned snapshot stage encoder requires the GIL at entry"
                );
                TORCH_CHECK(
                    PyLong_CheckExact(absolute_tick.ptr()),
                    "snapshot absolute tick must be an exact integer"
                );
                const auto tick = py::cast<std::int64_t>(absolute_tick);
                TORCH_CHECK(tick >= 0, "snapshot absolute tick is invalid");
                std::optional<std::size_t> fault;
                if (!fault_after_bytes.is_none()) {
                    TORCH_CHECK(
                        PyLong_CheckExact(fault_after_bytes.ptr()),
                        "snapshot stage fault injection must be an exact integer"
                    );
                    fault = py::cast<std::size_t>(fault_after_bytes);
                }
                std::optional<eiel::snapshot::encoder_v1::EncodingResult> result;
                {
                    py::gil_scoped_release release;
                    result.emplace(plan.encode_owned_tensors_to_stage_file(
                        state,
                        metrics,
                        tick,
                        stage_path,
                        encoding_policy,
                        fault
                    ));
                }
                plan.verify_owned_tensor_versions_at_python_boundary(
                    state, metrics, *result
                );
                return eiel::snapshot::encoder_v1::result_to_python(
                    plan, std::move(*result), false
                );
            },
            py::arg("state"),
            py::arg("metrics"),
            py::arg("absolute_tick"),
            py::arg("stage_path"),
            py::arg("encoding_policy"),
            py::arg("fault_after_bytes") = py::none()
        );
    module.def(
        "compile_snapshot_encoder_plan_v1",
        [](const py::bytes& canonical_plan_bytes, const py::dict& plan) {
            return std::make_unique<NativeSnapshotEncoderPlanV1>(
                canonical_plan_bytes, plan
            );
        },
        py::arg("canonical_plan_bytes"), py::arg("plan")
    );
}
