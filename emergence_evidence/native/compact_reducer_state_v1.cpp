#include "compact_reducer_core_v1.h"
#include "compact_reducer_internal_v1.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>
#include <zlib.h>

namespace {

using BirthReceipt = eiel_native_compact_birth_batch_receipt_v1;
using Failure = eiel_native_compact_failure_v1;
using Status = eiel_native_compact_reducer_status_v1;
using WorldReceipt = eiel_native_compact_world_receipt_v1;

constexpr std::size_t kGenomeBytes = EIEL_COMPACT_GENOME_ROW_BYTES_V1;
constexpr std::size_t kEventRowBytes = 32U;
constexpr std::size_t kMetricRowBytes = EIEL_COMPACT_DEATH_METRIC_ROW_BYTES_V1;
constexpr std::size_t kProvenanceBytes =
    EIEL_COMPACT_PACKED_BIRTH_PROVENANCE_BYTES_V1;
constexpr std::size_t kMetricFields = EIEL_COMPACT_DEATH_METRIC_FIELDS_V1;
constexpr std::size_t kMaxStrategies = EIEL_COMPACT_STRATEGY_MAX_IDS_V1;
constexpr std::size_t kMaxThresholds =
    EIEL_COMPACT_STRATEGY_MAX_THRESHOLDS_V1;
constexpr std::size_t kMaxStrategyGroups =
    EIEL_COMPACT_STRATEGY_MAX_GROUPS_V1;
constexpr std::size_t kMaxParentPairs =
    EIEL_COMPACT_STRATEGY_MAX_PARENT_PAIRS_V1;
constexpr std::size_t kMaxPolicyLoci =
    EIEL_COMPACT_SNAPSHOT_MAX_POLICY_LOCI_V1;
constexpr std::size_t kMaxTraitLoci =
    EIEL_COMPACT_SNAPSHOT_MAX_TRAIT_LOCI_V1;
constexpr std::size_t kPolicyLociPerHaplotype = 112U;
constexpr std::size_t kPolicyValueCount = 224U;
constexpr std::size_t kTraitLociPerHaplotype = 16U;
constexpr std::size_t kHereditaryBirthRecordWidth = 374U;
constexpr std::size_t kHereditaryBirthRecordHeaderBytes = 33U;
constexpr std::size_t kHereditaryBirthRecordOutputBufferBytes = 64U * 1024U;
constexpr char kHereditaryBirthRecordHeader[] =
    "EIEL_HEREDITARY_BIRTH_RECORDS_V1\n";
constexpr std::uint64_t kTombstone = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kMissingSlot = std::numeric_limits<std::uint32_t>::max();

using Digest256 = std::array<std::uint8_t, 32U>;

class GzipHereditaryBirthRecordSink {
public:
    explicit GzipHereditaryBirthRecordSink(std::string path)
        : path_(std::move(path)) {}

    GzipHereditaryBirthRecordSink(const GzipHereditaryBirthRecordSink&) = delete;
    GzipHereditaryBirthRecordSink& operator=(
        const GzipHereditaryBirthRecordSink&) = delete;

    ~GzipHereditaryBirthRecordSink() {
        if (fd_ >= 0) {
            if (stream_initialized_) deflateEnd(&stream_);
            (void)close_fd();
            if (!finalized_) ::unlink(path_.c_str());
        }
    }

    [[nodiscard]] bool open() noexcept {
        if (path_.empty() || path_.size() > 4096U
            || path_.find('\0') != std::string::npos) {
            return false;
        }
        fd_ = ::open(
            path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd_ < 0) return false;
        std::memset(&stream_, 0, sizeof(stream_));
        if (deflateInit2(
                &stream_, Z_BEST_SPEED, Z_DEFLATED, 15 + 16, 8,
                Z_DEFAULT_STRATEGY)
            != Z_OK) {
            abort_unpublished();
            return false;
        }
        stream_initialized_ = true;
        if (!compress(
                reinterpret_cast<const std::uint8_t*>(
                    kHereditaryBirthRecordHeader),
                kHereditaryBirthRecordHeaderBytes, Z_NO_FLUSH)) {
            abort_unpublished();
            return false;
        }
        return true;
    }

    [[nodiscard]] bool write_record(
        const std::uint16_t world_index,
        const std::int64_t tick,
        const std::uint64_t child_agent_id,
        const std::uint64_t parent_a_id,
        const std::uint64_t parent_b_id,
        const Digest256& parent_a_full_id,
        const Digest256& parent_b_full_id,
        const Digest256& child_full_id,
        const Digest256& parent_a_tracked_id,
        const Digest256& parent_b_tracked_id,
        const Digest256& child_tracked_id,
        const std::uint64_t lineage_depth,
        const std::uint32_t mutation_origin_count,
        const std::uint32_t recombination_pattern,
        const std::uint8_t* const birth_provenance) noexcept {
        if (!stream_initialized_ || birth_provenance == nullptr) return false;
        std::array<std::uint8_t, kHereditaryBirthRecordWidth> record{};
        write_u16(record.data(), world_index);
        write_u64(record.data() + 2U, static_cast<std::uint64_t>(tick));
        write_u64(record.data() + 10U, child_agent_id);
        write_u64(record.data() + 18U, parent_a_id);
        write_u64(record.data() + 26U, parent_b_id);
        // The compact native ABI does not carry transient agent-state IDs for
        // born parents.  The field remains explicitly null, while parent
        // agent IDs and exact full hereditary IDs are retained below.
        std::memcpy(record.data() + 98U, parent_a_full_id.data(), 32U);
        std::memcpy(record.data() + 130U, parent_b_full_id.data(), 32U);
        std::memcpy(record.data() + 162U, child_full_id.data(), 32U);
        std::memcpy(record.data() + 194U, parent_a_tracked_id.data(), 32U);
        std::memcpy(record.data() + 226U, parent_b_tracked_id.data(), 32U);
        std::memcpy(record.data() + 258U, child_tracked_id.data(), 32U);
        write_u64(record.data() + 290U, lineage_depth);
        write_u32(record.data() + 298U, mutation_origin_count);
        write_u32(record.data() + 302U, recombination_pattern);
        std::memcpy(record.data() + 306U, birth_provenance, kProvenanceBytes);
        if (!compress(record.data(), record.size(), Z_NO_FLUSH)) return false;
        ++record_count_;
        return true;
    }

    [[nodiscard]] bool finalize() noexcept {
        if (finalized_) return true;
        if (!stream_initialized_ || !compress(nullptr, 0U, Z_FINISH)) {
            abort_unpublished();
            return false;
        }
        const int deflate_status = deflateEnd(&stream_);
        stream_initialized_ = false;
        if (deflate_status != Z_OK) {
            abort_unpublished();
            return false;
        }
        if (::fsync(fd_) != 0) {
            abort_unpublished();
            return false;
        }
        if (!close_fd()) {
            ::unlink(path_.c_str());
            return false;
        }
        finalized_ = true;
        return true;
    }

private:
    static void write_u16(std::uint8_t* const output, const std::uint16_t value) {
        output[0] = static_cast<std::uint8_t>(value & 0xffU);
        output[1] = static_cast<std::uint8_t>((value >> 8U) & 0xffU);
    }

    static void write_u32(std::uint8_t* const output, const std::uint32_t value) {
        for (std::size_t byte = 0U; byte < 4U; ++byte) {
            output[byte] = static_cast<std::uint8_t>(value >> (byte * 8U));
        }
    }

    static void write_u64(std::uint8_t* const output, const std::uint64_t value) {
        for (std::size_t byte = 0U; byte < 8U; ++byte) {
            output[byte] = static_cast<std::uint8_t>(value >> (byte * 8U));
        }
    }

    [[nodiscard]] bool write_all(
        const std::uint8_t* const data, const std::size_t size) noexcept {
        std::size_t offset = 0U;
        while (offset < size) {
            const ssize_t written = ::write(fd_, data + offset, size - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) return false;
            offset += static_cast<std::size_t>(written);
        }
        return true;
    }

    [[nodiscard]] bool compress(
        const std::uint8_t* const data,
        const std::size_t size,
        const int flush) noexcept {
        if (!stream_initialized_
            || size > static_cast<std::size_t>(
                std::numeric_limits<uInt>::max())) {
            return false;
        }
        stream_.next_in = const_cast<Bytef*>(data);
        stream_.avail_in = static_cast<uInt>(size);
        while (stream_.avail_in > 0U || flush == Z_FINISH) {
            stream_.next_out = output_buffer_.data();
            stream_.avail_out = static_cast<uInt>(output_buffer_.size());
            const int status = deflate(&stream_, flush);
            if (status != Z_OK && !(flush == Z_FINISH && status == Z_STREAM_END)) {
                return false;
            }
            const std::size_t produced = output_buffer_.size()
                - static_cast<std::size_t>(stream_.avail_out);
            if (!write_all(output_buffer_.data(), produced)) return false;
            if (flush == Z_FINISH && status == Z_STREAM_END) return true;
            if (flush != Z_FINISH && stream_.avail_in == 0U) return true;
        }
        return true;
    }

    [[nodiscard]] bool close_fd() noexcept {
        if (fd_ < 0) return true;
        const int result = ::close(fd_);
        fd_ = -1;
        return result == 0;
    }

    void abort_unpublished() noexcept {
        if (stream_initialized_) {
            deflateEnd(&stream_);
            stream_initialized_ = false;
        }
        (void)close_fd();
        ::unlink(path_.c_str());
        finalized_ = false;
    }

    std::string path_;
    int fd_ = -1;
    z_stream stream_{};
    std::array<std::uint8_t, kHereditaryBirthRecordOutputBufferBytes>
        output_buffer_{};
    std::uint64_t record_count_ = 0U;
    bool stream_initialized_ = false;
    bool finalized_ = false;
};

struct AgentState {
    std::uint64_t agent_id = 0U;
    std::uint64_t generation = 0U;
    std::uint64_t strategy_mask = 0U;
    std::int64_t birth_tick = 0;
    std::array<std::int64_t, kMetricFields> latest_metrics{};
    std::uint64_t strategy_snapshot_epoch = 0U;
    bool mutation_lineage = false;
    bool born = false;
    bool reproduced = false;
    bool founder = false;
    bool latest_metrics_present = false;
    bool genotype_reproductive_event_recorded = false;
    std::array<std::uint8_t, kGenomeBytes> genome{};
    Digest256 genome_digest{};
    bool genome_digest_present = false;
    Digest256 genotype_digest{};
    bool genotype_digest_present = false;
};

struct SnapshotPolicyMapping {
    Digest256 policy_digest{};
    std::uint64_t strategy_mask = 0U;
};

struct SnapshotConfiguration {
    bool configured = false;
    bool genotype_attribution_only = false;
    bool genotype_focal_filter_enabled = false;
    std::string cell_id;
    std::vector<std::uint16_t> policy_loci;
    std::vector<std::uint16_t> trait_loci;
    std::vector<Digest256> genotype_focal_ids;
    std::uint8_t strategy_count = 0U;
    std::vector<SnapshotPolicyMapping> strategy_policies;
    std::vector<std::int64_t> thresholds;
    std::int64_t early_life_tick = 0;
};

struct StrategyGroupCounter {
    std::uint64_t mask = 0U;
    std::uint64_t count = 0U;
};

struct StrategyParentPairCounter {
    std::uint64_t first_mask = 0U;
    std::uint64_t second_mask = 0U;
    std::uint64_t count = 0U;
};

struct GenotypeCounter {
    std::uint64_t agent_count = 0U;
    std::uint64_t founder_count = 0U;
    std::uint64_t offspring_birth_count = 0U;
    std::uint64_t parent_birth_event_count = 0U;
    std::uint64_t reproductive_agent_count = 0U;
    std::uint64_t death_count = 0U;
    std::uint64_t terminal_alive_count = 0U;
    std::array<__int128, kMetricFields> metric_totals{};
};

struct StrategyState {
    bool enabled = false;
    bool finalized = false;
    std::uint8_t strategy_count = 0U;
    std::uint8_t threshold_count = 0U;
    std::int64_t early_life_tick = 0;
    std::int64_t first_birth_tick = -1;
    std::int64_t first_death_tick = -1;
    std::int64_t extinction_tick = -1;
    std::uint64_t total_agent_count = 0U;
    std::uint64_t founder_count = 0U;
    std::uint64_t birth_count = 0U;
    std::uint64_t death_count = 0U;
    std::uint64_t maximum_lineage_depth = 0U;
    std::uint64_t mixed_parent_birth_count = 0U;
    std::uint64_t snapshot_epoch = 0U;
    std::array<std::int64_t, kMaxThresholds> thresholds{};
    std::array<std::uint64_t, kMaxStrategies> agent_counts{};
    std::array<std::uint64_t, kMaxStrategies> offspring_born{};
    std::array<std::uint64_t, kMaxStrategies * kMaxThresholds>
        offspring_survived{};
    std::array<std::uint64_t, kMaxStrategies> max_depth_by_strategy{};
    std::array<std::uint64_t, kMaxStrategies> parent_birth_events{};
    std::array<std::uint64_t, kMaxStrategies> death_events{};
    std::array<std::uint64_t, kMaxStrategies> persistence_to_early{};
    std::array<std::uint64_t, kMaxStrategies> terminal_alive{};
    std::array<__int128, kMaxStrategies * kMetricFields> metric_totals{};
    std::array<StrategyGroupCounter, kMaxStrategyGroups> offspring_groups{};
    std::size_t offspring_group_count = 0U;
    std::array<StrategyParentPairCounter, kMaxParentPairs> parent_pairs{};
    std::size_t parent_pair_count = 0U;
    std::vector<std::uint16_t> genotype_policy_loci;
    std::vector<std::uint16_t> genotype_trait_loci;
    bool genotype_focal_filter_enabled = false;
    std::vector<Digest256> genotype_focal_ids;
    std::map<Digest256, GenotypeCounter> genotype_groups;
    std::map<Digest256, std::uint64_t> descendant_birth_counts;
    std::string genotype_result;
    bool genotype_result_ready = false;
};

[[nodiscard]] bool genotype_group_is_retained(
    const StrategyState& strategy, const Digest256& digest) noexcept {
    if (!strategy.genotype_focal_filter_enabled) return true;
    return std::binary_search(
        strategy.genotype_focal_ids.begin(),
        strategy.genotype_focal_ids.end(),
        digest);
}

struct MapEntry {
    std::uint64_t key = 0U;
    std::uint32_t slot = 0U;
};

[[nodiscard]] std::uint64_t hash_agent_id(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

class ActiveMap {
public:
    explicit ActiveMap(const std::uint32_t resident_capacity) {
        std::size_t extent = 8U;
        const std::size_t target =
            static_cast<std::size_t>(resident_capacity) * 4U;
        while (extent < target) extent *= 2U;
        entries_.resize(extent);
        mask_ = extent - 1U;
    }

    [[nodiscard]] std::uint32_t find(const std::uint64_t key) const noexcept {
        if (key == 0U || key == kTombstone) return kMissingSlot;
        std::size_t position = static_cast<std::size_t>(hash_agent_id(key)) & mask_;
        for (std::size_t visited = 0U; visited < entries_.size(); ++visited) {
            const MapEntry& entry = entries_[position];
            if (entry.key == 0U) return kMissingSlot;
            if (entry.key == key) return entry.slot;
            position = (position + 1U) & mask_;
        }
        return kMissingSlot;
    }

    [[nodiscard]] bool insert(
        const std::uint64_t key, const std::uint32_t slot) noexcept {
        if (key == 0U || key == kTombstone) return false;
        std::size_t position = static_cast<std::size_t>(hash_agent_id(key)) & mask_;
        std::size_t first_tombstone = entries_.size();
        for (std::size_t visited = 0U; visited < entries_.size(); ++visited) {
            MapEntry& entry = entries_[position];
            if (entry.key == key) return false;
            if (entry.key == kTombstone && first_tombstone == entries_.size()) {
                first_tombstone = position;
            }
            if (entry.key == 0U) {
                const std::size_t destination =
                    first_tombstone == entries_.size() ? position : first_tombstone;
                if (first_tombstone != entries_.size()) --tombstone_count_;
                entries_[destination] = MapEntry{key, slot};
                return true;
            }
            position = (position + 1U) & mask_;
        }
        if (first_tombstone != entries_.size()) {
            entries_[first_tombstone] = MapEntry{key, slot};
            --tombstone_count_;
            return true;
        }
        return false;
    }

    [[nodiscard]] bool erase(const std::uint64_t key) noexcept {
        std::size_t position = static_cast<std::size_t>(hash_agent_id(key)) & mask_;
        for (std::size_t visited = 0U; visited < entries_.size(); ++visited) {
            MapEntry& entry = entries_[position];
            if (entry.key == 0U) return false;
            if (entry.key == key) {
                entry.key = kTombstone;
                entry.slot = 0U;
                ++tombstone_count_;
                return true;
            }
            position = (position + 1U) & mask_;
        }
        return false;
    }

    void rebuild(const std::vector<AgentState>& agents) noexcept {
        if (tombstone_count_ <= entries_.size() / 2U) return;
        std::fill(entries_.begin(), entries_.end(), MapEntry{});
        tombstone_count_ = 0U;
        for (std::size_t slot = 0U; slot < agents.size(); ++slot) {
            if (agents[slot].agent_id != 0U) {
                const bool inserted = insert(
                    agents[slot].agent_id, static_cast<std::uint32_t>(slot));
                if (!inserted) std::terminate();
            }
        }
    }

private:
    std::vector<MapEntry> entries_;
    std::size_t mask_ = 0U;
    std::size_t tombstone_count_ = 0U;
};

struct WorldState {
    WorldState(
        const std::uint32_t capacity,
        const std::int64_t start_tick,
        std::vector<std::int64_t> segment_starts,
        std::vector<std::int32_t> segment_steps)
        : resident_capacity(capacity),
          next_absolute_tick(start_tick),
          mutation_segment_starts(std::move(segment_starts)),
          mutation_segment_steps(std::move(segment_steps)),
          agents(capacity),
          active_map(capacity) {
        free_slots.reserve(capacity);
        for (std::uint32_t slot = capacity; slot > 0U; --slot) {
            free_slots.push_back(slot - 1U);
        }
    }

    [[nodiscard]] std::int32_t mutation_step_at(
        const std::int64_t tick) noexcept {
        while (mutation_segment_cursor + 1U < mutation_segment_starts.size()
               && mutation_segment_starts[mutation_segment_cursor + 1U] <= tick) {
            ++mutation_segment_cursor;
        }
        return mutation_segment_steps[mutation_segment_cursor];
    }

    std::uint32_t resident_capacity;
    std::int64_t next_absolute_tick;
    std::vector<std::int64_t> mutation_segment_starts;
    std::vector<std::int32_t> mutation_segment_steps;
    std::size_t mutation_segment_cursor = 0U;
    std::vector<AgentState> agents;
    std::vector<std::uint32_t> free_slots;
    ActiveMap active_map;
    BirthReceipt birth_receipt{};
    std::array<__int128, kMetricFields> metric_totals{};
    std::uint64_t birth_count = 0U;
    std::uint64_t death_count = 0U;
    std::uint64_t recombination_birth_count = 0U;
    std::uint64_t carrier_reproduction_birth_count = 0U;
    std::uint64_t mutation_carrier_reproduction_birth_count = 0U;
    std::uint64_t unique_born_agents_reproduced_count = 0U;
    std::uint64_t unique_mutation_carriers_reproduced_count = 0U;
    std::uint64_t maximum_lineage_depth = 0U;
    std::uint64_t maximum_seen_agent_id = 0U;
    std::uint64_t dispatch_count = 0U;
    std::uint64_t event_count = 0U;
    std::uint64_t dispatch_tick_count = 0U;
    std::uint64_t snapshot_epoch = 0U;
    bool initialized = false;
    StrategyState strategy{};
    SnapshotConfiguration snapshot{};
    std::unique_ptr<GzipHereditaryBirthRecordSink> hereditary_birth_sink;
};

struct ReducerState {
    std::vector<WorldState> worlds;
    std::int32_t max_worker_threads = 1;
    bool poisoned = false;
    Failure failure{};
    std::mutex failure_mutex;
    std::string snapshot_result;
    bool snapshot_result_ready = false;
};

[[nodiscard]] std::uint32_t read_u32_le(const std::uint8_t* source) noexcept {
    return static_cast<std::uint32_t>(source[0])
        | (static_cast<std::uint32_t>(source[1]) << 8U)
        | (static_cast<std::uint32_t>(source[2]) << 16U)
        | (static_cast<std::uint32_t>(source[3]) << 24U);
}

[[nodiscard]] std::uint64_t read_u64_le(const std::uint8_t* source) noexcept {
    std::uint64_t value = 0U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
        value |= static_cast<std::uint64_t>(source[byte]) << (byte * 8U);
    }
    return value;
}

[[nodiscard]] std::int64_t read_i64_le(const std::uint8_t* source) noexcept {
    const std::uint64_t bits = read_u64_le(source);
    std::int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] std::uint32_t rotate_right(
    const std::uint32_t value, const unsigned int bits) noexcept {
    return (value >> bits) | (value << (32U - bits));
}

[[nodiscard]] Digest256 sha256_bytes(
    const std::uint8_t* const data, const std::size_t size) noexcept {
    static constexpr std::array<std::uint32_t, 64U> kRoundConstants{{
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
    std::array<std::uint32_t, 8U> hash{{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    }};
    const auto transform = [&](const std::uint8_t* const block) noexcept {
        std::array<std::uint32_t, 64U> words{};
        for (std::size_t index = 0U; index < 16U; ++index) {
            const std::size_t offset = index * 4U;
            words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U)
                | (static_cast<std::uint32_t>(block[offset + 1U]) << 16U)
                | (static_cast<std::uint32_t>(block[offset + 2U]) << 8U)
                | static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t first = words[index - 15U];
            const std::uint32_t second = words[index - 2U];
            const std::uint32_t sigma_zero =
                rotate_right(first, 7U) ^ rotate_right(first, 18U)
                ^ (first >> 3U);
            const std::uint32_t sigma_one =
                rotate_right(second, 17U) ^ rotate_right(second, 19U)
                ^ (second >> 10U);
            words[index] = words[index - 16U] + sigma_zero
                + words[index - 7U] + sigma_one;
        }
        std::uint32_t a = hash[0];
        std::uint32_t b = hash[1];
        std::uint32_t c = hash[2];
        std::uint32_t d = hash[3];
        std::uint32_t e = hash[4];
        std::uint32_t f = hash[5];
        std::uint32_t g = hash[6];
        std::uint32_t h = hash[7];
        for (std::size_t index = 0U; index < words.size(); ++index) {
            const std::uint32_t sum_one =
                rotate_right(e, 6U) ^ rotate_right(e, 11U)
                ^ rotate_right(e, 25U);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary_one = h + sum_one + choose
                + kRoundConstants[index] + words[index];
            const std::uint32_t sum_zero =
                rotate_right(a, 2U) ^ rotate_right(a, 13U)
                ^ rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary_two = sum_zero + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary_one;
            d = c;
            c = b;
            b = a;
            a = temporary_one + temporary_two;
        }
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    };
    std::size_t offset = 0U;
    while (size - offset >= 64U) {
        transform(data + offset);
        offset += 64U;
    }
    std::array<std::uint8_t, 128U> final_blocks{};
    const std::size_t remainder = size - offset;
    if (remainder > 0U) {
        std::memcpy(final_blocks.data(), data + offset, remainder);
    }
    final_blocks[remainder] = 0x80U;
    const std::size_t final_size = remainder < 56U ? 64U : 128U;
    const std::uint64_t bit_size = static_cast<std::uint64_t>(size) * 8U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
        final_blocks[final_size - 1U - byte] = static_cast<std::uint8_t>(
            bit_size >> (byte * 8U));
    }
    transform(final_blocks.data());
    if (final_size == 128U) transform(final_blocks.data() + 64U);
    Digest256 result{};
    for (std::size_t index = 0U; index < hash.size(); ++index) {
        result[index * 4U] = static_cast<std::uint8_t>(hash[index] >> 24U);
        result[index * 4U + 1U] =
            static_cast<std::uint8_t>(hash[index] >> 16U);
        result[index * 4U + 2U] =
            static_cast<std::uint8_t>(hash[index] >> 8U);
        result[index * 4U + 3U] = static_cast<std::uint8_t>(hash[index]);
    }
    return result;
}

[[nodiscard]] Digest256 genome_state_digest(
    const std::array<std::uint8_t, kGenomeBytes>& genome) noexcept {
    static constexpr char kHex[] = "0123456789abcdef";
    static constexpr std::string_view kPolicyPrefix = "{\"policy\":\"";
    static constexpr std::string_view kTraitPrefix = "\",\"trait\":\"";
    static constexpr std::string_view kSuffix = "\"}";
    std::array<std::uint8_t, 600U> canonical{};
    std::size_t cursor = 0U;
    const auto append_text = [&](const std::string_view value) noexcept {
        for (const char character : value) {
            canonical[cursor++] = static_cast<std::uint8_t>(character);
        }
    };
    const auto append_hex = [&](const std::uint8_t value) noexcept {
        canonical[cursor++] = static_cast<std::uint8_t>(kHex[value >> 4U]);
        canonical[cursor++] = static_cast<std::uint8_t>(kHex[value & 0x0fU]);
    };
    append_text(kPolicyPrefix);
    for (std::size_t index = 0U; index < 224U; ++index) {
        append_hex(genome[index]);
    }
    append_text(kTraitPrefix);
    for (std::size_t index = 224U; index < genome.size(); ++index) {
        append_hex(genome[index]);
    }
    append_text(kSuffix);
    if (cursor != canonical.size()) std::terminate();
    return sha256_bytes(canonical.data(), canonical.size());
}

[[nodiscard]] Digest256 tracked_genotype_state_digest(
    const std::array<std::uint8_t, kGenomeBytes>& genome,
    const SnapshotConfiguration& configuration) noexcept {
    // A low-level caller that has not supplied a tracked projection retains
    // the historical complete-genome identity.  Science bundles always bind
    // the tracked loci explicitly, so their direct attribution is keyed by
    // the exact requested diploid projection rather than by unbounded
    // untracked recombination states.
    if (configuration.policy_loci.empty()
        && configuration.trait_loci.empty()) {
        return genome_state_digest(genome);
    }
    std::array<std::uint8_t, 1024U> canonical{};
    std::size_t cursor = 0U;
    static constexpr std::array<std::uint8_t, 4U> kPrefix{{
        static_cast<std::uint8_t>('t'), static_cast<std::uint8_t>('g'),
        static_cast<std::uint8_t>('1'), 0U,
    }};
    const auto append_u16 = [&](const std::uint16_t value) noexcept {
        canonical[cursor++] = static_cast<std::uint8_t>(value & 0xffU);
        canonical[cursor++] = static_cast<std::uint8_t>(value >> 8U);
    };
    const auto append_i16 = [&](const std::int16_t value) noexcept {
        std::uint16_t bits = 0U;
        std::memcpy(&bits, &value, sizeof(bits));
        append_u16(bits);
    };
    for (const std::uint8_t value : kPrefix) canonical[cursor++] = value;
    append_u16(static_cast<std::uint16_t>(configuration.policy_loci.size()));
    for (const std::uint16_t locus : configuration.policy_loci) {
        append_u16(locus);
        append_i16(static_cast<std::int8_t>(genome[locus]));
        append_i16(static_cast<std::int8_t>(
            genome[kPolicyLociPerHaplotype + locus]));
    }
    append_u16(static_cast<std::uint16_t>(configuration.trait_loci.size()));
    for (const std::uint16_t locus : configuration.trait_loci) {
        append_u16(locus);
        append_i16(static_cast<std::int16_t>(
            genome[kPolicyValueCount + locus]));
        append_i16(static_cast<std::int16_t>(
            genome[kPolicyValueCount + kTraitLociPerHaplotype + locus]));
    }
    return sha256_bytes(canonical.data(), cursor);
}

[[nodiscard]] int hexadecimal_value(const char value) noexcept {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

[[nodiscard]] bool parse_sha256_id(
    const std::string_view value, Digest256* const output) noexcept {
    if (output == nullptr || value.size() != 71U
        || value.substr(0U, 7U) != "sha256:") {
        return false;
    }
    for (std::size_t index = 0U; index < output->size(); ++index) {
        const int high = hexadecimal_value(value[7U + index * 2U]);
        const int low = hexadecimal_value(value[8U + index * 2U]);
        if (high < 0 || low < 0) return false;
        (*output)[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

class JsonCursor {
public:
    JsonCursor(const char* const begin, const char* const end) noexcept
        : cursor_(begin), end_(end) {}

    [[nodiscard]] bool consume(const char expected) noexcept {
        if (cursor_ == end_ || *cursor_ != expected) return false;
        ++cursor_;
        return true;
    }

    [[nodiscard]] bool at_end() const noexcept { return cursor_ == end_; }

    [[nodiscard]] char peek() const noexcept {
        return cursor_ == end_ ? '\0' : *cursor_;
    }

    [[nodiscard]] bool parse_plain_string(
        std::string_view* const output) noexcept {
        if (output == nullptr || !consume('"')) return false;
        const char* const start = cursor_;
        while (cursor_ != end_ && *cursor_ != '"') {
            const unsigned char value = static_cast<unsigned char>(*cursor_);
            if (value < 0x20U || *cursor_ == '\\') return false;
            ++cursor_;
        }
        if (cursor_ == end_) return false;
        *output = std::string_view(
            start, static_cast<std::size_t>(cursor_ - start));
        ++cursor_;
        return true;
    }

    [[nodiscard]] bool parse_number_span(
        std::string_view* const output) noexcept {
        if (output == nullptr || cursor_ == end_) return false;
        const char* const start = cursor_;
        if (*cursor_ == '-') {
            ++cursor_;
            if (cursor_ == end_) return false;
        }
        if (*cursor_ == '0') {
            ++cursor_;
            if (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
                return false;
            }
        } else {
            if (*cursor_ < '1' || *cursor_ > '9') return false;
            do {
                ++cursor_;
            } while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9');
        }
        if (cursor_ != end_ && *cursor_ == '.') {
            ++cursor_;
            if (cursor_ == end_ || *cursor_ < '0' || *cursor_ > '9') {
                return false;
            }
            do {
                ++cursor_;
            } while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9');
        }
        if (cursor_ != end_ && (*cursor_ == 'e' || *cursor_ == 'E')) {
            ++cursor_;
            if (cursor_ != end_ && (*cursor_ == '+' || *cursor_ == '-')) {
                ++cursor_;
            }
            if (cursor_ == end_ || *cursor_ < '0' || *cursor_ > '9') {
                return false;
            }
            do {
                ++cursor_;
            } while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9');
        }
        *output = std::string_view(
            start, static_cast<std::size_t>(cursor_ - start));
        return true;
    }

    [[nodiscard]] bool parse_i64(std::int64_t* const output) noexcept {
        std::string_view token;
        if (!parse_number_span(&token)
            || token.find_first_of(".eE") != std::string_view::npos) {
            return false;
        }
        const auto parsed = std::from_chars(
            token.data(), token.data() + token.size(), *output);
        return parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size();
    }

    [[nodiscard]] bool parse_u64(std::uint64_t* const output) noexcept {
        std::string_view token;
        if (!parse_number_span(&token) || token.empty() || token.front() == '-'
            || token.find_first_of(".eE") != std::string_view::npos) {
            return false;
        }
        const auto parsed = std::from_chars(
            token.data(), token.data() + token.size(), *output);
        return parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size();
    }

    [[nodiscard]] bool skip_value(const unsigned int depth = 0U) noexcept {
        if (depth > 64U || cursor_ == end_) return false;
        if (*cursor_ == '"') return skip_string();
        if (*cursor_ == '{') {
            ++cursor_;
            if (consume('}')) return true;
            while (true) {
                if (!skip_string() || !consume(':') || !skip_value(depth + 1U)) {
                    return false;
                }
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (*cursor_ == '[') {
            ++cursor_;
            if (consume(']')) return true;
            while (true) {
                if (!skip_value(depth + 1U)) return false;
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (match_literal("true") || match_literal("false")
            || match_literal("null")) {
            return true;
        }
        std::string_view ignored;
        return parse_number_span(&ignored);
    }

private:
    [[nodiscard]] bool match_literal(const std::string_view value) noexcept {
        if (static_cast<std::size_t>(end_ - cursor_) < value.size()
            || std::memcmp(cursor_, value.data(), value.size()) != 0) {
            return false;
        }
        cursor_ += value.size();
        return true;
    }

    [[nodiscard]] bool skip_string() noexcept {
        if (!consume('"')) return false;
        while (cursor_ != end_) {
            const unsigned char value = static_cast<unsigned char>(*cursor_++);
            if (value == '"') return true;
            if (value < 0x20U) return false;
            if (value != '\\') continue;
            if (cursor_ == end_) return false;
            const char escaped = *cursor_++;
            if (escaped == '"' || escaped == '\\' || escaped == '/'
                || escaped == 'b' || escaped == 'f' || escaped == 'n'
                || escaped == 'r' || escaped == 't') {
                continue;
            }
            if (escaped != 'u' || end_ - cursor_ < 4) return false;
            for (int index = 0; index < 4; ++index) {
                if (hexadecimal_value(*cursor_++) < 0) return false;
            }
        }
        return false;
    }

    const char* cursor_;
    const char* end_;
};

enum class SnapshotRowKind {
    kInvalid,
    kWorldState,
    kActiveAgent,
    kGenomeCatalog,
};

struct SnapshotNumber {
    bool present = false;
    bool integer = false;
    std::int64_t integer_value = 0;
    double floating_value = 0.0;
};

struct ParsedSnapshotRow {
    SnapshotRowKind kind = SnapshotRowKind::kInvalid;
    std::string_view cell_id;
    std::int64_t absolute_tick = -1;
    std::uint64_t agent_id = 0U;
    Digest256 genome_digest{};
    bool genome_digest_present = false;
    std::array<std::int64_t, kMetricFields> metrics{};
    bool metrics_present = false;
    Digest256 policy_digest{};
    bool policy_digest_present = false;
    std::uint64_t parent_a_id = 0U;
    std::uint64_t parent_b_id = 0U;
    bool parent_a_present = false;
    bool parent_b_present = false;
    SnapshotNumber energy{};
    SnapshotNumber health{};
    SnapshotNumber injury{};
    std::array<std::uint8_t, kGenomeBytes> genome{};
    bool policy_genome_present = false;
    bool trait_genome_present = false;
};

[[nodiscard]] int metric_field_index(const std::string_view name) noexcept {
    if (name.size() == 14U && name.substr(0U, 7U) == "action_"
        && name[8U] == '_' && name.substr(9U) == "count"
        && name[7U] >= '0' && name[7U] <= '6') {
        return name[7U] - '0';
    }
    if (name == "alive_ticks") return 7;
    if (name == "reproduction_eligible_count") return 8;
    if (name == "mate_candidate_count") return 9;
    if (name == "birth_attempt_count") return 10;
    if (name == "successful_parent_count") return 11;
    if (name == "resource_uptake_total") return 12;
    if (name == "movement_cost_total") return 13;
    if (name == "movement_attempt_count") return 14;
    if (name == "movement_success_count") return 15;
    if (name == "health_damage_total") return 16;
    if (name == "energy_delta_total") return 17;
    if (name == "attack_attempt_count") return 18;
    if (name == "attack_hit_count") return 19;
    if (name == "births_as_parent_count") return 20;
    return -1;
}

[[nodiscard]] bool parse_metric_object(
    JsonCursor* const cursor,
    std::array<std::int64_t, kMetricFields>* const metrics) noexcept {
    if (cursor == nullptr || metrics == nullptr || !cursor->consume('{')) {
        return false;
    }
    std::uint32_t seen = 0U;
    if (cursor->consume('}')) return false;
    while (true) {
        std::string_view name;
        std::int64_t value = 0;
        if (!cursor->parse_plain_string(&name) || !cursor->consume(':')
            || !cursor->parse_i64(&value)) {
            return false;
        }
        const int field = metric_field_index(name);
        if (field < 0) return false;
        const auto index = static_cast<std::size_t>(field);
        const std::uint32_t bit = 1U << static_cast<unsigned int>(field);
        if ((seen & bit) != 0U || (field != 17 && (value < 0
            || static_cast<std::uint64_t>(value)
                > std::numeric_limits<std::uint32_t>::max()))) {
            return false;
        }
        (*metrics)[index] = value;
        seen |= bit;
        if (cursor->consume('}')) break;
        if (!cursor->consume(',')) return false;
    }
    return seen == ((1U << static_cast<unsigned int>(kMetricFields)) - 1U);
}

[[nodiscard]] bool parse_snapshot_number(
    JsonCursor* const cursor, SnapshotNumber* const output) noexcept {
    if (cursor == nullptr || output == nullptr) return false;
    const char first = cursor->peek();
    if (first != '-' && (first < '0' || first > '9')) {
        return cursor->skip_value();
    }
    std::string_view token;
    if (!cursor->parse_number_span(&token)) return false;
    output->present = true;
    if (token.find_first_of(".eE") == std::string_view::npos) {
        const auto parsed = std::from_chars(
            token.data(), token.data() + token.size(), output->integer_value);
        if (parsed.ec != std::errc{}
            || parsed.ptr != token.data() + token.size()) {
            return false;
        }
        output->integer = true;
        output->floating_value = static_cast<double>(output->integer_value);
        return true;
    }
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), output->floating_value,
        std::chars_format::general);
    return parsed.ec == std::errc{}
        && parsed.ptr == token.data() + token.size()
        && std::isfinite(output->floating_value);
}

template <std::size_t Count>
[[nodiscard]] bool parse_genome_array(
    JsonCursor* const cursor,
    const std::int64_t minimum,
    const std::int64_t maximum,
    std::array<std::int64_t, Count>* const values) noexcept {
    if (cursor == nullptr || values == nullptr || !cursor->consume('[')) {
        return false;
    }
    for (std::size_t index = 0U; index < Count; ++index) {
        std::int64_t value = 0;
        if (!cursor->parse_i64(&value) || value < minimum || value > maximum) {
            return false;
        }
        (*values)[index] = value;
        if (index + 1U < Count && !cursor->consume(',')) return false;
    }
    return cursor->consume(']');
}

[[nodiscard]] bool parse_snapshot_row(
    const char* const begin,
    const char* const end,
    ParsedSnapshotRow* const output) noexcept {
    if (begin == nullptr || end == nullptr || output == nullptr || begin >= end) {
        return false;
    }
    JsonCursor cursor(begin, end);
    if (!cursor.consume('{') || cursor.consume('}')) return false;
    bool tick_present = false;
    bool cell_present = false;
    std::string_view kind;
    std::array<std::int64_t, 224U> policy{};
    std::array<std::int64_t, 32U> trait{};
    while (true) {
        std::string_view key;
        if (!cursor.parse_plain_string(&key) || !cursor.consume(':')) return false;
        if (key == "absolute_tick") {
            if (tick_present || !cursor.parse_i64(&output->absolute_tick)) {
                return false;
            }
            tick_present = true;
        } else if (key == "agent_id") {
            if (output->agent_id != 0U || !cursor.parse_u64(&output->agent_id)) {
                return false;
            }
        } else if (key == "cell_id") {
            if (cell_present || !cursor.parse_plain_string(&output->cell_id)) {
                return false;
            }
            cell_present = true;
        } else if (key == "energy") {
            if (!parse_snapshot_number(&cursor, &output->energy)) return false;
        } else if (key == "genome_state_id") {
            std::string_view value;
            if (output->genome_digest_present
                || !cursor.parse_plain_string(&value)
                || !parse_sha256_id(value, &output->genome_digest)) {
                return false;
            }
            output->genome_digest_present = true;
        } else if (key == "health") {
            if (!parse_snapshot_number(&cursor, &output->health)) return false;
        } else if (key == "injury") {
            if (!parse_snapshot_number(&cursor, &output->injury)) return false;
        } else if (key == "lifetime_metrics") {
            if (output->metrics_present
                || !parse_metric_object(&cursor, &output->metrics)) {
                return false;
            }
            output->metrics_present = true;
        } else if (key == "parent_a_id") {
            if (output->parent_a_present
                || !cursor.parse_u64(&output->parent_a_id)) {
                return false;
            }
            output->parent_a_present = true;
        } else if (key == "parent_b_id") {
            if (output->parent_b_present
                || !cursor.parse_u64(&output->parent_b_id)) {
                return false;
            }
            output->parent_b_present = true;
        } else if (key == "policy") {
            if (output->policy_genome_present
                || !parse_genome_array(
                    &cursor, -128, 127, &policy)) {
                return false;
            }
            output->policy_genome_present = true;
            for (std::size_t index = 0U; index < policy.size(); ++index) {
                output->genome[index] = static_cast<std::uint8_t>(
                    static_cast<std::int8_t>(policy[index]));
            }
        } else if (key == "policy_id") {
            std::string_view value;
            if (output->policy_digest_present) return false;
            if (cursor.peek() == 'n') {
                if (!cursor.skip_value()) return false;
            } else if (!cursor.parse_plain_string(&value)
                       || !parse_sha256_id(value, &output->policy_digest)) {
                return false;
            } else {
                output->policy_digest_present = true;
            }
        } else if (key == "row_kind") {
            if (!kind.empty() || !cursor.parse_plain_string(&kind)) return false;
        } else if (key == "trait") {
            if (output->trait_genome_present
                || !parse_genome_array(
                    &cursor, -32768, 32767, &trait)) {
                return false;
            }
            output->trait_genome_present = true;
            for (std::size_t index = 0U; index < trait.size(); ++index) {
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::int16_t>(trait[index]));
                output->genome[224U + index * 2U] =
                    static_cast<std::uint8_t>(bits);
                output->genome[225U + index * 2U] =
                    static_cast<std::uint8_t>(bits >> 8U);
            }
        } else if (!cursor.skip_value()) {
            return false;
        }
        if (cursor.consume('}')) break;
        if (!cursor.consume(',')) return false;
    }
    if (!cursor.at_end() || !tick_present || !cell_present) return false;
    if (kind == "world_state") {
        output->kind = SnapshotRowKind::kWorldState;
        return true;
    }
    if (kind == "active_agent_state") {
        output->kind = SnapshotRowKind::kActiveAgent;
        return output->agent_id != 0U && output->genome_digest_present
            && output->metrics_present;
    }
    if (kind == "genome_catalog") {
        output->kind = SnapshotRowKind::kGenomeCatalog;
        return output->genome_digest_present && output->policy_genome_present
            && output->trait_genome_present;
    }
    return false;
}

struct DigestHash {
    [[nodiscard]] std::size_t operator()(
        const Digest256& value) const noexcept {
        std::uint64_t first = 0U;
        std::uint64_t second = 0U;
        std::memcpy(&first, value.data(), sizeof(first));
        std::memcpy(&second, value.data() + 16U, sizeof(second));
        return static_cast<std::size_t>(hash_agent_id(first ^ second));
    }
};

struct SnapshotActiveRow {
    std::uint64_t agent_id = 0U;
    Digest256 genome_digest{};
    std::array<std::int64_t, kMetricFields> metrics{};
    Digest256 policy_digest{};
    bool policy_digest_present = false;
    std::uint64_t parent_a_id = 0U;
    std::uint64_t parent_b_id = 0U;
    bool parent_ids_present = false;
    SnapshotNumber energy{};
    SnapshotNumber health{};
    SnapshotNumber injury{};
};

struct SnapshotNumericAccumulator {
    std::vector<SnapshotNumber> values;
    bool has_float = false;

    void add(const SnapshotNumber& value) {
        if (!value.present) return;
        values.push_back(value);
        has_float = has_float || !value.integer;
    }
};

using SnapshotHistogram = std::map<std::int32_t, std::uint64_t>;

struct SnapshotWorldProjection {
    std::uint32_t world_index = 0U;
    std::string world_state_row;
    std::uint64_t active_count = 0U;
    std::uint64_t catalog_count = 0U;
    std::array<__int128, kMetricFields> metric_totals{};
    std::uint64_t mutation_lineage_count = 0U;
    std::uint64_t maximum_lineage_depth = 0U;
    std::vector<std::array<SnapshotHistogram, 2U>> policy_histograms;
    std::vector<std::array<SnapshotHistogram, 2U>> trait_histograms;
    std::array<std::uint64_t, kMaxStrategies> active_by_strategy{};
    std::vector<std::uint64_t> initial_strategy_masks;
    SnapshotNumericAccumulator energy;
    SnapshotNumericAccumulator health;
    SnapshotNumericAccumulator injury;
};

struct SnapshotSegment {
    const char* begin = nullptr;
    const char* end = nullptr;
    std::int64_t first_row = 0;
    std::uint32_t world_index = 0U;
};

void append_u64_json(std::string* const output, const std::uint64_t value) {
    std::array<char, 32U> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec != std::errc{}) throw std::bad_alloc();
    output->append(buffer.data(), converted.ptr);
}

void append_i64_json(std::string* const output, const std::int64_t value) {
    std::array<char, 32U> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value);
    if (converted.ec != std::errc{}) throw std::bad_alloc();
    output->append(buffer.data(), converted.ptr);
}

void append_i128_json(std::string* const output, const __int128 value) {
    std::array<char, 64U> buffer{};
    char* cursor = buffer.data() + buffer.size();
    const bool negative = value < 0;
    unsigned __int128 magnitude = 0U;
    if (negative) {
        magnitude = static_cast<unsigned __int128>(-(value + 1)) + 1U;
    } else {
        magnitude = static_cast<unsigned __int128>(value);
    }
    do {
        const unsigned int digit = static_cast<unsigned int>(magnitude % 10U);
        *--cursor = static_cast<char>('0' + digit);
        magnitude /= 10U;
    } while (magnitude != 0U);
    if (negative) *--cursor = '-';
    output->append(cursor, buffer.data() + buffer.size());
}

void append_digest_string(
    std::string* const output, const Digest256& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    output->append("\"sha256:");
    for (const std::uint8_t value : digest) {
        output->push_back(kHex[value >> 4U]);
        output->push_back(kHex[value & 0x0fU]);
    }
    output->push_back('"');
}

constexpr std::array<std::string_view, kMetricFields> kMetricNames{{
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "action_4_count",
    "action_5_count",
    "action_6_count",
    "alive_ticks",
    "reproduction_eligible_count",
    "mate_candidate_count",
    "birth_attempt_count",
    "successful_parent_count",
    "resource_uptake_total",
    "movement_cost_total",
    "movement_attempt_count",
    "movement_success_count",
    "health_damage_total",
    "energy_delta_total",
    "attack_attempt_count",
    "attack_hit_count",
    "births_as_parent_count",
}};

void append_metric_totals_json(
    std::string* const output,
    const std::array<__int128, kMetricFields>& totals) {
    output->push_back('{');
    for (std::size_t index = 0U; index < kMetricFields; ++index) {
        if (index != 0U) output->push_back(',');
        output->push_back('"');
        output->append(kMetricNames[index]);
        output->append("\":");
        append_i128_json(output, totals[index]);
    }
    output->push_back('}');
}

void append_genotype_funnel_json(
    std::string* const output,
    const std::array<__int128, kMetricFields>& totals) {
    static constexpr std::array<std::size_t, 5U> kFunnelFields{{
        8U, 9U, 10U, 11U, 20U,
    }};
    output->push_back('{');
    for (std::size_t index = 0U; index < kFunnelFields.size(); ++index) {
        if (index != 0U) output->push_back(',');
        output->push_back('"');
        output->append(kMetricNames[kFunnelFields[index]]);
        output->append("\":");
        append_i128_json(output, totals[kFunnelFields[index]]);
    }
    output->push_back('}');
}

void append_genotype_counter_json(
    std::string* const output,
    const Digest256& digest,
    const GenotypeCounter& group) {
    output->append("{\"genome_state_id\":");
    append_digest_string(output, digest);
    output->append(",\"agent_count\":");
    append_u64_json(output, group.agent_count);
    output->append(",\"founder_count\":");
    append_u64_json(output, group.founder_count);
    output->append(",\"offspring_birth_count\":");
    append_u64_json(output, group.offspring_birth_count);
    output->append(",\"parent_birth_event_count\":");
    append_u64_json(output, group.parent_birth_event_count);
    output->append(",\"reproductive_agent_count\":");
    append_u64_json(output, group.reproductive_agent_count);
    output->append(",\"death_count\":");
    append_u64_json(output, group.death_count);
    output->append(",\"terminal_alive_count\":");
    append_u64_json(output, group.terminal_alive_count);
    output->append(",\"reproduction_funnel\":");
    append_genotype_funnel_json(output, group.metric_totals);
    output->append(",\"metric_totals\":");
    append_metric_totals_json(output, group.metric_totals);
    output->push_back('}');
}

[[nodiscard]] std::string render_genotype_result(
    const StrategyState& strategy) {
    GenotypeCounter totals{};
    for (const auto& entry : strategy.genotype_groups) {
        const GenotypeCounter& group = entry.second;
        totals.agent_count += group.agent_count;
        totals.founder_count += group.founder_count;
        totals.offspring_birth_count += group.offspring_birth_count;
        totals.parent_birth_event_count += group.parent_birth_event_count;
        totals.reproductive_agent_count += group.reproductive_agent_count;
        totals.death_count += group.death_count;
        totals.terminal_alive_count += group.terminal_alive_count;
        for (std::size_t field = 0U; field < kMetricFields; ++field) {
            totals.metric_totals[field] += group.metric_totals[field];
        }
    }
    std::string output;
    output.reserve(strategy.genotype_groups.size() * 512U + 512U);
    output.append(
        "{\"schema_version\":\"eiel_science_genotype_reproductive_attribution_v1\"");
    output.append(
        ",\"identity_mode\":\"exact_tracked_genotype_state_id_v1\"");
    output.append(",\"identity_scope\":{\"policy_loci\":[");
    for (std::size_t index = 0U;
         index < strategy.genotype_policy_loci.size(); ++index) {
        if (index != 0U) output.push_back(',');
        append_u64_json(
            &output,
            static_cast<std::uint64_t>(strategy.genotype_policy_loci[index]));
    }
    output.append("],\"trait_loci\":[");
    for (std::size_t index = 0U;
         index < strategy.genotype_trait_loci.size(); ++index) {
        if (index != 0U) output.push_back(',');
        append_u64_json(
            &output,
            static_cast<std::uint64_t>(strategy.genotype_trait_loci[index]));
    }
    output.append("]}");
    if (strategy.genotype_focal_filter_enabled) {
        output.append(",\"group_scope\":\"requested_focal_ids_v1\"");
    }
    output.append(",\"group_count\":");
    append_u64_json(
        &output, static_cast<std::uint64_t>(strategy.genotype_groups.size()));
    output.append(",\"groups\":[");
    std::size_t index = 0U;
    for (const auto& entry : strategy.genotype_groups) {
        if (index++ != 0U) output.push_back(',');
        append_genotype_counter_json(&output, entry.first, entry.second);
    }
    output.append("],\"descendant_birth_counts_by_parent_tracked_identity\":{");
    index = 0U;
    for (const auto& entry : strategy.descendant_birth_counts) {
        if (index++ != 0U) output.push_back(',');
        append_digest_string(&output, entry.first);
        output.push_back(':');
        append_u64_json(&output, entry.second);
    }
    output.append("},\"totals\":{");
    output.append("\"agent_count\":");
    append_u64_json(&output, totals.agent_count);
    output.append(",\"founder_count\":");
    append_u64_json(&output, totals.founder_count);
    output.append(",\"offspring_birth_count\":");
    append_u64_json(&output, totals.offspring_birth_count);
    output.append(",\"parent_birth_event_count\":");
    append_u64_json(&output, totals.parent_birth_event_count);
    output.append(",\"reproductive_agent_count\":");
    append_u64_json(&output, totals.reproductive_agent_count);
    output.append(",\"death_count\":");
    append_u64_json(&output, totals.death_count);
    output.append(",\"terminal_alive_count\":");
    append_u64_json(&output, totals.terminal_alive_count);
    output.append(",\"reproduction_funnel\":");
    append_genotype_funnel_json(&output, totals.metric_totals);
    output.append(",\"metric_totals\":");
    append_metric_totals_json(&output, totals.metric_totals);
    output.append("}}");
    return output;
}

void append_double_json(std::string* const output, const double value) {
    std::array<char, 64U> buffer{};
    const auto converted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (converted.ec != std::errc{}) throw std::bad_alloc();
    output->append(buffer.data(), converted.ptr);
}

void append_snapshot_number_json(
    std::string* const output, const SnapshotNumber& value) {
    if (value.integer) append_i64_json(output, value.integer_value);
    else append_double_json(output, value.floating_value);
}

void append_numeric_accumulator_json(
    std::string* const output, const SnapshotNumericAccumulator& values) {
    output->append("{\"count\":");
    append_u64_json(output, static_cast<std::uint64_t>(values.values.size()));
    if (values.values.empty()) {
        output->append(",\"integer_only\":true,\"maximum\":null,"
                       "\"minimum\":null,\"sum\":0}");
        return;
    }
    if (values.has_float) {
        output->append(",\"integer_only\":false,\"values\":[");
        for (std::size_t index = 0U; index < values.values.size(); ++index) {
            if (index != 0U) output->push_back(',');
            append_snapshot_number_json(output, values.values[index]);
        }
        output->append("]}");
        return;
    }
    __int128 sum = 0;
    std::int64_t minimum = values.values.front().integer_value;
    std::int64_t maximum = minimum;
    for (const SnapshotNumber& value : values.values) {
        sum += static_cast<__int128>(value.integer_value);
        minimum = std::min(minimum, value.integer_value);
        maximum = std::max(maximum, value.integer_value);
    }
    output->append(",\"integer_only\":true,\"maximum\":");
    append_i64_json(output, maximum);
    output->append(",\"minimum\":");
    append_i64_json(output, minimum);
    output->append(",\"sum\":");
    append_i128_json(output, sum);
    output->push_back('}');
}

void append_histograms_json(
    std::string* const output,
    const std::vector<std::uint16_t>& loci,
    const std::vector<std::array<SnapshotHistogram, 2U>>& histograms) {
    output->push_back('[');
    for (std::size_t locus_index = 0U; locus_index < loci.size(); ++locus_index) {
        if (locus_index != 0U) output->push_back(',');
        output->append("{\"alleles\":[");
        for (std::size_t allele = 0U; allele < 2U; ++allele) {
            if (allele != 0U) output->push_back(',');
            output->push_back('[');
            bool first = true;
            for (const auto& entry : histograms[locus_index][allele]) {
                if (!first) output->push_back(',');
                first = false;
                output->push_back('[');
                append_i64_json(output, entry.first);
                output->push_back(',');
                append_u64_json(output, entry.second);
                output->push_back(']');
            }
            output->push_back(']');
        }
        output->append("],\"locus\":");
        append_u64_json(output, loci[locus_index]);
        output->push_back('}');
    }
    output->push_back(']');
}

[[nodiscard]] std::string render_snapshot_world_projection(
    const SnapshotWorldProjection& projection,
    const SnapshotConfiguration& configuration) {
    std::string output;
    output.reserve(projection.world_state_row.size() + 8192U);
    output.append("{\"active_agent_count\":");
    append_u64_json(&output, projection.active_count);
    output.append(",\"active_by_strategy\":[");
    for (std::size_t index = 0U; index < configuration.strategy_count; ++index) {
        if (index != 0U) output.push_back(',');
        append_u64_json(&output, projection.active_by_strategy[index]);
    }
    output.append("],\"active_lifetime_metric_totals\":[");
    for (std::size_t index = 0U; index < kMetricFields; ++index) {
        if (index != 0U) output.push_back(',');
        append_i128_json(&output, projection.metric_totals[index]);
    }
    output.append("],\"energy\":");
    append_numeric_accumulator_json(&output, projection.energy);
    output.append(",\"genome_catalog_size\":");
    append_u64_json(&output, projection.catalog_count);
    output.append(",\"genome_reconstruction\":{\"active_agent_count\":");
    append_u64_json(&output, projection.active_count);
    output.append(",\"maximum_active_lineage_depth\":");
    append_u64_json(&output, projection.maximum_lineage_depth);
    output.append(",\"mutation_lineage_active_count\":");
    append_u64_json(&output, projection.mutation_lineage_count);
    output.append(",\"state_identity_check_count\":");
    append_u64_json(&output, projection.active_count);
    output.append(",\"tracked_policy_locus_histograms\":");
    append_histograms_json(
        &output, configuration.policy_loci, projection.policy_histograms);
    output.append(",\"tracked_trait_locus_histograms\":");
    append_histograms_json(
        &output, configuration.trait_loci, projection.trait_histograms);
    output.append("},\"health\":");
    append_numeric_accumulator_json(&output, projection.health);
    output.append(",\"initial_strategy_masks\":[");
    for (std::size_t index = 0U;
         index < projection.initial_strategy_masks.size(); ++index) {
        if (index != 0U) output.push_back(',');
        append_u64_json(&output, projection.initial_strategy_masks[index]);
    }
    output.append("],\"injury\":");
    append_numeric_accumulator_json(&output, projection.injury);
    output.append(",\"world_index\":");
    append_u64_json(&output, projection.world_index);
    output.append(",\"world_state_row\":");
    output.append(projection.world_state_row);
    output.push_back('}');
    return output;
}

[[nodiscard]] bool increment_u64(std::uint64_t* value) noexcept {
    if (*value == std::numeric_limits<std::uint64_t>::max()) return false;
    ++*value;
    return true;
}

[[nodiscard]] bool add_u64(
    std::uint64_t* target, const std::uint64_t value) noexcept {
    if (*target > std::numeric_limits<std::uint64_t>::max() - value) return false;
    *target += value;
    return true;
}

void store_i128(
    const __int128 value, eiel_native_compact_i128_v1* output) noexcept {
    const unsigned __int128 bits = static_cast<unsigned __int128>(value);
    output->low = static_cast<std::uint64_t>(bits);
    const std::uint64_t high_bits = static_cast<std::uint64_t>(bits >> 64U);
    std::memcpy(&output->high, &high_bits, sizeof(high_bits));
}

[[nodiscard]] ReducerState* mutable_state(void* handle) noexcept {
    return static_cast<ReducerState*>(handle);
}

[[nodiscard]] const ReducerState* immutable_state(const void* handle) noexcept {
    return static_cast<const ReducerState*>(handle);
}

void set_failure(
    ReducerState* state,
    const Status status,
    const std::uint32_t world_index,
    const std::int64_t tick,
    const std::int64_t row,
    const std::uint64_t agent_id,
    const std::uint64_t parent_a,
    const std::uint64_t parent_b) noexcept {
    std::lock_guard<std::mutex> lock(state->failure_mutex);
    if (state->failure.status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) return;
    state->failure = Failure{
        static_cast<std::int32_t>(status), world_index, tick, row,
        agent_id, parent_a, parent_b};
}

[[nodiscard]] bool strategy_mask_is_valid(
    const std::uint64_t mask, const std::uint8_t strategy_count) noexcept {
    if (mask == 0U || strategy_count == 0U || strategy_count > kMaxStrategies) {
        return false;
    }
    return strategy_count == 64U || (mask >> strategy_count) == 0U;
}

template <typename Function>
[[nodiscard]] bool for_each_strategy_bit(
    const std::uint64_t mask,
    const std::uint8_t strategy_count,
    Function&& function) noexcept {
    for (std::uint8_t strategy = 0U; strategy < strategy_count; ++strategy) {
        if (((mask >> strategy) & 1U) != 0U && !function(strategy)) return false;
    }
    return true;
}

[[nodiscard]] Status add_strategy_group(
    StrategyState* strategy, const std::uint64_t mask) noexcept {
    for (std::size_t index = 0U; index < strategy->offspring_group_count; ++index) {
        StrategyGroupCounter& entry = strategy->offspring_groups[index];
        if (entry.mask == mask) {
            return increment_u64(&entry.count)
                ? EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
                : EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
    }
    if (strategy->offspring_group_count == kMaxStrategyGroups) {
        return EIEL_NATIVE_COMPACT_REDUCER_CAPACITY_EXCEEDED_V1;
    }
    strategy->offspring_groups[strategy->offspring_group_count++] =
        StrategyGroupCounter{mask, 1U};
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] Status add_strategy_parent_pair(
    StrategyState* strategy,
    std::uint64_t first_mask,
    std::uint64_t second_mask) noexcept {
    if (second_mask < first_mask) std::swap(first_mask, second_mask);
    for (std::size_t index = 0U; index < strategy->parent_pair_count; ++index) {
        StrategyParentPairCounter& entry = strategy->parent_pairs[index];
        if (entry.first_mask == first_mask && entry.second_mask == second_mask) {
            return increment_u64(&entry.count)
                ? EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
                : EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
    }
    if (strategy->parent_pair_count == kMaxParentPairs) {
        return EIEL_NATIVE_COMPACT_REDUCER_CAPACITY_EXCEEDED_V1;
    }
    strategy->parent_pairs[strategy->parent_pair_count++] =
        StrategyParentPairCounter{first_mask, second_mask, 1U};
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] Status ensure_genotype_group(
    StrategyState* const strategy,
    const Digest256& digest,
    GenotypeCounter** const output) {
    if (strategy == nullptr || output == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    try {
        const auto result = strategy->genotype_groups.try_emplace(digest);
        *output = &result.first->second;
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    }
}

[[nodiscard]] Status add_genotype_agent(
    StrategyState* const strategy,
    const Digest256& digest,
    const bool founder) {
    if (strategy == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (!genotype_group_is_retained(*strategy, digest)) {
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    }
    GenotypeCounter* group = nullptr;
    const Status status = ensure_genotype_group(strategy, digest, &group);
    if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) return status;
    if (!increment_u64(&group->agent_count)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    if (founder) {
        if (!increment_u64(&group->founder_count)) {
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
    } else if (!increment_u64(&group->offspring_birth_count)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] Status add_genotype_parent_event(
    StrategyState* const strategy,
    const AgentState& parent,
    const bool first_reproductive_event) {
    if (!parent.genotype_digest_present) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    if (!genotype_group_is_retained(*strategy, parent.genotype_digest)) {
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    }
    GenotypeCounter* group = nullptr;
    const Status status = ensure_genotype_group(
        strategy, parent.genotype_digest, &group);
    if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) return status;
    if (!increment_u64(&group->parent_birth_event_count)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    if (first_reproductive_event
        && !increment_u64(&group->reproductive_agent_count)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] Status add_descendant_birth_count(
    StrategyState* const strategy, const Digest256& digest) {
    if (strategy == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (!genotype_group_is_retained(*strategy, digest)) {
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    }
    try {
        auto result = strategy->descendant_birth_counts.try_emplace(digest, 0U);
        if (!increment_u64(&result.first->second)) {
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    }
}

[[nodiscard]] Status close_genotype_agent(
    StrategyState* const strategy,
    const AgentState& agent,
    const std::array<std::int64_t, kMetricFields>& metrics,
    const bool terminal_alive) {
    if (!agent.genotype_digest_present) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    if (!genotype_group_is_retained(*strategy, agent.genotype_digest)) {
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    }
    const auto iterator = strategy->genotype_groups.find(agent.genotype_digest);
    if (iterator == strategy->genotype_groups.end()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    GenotypeCounter& group = iterator->second;
    for (std::size_t field = 0U; field < kMetricFields; ++field) {
        group.metric_totals[field] += static_cast<__int128>(metrics[field]);
    }
    if (terminal_alive) {
        if (!increment_u64(&group.terminal_alive_count)) {
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
    } else if (!increment_u64(&group.death_count)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] Status close_strategy_agent(
    StrategyState* strategy,
    const AgentState& agent,
    const std::array<std::int64_t, kMetricFields>& metrics,
    const std::int64_t death_tick,
    const bool terminal_alive) noexcept {
    const bool counters_ok = for_each_strategy_bit(
        agent.strategy_mask, strategy->strategy_count,
        [&](const std::uint8_t strategy_index) noexcept {
            const std::size_t index = strategy_index;
            for (std::size_t field = 0U; field < kMetricFields; ++field) {
                strategy->metric_totals[index * kMetricFields + field] +=
                    static_cast<__int128>(metrics[field]);
            }
            if (!terminal_alive && !increment_u64(&strategy->death_events[index])) {
                return false;
            }
            if (agent.birth_tick <= strategy->early_life_tick
                && (terminal_alive || death_tick > strategy->early_life_tick)
                && !increment_u64(&strategy->persistence_to_early[index])) {
                return false;
            }
            if (!agent.founder) {
                for (std::size_t threshold = 0U;
                     threshold < strategy->threshold_count; ++threshold) {
                    if ((terminal_alive
                         || death_tick > strategy->thresholds[threshold])
                        && !increment_u64(
                            &strategy->offspring_survived[
                                index * kMaxThresholds + threshold])) {
                        return false;
                    }
                }
            }
            if (terminal_alive
                && !increment_u64(&strategy->terminal_alive[index])) {
                return false;
            }
            return true;
        });
    return counters_ok
        ? EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
        : EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
}

[[nodiscard]] std::uint64_t snapshot_strategy_mask(
    const SnapshotConfiguration& configuration,
    const Digest256& policy_digest) noexcept {
    for (const SnapshotPolicyMapping& mapping : configuration.strategy_policies) {
        if (mapping.policy_digest == policy_digest) return mapping.strategy_mask;
    }
    return 0U;
}

[[nodiscard]] Status initialize_snapshot_strategy(
    WorldState* const world,
    const SnapshotConfiguration& configuration,
    const std::vector<SnapshotActiveRow>& active,
    SnapshotWorldProjection* const projection) noexcept {
    if (configuration.strategy_count == 0U) return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    StrategyState strategy{};
    strategy.enabled = true;
    strategy.strategy_count = configuration.strategy_count;
    strategy.threshold_count = static_cast<std::uint8_t>(
        configuration.thresholds.size());
    strategy.early_life_tick = configuration.early_life_tick;
    strategy.genotype_policy_loci = configuration.policy_loci;
    strategy.genotype_trait_loci = configuration.trait_loci;
    strategy.genotype_focal_filter_enabled =
        configuration.genotype_focal_filter_enabled;
    strategy.genotype_focal_ids = configuration.genotype_focal_ids;
    strategy.total_agent_count = static_cast<std::uint64_t>(active.size());
    strategy.founder_count = static_cast<std::uint64_t>(active.size());
    strategy.snapshot_epoch = world->snapshot_epoch;
    for (std::size_t index = 0U; index < configuration.thresholds.size(); ++index) {
        strategy.thresholds[index] = configuration.thresholds[index];
    }
    projection->initial_strategy_masks.reserve(active.size());
    for (const SnapshotActiveRow& row : active) {
        if ((!configuration.genotype_attribution_only
             && !row.policy_digest_present)
            || !row.parent_ids_present || row.parent_a_id != 0U
            || row.parent_b_id != 0U) {
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        const std::uint64_t mask = configuration.genotype_attribution_only
            ? 1U
            : snapshot_strategy_mask(configuration, row.policy_digest);
        if (!strategy_mask_is_valid(mask, configuration.strategy_count)) {
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        const std::uint32_t slot = world->active_map.find(row.agent_id);
        if (slot == kMissingSlot) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        AgentState& agent = world->agents[slot];
        agent.strategy_mask = mask;
        agent.birth_tick = world->next_absolute_tick;
        agent.founder = true;
        agent.genotype_digest = tracked_genotype_state_digest(
            agent.genome, configuration);
        agent.genotype_digest_present = true;
        const Status genotype_status = add_genotype_agent(
            &strategy, agent.genotype_digest, true);
        if (genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            return genotype_status;
        }
        const bool counted = for_each_strategy_bit(
            mask, configuration.strategy_count,
            [&](const std::uint8_t strategy_index) noexcept {
                return increment_u64(
                    &strategy.agent_counts[
                        static_cast<std::size_t>(strategy_index)]);
            });
        if (!counted) return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        projection->initial_strategy_masks.push_back(mask);
    }
    world->strategy = strategy;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] std::int16_t read_i16_le(const std::uint8_t* const source) noexcept {
    const std::uint16_t bits = static_cast<std::uint16_t>(source[0])
        | static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(source[1]) << 8U);
    std::int16_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] Status process_snapshot_segment(
    ReducerState* const state,
    const SnapshotSegment& segment,
    const std::int64_t absolute_tick,
    SnapshotWorldProjection* const projection) {
    WorldState& world = state->worlds[segment.world_index];
    const SnapshotConfiguration& configuration = world.snapshot;
    const auto fail = [&](const Status status, const std::int64_t row,
                          const std::uint64_t agent_id = 0U) noexcept {
        set_failure(
            state, status, segment.world_index, absolute_tick, row,
            agent_id, 0U, 0U);
        return status;
    };
    if (!configuration.configured) {
        return fail(
            EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1, segment.first_row);
    }
    projection->world_index = segment.world_index;
    projection->policy_histograms.resize(configuration.policy_loci.size());
    projection->trait_histograms.resize(configuration.trait_loci.size());
    std::vector<SnapshotActiveRow> active;
    active.reserve(world.resident_capacity);
    std::unordered_set<std::uint64_t> active_ids;
    active_ids.reserve(static_cast<std::size_t>(world.resident_capacity) * 2U);
    std::unordered_map<
        Digest256, std::array<std::uint8_t, kGenomeBytes>, DigestHash> catalogs;
    catalogs.reserve(static_cast<std::size_t>(world.resident_capacity) * 2U);
    bool world_state_seen = false;
    const char* line_begin = segment.begin;
    std::int64_t row_index = segment.first_row;
    while (line_begin < segment.end) {
        const void* const newline_raw = std::memchr(
            line_begin, '\n', static_cast<std::size_t>(segment.end - line_begin));
        const char* const line_end = newline_raw == nullptr
            ? segment.end
            : static_cast<const char*>(newline_raw);
        if (line_begin == line_end) {
            return fail(EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1, row_index);
        }
        ParsedSnapshotRow row;
        if (!parse_snapshot_row(line_begin, line_end, &row)
            || row.absolute_tick != absolute_tick
            || row.cell_id != configuration.cell_id) {
            return fail(
                EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                row_index, row.agent_id);
        }
        if (row.kind == SnapshotRowKind::kWorldState) {
            if (world_state_seen || row_index != segment.first_row) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1, row_index);
            }
            world_state_seen = true;
            projection->world_state_row.assign(line_begin, line_end);
        } else if (row.kind == SnapshotRowKind::kActiveAgent) {
            if (!world_state_seen || active_ids.find(row.agent_id) != active_ids.end()) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1,
                    row_index, row.agent_id);
            }
            active_ids.insert(row.agent_id);
            SnapshotActiveRow active_row;
            active_row.agent_id = row.agent_id;
            active_row.genome_digest = row.genome_digest;
            active_row.metrics = row.metrics;
            active_row.policy_digest = row.policy_digest;
            active_row.policy_digest_present = row.policy_digest_present;
            active_row.parent_a_id = row.parent_a_id;
            active_row.parent_b_id = row.parent_b_id;
            active_row.parent_ids_present =
                row.parent_a_present && row.parent_b_present;
            active_row.energy = row.energy;
            active_row.health = row.health;
            active_row.injury = row.injury;
            active.push_back(active_row);
            for (std::size_t field = 0U; field < kMetricFields; ++field) {
                projection->metric_totals[field] +=
                    static_cast<__int128>(row.metrics[field]);
            }
            projection->energy.add(row.energy);
            projection->health.add(row.health);
            projection->injury.add(row.injury);
        } else if (row.kind == SnapshotRowKind::kGenomeCatalog) {
            if (!world_state_seen || genome_state_digest(row.genome)
                    != row.genome_digest
                || catalogs.find(row.genome_digest) != catalogs.end()) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1, row_index);
            }
            catalogs.emplace(row.genome_digest, row.genome);
        } else {
            return fail(EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1, row_index);
        }
        ++row_index;
        line_begin = newline_raw == nullptr ? segment.end : line_end + 1;
    }
    if (!world_state_seen
        || active.size() > static_cast<std::size_t>(world.resident_capacity)) {
        return fail(
            EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1, segment.first_row);
    }
    projection->active_count = static_cast<std::uint64_t>(active.size());
    projection->catalog_count = static_cast<std::uint64_t>(catalogs.size());
    const bool initializing = !world.initialized;
    if (initializing) {
        if (absolute_tick != world.next_absolute_tick || catalogs.empty()) {
            return fail(
                EIEL_NATIVE_COMPACT_REDUCER_NONCONTIGUOUS_TICK_V1,
                segment.first_row);
        }
        for (const SnapshotActiveRow& row : active) {
            const auto genome = catalogs.find(row.genome_digest);
            if (genome == catalogs.end() || world.free_slots.empty()) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                    segment.first_row, row.agent_id);
            }
            const std::uint32_t slot = world.free_slots.back();
            world.free_slots.pop_back();
            AgentState& agent = world.agents[slot];
            agent = AgentState{};
            agent.agent_id = row.agent_id;
            agent.genome = genome->second;
            agent.genome_digest = row.genome_digest;
            agent.genome_digest_present = true;
            if (!world.active_map.insert(row.agent_id, slot)) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1,
                    segment.first_row, row.agent_id);
            }
            world.maximum_seen_agent_id = std::max(
                world.maximum_seen_agent_id, row.agent_id);
        }
        world.initialized = true;
    } else if (world.next_absolute_tick != absolute_tick
               || active.size()
                    != static_cast<std::size_t>(world.resident_capacity)
                        - world.free_slots.size()) {
        return fail(
            EIEL_NATIVE_COMPACT_REDUCER_NONCONTIGUOUS_TICK_V1,
            segment.first_row);
    }
    if (world.snapshot_epoch == std::numeric_limits<std::uint64_t>::max()) {
        return fail(
            EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1, segment.first_row);
    }
    const std::uint64_t snapshot_epoch = ++world.snapshot_epoch;
    if (initializing) {
        const Status strategy_status = initialize_snapshot_strategy(
            &world, configuration, active, projection);
        if (strategy_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            return fail(strategy_status, segment.first_row);
        }
    } else if ((configuration.strategy_count == 0U) != !world.strategy.enabled) {
        return fail(
            EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1, segment.first_row);
    }
    for (const SnapshotActiveRow& row : active) {
        const std::uint32_t slot = world.active_map.find(row.agent_id);
        if (slot == kMissingSlot) {
            return fail(
                EIEL_NATIVE_COMPACT_REDUCER_INACTIVE_AGENT_V1,
                segment.first_row, row.agent_id);
        }
        AgentState& agent = world.agents[slot];
        if (agent.strategy_snapshot_epoch == snapshot_epoch) {
            return fail(
                EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1,
                segment.first_row, row.agent_id);
        }
        if (!agent.genome_digest_present) {
            agent.genome_digest = genome_state_digest(agent.genome);
            agent.genome_digest_present = true;
        }
        if (agent.genome_digest != row.genome_digest) {
            return fail(
                EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                segment.first_row, row.agent_id);
        }
        agent.latest_metrics = row.metrics;
        agent.latest_metrics_present = true;
        agent.strategy_snapshot_epoch = snapshot_epoch;
        projection->mutation_lineage_count +=
            static_cast<std::uint64_t>(agent.mutation_lineage);
        projection->maximum_lineage_depth = std::max(
            projection->maximum_lineage_depth, agent.generation);
        for (std::size_t locus_index = 0U;
             locus_index < configuration.policy_loci.size(); ++locus_index) {
            const std::size_t locus = configuration.policy_loci[locus_index];
            for (std::size_t allele = 0U; allele < 2U; ++allele) {
                const std::size_t genome_index = locus + allele * 112U;
                std::int8_t value = 0;
                std::memcpy(&value, &agent.genome[genome_index], sizeof(value));
                ++projection->policy_histograms[locus_index][allele][value];
            }
        }
        for (std::size_t locus_index = 0U;
             locus_index < configuration.trait_loci.size(); ++locus_index) {
            const std::size_t locus = configuration.trait_loci[locus_index];
            for (std::size_t allele = 0U; allele < 2U; ++allele) {
                const std::size_t genome_index =
                    224U + (locus + allele * 16U) * 2U;
                const std::int16_t value = read_i16_le(
                    agent.genome.data() + genome_index);
                ++projection->trait_histograms[locus_index][allele][value];
            }
        }
        if (world.strategy.enabled) {
            if (!strategy_mask_is_valid(
                    agent.strategy_mask, world.strategy.strategy_count)) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                    segment.first_row, row.agent_id);
            }
            const bool counted = for_each_strategy_bit(
                agent.strategy_mask, world.strategy.strategy_count,
                [&](const std::uint8_t strategy_index) noexcept {
                    return increment_u64(
                        &projection->active_by_strategy[
                            static_cast<std::size_t>(strategy_index)]);
                });
            if (!counted) {
                return fail(
                    EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                    segment.first_row, row.agent_id);
            }
        }
    }
    for (const AgentState& agent : world.agents) {
        if (agent.agent_id != 0U
            && agent.strategy_snapshot_epoch != snapshot_epoch) {
            return fail(
                EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                segment.first_row, agent.agent_id);
        }
    }
    if (world.strategy.enabled) world.strategy.snapshot_epoch = snapshot_epoch;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

[[nodiscard]] Status process_world(
    ReducerState* state,
    const std::uint32_t world_index,
    const std::int64_t start_tick,
    const std::int64_t end_tick,
    const std::uint8_t* events,
    const std::int64_t first,
    const std::int64_t stop,
    const std::uint8_t* metrics,
    const std::int64_t death_base,
    const std::uint8_t* provenance,
    const std::int64_t birth_base) noexcept {
    WorldState& world = state->worlds[world_index];
    std::int64_t birth_used = 0;
    std::int64_t death_used = 0;
    for (std::int64_t row = first; row < stop; ++row) {
        const std::uint8_t* event = events
            + static_cast<std::size_t>(row) * kEventRowBytes;
        const std::uint64_t agent_id = read_u64_le(event);
        const std::uint64_t parent_a_id = read_u64_le(event + 8U);
        const std::uint64_t parent_b_id = read_u64_le(event + 16U);
        const std::uint64_t packed = read_u64_le(event + 24U);
        const std::uint64_t event_kind = (packed >> 6U) & 0x3U;
        const std::int64_t tick = start_tick
            + static_cast<std::int64_t>(packed & 0x3fU);
        if (event_kind == 1U) {
            const std::uint32_t slot = world.active_map.find(agent_id);
            if (slot == kMissingSlot) {
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INACTIVE_AGENT_V1,
                    world_index, tick, row, agent_id, parent_a_id, parent_b_id);
                return EIEL_NATIVE_COMPACT_REDUCER_INACTIVE_AGENT_V1;
            }
            const std::uint8_t* cursor = metrics
                + static_cast<std::size_t>(death_base + death_used)
                    * kMetricRowBytes;
            std::array<std::int64_t, kMetricFields> death_metrics{};
            for (std::size_t field = 0U; field < kMetricFields; ++field) {
                if (field == 17U) {
                    death_metrics[field] = read_i64_le(cursor);
                    cursor += 8U;
                } else {
                    death_metrics[field] =
                        static_cast<std::int64_t>(read_u32_le(cursor));
                    cursor += 4U;
                }
                world.metric_totals[field] +=
                    static_cast<__int128>(death_metrics[field]);
            }
            AgentState& agent = world.agents[slot];
            if (world.strategy.enabled) {
                const Status strategy_status = close_strategy_agent(
                    &world.strategy, agent, death_metrics, tick, false);
                Status genotype_status = strategy_status;
                if (genotype_status == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                    genotype_status = close_genotype_agent(
                        &world.strategy, agent, death_metrics, false);
                }
                if (genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
                    || !increment_u64(&world.strategy.death_count)) {
                    const Status failure_status =
                        genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
                        ? genotype_status
                        : EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
                    set_failure(
                        state, failure_status, world_index, tick, row,
                        agent_id, parent_a_id, parent_b_id);
                    return failure_status;
                }
                if (world.strategy.first_death_tick < 0) {
                    world.strategy.first_death_tick = tick;
                }
            }
            agent.agent_id = 0U;
            if (!world.active_map.erase(agent_id)) {
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                    world_index, tick, row, agent_id, parent_a_id, parent_b_id);
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
            }
            world.free_slots.push_back(slot);
            if (world.strategy.enabled
                && world.strategy.extinction_tick < 0
                && world.free_slots.size() == world.resident_capacity) {
                world.strategy.extinction_tick = tick;
            }
            if (!increment_u64(&world.death_count)) {
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                    world_index, tick, row, agent_id, parent_a_id, parent_b_id);
                return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
            }
            ++death_used;
            continue;
        }
        if (event_kind != 2U || agent_id == 0U || parent_a_id == 0U
            || parent_b_id == 0U) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1;
        }
        if (agent_id <= world.maximum_seen_agent_id) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1;
        }
        const std::uint32_t parent_a_slot = world.active_map.find(parent_a_id);
        const std::uint32_t parent_b_slot = world.active_map.find(parent_b_id);
        if (parent_a_slot == kMissingSlot || parent_b_slot == kMissingSlot) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INACTIVE_AGENT_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_INACTIVE_AGENT_V1;
        }
        if (world.free_slots.empty()) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_CAPACITY_EXCEEDED_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_CAPACITY_EXCEEDED_V1;
        }
        AgentState& parent_a = world.agents[parent_a_slot];
        AgentState& parent_b = world.agents[parent_b_slot];
        const std::uint32_t child_slot = world.free_slots.back();
        AgentState& child = world.agents[child_slot];
        const std::uint8_t* birth_provenance = provenance
            + static_cast<std::size_t>(birth_base + birth_used) * kProvenanceBytes;
        const std::uint64_t mutation_origin_count_before =
            world.birth_receipt.mutation_origin_count;
        const Status reconstruction_status =
            eiel_compact_reducer_internal_v1::reconstruct_birth(
                parent_a.genome.data(), parent_b.genome.data(), birth_provenance,
                world.mutation_step_at(tick), child.genome.data(),
                &world.birth_receipt);
        if (reconstruction_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            set_failure(
                state, reconstruction_status, world_index, tick, row,
                agent_id, parent_a_id, parent_b_id);
            return reconstruction_status;
        }
        child.genome_digest = genome_state_digest(child.genome);
        child.genome_digest_present = true;
        child.genotype_digest = tracked_genotype_state_digest(
            child.genome, world.snapshot);
        child.genotype_digest_present = true;
        const bool parent_born = parent_a.born || parent_b.born;
        const bool parent_mutation_lineage =
            parent_a.mutation_lineage || parent_b.mutation_lineage;
        const std::uint64_t generation =
            std::max(parent_a.generation, parent_b.generation) + 1U;
        if (generation == 0U) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
        if (parent_born
            && !increment_u64(&world.carrier_reproduction_birth_count)) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
        if (parent_mutation_lineage
            && !increment_u64(
                &world.mutation_carrier_reproduction_birth_count)) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
        const auto mark_reproduced = [&world](AgentState& parent) noexcept {
            if (!parent.born || parent.reproduced) return true;
            parent.reproduced = true;
            if (!increment_u64(&world.unique_born_agents_reproduced_count)) {
                return false;
            }
            if (parent.mutation_lineage
                && !increment_u64(
                    &world.unique_mutation_carriers_reproduced_count)) {
                return false;
            }
            return true;
        };
        if (!mark_reproduced(parent_a) || !mark_reproduced(parent_b)) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
        std::uint64_t child_strategy_mask = 0U;
        if (world.strategy.enabled) {
            if (!strategy_mask_is_valid(
                    parent_a.strategy_mask, world.strategy.strategy_count)
                || !strategy_mask_is_valid(
                    parent_b.strategy_mask, world.strategy.strategy_count)) {
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                    world_index, tick, row, agent_id, parent_a_id, parent_b_id);
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
            }
            child_strategy_mask =
                parent_a.strategy_mask | parent_b.strategy_mask;
            Status strategy_status = add_strategy_group(
                &world.strategy, child_strategy_mask);
            if (strategy_status == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                strategy_status = add_strategy_parent_pair(
                    &world.strategy, parent_a.strategy_mask,
                    parent_b.strategy_mask);
            }
            bool counters_ok =
                strategy_status == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
                && increment_u64(&world.strategy.total_agent_count)
                && increment_u64(&world.strategy.birth_count);
            if (counters_ok && parent_a.strategy_mask != parent_b.strategy_mask) {
                counters_ok = increment_u64(
                    &world.strategy.mixed_parent_birth_count);
            }
            if (counters_ok) {
                counters_ok = for_each_strategy_bit(
                    child_strategy_mask, world.strategy.strategy_count,
                    [&](const std::uint8_t strategy_index) noexcept {
                        const std::size_t index = strategy_index;
                        if (!increment_u64(&world.strategy.agent_counts[index])
                            || !increment_u64(
                                &world.strategy.offspring_born[index])) {
                            return false;
                        }
                        world.strategy.max_depth_by_strategy[index] = std::max(
                            world.strategy.max_depth_by_strategy[index],
                            generation);
                        return true;
                    });
            }
            const auto count_parent = [&world](
                                          const std::uint64_t mask) noexcept {
                return for_each_strategy_bit(
                    mask, world.strategy.strategy_count,
                    [&world](const std::uint8_t strategy_index) noexcept {
                        return increment_u64(
                            &world.strategy.parent_birth_events[
                                static_cast<std::size_t>(strategy_index)]);
                    });
            };
            if (counters_ok) counters_ok = count_parent(parent_a.strategy_mask);
            if (counters_ok) counters_ok = count_parent(parent_b.strategy_mask);
            if (!counters_ok) {
                if (strategy_status == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                    strategy_status =
                        EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
                }
                set_failure(
                    state, strategy_status, world_index, tick, row,
                    agent_id, parent_a_id, parent_b_id);
                return strategy_status;
            }
            Status genotype_status = add_genotype_agent(
                &world.strategy, child.genotype_digest, false);
            if (genotype_status == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                const bool parent_a_first =
                    !parent_a.genotype_reproductive_event_recorded;
                genotype_status = add_genotype_parent_event(
                    &world.strategy, parent_a, parent_a_first);
                if (genotype_status
                    == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                    parent_a.genotype_reproductive_event_recorded = true;
                    const bool parent_b_first =
                        !parent_b.genotype_reproductive_event_recorded;
                    genotype_status = add_genotype_parent_event(
                        &world.strategy, parent_b, parent_b_first);
                    if (genotype_status
                        == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                        parent_b.genotype_reproductive_event_recorded = true;
                    }
                }
            }
            if (genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                set_failure(
                    state, genotype_status, world_index, tick, row,
                    agent_id, parent_a_id, parent_b_id);
                return genotype_status;
            }
            genotype_status = add_descendant_birth_count(
                &world.strategy, parent_a.genotype_digest);
            if (genotype_status == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1
                && parent_b.genotype_digest != parent_a.genotype_digest) {
                genotype_status = add_descendant_birth_count(
                    &world.strategy, parent_b.genotype_digest);
            }
            if (genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                set_failure(
                    state, genotype_status, world_index, tick, row,
                    agent_id, parent_a_id, parent_b_id);
                return genotype_status;
            }
            world.strategy.maximum_lineage_depth = std::max(
                world.strategy.maximum_lineage_depth, generation);
            if (world.strategy.first_birth_tick < 0) {
                world.strategy.first_birth_tick = tick;
            }
        }
        child.agent_id = agent_id;
        child.generation = generation;
        child.strategy_mask = child_strategy_mask;
        child.birth_tick = tick;
        // The cumulative receipt cannot identify this row after prior births.
        // Inspecting its packed mutation codes gives the exact row-local fact.
        bool row_has_mutation = false;
        for (std::size_t byte = 0U; byte < 64U; ++byte) {
            const std::uint8_t value = birth_provenance[byte];
            if ((value & 0x3U) != 0U || ((value >> 2U) & 0x3U) != 0U
                || ((value >> 4U) & 0x3U) != 0U
                || ((value >> 6U) & 0x3U) != 0U) {
                row_has_mutation = true;
                break;
            }
        }
        child.mutation_lineage = parent_mutation_lineage || row_has_mutation;
        if (world.hereditary_birth_sink != nullptr) {
            if (world.birth_receipt.mutation_origin_count
                    < mutation_origin_count_before
                || world.birth_receipt.mutation_origin_count
                    - mutation_origin_count_before
                    > std::numeric_limits<std::uint32_t>::max()
                || !parent_a.genome_digest_present
                || !parent_b.genome_digest_present
                || !parent_a.genotype_digest_present
                || !parent_b.genotype_digest_present) {
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                    world_index, tick, row, agent_id, parent_a_id, parent_b_id);
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
            }
            const std::uint32_t mutation_origin_count = static_cast<std::uint32_t>(
                world.birth_receipt.mutation_origin_count
                - mutation_origin_count_before);
            const std::uint32_t recombination_pattern = read_u32_le(
                birth_provenance + 64U);
            if (!world.hereditary_birth_sink->write_record(
                    0U, tick, agent_id,
                    parent_a_id, parent_b_id, parent_a.genome_digest,
                    parent_b.genome_digest, child.genome_digest,
                    parent_a.genotype_digest, parent_b.genotype_digest,
                    child.genotype_digest, generation, mutation_origin_count,
                    recombination_pattern, birth_provenance)) {
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                    world_index, tick, row, agent_id, parent_a_id, parent_b_id);
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
            }
        }
        child.born = true;
        child.reproduced = false;
        child.founder = false;
        child.latest_metrics_present = false;
        child.genotype_reproductive_event_recorded = false;
        child.strategy_snapshot_epoch = 0U;
        if (!world.active_map.insert(agent_id, child_slot)) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        world.maximum_seen_agent_id = agent_id;
        world.free_slots.pop_back();
        if (!increment_u64(&world.birth_count)
            || !increment_u64(&world.recombination_birth_count)) {
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                world_index, tick, row, agent_id, parent_a_id, parent_b_id);
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
        world.maximum_lineage_depth =
            std::max(world.maximum_lineage_depth, generation);
        ++birth_used;
    }
    world.active_map.rebuild(world.agents);
    const std::uint64_t row_count = static_cast<std::uint64_t>(stop - first);
    if (!increment_u64(&world.dispatch_count)
        || !add_u64(&world.event_count, row_count)
        || !add_u64(
            &world.dispatch_tick_count,
            static_cast<std::uint64_t>(end_tick - start_tick))) {
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
            world_index, end_tick, stop, 0U, 0U, 0U);
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    world.next_absolute_tick = end_tick;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

}  // namespace

extern "C" {

std::int32_t eiel_native_compact_state_create_v1(
    const std::int64_t world_count,
    const std::uint32_t* const resident_capacities,
    const std::int64_t* const start_next_absolute_ticks,
    const std::int64_t* const mutation_segment_offsets,
    const std::int64_t* const mutation_segment_start_ticks,
    const std::int32_t* const mutation_segment_steps,
    const std::int64_t mutation_segment_count,
    const std::int32_t max_worker_threads,
    void** const output_handle) {
    if (output_handle == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    *output_handle = nullptr;
    if (world_count < 1
        || world_count > std::numeric_limits<std::int32_t>::max()
        || mutation_segment_count < world_count
        || max_worker_threads < 1) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (resident_capacities == nullptr || start_next_absolute_ticks == nullptr
        || mutation_segment_offsets == nullptr
        || mutation_segment_start_ticks == nullptr
        || mutation_segment_steps == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (mutation_segment_offsets[0] != 0
        || mutation_segment_offsets[world_count] != mutation_segment_count) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    try {
        std::unique_ptr<ReducerState> state(new ReducerState());
        state->worlds.reserve(static_cast<std::size_t>(world_count));
        state->max_worker_threads = max_worker_threads;
        for (std::int64_t world = 0; world < world_count; ++world) {
            const std::int64_t first = mutation_segment_offsets[world];
            const std::int64_t stop = mutation_segment_offsets[world + 1];
            if (resident_capacities[world] < 1U
                || start_next_absolute_ticks[world] < 0
                || first < 0 || stop <= first || stop > mutation_segment_count) {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
            }
            std::vector<std::int64_t> starts;
            std::vector<std::int32_t> steps;
            starts.reserve(static_cast<std::size_t>(stop - first));
            steps.reserve(static_cast<std::size_t>(stop - first));
            for (std::int64_t segment = first; segment < stop; ++segment) {
                if (mutation_segment_steps[segment] < 1
                    || (segment > first
                        && mutation_segment_start_ticks[segment]
                            <= mutation_segment_start_ticks[segment - 1])) {
                    return EIEL_NATIVE_COMPACT_REDUCER_INVALID_MUTATION_STEP_V1;
                }
                starts.push_back(mutation_segment_start_ticks[segment]);
                steps.push_back(mutation_segment_steps[segment]);
            }
            if (starts.front() > start_next_absolute_ticks[world]) {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_MUTATION_STEP_V1;
            }
            state->worlds.emplace_back(
                resident_capacities[world], start_next_absolute_ticks[world],
                std::move(starts), std::move(steps));
        }
        *output_handle = state.release();
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
}

void eiel_native_compact_state_destroy_v1(void* const handle) {
    delete mutable_state(handle);
}

std::int32_t eiel_native_compact_state_initialize_world_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::uint64_t* const agent_ids,
    const std::uint8_t* const packed_genomes,
    const std::int64_t active_agent_count) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    if (world.initialized) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    if (active_agent_count < 0
        || static_cast<std::uint64_t>(active_agent_count)
            > world.resident_capacity) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (active_agent_count > 0
        && (agent_ids == nullptr || packed_genomes == nullptr)) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    for (std::int64_t index = 0; index < active_agent_count; ++index) {
        const std::uint64_t agent_id = agent_ids[index];
        if (agent_id == 0U || agent_id == kTombstone
            || world.active_map.find(agent_id) != kMissingSlot) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1,
                static_cast<std::uint32_t>(world_index),
                world.next_absolute_tick, index, agent_id, 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1;
        }
        const std::uint32_t slot = world.free_slots.back();
        world.free_slots.pop_back();
        AgentState& agent = world.agents[slot];
        agent.agent_id = agent_id;
        std::memcpy(
            agent.genome.data(),
            packed_genomes + static_cast<std::size_t>(index) * kGenomeBytes,
            kGenomeBytes);
        agent.genome_digest_present = false;
        if (!world.active_map.insert(agent_id, slot)) {
            state->poisoned = true;
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        world.maximum_seen_agent_id = std::max(
            world.maximum_seen_agent_id, agent_id);
    }
    world.initialized = true;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_configure_snapshot_world_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::uint8_t* const cell_id,
    const std::int64_t cell_id_bytes,
    const std::uint16_t* const tracked_policy_loci,
    const std::int32_t tracked_policy_locus_count,
    const std::uint16_t* const tracked_trait_loci,
    const std::int32_t tracked_trait_locus_count,
    const std::int32_t strategy_count,
    const std::uint8_t* const strategy_policy_sha256_rows,
    const std::uint64_t* const strategy_policy_masks,
    const std::int32_t strategy_policy_count,
    const std::int64_t* const threshold_ticks,
    const std::int32_t threshold_count,
    const std::int64_t early_life_tick) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    if (cell_id_bytes < 1 || cell_id_bytes > 4096
        || tracked_policy_locus_count < 0
        || tracked_policy_locus_count > static_cast<std::int32_t>(kMaxPolicyLoci)
        || tracked_trait_locus_count < 0
        || tracked_trait_locus_count > static_cast<std::int32_t>(kMaxTraitLoci)
        || strategy_count < 0
        || strategy_count > static_cast<std::int32_t>(kMaxStrategies)
        || strategy_policy_count < 0
        || strategy_policy_count > static_cast<std::int32_t>(kMaxStrategies)
        || threshold_count < 0
        || threshold_count > static_cast<std::int32_t>(kMaxThresholds)
        || early_life_tick < 0) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (cell_id == nullptr
        || (tracked_policy_locus_count > 0 && tracked_policy_loci == nullptr)
        || (tracked_trait_locus_count > 0 && tracked_trait_loci == nullptr)
        || (strategy_policy_count > 0
            && (strategy_policy_sha256_rows == nullptr
                || strategy_policy_masks == nullptr))
        || (threshold_count > 0 && threshold_ticks == nullptr)) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if ((strategy_count == 0
         && (strategy_policy_count != 0 || threshold_count != 0))
        || (strategy_count > 0
            && (strategy_policy_count < 1 || threshold_count < 1))) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    if (world.initialized || world.dispatch_count != 0U
        || world.snapshot.configured) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    try {
        SnapshotConfiguration configuration;
        configuration.cell_id.assign(
            reinterpret_cast<const char*>(cell_id),
            static_cast<std::size_t>(cell_id_bytes));
        for (const char character : configuration.cell_id) {
            const auto value = static_cast<unsigned char>(character);
            if (value < 0x20U || value > 0x7eU || value == '"' || value == '\\') {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
            }
        }
        for (std::size_t index = 0U; index < state->worlds.size(); ++index) {
            if (index != static_cast<std::size_t>(world_index)
                && state->worlds[index].snapshot.configured
                && state->worlds[index].snapshot.cell_id
                    == configuration.cell_id) {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
            }
        }
        if (tracked_policy_locus_count > 0) {
            configuration.policy_loci.assign(
                tracked_policy_loci,
                tracked_policy_loci + tracked_policy_locus_count);
        }
        if (tracked_trait_locus_count > 0) {
            configuration.trait_loci.assign(
                tracked_trait_loci,
                tracked_trait_loci + tracked_trait_locus_count);
        }
        if (!std::is_sorted(
                configuration.policy_loci.begin(),
                configuration.policy_loci.end())
            || std::adjacent_find(
                configuration.policy_loci.begin(),
                configuration.policy_loci.end())
                != configuration.policy_loci.end()
            || (!configuration.policy_loci.empty()
                && configuration.policy_loci.back() >= 112U)
            || !std::is_sorted(
                configuration.trait_loci.begin(),
                configuration.trait_loci.end())
            || std::adjacent_find(
                configuration.trait_loci.begin(),
                configuration.trait_loci.end())
                != configuration.trait_loci.end()
            || (!configuration.trait_loci.empty()
                && configuration.trait_loci.back() >= 16U)) {
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
        }
        configuration.strategy_count = static_cast<std::uint8_t>(strategy_count);
        configuration.strategy_policies.reserve(
            static_cast<std::size_t>(strategy_policy_count));
        for (std::int32_t index = 0; index < strategy_policy_count; ++index) {
            SnapshotPolicyMapping mapping;
            std::memcpy(
                mapping.policy_digest.data(),
                strategy_policy_sha256_rows
                    + static_cast<std::size_t>(index) * mapping.policy_digest.size(),
                mapping.policy_digest.size());
            mapping.strategy_mask = strategy_policy_masks[index];
            if (!strategy_mask_is_valid(
                    mapping.strategy_mask, configuration.strategy_count)
                || std::any_of(
                    configuration.strategy_policies.begin(),
                    configuration.strategy_policies.end(),
                    [&](const SnapshotPolicyMapping& existing) noexcept {
                        return existing.policy_digest == mapping.policy_digest;
                    })) {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
            }
            configuration.strategy_policies.push_back(mapping);
        }
        if (threshold_count > 0) {
            configuration.thresholds.assign(
                threshold_ticks, threshold_ticks + threshold_count);
        }
        for (std::size_t index = 0U; index < configuration.thresholds.size();
             ++index) {
            if (configuration.thresholds[index] < 0
                || (index > 0U
                    && configuration.thresholds[index]
                        <= configuration.thresholds[index - 1U])) {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
            }
        }
        configuration.early_life_tick = early_life_tick;
        configuration.configured = true;
        world.snapshot = std::move(configuration);
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
}

std::int32_t eiel_native_compact_state_configure_genotype_attribution_v1(
    void* const handle,
    const std::int64_t world_index) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    if (!world.snapshot.configured || world.snapshot.genotype_attribution_only
        || world.snapshot.strategy_count != 0U
        || !world.snapshot.strategy_policies.empty()
        || !world.snapshot.thresholds.empty() || world.initialized
        || world.dispatch_count != 0U) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    world.snapshot.genotype_attribution_only = true;
    world.snapshot.strategy_count = 1U;
    world.snapshot.early_life_tick = 0;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_configure_genotype_focal_ids_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::uint8_t* const focal_ids,
    const std::int64_t focal_id_count) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    if (focal_ids == nullptr || focal_id_count < 1 || focal_id_count > 256) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    if (!world.snapshot.genotype_attribution_only
        || world.snapshot.genotype_focal_filter_enabled || world.initialized
        || world.dispatch_count != 0U) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    try {
        world.snapshot.genotype_focal_ids.clear();
        world.snapshot.genotype_focal_ids.reserve(
            static_cast<std::size_t>(focal_id_count));
        for (std::int64_t index = 0; index < focal_id_count; ++index) {
            Digest256 digest{};
            std::memcpy(
                digest.data(), focal_ids + static_cast<std::size_t>(index) * 32U,
                digest.size());
            if (!world.snapshot.genotype_focal_ids.empty()
                && digest <= world.snapshot.genotype_focal_ids.back()) {
                world.snapshot.genotype_focal_ids.clear();
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
            }
            world.snapshot.genotype_focal_ids.push_back(digest);
        }
        world.snapshot.genotype_focal_filter_enabled = true;
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
}

std::int32_t eiel_native_compact_state_configure_hereditary_birth_output_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::uint8_t* const path,
    const std::int64_t path_bytes) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    if (path == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (path_bytes < 1 || path_bytes > 4096) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    if (!world.snapshot.genotype_attribution_only || world.initialized
        || world.dispatch_count != 0U || world.hereditary_birth_sink != nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    try {
        auto sink = std::make_unique<GzipHereditaryBirthRecordSink>(std::string(
            reinterpret_cast<const char*>(path),
            static_cast<std::size_t>(path_bytes)));
        if (!sink->open()) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        world.hereditary_birth_sink = std::move(sink);
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
}

std::int32_t eiel_native_compact_state_finalize_hereditary_birth_output_v1(
    void* const handle,
    const std::int64_t world_index) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    if (!world.snapshot.genotype_attribution_only
        || world.hereditary_birth_sink == nullptr
        || !world.snapshot.configured || !world.initialized
        || world.next_absolute_tick < 0) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    if (!world.hereditary_birth_sink->finalize()) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
            static_cast<std::uint32_t>(world_index), world.next_absolute_tick,
            0, 0U, 0U, 0U);
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_process_snapshot_v1(
    void* const handle,
    const std::int64_t absolute_tick,
    const std::uint8_t* const canonical_json_lines,
    const std::int64_t canonical_json_bytes,
    const std::int64_t row_count) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    state->snapshot_result.clear();
    state->snapshot_result_ready = false;
    if (absolute_tick < 0 || canonical_json_bytes < 1 || row_count < 1) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (canonical_json_lines == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    try {
        const char* const input_begin =
            reinterpret_cast<const char*>(canonical_json_lines);
        const char* const input_end = input_begin + canonical_json_bytes;
        static constexpr std::string_view kWorldMarker =
            "\"row_kind\":\"world_state\"";
        std::vector<SnapshotSegment> segments;
        segments.reserve(state->worlds.size());
        std::int64_t observed_rows = 0;
        const char* line_begin = input_begin;
        SnapshotSegment current;
        bool current_present = false;
        std::unordered_set<std::uint32_t> seen_worlds;
        seen_worlds.reserve(state->worlds.size() * 2U);
        while (line_begin < input_end) {
            const void* const newline_raw = std::memchr(
                line_begin, '\n',
                static_cast<std::size_t>(input_end - line_begin));
            const char* const line_end = newline_raw == nullptr
                ? input_end
                : static_cast<const char*>(newline_raw);
            if (line_begin == line_end) {
                state->poisoned = true;
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                    0U, absolute_tick, observed_rows, 0U, 0U, 0U);
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1;
            }
            const bool is_world_start = std::search(
                line_begin, line_end,
                kWorldMarker.begin(), kWorldMarker.end()) != line_end;
            if (is_world_start) {
                ParsedSnapshotRow first_row;
                if (!parse_snapshot_row(line_begin, line_end, &first_row)
                    || first_row.kind != SnapshotRowKind::kWorldState
                    || first_row.absolute_tick != absolute_tick) {
                    state->poisoned = true;
                    set_failure(
                        state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                        0U, absolute_tick, observed_rows, 0U, 0U, 0U);
                    return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1;
                }
                std::size_t world_index = state->worlds.size();
                for (std::size_t index = 0U; index < state->worlds.size(); ++index) {
                    if (state->worlds[index].snapshot.configured
                        && first_row.cell_id
                            == state->worlds[index].snapshot.cell_id) {
                        world_index = index;
                        break;
                    }
                }
                if (world_index == state->worlds.size()
                    || !seen_worlds.insert(
                        static_cast<std::uint32_t>(world_index)).second) {
                    state->poisoned = true;
                    set_failure(
                        state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1,
                        static_cast<std::uint32_t>(
                            world_index == state->worlds.size() ? 0U : world_index),
                        absolute_tick, observed_rows, 0U, 0U, 0U);
                    return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
                }
                if (current_present) {
                    current.end = line_begin;
                    segments.push_back(current);
                } else if (line_begin != input_begin) {
                    state->poisoned = true;
                    set_failure(
                        state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                        0U, absolute_tick, observed_rows, 0U, 0U, 0U);
                    return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1;
                }
                current = SnapshotSegment{
                    line_begin, nullptr, observed_rows,
                    static_cast<std::uint32_t>(world_index)};
                current_present = true;
            } else if (!current_present) {
                state->poisoned = true;
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                    0U, absolute_tick, observed_rows, 0U, 0U, 0U);
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1;
            }
            ++observed_rows;
            line_begin = newline_raw == nullptr ? input_end : line_end + 1;
        }
        if (!current_present || observed_rows != row_count) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1,
                0U, absolute_tick, observed_rows, 0U, 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
        }
        current.end = input_end;
        segments.push_back(current);
        std::vector<SnapshotWorldProjection> projections(segments.size());
        std::atomic<std::size_t> next_segment{0U};
        std::atomic<std::int32_t> first_status{
            EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1};
        const std::int32_t worker_count = std::max<std::int32_t>(
            1, std::min<std::int32_t>(
                state->max_worker_threads,
                static_cast<std::int32_t>(segments.size())));
        const auto worker = [&]() noexcept {
            while (first_status.load(std::memory_order_acquire)
                   == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                const std::size_t segment_index =
                    next_segment.fetch_add(1U, std::memory_order_relaxed);
                if (segment_index >= segments.size()) return;
                Status status = EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
                try {
                    status = process_snapshot_segment(
                        state, segments[segment_index], absolute_tick,
                        &projections[segment_index]);
                } catch (const std::bad_alloc&) {
                    status = EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
                    set_failure(
                        state, status, segments[segment_index].world_index,
                        absolute_tick, segments[segment_index].first_row,
                        0U, 0U, 0U);
                } catch (...) {
                    status = EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
                    set_failure(
                        state, status, segments[segment_index].world_index,
                        absolute_tick, segments[segment_index].first_row,
                        0U, 0U, 0U);
                }
                if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                    std::int32_t expected = EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
                    first_status.compare_exchange_strong(
                        expected, static_cast<std::int32_t>(status),
                        std::memory_order_release, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (worker_count == 1) {
            worker();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(worker_count));
            try {
                for (std::int32_t index = 0; index < worker_count; ++index) {
                    threads.emplace_back(worker);
                }
            } catch (...) {
                for (std::thread& thread : threads) {
                    if (thread.joinable()) thread.join();
                }
                state->poisoned = true;
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_THREAD_FAILURE_V1,
                    0U, absolute_tick, 0, 0U, 0U, 0U);
                return EIEL_NATIVE_COMPACT_REDUCER_THREAD_FAILURE_V1;
            }
            for (std::thread& thread : threads) thread.join();
        }
        const std::int32_t status = first_status.load(std::memory_order_acquire);
        if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            state->poisoned = true;
            return status;
        }
        std::sort(
            projections.begin(), projections.end(),
            [](const SnapshotWorldProjection& left,
               const SnapshotWorldProjection& right) noexcept {
                return left.world_index < right.world_index;
            });
        std::string result;
        result.reserve(segments.size() * 8192U);
        result.append("{\"absolute_tick\":");
        append_i64_json(&result, absolute_tick);
        result.append(",\"schema_version\":"
                      "\"eiel_native_compact_snapshot_result_v1\",\"worlds\":[");
        for (std::size_t index = 0U; index < projections.size(); ++index) {
            if (index != 0U) result.push_back(',');
            const SnapshotWorldProjection& projection = projections[index];
            result.append(render_snapshot_world_projection(
                projection, state->worlds[projection.world_index].snapshot));
        }
        result.append("]}");
        state->snapshot_result = std::move(result);
        state->snapshot_result_ready = true;
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1,
            0U, absolute_tick, 0, 0U, 0U, 0U);
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
            0U, absolute_tick, 0, 0U, 0U, 0U);
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
}

std::int32_t eiel_native_compact_state_snapshot_result_size_v1(
    const void* const handle, std::int64_t* const output_size) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr || output_size == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (!state->snapshot_result_ready
        || state->snapshot_result.size()
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    *output_size = static_cast<std::int64_t>(state->snapshot_result.size());
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_copy_snapshot_result_v1(
    const void* const handle,
    std::uint8_t* const output,
    const std::int64_t output_capacity) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (!state->snapshot_result_ready || output_capacity < 0
        || static_cast<std::uint64_t>(output_capacity)
            < state->snapshot_result.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (!state->snapshot_result.empty() && output == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    std::memcpy(output, state->snapshot_result.data(), state->snapshot_result.size());
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_process_dispatch_v1(
    void* const handle,
    const std::int64_t start_tick,
    const std::int64_t end_tick,
    const std::uint8_t* const event_rows,
    const std::int64_t event_bytes,
    const std::uint8_t* const death_metric_rows,
    const std::int64_t death_metric_bytes,
    const std::uint8_t* const provenance_rows,
    const std::int64_t provenance_bytes,
    const std::int64_t* const world_offsets,
    const std::int64_t world_offset_count) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    const std::int64_t world_count = static_cast<std::int64_t>(state->worlds.size());
    if (end_tick <= start_tick || end_tick - start_tick > 64
        || event_bytes < 0 || event_bytes % static_cast<std::int64_t>(kEventRowBytes)
        || death_metric_bytes < 0
        || death_metric_bytes % static_cast<std::int64_t>(kMetricRowBytes)
        || provenance_bytes < 0
        || provenance_bytes % static_cast<std::int64_t>(kProvenanceBytes)
        || world_offset_count != world_count + 1 || world_offsets == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if ((event_bytes > 0 && event_rows == nullptr)
        || (death_metric_bytes > 0 && death_metric_rows == nullptr)
        || (provenance_bytes > 0 && provenance_rows == nullptr)) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    const std::int64_t event_count =
        event_bytes / static_cast<std::int64_t>(kEventRowBytes);
    if (world_offsets[0] != 0 || world_offsets[world_count] != event_count) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    std::vector<std::int64_t> birth_counts(static_cast<std::size_t>(world_count));
    std::vector<std::int64_t> death_counts(static_cast<std::size_t>(world_count));
    std::vector<std::int64_t> birth_bases(static_cast<std::size_t>(world_count));
    std::vector<std::int64_t> death_bases(static_cast<std::size_t>(world_count));
    try {
        for (std::int64_t world_index = 0; world_index < world_count; ++world_index) {
            WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
            const std::int64_t first = world_offsets[world_index];
            const std::int64_t stop = world_offsets[world_index + 1];
            if (!world.initialized || world.next_absolute_tick != start_tick) {
                state->poisoned = true;
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_NONCONTIGUOUS_TICK_V1,
                    static_cast<std::uint32_t>(world_index), start_tick,
                    first, 0U, 0U, 0U);
                return EIEL_NATIVE_COMPACT_REDUCER_NONCONTIGUOUS_TICK_V1;
            }
            if (first < 0 || stop < first || stop > event_count) {
                return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
            }
            std::int64_t prior_tick = start_tick - 1;
            for (std::int64_t row = first; row < stop; ++row) {
                const std::uint8_t* event = event_rows
                    + static_cast<std::size_t>(row) * kEventRowBytes;
                const std::uint64_t packed = read_u64_le(event + 24U);
                const std::uint64_t kind = (packed >> 6U) & 0x3U;
                const std::int64_t tick = start_tick
                    + static_cast<std::int64_t>(packed & 0x3fU);
                if (tick < prior_tick || tick < start_tick || tick >= end_tick
                    || (kind != 1U && kind != 2U)) {
                    state->poisoned = true;
                    set_failure(
                        state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1,
                        static_cast<std::uint32_t>(world_index), tick, row,
                        read_u64_le(event), read_u64_le(event + 8U),
                        read_u64_le(event + 16U));
                    return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1;
                }
                prior_tick = tick;
                if (kind == 1U) ++death_counts[static_cast<std::size_t>(world_index)];
                else ++birth_counts[static_cast<std::size_t>(world_index)];
            }
        }
        std::int64_t birth_total = 0;
        std::int64_t death_total = 0;
        for (std::int64_t world = 0; world < world_count; ++world) {
            birth_bases[static_cast<std::size_t>(world)] = birth_total;
            death_bases[static_cast<std::size_t>(world)] = death_total;
            birth_total += birth_counts[static_cast<std::size_t>(world)];
            death_total += death_counts[static_cast<std::size_t>(world)];
        }
        if (birth_total
                != provenance_bytes / static_cast<std::int64_t>(kProvenanceBytes)
            || death_total
                != death_metric_bytes / static_cast<std::int64_t>(kMetricRowBytes)) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_AUXILIARY_CURSOR_V1,
                0U, start_tick, 0, 0U, 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_AUXILIARY_CURSOR_V1;
        }
        std::atomic<std::int64_t> next_world{0};
        std::atomic<std::int32_t> first_status{
            EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1};
        const std::int32_t worker_count = std::max<std::int32_t>(
            1, std::min<std::int32_t>(
                state->max_worker_threads,
                static_cast<std::int32_t>(world_count)));
        const auto worker = [&]() noexcept {
            while (first_status.load(std::memory_order_acquire)
                   == EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                const std::int64_t world_index =
                    next_world.fetch_add(1, std::memory_order_relaxed);
                if (world_index >= world_count) return;
                const Status status = process_world(
                    state, static_cast<std::uint32_t>(world_index),
                    start_tick, end_tick, event_rows,
                    world_offsets[world_index], world_offsets[world_index + 1],
                    death_metric_rows,
                    death_bases[static_cast<std::size_t>(world_index)],
                    provenance_rows,
                    birth_bases[static_cast<std::size_t>(world_index)]);
                if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
                    std::int32_t expected = EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
                    first_status.compare_exchange_strong(
                        expected, static_cast<std::int32_t>(status),
                        std::memory_order_release, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (worker_count == 1) {
            worker();
        } else {
            std::vector<std::thread> threads;
            threads.reserve(static_cast<std::size_t>(worker_count));
            try {
                for (std::int32_t index = 0; index < worker_count; ++index) {
                    threads.emplace_back(worker);
                }
            } catch (...) {
                for (std::thread& thread : threads) {
                    if (thread.joinable()) thread.join();
                }
                state->poisoned = true;
                set_failure(
                    state, EIEL_NATIVE_COMPACT_REDUCER_THREAD_FAILURE_V1,
                    0U, start_tick, 0, 0U, 0U, 0U);
                return EIEL_NATIVE_COMPACT_REDUCER_THREAD_FAILURE_V1;
            }
            for (std::thread& thread : threads) thread.join();
        }
        const std::int32_t status = first_status.load(std::memory_order_acquire);
        if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            state->poisoned = true;
            return status;
        }
        return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    } catch (const std::bad_alloc&) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1,
            0U, start_tick, 0, 0U, 0U, 0U);
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
            0U, start_tick, 0, 0U, 0U, 0U);
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
}

std::int32_t eiel_native_compact_state_world_receipt_v1(
    const void* const handle,
    const std::int64_t world_index,
    WorldReceipt* const output_receipt) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr || output_receipt == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    const WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    WorldReceipt receipt{};
    receipt.birth_count = world.birth_count;
    receipt.death_count = world.death_count;
    receipt.recombination_birth_count = world.recombination_birth_count;
    receipt.carrier_reproduction_birth_count =
        world.carrier_reproduction_birth_count;
    receipt.mutation_carrier_reproduction_birth_count =
        world.mutation_carrier_reproduction_birth_count;
    receipt.unique_born_agents_reproduced_count =
        world.unique_born_agents_reproduced_count;
    receipt.unique_mutation_carriers_reproduced_count =
        world.unique_mutation_carriers_reproduced_count;
    receipt.maximum_lineage_depth = world.maximum_lineage_depth;
    receipt.dispatch_count = world.dispatch_count;
    receipt.event_count = world.event_count;
    receipt.dispatch_tick_count = world.dispatch_tick_count;
    receipt.next_absolute_tick = world.next_absolute_tick;
    receipt.active_agent_count =
        static_cast<std::uint64_t>(world.resident_capacity)
        - static_cast<std::uint64_t>(world.free_slots.size());
    receipt.births = world.birth_receipt;
    for (std::size_t field = 0U; field < kMetricFields; ++field) {
        store_i128(world.metric_totals[field],
                   &receipt.completed_lifetime_metric_totals[field]);
    }
    *output_receipt = receipt;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_export_active_v1(
    const void* const handle,
    const std::int64_t world_index,
    eiel_native_compact_active_record_v1* const output_records,
    const std::int64_t record_capacity,
    std::int64_t* const output_record_count) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr || output_record_count == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()
        || record_capacity < 0) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    const WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    const std::int64_t active_count = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(world.resident_capacity)
        - static_cast<std::uint64_t>(world.free_slots.size()));
    *output_record_count = active_count;
    if (record_capacity < active_count) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (active_count > 0 && output_records == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    std::int64_t output_index = 0;
    for (const AgentState& agent : world.agents) {
        if (agent.agent_id == 0U) continue;
        eiel_native_compact_active_record_v1& output =
            output_records[output_index++];
        output = eiel_native_compact_active_record_v1{};
        output.agent_id = agent.agent_id;
        output.generation = agent.generation;
        output.strategy_mask = agent.strategy_mask;
        output.birth_tick = agent.birth_tick;
        output.mutation_lineage = static_cast<std::uint8_t>(agent.mutation_lineage);
        output.born = static_cast<std::uint8_t>(agent.born);
        output.reproduced = static_cast<std::uint8_t>(agent.reproduced);
        output.founder = static_cast<std::uint8_t>(agent.founder);
        output.latest_metrics_present =
            static_cast<std::uint8_t>(agent.latest_metrics_present);
        std::memcpy(output.genome, agent.genome.data(), kGenomeBytes);
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_configure_strategy_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::int32_t strategy_count,
    const std::uint64_t* const founder_agent_ids,
    const std::uint64_t* const founder_strategy_masks,
    const std::int64_t founder_count,
    const std::int64_t* const threshold_ticks,
    const std::int32_t threshold_count,
    const std::int64_t early_life_tick) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    const std::int64_t active_count = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(world.resident_capacity)
        - static_cast<std::uint64_t>(world.free_slots.size()));
    if (!world.initialized || world.dispatch_count != 0U
        || world.strategy.enabled || strategy_count < 1
        || strategy_count > static_cast<std::int32_t>(kMaxStrategies)
        || founder_count != active_count || threshold_count < 1
        || threshold_count > static_cast<std::int32_t>(kMaxThresholds)
        || early_life_tick < 0) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    if ((founder_count > 0
         && (founder_agent_ids == nullptr || founder_strategy_masks == nullptr))
        || threshold_ticks == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    for (std::int32_t threshold = 0; threshold < threshold_count; ++threshold) {
        if (threshold_ticks[threshold] < 0
            || (threshold > 0
                && threshold_ticks[threshold] <= threshold_ticks[threshold - 1])) {
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
        }
    }
    const auto strategy_width = static_cast<std::uint8_t>(strategy_count);
    for (AgentState& agent : world.agents) agent.strategy_snapshot_epoch = 0U;
    for (std::int64_t index = 0; index < founder_count; ++index) {
        const std::uint32_t slot = world.active_map.find(founder_agent_ids[index]);
        if (slot == kMissingSlot
            || !strategy_mask_is_valid(
                founder_strategy_masks[index], strategy_width)
            || world.agents[slot].strategy_snapshot_epoch != 0U) {
            for (AgentState& agent : world.agents) {
                agent.strategy_snapshot_epoch = 0U;
            }
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        world.agents[slot].strategy_snapshot_epoch = 1U;
    }
    if (std::any_of(
            world.agents.begin(), world.agents.end(),
            [](const AgentState& agent) {
                return agent.agent_id != 0U
                    && agent.strategy_snapshot_epoch != 1U;
            })) {
        for (AgentState& agent : world.agents) agent.strategy_snapshot_epoch = 0U;
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    world.strategy = StrategyState{};
    StrategyState& strategy = world.strategy;
    strategy.enabled = true;
    strategy.strategy_count = strategy_width;
    strategy.threshold_count = static_cast<std::uint8_t>(threshold_count);
    strategy.early_life_tick = early_life_tick;
    strategy.genotype_policy_loci = world.snapshot.policy_loci;
    strategy.genotype_trait_loci = world.snapshot.trait_loci;
    strategy.genotype_focal_filter_enabled =
        world.snapshot.genotype_focal_filter_enabled;
    strategy.genotype_focal_ids = world.snapshot.genotype_focal_ids;
    strategy.total_agent_count = static_cast<std::uint64_t>(founder_count);
    strategy.founder_count = static_cast<std::uint64_t>(founder_count);
    for (std::int32_t threshold = 0; threshold < threshold_count; ++threshold) {
        strategy.thresholds[static_cast<std::size_t>(threshold)] =
            threshold_ticks[threshold];
    }
    for (std::int64_t index = 0; index < founder_count; ++index) {
        AgentState& agent = world.agents[
            world.active_map.find(founder_agent_ids[index])];
        agent.strategy_mask = founder_strategy_masks[index];
        agent.birth_tick = world.next_absolute_tick;
        agent.founder = true;
        agent.latest_metrics_present = false;
        agent.strategy_snapshot_epoch = 0U;
        agent.genotype_digest = tracked_genotype_state_digest(
            agent.genome, world.snapshot);
        agent.genotype_digest_present = true;
        const Status genotype_status = add_genotype_agent(
            &strategy, agent.genotype_digest, true);
        if (genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            state->poisoned = true;
            set_failure(
                state, genotype_status, static_cast<std::uint32_t>(world_index),
                world.next_absolute_tick, index, agent.agent_id, 0U, 0U);
            return genotype_status;
        }
        const bool counted = for_each_strategy_bit(
            agent.strategy_mask, strategy.strategy_count,
            [&](const std::uint8_t strategy_index) noexcept {
                return increment_u64(
                    &strategy.agent_counts[
                        static_cast<std::size_t>(strategy_index)]);
            });
        if (!counted) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1,
                static_cast<std::uint32_t>(world_index),
                world.next_absolute_tick, index, agent.agent_id, 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_observe_strategy_snapshot_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::uint64_t* const active_agent_ids,
    const std::uint8_t* const active_metric_rows,
    const std::int64_t active_agent_count) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    const std::int64_t expected_count = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(world.resident_capacity)
        - static_cast<std::uint64_t>(world.free_slots.size()));
    if (!world.strategy.enabled || world.strategy.finalized
        || active_agent_count != expected_count) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    if (active_agent_count > 0
        && (active_agent_ids == nullptr || active_metric_rows == nullptr)) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (world.strategy.snapshot_epoch
        == std::numeric_limits<std::uint64_t>::max()) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    const std::uint64_t epoch = ++world.strategy.snapshot_epoch;
    for (std::int64_t index = 0; index < active_agent_count; ++index) {
        const std::uint32_t slot = world.active_map.find(active_agent_ids[index]);
        if (slot == kMissingSlot
            || world.agents[slot].strategy_snapshot_epoch == epoch) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                static_cast<std::uint32_t>(world_index),
                world.next_absolute_tick, index, active_agent_ids[index], 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        world.agents[slot].strategy_snapshot_epoch = epoch;
    }
    for (const AgentState& agent : world.agents) {
        if (agent.agent_id != 0U && agent.strategy_snapshot_epoch != epoch) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                static_cast<std::uint32_t>(world_index),
                world.next_absolute_tick, 0, agent.agent_id, 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
    }
    for (std::int64_t index = 0; index < active_agent_count; ++index) {
        AgentState& agent = world.agents[
            world.active_map.find(active_agent_ids[index])];
        const std::uint8_t* cursor = active_metric_rows
            + static_cast<std::size_t>(index) * kMetricRowBytes;
        for (std::size_t field = 0U; field < kMetricFields; ++field) {
            if (field == 17U) {
                agent.latest_metrics[field] = read_i64_le(cursor);
                cursor += 8U;
            } else {
                agent.latest_metrics[field] =
                    static_cast<std::int64_t>(read_u32_le(cursor));
                cursor += 4U;
            }
        }
        agent.latest_metrics_present = true;
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_finalize_strategy_v1(
    void* const handle,
    const std::int64_t world_index,
    const std::int64_t terminal_tick) {
    ReducerState* const state = mutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    WorldState& world = state->worlds[static_cast<std::size_t>(world_index)];
    StrategyState& strategy = world.strategy;
    if (!strategy.enabled || strategy.finalized
        || terminal_tick != world.next_absolute_tick
        || strategy.birth_count != world.birth_count
        || strategy.death_count != world.death_count
        || strategy.total_agent_count
            != strategy.founder_count + strategy.birth_count) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    for (std::size_t threshold = 0U; threshold < strategy.threshold_count;
         ++threshold) {
        if (strategy.thresholds[threshold] > terminal_tick) {
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
        }
    }
    for (const AgentState& agent : world.agents) {
        if (agent.agent_id == 0U) continue;
        if (!agent.latest_metrics_present) {
            state->poisoned = true;
            set_failure(
                state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
                static_cast<std::uint32_t>(world_index), terminal_tick,
                0, agent.agent_id, 0U, 0U);
            return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
        }
        const Status status = close_strategy_agent(
            &strategy, agent, agent.latest_metrics, terminal_tick, true);
        if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            state->poisoned = true;
            set_failure(
                state, status, static_cast<std::uint32_t>(world_index),
                terminal_tick, 0, agent.agent_id, 0U, 0U);
            return status;
        }
        const Status genotype_status = close_genotype_agent(
            &strategy, agent, agent.latest_metrics, true);
        if (genotype_status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            state->poisoned = true;
            set_failure(
                state, genotype_status, static_cast<std::uint32_t>(world_index),
                terminal_tick, 0, agent.agent_id, 0U, 0U);
            return genotype_status;
        }
    }
    try {
        strategy.genotype_result = render_genotype_result(strategy);
        strategy.genotype_result_ready = true;
    } catch (const std::bad_alloc&) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1,
            static_cast<std::uint32_t>(world_index), terminal_tick, 0, 0U, 0U,
            0U);
        return EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1;
    } catch (...) {
        state->poisoned = true;
        set_failure(
            state, EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1,
            static_cast<std::uint32_t>(world_index), terminal_tick, 0, 0U, 0U,
            0U);
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    strategy.finalized = true;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_strategy_receipt_v1(
    const void* const handle,
    const std::int64_t world_index,
    eiel_native_compact_strategy_receipt_v1* const output_receipt) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr || output_receipt == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    const StrategyState& strategy =
        state->worlds[static_cast<std::size_t>(world_index)].strategy;
    if (!strategy.enabled) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    eiel_native_compact_strategy_receipt_v1 receipt{};
    receipt.enabled = 1U;
    receipt.finalized = static_cast<std::uint8_t>(strategy.finalized);
    receipt.strategy_count = strategy.strategy_count;
    receipt.threshold_count = strategy.threshold_count;
    receipt.early_life_tick = strategy.early_life_tick;
    receipt.first_birth_tick = strategy.first_birth_tick;
    receipt.first_death_tick = strategy.first_death_tick;
    receipt.extinction_tick = strategy.extinction_tick;
    receipt.total_agent_count = strategy.total_agent_count;
    receipt.founder_count = strategy.founder_count;
    receipt.birth_count = strategy.birth_count;
    receipt.death_count = strategy.death_count;
    receipt.maximum_lineage_depth = strategy.maximum_lineage_depth;
    receipt.mixed_parent_birth_count = strategy.mixed_parent_birth_count;
    for (std::size_t threshold = 0U; threshold < kMaxThresholds; ++threshold) {
        receipt.threshold_ticks[threshold] = strategy.thresholds[threshold];
    }
    for (std::size_t index = 0U; index < kMaxStrategies; ++index) {
        receipt.agent_counts_by_strategy[index] = strategy.agent_counts[index];
        receipt.offspring_born_by_strategy[index] = strategy.offspring_born[index];
        receipt.maximum_lineage_depth_by_strategy[index] =
            strategy.max_depth_by_strategy[index];
        receipt.parent_birth_events_by_strategy[index] =
            strategy.parent_birth_events[index];
        receipt.death_events_by_strategy[index] = strategy.death_events[index];
        receipt.persistence_to_early_life_by_strategy[index] =
            strategy.persistence_to_early[index];
        receipt.terminal_alive_by_strategy[index] =
            strategy.terminal_alive[index];
    }
    for (std::size_t index = 0U;
         index < kMaxStrategies * kMaxThresholds; ++index) {
        receipt.offspring_survived_by_strategy_threshold[index] =
            strategy.offspring_survived[index];
    }
    for (std::size_t index = 0U;
         index < kMaxStrategies * kMetricFields; ++index) {
        store_i128(
            strategy.metric_totals[index],
            &receipt.terminal_metric_totals_by_strategy[index]);
    }
    receipt.offspring_group_count = strategy.offspring_group_count;
    for (std::size_t index = 0U; index < strategy.offspring_group_count; ++index) {
        receipt.offspring_groups[index] =
            eiel_native_compact_strategy_group_count_v1{
                strategy.offspring_groups[index].mask,
                strategy.offspring_groups[index].count};
    }
    receipt.parent_pair_count = strategy.parent_pair_count;
    for (std::size_t index = 0U; index < strategy.parent_pair_count; ++index) {
        receipt.parent_pairs[index] =
            eiel_native_compact_strategy_parent_pair_count_v1{
                strategy.parent_pairs[index].first_mask,
                strategy.parent_pairs[index].second_mask,
                strategy.parent_pairs[index].count};
    }
    *output_receipt = receipt;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_genotype_result_size_v1(
    const void* const handle,
    const std::int64_t world_index,
    std::int64_t* const output_size) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr || output_size == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    const StrategyState& strategy =
        state->worlds[static_cast<std::size_t>(world_index)].strategy;
    if (!strategy.enabled || !strategy.finalized
        || !strategy.genotype_result_ready
        || strategy.genotype_result.size()
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1;
    }
    *output_size = static_cast<std::int64_t>(strategy.genotype_result.size());
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_copy_genotype_result_v1(
    const void* const handle,
    const std::int64_t world_index,
    std::uint8_t* const output,
    const std::int64_t output_capacity) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    if (state->poisoned) return EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1;
    if (world_index < 0
        || static_cast<std::size_t>(world_index) >= state->worlds.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1;
    }
    const StrategyState& strategy =
        state->worlds[static_cast<std::size_t>(world_index)].strategy;
    if (!strategy.enabled || !strategy.finalized
        || !strategy.genotype_result_ready || output_capacity < 0
        || static_cast<std::uint64_t>(output_capacity)
            < strategy.genotype_result.size()) {
        return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    }
    if (!strategy.genotype_result.empty() && output == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    std::memcpy(
        output, strategy.genotype_result.data(), strategy.genotype_result.size());
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_state_failure_v1(
    const void* const handle,
    Failure* const output_failure) {
    const ReducerState* const state = immutable_state(handle);
    if (state == nullptr || output_failure == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    *output_failure = state->failure;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

}  // extern "C"
