#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace eiel::cuda_architecture_b::resident_event_archive_v1 {

constexpr std::size_t kEventColumnCountV1 = 22;
constexpr std::size_t kTypedSequenceColumnV1 = 9;
constexpr int64_t kEventRowBytesV1 = 141;
constexpr int64_t kHostEventArchiveBudgetBytesV1 = 4LL * 1024 * 1024 * 1024;

constexpr std::array<torch::ScalarType, kEventColumnCountV1> kEventColumnDtypesV1{
    torch::kInt32,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kFloat32,
    torch::kInt32,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt32,
    torch::kInt32,
    torch::kInt32,
    torch::kInt8,
    torch::kFloat32,
};

using EventChunkV1 = std::vector<torch::Tensor>;
using EventCellArchiveV1 = std::vector<EventChunkV1>;

struct EventArchiveV1 {
    std::vector<EventCellArchiveV1> cells;
    std::vector<int64_t> archived_rows;
    int64_t archived_bytes = 0;
};

struct EventDrainReceiptV1 {
    std::vector<int64_t> active_count_before;
    std::vector<int64_t> sequence_base_before;
    std::vector<int64_t> sequence_base_after;
    int64_t copied_bytes = 0;
    int64_t cumulative_archive_bytes = 0;
    int64_t remaining_budget_bytes = 0;
};

// The external sink is deliberately column-blocked rather than row-packed:
// each record can be written directly from one bounded D2H column copy, and
// recovery only needs to scan record headers and skip fixed-size payloads.
// Header layout is shared with cuda_architecture_b_event_sink_v1.py:
// magic[8], uint32 version, uint32 cells (0 means dynamic), uint32 columns,
// uint32 row_bytes; records are uint32 cell, uint32 reserved, uint64 rows,
// followed by one contiguous payload block per event column.
struct EventSinkV1 {
    std::string path;
    int64_t cells = 0;
    std::vector<int64_t> archived_rows;
    int64_t archived_bytes = 0;

    EventSinkV1() = default;

    EventSinkV1(const std::string& sink_path, const int64_t cell_count)
        : path(sink_path), cells(cell_count) {
        TORCH_CHECK(!path.empty(), "external event sink path must not be empty");
        TORCH_CHECK(cells > 0, "external event sink cell count must be positive");
        archived_rows.assign(static_cast<std::size_t>(cells), 0);
        open_or_create();
    }

    static void write_header(std::ofstream& stream, const int64_t cell_count) {
        constexpr char magic[8] = {'E', 'I', 'E', 'L', 'E', 'V', 'T', '1'};
        const uint32_t version = 1;
        const uint32_t cells = cell_count > 0
            ? static_cast<uint32_t>(cell_count)
            : 0U;
        const uint32_t columns = static_cast<uint32_t>(kEventColumnCountV1);
        const uint32_t row_bytes = static_cast<uint32_t>(kEventRowBytesV1);
        stream.write(magic, sizeof(magic));
        stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
        stream.write(reinterpret_cast<const char*>(&cells), sizeof(cells));
        stream.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
        stream.write(reinterpret_cast<const char*>(&row_bytes), sizeof(row_bytes));
        TORCH_CHECK(stream.good(), "cannot create external event sink header");
    }

    void open_or_create() {
        const std::filesystem::path target(path);
        if (target.has_parent_path()) {
            std::error_code error;
            std::filesystem::create_directories(target.parent_path(), error);
            TORCH_CHECK(
                !error,
                "cannot create external event sink parent directory: ",
                error.message()
            );
        }
        std::ifstream input(path, std::ios::binary);
        if (!input.good()) {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            TORCH_CHECK(output.good(), "cannot create external event sink: ", path);
            write_header(output, 0);
            return;
        }
        char magic[8]{};
        uint32_t version = 0;
        uint32_t header_cells = 0;
        uint32_t columns = 0;
        uint32_t row_bytes = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        input.read(reinterpret_cast<char*>(&header_cells), sizeof(header_cells));
        input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
        input.read(reinterpret_cast<char*>(&row_bytes), sizeof(row_bytes));
        TORCH_CHECK(input.good(), "external event sink header is truncated");
        constexpr char expected_magic[8] = {'E', 'I', 'E', 'L', 'E', 'V', 'T', '1'};
        TORCH_CHECK(
            std::equal(std::begin(magic), std::end(magic), std::begin(expected_magic)),
            "external event sink magic is invalid"
        );
        TORCH_CHECK(version == 1, "external event sink version is unsupported");
        TORCH_CHECK(
            columns == kEventColumnCountV1 && row_bytes == kEventRowBytesV1,
            "external event sink column contract is invalid"
        );
        TORCH_CHECK(
            header_cells == 0 || header_cells == static_cast<uint32_t>(cells),
            "external event sink cell count changed"
        );
        while (true) {
            uint32_t cell = 0;
            uint32_t reserved = 0;
            uint64_t rows = 0;
            input.read(reinterpret_cast<char*>(&cell), sizeof(cell));
            if (input.eof()) {
                break;
            }
            TORCH_CHECK(input.good(), "external event sink record header is truncated");
            input.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
            input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            TORCH_CHECK(input.good(), "external event sink record header is truncated");
            (void) reserved;
            TORCH_CHECK(cell < static_cast<uint32_t>(cells), "external event sink cell is invalid");
            TORCH_CHECK(
                rows <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
                    / static_cast<uint64_t>(kEventRowBytesV1),
                "external event sink row count overflows byte accounting"
            );
            const auto payload = static_cast<std::streamoff>(
                rows * static_cast<uint64_t>(kEventRowBytesV1)
            );
            input.seekg(payload, std::ios::cur);
            TORCH_CHECK(input.good(), "external event sink record payload is truncated");
            archived_rows[static_cast<std::size_t>(cell)] = checked_add_rows(
                archived_rows[static_cast<std::size_t>(cell)],
                static_cast<int64_t>(rows)
            );
            archived_bytes = checked_add_bytes(
                archived_bytes,
                static_cast<int64_t>(rows) * kEventRowBytesV1
            );
        }
    }

    static int64_t checked_add_rows(const int64_t left, const int64_t right) {
        TORCH_CHECK(right >= 0 && left <= std::numeric_limits<int64_t>::max() - right,
                    "external event sink row count overflow");
        return left + right;
    }

    static int64_t checked_add_bytes(const int64_t left, const int64_t right) {
        TORCH_CHECK(right >= 0 && left <= std::numeric_limits<int64_t>::max() - right,
                    "external event sink byte count overflow");
        return left + right;
    }

    bool last_record_matches(
        const std::vector<torch::Tensor>& columns,
        const int64_t cell,
        const int64_t expected_rows
    ) const {
        TORCH_CHECK(
            columns.size() == kEventColumnCountV1,
            "external event sink retry requires 22 columns"
        );
        std::ifstream input(path, std::ios::binary);
        TORCH_CHECK(input.good(), "cannot inspect external event sink: ", path);
        input.seekg(0, std::ios::beg);
        char magic[8]{};
        uint32_t version = 0;
        uint32_t header_cells = 0;
        uint32_t column_count = 0;
        uint32_t row_bytes = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        input.read(reinterpret_cast<char*>(&header_cells), sizeof(header_cells));
        input.read(reinterpret_cast<char*>(&column_count), sizeof(column_count));
        input.read(reinterpret_cast<char*>(&row_bytes), sizeof(row_bytes));
        TORCH_CHECK(
            input.good()
                && version == 1
                && column_count == kEventColumnCountV1
                && row_bytes == kEventRowBytesV1,
            "external event sink retry header is invalid"
        );
        std::streampos last_payload = std::streampos(-1);
        int64_t last_rows = 0;
        while (true) {
            uint32_t record_cell = 0;
            uint32_t reserved = 0;
            uint64_t record_rows = 0;
            input.read(reinterpret_cast<char*>(&record_cell), sizeof(record_cell));
            if (input.eof()) {
                break;
            }
            TORCH_CHECK(input.good(), "external event sink retry record is truncated");
            input.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
            input.read(reinterpret_cast<char*>(&record_rows), sizeof(record_rows));
            TORCH_CHECK(
                input.good() && reserved == 0,
                "external event sink retry record header is invalid"
            );
            const auto payload = static_cast<std::streamoff>(
                record_rows * static_cast<uint64_t>(kEventRowBytesV1)
            );
            const auto payload_start = input.tellg();
            if (record_cell == static_cast<uint32_t>(cell)) {
                last_payload = payload_start;
                last_rows = static_cast<int64_t>(record_rows);
            }
            input.seekg(payload, std::ios::cur);
            TORCH_CHECK(
                input.good(),
                "external event sink retry payload is truncated"
            );
        }
        if (last_payload == std::streampos(-1)) {
            return expected_rows == 0;
        }
        if (last_rows != expected_rows) {
            return false;
        }
        std::streamoff column_offset = static_cast<std::streamoff>(last_payload);
        for (std::size_t index = 0; index < columns.size(); ++index) {
            const auto& column = columns[index];
            auto expected = column[cell]
                .narrow(0, 0, expected_rows)
                .detach()
                .cpu()
                .contiguous();
            const auto byte_count = static_cast<std::streamsize>(
                expected.numel() * static_cast<int64_t>(expected.element_size())
            );
            std::vector<char> actual(static_cast<std::size_t>(byte_count));
            input.seekg(column_offset, std::ios::beg);
            input.read(actual.data(), byte_count);
            if (!input.good()) {
                return false;
            }
            if (
                byte_count > 0
                && std::memcmp(
                    actual.data(),
                    expected.data_ptr(),
                    static_cast<std::size_t>(byte_count)
                ) != 0
            ) {
                return false;
            }
            column_offset += byte_count;
        }
        return true;
    }
};

inline int64_t checked_event_payload_bytes_v1(
    const std::vector<int64_t>& counts
) {
    int64_t rows = 0;
    for (const int64_t count : counts) {
        TORCH_CHECK(count >= 0, "event row count must be nonnegative");
        TORCH_CHECK(
            count <= std::numeric_limits<int64_t>::max() - rows,
            "event row count overflow"
        );
        rows += count;
    }
    TORCH_CHECK(
        rows <= std::numeric_limits<int64_t>::max() / kEventRowBytesV1,
        "event payload byte count overflow"
    );
    return rows * kEventRowBytesV1;
}

inline std::vector<int64_t> host_int64_values_v1(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
    auto host = tensor.detach().cpu().contiguous();
    TORCH_CHECK(host.scalar_type() == torch::kInt64, name, " must be int64");
    std::vector<int64_t> values(static_cast<std::size_t>(host.numel()));
    const auto* source = host.data_ptr<int64_t>();
    std::copy(source, source + host.numel(), values.begin());
    return values;
}

inline std::vector<int32_t> host_int32_values_v1(
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
    auto host = tensor.detach().cpu().contiguous();
    TORCH_CHECK(host.scalar_type() == torch::kInt32, name, " must be int32");
    std::vector<int32_t> values(static_cast<std::size_t>(host.numel()));
    const auto* source = host.data_ptr<int32_t>();
    std::copy(source, source + host.numel(), values.begin());
    return values;
}

inline std::pair<int64_t, int64_t> validate_event_bank_v1(
    const std::vector<torch::Tensor>& columns,
    const torch::Tensor& active_count,
    const torch::Tensor& sequence_base,
    const torch::Tensor& status
) {
    TORCH_CHECK(
        columns.size() == kEventColumnCountV1,
        "resident event bank must contain 22 columns"
    );
    TORCH_CHECK(columns.front().defined(), "resident event columns must be defined");
    TORCH_CHECK(columns.front().dim() == 2, "resident event columns must be batched rank-2");
    const int64_t cells = columns.front().size(0);
    const int64_t capacity = columns.front().size(1);
    TORCH_CHECK(cells > 0 && capacity > 0, "resident event dimensions must be positive");
    const auto device = columns.front().device();
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const auto& column = columns[index];
        TORCH_CHECK(column.defined(), "resident event column must be defined");
        TORCH_CHECK(
            column.scalar_type() == kEventColumnDtypesV1[index],
            "resident event column has the wrong dtype"
        );
        TORCH_CHECK(
            column.sizes() == torch::IntArrayRef({cells, capacity}),
            "resident event columns must share shape [cells,capacity]"
        );
        TORCH_CHECK(column.is_contiguous(), "resident event columns must be contiguous");
        TORCH_CHECK(column.device() == device, "resident event columns must share one device");
        TORCH_CHECK(!column.requires_grad(), "resident event columns must not require gradients");
    }
    const auto validate_counter = [cells, device](
                                      const torch::Tensor& tensor,
                                      const torch::ScalarType dtype,
                                      const char* name
                                  ) {
        TORCH_CHECK(tensor.defined(), name, " must be defined");
        TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
        TORCH_CHECK(tensor.sizes() == torch::IntArrayRef({cells}), name, " must have shape [cells]");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.device() == device, name, " must share the event device");
        TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
    };
    validate_counter(active_count, torch::kInt64, "event active_count");
    validate_counter(sequence_base, torch::kInt64, "event sequence_base");
    validate_counter(status, torch::kInt32, "event status");
    return {cells, capacity};
}

inline EventDrainReceiptV1 drain_event_window_v1(
    const std::vector<torch::Tensor>& columns,
    torch::Tensor active_count,
    torch::Tensor sequence_base,
    const torch::Tensor& status,
    EventArchiveV1& archive,
    const int64_t budget_bytes = kHostEventArchiveBudgetBytesV1
) {
    const auto [cells, capacity] = validate_event_bank_v1(
        columns,
        active_count,
        sequence_base,
        status
    );
    const auto counts = host_int64_values_v1(active_count, "event active_count");
    const auto bases = host_int64_values_v1(sequence_base, "event sequence_base");
    const auto statuses = host_int32_values_v1(status, "event status");
    TORCH_CHECK(budget_bytes > 0, "host event archive budget must be positive");
    TORCH_CHECK(
        archive.cells.empty() || static_cast<int64_t>(archive.cells.size()) == cells,
        "resident event archive cell count changed"
    );
    if (archive.cells.empty()) {
        archive.cells.resize(static_cast<std::size_t>(cells));
        archive.archived_rows.assign(static_cast<std::size_t>(cells), 0);
    }
    TORCH_CHECK(
        archive.archived_bytes >= 0 && archive.archived_bytes <= budget_bytes,
        "host event archive byte accounting is contradictory"
    );
    const int64_t incoming_bytes = checked_event_payload_bytes_v1(counts);
    TORCH_CHECK(
        incoming_bytes <= budget_bytes - archive.archived_bytes,
        "host event archive budget exceeded before drain"
    );

    std::vector<EventChunkV1> staged(static_cast<std::size_t>(cells));
    std::vector<int64_t> bases_after(static_cast<std::size_t>(cells));
    int64_t copied_bytes = 0;
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto position = static_cast<std::size_t>(cell);
        TORCH_CHECK(counts[position] >= 0 && counts[position] <= capacity, "event active_count is outside physical capacity");
        TORCH_CHECK(bases[position] >= 0, "event sequence_base must be nonnegative");
        TORCH_CHECK(statuses[position] == 0, "event status must be zero before drain");
        TORCH_CHECK(archive.archived_rows[position] == bases[position], "event archive row count disagrees with sequence_base");
        TORCH_CHECK(
            bases[position] <= std::numeric_limits<int64_t>::max() - counts[position],
            "event sequence_base overflow"
        );
        auto& chunk = staged[position];
        chunk.reserve(kEventColumnCountV1);
        for (const auto& column : columns) {
            auto value = column[cell]
                .narrow(0, 0, counts[position])
                .detach()
                .cpu()
                .clone()
                .contiguous();
            copied_bytes += value.numel() * static_cast<int64_t>(value.element_size());
            chunk.push_back(std::move(value));
        }
        const auto& sequence = chunk.at(kTypedSequenceColumnV1);
        const auto* sequence_values = sequence.data_ptr<int64_t>();
        for (int64_t row = 0; row < counts[position]; ++row) {
            TORCH_CHECK(
                sequence_values[row] == bases[position] + row,
                "active typed event sequence is not contiguous from sequence_base"
            );
        }
        bases_after[position] = bases[position] + counts[position];
    }

    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto position = static_cast<std::size_t>(cell);
        if (counts[position] > 0) {
            archive.cells[position].push_back(std::move(staged[position]));
        }
        archive.archived_rows[position] = bases_after[position];
    }
    TORCH_CHECK(
        copied_bytes == incoming_bytes,
        "host event archive copied-byte accounting mismatch"
    );
    archive.archived_bytes += copied_bytes;
    sequence_base.add_(active_count);
    active_count.zero_();
    return {
        counts,
        bases,
        bases_after,
        copied_bytes,
        archive.archived_bytes,
        budget_bytes - archive.archived_bytes,
    };
}

inline EventDrainReceiptV1 drain_event_window_to_sink_v1(
    const std::vector<torch::Tensor>& columns,
    torch::Tensor active_count,
    torch::Tensor sequence_base,
    const torch::Tensor& status,
    EventSinkV1& sink
) {
    const auto [cells, capacity] = validate_event_bank_v1(
        columns,
        active_count,
        sequence_base,
        status
    );
    TORCH_CHECK(cells == sink.cells, "external event sink cell count disagrees");
    const auto counts = host_int64_values_v1(active_count, "event active_count");
    const auto bases = host_int64_values_v1(sequence_base, "event sequence_base");
    const auto statuses = host_int32_values_v1(status, "event status");
    std::vector<int64_t> bases_after(static_cast<std::size_t>(cells));
    int64_t maximum_count = 0;
    bool append_required = false;
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto position = static_cast<std::size_t>(cell);
        TORCH_CHECK(counts[position] >= 0 && counts[position] <= capacity,
                    "event active_count is outside physical capacity");
        TORCH_CHECK(bases[position] >= 0, "event sequence_base must be nonnegative");
        TORCH_CHECK(statuses[position] == 0, "event status must be zero before drain");
        TORCH_CHECK(
            bases[position] <= std::numeric_limits<int64_t>::max() - counts[position],
            "event sequence_base overflow"
        );
        const int64_t expected_frontier = bases[position] + counts[position];
        if (counts[position] == 0) {
            TORCH_CHECK(
                sink.archived_rows[position] == bases[position],
                "external event sink row count disagrees with sequence_base"
            );
            bases_after[position] = bases[position];
        } else if (sink.archived_rows[position] == expected_frontier) {
            TORCH_CHECK(
                sink.last_record_matches(columns, cell, counts[position]),
                "external event sink same-frontier retry does not match its durable record"
            );
            bases_after[position] = expected_frontier;
        } else {
            TORCH_CHECK(
                sink.archived_rows[position] == bases[position],
                "external event sink row count disagrees with sequence_base"
            );
            append_required = true;
            maximum_count = std::max(maximum_count, counts[position]);
        }
    }

    // One bounded D2H copy per column replaces one copy per cell and column.
    // Only the active common prefix is transferred; unused ring capacity never
    // enters host memory.
    std::vector<torch::Tensor> staged_columns;
    if (maximum_count > 0) {
        staged_columns.reserve(kEventColumnCountV1);
        for (const auto& column : columns) {
            staged_columns.push_back(
                column.narrow(1, 0, maximum_count)
                    .detach()
                    .cpu()
                    .contiguous()
            );
        }
    }

    int64_t copied_bytes = 0;
    std::ofstream output;
    if (append_required) {
        output.open(sink.path, std::ios::binary | std::ios::app);
        TORCH_CHECK(output.good(), "cannot append external event sink: ", sink.path);
    }
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto position = static_cast<std::size_t>(cell);
        const int64_t count = counts[position];
        const int64_t expected_frontier = bases[position] + count;
        if (count == 0 || sink.archived_rows[position] == expected_frontier) {
            continue;
        }
        const auto sequence = staged_columns.at(kTypedSequenceColumnV1)[cell];
        const auto* sequence_values = sequence.data_ptr<int64_t>();
        for (int64_t row = 0; row < count; ++row) {
            TORCH_CHECK(
                sequence_values[row] == bases[position] + row,
                "active typed event sequence is not contiguous from sequence_base"
            );
        }
        const uint32_t cell_value = static_cast<uint32_t>(cell);
        const uint32_t reserved = 0;
        const uint64_t row_value = static_cast<uint64_t>(count);
        output.write(reinterpret_cast<const char*>(&cell_value), sizeof(cell_value));
        output.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
        output.write(reinterpret_cast<const char*>(&row_value), sizeof(row_value));
        for (const auto& staged : staged_columns) {
            const auto value = staged[cell].narrow(0, 0, count);
            const int64_t byte_count =
                value.numel() * static_cast<int64_t>(value.element_size());
            copied_bytes = EventSinkV1::checked_add_bytes(
                copied_bytes, byte_count
            );
            output.write(
                reinterpret_cast<const char*>(value.data_ptr()),
                static_cast<std::streamsize>(byte_count)
            );
        }
        bases_after[position] = expected_frontier;
    }
    if (append_required) {
        output.flush();
        TORCH_CHECK(output.good(), "external event sink append failed: ", sink.path);
        output.close();
        TORCH_CHECK(output.good(), "external event sink close failed: ", sink.path);
        const int durable_fd = ::open(sink.path.c_str(), O_WRONLY | O_CLOEXEC);
        TORCH_CHECK(
            durable_fd >= 0,
            "cannot reopen external event sink for durability: ", sink.path
        );
        const int sync_status = ::fsync(durable_fd);
        const int close_status = ::close(durable_fd);
        TORCH_CHECK(
            sync_status == 0 && close_status == 0,
            "external event sink durability acknowledgement failed: ", sink.path
        );
        for (int64_t cell = 0; cell < cells; ++cell) {
            sink.archived_rows[static_cast<std::size_t>(cell)] =
                bases_after[static_cast<std::size_t>(cell)];
        }
    }
    sink.archived_bytes = EventSinkV1::checked_add_bytes(
        sink.archived_bytes,
        copied_bytes
    );
    sequence_base.add_(active_count);
    active_count.zero_();
    return {
        counts,
        bases,
        bases_after,
        copied_bytes,
        sink.archived_bytes,
        -1,
    };
}

inline std::vector<EventChunkV1> materialize_event_history_v1(
    const std::vector<torch::Tensor>& columns,
    const torch::Tensor& active_count,
    const torch::Tensor& sequence_base,
    const torch::Tensor& status,
    const EventArchiveV1& archive,
    const int64_t budget_bytes = kHostEventArchiveBudgetBytesV1
) {
    const auto [cells, capacity] = validate_event_bank_v1(
        columns,
        active_count,
        sequence_base,
        status
    );
    (void) capacity;
    const auto counts = host_int64_values_v1(active_count, "event active_count");
    const auto bases = host_int64_values_v1(sequence_base, "event sequence_base");
    const auto statuses = host_int32_values_v1(status, "event status");
    TORCH_CHECK(budget_bytes > 0, "host event archive budget must be positive");
    TORCH_CHECK(
        archive.archived_bytes >= 0 && archive.archived_bytes <= budget_bytes,
        "host event archive byte accounting is contradictory"
    );
    const int64_t active_bytes = checked_event_payload_bytes_v1(counts);
    TORCH_CHECK(
        active_bytes <= budget_bytes - archive.archived_bytes,
        "host event archive budget exceeded before materialization"
    );
    TORCH_CHECK(static_cast<int64_t>(archive.cells.size()) == cells, "event archive cell count is contradictory");
    std::vector<EventChunkV1> histories(static_cast<std::size_t>(cells));
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto position = static_cast<std::size_t>(cell);
        TORCH_CHECK(statuses[position] == 0, "event status must be zero before materialization");
        TORCH_CHECK(archive.archived_rows[position] == bases[position], "event archive row count disagrees with sequence_base");
        auto& history = histories[position];
        history.reserve(kEventColumnCountV1);
        for (std::size_t column_index = 0; column_index < kEventColumnCountV1; ++column_index) {
            std::vector<torch::Tensor> parts;
            for (const auto& chunk : archive.cells[position]) {
                parts.push_back(chunk.at(column_index));
            }
            if (counts[position] > 0) {
                parts.push_back(
                    columns[column_index][cell]
                        .narrow(0, 0, counts[position])
                        .detach()
                        .cpu()
                        .clone()
                        .contiguous()
                );
            }
            history.push_back(
                parts.empty()
                ? torch::empty(
                    {0},
                    torch::TensorOptions()
                        .dtype(kEventColumnDtypesV1[column_index])
                        .device(torch::kCPU)
                )
                : torch::cat(parts).contiguous()
            );
        }
        const auto total_rows = bases[position] + counts[position];
        TORCH_CHECK(
            history.at(kTypedSequenceColumnV1).numel() == total_rows,
            "materialized event history row count is contradictory"
        );
        const auto* sequence = history.at(kTypedSequenceColumnV1).data_ptr<int64_t>();
        for (int64_t row = 0; row < total_rows; ++row) {
            TORCH_CHECK(sequence[row] == row, "materialized typed event history sequence is not exact");
        }
    }
    return histories;
}

inline pybind11::dict drain_event_window_once_v1(
    const std::vector<torch::Tensor>& columns,
    torch::Tensor active_count,
    torch::Tensor sequence_base,
    const torch::Tensor& status
) {
    EventArchiveV1 archive;
    const auto receipt = drain_event_window_v1(
        columns,
        active_count,
        sequence_base,
        status,
        archive
    );
    pybind11::list chunks;
    for (const auto& cell : archive.cells) {
        pybind11::list cell_chunks;
        for (const auto& chunk : cell) {
            pybind11::list columns_out;
            for (const auto& column : chunk) {
                columns_out.append(column);
            }
            cell_chunks.append(std::move(columns_out));
        }
        chunks.append(std::move(cell_chunks));
    }
    pybind11::dict out;
    out["schema_version"] = "eiel_cuda_architecture_b_resident_event_archive_v1";
    out["event_chunks"] = std::move(chunks);
    out["active_count_before"] = receipt.active_count_before;
    out["event_sequence_base_before"] = receipt.sequence_base_before;
    out["event_sequence_base_after"] = receipt.sequence_base_after;
    out["host_archive_bytes_appended"] = receipt.copied_bytes;
    out["host_archive_cumulative_bytes"] = receipt.cumulative_archive_bytes;
    out["host_archive_remaining_budget_bytes"] = receipt.remaining_budget_bytes;
    out["host_archive_budget_bytes"] = kHostEventArchiveBudgetBytesV1;
    out["source_columns_preserved"] = true;
    return out;
}

inline pybind11::dict drain_event_window_once_with_budget_v1(
    const std::vector<torch::Tensor>& columns,
    torch::Tensor active_count,
    torch::Tensor sequence_base,
    const torch::Tensor& status,
    const int64_t budget_bytes
) {
    EventArchiveV1 archive;
    const auto receipt = drain_event_window_v1(
        columns,
        active_count,
        sequence_base,
        status,
        archive,
        budget_bytes
    );
    pybind11::dict out;
    out["schema_version"] = "eiel_cuda_architecture_b_resident_event_archive_v1";
    out["active_count_before"] = receipt.active_count_before;
    out["event_sequence_base_before"] = receipt.sequence_base_before;
    out["event_sequence_base_after"] = receipt.sequence_base_after;
    out["host_archive_bytes_appended"] = receipt.copied_bytes;
    out["host_archive_cumulative_bytes"] = receipt.cumulative_archive_bytes;
    out["host_archive_remaining_budget_bytes"] = receipt.remaining_budget_bytes;
    out["host_archive_budget_bytes"] = budget_bytes;
    out["source_columns_preserved"] = true;
    return out;
}

}  // namespace eiel::cuda_architecture_b::resident_event_archive_v1
