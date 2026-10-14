#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <unistd.h>

#include <cuda.h>
#include <cupti.h>

static_assert(CUPTI_API_VERSION == 26,
              "This collector targets the CUDA 12.8 CUPTI activity ABI");

namespace {

constexpr size_t kActivityBufferSize = 8U * 1024U * 1024U;
constexpr size_t kActivityBufferCount = 8;
constexpr size_t kDeviceBufferPoolLimit = 8;
constexpr size_t kDeviceBufferPreallocate = 3;
constexpr size_t kMaxJsonStringBytes = 64U * 1024U;

enum class InitState : uint32_t {
  kNever = 0,
  kInitializing = 1,
  kReady = 2,
  kFailed = 3,
};

struct BufferSlot {
  uint8_t *data = nullptr;
  std::atomic<bool> in_use{false};
};

struct Collector {
  std::mutex initialize_mutex;
  std::mutex output_mutex;
  std::mutex dropped_mutex;
  std::atomic<InitState> state{InitState::kNever};
  int output_fd = -1;
  const char *trace_path = nullptr;
  std::array<BufferSlot, kActivityBufferCount> buffers{};
  std::array<CUpti_ActivityKind, 8> enabled_kinds{};
  size_t enabled_kind_count = 0;

  std::atomic<uint64_t> buffers_requested{0};
  std::atomic<uint64_t> buffers_completed{0};
  std::atomic<uint64_t> buffer_requests_denied{0};
  std::atomic<uint64_t> invalid_buffer_returns{0};
  std::atomic<uint64_t> records_seen{0};
  std::atomic<uint64_t> records_serialized{0};
  std::atomic<uint64_t> driver_api_records{0};
  std::atomic<uint64_t> runtime_api_records{0};
  std::atomic<uint64_t> kernel_records{0};
  std::atomic<uint64_t> memcpy_records{0};
  std::atomic<uint64_t> memory_records{0};
  std::atomic<uint64_t> synchronization_records{0};
  std::atomic<uint64_t> cuda_event_records{0};
  std::atomic<uint64_t> memset_records{0};
  std::atomic<uint64_t> unknown_records{0};
  std::atomic<uint64_t> dropped_records{0};
  std::atomic<uint64_t> parse_errors{0};
  std::atomic<uint64_t> output_errors{0};
  std::atomic<bool> output_error_reported{false};
};

Collector g_collector;

const char *CuptiResultName(CUptiResult result) {
  const char *name = nullptr;
  if (cuptiGetResultString(result, &name) == CUPTI_SUCCESS && name != nullptr) {
    return name;
  }
  return "unknown CUPTI error";
}

void ReportCuptiError(const char *operation, CUptiResult result) {
  std::fprintf(stderr, "EIEL CUPTI collector: %s failed: %s (%d)\n",
               operation, CuptiResultName(result), static_cast<int>(result));
}

void ReportSystemError(const char *operation) {
  std::fprintf(stderr, "EIEL CUPTI collector: %s failed: %s (%d)\n",
               operation, std::strerror(errno), errno);
}

void AppendUnsigned(std::string &output, uint64_t value) {
  char buffer[32];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  output.append(buffer, result.ptr);
}

void AppendSigned(std::string &output, int64_t value) {
  char buffer[32];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  output.append(buffer, result.ptr);
}

void AppendHex(std::string &output, uint64_t value) {
  char buffer[32];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, 16);
  output.append("\"0x");
  output.append(buffer, result.ptr);
  output.push_back('"');
}

void AppendEscapedCString(std::string &output, const char *value) {
  if (value == nullptr) {
    output.append("null");
    return;
  }

  static constexpr char kHex[] = "0123456789abcdef";
  output.push_back('"');
  size_t index = 0;
  for (; index < kMaxJsonStringBytes && value[index] != '\0'; ++index) {
    const uint8_t byte = static_cast<uint8_t>(value[index]);
    switch (byte) {
      case '"':
        output.append("\\\"");
        break;
      case '\\':
        output.append("\\\\");
        break;
      case '\b':
        output.append("\\b");
        break;
      case '\f':
        output.append("\\f");
        break;
      case '\n':
        output.append("\\n");
        break;
      case '\r':
        output.append("\\r");
        break;
      case '\t':
        output.append("\\t");
        break;
      default:
        if (byte < 0x20 || byte >= 0x7f) {
          output.append("\\u00");
          output.push_back(kHex[(byte >> 4) & 0xf]);
          output.push_back(kHex[byte & 0xf]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  if (index == kMaxJsonStringBytes) {
    output.append("...[truncated]");
  }
  output.push_back('"');
}

class JsonObject {
 public:
  explicit JsonObject(std::string &output) : output_(output) {
    output_.push_back('{');
  }

  void AddString(const char *key, const char *value) {
    AddKey(key);
    AppendEscapedCString(output_, value);
  }

  void AddUnsigned(const char *key, uint64_t value) {
    AddKey(key);
    AppendUnsigned(output_, value);
  }

  void AddSigned(const char *key, int64_t value) {
    AddKey(key);
    AppendSigned(output_, value);
  }

  void AddHex(const char *key, uint64_t value) {
    AddKey(key);
    AppendHex(output_, value);
  }

  void AddBool(const char *key, bool value) {
    AddKey(key);
    output_.append(value ? "true" : "false");
  }

  void Finish() {
    output_.append("}\n");
  }

 private:
  void AddKey(const char *key) {
    if (!first_) {
      output_.push_back(',');
    }
    first_ = false;
    output_.push_back('"');
    output_.append(key);
    output_.append("\":");
  }

  std::string &output_;
  bool first_ = true;
};

bool WriteBatch(std::string_view batch) {
  if (batch.empty()) {
    return true;
  }

  std::lock_guard<std::mutex> lock(g_collector.output_mutex);
  size_t written = 0;
  while (written < batch.size()) {
    const ssize_t result =
        ::write(g_collector.output_fd, batch.data() + written,
                batch.size() - written);
    if (result > 0) {
      written += static_cast<size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }

    g_collector.output_errors.fetch_add(1, std::memory_order_relaxed);
    if (!g_collector.output_error_reported.exchange(true,
                                                    std::memory_order_relaxed)) {
      ReportSystemError("write trace output");
    }
    return false;
  }
  return true;
}

const char *MemcpyKindName(uint8_t kind) {
  switch (static_cast<CUpti_ActivityMemcpyKind>(kind)) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "host_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "device_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "host_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "array_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "array_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "array_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "device_to_array";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "device_to_device";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "host_to_host";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "peer_to_peer";
    case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

const char *MemoryKindName(uint32_t kind) {
  switch (static_cast<CUpti_ActivityMemoryKind>(kind)) {
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "device_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "managed_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

const char *MemoryOperationName(CUpti_ActivityMemoryOperationType type) {
  switch (type) {
    case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION:
      return "allocation";
    case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE:
      return "release";
    case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID:
    default:
      return "invalid";
  }
}

const char *SynchronizationTypeName(
    CUpti_ActivitySynchronizationType type) {
  switch (type) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
      return "event_synchronize";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
      return "stream_wait_event";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
      return "stream_synchronize";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
      return "context_synchronize";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN:
    default:
      return "unknown";
  }
}

void SerializeApi(const CUpti_ActivityAPI &record, std::string &batch) {
  const bool is_driver = record.kind == CUPTI_ACTIVITY_KIND_DRIVER;
  const CUpti_CallbackDomain domain =
      is_driver ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API;
  const char *api_name = nullptr;
  const CUptiResult name_result =
      cuptiGetCallbackName(domain, record.cbid, &api_name);

  JsonObject json(batch);
  json.AddString("record_type", "api");
  json.AddString("api_domain", is_driver ? "driver" : "runtime");
  json.AddString("api_name", name_result == CUPTI_SUCCESS ? api_name : nullptr);
  json.AddUnsigned("cbid", record.cbid);
  json.AddUnsigned("return_value", record.returnValue);
  json.AddUnsigned("start_ns", record.start);
  json.AddUnsigned("end_ns", record.end);
  json.AddUnsigned("process_id", record.processId);
  json.AddUnsigned("thread_id", record.threadId);
  json.AddUnsigned("correlation_id", record.correlationId);
  if (name_result != CUPTI_SUCCESS) {
    json.AddUnsigned("api_name_error", name_result);
  }
  json.Finish();

  if (is_driver) {
    g_collector.driver_api_records.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_collector.runtime_api_records.fetch_add(1, std::memory_order_relaxed);
  }
}

void SerializeKernel(const CUpti_ActivityKernel9 &record, std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "kernel");
  json.AddString("activity_kind",
                 record.kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                     ? "concurrent_kernel"
                     : "kernel");
  json.AddString("name", record.name);
  json.AddUnsigned("start_ns", record.start);
  json.AddUnsigned("end_ns", record.end);
  json.AddUnsigned("completed_ns", record.completed);
  json.AddUnsigned("queued_ns", record.queued);
  json.AddUnsigned("submitted_ns", record.submitted);
  json.AddUnsigned("device_id", record.deviceId);
  json.AddUnsigned("context_id", record.contextId);
  json.AddUnsigned("stream_id", record.streamId);
  json.AddSigned("grid_x", record.gridX);
  json.AddSigned("grid_y", record.gridY);
  json.AddSigned("grid_z", record.gridZ);
  json.AddSigned("block_x", record.blockX);
  json.AddSigned("block_y", record.blockY);
  json.AddSigned("block_z", record.blockZ);
  json.AddUnsigned("registers_per_thread", record.registersPerThread);
  json.AddSigned("static_shared_memory_bytes", record.staticSharedMemory);
  json.AddSigned("dynamic_shared_memory_bytes", record.dynamicSharedMemory);
  json.AddUnsigned("local_memory_per_thread_bytes",
                   record.localMemoryPerThread);
  json.AddUnsigned("local_memory_total_bytes", record.localMemoryTotal_v2);
  json.AddUnsigned("correlation_id", record.correlationId);
  json.AddSigned("grid_id", record.gridId);
  json.AddUnsigned("launch_type", record.launchType);
  json.AddUnsigned("graph_node_id", record.graphNodeId);
  json.AddUnsigned("graph_id", record.graphId);
  json.AddUnsigned("channel_id", record.channelID);
  json.AddUnsigned("channel_type", record.channelType);
  json.Finish();
  g_collector.kernel_records.fetch_add(1, std::memory_order_relaxed);
}

void SerializeMemcpy(const CUpti_ActivityMemcpy6 &record, std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "memcpy");
  json.AddString("direction", MemcpyKindName(record.copyKind));
  json.AddUnsigned("copy_kind", record.copyKind);
  json.AddString("source_memory_kind", MemoryKindName(record.srcKind));
  json.AddUnsigned("source_memory_kind_id", record.srcKind);
  json.AddString("destination_memory_kind", MemoryKindName(record.dstKind));
  json.AddUnsigned("destination_memory_kind_id", record.dstKind);
  json.AddUnsigned("flags", record.flags);
  json.AddUnsigned("bytes", record.bytes);
  json.AddUnsigned("start_ns", record.start);
  json.AddUnsigned("end_ns", record.end);
  json.AddUnsigned("device_id", record.deviceId);
  json.AddUnsigned("context_id", record.contextId);
  json.AddUnsigned("stream_id", record.streamId);
  json.AddUnsigned("correlation_id", record.correlationId);
  json.AddUnsigned("runtime_correlation_id", record.runtimeCorrelationId);
  json.AddUnsigned("graph_node_id", record.graphNodeId);
  json.AddUnsigned("graph_id", record.graphId);
  json.AddUnsigned("channel_id", record.channelID);
  json.AddUnsigned("channel_type", record.channelType);
  json.AddUnsigned("copy_count", record.copyCount);
  json.Finish();
  g_collector.memcpy_records.fetch_add(1, std::memory_order_relaxed);
}

void SerializeMemory(const CUpti_ActivityMemory4 &record, std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "memory");
  json.AddString("operation",
                 MemoryOperationName(record.memoryOperationType));
  json.AddUnsigned("operation_id", record.memoryOperationType);
  json.AddString("memory_kind", MemoryKindName(record.memoryKind));
  json.AddUnsigned("memory_kind_id", record.memoryKind);
  json.AddUnsigned("correlation_id", record.correlationId);
  json.AddUnsigned("address", record.address);
  json.AddHex("address_hex", record.address);
  json.AddUnsigned("bytes", record.bytes);
  json.AddUnsigned("timestamp_ns", record.timestamp);
  json.AddUnsigned("pc", record.PC);
  json.AddHex("pc_hex", record.PC);
  json.AddUnsigned("process_id", record.processId);
  json.AddUnsigned("device_id", record.deviceId);
  json.AddUnsigned("context_id", record.contextId);
  json.AddUnsigned("stream_id", record.streamId);
  json.AddString("name", record.name);
  json.AddBool("is_async", record.isAsync != 0);
  json.AddString("source", record.source);
  json.Finish();
  g_collector.memory_records.fetch_add(1, std::memory_order_relaxed);
}

void SerializeSynchronization(const CUpti_ActivitySynchronization2 &record,
                              std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "synchronization");
  json.AddString("synchronization_type",
                 SynchronizationTypeName(record.type));
  json.AddUnsigned("synchronization_type_id", record.type);
  json.AddUnsigned("start_ns", record.start);
  json.AddUnsigned("end_ns", record.end);
  json.AddUnsigned("correlation_id", record.correlationId);
  json.AddUnsigned("context_id", record.contextId);
  json.AddUnsigned("stream_id", record.streamId);
  json.AddUnsigned("cuda_event_id", record.cudaEventId);
  json.AddUnsigned("cuda_event_sync_id", record.cudaEventSyncId);
  json.AddUnsigned("return_value", record.returnValue);
  json.Finish();
  g_collector.synchronization_records.fetch_add(
      1, std::memory_order_relaxed);
}

void SerializeCudaEvent(const CUpti_ActivityCudaEvent2 &record,
                        std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "cuda_event");
  json.AddUnsigned("correlation_id", record.correlationId);
  json.AddUnsigned("context_id", record.contextId);
  json.AddUnsigned("stream_id", record.streamId);
  json.AddUnsigned("event_id", record.eventId);
  json.AddUnsigned("device_id", record.deviceId);
  json.AddUnsigned("device_timestamp_ns", record.deviceTimestamp);
  json.AddUnsigned("cuda_event_sync_id", record.cudaEventSyncId);
  json.Finish();
  g_collector.cuda_event_records.fetch_add(1, std::memory_order_relaxed);
}

void SerializeMemset(const CUpti_ActivityMemset4 &record, std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "memset");
  json.AddUnsigned("value", record.value);
  json.AddUnsigned("bytes", record.bytes);
  json.AddUnsigned("start_ns", record.start);
  json.AddUnsigned("end_ns", record.end);
  json.AddUnsigned("device_id", record.deviceId);
  json.AddUnsigned("context_id", record.contextId);
  json.AddUnsigned("stream_id", record.streamId);
  json.AddUnsigned("correlation_id", record.correlationId);
  json.AddUnsigned("flags", record.flags);
  json.AddString("memory_kind", MemoryKindName(record.memoryKind));
  json.AddUnsigned("memory_kind_id", record.memoryKind);
  json.AddUnsigned("graph_node_id", record.graphNodeId);
  json.AddUnsigned("graph_id", record.graphId);
  json.AddUnsigned("channel_id", record.channelID);
  json.AddUnsigned("channel_type", record.channelType);
  json.Finish();
  g_collector.memset_records.fetch_add(1, std::memory_order_relaxed);
}

void SerializeUnknown(const CUpti_Activity &record, size_t valid_size,
                      std::string &batch) {
  JsonObject json(batch);
  json.AddString("record_type", "collector_error");
  json.AddString("error", "unexpected activity kind");
  json.AddUnsigned("activity_kind", record.kind);
  json.AddUnsigned("buffer_valid_bytes", valid_size);
  json.Finish();
  g_collector.unknown_records.fetch_add(1, std::memory_order_relaxed);
}

void SerializeRecord(const CUpti_Activity &record, size_t valid_size,
                     std::string &batch) {
  switch (record.kind) {
    case CUPTI_ACTIVITY_KIND_DRIVER:
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      SerializeApi(reinterpret_cast<const CUpti_ActivityAPI &>(record), batch);
      break;
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      SerializeKernel(reinterpret_cast<const CUpti_ActivityKernel9 &>(record),
                      batch);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      SerializeMemcpy(reinterpret_cast<const CUpti_ActivityMemcpy6 &>(record),
                      batch);
      break;
    case CUPTI_ACTIVITY_KIND_MEMORY2:
      SerializeMemory(reinterpret_cast<const CUpti_ActivityMemory4 &>(record),
                      batch);
      break;
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
      SerializeSynchronization(
          reinterpret_cast<const CUpti_ActivitySynchronization2 &>(record),
          batch);
      break;
    case CUPTI_ACTIVITY_KIND_CUDA_EVENT:
      SerializeCudaEvent(
          reinterpret_cast<const CUpti_ActivityCudaEvent2 &>(record), batch);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      SerializeMemset(reinterpret_cast<const CUpti_ActivityMemset4 &>(record),
                      batch);
      break;
    default:
      SerializeUnknown(record, valid_size, batch);
      break;
  }
  g_collector.records_serialized.fetch_add(1, std::memory_order_relaxed);
}

bool ReleaseBuffer(uint8_t *buffer) {
  for (BufferSlot &slot : g_collector.buffers) {
    if (slot.data == buffer) {
      slot.in_use.store(false, std::memory_order_release);
      return true;
    }
  }
  g_collector.invalid_buffer_returns.fetch_add(1,
                                               std::memory_order_relaxed);
  return false;
}

void AppendDroppedRecord(std::string &batch, size_t dropped) {
  if (dropped == 0) {
    return;
  }
  g_collector.dropped_records.fetch_add(dropped, std::memory_order_relaxed);
  JsonObject json(batch);
  json.AddString("record_type", "dropped_records");
  json.AddUnsigned("count", dropped);
  json.Finish();
}

void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size,
                              size_t *max_num_records) {
  g_collector.buffers_requested.fetch_add(1, std::memory_order_relaxed);
  for (BufferSlot &slot : g_collector.buffers) {
    bool expected = false;
    if (slot.in_use.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
      *buffer = slot.data;
      *size = kActivityBufferSize;
      *max_num_records = 0;
      return;
    }
  }

  *buffer = nullptr;
  *size = 0;
  *max_num_records = 0;
  g_collector.buffer_requests_denied.fetch_add(1,
                                               std::memory_order_relaxed);
}

void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id,
                              uint8_t *buffer, size_t /*size*/,
                              size_t valid_size) {
  g_collector.buffers_completed.fetch_add(1, std::memory_order_relaxed);
  try {
    std::string batch;
    if (valid_size != 0) {
      batch.reserve(valid_size + valid_size / 2);
      CUpti_Activity *record = nullptr;
      for (;;) {
        const CUptiResult result =
            cuptiActivityGetNextRecord(buffer, valid_size, &record);
        if (result == CUPTI_SUCCESS) {
          g_collector.records_seen.fetch_add(1, std::memory_order_relaxed);
          SerializeRecord(*record, valid_size, batch);
          continue;
        }
        if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
          break;
        }

        g_collector.parse_errors.fetch_add(1, std::memory_order_relaxed);
        JsonObject json(batch);
        json.AddString("record_type", "collector_error");
        json.AddString("error", "cuptiActivityGetNextRecord failed");
        json.AddUnsigned("cupti_result", result);
        json.AddString("cupti_result_name", CuptiResultName(result));
        json.AddUnsigned("buffer_valid_bytes", valid_size);
        json.Finish();
        break;
      }
    }

    size_t dropped = 0;
    CUptiResult dropped_result;
    {
      std::lock_guard<std::mutex> lock(g_collector.dropped_mutex);
      dropped_result =
          cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);
    }
    if (dropped_result == CUPTI_SUCCESS) {
      AppendDroppedRecord(batch, dropped);
    } else {
      g_collector.parse_errors.fetch_add(1, std::memory_order_relaxed);
      JsonObject json(batch);
      json.AddString("record_type", "collector_error");
      json.AddString("error", "cuptiActivityGetNumDroppedRecords failed");
      json.AddUnsigned("cupti_result", dropped_result);
      json.AddString("cupti_result_name", CuptiResultName(dropped_result));
      json.Finish();
    }

    ReleaseBuffer(buffer);
    WriteBatch(batch);
  } catch (...) {
    g_collector.parse_errors.fetch_add(1, std::memory_order_relaxed);
    ReleaseBuffer(buffer);
    std::fprintf(stderr,
                 "EIEL CUPTI collector: exception while processing an "
                 "activity buffer\n");
  }
}

bool AllocateBufferPool() {
  for (BufferSlot &slot : g_collector.buffers) {
    void *allocation = nullptr;
    const int result =
        ::posix_memalign(&allocation, 64, kActivityBufferSize);
    if (result != 0) {
      std::fprintf(stderr,
                   "EIEL CUPTI collector: activity buffer allocation failed: "
                   "%s (%d)\n",
                   std::strerror(result), result);
      return false;
    }
    slot.data = static_cast<uint8_t *>(allocation);
  }
  return true;
}

bool SetSizeAttribute(CUpti_ActivityAttribute attribute, size_t value,
                      const char *name) {
  size_t value_size = sizeof(value);
  const CUptiResult result =
      cuptiActivitySetAttribute(attribute, &value_size, &value);
  if (result != CUPTI_SUCCESS) {
    ReportCuptiError(name, result);
    return false;
  }
  return true;
}

void RollBackActivities() {
  while (g_collector.enabled_kind_count != 0) {
    const CUpti_ActivityKind kind =
        g_collector.enabled_kinds[--g_collector.enabled_kind_count];
    const CUptiResult result = cuptiActivityDisable(kind);
    if (result != CUPTI_SUCCESS) {
      ReportCuptiError("cuptiActivityDisable during rollback", result);
    }
  }
}

bool EnableRequiredActivity(CUpti_ActivityKind kind, const char *name) {
  const CUptiResult result = cuptiActivityEnable(kind);
  if (result != CUPTI_SUCCESS) {
    ReportCuptiError(name, result);
    return false;
  }
  g_collector.enabled_kinds[g_collector.enabled_kind_count++] = kind;
  return true;
}

void AppendFinalMetadata(std::string &batch, CUptiResult flush_result,
                         CUptiResult dropped_result, size_t final_dropped) {
  JsonObject json(batch);
  json.AddString("record_type", "metadata");
  json.AddString("event", "final");
  json.AddUnsigned("cupti_api_version", CUPTI_API_VERSION);
  json.AddUnsigned("process_id", static_cast<uint64_t>(::getpid()));
  json.AddUnsigned("force_flush_result", flush_result);
  json.AddString("force_flush_result_name", CuptiResultName(flush_result));
  json.AddUnsigned("final_dropped_query_result", dropped_result);
  json.AddString("final_dropped_query_result_name",
                 CuptiResultName(dropped_result));
  json.AddUnsigned("final_dropped_records", final_dropped);
  json.AddUnsigned(
      "dropped_records_total",
      g_collector.dropped_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "buffers_requested",
      g_collector.buffers_requested.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "buffers_completed",
      g_collector.buffers_completed.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "buffer_requests_denied",
      g_collector.buffer_requests_denied.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "invalid_buffer_returns",
      g_collector.invalid_buffer_returns.load(std::memory_order_relaxed));
  json.AddUnsigned("records_seen",
                   g_collector.records_seen.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "records_serialized",
      g_collector.records_serialized.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "driver_api_records",
      g_collector.driver_api_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "runtime_api_records",
      g_collector.runtime_api_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "kernel_records",
      g_collector.kernel_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "memcpy_records",
      g_collector.memcpy_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "memory_records",
      g_collector.memory_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "synchronization_records",
      g_collector.synchronization_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "cuda_event_records",
      g_collector.cuda_event_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "memset_records",
      g_collector.memset_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "unknown_records",
      g_collector.unknown_records.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "parse_errors",
      g_collector.parse_errors.load(std::memory_order_relaxed));
  json.AddUnsigned(
      "output_errors_before_final",
      g_collector.output_errors.load(std::memory_order_relaxed));
  json.Finish();
}

void AtExitHandler() {
  if (g_collector.state.load(std::memory_order_acquire) !=
      InitState::kReady) {
    return;
  }

  const CUptiResult flush_result =
      cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (flush_result != CUPTI_SUCCESS) {
    ReportCuptiError("cuptiActivityFlushAll forced at exit", flush_result);
  }

  size_t final_dropped = 0;
  CUptiResult dropped_result;
  {
    std::lock_guard<std::mutex> lock(g_collector.dropped_mutex);
    dropped_result =
        cuptiActivityGetNumDroppedRecords(nullptr, 0, &final_dropped);
  }
  if (dropped_result != CUPTI_SUCCESS) {
    ReportCuptiError("final cuptiActivityGetNumDroppedRecords",
                     dropped_result);
  } else {
    g_collector.dropped_records.fetch_add(final_dropped,
                                          std::memory_order_relaxed);
  }

  try {
    std::string batch;
    batch.reserve(2048);
    if (final_dropped != 0) {
      JsonObject dropped_json(batch);
      dropped_json.AddString("record_type", "dropped_records");
      dropped_json.AddUnsigned("count", final_dropped);
      dropped_json.AddString("source", "final_query");
      dropped_json.Finish();
    }
    AppendFinalMetadata(batch, flush_result, dropped_result, final_dropped);
    WriteBatch(batch);
  } catch (...) {
    std::fprintf(stderr,
                 "EIEL CUPTI collector: exception while writing final metadata\n");
  }
}

bool WriteInitializationMetadata(bool memset_enabled,
                                 CUptiResult memset_result,
                                 uint32_t runtime_cupti_version) {
  std::string batch;
  batch.reserve(1024);
  JsonObject json(batch);
  json.AddString("record_type", "metadata");
  json.AddString("event", "initialize");
  json.AddString("trace_path", g_collector.trace_path);
  json.AddUnsigned("process_id", static_cast<uint64_t>(::getpid()));
  json.AddUnsigned("cupti_api_version", CUPTI_API_VERSION);
  json.AddUnsigned("cupti_runtime_version", runtime_cupti_version);
  json.AddUnsigned("host_activity_buffer_bytes", kActivityBufferSize);
  json.AddUnsigned("host_activity_buffer_count", kActivityBufferCount);
  json.AddUnsigned("device_activity_buffer_bytes", kActivityBufferSize);
  json.AddUnsigned("device_activity_buffer_pool_limit",
                   kDeviceBufferPoolLimit);
  json.AddUnsigned("device_activity_buffer_preallocate",
                   kDeviceBufferPreallocate);
  json.AddString(
      "enabled_activities",
      "DRIVER,RUNTIME,CONCURRENT_KERNEL,MEMCPY,MEMORY2,SYNCHRONIZATION,"
      "CUDA_EVENT_IMPLICIT");
  json.AddBool("memset_enabled", memset_enabled);
  json.AddUnsigned("memset_enable_result", memset_result);
  json.AddBool("overhead_enabled", false);
  json.AddString("synchronization_record_layout",
                 "CUpti_ActivitySynchronization2");
  json.Finish();
  return WriteBatch(batch);
}

bool InitializeCollector() {
  const char *path = std::getenv("EIEL_CUPTI_TRACE_PATH");
  if (path == nullptr || path[0] == '\0') {
    std::fprintf(stderr,
                 "EIEL CUPTI collector: EIEL_CUPTI_TRACE_PATH is required\n");
    return false;
  }

  g_collector.trace_path = path;
  g_collector.output_fd =
      ::open(path, O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
  if (g_collector.output_fd < 0) {
    ReportSystemError("open EIEL_CUPTI_TRACE_PATH");
    return false;
  }

  if (!AllocateBufferPool()) {
    return false;
  }
  if (std::atexit(AtExitHandler) != 0) {
    std::fprintf(stderr,
                 "EIEL CUPTI collector: failed to register exit handler\n");
    return false;
  }

  if (!SetSizeAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                        kActivityBufferSize,
                        "set CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE") ||
      !SetSizeAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                        kDeviceBufferPoolLimit,
                        "set CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT") ||
      !SetSizeAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE,
                        kDeviceBufferPreallocate,
                        "set CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE")) {
    return false;
  }

  CUptiResult result =
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
  if (result != CUPTI_SUCCESS) {
    ReportCuptiError("cuptiActivityRegisterCallbacks", result);
    return false;
  }

  const std::array<std::pair<CUpti_ActivityKind, const char *>, 6> required = {{
      {CUPTI_ACTIVITY_KIND_DRIVER, "enable DRIVER activity"},
      {CUPTI_ACTIVITY_KIND_RUNTIME, "enable RUNTIME activity"},
      {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
       "enable CONCURRENT_KERNEL activity"},
      {CUPTI_ACTIVITY_KIND_MEMCPY, "enable MEMCPY activity"},
      {CUPTI_ACTIVITY_KIND_MEMORY2, "enable MEMORY2 activity"},
      {CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
       "enable SYNCHRONIZATION activity"},
  }};
  for (const auto &entry : required) {
    if (!EnableRequiredActivity(entry.first, entry.second)) {
      RollBackActivities();
      return false;
    }
  }

  const CUptiResult memset_result =
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
  const bool memset_enabled = memset_result == CUPTI_SUCCESS;
  if (memset_enabled) {
    g_collector.enabled_kinds[g_collector.enabled_kind_count++] =
        CUPTI_ACTIVITY_KIND_MEMSET;
  } else {
    ReportCuptiError("enable optional MEMSET activity", memset_result);
  }

  uint32_t runtime_cupti_version = 0;
  result = cuptiGetVersion(&runtime_cupti_version);
  if (result != CUPTI_SUCCESS) {
    ReportCuptiError("cuptiGetVersion", result);
    RollBackActivities();
    return false;
  }
  if (runtime_cupti_version != CUPTI_API_VERSION) {
    std::fprintf(stderr,
                 "EIEL CUPTI collector: header/runtime API version mismatch: "
                 "%u != %u\n",
                 CUPTI_API_VERSION, runtime_cupti_version);
    RollBackActivities();
    return false;
  }

  if (!WriteInitializationMetadata(memset_enabled, memset_result,
                                   runtime_cupti_version)) {
    RollBackActivities();
    return false;
  }
  return true;
}

}  // namespace

extern "C" int InitializeInjection() {
  std::lock_guard<std::mutex> lock(g_collector.initialize_mutex);
  const InitState state = g_collector.state.load(std::memory_order_acquire);
  if (state == InitState::kReady) {
    return 1;
  }
  if (state == InitState::kInitializing || state == InitState::kFailed) {
    return 0;
  }

  g_collector.state.store(InitState::kInitializing,
                          std::memory_order_release);
  try {
    if (!InitializeCollector()) {
      g_collector.state.store(InitState::kFailed, std::memory_order_release);
      return 0;
    }
  } catch (...) {
    RollBackActivities();
    std::fprintf(stderr,
                 "EIEL CUPTI collector: exception during initialization\n");
    g_collector.state.store(InitState::kFailed, std::memory_order_release);
    return 0;
  }

  g_collector.state.store(InitState::kReady, std::memory_order_release);
  return 1;
}
