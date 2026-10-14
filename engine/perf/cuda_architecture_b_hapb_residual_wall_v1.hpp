#ifndef EIEL_CUDA_ARCHITECTURE_B_HAPB_RESIDUAL_WALL_V1_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_HAPB_RESIDUAL_WALL_V1_HPP_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>

namespace eiel::cuda_architecture_b::hapb_residual_wall_v1 {

inline std::atomic<std::int64_t> classify_ns{0};
inline std::atomic<std::int64_t> gather_ns{0};
inline std::atomic<std::int64_t> h2d_ns{0};
inline std::atomic<std::int64_t> compact_and_sync_ns{0};
inline std::atomic<std::int64_t> refill_total_ns{0};
inline std::atomic<std::int64_t> settlement_ns{0};
inline std::atomic<std::int64_t> prepare_sync_ns{0};
inline std::atomic<std::int64_t> resume_launch_ns{0};
inline std::atomic<std::int64_t> birth_prepare_ns{0};
inline std::atomic<std::int64_t> birth_host_ns{0};
inline std::atomic<std::int64_t> birth_settle_ns{0};

inline void add_ns(std::atomic<std::int64_t>& slot, const std::int64_t ns) {
    slot.fetch_add(ns, std::memory_order_relaxed);
}

class ScopeTimerV1 {
public:
    explicit ScopeTimerV1(std::atomic<std::int64_t>& slot)
        : slot_(slot), start_(std::chrono::steady_clock::now()) {}
    ~ScopeTimerV1() {
        add_ns(
            slot_,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start_
            ).count()
        );
    }
    ScopeTimerV1(const ScopeTimerV1&) = delete;
    ScopeTimerV1& operator=(const ScopeTimerV1&) = delete;

private:
    std::atomic<std::int64_t>& slot_;
    std::chrono::steady_clock::time_point start_;
};

inline std::map<std::string, std::int64_t> snapshot_v1() {
    return {
        {"classify_ns", classify_ns.load()},
        {"gather_ns", gather_ns.load()},
        {"h2d_ns", h2d_ns.load()},
        {"compact_and_sync_ns", compact_and_sync_ns.load()},
        {"refill_total_ns", refill_total_ns.load()},
        {"settlement_ns", settlement_ns.load()},
        {"prepare_sync_ns", prepare_sync_ns.load()},
        {"resume_launch_ns", resume_launch_ns.load()},
        {"birth_prepare_ns", birth_prepare_ns.load()},
        {"birth_host_ns", birth_host_ns.load()},
        {"birth_settle_ns", birth_settle_ns.load()},
    };
}

inline void reset_v1() {
    classify_ns.store(0);
    gather_ns.store(0);
    h2d_ns.store(0);
    compact_and_sync_ns.store(0);
    refill_total_ns.store(0);
    settlement_ns.store(0);
    prepare_sync_ns.store(0);
    resume_launch_ns.store(0);
    birth_prepare_ns.store(0);
    birth_host_ns.store(0);
    birth_settle_ns.store(0);
}

}  // namespace eiel::cuda_architecture_b::hapb_residual_wall_v1

#endif
