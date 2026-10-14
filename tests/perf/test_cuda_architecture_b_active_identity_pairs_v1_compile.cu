#include "../../engine/perf/cuda_architecture_b_active_identity_pairs_v1.cuh"

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace identity =
    eiel::cuda_architecture_b::active_identity_pairs_v1;

static_assert(identity::kResolveFound == 0);
static_assert(identity::kResolveAbsent == 1);
static_assert(identity::kResolveInvalid == 2);
static_assert(std::is_standard_layout_v<identity::ActiveIdentityPairsV1>);
static_assert(std::is_trivially_copyable_v<identity::ActiveIdentityPairsV1>);
static_assert(identity::agent_id_fits_grid_v1(
    std::numeric_limits<int32_t>::max()
));
static_assert(!identity::agent_id_fits_grid_v1(
    static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1
));
static_assert(identity::birth_id_range_fits_grid_v1(
    std::numeric_limits<int32_t>::max(), 1
));
static_assert(!identity::birth_id_range_fits_grid_v1(
    std::numeric_limits<int32_t>::max(), 2
));

int main() {
    const int64_t ids[3] = {7, 2000000000LL, 2147483647LL};
    const int32_t slots[3] = {2, 0, 1};
    const int64_t slot_to_id[3] = {
        2000000000LL, 2147483647LL, 7,
    };
    const bool alive[3] = {true, true, true};
    const identity::ActiveIdentityPairsV1 pairs{
        ids, slots, slot_to_id, alive, 3, 3, 3,
    };
    assert(identity::valid_active_identity_pairs_v1(pairs));
    const identity::ResolveResultV1 found =
        identity::resolve_active_agent_slot_v1(pairs, 2000000000LL);
    assert(found.status == identity::kResolveFound);
    assert(found.slot == 0);
    assert(
        identity::resolve_active_agent_slot_v1(pairs, 8).status
        == identity::kResolveAbsent
    );

    const int64_t duplicate_ids[3] = {7, 7, 2147483647LL};
    const identity::ActiveIdentityPairsV1 duplicate{
        duplicate_ids, slots, slot_to_id, alive, 3, 3, 3,
    };
    assert(!identity::valid_active_identity_pairs_v1(duplicate));
    assert(
        identity::resolve_active_agent_slot_v1(duplicate, 7).status
        == identity::kResolveInvalid
    );

    const int32_t corrupt_slots[3] = {1, 0, 2};
    const identity::ActiveIdentityPairsV1 corrupt{
        ids, corrupt_slots, slot_to_id, alive, 3, 3, 3,
    };
    assert(!identity::valid_active_identity_pairs_v1(corrupt));
    assert(
        identity::resolve_active_agent_slot_v1(corrupt, 7).status
        == identity::kResolveInvalid
    );
    return 0;
}
