#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define EIEL_INHERITANCE_HD __host__ __device__
#else
#define EIEL_INHERITANCE_HD
#endif

namespace eiel::inheritance_linkage_v1 {

inline constexpr const char* kRegistryVersion =
    "eiel_inheritance_registry_v1";
inline constexpr const char* kLegacyProfileId = "legacy_midpoint_v1";
inline constexpr const char* kLinkedProfileId = "linked_parent_copy_v1";
inline constexpr std::uint32_t kLinkageGroupCount = 7;
inline constexpr std::uint32_t kLegacyAddedDonorDrawCount = 0;
inline constexpr std::uint32_t kLinkedDonorDrawCount = 7;
inline constexpr std::uint8_t kNoDonorOrdinal = 0xff;

enum class InheritanceModeV1 : std::uint8_t {
    legacy_midpoint_v1 = 0,
    linked_parent_copy_v1 = 1,
};

enum class LegacyDispatchV1 : std::uint8_t {
    midpoint_scalar = 0,
    existing_mixed_controller = 1,
    existing_record_recombination = 2,
};

enum class SelectionKindV1 : std::uint8_t {
    legacy_dispatch = 0,
    parent_donor = 1,
};

enum class SelectionStatusV1 : std::uint8_t {
    ok = 0,
    invalid_mode = 1,
    invalid_donor_budget = 2,
    invalid_donor_bit = 3,
    invalid_linkage_group = 4,
    unsupported_legacy_field_operator = 5,
};

struct LinkageSelectionV1 {
    SelectionStatusV1 status;
    SelectionKindV1 kind;
    std::uint8_t donor_ordinal;
    std::uint32_t consumed_donor_draws;
};

template <typename T>
struct ScalarSelectionV1 {
    SelectionStatusV1 status;
    T value;
    std::uint8_t donor_ordinal;
    std::uint32_t consumed_donor_draws;
};

struct LinkageGroupDescriptorV1 {
    const char* block_id;
    const char* const* fields;
    std::uint32_t field_count;
    LegacyDispatchV1 legacy_dispatch;
};

inline constexpr const char* kBehavioralStrategyFields[] = {
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
};
inline constexpr const char* kRegulationPredictionFields[] = {
    "drive_inertia",
    "drive_noise",
    "drive_logit_gain",
    "pred_lr",
    "drive_sp_hunger",
    "drive_sp_pain",
    "drive_sp_fatigue",
    "drive_sp_fear",
    "exp_prior_strength",
    "exp_prior_noise",
    "drive_birth_noise",
};
inline constexpr const char* kLifeHistoryFields[] = {
    "reproduction_threshold",
};
inline constexpr const char* kMutationRateFields[] = {
    "mutation_rate",
};
inline constexpr const char* kMemoryControllerArchitectureFields[] = {
    "hidden_size",
    "memory_dim",
    "use_gru",
    "policy_controller",
};
inline constexpr const char* kResourceHandlingFields[] = {
    "resource_handling_axis",
};
inline constexpr const char* kFamilyHeadStrategyFields[] = {
    "family_head_residuals",
};

inline constexpr LinkageGroupDescriptorV1 kLinkageGroups[] = {
    {
        "behavioral_strategy_v1",
        kBehavioralStrategyFields,
        3,
        LegacyDispatchV1::midpoint_scalar,
    },
    {
        "regulation_prediction_v1",
        kRegulationPredictionFields,
        11,
        LegacyDispatchV1::midpoint_scalar,
    },
    {
        "life_history_v1",
        kLifeHistoryFields,
        1,
        LegacyDispatchV1::midpoint_scalar,
    },
    {
        "mutation_rate_v1",
        kMutationRateFields,
        1,
        LegacyDispatchV1::midpoint_scalar,
    },
    {
        "memory_controller_architecture_v1",
        kMemoryControllerArchitectureFields,
        4,
        LegacyDispatchV1::existing_mixed_controller,
    },
    {
        "resource_handling_v1",
        kResourceHandlingFields,
        1,
        LegacyDispatchV1::midpoint_scalar,
    },
    {
        "family_head_strategy_v1",
        kFamilyHeadStrategyFields,
        1,
        LegacyDispatchV1::existing_record_recombination,
    },
};

static_assert(
    sizeof(kLinkageGroups) / sizeof(kLinkageGroups[0]) ==
    kLinkageGroupCount);
static_assert(kLinkedDonorDrawCount == kLinkageGroupCount);

EIEL_INHERITANCE_HD constexpr std::uint32_t required_donor_draws_v1(
    InheritanceModeV1 mode) {
    switch (mode) {
        case InheritanceModeV1::legacy_midpoint_v1:
            return kLegacyAddedDonorDrawCount;
        case InheritanceModeV1::linked_parent_copy_v1:
            return kLinkedDonorDrawCount;
    }
    return static_cast<std::uint32_t>(-1);
}

EIEL_INHERITANCE_HD constexpr SelectionStatusV1
validate_donor_schedule_v1(
    InheritanceModeV1 mode,
    const std::uint8_t* donor_bits,
    std::uint32_t donor_count) {
    const auto required = required_donor_draws_v1(mode);
    if (required == static_cast<std::uint32_t>(-1)) {
        return SelectionStatusV1::invalid_mode;
    }
    if (donor_count != required) {
        return SelectionStatusV1::invalid_donor_budget;
    }
    if (required == 0) {
        return SelectionStatusV1::ok;
    }
    if (donor_bits == nullptr) {
        return SelectionStatusV1::invalid_donor_budget;
    }
    for (std::uint32_t index = 0; index < required; ++index) {
        if (donor_bits[index] > 1) {
            return SelectionStatusV1::invalid_donor_bit;
        }
    }
    return SelectionStatusV1::ok;
}

EIEL_INHERITANCE_HD constexpr LegacyDispatchV1 legacy_dispatch_v1(
    std::uint32_t linkage_group) {
    switch (linkage_group) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 5:
            return LegacyDispatchV1::midpoint_scalar;
        case 4:
            return LegacyDispatchV1::existing_mixed_controller;
        case 6:
            return LegacyDispatchV1::existing_record_recombination;
    }
    return LegacyDispatchV1::existing_mixed_controller;
}

EIEL_INHERITANCE_HD constexpr LinkageSelectionV1 select_linkage_v1(
    InheritanceModeV1 mode,
    std::uint32_t linkage_group,
    const std::uint8_t* donor_bits,
    std::uint32_t donor_count) {
    const auto schedule_status =
        validate_donor_schedule_v1(mode, donor_bits, donor_count);
    if (schedule_status != SelectionStatusV1::ok) {
        return {
            schedule_status,
            SelectionKindV1::legacy_dispatch,
            kNoDonorOrdinal,
            0,
        };
    }
    if (linkage_group >= kLinkageGroupCount) {
        return {
            SelectionStatusV1::invalid_linkage_group,
            SelectionKindV1::legacy_dispatch,
            kNoDonorOrdinal,
            0,
        };
    }
    if (mode == InheritanceModeV1::legacy_midpoint_v1) {
        return {
            SelectionStatusV1::ok,
            SelectionKindV1::legacy_dispatch,
            kNoDonorOrdinal,
            kLegacyAddedDonorDrawCount,
        };
    }
    return {
        SelectionStatusV1::ok,
        SelectionKindV1::parent_donor,
        donor_bits[linkage_group],
        kLinkedDonorDrawCount,
    };
}

template <typename T>
EIEL_INHERITANCE_HD constexpr ScalarSelectionV1<T> select_scalar_v1(
    InheritanceModeV1 mode,
    std::uint32_t linkage_group,
    const std::uint8_t* donor_bits,
    std::uint32_t donor_count,
    T canonical_parent_0,
    T canonical_parent_1) {
    static_assert(
        std::is_floating_point<T>::value,
        "select_scalar_v1 is only valid for floating-point scalar loci");
    const auto selection =
        select_linkage_v1(mode, linkage_group, donor_bits, donor_count);
    if (selection.status != SelectionStatusV1::ok) {
        return {
            selection.status,
            T{},
            kNoDonorOrdinal,
            selection.consumed_donor_draws,
        };
    }
    if (selection.kind == SelectionKindV1::parent_donor) {
        return {
            SelectionStatusV1::ok,
            selection.donor_ordinal == 0
                ? canonical_parent_0
                : canonical_parent_1,
            selection.donor_ordinal,
            selection.consumed_donor_draws,
        };
    }
    if (legacy_dispatch_v1(linkage_group) !=
        LegacyDispatchV1::midpoint_scalar) {
        return {
            SelectionStatusV1::unsupported_legacy_field_operator,
            T{},
            kNoDonorOrdinal,
            kLegacyAddedDonorDrawCount,
        };
    }
    return {
        SelectionStatusV1::ok,
        static_cast<T>(
            (canonical_parent_0 + canonical_parent_1) /
            static_cast<T>(2)),
        kNoDonorOrdinal,
        kLegacyAddedDonorDrawCount,
    };
}

inline constexpr const LinkageGroupDescriptorV1* group_descriptor_v1(
    std::uint32_t linkage_group) {
    return linkage_group < kLinkageGroupCount
        ? &kLinkageGroups[linkage_group]
        : nullptr;
}

}  // namespace eiel::inheritance_linkage_v1

#undef EIEL_INHERITANCE_HD
