#pragma once

#include "cuda_architecture_b_compact_summary_host_template_renderer_v1.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1 {

namespace renderer = compact_summary_host_template_renderer_v1;

inline constexpr std::uint32_t kHostTemplateWireAbiVersionV1 = 1;
inline constexpr std::uint32_t kHostTemplateSurfaceCountV1 = 6;
inline constexpr std::uint64_t kMaximumHostTemplateCoordinatesV1 = 4096;
inline constexpr std::uint64_t kNoRuntimeRowV1 = UINT64_MAX;
inline constexpr std::uint32_t kNoRuntimeColumnV1 = UINT32_MAX;

enum HostTemplateProgramWireColumnV1 : std::uint32_t {
    kProgramAbiVersionV1 = 0,
    kProgramEncodingV1,
    kProgramSurfaceV1,
    kProgramGenerationV1,
    kProgramCellV1,
    kProgramTokenOffsetV1,
    kProgramTokenCountV1,
    kProgramPartOffsetV1,
    kProgramPartCountV1,
    kProgramConstraintOffsetV1,
    kProgramConstraintCountV1,
    kProgramLiteralUtf8BytesV1,
    kProgramMaximumRenderedUtf8BytesV1,
    kProgramWireColumnCountV1,
};

enum HostTemplateTokenWireColumnV1 : std::uint32_t {
    kTokenIndexV1 = 0,
    kTokenTypeV1,
    kTokenValidationKindV1,
    kTokenRankV1,
    kTokenRenderPackDepthV1,
    kTokenMaximumUtf8BytesV1,
    kTokenSourcePathPoolOffsetV1,
    kTokenSourcePathUtf8BytesV1,
    kTokenRuleParameterV1,
    kTokenMinimumLowV1,
    kTokenMinimumHighV1,
    kTokenMaximumLowV1,
    kTokenMaximumHighV1,
    kTokenWireColumnCountV1,
};

enum HostTemplatePartWireColumnV1 : std::uint32_t {
    kPartKindV1 = 0,
    kPartTokenIndexV1,
    kPartLiteralPoolOffsetV1,
    kPartLiteralUtf8BytesV1,
    kPartReservedZeroV1,
    kPartWireColumnCountV1,
};

enum HostTemplateConstraintWireColumnV1 : std::uint32_t {
    kConstraintKindV1 = 0,
    kConstraintLeftPathPoolOffsetV1,
    kConstraintLeftPathUtf8BytesV1,
    kConstraintRightPathPoolOffsetV1,
    kConstraintRightPathUtf8BytesV1,
    kConstraintWireColumnCountV1,
};

struct HostTemplateWireTableV1 {
    const std::uint64_t* data;
    std::uint64_t row_count;
    std::uint32_t column_count;
    std::uint32_t reserved;
};

struct HostTemplateWireBytePoolV1 {
    const std::uint8_t* data;
    std::uint64_t byte_count;
    std::uint64_t reserved;
};

struct HostTemplateWireViewV1 {
    std::uint32_t abi_version;
    std::uint32_t reserved0;
    std::uint32_t generation_count;
    std::uint32_t cell_count;
    HostTemplateWireTableV1 programs;
    HostTemplateWireTableV1 tokens;
    HostTemplateWireTableV1 parts;
    HostTemplateWireTableV1 constraints;
    HostTemplateWireBytePoolV1 byte_pool;
    std::uint64_t reserved1;
};

enum class HostTemplateRuntimeStatusV1 : std::uint32_t {
    kSuccess = 0,
    kNotPrepared = 1,
    kNullArgument = 2,
    kUnsupportedHostEndian = 3,
    kUnsupportedWireAbi = 4,
    kWireShape = 5,
    kReservedFieldNonzero = 6,
    kCoordinateBounds = 7,
    kCoordinateOrder = 8,
    kRowSliceBounds = 9,
    kRowSliceOrder = 10,
    kUnsupportedProgramAbi = 11,
    kUnsupportedEncoding = 12,
    kUnsupportedSurface = 13,
    kScalarNarrowing = 14,
    kUnsupportedTokenType = 15,
    kUnsupportedValidationKind = 16,
    kTokenDeclarationContradiction = 17,
    kUnsupportedPartKind = 18,
    kPartContradiction = 19,
    kUnsupportedConstraintKind = 20,
    kConstraintContradiction = 21,
    kBytePoolBounds = 22,
    kBytePoolTraversal = 23,
    kInvalidUtf8 = 24,
    kInvalidSourcePath = 25,
    kRenderedBoundContradiction = 26,
    kUnownedStorage = 27,
    kAllocationFailure = 28,
    kRendererRejected = 29,
    kInternalException = 30,
};

enum class HostTemplateRuntimeFailureComponentV1 : std::uint32_t {
    kNone = 0,
    kWireView = 1,
    kPrograms = 2,
    kTokens = 3,
    kParts = 4,
    kConstraints = 5,
    kBytePool = 6,
    kAllocation = 7,
    kRenderer = 8,
};

struct HostTemplateRuntimeFailureV1 {
    std::uint32_t status;
    std::uint32_t component;
    std::uint64_t row;
    std::uint32_t column;
    std::uint32_t renderer_status;
    std::uint32_t renderer_component;
    std::uint32_t renderer_index;
    std::uint32_t reserved;
};

struct HostTemplateProgramBindingV1 {
    const renderer::HostTemplateProgramV1* program;
    renderer::HostTemplateTokenValueV1* token_values;
    std::uint32_t token_value_count;
    std::uint32_t reserved;
};

struct HostTemplateTokenStorageV1 {
    renderer::HostTemplateTokenValueV1* value;
    std::uint64_t* shape_dimensions;
    std::uint32_t shape_dimension_capacity;
    std::uint8_t* sha256_hex;
    std::uint32_t sha256_hex_capacity;
    std::uint32_t reserved;
};

class HostTemplateRuntimeV1;

HostTemplateRuntimeStatusV1 prepare_host_template_runtime_v1(
    HostTemplateWireViewV1 wire,
    HostTemplateRuntimeV1* runtime
) noexcept;

HostTemplateRuntimeStatusV1 host_template_runtime_status_v1(
    const HostTemplateRuntimeV1* runtime
) noexcept;

HostTemplateRuntimeFailureV1 host_template_runtime_failure_v1(
    const HostTemplateRuntimeV1* runtime
) noexcept;

std::uint64_t host_template_program_count_v1(
    const HostTemplateRuntimeV1* runtime
) noexcept;

HostTemplateProgramBindingV1 lookup_host_template_program_v1(
    HostTemplateRuntimeV1* runtime,
    std::uint32_t generation,
    std::uint32_t cell,
    std::uint32_t surface
) noexcept;

HostTemplateTokenStorageV1 lookup_host_template_token_storage_v1(
    HostTemplateRuntimeV1* runtime,
    std::uint32_t generation,
    std::uint32_t cell,
    std::uint32_t surface,
    std::uint32_t token_index
) noexcept;

const char* host_template_runtime_status_name_v1(
    HostTemplateRuntimeStatusV1 status
) noexcept;

class HostTemplateRuntimeV1 final {
public:
    HostTemplateRuntimeV1() noexcept = default;
    ~HostTemplateRuntimeV1() = default;

    HostTemplateRuntimeV1(const HostTemplateRuntimeV1&) = delete;
    HostTemplateRuntimeV1& operator=(const HostTemplateRuntimeV1&) = delete;
    HostTemplateRuntimeV1(HostTemplateRuntimeV1&&) = delete;
    HostTemplateRuntimeV1& operator=(HostTemplateRuntimeV1&&) = delete;

private:
    friend HostTemplateRuntimeStatusV1 prepare_host_template_runtime_v1(
        HostTemplateWireViewV1,
        HostTemplateRuntimeV1*
    ) noexcept;
    friend HostTemplateRuntimeStatusV1 host_template_runtime_status_v1(
        const HostTemplateRuntimeV1*
    ) noexcept;
    friend HostTemplateRuntimeFailureV1 host_template_runtime_failure_v1(
        const HostTemplateRuntimeV1*
    ) noexcept;
    friend std::uint64_t host_template_program_count_v1(
        const HostTemplateRuntimeV1*
    ) noexcept;
    friend HostTemplateProgramBindingV1 lookup_host_template_program_v1(
        HostTemplateRuntimeV1*,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t
    ) noexcept;
    friend HostTemplateTokenStorageV1 lookup_host_template_token_storage_v1(
        HostTemplateRuntimeV1*,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t,
        std::uint32_t
    ) noexcept;

    HostTemplateRuntimeStatusV1 status_ =
        HostTemplateRuntimeStatusV1::kNotPrepared;
    HostTemplateRuntimeFailureV1 failure_ = {
        static_cast<std::uint32_t>(HostTemplateRuntimeStatusV1::kNotPrepared),
        static_cast<std::uint32_t>(
            HostTemplateRuntimeFailureComponentV1::kNone
        ),
        kNoRuntimeRowV1,
        kNoRuntimeColumnV1,
        static_cast<std::uint32_t>(renderer::HostTemplateStatusV1::kSuccess),
        static_cast<std::uint32_t>(
            renderer::HostTemplateFailureComponentV1::kNone
        ),
        renderer::kNoIndexV1,
        0,
    };
    std::uint32_t generation_count_ = 0;
    std::uint32_t cell_count_ = 0;
    std::vector<renderer::HostTemplateProgramV1> programs_;
    std::vector<renderer::HostTemplateTokenSpecV1> tokens_;
    std::vector<renderer::HostTemplateTokenValueV1> token_values_;
    std::vector<renderer::HostTemplatePartV1> parts_;
    std::vector<renderer::HostTemplateConstraintV1> constraints_;
    std::vector<std::uint8_t> byte_pool_;
    std::vector<std::uint64_t> shape_dimensions_;
    std::vector<std::uint8_t> sha256_hex_;
};

static_assert(kProgramWireColumnCountV1 == 13);
static_assert(kTokenWireColumnCountV1 == 13);
static_assert(kPartWireColumnCountV1 == 5);
static_assert(kConstraintWireColumnCountV1 == 5);
static_assert(std::is_standard_layout_v<HostTemplateWireTableV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateWireTableV1>);
static_assert(std::is_standard_layout_v<HostTemplateWireBytePoolV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateWireBytePoolV1>);
static_assert(std::is_standard_layout_v<HostTemplateWireViewV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateWireViewV1>);
static_assert(std::is_standard_layout_v<HostTemplateRuntimeFailureV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateRuntimeFailureV1>);
static_assert(std::is_standard_layout_v<HostTemplateProgramBindingV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateProgramBindingV1>);
static_assert(std::is_standard_layout_v<HostTemplateTokenStorageV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateTokenStorageV1>);

}  // namespace eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1
