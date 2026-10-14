#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1 {

inline constexpr std::uint32_t kHostTemplateRendererAbiVersionV1 = 1;
inline constexpr std::uint32_t kDigestBytesV1 = 32;
inline constexpr std::uint32_t kNoIndexV1 = UINT32_MAX;
inline constexpr std::uint32_t kMaximumTokensPerTemplateV1 = 8192;
inline constexpr std::uint64_t kMaximumLiteralUtf8BytesPerTemplateV1 =
    8ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kMaximumRenderedUtf8BytesPerTemplateV1 =
    16ULL * 1024ULL * 1024ULL;
inline constexpr std::uint32_t kMaximumSourcePathUtf8BytesV1 = 2048;

// Exact lowering of eiel_compact_summary_host_template_v1. The Python string
// spellings are retained here to make the pre-dispatch translation explicit.
enum class HostTemplateSurfaceV1 : std::uint32_t {
    kCheckpointTrajectory = 1,       // checkpoint_trajectory
    kCanonicalCheckpointStorage = 2, // canonical_checkpoint_storage
    kNextGenerationSource = 3,       // next_generation_source_projection
    kRng = 4,                        // rng
    kActiveAgentIds = 5,             // active_agent_ids
    kPolicyMutationTape = 6,         // policy_mutation_tape
};

enum class HostTemplateEncodingV1 : std::uint32_t {
    kUtf8 = 1,
};

enum class HostTemplateTokenTypeV1 : std::uint32_t {
    kInteger = 1,
    kBoolean = 2,
    kPcgUint128 = 3,
    kPythonMtWord = 4,
    kTensorShape = 5,
    kSha256Hex = 6,
    kNullableFloat64 = 7,
};

enum class HostTemplateValidationKindV1 : std::uint32_t {
    kIntegerRange = 1,
    kExactBoolean = 2,
    kTensorShape = 3,
    kLowercaseSha256Hex = 4,
    kFiniteOptionalFloat64 = 5,
};

enum class HostTemplatePartKindV1 : std::uint32_t {
    kLiteralUtf8 = 1,
    kDynamicToken = 2,
};

enum class HostTemplateConstraintKindV1 : std::uint32_t {
    kValuesEqual = 1,
    kShapeFirstDimensionEqualsInteger = 2,
};

enum class HostTemplateRenderModeV1 : std::uint32_t {
    kHashOnly = 0,
    kRenderAndHash = 1,
};

enum class HostTemplateStatusV1 : std::uint32_t {
    kSuccess = 0,
    kNullArgument = 1,
    kUnsupportedAbiVersion = 2,
    kUnsupportedEncoding = 3,
    kUnsupportedSurface = 4,
    kReservedFieldNonzero = 5,
    kProgramBounds = 6,
    kMalformedProgram = 7,
    kInvalidUtf8 = 8,
    kUnsupportedTokenType = 9,
    kUnsupportedValidationRule = 10,
    kTokenDeclarationContradiction = 11,
    kTokenValueCountMismatch = 12,
    kTokenValueTypeMismatch = 13,
    kTokenValueBounds = 14,
    kTokenShapeContradiction = 15,
    kTokenSha256Contradiction = 16,
    kUnsupportedPartKind = 17,
    kPartBounds = 18,
    kUnsupportedConstraintKind = 19,
    kConstraintPathContradiction = 20,
    kConstraintTypeContradiction = 21,
    kConstraintValueContradiction = 22,
    kRenderedBoundContradiction = 23,
    kOutputOverrun = 24,
    kOutputModeContradiction = 25,
};

enum class HostTemplateFailureComponentV1 : std::uint32_t {
    kNone = 0,
    kRequest = 1,
    kProgram = 2,
    kTokenDeclaration = 3,
    kTokenValue = 4,
    kPart = 5,
    kConstraint = 6,
    kOutput = 7,
};

// validation_kind and its numeric fields are a direct POD lowering of the
// compiler's validation_rule mapping. rule_parameter is the SHA-256 hex length
// for kLowercaseSha256Hex and zero for every other rule.
struct HostTemplateTokenSpecV1 {
    std::uint32_t token_index;
    std::uint32_t token_type;
    std::uint32_t validation_kind;
    std::uint32_t rank;
    std::uint32_t render_pack_depth;
    std::uint32_t maximum_utf8_bytes;
    std::uint32_t source_path_utf8_bytes;
    std::uint32_t rule_parameter;
    std::uint64_t minimum_low;
    std::uint64_t minimum_high;
    std::uint64_t maximum_low;
    std::uint64_t maximum_high;
    const std::uint8_t* source_path_utf8;
    std::uint64_t reserved;
};

// The caller supplies one value per token in token_index order. Integer-like
// values use scalar_low/scalar_high, nullable float64 uses scalar_low for the
// IEEE-754 bits and scalar_high for canonical 0/1 presence, tensor shapes use
// shape_dimensions, and SHA-256 values use exactly 64 lowercase ASCII bytes in
// sha256_hex.
struct HostTemplateTokenValueV1 {
    std::uint32_t token_type;
    std::uint32_t shape_rank;
    std::uint64_t scalar_low;
    std::uint64_t scalar_high;
    const std::uint64_t* shape_dimensions;
    const std::uint8_t* sha256_hex;
    std::uint32_t sha256_hex_bytes;
    std::uint32_t reserved;
};

struct HostTemplatePartV1 {
    std::uint32_t kind;
    std::uint32_t token_index;
    const std::uint8_t* literal_utf8;
    std::uint64_t literal_utf8_bytes;
    std::uint64_t reserved;
};

// left/right retain the compiler's source-path grammar. For values_equal they
// are left_source_path/right_source_path. For the shape/cursor relation they
// are shape_source_path/integer_source_path.
struct HostTemplateConstraintV1 {
    std::uint32_t kind;
    std::uint32_t reserved0;
    const std::uint8_t* left_source_path_utf8;
    std::uint32_t left_source_path_utf8_bytes;
    std::uint32_t reserved1;
    const std::uint8_t* right_source_path_utf8;
    std::uint32_t right_source_path_utf8_bytes;
    std::uint32_t reserved2;
    std::uint64_t reserved3;
};

struct HostTemplateProgramV1 {
    std::uint32_t abi_version;
    std::uint32_t encoding;
    std::uint32_t surface;
    std::uint32_t reserved0;
    std::int64_t generation;
    std::int64_t cell;
    std::uint32_t token_count;
    std::uint32_t part_count;
    std::uint32_t constraint_count;
    std::uint32_t reserved1;
    std::uint64_t literal_utf8_bytes;
    std::uint64_t maximum_rendered_utf8_bytes;
    const HostTemplateTokenSpecV1* tokens;
    const HostTemplatePartV1* parts;
    const HostTemplateConstraintV1* constraints;
    std::uint64_t reserved2;
};

struct HostTemplateResultV1 {
    std::uint32_t status;
    std::uint32_t failed_component;
    std::uint32_t failed_index;
    std::uint32_t reserved0;
    std::uint64_t rendered_utf8_bytes;
    std::uint8_t digest_sha256[kDigestBytesV1];
    std::uint64_t reserved1;
};

struct HostTemplateRequestV1 {
    std::uint32_t abi_version;
    std::uint32_t mode;
    const HostTemplateProgramV1* program;
    const HostTemplateTokenValueV1* token_values;
    std::uint32_t token_value_count;
    std::uint32_t reserved0;
    std::uint8_t* output_utf8;
    std::uint64_t output_capacity;
    HostTemplateResultV1* result;
    std::uint64_t reserved1;
};

HostTemplateStatusV1 render_and_hash_host_template_v1(
    const HostTemplateRequestV1* request
) noexcept;

void render_and_hash_host_template_callback_v1(void* user_data) noexcept;

static_assert(std::is_standard_layout_v<HostTemplateTokenSpecV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateTokenSpecV1>);
static_assert(sizeof(HostTemplateTokenSpecV1) == 80);
static_assert(std::is_standard_layout_v<HostTemplateTokenValueV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateTokenValueV1>);
static_assert(sizeof(HostTemplateTokenValueV1) == 48);
static_assert(std::is_standard_layout_v<HostTemplatePartV1>);
static_assert(std::is_trivially_copyable_v<HostTemplatePartV1>);
static_assert(sizeof(HostTemplatePartV1) == 32);
static_assert(std::is_standard_layout_v<HostTemplateConstraintV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateConstraintV1>);
static_assert(sizeof(HostTemplateConstraintV1) == 48);
static_assert(std::is_standard_layout_v<HostTemplateProgramV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateProgramV1>);
static_assert(sizeof(HostTemplateProgramV1) == 96);
static_assert(std::is_standard_layout_v<HostTemplateResultV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateResultV1>);
static_assert(sizeof(HostTemplateResultV1) == 64);
static_assert(std::is_standard_layout_v<HostTemplateRequestV1>);
static_assert(std::is_trivially_copyable_v<HostTemplateRequestV1>);
static_assert(sizeof(HostTemplateRequestV1) == 64);

}  // namespace eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1
