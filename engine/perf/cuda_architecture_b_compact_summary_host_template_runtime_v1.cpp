#include "cuda_architecture_b_compact_summary_host_template_runtime_v1.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1 {
namespace {

using FailureComponent = HostTemplateRuntimeFailureComponentV1;
using RuntimeStatus = HostTemplateRuntimeStatusV1;

struct ValidationFailureV1 {
    RuntimeStatus status;
    FailureComponent component;
    std::uint64_t row;
    std::uint32_t column;
    std::uint32_t renderer_status;
    std::uint32_t renderer_component;
    std::uint32_t renderer_index;
};

struct WireTotalsV1 {
    std::uint64_t shape_dimension_count;
    std::uint64_t sha256_hex_bytes;
};

constexpr ValidationFailureV1 success() noexcept {
    return {
        RuntimeStatus::kSuccess,
        FailureComponent::kNone,
        kNoRuntimeRowV1,
        kNoRuntimeColumnV1,
        static_cast<std::uint32_t>(renderer::HostTemplateStatusV1::kSuccess),
        static_cast<std::uint32_t>(
            renderer::HostTemplateFailureComponentV1::kNone
        ),
        renderer::kNoIndexV1,
    };
}

constexpr ValidationFailureV1 failure(
    const RuntimeStatus status,
    const FailureComponent component,
    const std::uint64_t row = kNoRuntimeRowV1,
    const std::uint32_t column = kNoRuntimeColumnV1
) noexcept {
    return {
        status,
        component,
        row,
        column,
        static_cast<std::uint32_t>(renderer::HostTemplateStatusV1::kSuccess),
        static_cast<std::uint32_t>(
            renderer::HostTemplateFailureComponentV1::kNone
        ),
        renderer::kNoIndexV1,
    };
}

bool host_is_little_endian() noexcept {
    const std::uint16_t value = 1U;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1U;
}

bool checked_add(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t* const result
) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *result = left + right;
    return true;
}

bool fits_size_t(const std::uint64_t value) noexcept {
    return value <= static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max()
    );
}

bool valid_table_shape(
    const HostTemplateWireTableV1& table,
    const std::uint32_t expected_columns
) noexcept {
    if (table.column_count != expected_columns
        || (table.row_count == 0U) != (table.data == nullptr)
        || !fits_size_t(table.row_count)
        || table.row_count > std::numeric_limits<std::size_t>::max()
            / expected_columns) {
        return false;
    }
    return table.data == nullptr
        || reinterpret_cast<std::uintptr_t>(table.data)
            % alignof(std::uint64_t) == 0U;
}

const std::uint64_t* table_row(
    const HostTemplateWireTableV1& table,
    const std::uint64_t row
) noexcept {
    return table.data + static_cast<std::size_t>(row) * table.column_count;
}

bool pool_span_is_valid(
    const HostTemplateWireBytePoolV1& pool,
    const std::uint64_t offset,
    const std::uint64_t byte_count
) noexcept {
    return offset <= pool.byte_count && byte_count <= pool.byte_count - offset;
}

const std::uint8_t* pool_span(
    const HostTemplateWireBytePoolV1& pool,
    const std::uint64_t offset
) noexcept {
    return pool.data + static_cast<std::size_t>(offset);
}

bool valid_utf8(
    const std::uint8_t* const bytes,
    const std::uint64_t size
) noexcept {
    if (size != 0U && bytes == nullptr) {
        return false;
    }
    std::uint64_t cursor = 0;
    while (cursor < size) {
        const std::uint8_t first = bytes[cursor++];
        if (first <= 0x7fU) {
            continue;
        }
        std::uint32_t remaining = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (first >= 0xc2U && first <= 0xdfU) {
            remaining = 1;
            codepoint = first & 0x1fU;
            minimum = 0x80U;
        } else if (first >= 0xe0U && first <= 0xefU) {
            remaining = 2;
            codepoint = first & 0x0fU;
            minimum = 0x800U;
        } else if (first >= 0xf0U && first <= 0xf4U) {
            remaining = 3;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        } else {
            return false;
        }
        if (remaining > size - cursor) {
            return false;
        }
        for (std::uint32_t index = 0; index < remaining; ++index) {
            const std::uint8_t continuation = bytes[cursor++];
            if ((continuation & 0xc0U) != 0x80U) {
                return false;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3fU);
        }
        if (codepoint < minimum || codepoint > 0x10ffffU
            || (codepoint >= 0xd800U && codepoint <= 0xdfffU)) {
            return false;
        }
    }
    return true;
}

bool valid_source_path(
    const std::uint8_t* const bytes,
    const std::uint64_t size
) noexcept {
    return size != 0U
        && size <= renderer::kMaximumSourcePathUtf8BytesV1
        && bytes != nullptr
        && bytes[0] == static_cast<std::uint8_t>('/')
        && valid_utf8(bytes, size);
}

bool spans_equal(
    const std::uint8_t* const left,
    const std::uint64_t left_size,
    const std::uint8_t* const right,
    const std::uint64_t right_size
) noexcept {
    return left_size == right_size
        && (left_size == 0U
            || std::memcmp(
                left,
                right,
                static_cast<std::size_t>(left_size)
            ) == 0);
}

bool valid_rule_words(const std::uint64_t* const token) noexcept {
    const std::uint64_t type = token[kTokenTypeV1];
    const std::uint64_t validation = token[kTokenValidationKindV1];
    const std::uint64_t rank = token[kTokenRankV1];
    const std::uint64_t parameter = token[kTokenRuleParameterV1];
    const std::uint64_t minimum_low = token[kTokenMinimumLowV1];
    const std::uint64_t minimum_high = token[kTokenMinimumHighV1];
    const std::uint64_t maximum_low = token[kTokenMaximumLowV1];
    const std::uint64_t maximum_high = token[kTokenMaximumHighV1];
    if (minimum_low != 0U || minimum_high != 0U) {
        return false;
    }
    switch (static_cast<renderer::HostTemplateTokenTypeV1>(type)) {
        case renderer::HostTemplateTokenTypeV1::kInteger:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kIntegerRange
                )
                && rank == 0U && parameter == 0U
                && maximum_low == UINT64_MAX && maximum_high == 0U;
        case renderer::HostTemplateTokenTypeV1::kBoolean:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kExactBoolean
                )
                && rank == 0U && parameter == 0U
                && maximum_low == 0U && maximum_high == 0U;
        case renderer::HostTemplateTokenTypeV1::kPcgUint128:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kIntegerRange
                )
                && rank == 0U && parameter == 0U
                && maximum_low == UINT64_MAX && maximum_high == UINT64_MAX;
        case renderer::HostTemplateTokenTypeV1::kPythonMtWord:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kIntegerRange
                )
                && rank == 0U && parameter == 0U
                && maximum_low == UINT32_MAX && maximum_high == 0U;
        case renderer::HostTemplateTokenTypeV1::kTensorShape:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kTensorShape
                )
                && parameter == 0U
                && maximum_low == UINT64_MAX && maximum_high == 0U;
        case renderer::HostTemplateTokenTypeV1::kSha256Hex:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kLowercaseSha256Hex
                )
                && rank == 0U && parameter == 64U
                && maximum_low == 0U && maximum_high == 0U;
        case renderer::HostTemplateTokenTypeV1::kNullableFloat64:
            return validation == static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kFiniteOptionalFloat64
                )
                && rank == 0U && parameter == 0U
                && maximum_low == 0U && maximum_high == 0U;
    }
    return false;
}

std::int64_t find_local_token(
    const HostTemplateWireViewV1& wire,
    const std::uint64_t token_offset,
    const std::uint64_t token_count,
    const std::uint8_t* const path,
    const std::uint64_t path_size
) noexcept {
    for (std::uint64_t local = 0; local < token_count; ++local) {
        const std::uint64_t* const token = table_row(
            wire.tokens,
            token_offset + local
        );
        const std::uint64_t offset = token[kTokenSourcePathPoolOffsetV1];
        const std::uint64_t size = token[kTokenSourcePathUtf8BytesV1];
        if (spans_equal(pool_span(wire.byte_pool, offset), size, path, path_size)) {
            return static_cast<std::int64_t>(local);
        }
    }
    return -1;
}

ValidationFailureV1 validate_wire(
    const HostTemplateWireViewV1& wire,
    WireTotalsV1* const totals
) noexcept {
    if (!host_is_little_endian()) {
        return failure(
            RuntimeStatus::kUnsupportedHostEndian,
            FailureComponent::kWireView
        );
    }
    if (wire.abi_version != kHostTemplateWireAbiVersionV1) {
        return failure(
            RuntimeStatus::kUnsupportedWireAbi,
            FailureComponent::kWireView,
            kNoRuntimeRowV1,
            0
        );
    }
    if (wire.reserved0 != 0U || wire.reserved1 != 0U
        || wire.programs.reserved != 0U || wire.tokens.reserved != 0U
        || wire.parts.reserved != 0U || wire.constraints.reserved != 0U
        || wire.byte_pool.reserved != 0U) {
        return failure(
            RuntimeStatus::kReservedFieldNonzero,
            FailureComponent::kWireView
        );
    }
    if (!valid_table_shape(wire.programs, kProgramWireColumnCountV1)
        || !valid_table_shape(wire.tokens, kTokenWireColumnCountV1)
        || !valid_table_shape(wire.parts, kPartWireColumnCountV1)
        || !valid_table_shape(wire.constraints, kConstraintWireColumnCountV1)
        || (wire.byte_pool.byte_count == 0U) != (wire.byte_pool.data == nullptr)
        || !fits_size_t(wire.byte_pool.byte_count)) {
        return failure(
            RuntimeStatus::kWireShape,
            FailureComponent::kWireView
        );
    }
    const std::uint64_t coordinate_count =
        static_cast<std::uint64_t>(wire.generation_count) * wire.cell_count;
    if (wire.generation_count == 0U || wire.cell_count == 0U
        || coordinate_count > kMaximumHostTemplateCoordinatesV1) {
        return failure(
            RuntimeStatus::kCoordinateBounds,
            FailureComponent::kWireView
        );
    }
    if (wire.programs.row_count
        != coordinate_count * kHostTemplateSurfaceCountV1) {
        return failure(
            RuntimeStatus::kWireShape,
            FailureComponent::kPrograms
        );
    }

    std::uint64_t token_cursor = 0;
    std::uint64_t part_cursor = 0;
    std::uint64_t constraint_cursor = 0;
    std::uint64_t pool_cursor = 0;
    totals->shape_dimension_count = 0;
    totals->sha256_hex_bytes = 0;
    for (std::uint64_t program_index = 0;
         program_index < wire.programs.row_count;
         ++program_index) {
        const std::uint64_t* const program = table_row(
            wire.programs,
            program_index
        );
        if (program[kProgramAbiVersionV1]
            != renderer::kHostTemplateRendererAbiVersionV1) {
            return failure(
                RuntimeStatus::kUnsupportedProgramAbi,
                FailureComponent::kPrograms,
                program_index,
                kProgramAbiVersionV1
            );
        }
        if (program[kProgramEncodingV1] != static_cast<std::uint32_t>(
                renderer::HostTemplateEncodingV1::kUtf8
            )) {
            return failure(
                RuntimeStatus::kUnsupportedEncoding,
                FailureComponent::kPrograms,
                program_index,
                kProgramEncodingV1
            );
        }
        const std::uint64_t coordinate_index =
            program_index / kHostTemplateSurfaceCountV1;
        const std::uint64_t expected_surface =
            program_index % kHostTemplateSurfaceCountV1 + 1U;
        const std::uint64_t expected_generation =
            coordinate_index / wire.cell_count;
        const std::uint64_t expected_cell = coordinate_index % wire.cell_count;
        if (program[kProgramSurfaceV1] < 1U
            || program[kProgramSurfaceV1] > kHostTemplateSurfaceCountV1) {
            return failure(
                RuntimeStatus::kUnsupportedSurface,
                FailureComponent::kPrograms,
                program_index,
                kProgramSurfaceV1
            );
        }
        if (program[kProgramSurfaceV1] != expected_surface
            || program[kProgramGenerationV1] != expected_generation
            || program[kProgramCellV1] != expected_cell) {
            return failure(
                RuntimeStatus::kCoordinateOrder,
                FailureComponent::kPrograms,
                program_index,
                program[kProgramSurfaceV1] != expected_surface
                    ? kProgramSurfaceV1
                    : (program[kProgramGenerationV1] != expected_generation
                        ? kProgramGenerationV1
                        : kProgramCellV1)
            );
        }

        const std::uint64_t token_offset = program[kProgramTokenOffsetV1];
        const std::uint64_t token_count = program[kProgramTokenCountV1];
        const std::uint64_t part_offset = program[kProgramPartOffsetV1];
        const std::uint64_t part_count = program[kProgramPartCountV1];
        const std::uint64_t constraint_offset =
            program[kProgramConstraintOffsetV1];
        const std::uint64_t constraint_count =
            program[kProgramConstraintCountV1];
        if (token_offset != token_cursor || part_offset != part_cursor
            || constraint_offset != constraint_cursor) {
            return failure(
                RuntimeStatus::kRowSliceOrder,
                FailureComponent::kPrograms,
                program_index
            );
        }
        if (token_count > renderer::kMaximumTokensPerTemplateV1
            || part_count == 0U
            || part_count > renderer::kMaximumRenderedUtf8BytesPerTemplateV1
            || part_count > UINT32_MAX
            || constraint_count > token_count
            || token_count > wire.tokens.row_count - token_cursor
            || part_count > wire.parts.row_count - part_cursor
            || constraint_count
                > wire.constraints.row_count - constraint_cursor) {
            return failure(
                RuntimeStatus::kRowSliceBounds,
                FailureComponent::kPrograms,
                program_index
            );
        }
        const std::uint64_t declared_literal =
            program[kProgramLiteralUtf8BytesV1];
        const std::uint64_t declared_maximum =
            program[kProgramMaximumRenderedUtf8BytesV1];
        if (declared_literal
                > renderer::kMaximumLiteralUtf8BytesPerTemplateV1
            || declared_maximum
                > renderer::kMaximumRenderedUtf8BytesPerTemplateV1) {
            return failure(
                RuntimeStatus::kRenderedBoundContradiction,
                FailureComponent::kPrograms,
                program_index
            );
        }

        std::uint64_t compiled_maximum = declared_literal;
        for (std::uint64_t local = 0; local < token_count; ++local) {
            const std::uint64_t global = token_cursor + local;
            const std::uint64_t* const token = table_row(wire.tokens, global);
            if (token[kTokenIndexV1] != local) {
                return failure(
                    RuntimeStatus::kTokenDeclarationContradiction,
                    FailureComponent::kTokens,
                    global,
                    kTokenIndexV1
                );
            }
            if (token[kTokenTypeV1] < static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kInteger
                )
                || token[kTokenTypeV1] > static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kNullableFloat64
                )) {
                return failure(
                    RuntimeStatus::kUnsupportedTokenType,
                    FailureComponent::kTokens,
                    global,
                    kTokenTypeV1
                );
            }
            if (token[kTokenValidationKindV1] < static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kIntegerRange
                )
                || token[kTokenValidationKindV1] > static_cast<std::uint32_t>(
                    renderer::HostTemplateValidationKindV1::kFiniteOptionalFloat64
                )) {
                return failure(
                    RuntimeStatus::kUnsupportedValidationKind,
                    FailureComponent::kTokens,
                    global,
                    kTokenValidationKindV1
                );
            }
            if (token[kTokenRankV1] > UINT32_MAX
                || token[kTokenRenderPackDepthV1] > 2U
                || token[kTokenMaximumUtf8BytesV1] > UINT32_MAX
                || token[kTokenMaximumUtf8BytesV1]
                    > renderer::kMaximumRenderedUtf8BytesPerTemplateV1
                || token[kTokenSourcePathUtf8BytesV1] > UINT32_MAX
                || token[kTokenRuleParameterV1] > UINT32_MAX) {
                return failure(
                    RuntimeStatus::kScalarNarrowing,
                    FailureComponent::kTokens,
                    global
                );
            }
            if (!valid_rule_words(token)) {
                return failure(
                    RuntimeStatus::kTokenDeclarationContradiction,
                    FailureComponent::kTokens,
                    global,
                    kTokenValidationKindV1
                );
            }
            const std::uint64_t path_offset =
                token[kTokenSourcePathPoolOffsetV1];
            const std::uint64_t path_size =
                token[kTokenSourcePathUtf8BytesV1];
            if (!pool_span_is_valid(wire.byte_pool, path_offset, path_size)) {
                return failure(
                    RuntimeStatus::kBytePoolBounds,
                    FailureComponent::kTokens,
                    global,
                    kTokenSourcePathPoolOffsetV1
                );
            }
            if (path_offset != pool_cursor) {
                return failure(
                    RuntimeStatus::kBytePoolTraversal,
                    FailureComponent::kTokens,
                    global,
                    kTokenSourcePathPoolOffsetV1
                );
            }
            const std::uint8_t* const path = pool_span(
                wire.byte_pool,
                path_offset
            );
            if (!valid_utf8(path, path_size)) {
                return failure(
                    RuntimeStatus::kInvalidUtf8,
                    FailureComponent::kTokens,
                    global,
                    kTokenSourcePathPoolOffsetV1
                );
            }
            if (!valid_source_path(path, path_size)) {
                return failure(
                    RuntimeStatus::kInvalidSourcePath,
                    FailureComponent::kTokens,
                    global,
                    kTokenSourcePathPoolOffsetV1
                );
            }
            for (std::uint64_t prior = 0; prior < local; ++prior) {
                const std::uint64_t* const prior_token = table_row(
                    wire.tokens,
                    token_cursor + prior
                );
                if (spans_equal(
                        path,
                        path_size,
                        pool_span(
                            wire.byte_pool,
                            prior_token[kTokenSourcePathPoolOffsetV1]
                        ),
                        prior_token[kTokenSourcePathUtf8BytesV1]
                    )) {
                    return failure(
                        RuntimeStatus::kTokenDeclarationContradiction,
                        FailureComponent::kTokens,
                        global,
                        kTokenSourcePathPoolOffsetV1
                    );
                }
            }
            pool_cursor += path_size;
            if (!checked_add(
                    compiled_maximum,
                    token[kTokenMaximumUtf8BytesV1],
                    &compiled_maximum
                )) {
                return failure(
                    RuntimeStatus::kRenderedBoundContradiction,
                    FailureComponent::kPrograms,
                    program_index
                );
            }
            if (token[kTokenTypeV1] == static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kTensorShape
                )
                && !checked_add(
                    totals->shape_dimension_count,
                    token[kTokenRankV1],
                    &totals->shape_dimension_count
                )) {
                return failure(
                    RuntimeStatus::kScalarNarrowing,
                    FailureComponent::kTokens,
                    global,
                    kTokenRankV1
                );
            }
            if (token[kTokenTypeV1] == static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kSha256Hex
                )
                && !checked_add(
                    totals->sha256_hex_bytes,
                    64U,
                    &totals->sha256_hex_bytes
                )) {
                return failure(
                    RuntimeStatus::kScalarNarrowing,
                    FailureComponent::kTokens,
                    global
                );
            }
        }
        if (compiled_maximum != declared_maximum) {
            return failure(
                RuntimeStatus::kRenderedBoundContradiction,
                FailureComponent::kPrograms,
                program_index,
                kProgramMaximumRenderedUtf8BytesV1
            );
        }

        std::uint64_t seen_tokens[
            (renderer::kMaximumTokensPerTemplateV1 + 63U) / 64U
        ] = {};
        std::uint64_t actual_literal = 0;
        bool previous_literal = false;
        for (std::uint64_t local = 0; local < part_count; ++local) {
            const std::uint64_t global = part_cursor + local;
            const std::uint64_t* const part = table_row(wire.parts, global);
            if (part[kPartReservedZeroV1] != 0U) {
                return failure(
                    RuntimeStatus::kReservedFieldNonzero,
                    FailureComponent::kParts,
                    global,
                    kPartReservedZeroV1
                );
            }
            if (part[kPartKindV1] == static_cast<std::uint32_t>(
                    renderer::HostTemplatePartKindV1::kLiteralUtf8
                )) {
                if (previous_literal || part[kPartTokenIndexV1] != UINT32_MAX
                    || part[kPartLiteralUtf8BytesV1] == 0U) {
                    return failure(
                        RuntimeStatus::kPartContradiction,
                        FailureComponent::kParts,
                        global
                    );
                }
                const std::uint64_t offset = part[kPartLiteralPoolOffsetV1];
                const std::uint64_t size = part[kPartLiteralUtf8BytesV1];
                if (!pool_span_is_valid(wire.byte_pool, offset, size)) {
                    return failure(
                        RuntimeStatus::kBytePoolBounds,
                        FailureComponent::kParts,
                        global,
                        kPartLiteralPoolOffsetV1
                    );
                }
                if (offset != pool_cursor) {
                    return failure(
                        RuntimeStatus::kBytePoolTraversal,
                        FailureComponent::kParts,
                        global,
                        kPartLiteralPoolOffsetV1
                    );
                }
                if (!valid_utf8(pool_span(wire.byte_pool, offset), size)) {
                    return failure(
                        RuntimeStatus::kInvalidUtf8,
                        FailureComponent::kParts,
                        global,
                        kPartLiteralPoolOffsetV1
                    );
                }
                pool_cursor += size;
                if (!checked_add(actual_literal, size, &actual_literal)) {
                    return failure(
                        RuntimeStatus::kRenderedBoundContradiction,
                        FailureComponent::kParts,
                        global
                    );
                }
                previous_literal = true;
            } else if (part[kPartKindV1] == static_cast<std::uint32_t>(
                    renderer::HostTemplatePartKindV1::kDynamicToken
                )) {
                if (part[kPartTokenIndexV1] >= token_count
                    || part[kPartLiteralPoolOffsetV1] != 0U
                    || part[kPartLiteralUtf8BytesV1] != 0U) {
                    return failure(
                        RuntimeStatus::kPartContradiction,
                        FailureComponent::kParts,
                        global
                    );
                }
                const std::uint64_t token_index = part[kPartTokenIndexV1];
                seen_tokens[token_index / 64U]
                    |= 1ULL << (token_index % 64U);
                previous_literal = false;
            } else {
                return failure(
                    RuntimeStatus::kUnsupportedPartKind,
                    FailureComponent::kParts,
                    global,
                    kPartKindV1
                );
            }
        }
        if (actual_literal != declared_literal) {
            return failure(
                RuntimeStatus::kRenderedBoundContradiction,
                FailureComponent::kPrograms,
                program_index,
                kProgramLiteralUtf8BytesV1
            );
        }
        for (std::uint64_t token = 0; token < token_count; ++token) {
            if ((seen_tokens[token / 64U] & (1ULL << (token % 64U))) == 0U) {
                return failure(
                    RuntimeStatus::kPartContradiction,
                    FailureComponent::kParts,
                    part_cursor
                );
            }
        }
        if (constraint_count != 0U
            && program[kProgramSurfaceV1] != static_cast<std::uint32_t>(
                renderer::HostTemplateSurfaceV1::kRng
            )
            && program[kProgramSurfaceV1] != static_cast<std::uint32_t>(
                renderer::HostTemplateSurfaceV1::kPolicyMutationTape
            )) {
            return failure(
                RuntimeStatus::kConstraintContradiction,
                FailureComponent::kPrograms,
                program_index,
                kProgramConstraintCountV1
            );
        }

        for (std::uint64_t local = 0; local < constraint_count; ++local) {
            const std::uint64_t global = constraint_cursor + local;
            const std::uint64_t* const constraint = table_row(
                wire.constraints,
                global
            );
            const std::uint64_t kind = constraint[kConstraintKindV1];
            if (kind < static_cast<std::uint32_t>(
                    renderer::HostTemplateConstraintKindV1::kValuesEqual
                )
                || kind > static_cast<std::uint32_t>(
                    renderer::HostTemplateConstraintKindV1::
                        kShapeFirstDimensionEqualsInteger
                )) {
                return failure(
                    RuntimeStatus::kUnsupportedConstraintKind,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintKindV1
                );
            }
            const std::uint64_t left_offset =
                constraint[kConstraintLeftPathPoolOffsetV1];
            const std::uint64_t left_size =
                constraint[kConstraintLeftPathUtf8BytesV1];
            const std::uint64_t right_offset =
                constraint[kConstraintRightPathPoolOffsetV1];
            const std::uint64_t right_size =
                constraint[kConstraintRightPathUtf8BytesV1];
            if (left_size > UINT32_MAX || right_size > UINT32_MAX) {
                return failure(
                    RuntimeStatus::kScalarNarrowing,
                    FailureComponent::kConstraints,
                    global
                );
            }
            if (!pool_span_is_valid(wire.byte_pool, left_offset, left_size)
                || !pool_span_is_valid(wire.byte_pool, right_offset, right_size)) {
                return failure(
                    RuntimeStatus::kBytePoolBounds,
                    FailureComponent::kConstraints,
                    global
                );
            }
            if (left_offset != pool_cursor) {
                return failure(
                    RuntimeStatus::kBytePoolTraversal,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintLeftPathPoolOffsetV1
                );
            }
            const std::uint8_t* const left = pool_span(
                wire.byte_pool,
                left_offset
            );
            if (!valid_utf8(left, left_size)) {
                return failure(
                    RuntimeStatus::kInvalidUtf8,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintLeftPathPoolOffsetV1
                );
            }
            if (!valid_source_path(left, left_size)) {
                return failure(
                    RuntimeStatus::kInvalidSourcePath,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintLeftPathPoolOffsetV1
                );
            }
            pool_cursor += left_size;
            if (right_offset != pool_cursor) {
                return failure(
                    RuntimeStatus::kBytePoolTraversal,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintRightPathPoolOffsetV1
                );
            }
            const std::uint8_t* const right = pool_span(
                wire.byte_pool,
                right_offset
            );
            if (!valid_utf8(right, right_size)) {
                return failure(
                    RuntimeStatus::kInvalidUtf8,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintRightPathPoolOffsetV1
                );
            }
            if (!valid_source_path(right, right_size)) {
                return failure(
                    RuntimeStatus::kInvalidSourcePath,
                    FailureComponent::kConstraints,
                    global,
                    kConstraintRightPathPoolOffsetV1
                );
            }
            pool_cursor += right_size;
            if (spans_equal(left, left_size, right, right_size)) {
                return failure(
                    RuntimeStatus::kConstraintContradiction,
                    FailureComponent::kConstraints,
                    global
                );
            }
            const std::int64_t left_token = find_local_token(
                wire,
                token_offset,
                token_count,
                left,
                left_size
            );
            const std::int64_t right_token = find_local_token(
                wire,
                token_offset,
                token_count,
                right,
                right_size
            );
            if (left_token < 0 || right_token < 0) {
                return failure(
                    RuntimeStatus::kConstraintContradiction,
                    FailureComponent::kConstraints,
                    global
                );
            }
            const std::uint64_t* const left_spec = table_row(
                wire.tokens,
                token_offset + static_cast<std::uint64_t>(left_token)
            );
            const std::uint64_t* const right_spec = table_row(
                wire.tokens,
                token_offset + static_cast<std::uint64_t>(right_token)
            );
            if (kind == static_cast<std::uint32_t>(
                    renderer::HostTemplateConstraintKindV1::kValuesEqual
                )) {
                if (left_spec[kTokenTypeV1] != right_spec[kTokenTypeV1]
                    || left_spec[kTokenRankV1] != right_spec[kTokenRankV1]) {
                    return failure(
                        RuntimeStatus::kConstraintContradiction,
                        FailureComponent::kConstraints,
                        global
                    );
                }
            } else if (program[kProgramSurfaceV1]
                    != static_cast<std::uint32_t>(
                        renderer::HostTemplateSurfaceV1::kPolicyMutationTape
                    )
                || left_spec[kTokenTypeV1] != static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kTensorShape
                )
                || left_spec[kTokenRankV1] == 0U
                || right_spec[kTokenTypeV1] != static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kInteger
                )) {
                return failure(
                    RuntimeStatus::kConstraintContradiction,
                    FailureComponent::kConstraints,
                    global
                );
            }
        }

        token_cursor += token_count;
        part_cursor += part_count;
        constraint_cursor += constraint_count;
    }
    if (token_cursor != wire.tokens.row_count
        || part_cursor != wire.parts.row_count
        || constraint_cursor != wire.constraints.row_count
        || pool_cursor != wire.byte_pool.byte_count) {
        return failure(
            RuntimeStatus::kUnownedStorage,
            FailureComponent::kWireView
        );
    }
    if (!fits_size_t(totals->shape_dimension_count)
        || !fits_size_t(totals->sha256_hex_bytes)) {
        return failure(
            RuntimeStatus::kScalarNarrowing,
            FailureComponent::kAllocation
        );
    }
    return success();
}

}  // namespace

HostTemplateRuntimeStatusV1 prepare_host_template_runtime_v1(
    const HostTemplateWireViewV1 wire,
    HostTemplateRuntimeV1* const runtime
) noexcept {
    if (runtime == nullptr) {
        return RuntimeStatus::kNullArgument;
    }

    auto clear_storage = [runtime]() noexcept {
        runtime->programs_.clear();
        runtime->tokens_.clear();
        runtime->token_values_.clear();
        runtime->parts_.clear();
        runtime->constraints_.clear();
        runtime->byte_pool_.clear();
        runtime->shape_dimensions_.clear();
        runtime->sha256_hex_.clear();
        runtime->generation_count_ = 0;
        runtime->cell_count_ = 0;
    };
    auto record = [runtime, &clear_storage](
        const ValidationFailureV1& outcome
    ) noexcept {
        clear_storage();
        runtime->status_ = outcome.status;
        runtime->failure_ = {
            static_cast<std::uint32_t>(outcome.status),
            static_cast<std::uint32_t>(outcome.component),
            outcome.row,
            outcome.column,
            outcome.renderer_status,
            outcome.renderer_component,
            outcome.renderer_index,
            0,
        };
        return outcome.status;
    };

    clear_storage();
    runtime->status_ = RuntimeStatus::kNotPrepared;
    WireTotalsV1 totals{};
    ValidationFailureV1 outcome = validate_wire(wire, &totals);
    if (outcome.status != RuntimeStatus::kSuccess) {
        return record(outcome);
    }

    try {
        runtime->programs_.resize(
            static_cast<std::size_t>(wire.programs.row_count)
        );
        runtime->tokens_.resize(
            static_cast<std::size_t>(wire.tokens.row_count)
        );
        runtime->token_values_.resize(
            static_cast<std::size_t>(wire.tokens.row_count)
        );
        runtime->parts_.resize(
            static_cast<std::size_t>(wire.parts.row_count)
        );
        runtime->constraints_.resize(
            static_cast<std::size_t>(wire.constraints.row_count)
        );
        runtime->byte_pool_.resize(
            static_cast<std::size_t>(wire.byte_pool.byte_count)
        );
        runtime->shape_dimensions_.resize(
            static_cast<std::size_t>(totals.shape_dimension_count)
        );
        runtime->sha256_hex_.resize(
            static_cast<std::size_t>(totals.sha256_hex_bytes)
        );
    } catch (const std::bad_alloc&) {
        return record(failure(
            RuntimeStatus::kAllocationFailure,
            FailureComponent::kAllocation
        ));
    } catch (const std::length_error&) {
        return record(failure(
            RuntimeStatus::kAllocationFailure,
            FailureComponent::kAllocation
        ));
    } catch (...) {
        return record(failure(
            RuntimeStatus::kInternalException,
            FailureComponent::kAllocation
        ));
    }

    if (wire.byte_pool.byte_count != 0U) {
        std::memcpy(
            runtime->byte_pool_.data(),
            wire.byte_pool.data,
            static_cast<std::size_t>(wire.byte_pool.byte_count)
        );
    }
    for (std::uint64_t index = 0; index < wire.tokens.row_count; ++index) {
        const std::uint64_t* const source = table_row(wire.tokens, index);
        auto& token = runtime->tokens_[static_cast<std::size_t>(index)];
        token = {};
        token.token_index = static_cast<std::uint32_t>(source[kTokenIndexV1]);
        token.token_type = static_cast<std::uint32_t>(source[kTokenTypeV1]);
        token.validation_kind = static_cast<std::uint32_t>(
            source[kTokenValidationKindV1]
        );
        token.rank = static_cast<std::uint32_t>(source[kTokenRankV1]);
        token.render_pack_depth = static_cast<std::uint32_t>(
            source[kTokenRenderPackDepthV1]
        );
        token.maximum_utf8_bytes = static_cast<std::uint32_t>(
            source[kTokenMaximumUtf8BytesV1]
        );
        token.source_path_utf8_bytes = static_cast<std::uint32_t>(
            source[kTokenSourcePathUtf8BytesV1]
        );
        token.rule_parameter = static_cast<std::uint32_t>(
            source[kTokenRuleParameterV1]
        );
        token.minimum_low = source[kTokenMinimumLowV1];
        token.minimum_high = source[kTokenMinimumHighV1];
        token.maximum_low = source[kTokenMaximumLowV1];
        token.maximum_high = source[kTokenMaximumHighV1];
        token.source_path_utf8 = runtime->byte_pool_.data()
            + static_cast<std::size_t>(source[kTokenSourcePathPoolOffsetV1]);
    }
    for (std::uint64_t index = 0; index < wire.parts.row_count; ++index) {
        const std::uint64_t* const source = table_row(wire.parts, index);
        auto& part = runtime->parts_[static_cast<std::size_t>(index)];
        part = {};
        part.kind = static_cast<std::uint32_t>(source[kPartKindV1]);
        part.token_index = static_cast<std::uint32_t>(
            source[kPartTokenIndexV1]
        );
        part.literal_utf8_bytes = source[kPartLiteralUtf8BytesV1];
        if (part.kind == static_cast<std::uint32_t>(
                renderer::HostTemplatePartKindV1::kLiteralUtf8
            )) {
            part.literal_utf8 = runtime->byte_pool_.data()
                + static_cast<std::size_t>(source[kPartLiteralPoolOffsetV1]);
        }
    }
    for (std::uint64_t index = 0; index < wire.constraints.row_count; ++index) {
        const std::uint64_t* const source = table_row(wire.constraints, index);
        auto& constraint = runtime->constraints_[static_cast<std::size_t>(index)];
        constraint = {};
        constraint.kind = static_cast<std::uint32_t>(
            source[kConstraintKindV1]
        );
        constraint.left_source_path_utf8 = runtime->byte_pool_.data()
            + static_cast<std::size_t>(
                source[kConstraintLeftPathPoolOffsetV1]
            );
        constraint.left_source_path_utf8_bytes = static_cast<std::uint32_t>(
            source[kConstraintLeftPathUtf8BytesV1]
        );
        constraint.right_source_path_utf8 = runtime->byte_pool_.data()
            + static_cast<std::size_t>(
                source[kConstraintRightPathPoolOffsetV1]
            );
        constraint.right_source_path_utf8_bytes = static_cast<std::uint32_t>(
            source[kConstraintRightPathUtf8BytesV1]
        );
    }
    for (std::uint64_t index = 0; index < wire.programs.row_count; ++index) {
        const std::uint64_t* const source = table_row(wire.programs, index);
        auto& program = runtime->programs_[static_cast<std::size_t>(index)];
        program = {};
        program.abi_version = static_cast<std::uint32_t>(
            source[kProgramAbiVersionV1]
        );
        program.encoding = static_cast<std::uint32_t>(
            source[kProgramEncodingV1]
        );
        program.surface = static_cast<std::uint32_t>(
            source[kProgramSurfaceV1]
        );
        program.generation = static_cast<std::int64_t>(
            source[kProgramGenerationV1]
        );
        program.cell = static_cast<std::int64_t>(source[kProgramCellV1]);
        program.token_count = static_cast<std::uint32_t>(
            source[kProgramTokenCountV1]
        );
        program.part_count = static_cast<std::uint32_t>(
            source[kProgramPartCountV1]
        );
        program.constraint_count = static_cast<std::uint32_t>(
            source[kProgramConstraintCountV1]
        );
        program.literal_utf8_bytes = source[kProgramLiteralUtf8BytesV1];
        program.maximum_rendered_utf8_bytes =
            source[kProgramMaximumRenderedUtf8BytesV1];
        if (program.token_count != 0U) {
            program.tokens = runtime->tokens_.data()
                + static_cast<std::size_t>(source[kProgramTokenOffsetV1]);
        }
        program.parts = runtime->parts_.data()
            + static_cast<std::size_t>(source[kProgramPartOffsetV1]);
        if (program.constraint_count != 0U) {
            program.constraints = runtime->constraints_.data()
                + static_cast<std::size_t>(source[kProgramConstraintOffsetV1]);
        }
    }

    std::size_t shape_cursor = 0;
    std::size_t sha_cursor = 0;
    for (std::size_t index = 0; index < runtime->tokens_.size(); ++index) {
        const auto& token = runtime->tokens_[index];
        auto& value = runtime->token_values_[index];
        value = {};
        value.token_type = token.token_type;
        switch (static_cast<renderer::HostTemplateTokenTypeV1>(token.token_type)) {
            case renderer::HostTemplateTokenTypeV1::kInteger:
                value.scalar_low = UINT64_MAX;
                break;
            case renderer::HostTemplateTokenTypeV1::kBoolean:
                value.scalar_low = 0U;
                break;
            case renderer::HostTemplateTokenTypeV1::kPcgUint128:
                value.scalar_low = UINT64_MAX;
                value.scalar_high = UINT64_MAX;
                break;
            case renderer::HostTemplateTokenTypeV1::kPythonMtWord:
                value.scalar_low = UINT32_MAX;
                break;
            case renderer::HostTemplateTokenTypeV1::kTensorShape:
                value.shape_rank = token.rank;
                if (token.rank != 0U) {
                    value.shape_dimensions =
                        runtime->shape_dimensions_.data() + shape_cursor;
                    std::fill_n(
                        runtime->shape_dimensions_.data() + shape_cursor,
                        token.rank,
                        UINT64_MAX
                    );
                    shape_cursor += token.rank;
                }
                break;
            case renderer::HostTemplateTokenTypeV1::kSha256Hex:
                value.sha256_hex = runtime->sha256_hex_.data() + sha_cursor;
                value.sha256_hex_bytes = 64U;
                std::fill_n(
                    runtime->sha256_hex_.data() + sha_cursor,
                    64U,
                    static_cast<std::uint8_t>('f')
                );
                sha_cursor += 64U;
                break;
            case renderer::HostTemplateTokenTypeV1::kNullableFloat64: {
                const double representative = -std::numeric_limits<double>::max();
                std::memcpy(
                    &value.scalar_low,
                    &representative,
                    sizeof(representative)
                );
                value.scalar_high = 1U;
                break;
            }
        }
    }

    for (std::size_t index = 0; index < runtime->programs_.size(); ++index) {
        const auto& program = runtime->programs_[index];
        const std::size_t token_offset = program.token_count == 0U
            ? 0U
            : static_cast<std::size_t>(program.tokens - runtime->tokens_.data());
        renderer::HostTemplateResultV1 result{};
        renderer::HostTemplateRequestV1 request{};
        request.abi_version = renderer::kHostTemplateRendererAbiVersionV1;
        request.mode = static_cast<std::uint32_t>(
            renderer::HostTemplateRenderModeV1::kHashOnly
        );
        request.program = &program;
        request.token_values = program.token_count == 0U
            ? nullptr
            : runtime->token_values_.data() + token_offset;
        request.token_value_count = program.token_count;
        request.result = &result;
        const renderer::HostTemplateStatusV1 renderer_status =
            renderer::render_and_hash_host_template_v1(&request);
        if (renderer_status != renderer::HostTemplateStatusV1::kSuccess) {
            ValidationFailureV1 rejected = failure(
                RuntimeStatus::kRendererRejected,
                FailureComponent::kRenderer,
                index
            );
            rejected.renderer_status = static_cast<std::uint32_t>(
                renderer_status
            );
            rejected.renderer_component = result.failed_component;
            rejected.renderer_index = result.failed_index;
            return record(rejected);
        }
    }

    runtime->generation_count_ = wire.generation_count;
    runtime->cell_count_ = wire.cell_count;
    runtime->status_ = RuntimeStatus::kSuccess;
    runtime->failure_ = {
        static_cast<std::uint32_t>(RuntimeStatus::kSuccess),
        static_cast<std::uint32_t>(FailureComponent::kNone),
        kNoRuntimeRowV1,
        kNoRuntimeColumnV1,
        static_cast<std::uint32_t>(renderer::HostTemplateStatusV1::kSuccess),
        static_cast<std::uint32_t>(
            renderer::HostTemplateFailureComponentV1::kNone
        ),
        renderer::kNoIndexV1,
        0,
    };
    return RuntimeStatus::kSuccess;
}

HostTemplateRuntimeStatusV1 host_template_runtime_status_v1(
    const HostTemplateRuntimeV1* const runtime
) noexcept {
    return runtime == nullptr ? RuntimeStatus::kNullArgument : runtime->status_;
}

HostTemplateRuntimeFailureV1 host_template_runtime_failure_v1(
    const HostTemplateRuntimeV1* const runtime
) noexcept {
    if (runtime != nullptr) {
        return runtime->failure_;
    }
    return {
        static_cast<std::uint32_t>(RuntimeStatus::kNullArgument),
        static_cast<std::uint32_t>(FailureComponent::kWireView),
        kNoRuntimeRowV1,
        kNoRuntimeColumnV1,
        static_cast<std::uint32_t>(renderer::HostTemplateStatusV1::kSuccess),
        static_cast<std::uint32_t>(
            renderer::HostTemplateFailureComponentV1::kNone
        ),
        renderer::kNoIndexV1,
        0,
    };
}

std::uint64_t host_template_program_count_v1(
    const HostTemplateRuntimeV1* const runtime
) noexcept {
    return runtime != nullptr && runtime->status_ == RuntimeStatus::kSuccess
        ? runtime->programs_.size()
        : 0U;
}

HostTemplateProgramBindingV1 lookup_host_template_program_v1(
    HostTemplateRuntimeV1* const runtime,
    const std::uint32_t generation,
    const std::uint32_t cell,
    const std::uint32_t surface
) noexcept {
    HostTemplateProgramBindingV1 binding{};
    if (runtime == nullptr || runtime->status_ != RuntimeStatus::kSuccess
        || generation >= runtime->generation_count_
        || cell >= runtime->cell_count_
        || surface < 1U || surface > kHostTemplateSurfaceCountV1) {
        return binding;
    }
    const std::uint64_t index =
        (static_cast<std::uint64_t>(generation) * runtime->cell_count_ + cell)
            * kHostTemplateSurfaceCountV1
        + surface - 1U;
    auto& program = runtime->programs_[static_cast<std::size_t>(index)];
    binding.program = &program;
    binding.token_value_count = program.token_count;
    if (program.token_count != 0U) {
        const std::size_t token_offset = static_cast<std::size_t>(
            program.tokens - runtime->tokens_.data()
        );
        binding.token_values = runtime->token_values_.data() + token_offset;
    }
    return binding;
}

HostTemplateTokenStorageV1 lookup_host_template_token_storage_v1(
    HostTemplateRuntimeV1* const runtime,
    const std::uint32_t generation,
    const std::uint32_t cell,
    const std::uint32_t surface,
    const std::uint32_t token_index
) noexcept {
    HostTemplateTokenStorageV1 storage{};
    const HostTemplateProgramBindingV1 program =
        lookup_host_template_program_v1(runtime, generation, cell, surface);
    if (program.program == nullptr || token_index >= program.token_value_count) {
        return storage;
    }
    auto& value = program.token_values[token_index];
    storage.value = &value;
    if (value.token_type == static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kTensorShape
        )) {
        storage.shape_dimensions = const_cast<std::uint64_t*>(
            value.shape_dimensions
        );
        storage.shape_dimension_capacity = value.shape_rank;
    } else if (value.token_type == static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kSha256Hex
        )) {
        storage.sha256_hex = const_cast<std::uint8_t*>(value.sha256_hex);
        storage.sha256_hex_capacity = value.sha256_hex_bytes;
    }
    return storage;
}

const char* host_template_runtime_status_name_v1(
    const HostTemplateRuntimeStatusV1 status
) noexcept {
    switch (status) {
        case RuntimeStatus::kSuccess: return "success";
        case RuntimeStatus::kNotPrepared: return "not_prepared";
        case RuntimeStatus::kNullArgument: return "null_argument";
        case RuntimeStatus::kUnsupportedHostEndian:
            return "unsupported_host_endian";
        case RuntimeStatus::kUnsupportedWireAbi: return "unsupported_wire_abi";
        case RuntimeStatus::kWireShape: return "wire_shape";
        case RuntimeStatus::kReservedFieldNonzero:
            return "reserved_field_nonzero";
        case RuntimeStatus::kCoordinateBounds: return "coordinate_bounds";
        case RuntimeStatus::kCoordinateOrder: return "coordinate_order";
        case RuntimeStatus::kRowSliceBounds: return "row_slice_bounds";
        case RuntimeStatus::kRowSliceOrder: return "row_slice_order";
        case RuntimeStatus::kUnsupportedProgramAbi:
            return "unsupported_program_abi";
        case RuntimeStatus::kUnsupportedEncoding: return "unsupported_encoding";
        case RuntimeStatus::kUnsupportedSurface: return "unsupported_surface";
        case RuntimeStatus::kScalarNarrowing: return "scalar_narrowing";
        case RuntimeStatus::kUnsupportedTokenType:
            return "unsupported_token_type";
        case RuntimeStatus::kUnsupportedValidationKind:
            return "unsupported_validation_kind";
        case RuntimeStatus::kTokenDeclarationContradiction:
            return "token_declaration_contradiction";
        case RuntimeStatus::kUnsupportedPartKind:
            return "unsupported_part_kind";
        case RuntimeStatus::kPartContradiction: return "part_contradiction";
        case RuntimeStatus::kUnsupportedConstraintKind:
            return "unsupported_constraint_kind";
        case RuntimeStatus::kConstraintContradiction:
            return "constraint_contradiction";
        case RuntimeStatus::kBytePoolBounds: return "byte_pool_bounds";
        case RuntimeStatus::kBytePoolTraversal: return "byte_pool_traversal";
        case RuntimeStatus::kInvalidUtf8: return "invalid_utf8";
        case RuntimeStatus::kInvalidSourcePath: return "invalid_source_path";
        case RuntimeStatus::kRenderedBoundContradiction:
            return "rendered_bound_contradiction";
        case RuntimeStatus::kUnownedStorage: return "unowned_storage";
        case RuntimeStatus::kAllocationFailure: return "allocation_failure";
        case RuntimeStatus::kRendererRejected: return "renderer_rejected";
        case RuntimeStatus::kInternalException: return "internal_exception";
    }
    return "unknown";
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1
