#ifndef EIEL_CUDA_ARCHITECTURE_B_GMNW_CONTROL_MATRIX_V1_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_GMNW_CONTROL_MATRIX_V1_HPP_

#include <cstdint>

namespace eiel::cuda_architecture_b::gmnw_control_matrix_v1 {

inline constexpr std::int32_t kAbiVersionV1 = 1;
inline constexpr std::int8_t kPersistentGruArmV1 = 0;
inline constexpr std::int8_t kResetGruArmV1 = 1;
inline constexpr std::int8_t kMatchedMlpArmV1 = 2;
inline constexpr std::int8_t kMixedSelectionArmV1 = 3;
inline constexpr std::int8_t kPersistentGruImmediateArmV1 = 4;
inline constexpr std::int8_t kMatchedMlpImmediateArmV1 = 5;
inline constexpr std::int8_t kMixedSelectionImmediateArmV1 = 6;
inline constexpr std::int64_t kMatchedMlpObservationWidthV1 = 12;
inline constexpr std::int64_t kMatchedMlpHiddenWidthV1 = 512;
inline constexpr std::int64_t kMatchedMlpActionCountV1 = 7;
inline constexpr std::int64_t kMatchedMlpPackedWidthV1 = 272903;
inline constexpr std::int64_t kMatchedMlpNeuralScratchWidthV1 = 1024;
inline constexpr std::int64_t kMatchedMlpAuditTicksV1 = 160;
inline constexpr std::int64_t kMatchedMlpAuditWordsPerTickV1 = 2;
inline constexpr std::int64_t kMatchedMlpScratchWidthV1 =
    kMatchedMlpNeuralScratchWidthV1
    + kMatchedMlpAuditTicksV1 * kMatchedMlpAuditWordsPerTickV1;

struct GmnwControlMatrixContextV1 {
    bool enabled;
    const std::int8_t* policy_arm;
    const float* matched_mlp_pack;
    float* matched_mlp_scratch;
    std::int32_t* status;
    std::int64_t cells;
};

}  // namespace eiel::cuda_architecture_b::gmnw_control_matrix_v1

#endif
