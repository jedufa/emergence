#include <torch/extension.h>

#include <vector>

void tensor_world_c4_resident_population_r11_graph_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, std::vector<torch::Tensor> identity,
    std::vector<torch::Tensor> rng_staging,
    std::vector<torch::Tensor> event_ring,
    torch::Tensor status,
    torch::Tensor persistent_outer_status,
    torch::Tensor generation, torch::Tensor tick, torch::Tensor requested_ticks,
    torch::Tensor completed_ticks, int64_t max_births,
    double parent_energy_share, double minimum_energy_buffer,
    int64_t cooldown_steps, double start_health, double max_energy,
    int64_t life_history_mode, int64_t maturity_start_ticks,
    int64_t maturity_rise_ticks, int64_t fertility_decline_start_ticks,
    int64_t fertility_decline_ticks, double elder_fertility_floor,
    int64_t environment_stream, int64_t runner_stream);

std::vector<int64_t> tensor_world_c4_resident_r11_newborn_fanout_contract();
int64_t tensor_world_c4_resident_r11_graph_node_count(int64_t raw_graph);
void tensor_world_c4_resident_r11_merge_window_control_cuda(
    torch::Tensor preflight_status, torch::Tensor global_status,
    torch::Tensor c4_status, torch::Tensor commit_status,
    torch::Tensor stop_requested, torch::Tensor first_error_stage,
    torch::Tensor first_error_world, torch::Tensor first_error_code,
    torch::Tensor first_error_tick, torch::Tensor tick,
    torch::Tensor completed_ticks, torch::Tensor current_tick,
    torch::Tensor receipt_write_index);
void tensor_world_c4_resident_r11_advance_window_clock_cuda(
    torch::Tensor global_status, torch::Tensor tick, torch::Tensor generation);
void tensor_world_c4_resident_r11_prepare_v2_window_replay_cuda(
    torch::Tensor global_status, torch::Tensor completed_ticks,
    torch::Tensor receipt_write_index, torch::Tensor generation,
    torch::Tensor generation_base, torch::Tensor live_consumer,
    torch::Tensor live_producer, torch::Tensor shadow_consumer,
    torch::Tensor shadow_producer);
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
void tensor_world_c4_resident_r11_test_only_configure_phase_timing_cuda(
    torch::Tensor markers);
void tensor_world_c4_resident_r11_test_only_clear_phase_timing_cuda();
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
void tensor_world_c4_resident_r11_test_only_configure_active_preflight_cuda(
    torch::Tensor active_count, torch::Tensor alive_sum,
    torch::Tensor tail_valid, torch::Tensor order_valid,
    torch::Tensor device_pointers, torch::Tensor checksum, int64_t mode);
std::vector<int64_t>
tensor_world_c4_resident_r11_test_only_active_preflight_enqueue_metadata();
void tensor_world_c4_resident_r11_test_only_clear_active_preflight_cuda();
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK
void tensor_world_c4_resident_r11_test_only_identity_capture_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, std::vector<torch::Tensor> identity,
    torch::Tensor status);
void tensor_world_c4_resident_r11_test_only_identity_validate_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, std::vector<torch::Tensor> identity,
    torch::Tensor status);
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
void tensor_world_c4_resident_r11_test_only_configure_alias_guard_cuda(
    torch::Tensor control, torch::Tensor local_status,
    torch::Tensor planned_births, torch::Tensor writer_reached,
    torch::Tensor writer_skipped, torch::Tensor outer_status,
    torch::Tensor injected);
void tensor_world_c4_resident_r11_test_only_clear_alias_guard_cuda();
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Tensor World C4 R11 fixed-graph population transaction";
    module.def(
        "population_transaction_device_time",
        &tensor_world_c4_resident_population_r11_graph_cuda);
    module.def(
        "newborn_fanout_contract",
        &tensor_world_c4_resident_r11_newborn_fanout_contract);
    module.def(
        "captured_graph_node_count",
        &tensor_world_c4_resident_r11_graph_node_count);
    module.def(
        "merge_window_control",
        &tensor_world_c4_resident_r11_merge_window_control_cuda);
    module.def(
        "advance_window_clock",
        &tensor_world_c4_resident_r11_advance_window_clock_cuda);
    module.def(
        "prepare_v2_window_replay",
        &tensor_world_c4_resident_r11_prepare_v2_window_replay_cuda);
#ifdef EIEL_C4_PHASE_B_SURFACE_TIMING_HOOK
    module.def(
        "_test_only_configure_phase_timing",
        &tensor_world_c4_resident_r11_test_only_configure_phase_timing_cuda);
    module.def(
        "_test_only_clear_phase_timing",
        &tensor_world_c4_resident_r11_test_only_clear_phase_timing_cuda);
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_ACTIVE_PREFLIGHT_HOOK
    module.def(
        "_test_only_configure_active_preflight",
        &tensor_world_c4_resident_r11_test_only_configure_active_preflight_cuda);
    module.def(
        "_test_only_active_preflight_enqueue_metadata",
        &tensor_world_c4_resident_r11_test_only_active_preflight_enqueue_metadata);
    module.def(
        "_test_only_clear_active_preflight",
        &tensor_world_c4_resident_r11_test_only_clear_active_preflight_cuda);
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_IDENTITY_HOOK
    module.def(
        "_test_only_identity_capture",
        &tensor_world_c4_resident_r11_test_only_identity_capture_cuda);
    module.def(
        "_test_only_identity_validate",
        &tensor_world_c4_resident_r11_test_only_identity_validate_cuda);
#endif
#ifdef EIEL_C4_R11_TEST_ONLY_ALIAS_GUARD_HOOK
    module.def(
        "_test_only_configure_alias_guard",
        &tensor_world_c4_resident_r11_test_only_configure_alias_guard_cuda);
    module.def(
        "_test_only_clear_alias_guard",
        &tensor_world_c4_resident_r11_test_only_clear_alias_guard_cuda);
#endif
}
