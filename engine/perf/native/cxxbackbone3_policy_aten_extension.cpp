#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> cxxbackbone3_gru_policy_aten_v1(
    torch::Tensor x,
    torch::Tensor h_prev,
    torch::Tensor pack,
    torch::Tensor mem_dims,
    torch::Tensor aggr,
    torch::Tensor temp,
    torch::Tensor drives,
    torch::Tensor gain,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const int64_t obs,
    const int64_t hidden,
    const int64_t actions
) {
    namespace idx = torch::indexing;
    x = x.contiguous();
    h_prev = h_prev.contiguous();
    pack = pack.contiguous();
    mem_dims = mem_dims.contiguous();
    aggr = aggr.contiguous();
    temp = temp.contiguous();
    drives = drives.contiguous();
    gain = gain.contiguous();

    auto options_long = torch::TensorOptions().dtype(torch::kLong).device(h_prev.device());
    auto arange = torch::arange(hidden, options_long).unsqueeze(0);
    auto mask = (arange < torch::clamp(mem_dims, 0, hidden).unsqueeze(1)).to(h_prev.dtype());
    h_prev = h_prev * mask;

    int64_t off = 0;
    auto enc_w = pack.slice(1, off, off + hidden * obs).view({-1, hidden, obs});
    off += hidden * obs;
    auto enc_b = pack.slice(1, off, off + hidden).view({-1, hidden});
    off += hidden;
    auto wih = pack.slice(1, off, off + 3 * hidden * hidden).view({-1, 3 * hidden, hidden});
    off += 3 * hidden * hidden;
    auto bih = pack.slice(1, off, off + 3 * hidden).view({-1, 3 * hidden});
    off += 3 * hidden;
    auto whh = pack.slice(1, off, off + 3 * hidden * hidden).view({-1, 3 * hidden, hidden});
    off += 3 * hidden * hidden;
    auto bhh = pack.slice(1, off, off + 3 * hidden).view({-1, 3 * hidden});
    off += 3 * hidden;
    auto head_w = pack.slice(1, off, off + actions * hidden).view({-1, actions, hidden});
    off += actions * hidden;
    auto head_b = pack.slice(1, off, off + actions).view({-1, actions});

    auto enc_linear = torch::bmm(enc_w, x.unsqueeze(-1)).squeeze(-1) + enc_b;
    auto z_in = torch::tanh(enc_linear);
    auto input_linear = torch::bmm(wih, z_in.unsqueeze(-1)).squeeze(-1) + bih;
    auto hidden_linear = torch::bmm(whh, h_prev.unsqueeze(-1)).squeeze(-1) + bhh;
    auto input_chunks = input_linear.chunk(3, 1);
    auto hidden_chunks = hidden_linear.chunk(3, 1);
    auto reset_preact = input_chunks[0] + hidden_chunks[0];
    auto update_preact = input_chunks[1] + hidden_chunks[1];
    auto reset_gate = torch::sigmoid(reset_preact);
    auto update_gate = torch::sigmoid(update_preact);
    auto reset_hidden = reset_gate * hidden_chunks[2];
    auto candidate_preact = input_chunks[2] + reset_hidden;
    auto candidate = torch::tanh(candidate_preact);
    auto interp_left = (1.0 - update_gate) * candidate;
    auto interp_right = update_gate * h_prev;
    auto h_next = interp_left + interp_right;
    auto logits_base = torch::bmm(head_w, h_next.unsqueeze(-1)).squeeze(-1) + head_b;
    auto logits_policy = logits_base.clone();
    logits_policy.index_put_({idx::Slice(), 6}, logits_policy.index({idx::Slice(), 6}) + aggr);
    logits_policy = logits_policy / torch::clamp(temp, 0.05).unsqueeze(-1);
    auto logits = logits_policy.clone();
    logits.index_put_({idx::Slice(), 4}, logits.index({idx::Slice(), 4}) + gain * (w_hunger_gather * drives.index({idx::Slice(), 0})));
    logits.index_put_(
        {idx::Slice(), 5},
        logits.index({idx::Slice(), 5}) + gain * (w_pain_rest * drives.index({idx::Slice(), 1}) + w_fatigue_rest * drives.index({idx::Slice(), 2}))
    );
    logits.index_put_(
        {idx::Slice(), 6},
        logits.index({idx::Slice(), 6}) - gain * (w_pain_attack * drives.index({idx::Slice(), 1}) + w_fear_attack * drives.index({idx::Slice(), 3}))
    );
    auto actions_out = std::get<1>(logits.max(1)).to(torch::kInt32);
    return {
        enc_linear,
        z_in,
        input_linear,
        hidden_linear,
        reset_preact,
        update_preact,
        input_chunks[2],
        hidden_chunks[2],
        reset_gate,
        update_gate,
        reset_hidden,
        candidate_preact,
        candidate,
        interp_left,
        interp_right,
        h_next,
        logits_base,
        logits_policy,
        logits,
        actions_out,
    };
}

std::vector<torch::Tensor> cxxbackbone4_mlp_policy_aten_v1(
    torch::Tensor x,
    torch::Tensor pack,
    torch::Tensor aggr,
    torch::Tensor temp,
    torch::Tensor drives,
    torch::Tensor gain,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const int64_t obs,
    const int64_t hidden,
    const int64_t actions
) {
    namespace idx = torch::indexing;
    x = x.contiguous();
    pack = pack.contiguous();
    aggr = aggr.contiguous();
    temp = temp.contiguous();
    drives = drives.contiguous();
    gain = gain.contiguous();

    int64_t off = 0;
    auto w1 = pack.slice(1, off, off + hidden * obs).view({-1, hidden, obs});
    off += hidden * obs;
    auto b1 = pack.slice(1, off, off + hidden).view({-1, hidden});
    off += hidden;
    auto w2 = pack.slice(1, off, off + hidden * hidden).view({-1, hidden, hidden});
    off += hidden * hidden;
    auto b2 = pack.slice(1, off, off + hidden).view({-1, hidden});
    off += hidden;
    auto w3 = pack.slice(1, off, off + actions * hidden).view({-1, actions, hidden});
    off += actions * hidden;
    auto b3 = pack.slice(1, off, off + actions).view({-1, actions});

    auto h1_linear = torch::bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1;
    auto h1 = torch::tanh(h1_linear);
    auto h2_linear = torch::bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2;
    auto h2 = torch::tanh(h2_linear);
    auto logits_base = torch::bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3;
    auto logits_policy = logits_base.clone();
    logits_policy.index_put_({idx::Slice(), 6}, logits_policy.index({idx::Slice(), 6}) + aggr);
    logits_policy = logits_policy / torch::clamp(temp, 0.05).unsqueeze(-1);
    auto logits = logits_policy.clone();
    logits.index_put_({idx::Slice(), 4}, logits.index({idx::Slice(), 4}) + gain * (w_hunger_gather * drives.index({idx::Slice(), 0})));
    logits.index_put_(
        {idx::Slice(), 5},
        logits.index({idx::Slice(), 5}) + gain * (w_pain_rest * drives.index({idx::Slice(), 1}) + w_fatigue_rest * drives.index({idx::Slice(), 2}))
    );
    logits.index_put_(
        {idx::Slice(), 6},
        logits.index({idx::Slice(), 6}) - gain * (w_pain_attack * drives.index({idx::Slice(), 1}) + w_fear_attack * drives.index({idx::Slice(), 3}))
    );
    auto actions_out = std::get<1>(logits.max(1)).to(torch::kInt32);
    return {h1_linear, h1, h2_linear, h2, logits_base, logits_policy, logits, actions_out};
}
