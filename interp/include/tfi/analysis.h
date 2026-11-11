// Post-training reverse-engineering tools.
//
// Activation-based (need inputs): logit lens, direct logit attribution, attention-head
// pattern scores, activation patching, head ablation.
// Weight-based (no inputs): per-head QK/OV matrices, Q/K/V composition scores between
// heads, OV copying scores, positional QK patterns.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tfi/model.h"

namespace tfi {

// Logit(target) - logit(other) at `pos`; with other < 0, the log-probability of target.
float logit_metric(const Tensor& logits, int64_t pos, int64_t target, int64_t other = -1);
// Mean next-token cross-entropy over the sequence.
float next_token_loss(const Tensor& logits, const Tokens& tokens);

// --- logit lens ---------------------------------------------------------------------
// Decode the residual stream after every sublayer with the final norm and unembedding.
struct LensEntry {
    std::string stage;  // "embed", "L0.attn", "L0.mlp", ...
    std::vector<int64_t> top_tokens;
    std::vector<float> top_probs;
    float target_prob = -1;    // when a target is given
    int64_t target_rank = -1;  // 0 = top prediction
};
std::vector<LensEntry> logit_lens(const Model& model, const Tokens& tokens, int64_t pos, int64_t target = -1,
                                  int64_t k = 5);

// --- direct logit attribution --------------------------------------------------------
// Split the final logit difference into contributions from the embeddings, every attention
// head (or whole attention layer for sandwich norm), every MLP and the norm bias, treating
// the final norm's scale as fixed. Values sum to `actual`. Not defined for post-norm models.
struct DLAResult {
    std::vector<std::string> labels;
    std::vector<float> values;
    float sum = 0;     // sum of values
    float actual = 0;  // the model's logit difference before any final softcap
};
DLAResult direct_logit_attribution(const Model& model, const Tokens& tokens, int64_t pos, int64_t target,
                                   int64_t other = -1);

// --- attention-head pattern scores ------------------------------------------------------
// Averaged over random sequences made of a segment repeated twice, at queries in the repeat.
struct TwoHop {
    std::string writer, reader;  // e.g. "L0H3", "L1H1"
    float score;                 // virtual attention (P_reader @ P_writer) to the token to copy
};
struct HeadScores {
    Tensor previous_token;   // (n_layers, n_heads) attention to t-1
    Tensor induction;        // to the token after the earlier copy of the current token
    Tensor duplicate_token;  // to the earlier copy of the current token
    Tensor first_token;      // to position 0 (attention sink)
    Tensor self;             // to t itself
    std::vector<TwoHop> top_two_hop;  // best writer->reader pairs, any lag
};
HeadScores head_scores(const Model& model, int64_t segment_len = 24, int64_t n_samples = 8, uint64_t seed = 0);

// --- activation patching ----------------------------------------------------------------
// Run the corrupted input, copy one activation from the clean run, and measure how much of
// the clean metric comes back: 0 = corrupted behaviour, 1 = clean behaviour.
struct PatchResult {
    std::vector<std::string> row_labels, col_labels;
    Tensor values;  // (rows, cols); NaN where a site does not exist
    float clean_metric = 0, corrupt_metric = 0;
};
// site: "resid_pre", "resid_mid", "resid_post", "attn_out" or "mlp_out". Rows are layers, columns positions.
PatchResult patch_residual(const Model& model, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                           int64_t pos, int64_t target, int64_t other = -1);
// Patch each head's output (hook_z, all positions). Rows are layers, columns heads.
PatchResult patch_heads(const Model& model, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other = -1);

// --- ablation -----------------------------------------------------------------------------
enum class Ablation { Zero, Mean };
// Increase in next-token loss when each head's output (hook_z) is ablated: (n_layers, n_heads).
Tensor head_ablation(const Model& model, const Tokens& tokens, Ablation mode);

// --- weight-based circuits -------------------------------------------------------------------
// One head's matrices with the attention-input norm's weight folded into W_Q/W_K/W_V.
// W_Q, W_K, W_V are (d_head, d_model); W_O is (d_model, d_head).
struct HeadWeights {
    Tensor W_Q, W_K, W_V, W_O;
};
HeadWeights head_weights(const Model& model, int64_t layer, int64_t head);
Tensor ov_matrix(const HeadWeights& h);  // (d_model, d_model) = W_O W_V
Tensor qk_matrix(const HeadWeights& h);  // (d_model, d_model) = W_Q^T W_K, score = x_q^T M x_k

// Composition scores (Elhage et al. 2021): entry [i][j] measures how much head j reads the
// output of an earlier head i through its queries (q), keys (k) or values (v). NaN unless
// head i sits in an earlier layer than head j. Compare against `random_baseline`.
struct Composition {
    std::vector<std::string> labels;  // "L0H0", "L0H1", ...
    Tensor q, k, v;                   // (n_layers*n_heads, n_layers*n_heads)
};
Composition composition_scores(const Model& model);
// Mean composition score for random Gaussian heads of the same shapes.
float composition_random_baseline(const Model& model, int64_t n_pairs = 64, uint64_t seed = 0);

// Fraction of (up to max_tokens evenly spaced) tokens that the path
// embed -> OV(path[0]) -> OV(path[1]) -> ... -> unembed ranks first as themselves.
// A path is a list of (layer, head), earliest first. Norm scales and centering are ignored.
float ov_path_copying(const Model& model, const std::vector<std::pair<int64_t, int64_t>>& path,
                      int64_t max_tokens = 1024);
// ov_path_copying for every single head: (n_layers, n_heads).
Tensor ov_copying_scores(const Model& model, int64_t max_tokens = 1024);

// Learned positions only: the (T, T) causal attention pattern of a head when queries and keys
// carry nothing but their positional embedding.
Tensor positional_qk(const Model& model, int64_t layer, int64_t head, int64_t T);

}  // namespace tfi
