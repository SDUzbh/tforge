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

// Patching, ablation and head_means run their independent forward passes on this many threads.
// The default (n = 0) is the TFI_THREADS environment variable, else every core. Results are the
// same for any thread count.
void set_num_threads(int n);
int num_threads();

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
    // Each matrix is (n_layers, widest layer's head count); NaN where a layer has fewer heads.
    Tensor previous_token;   // attention to t-1
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
    Tensor effect;  // (rows, cols): patched metric - corrupted metric; NaN only where a site does not exist
    Tensor values;  // effect / gap(): 0 = corrupted behaviour, 1 = clean. All NaN when !normalized()
    float clean_metric = 0, corrupt_metric = 0;

    float gap() const { return clean_metric - corrupt_metric; }
    // False when clean and corrupted metrics are (numerically) equal: the corruption doesn't change
    // the metric, so there is nothing to restore and `values` is undefined. Use `effect` instead.
    // A small but non-zero gap still normalizes, but amplifies noise; check gap() before trusting `values`.
    bool normalized() const;
};
// site: "resid_pre", "resid_mid", "resid_post", "attn_out" or "mlp_out". Rows are layers, columns positions.
PatchResult patch_residual(const Model& model, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                           int64_t pos, int64_t target, int64_t other = -1);
// Patch each head's output (hook_z, all positions). Rows are layers, columns heads.
PatchResult patch_heads(const Model& model, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other = -1);

// --- attribution patching ----------------------------------------------------------------
// d logit_metric(pos, target, other) / d(activation) for every hook point the filter accepts
// (see Model::run_with_gradients). Returns the metric.
float metric_gradients(const Model& model, const Tokens& tokens, int64_t pos, int64_t target, int64_t other,
                       ActivationCache& grads, NameFilter filter = nullptr, ActivationCache* acts = nullptr);

// Attribution patching (Nanda 2023; Syed et al. 2023): the same tables as patch_residual and
// patch_heads, estimated to first order from two forward passes and one backward pass instead of
// one forward pass per site:
//     effect ~ (clean activation - corrupted activation) . d metric / d activation (corrupted run)
// Exact when everything downstream of a site is linear in it (e.g. the last layer of a model without
// a final norm), and usually good for small changes. It can be badly off where a change saturates a
// softmax or crosses a nonlinearity, typically early layers and attention patterns, so confirm the
// sites it ranks highest with exact patching.
//
// steps > 1 averages the gradient over `steps` runs with token embeddings interpolated from the
// corrupted input towards the clean one (alpha = 0, 1/steps, ...), as in EAP-IG (Hanna et al. 2024):
// an integrated-gradients estimate that tracks large, saturating changes much better, for one more
// forward + backward pass per step.
PatchResult attribution_patch_residual(const Model& model, const Tokens& clean, const Tokens& corrupt,
                                       const std::string& site, int64_t pos, int64_t target, int64_t other = -1,
                                       int64_t steps = 1);
PatchResult attribution_patch_heads(const Model& model, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other = -1, int64_t steps = 1);

// --- ablation -----------------------------------------------------------------------------
enum class Ablation { Zero, Mean };

// Every head's average output (hook_z) over reference sequences, used for mean ablation.
// The mean comes from separate sequences, never from the input being ablated: averaging over the
// input's own positions would copy later tokens into earlier positions and break causality.
// If every reference has length T, means are per position: one (T, H, D) tensor per layer.
// Otherwise there is a single mean over all positions: (1, H, D) per layer.
struct HeadMeans {
    std::vector<Tensor> layers;
    bool per_position() const { return !layers.empty() && layers[0].shape[0] > 1; }
};
HeadMeans head_means(const Model& model, const std::vector<Tokens>& reference, int64_t T);

// Hooks that replace one head's output at every position with zeros, or with its reference mean
// (`means` is required for Ablation::Mean and must outlive the returned hooks). Combine with
// other hooks freely.
Hooks ablate_head(const Model& model, int64_t layer, int64_t head, Ablation mode, const HeadMeans* means = nullptr);

// Increase in next-token loss when each head is ablated: (n_layers, max heads), NaN where a layer
// has fewer heads. Mean ablation needs `reference` sequences from the same distribution as `tokens`.
Tensor head_ablation(const Model& model, const Tokens& tokens, Ablation mode,
                     const std::vector<Tokens>& reference = {});

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
    Tensor q, k, v;                   // (total heads, total heads), heads ordered as in labels
};
Composition composition_scores(const Model& model);
// Mean composition score for random Gaussian heads of the same shapes.
float composition_random_baseline(const Model& model, int64_t n_pairs = 64, uint64_t seed = 0);

// Fraction of (up to max_tokens evenly spaced) tokens that the path
// embed -> OV(path[0]) -> OV(path[1]) -> ... -> unembed ranks first as themselves.
// A path is a list of (layer, head), earliest first. Norm scales and centering are ignored.
float ov_path_copying(const Model& model, const std::vector<std::pair<int64_t, int64_t>>& path,
                      int64_t max_tokens = 1024);
// ov_path_copying for every single head: (n_layers, max heads), NaN where a layer has fewer heads.
Tensor ov_copying_scores(const Model& model, int64_t max_tokens = 1024);

// Learned positions only: the (T, T) causal attention pattern of a head when queries and keys
// carry nothing but their positional embedding.
Tensor positional_qk(const Model& model, int64_t layer, int64_t head, int64_t T);

}  // namespace tfi
