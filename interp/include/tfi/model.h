// Inference-only re-implementation of tforge.DecoderLM with named hook points.
//
// Hook names and activation semantics match tforge exactly (minus the batch dimension),
// e.g. "blocks.0.attn.hook_pattern" is (H, T, T) here and (1, H, T, T) in PyTorch.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tfi/config.h"
#include "tfi/tensor.h"

namespace tfi {

namespace detail {
struct RowBatch;  // internal (src/ops.h)
class Gpu;        // internal (src/gpu.h, macOS)

// A model's GPU context, created on first use. Copies start empty: the context caches the
// weights by address, and a copy's weights live elsewhere.
struct GpuSlot {
    std::shared_ptr<Gpu> gpu;
    bool checked = false;
    GpuSlot() = default;
    GpuSlot(const GpuSlot&) {}
    GpuSlot& operator=(const GpuSlot&) {
        gpu.reset();
        checked = false;
        return *this;
    }
};
}  // namespace detail

using Tokens = std::vector<int64_t>;
using ActivationCache = std::map<std::string, Tensor>;
// Called with the hook name and the activation; may modify the activation in place.
using HookFn = std::function<void(const std::string& name, Tensor& act)>;
using NameFilter = std::function<bool(const std::string& name)>;

class Hooks {
  public:
    Hooks& add(const std::string& name, HookFn fn);       // exact hook name
    Hooks& add_filter(NameFilter filter, HookFn fn);      // every hook the filter accepts
    Hooks& extend(const Hooks& other);
    void run(const std::string& name, Tensor& act) const;  // in registration order
    bool matches(const std::string& name) const;            // whether run(name) would call anything
    bool empty() const { return entries_.empty(); }

  private:
    std::vector<std::pair<NameFilter, HookFn>> entries_;
};

class Model {
  public:
    Model(ModelConfig cfg, std::map<std::string, Tensor> weights);
    static Model load(const std::string& path);  // safetensors file written by tforge.export

    const ModelConfig& cfg() const { return cfg_; }
    const ModelConfig& layer_cfg(int64_t layer) const { return layer_cfgs_.at(layer); }

    // Logits (T, vocab) for one sequence. Hooks may read or overwrite activations.
    Tensor forward(const Tokens& tokens, const Hooks* hooks = nullptr) const;
    // Run only blocks layer..n_layers-1 (then the final norm and unembedding), starting from
    // `resid`, the (T, d_model) input to block `layer`: what hook_resid_pre of that block sees.
    // With a cached blocks.<layer>.hook_resid_pre this reproduces forward() exactly while skipping
    // the earlier layers, which is how ablation avoids recomputing them (patching uses forward_rows).
    Tensor forward_from(int64_t layer, Tensor resid, const Hooks* hooks = nullptr) const;
    // Internal, for patching: run blocks layer..n_layers-1 on several sequences' rows stacked into
    // one (rows, d_model) residual (see detail::RowBatch). Hooks see the stacked activations
    // (hook_pattern is not called). Returns the logits of each segment's last row: (segments, vocab).
    Tensor forward_rows(int64_t layer, Tensor resid, const detail::RowBatch& batch, const Hooks* hooks = nullptr) const;
    // Internal, for ablation: like forward_rows, but returns every row after the final norm,
    // (rows, d_model), for the caller to unembed (logits_from_final) in pieces.
    Tensor forward_rows_final(int64_t layer, Tensor resid, const detail::RowBatch& batch,
                              const Hooks* hooks = nullptr) const;
    // Internal, for attribution patching (macOS, needs batch.gpu): forward and backward passes of
    // several whole sequences stacked as in forward_rows (every segment starts at position 0), from
    // `x0`, the (rows, d_model) input to block 0, everything but attention on the GPU.
    // `dmetric(s, logits)` gives dF/d(logits) for segment s's scalar F, from the logits (1, vocab) of
    // its last row. Returns each segment's gradients, as run_with_gradients would, for the hook points
    // the filter accepts among hook_embed, hook_resid_pre / _mid / _post, hook_attn_out, hook_mlp_out
    // and attn.hook_z (no others).
    std::vector<ActivationCache> gradients_rows(const Tensor& x0, const detail::RowBatch& batch,
                                                const std::function<Tensor(int64_t, const Tensor&)>& dmetric,
                                                const NameFilter& filter) const;
    // Internal (macOS): this model's GPU context for batched patching, created on first use and
    // kept with its copies of the weights; nullptr without a Metal device or on other systems.
    detail::Gpu* gpu() const;
    // Like forward, but also stores activations (after `hooks` ran) for hooks the filter accepts.
    Tensor run_with_cache(const Tokens& tokens, ActivationCache& cache, NameFilter filter = nullptr,
                          const Hooks* hooks = nullptr) const;
    std::vector<std::string> hook_names() const;

    // Forward pass, then a backward pass from a gradient on the logits: `dlogits(logits)` returns
    // dF/dlogits (T, vocab) for some scalar function F of the logits. For every hook point the
    // filter accepts, `grads` receives dF/d(activation), the activation's shape. Each hook point is
    // its own variable: e.g. hook_resid_pre's gradient counts every path from it, hook_attn_in's
    // only the path through attention (as PyTorch autograd on tforge's hook points gives).
    // The forward pass is forward()'s. `hooks` may edit activations; the backward pass then treats
    // each edited value as the activation itself (e.g. replacing hook_embed runs the model on other
    // embeddings). If `acts` is given it receives every activation. Returns the logits.
    Tensor run_with_gradients(const Tokens& tokens, const std::function<Tensor(const Tensor& logits)>& dlogits,
                              ActivationCache& grads, NameFilter filter = nullptr,
                              ActivationCache* acts = nullptr, const Hooks* hooks = nullptr) const;

    const Tensor& weight(const std::string& name) const;  // throws, naming the missing tensor
    const Tensor* weight_or_null(const std::string& name) const;
    const Tensor& unembed() const;  // (vocab, d_model); the embedding when tied
    int64_t num_parameters() const;
    int64_t max_heads() const;  // largest head count of any layer (layer_overrides may change n_heads)
    const std::map<std::string, Tensor>& weights() const { return w_; }

    // Optional vocabulary (one string per token id), read from the file when tforge.export
    // was given one. Without it, tokens are plain ids.
    const std::vector<std::string>& vocab() const { return vocab_; }
    void set_vocab(std::vector<std::string> vocab);
    std::string token_str(int64_t id) const;     // the token's text, or its id
    Tokens tokenize(const std::string& text) const;  // greedy longest match; needs a vocabulary

    // Building blocks reused by the analysis tools.
    Tensor norm(const std::string& prefix, const ModelConfig& c, const Tensor& x) const;  // over the last dim
    Tensor final_norm(const Tensor& resid) const;    // ln_final (identity for post-norm models)
    Tensor logits_from_final(const Tensor& normed) const;  // unembed + final softcap
    // The same, with the unembedding on the GPU when `batch` allows it (as in forward_rows).
    Tensor logits_from_final(const Tensor& normed, const detail::RowBatch* batch) const;
    // The input to block 0, (T, d_model): token (+ positional) embeddings, as hook_resid_pre of
    // block 0 sees it. Calls hook_embed and hook_pos_embed.
    Tensor embed(const Tokens& tokens, const Hooks* hooks = nullptr) const;

  private:
    ModelConfig cfg_;
    std::vector<ModelConfig> layer_cfgs_;
    std::map<std::string, Tensor> w_;
    std::vector<std::string> vocab_;
    mutable detail::GpuSlot gpu_;

    Tensor attention(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const;
    // Attention from the projections q (T, H*D), k and v (T, Hkv*D): qk_norm, RoPE, hook_q/k/v, the
    // pattern and hook_z. Returns z as (T, H*D), before o_proj.
    Tensor attend(int64_t layer, Tensor q, Tensor k, Tensor v, const Hooks* hooks, const detail::RowBatch* batch) const;
    // Blocks layer..n_layers-1 of a batched pass; returns the residual stream after the last one.
    Tensor run_rows(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks) const;
    Tensor mlp(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const;
    Tensor block(int64_t layer, Tensor x, const Hooks* hooks, const detail::RowBatch* batch = nullptr) const;
    Tensor positional_embedding(int64_t T) const;
};

}  // namespace tfi
