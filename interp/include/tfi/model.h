// Inference-only re-implementation of tforge.DecoderLM with named hook points.
//
// Hook names and activation semantics match tforge exactly (minus the batch dimension),
// e.g. "blocks.0.attn.hook_pattern" is (H, T, T) here and (1, H, T, T) in PyTorch.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tfi/config.h"
#include "tfi/tensor.h"

namespace tfi {

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
    // Like forward, but also stores activations (after `hooks` ran) for hooks the filter accepts.
    Tensor run_with_cache(const Tokens& tokens, ActivationCache& cache, NameFilter filter = nullptr,
                          const Hooks* hooks = nullptr) const;
    std::vector<std::string> hook_names() const;

    const Tensor& weight(const std::string& name) const;  // throws, naming the missing tensor
    const Tensor* weight_or_null(const std::string& name) const;
    const Tensor& unembed() const;  // (vocab, d_model); the embedding when tied
    int64_t num_parameters() const;
    const std::map<std::string, Tensor>& weights() const { return w_; }

    // Building blocks reused by the analysis tools.
    Tensor norm(const std::string& prefix, const ModelConfig& c, const Tensor& x) const;  // over the last dim
    Tensor final_norm(const Tensor& resid) const;    // ln_final (identity for post-norm models)
    Tensor logits_from_final(const Tensor& normed) const;  // unembed + final softcap

  private:
    ModelConfig cfg_;
    std::vector<ModelConfig> layer_cfgs_;
    std::map<std::string, Tensor> w_;

    Tensor attention(int64_t layer, const Tensor& x, const Hooks* hooks) const;
    Tensor mlp(int64_t layer, const Tensor& x, const Hooks* hooks) const;
    Tensor block(int64_t layer, Tensor x, const Hooks* hooks) const;
    Tensor positional_embedding(int64_t T) const;
};

}  // namespace tfi
