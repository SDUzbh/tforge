// Mirror of indranet.config.ModelConfig, parsed from the JSON stored in exported models.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "tfi/json.h"

namespace tfi {

struct ModelConfig {
    int64_t vocab_size = 32000;
    int64_t d_model = 512;
    int64_t n_layers = 8;
    int64_t n_heads = 8;
    std::optional<int64_t> n_kv_heads;
    std::optional<int64_t> d_head;
    std::optional<int64_t> d_mlp;
    int64_t mlp_multiple_of = 64;
    int64_t max_seq_len = 2048;

    std::string norm = "rmsnorm";
    double norm_eps = 1e-5;
    std::string norm_position = "pre";
    std::string positional = "rope";
    double rope_theta = 10000.0;
    double rope_fraction = 1.0;
    std::string attention = "causal";
    std::string mlp = "swiglu";
    std::string activation = "gelu";

    bool qk_norm = false;
    std::optional<int64_t> sliding_window;
    std::optional<double> attn_scale;
    std::optional<double> attn_logit_softcap;

    bool parallel_residual = false;
    bool bias = false;
    bool tie_embeddings = true;
    bool embed_scale = false;
    std::optional<double> final_logit_softcap;

    std::map<int64_t, json::Value> layer_overrides;  // layer -> object of field overrides

    int64_t head_dim() const { return d_head ? *d_head : d_model / n_heads; }
    int64_t kv_heads() const { return n_kv_heads ? *n_kv_heads : n_heads; }
    bool has_mlp() const { return mlp != "none"; }
    bool embedding_positional() const { return positional == "learned" || positional == "sinusoidal"; }
    // The activation the MLP actually uses ("swiglu" forces silu, "geglu" a GELU variant).
    std::string mlp_activation() const;

    ModelConfig layer_config(int64_t layer) const;  // this config plus that layer's overrides
    void apply(const json::Value& fields);          // set every field present in a JSON object
    void validate() const;

    static ModelConfig from_json(const std::string& text);
};

}  // namespace tfi
