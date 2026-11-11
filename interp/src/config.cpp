#include "tfi/config.h"

#include <set>
#include <stdexcept>

namespace tfi {

namespace {

template <typename T>
std::optional<T> opt_int(const json::Value& v) {
    if (v.is_null()) return std::nullopt;
    return static_cast<T>(v.as_int());
}

std::optional<double> opt_num(const json::Value& v) {
    if (v.is_null()) return std::nullopt;
    return v.as_number();
}

// Fields tforge writes that only matter for training.
const std::set<std::string> kTrainingOnly = {"use_sdpa", "dropout", "attn_dropout", "init_std", "scale_residual_init"};

}  // namespace

std::string ModelConfig::mlp_activation() const {
    if (mlp == "swiglu") return "silu";
    if (mlp == "geglu") return activation.rfind("gelu", 0) == 0 ? activation : "gelu";
    return activation;
}

void ModelConfig::apply(const json::Value& f) {
    if (!f.is_object()) throw std::runtime_error("config: expected a JSON object");
    for (size_t i = 0; i < f.keys.size(); ++i) {
        const std::string& k = f.keys[i];
        const json::Value& v = f.values[i];
        if (k == "vocab_size") vocab_size = v.as_int();
        else if (k == "d_model") d_model = v.as_int();
        else if (k == "n_layers") n_layers = v.as_int();
        else if (k == "n_heads") n_heads = v.as_int();
        else if (k == "n_kv_heads") n_kv_heads = opt_int<int64_t>(v);
        else if (k == "d_head") d_head = opt_int<int64_t>(v);
        else if (k == "d_mlp") d_mlp = opt_int<int64_t>(v);
        else if (k == "mlp_multiple_of") mlp_multiple_of = v.as_int();
        else if (k == "max_seq_len") max_seq_len = v.as_int();
        else if (k == "norm") norm = v.as_string();
        else if (k == "norm_eps") norm_eps = v.as_number();
        else if (k == "norm_position") norm_position = v.as_string();
        else if (k == "positional") positional = v.as_string();
        else if (k == "rope_theta") rope_theta = v.as_number();
        else if (k == "rope_fraction") rope_fraction = v.as_number();
        else if (k == "attention") attention = v.as_string();
        else if (k == "mlp") mlp = v.as_string();
        else if (k == "activation") activation = v.as_string();
        else if (k == "qk_norm") qk_norm = v.as_bool();
        else if (k == "sliding_window") sliding_window = opt_int<int64_t>(v);
        else if (k == "attn_scale") attn_scale = opt_num(v);
        else if (k == "attn_logit_softcap") attn_logit_softcap = opt_num(v);
        else if (k == "parallel_residual") parallel_residual = v.as_bool();
        else if (k == "bias") bias = v.as_bool();
        else if (k == "tie_embeddings") tie_embeddings = v.as_bool();
        else if (k == "embed_scale") embed_scale = v.as_bool();
        else if (k == "final_logit_softcap") final_logit_softcap = opt_num(v);
        else if (k == "layer_overrides") {
            layer_overrides.clear();
            for (size_t j = 0; j < v.keys.size(); ++j) layer_overrides[std::stoll(v.keys[j])] = v.values[j];
        } else if (!kTrainingOnly.count(k)) {
            throw std::runtime_error("config: unknown field '" + k + "' (exported by a newer tforge?)");
        }
    }
}

ModelConfig ModelConfig::layer_config(int64_t layer) const {
    auto it = layer_overrides.find(layer);
    if (it == layer_overrides.end()) return *this;
    ModelConfig c = *this;
    c.layer_overrides.clear();
    c.apply(it->second);
    return c;
}

void ModelConfig::validate() const {
    auto one_of = [](const std::string& field, const std::string& v, const std::set<std::string>& ok) {
        if (!ok.count(v)) throw std::runtime_error("config: " + field + " '" + v + "' is not supported by tfi");
    };
    one_of("norm", norm, {"rmsnorm", "layernorm", "none"});
    one_of("norm_position", norm_position, {"pre", "post", "sandwich"});
    one_of("positional", positional, {"rope", "alibi", "learned", "sinusoidal", "none"});
    one_of("attention", attention, {"causal"});
    one_of("mlp", mlp, {"mlp", "glu", "swiglu", "geglu", "none"});
    one_of("activation", mlp_activation(), {"relu", "gelu", "gelu_tanh", "silu", "tanh", "relu2"});
    if (n_heads % kv_heads()) throw std::runtime_error("config: n_heads must be a multiple of n_kv_heads");
    if (!d_head && d_model % n_heads) throw std::runtime_error("config: d_model not divisible by n_heads");
}

ModelConfig ModelConfig::from_json(const std::string& text) {
    ModelConfig c;
    c.apply(json::parse(text));
    c.validate();
    for (const auto& kv : c.layer_overrides) {
        if (kv.first < 0 || kv.first >= c.n_layers) throw std::runtime_error("config: layer override out of range");
        c.layer_config(kv.first).validate();
    }
    return c;
}

}  // namespace tfi
