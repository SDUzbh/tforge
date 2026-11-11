#include "tfi/model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "tfi/safetensors.h"

namespace tfi {

// --- hooks ---------------------------------------------------------------------

Hooks& Hooks::add(const std::string& name, HookFn fn) {
    entries_.emplace_back([name](const std::string& n) { return n == name; }, std::move(fn));
    return *this;
}

Hooks& Hooks::add_filter(NameFilter filter, HookFn fn) {
    entries_.emplace_back(std::move(filter), std::move(fn));
    return *this;
}

Hooks& Hooks::extend(const Hooks& other) {
    entries_.insert(entries_.end(), other.entries_.begin(), other.entries_.end());
    return *this;
}

void Hooks::run(const std::string& name, Tensor& act) const {
    for (const auto& e : entries_)
        if (!e.first || e.first(name)) e.second(name, act);
}

namespace {

void hook(const Hooks* hooks, const std::string& name, Tensor& act) {
    if (hooks) hooks->run(name, act);
}

std::string layer_prefix(int64_t layer) { return "blocks." + std::to_string(layer) + "."; }

// M_PI and friends are POSIX extensions that strict -std=c++17 builds (e.g. glibc) do not define.
constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtHalf = 0.70710678118654752440;

void activate_(Tensor& t, const std::string& act) {
    if (act == "relu") {
        for (float& v : t.data) v = v > 0 ? v : 0.f;
    } else if (act == "gelu") {
        for (float& v : t.data) v = 0.5f * v * (1.f + std::erf(v * float(kSqrtHalf)));
    } else if (act == "gelu_tanh") {
        const float k = std::sqrt(2.f / float(kPi));
        for (float& v : t.data) v = 0.5f * v * (1.f + std::tanh(k * (v + 0.044715f * v * v * v)));
    } else if (act == "silu") {
        for (float& v : t.data) v = v / (1.f + std::exp(-v));
    } else if (act == "tanh") {
        for (float& v : t.data) v = std::tanh(v);
    } else if (act == "relu2") {
        for (float& v : t.data) v = v > 0 ? v * v : 0.f;
    } else {
        throw std::runtime_error("unsupported activation " + act);
    }
}

// (T, H*D) -> (H, T, D)
Tensor split_heads(const Tensor& x, int64_t H, int64_t D) {
    const int64_t T = x.shape[0];
    Tensor out({H, T, D});
    for (int64_t t = 0; t < T; ++t)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t d = 0; d < D; ++d) out.data[(h * T + t) * D + d] = x.data[t * H * D + h * D + d];
    return out;
}

// Rotate (H, T, D) in place at positions 0..T-1, tforge's "rotate half" convention.
void apply_rope_(Tensor& x, const ModelConfig& c) {
    const int64_t H = x.shape[0], T = x.shape[1], D = x.shape[2];
    int64_t rot = static_cast<int64_t>(D * c.rope_fraction);
    rot -= rot % 2;
    if (rot == 0) throw std::runtime_error("rope_fraction * head_dim leaves no dimensions to rotate");
    const int64_t half = rot / 2;
    std::vector<float> inv_freq(half);
    for (int64_t i = 0; i < half; ++i)
        inv_freq[i] = 1.0f / std::pow(static_cast<float>(c.rope_theta), static_cast<float>(2 * i) / static_cast<float>(rot));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < half; ++i) {
            const float a = static_cast<float>(t) * inv_freq[i], cs = std::cos(a), sn = std::sin(a);
            for (int64_t h = 0; h < H; ++h) {
                float* v = x.ptr() + (h * T + t) * D;
                const float x1 = v[i], x2 = v[i + half];
                v[i] = x1 * cs - x2 * sn;
                v[i + half] = x2 * cs + x1 * sn;
            }
        }
}

std::vector<float> alibi_slopes(int64_t n) {
    auto pow2 = [](int64_t m) {
        std::vector<double> s(m);
        const double start = std::pow(2.0, -8.0 / double(m));
        for (int64_t i = 0; i < m; ++i) s[i] = std::pow(start, double(i + 1));
        return s;
    };
    std::vector<double> s;
    const double lg = std::log2(double(n));
    if (lg == std::floor(lg)) s = pow2(n);
    else {
        const int64_t closest = int64_t(1) << int64_t(std::floor(lg));
        s = pow2(closest);
        const std::vector<double> extra = pow2(2 * closest);
        for (int64_t i = 0; i < n - closest; ++i) s.push_back(extra[2 * i]);
    }
    return std::vector<float>(s.begin(), s.end());
}

}  // namespace

// --- model -----------------------------------------------------------------------

Model::Model(ModelConfig cfg, std::map<std::string, Tensor> weights) : cfg_(std::move(cfg)), w_(std::move(weights)) {
    cfg_.validate();
    for (int64_t i = 0; i < cfg_.n_layers; ++i) layer_cfgs_.push_back(cfg_.layer_config(i));
    const Tensor& E = weight("embed.weight");
    if (E.shape != Shape{cfg_.vocab_size, cfg_.d_model})
        throw std::runtime_error("embed.weight has shape " + shape_str(E.shape) + ", config says " +
                                 shape_str({cfg_.vocab_size, cfg_.d_model}));
    if (unembed().shape != E.shape) throw std::runtime_error("unembed.weight shape does not match the embedding");
}

Model Model::load(const std::string& path) {
    SafeTensors st = load_safetensors(path);
    auto it = st.metadata.find("config");
    if (it == st.metadata.end())
        throw std::runtime_error(path + " has no tforge config; export it with tforge.export.export_model");
    return Model(ModelConfig::from_json(it->second), std::move(st.tensors));
}

const Tensor* Model::weight_or_null(const std::string& name) const {
    auto it = w_.find(name);
    return it == w_.end() ? nullptr : &it->second;
}

const Tensor& Model::weight(const std::string& name) const {
    const Tensor* t = weight_or_null(name);
    if (!t) throw std::runtime_error("model file is missing tensor '" + name + "'");
    return *t;
}

const Tensor& Model::unembed() const {
    if (const Tensor* u = weight_or_null("unembed.weight")) return *u;
    if (!cfg_.tie_embeddings) throw std::runtime_error("model file is missing tensor 'unembed.weight'");
    return weight("embed.weight");
}

int64_t Model::num_parameters() const {
    int64_t n = 0;
    for (const auto& kv : w_) n += kv.second.numel();
    return n;
}

std::vector<std::string> Model::hook_names() const {
    std::vector<std::string> names = {"hook_embed"};
    if (cfg_.embedding_positional()) names.push_back("hook_pos_embed");
    for (int64_t i = 0; i < cfg_.n_layers; ++i) {
        const std::string p = layer_prefix(i);
        for (const char* n : {"hook_resid_pre", "hook_attn_in", "attn.hook_q", "attn.hook_k", "attn.hook_v",
                              "attn.hook_pattern", "attn.hook_z", "hook_attn_out", "hook_resid_mid"})
            names.push_back(p + n);
        if (layer_cfgs_[i].has_mlp())
            for (const char* n : {"hook_mlp_in", "mlp.hook_pre", "mlp.hook_post", "hook_mlp_out"}) names.push_back(p + n);
        names.push_back(p + "hook_resid_post");
    }
    names.push_back("hook_final");
    return names;
}

Tensor Model::norm(const std::string& prefix, const ModelConfig& c, const Tensor& x) const {
    if (c.norm == "none") return x;
    const Tensor& w = weight(prefix + ".weight");
    const Tensor* b = weight_or_null(prefix + ".bias");
    const int64_t d = x.size(-1), n = x.numel() / d;
    if (w.numel() != d) throw std::runtime_error(prefix + ".weight does not match the normalized size");
    const float eps = static_cast<float>(c.norm_eps);
    Tensor y(x.shape);
    for (int64_t r = 0; r < n; ++r) {
        const float* xi = x.ptr() + r * d;
        float* yi = y.ptr() + r * d;
        if (c.norm == "layernorm") {
            double mean = 0, var = 0;
            for (int64_t j = 0; j < d; ++j) mean += xi[j];
            mean /= d;
            for (int64_t j = 0; j < d; ++j) var += (xi[j] - mean) * (xi[j] - mean);
            var /= d;
            const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
            for (int64_t j = 0; j < d; ++j)
                yi[j] = (xi[j] - static_cast<float>(mean)) * inv * w.data[j] + (b ? b->data[j] : 0.f);
        } else {  // rmsnorm
            double ms = 0;
            for (int64_t j = 0; j < d; ++j) ms += double(xi[j]) * xi[j];
            const float inv = 1.0f / std::sqrt(static_cast<float>(ms / d) + eps);
            for (int64_t j = 0; j < d; ++j) yi[j] = xi[j] * inv * w.data[j];
        }
    }
    return y;
}

Tensor Model::positional_embedding(int64_t T) const {
    const int64_t d = cfg_.d_model;
    if (cfg_.positional == "learned") {
        if (T > cfg_.max_seq_len)
            throw std::runtime_error("sequence length " + std::to_string(T) + " exceeds max_seq_len " +
                                     std::to_string(cfg_.max_seq_len));
        return rows(weight("pos_embed.weight"), 0, T);
    }
    // sinusoidal: [sin(pos * f_i), cos(pos * f_i)], f_i = exp(-ln(10000) * i / half), zero-padded if d is odd
    const int64_t half = d / 2;
    Tensor pe({T, d});
    for (int64_t i = 0; i < half; ++i) {
        const float f = std::exp(static_cast<float>(-std::log(10000.0)) * static_cast<float>(i) / static_cast<float>(half));
        for (int64_t t = 0; t < T; ++t) {
            const float a = static_cast<float>(t) * f;
            pe.data[t * d + i] = std::sin(a);
            pe.data[t * d + half + i] = std::cos(a);
        }
    }
    return pe;
}

Tensor Model::attention(int64_t layer, const Tensor& x, const Hooks* hooks) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string p = layer_prefix(layer) + "attn.";
    const int64_t T = x.shape[0], H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim(), rep = H / Hkv;

    Tensor q = split_heads(linear(x, weight(p + "q_proj.weight"), weight_or_null(p + "q_proj.bias")), H, D);
    Tensor k = split_heads(linear(x, weight(p + "k_proj.weight"), weight_or_null(p + "k_proj.bias")), Hkv, D);
    Tensor v = split_heads(linear(x, weight(p + "v_proj.weight"), weight_or_null(p + "v_proj.bias")), Hkv, D);
    if (c.qk_norm) {
        ModelConfig qc = c;
        if (qc.norm == "none") qc.norm = "rmsnorm";
        q = norm(p + "q_norm", qc, q);
        k = norm(p + "k_norm", qc, k);
    }
    if (c.positional == "rope") {
        apply_rope_(q, c);
        apply_rope_(k, c);
    }
    hook(hooks, p + "hook_q", q);
    hook(hooks, p + "hook_k", k);
    hook(hooks, p + "hook_v", v);

    const float scale = static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(D)));
    const std::vector<float> slopes = c.positional == "alibi" ? alibi_slopes(H) : std::vector<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();

    Tensor pattern({H, T, T});
    for (int64_t h = 0; h < H; ++h) {
        const int64_t g = h / rep;
        Tensor qh({T, D}, std::vector<float>(q.data.begin() + h * T * D, q.data.begin() + (h + 1) * T * D));
        Tensor kh({T, D}, std::vector<float>(k.data.begin() + g * T * D, k.data.begin() + (g + 1) * T * D));
        Tensor s = matmul(qh, kh, /*transpose_b=*/true);
        for (int64_t i = 0; i < T; ++i) {
            float* rowp = pattern.ptr() + (h * T + i) * T;
            for (int64_t j = 0; j < T; ++j) {
                float val = s.data[i * T + j] * scale;
                if (c.attn_logit_softcap) {
                    const float cap = static_cast<float>(*c.attn_logit_softcap);
                    val = cap * std::tanh(val / cap);
                }
                if (!slopes.empty()) val += -slopes[h] * static_cast<float>(std::max<int64_t>(i - j, 0));
                const bool allowed = j <= i && (!c.sliding_window || i - j < *c.sliding_window);
                rowp[j] = allowed ? val : neg_inf;
            }
            softmax_(rowp, T);
        }
    }
    hook(hooks, p + "hook_pattern", pattern);

    Tensor z({T, H, D});
    for (int64_t h = 0; h < H; ++h) {
        const int64_t g = h / rep;
        Tensor ph({T, T}, std::vector<float>(pattern.data.begin() + h * T * T, pattern.data.begin() + (h + 1) * T * T));
        Tensor vh({T, D}, std::vector<float>(v.data.begin() + g * T * D, v.data.begin() + (g + 1) * T * D));
        Tensor zh = matmul(ph, vh);
        for (int64_t t = 0; t < T; ++t)
            std::copy(zh.ptr() + t * D, zh.ptr() + (t + 1) * D, z.ptr() + (t * H + h) * D);
    }
    hook(hooks, p + "hook_z", z);
    return linear(z.reshape({T, H * D}), weight(p + "o_proj.weight"), weight_or_null(p + "o_proj.bias"));
}

Tensor Model::mlp(int64_t layer, const Tensor& x, const Hooks* hooks) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string p = layer_prefix(layer) + "mlp.";
    const std::string act = c.mlp_activation();
    Tensor post;
    if (c.mlp == "mlp") {
        Tensor pre = linear(x, weight(p + "up.weight"), weight_or_null(p + "up.bias"));
        hook(hooks, p + "hook_pre", pre);
        post = pre;
        activate_(post, act);
    } else {  // glu, swiglu, geglu: down(act(gate(x)) * up(x))
        Tensor pre = linear(x, weight(p + "gate.weight"), weight_or_null(p + "gate.bias"));
        hook(hooks, p + "hook_pre", pre);
        post = pre;
        activate_(post, act);
        post = mul(post, linear(x, weight(p + "up.weight"), weight_or_null(p + "up.bias")));
    }
    hook(hooks, p + "hook_post", post);
    return linear(post, weight(p + "down.weight"), weight_or_null(p + "down.bias"));
}

Tensor Model::block(int64_t layer, Tensor x, const Hooks* hooks) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string p = layer_prefix(layer);
    const bool post = c.norm_position == "post", sandwich = c.norm_position == "sandwich";
    hook(hooks, p + "hook_resid_pre", x);

    Tensor attn_in = post ? x : norm(p + "ln1", c, x);
    hook(hooks, p + "hook_attn_in", attn_in);
    Tensor attn_out = attention(layer, attn_in, hooks);
    if (sandwich) attn_out = norm(p + "ln1_post", c, attn_out);
    hook(hooks, p + "hook_attn_out", attn_out);

    auto run_mlp = [&](Tensor mlp_in) {
        hook(hooks, p + "hook_mlp_in", mlp_in);
        Tensor out = mlp(layer, mlp_in, hooks);
        if (sandwich) out = norm(p + "ln2_post", c, out);
        hook(hooks, p + "hook_mlp_out", out);
        return out;
    };

    if (c.parallel_residual) {
        Tensor mid = add(x, attn_out);
        hook(hooks, p + "hook_resid_mid", mid);
        Tensor out = c.has_mlp() ? add(mid, run_mlp(norm(p + "ln2", c, x))) : mid;
        hook(hooks, p + "hook_resid_post", out);
        return out;
    }

    add_(x, attn_out);
    if (post) x = norm(p + "ln1", c, x);
    hook(hooks, p + "hook_resid_mid", x);
    if (c.has_mlp()) {
        add_(x, run_mlp(post ? x : norm(p + "ln2", c, x)));
        if (post) x = norm(p + "ln2", c, x);
    }
    hook(hooks, p + "hook_resid_post", x);
    return x;
}

Tensor Model::final_norm(const Tensor& resid) const {
    if (cfg_.norm_position == "post") return resid;  // post-norm blocks already end in a norm
    return norm("ln_final", cfg_, resid);
}

Tensor Model::logits_from_final(const Tensor& normed) const {
    Tensor logits = linear(normed, unembed());
    if (cfg_.final_logit_softcap) {
        const float cap = static_cast<float>(*cfg_.final_logit_softcap);
        for (float& v : logits.data) v = cap * std::tanh(v / cap);
    }
    return logits;
}

Tensor Model::forward(const Tokens& tokens, const Hooks* hooks) const {
    const int64_t T = static_cast<int64_t>(tokens.size()), d = cfg_.d_model;
    if (T == 0) throw std::invalid_argument("forward needs at least one token");
    const Tensor& E = weight("embed.weight");
    Tensor x({T, d});
    for (int64_t t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= cfg_.vocab_size)
            throw std::out_of_range("token " + std::to_string(tokens[t]) + " is outside the vocabulary");
        std::copy(E.ptr() + tokens[t] * d, E.ptr() + (tokens[t] + 1) * d, x.ptr() + t * d);
    }
    if (cfg_.embed_scale) x = scale(x, static_cast<float>(std::sqrt(double(d))));
    hook(hooks, "hook_embed", x);
    if (cfg_.embedding_positional()) {
        Tensor pe = positional_embedding(T);
        hook(hooks, "hook_pos_embed", pe);
        add_(x, pe);
    }
    for (int64_t i = 0; i < cfg_.n_layers; ++i) x = block(i, std::move(x), hooks);
    Tensor f = final_norm(x);
    hook(hooks, "hook_final", f);
    return logits_from_final(f);
}

Tensor Model::run_with_cache(const Tokens& tokens, ActivationCache& cache, NameFilter filter, const Hooks* hooks) const {
    Hooks all;
    if (hooks) all.extend(*hooks);
    all.add_filter(std::move(filter), [&cache](const std::string& name, Tensor& act) { cache[name] = act; });
    return forward(tokens, &all);
}

}  // namespace tfi
