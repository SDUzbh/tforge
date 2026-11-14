#include "tfi/model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "tfi/safetensors.h"
#include "ops.h"
#include "vmath.h"
#include "gpu.h"

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

bool Hooks::matches(const std::string& name) const {
    for (const auto& e : entries_)
        if (!e.first || e.first(name)) return true;
    return false;
}

namespace {

void hook(const Hooks* hooks, const std::string& name, Tensor& act) {
    if (hooks) hooks->run(name, act);
}

std::string layer_prefix(int64_t layer) { return "blocks." + std::to_string(layer) + "."; }

// M_PI and friends are POSIX extensions that strict -std=c++17 builds (e.g. glibc) do not define.
constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtHalf = 0.70710678118654752440;

}  // namespace

namespace detail {

void activate_(Tensor& t, const std::string& act) {
    if (act == "relu") {
        for (float& v : t.data) v = v > 0 ? v : 0.f;
    } else if (act == "gelu") {
        for (float& v : t.data) v = 0.5f * v * (1.f + vm::erf(v * float(kSqrtHalf)));
    } else if (act == "gelu_tanh") {
        const float k = std::sqrt(2.f / float(kPi));
        for (float& v : t.data) v = 0.5f * v * (1.f + vm::tanh(k * (v + 0.044715f * v * v * v)));
    } else if (act == "silu") {
        for (float& v : t.data) v = v / (1.f + vm::exp(-v));
    } else if (act == "tanh") {
        for (float& v : t.data) v = vm::tanh(v);
    } else if (act == "relu2") {
        for (float& v : t.data) v = v > 0 ? v * v : 0.f;
    } else {
        throw std::runtime_error("unsupported activation " + act);
    }
}

void activation_grad_(Tensor& t, const std::string& act) {
    if (act == "relu") {
        for (float& v : t.data) v = v > 0 ? 1.f : 0.f;
    } else if (act == "gelu") {  // Phi(x) + x phi(x)
        const float inv_sqrt_2pi = static_cast<float>(1.0 / std::sqrt(2.0 * kPi));
        for (float& v : t.data) v = 0.5f * (1.f + vm::erf(v * float(kSqrtHalf))) + v * inv_sqrt_2pi * vm::exp(-0.5f * v * v);
    } else if (act == "gelu_tanh") {
        const float k = std::sqrt(2.f / float(kPi));
        for (float& v : t.data) {
            const float th = vm::tanh(k * (v + 0.044715f * v * v * v));
            v = 0.5f * (1.f + th) + 0.5f * v * (1.f - th * th) * k * (1.f + 3.f * 0.044715f * v * v);
        }
    } else if (act == "silu") {
        for (float& v : t.data) {
            const float s = 1.f / (1.f + vm::exp(-v));
            v = s * (1.f + v * (1.f - s));
        }
    } else if (act == "tanh") {
        for (float& v : t.data) {
            const float th = vm::tanh(v);
            v = 1.f - th * th;
        }
    } else if (act == "relu2") {
        for (float& v : t.data) v = v > 0 ? 2.f * v : 0.f;
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

Tensor merge_heads(const Tensor& x) {
    const int64_t H = x.shape[0], T = x.shape[1], D = x.shape[2];
    Tensor out({T, H * D});
    for (int64_t h = 0; h < H; ++h)
        for (int64_t t = 0; t < T; ++t)
            std::copy(x.ptr() + (h * T + t) * D, x.ptr() + (h * T + t + 1) * D, out.ptr() + t * H * D + h * D);
    return out;
}

void apply_rope_(Tensor& x, const ModelConfig& c, bool inverse, const std::vector<int64_t>* positions) {
    const int64_t H = x.shape[0], T = x.shape[1], D = x.shape[2];
    int64_t rot = static_cast<int64_t>(D * c.rope_fraction);
    rot -= rot % 2;
    if (rot == 0) throw std::runtime_error("rope_fraction * head_dim leaves no dimensions to rotate");
    const int64_t half = rot / 2;
    std::vector<float> inv_freq(half);
    for (int64_t i = 0; i < half; ++i)
        inv_freq[i] = 1.0f / std::pow(static_cast<float>(c.rope_theta), static_cast<float>(2 * i) / static_cast<float>(rot));
    // cos / sin per position, computed once even when many rows share a position.
    int64_t n_pos = T;
    if (positions)
        for (int64_t p : *positions) n_pos = std::max(n_pos, p + 1);
    std::vector<float> cs(n_pos * half), sn(n_pos * half);
    for (int64_t t = 0; t < n_pos; ++t)
        for (int64_t i = 0; i < half; ++i) {
            const float a = static_cast<float>(t) * inv_freq[i];
            cs[t * half + i] = std::cos(a);
            sn[t * half + i] = inverse ? -std::sin(a) : std::sin(a);
        }
    for (int64_t t = 0; t < T; ++t) {
        const int64_t pt = positions ? (*positions)[t] : t;
        for (int64_t i = 0; i < half; ++i) {
            const float c_ = cs[pt * half + i], s_ = sn[pt * half + i];
            for (int64_t h = 0; h < H; ++h) {
                float* v = x.ptr() + (h * T + t) * D;
                const float x1 = v[i], x2 = v[i + half];
                v[i] = x1 * c_ - x2 * s_;
                v[i + half] = x2 * c_ + x1 * s_;
            }
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

}  // namespace detail

using detail::activate_;
using detail::alibi_slopes;
using detail::apply_rope_;
using detail::split_heads;

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
        throw std::runtime_error(path + " has no IndraNet model config; export it with indranet.export.export_model");
    Model m(ModelConfig::from_json(it->second), std::move(st.tensors));
    auto v = st.metadata.find("vocab");
    if (v != st.metadata.end()) {
        std::vector<std::string> vocab;
        for (const auto& s : json::parse(v->second).array) vocab.push_back(s.as_string());
        m.set_vocab(std::move(vocab));
    }
    return m;
}

void Model::set_vocab(std::vector<std::string> vocab) {
    if (!vocab.empty() && static_cast<int64_t>(vocab.size()) != cfg_.vocab_size)
        throw std::runtime_error("vocabulary has " + std::to_string(vocab.size()) + " entries, model has " +
                                 std::to_string(cfg_.vocab_size));
    vocab_ = std::move(vocab);
}

std::string Model::token_str(int64_t id) const {
    if (id >= 0 && id < static_cast<int64_t>(vocab_.size())) return vocab_[id];
    return std::to_string(id);
}

Tokens Model::tokenize(const std::string& text) const {
    if (vocab_.empty()) throw std::runtime_error("this model file has no vocabulary; pass token ids instead");
    Tokens out;
    size_t i = 0;
    while (i < text.size()) {
        int64_t best = -1;
        size_t best_len = 0;
        for (size_t id = 0; id < vocab_.size(); ++id) {
            const std::string& tok = vocab_[id];
            if (tok.size() > best_len && text.compare(i, tok.size(), tok) == 0) {
                best = static_cast<int64_t>(id);
                best_len = tok.size();
            }
        }
        if (best < 0) throw std::runtime_error("no token matches the text at offset " + std::to_string(i) + ": '" + text.substr(i, 10) + "'");
        out.push_back(best);
        i += best_len;
    }
    return out;
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

int64_t Model::max_heads() const {
    int64_t h = 0;
    for (const auto& c : layer_cfgs_) h = std::max(h, c.n_heads);
    return h;
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
    const float* wp = w.ptr();
    const bool layernorm = c.norm == "layernorm";
    const float* bp = layernorm && b ? b->ptr() : nullptr;  // rmsnorm has no bias
    for (int64_t r = 0; r < n; ++r) {
        const float* xi = x.ptr() + r * d;
        float* yi = y.ptr() + r * d;
        // Two passes (mean, then squared deviations), each summed in 8 lanes (see vmath.h).
        const float mean = layernorm ? vm::sum(xi, d) / static_cast<float>(d) : 0.f;
        const float inv = 1.0f / std::sqrt(vm::sum_sq_dev(xi, d, mean) / static_cast<float>(d) + eps);
        if (bp)
            for (int64_t j = 0; j < d; ++j) yi[j] = (xi[j] - mean) * inv * wp[j] + bp[j];
        else
            for (int64_t j = 0; j < d; ++j) yi[j] = (xi[j] - mean) * inv * wp[j];
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

namespace {

// linear() with the weight `prefix`.weight (and .bias if present); on the GPU for large batches.
Tensor project(const Model& m, const Tensor& x, const std::string& prefix, const detail::RowBatch* batch) {
    const Tensor& W = m.weight(prefix + ".weight");
    const Tensor* b = m.weight_or_null(prefix + ".bias");
#ifdef TFI_USE_ACCELERATE
    if (batch && batch->gpu && x.size(-1) && x.numel() / x.size(-1) >= detail::gpu_min_rows())
        return batch->gpu->linear(x, W, b, batch->fp16);
#else
    (void)batch;
#endif
    return linear(x, W, b);
}

// project() for several weights that read the same x; on the GPU they share one submission.
std::vector<Tensor> project_many(const Model& m, const Tensor& x, const std::vector<std::string>& prefixes,
                                 const detail::RowBatch* batch) {
#ifdef TFI_USE_ACCELERATE
    if (batch && batch->gpu && x.size(-1) && x.numel() / x.size(-1) >= detail::gpu_min_rows()) {
        std::vector<const Tensor*> Ws, bs;
        for (const std::string& p : prefixes) {
            Ws.push_back(&m.weight(p + ".weight"));
            bs.push_back(m.weight_or_null(p + ".bias"));
        }
        return batch->gpu->linear_many(x, Ws, bs, batch->fp16);
    }
#endif
    std::vector<Tensor> out;
    for (const std::string& p : prefixes) out.push_back(project(m, x, p, batch));
    return out;
}

// Attention of n queries at positions first, first+1, ... over keys 0..first+n-1 for one head.
// Keys and values of positions before `first` come from (kp, vp), the rest from (kn, vn), all
// rows of D floats. `probs` (n rows, stride ld >= first + n) receives the attention pattern: each
// row is masked and softmaxed over the keys it may attend to (j <= its position, within the
// window); disallowed entries are exactly 0.
struct HeadAttention {
    int64_t D;
    float scale, cap;  // cap 0 = no softcap
    float slope;       // ALiBi slope, 0 = none
    int64_t window;    // 0 = none

    void pattern(const float* q, int64_t n, int64_t first, const float* kp, const float* kn, float* probs,
                 int64_t ld) const {
        if (first) detail::gemm(/*trans_b=*/true, n, first, D, q, D, kp, D, probs, ld);
        detail::gemm(/*trans_b=*/true, n, n, D, q, D, kn, D, probs + first, ld);
        for (int64_t r = 0; r < n; ++r) {
            const int64_t i = first + r;  // query position
            float* rowp = probs + r * ld;
            const int64_t j0 = window ? std::max<int64_t>(0, i - window + 1) : 0;
            float* seg = rowp + j0;
            const int64_t m = i + 1 - j0;
            for (int64_t j = 0; j < m; ++j) seg[j] *= scale;
            if (cap)
                for (int64_t j = 0; j < m; ++j) seg[j] = cap * vm::tanh(seg[j] / cap);
            if (slope != 0.f)
                for (int64_t j = 0; j < m; ++j) seg[j] += -slope * static_cast<float>(i - j0 - j);
            vm::softmax(seg, m);
            std::fill(rowp, seg, 0.f);
            std::fill(rowp + i + 1, rowp + first + n, 0.f);
        }
    }

    // out (n rows, stride ldo) = probs @ values.
    void apply(const float* probs, int64_t n, int64_t first, int64_t ld, const float* vp, const float* vn, float* out,
               int64_t ldo) const {
        if (first) detail::gemm(/*trans_b=*/false, n, D, first, probs, ld, vp, D, out, ldo);
        detail::gemm(/*trans_b=*/false, n, D, n, probs + first, ld, vn, D, out, ldo, /*accumulate=*/first > 0);
    }
};

}  // namespace

Tensor Model::attention(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const {
    const std::string p = layer_prefix(layer) + "attn.";
    std::vector<Tensor> qkv = project_many(*this, x, {p + "q_proj", p + "k_proj", p + "v_proj"}, batch);
    Tensor z = attend(layer, std::move(qkv[0]), std::move(qkv[1]), std::move(qkv[2]), hooks, batch);
    return project(*this, z, p + "o_proj", batch);
}

Tensor Model::attend(int64_t layer, Tensor q2, Tensor k2, Tensor v2, const Hooks* hooks,
                     const detail::RowBatch* batch) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string p = layer_prefix(layer) + "attn.";
    const int64_t T = q2.shape[0], H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim(), rep = H / Hkv;

    Tensor q = split_heads(q2, H, D);
    Tensor k = split_heads(k2, Hkv, D);
    Tensor v = split_heads(v2, Hkv, D);
    q2 = k2 = v2 = Tensor();
    if (c.qk_norm) {
        ModelConfig qc = c;
        if (qc.norm == "none") qc.norm = "rmsnorm";
        q = norm(p + "q_norm", qc, q);
        k = norm(p + "k_norm", qc, k);
    }
    if (c.positional == "rope") {
        std::vector<int64_t> positions;
        if (batch)
            for (size_t s = 0; s < batch->start.size(); ++s)
                for (int64_t r = 0; r < batch->len[s]; ++r) positions.push_back(batch->start[s] + r);
        apply_rope_(q, c, false, batch ? &positions : nullptr);
        apply_rope_(k, c, false, batch ? &positions : nullptr);
    }
    hook(hooks, p + "hook_q", q);
    hook(hooks, p + "hook_k", k);
    hook(hooks, p + "hook_v", v);

    const std::vector<float> slopes = c.positional == "alibi" ? alibi_slopes(H) : std::vector<float>();
    HeadAttention ha{D, static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(D))),
                     c.attn_logit_softcap ? static_cast<float>(*c.attn_logit_softcap) : 0.f, 0.f,
                     c.sliding_window ? *c.sliding_window : 0};
    Tensor z({T, H, D});
    if (!batch) {
        Tensor pattern({H, T, T});
        for (int64_t h = 0; h < H; ++h) {
            ha.slope = slopes.empty() ? 0.f : slopes[h];
            const float* kh = k.ptr() + (h / rep) * T * D;
            ha.pattern(q.ptr() + h * T * D, T, 0, nullptr, kh, pattern.ptr() + h * T * T, T);
        }
        hook(hooks, p + "hook_pattern", pattern);
        for (int64_t h = 0; h < H; ++h) {
            const float* vh = v.ptr() + (h / rep) * T * D;
            ha.apply(pattern.ptr() + h * T * T, T, 0, T, nullptr, vh, z.ptr() + h * D, H * D);
        }
    } else {  // stacked segments: no hook_pattern (each segment has its own pattern)
        std::vector<float> probs;
        for (size_t s = 0; s < batch->start.size(); ++s) {
            const int64_t first = batch->start[s], n = batch->len[s], off = batch->offset[s], nk = first + n;
            const Tensor* pk = first ? batch->prefix_k.at(layer) : nullptr;
            const Tensor* pv = first ? batch->prefix_v.at(layer) : nullptr;
            const int64_t Tp = pk ? pk->shape[1] : 0;
            probs.resize(n * nk);
            for (int64_t h = 0; h < H; ++h) {
                const int64_t g = h / rep;
                ha.slope = slopes.empty() ? 0.f : slopes[h];
                ha.pattern(q.ptr() + (h * T + off) * D, n, first, pk ? pk->ptr() + g * Tp * D : nullptr,
                           k.ptr() + (g * T + off) * D, probs.data(), nk);
                ha.apply(probs.data(), n, first, nk, pv ? pv->ptr() + g * Tp * D : nullptr, v.ptr() + (g * T + off) * D,
                         z.ptr() + (off * H + h) * D, H * D);
            }
        }
    }
    hook(hooks, p + "hook_z", z);
    return std::move(z).reshape({T, H * D});
}

Tensor Model::mlp(int64_t layer, const Tensor& x, const Hooks* hooks, const detail::RowBatch* batch) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string p = layer_prefix(layer) + "mlp.";
    const std::string act = c.mlp_activation();
    Tensor post;
    if (c.mlp == "mlp") {
        Tensor pre = project(*this, x, p + "up", batch);
        hook(hooks, p + "hook_pre", pre);
        post = std::move(pre);
        activate_(post, act);
    } else {  // glu, swiglu, geglu: down(act(gate(x)) * up(x))
        std::vector<Tensor> gate_up = project_many(*this, x, {p + "gate", p + "up"}, batch);
        hook(hooks, p + "hook_pre", gate_up[0]);
        post = std::move(gate_up[0]);
        activate_(post, act);
        post = mul(post, gate_up[1]);
    }
    hook(hooks, p + "hook_post", post);
    return project(*this, post, p + "down", batch);
}

Tensor Model::block(int64_t layer, Tensor x, const Hooks* hooks, const detail::RowBatch* batch) const {
    const ModelConfig& c = layer_cfgs_[layer];
    const std::string p = layer_prefix(layer);
    const bool post = c.norm_position == "post", sandwich = c.norm_position == "sandwich";
    hook(hooks, p + "hook_resid_pre", x);

    Tensor attn_in = post ? x : norm(p + "ln1", c, x);
    hook(hooks, p + "hook_attn_in", attn_in);
    Tensor attn_out = attention(layer, attn_in, hooks, batch);
    if (sandwich) attn_out = norm(p + "ln1_post", c, attn_out);
    hook(hooks, p + "hook_attn_out", attn_out);

    auto run_mlp = [&](Tensor mlp_in) {
        hook(hooks, p + "hook_mlp_in", mlp_in);
        Tensor out = mlp(layer, mlp_in, hooks, batch);
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

Tensor Model::logits_from_final(const Tensor& normed) const { return logits_from_final(normed, nullptr); }

Tensor Model::logits_from_final(const Tensor& normed, const detail::RowBatch* batch) const {
    Tensor logits;
#ifdef TFI_USE_ACCELERATE
    if (batch && batch->gpu && normed.size(-1) && normed.numel() / normed.size(-1) >= detail::gpu_min_rows())
        logits = batch->gpu->linear(normed, unembed(), nullptr, batch->fp16);
    else
#else
    (void)batch;
#endif
        logits = linear(normed, unembed());
    if (cfg_.final_logit_softcap) {
        const float cap = static_cast<float>(*cfg_.final_logit_softcap);
        for (float& v : logits.data) v = cap * vm::tanh(v / cap);
    }
    return logits;
}

Tensor Model::forward(const Tokens& tokens, const Hooks* hooks) const {
    return forward_from(0, embed(tokens, hooks), hooks);
}

Tensor Model::embed(const Tokens& tokens, const Hooks* hooks) const {
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
    return x;
}

Tensor Model::forward_from(int64_t layer, Tensor x, const Hooks* hooks) const {
    if (layer < 0 || layer > cfg_.n_layers) throw std::out_of_range("forward_from: layer out of range");
    if (x.dim() != 2 || x.shape[1] != cfg_.d_model)
        throw std::invalid_argument("forward_from: residual must be (T, d_model), got " + shape_str(x.shape));
    for (int64_t i = layer; i < cfg_.n_layers; ++i) x = block(i, std::move(x), hooks);
    Tensor f = final_norm(x);
    hook(hooks, "hook_final", f);
    return logits_from_final(f);
}

detail::Gpu* Model::gpu() const {
#ifdef TFI_USE_ACCELERATE
    static std::mutex mutex;  // one-time creation; Gpu itself is thread-safe
    std::lock_guard<std::mutex> lock(mutex);
    if (!gpu_.checked) {
        gpu_.gpu = detail::Gpu::create();
        gpu_.checked = true;
    }
    return gpu_.gpu.get();
#else
    return nullptr;
#endif
}

namespace {
void check_rows(const Model& m, int64_t layer, const Tensor& x, const detail::RowBatch& batch) {
    if (layer < 0 || layer >= m.cfg().n_layers) throw std::out_of_range("forward_rows: layer out of range");
    if (x.dim() != 2 || x.shape[1] != m.cfg().d_model || x.shape[0] != batch.rows())
        throw std::invalid_argument("forward_rows: residual must be (rows, d_model), got " + shape_str(x.shape));
}
}  // namespace

namespace {
std::atomic<bool> g_gpu_resident{true};
}  // namespace

void detail::set_gpu_resident(bool on) { g_gpu_resident = on; }

#ifdef TFI_USE_ACCELERATE
namespace {

using detail::GpuAct;

GpuAct gpu_act(const std::string& act) {
    if (act == "relu") return GpuAct::Relu;
    if (act == "gelu") return GpuAct::Gelu;
    if (act == "gelu_tanh") return GpuAct::GeluTanh;
    if (act == "silu") return GpuAct::Silu;
    if (act == "tanh") return GpuAct::Tanh;
    if (act == "relu2") return GpuAct::Relu2;
    throw std::runtime_error("unsupported activation " + act);
}

// Gpu::Seq slots of a batched pass. Inference (run_rows) uses fixed slots: the residual stream
// alternates between kResid0 and kResid1 (each step reads one and writes the other), and a
// parallel-residual MLP's output between kPar0 and kPar1, so that a step never overwrites its own
// inputs (Gpu::Seq::run may run it twice). Gradient passes (gradients_rows) give each layer's
// activations and gradients slots of their own, from kLayerSlots on, for the backward pass.
enum Slot : int { kResid0, kResid1, kAttnIn, kQ, kK, kV, kZ, kAttnOut, kAttnNormed, kMlpIn, kGate, kUp, kMlpOut,
                  kMlpRaw, kPar0, kPar1, kTmp,
                  kDZ, kDQ, kDK, kDV, kDIn, kDPost, kDPre, kDUp, kDMlpIn, kTmpB, kTmpC, kTmpD, kTop,  // backward
                  kLayerSlots = 64 };
constexpr int kSlotsPerLayer = 12;

// Where one block's activations go.
struct LayerSlots {
    int x, out;            // the residual stream in and out
    int attn_raw;          // o_proj's output (before any sandwich norm)
    int mid;               // hook_resid_mid (may be `out`)
    int pre, up;           // the MLP's pre-activation (GLU: the gate's) and a GLU's up projection
    int mlp_raw, mlp_out;  // the MLP's output before and after a sandwich norm
};

// Inference: block `layer` reads slot `cur` and writes `next`, everything else in scratch slots.
LayerSlots inference_slots(const ModelConfig& c, int64_t layer, int cur, int next) {
    const int par = kPar0 + static_cast<int>(layer & 1);  // must survive from before to after attention
    const bool sandwich = c.norm_position == "sandwich";
    return {cur, next, kAttnOut, next, kGate, kUp, c.parallel_residual && !sandwich ? par : kMlpRaw,
            c.parallel_residual ? par : kMlpOut};
}

// Gradient passes: every activation the backward pass needs has a slot of its own.
LayerSlots saved_slots(int64_t layer) {
    const int b = kLayerSlots + kSlotsPerLayer * static_cast<int>(layer);
    return {b, b + kSlotsPerLayer, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6};
}
struct GradSlots {
    int resid_pre, mid, attn_out, mlp_out;  // gradients at these hook points (mid is scratch with parallel residual)
};
GradSlots grad_slots(int64_t layer) {
    const int b = kLayerSlots + kSlotsPerLayer * static_cast<int>(layer);
    return {b + 7, b + 8, b + 9, b + 10};
}

// The GPU half of each block (everything but attention) for a batched pass, built from the same
// weights and config as Model::block, and its backward pass (as grad.cpp's Backward::block).
struct GpuBlocks {
    const Model& m;
    detail::Gpu::Seq& s;
    int64_t R, d;

    static std::string prefix(int64_t layer) { return "blocks." + std::to_string(layer) + "."; }
    static bool sandwich_norm(const ModelConfig& c) { return c.norm_position == "sandwich" && c.norm != "none"; }

    // norm(prefix, c, x) of slot `src` into `dst`; with norm "none", returns `src` unchanged.
    int norm(const std::string& name, const ModelConfig& c, int src, int dst) {
        if (c.norm == "none") return src;
        const Tensor& w = m.weight(name + ".weight");
        if (w.numel() != d) throw std::runtime_error(name + ".weight does not match the normalized size");
        const bool layernorm = c.norm == "layernorm";
        s.reserve(dst, R * d);
        s.norm(src, dst, R, w, layernorm ? m.weight_or_null(name + ".bias") : nullptr, layernorm,
               static_cast<float>(c.norm_eps));
        return dst;
    }
    // out = norm(name, a + b)
    void add_norm(const std::string& name, const ModelConfig& c, int a, int b, int out) {
        s.reserve(out, R * d);
        if (c.norm == "none") {
            s.add(a, b, out, R * d);
            return;
        }
        s.reserve(kTmp, R * d);
        s.add(a, b, kTmp, R * d);
        norm(name, c, kTmp, out);
    }
    void linear(int x, int y, const std::string& name) {
        const Tensor& W = m.weight(name + ".weight");
        if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
        s.reserve(y, R * W.shape[0]);
        s.linear(x, y, R, W, m.weight_or_null(name + ".bias"));
    }
    // Model::mlp (with sandwich's ln2_post) of slot `in`; returns the slot holding the result.
    int mlp(int64_t layer, int in, const LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer) + "mlp.";
        const GpuAct act = gpu_act(c.mlp_activation());
        const bool glu = c.mlp != "mlp";
        linear(in, ls.pre, p + (glu ? "gate" : "up"));
        const int64_t n = R * m.weight(p + (glu ? "gate" : "up") + ".weight").shape[0];
        s.reserve(kGate, n);
        if (ls.pre != kGate) s.copy(ls.pre, kGate, n);  // keep the pre-activation
        if (glu) {
            linear(in, ls.up, p + "up");
            s.glu(kGate, ls.up, n, act);
        } else {
            s.activate(kGate, n, act);
        }
        linear(kGate, ls.mlp_raw, p + "down");
        return sandwich_norm(c) ? norm(prefix(layer) + "ln2_post", c, ls.mlp_raw, ls.mlp_out) : ls.mlp_raw;
    }
    // Block `layer` up to the attention projections (into kQ, kK, kV); a parallel-residual MLP also
    // runs here, since it reads the block's input.
    void before_attention(int64_t layer, const LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        const int in = c.norm_position == "post" ? ls.x : norm(p + "ln1", c, ls.x, kAttnIn);
        linear(in, kQ, p + "attn.q_proj");
        linear(in, kK, p + "attn.k_proj");
        linear(in, kV, p + "attn.v_proj");
        if (c.parallel_residual && c.has_mlp()) mlp(layer, norm(p + "ln2", c, ls.x, kMlpIn), ls);
    }
    // The rest of block `layer` from z (slot kZ): o_proj, residual adds, norms and the MLP.
    void after_attention(int64_t layer, const LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        const bool post = c.norm_position == "post";
        linear(kZ, ls.attn_raw, p + "attn.o_proj");
        const int attn = c.norm_position == "sandwich" ? norm(p + "ln1_post", c, ls.attn_raw, kAttnNormed) : ls.attn_raw;
        s.reserve(ls.out, R * d);
        if (c.parallel_residual) {
            s.add(ls.x, attn, ls.out, R * d);
            if (c.has_mlp()) s.add(ls.out, sandwich_norm(c) ? ls.mlp_out : ls.mlp_raw, ls.out, R * d);
            return;
        }
        s.reserve(ls.mid, R * d);
        if (post) add_norm(p + "ln1", c, ls.x, attn, ls.mid);
        else s.add(ls.x, attn, ls.mid, R * d);
        if (!c.has_mlp()) {
            if (ls.mid != ls.out) s.copy(ls.mid, ls.out, R * d);
            return;
        }
        const int o = mlp(layer, post ? ls.mid : norm(p + "ln2", c, ls.mid, kMlpIn), ls);
        if (post) add_norm(p + "ln2", c, ls.mid, o, ls.out);
        else s.add(ls.mid, o, ls.out, R * d);
    }

    // --- backward ---
    // norm_backward(name, c, x, dy) into `dst`; with norm "none", returns `dy`.
    int norm_bwd(const std::string& name, const ModelConfig& c, int x, int dy, int dst) {
        if (c.norm == "none") return dy;
        const Tensor& w = m.weight(name + ".weight");
        if (w.numel() != d) throw std::runtime_error(name + ".weight does not match the normalized size");
        s.reserve(dst, R * d);
        s.norm_backward(x, dy, dst, R, w, c.norm == "layernorm", static_cast<float>(c.norm_eps));
        return dst;
    }
    void linear_t(int dy, int dx, const std::string& name) {
        const Tensor& W = m.weight(name + ".weight");
        s.reserve(dx, R * W.shape[1]);
        s.linear_t(dy, dx, R, W);
    }
    // dL/d(MLP input) into kDMlpIn from dL/d(MLP output, after any sandwich norm) in g_out.
    int mlp_bwd(int64_t layer, int g_out, const LayerSlots& ls) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer) + "mlp.";
        const GpuAct act = gpu_act(c.mlp_activation());
        const int g = c.norm_position == "sandwich" ? norm_bwd(prefix(layer) + "ln2_post", c, ls.mlp_raw, g_out, kTmpD) : g_out;
        linear_t(g, kDPost, p + "down");
        const int64_t n = R * m.weight(p + "down.weight").shape[1];
        s.reserve(kDPre, n);
        if (c.mlp == "mlp") {
            s.act_backward(kDPost, ls.pre, kDPre, n, act);
            linear_t(kDPre, kDMlpIn, p + "up");
        } else {
            s.reserve(kDUp, n);
            s.glu_backward(kDPost, ls.pre, ls.up, kDPre, kDUp, n, act);
            linear_t(kDPre, kDMlpIn, p + "gate");
            linear_t(kDUp, kTmpC, p + "up");
            s.add(kDMlpIn, kTmpC, kDMlpIn, R * d);
        }
        return kDMlpIn;
    }
    // Gradient slots of the hook points between the attention and the block's output.
    struct Reported {
        int mid = -1, attn_out = -1, mlp_out = -1;
    };
    // Block `layer`'s backward pass from dL/d(hook_resid_post) in `g_post` down to dL/dz (into kDZ).
    // Returns the slot that dL/d(hook_resid_pre) adds the attention path to (backward_after).
    int backward_before(int64_t layer, const LayerSlots& ls, const GradSlots& gs, int g_post, Reported& rep) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        const bool post = c.norm_position == "post";
        int base, gx;
        if (c.parallel_residual) {  // mid = x + attn(ln1 x); out = mid + mlp(ln2 x)
            rep.mid = rep.attn_out = g_post;
            base = gx = g_post;
            if (c.has_mlp()) {
                rep.mlp_out = g_post;
                const int t = norm_bwd(p + "ln2", c, ls.x, mlp_bwd(layer, g_post, ls), kTmpB);
                s.reserve(gs.mid, R * d);
                s.add(g_post, t, gs.mid, R * d);
                base = gs.mid;
            }
        } else {
            int g_mid = g_post;
            if (c.has_mlp()) {
                int g_raw = g_post;  // gradient of mid + mlp_out
                if (post) {
                    s.reserve(kTmp, R * d);
                    s.add(ls.mid, ls.mlp_raw, kTmp, R * d);
                    g_raw = norm_bwd(p + "ln2", c, kTmp, g_post, gs.mlp_out);
                }
                rep.mlp_out = g_raw;
                const int d_in = mlp_bwd(layer, g_raw, ls);
                const int t = post ? d_in : norm_bwd(p + "ln2", c, ls.mid, d_in, kTmpB);
                s.reserve(gs.mid, R * d);
                s.add(g_raw, t, gs.mid, R * d);
                g_mid = gs.mid;
            }
            rep.mid = g_mid;
            gx = g_mid;  // gradient of x + attn_out
            if (post) {
                s.reserve(kTmp, R * d);
                s.add(ls.x, ls.attn_raw, kTmp, R * d);
                gx = norm_bwd(p + "ln1", c, kTmp, g_mid, gs.attn_out);
            }
            rep.attn_out = gx;
            base = gx;
        }
        const int g = c.norm_position == "sandwich" ? norm_bwd(p + "ln1_post", c, ls.attn_raw, gx, kTmpD) : gx;
        linear_t(g, kDZ, p + "attn.o_proj");
        return base;
    }
    // The rest of block `layer`'s backward pass from the q, k, v projections' gradients (kDQ, kDK,
    // kDV): dL/d(hook_resid_pre) = base + the attention path, into gs.resid_pre.
    void backward_after(int64_t layer, const LayerSlots& ls, const GradSlots& gs, int base) {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = prefix(layer);
        linear_t(kDQ, kDIn, p + "attn.q_proj");
        linear_t(kDK, kTmpB, p + "attn.k_proj");
        s.add(kDIn, kTmpB, kDIn, R * d);
        linear_t(kDV, kTmpB, p + "attn.v_proj");
        s.add(kDIn, kTmpB, kDIn, R * d);
        const int t = c.norm_position == "post" ? kDIn : norm_bwd(p + "ln1", c, ls.x, kDIn, kTmpC);
        s.reserve(gs.resid_pre, R * d);
        s.add(base, t, gs.resid_pre, R * d);
    }
};

// Hook points that Model::block calls on activations the GPU path keeps on the GPU.
bool gpu_side_hooked(const Hooks* hooks, int64_t layer) {
    if (!hooks) return false;
    const std::string p = layer_prefix(layer);
    for (const char* name : {"hook_attn_in", "hook_attn_out", "hook_resid_mid", "hook_mlp_in", "mlp.hook_pre",
                             "mlp.hook_post", "hook_mlp_out", "hook_resid_post"})
        if (hooks->matches(p + name)) return true;
    return false;
}

}  // namespace
#endif

// On the GPU, each run of blocks with no hook on a GPU-side activation keeps the residual stream on
// the GPU: one submission per block (Gpu::Seq), the CPU doing only attention in between. A block
// with such a hook, or the CPU path, runs Model::block (linear layers still go to the GPU).
Tensor Model::run_rows(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks) const {
    const int64_t L = cfg_.n_layers;
#ifdef TFI_USE_ACCELERATE
    const int64_t R = batch.rows(), d = cfg_.d_model;
    if (g_gpu_resident.load() && batch.gpu && R >= detail::gpu_min_rows()) {
        detail::Gpu::Seq s(*batch.gpu);
        GpuBlocks gb{*this, s, R, d};
        int64_t i = layer;
        while (i < L) {
            if (gpu_side_hooked(hooks, i)) {
                x = block(i, std::move(x), hooks, &batch);
                ++i;
                continue;
            }
            // Blocks i..j-1 stay on the GPU; a later hook_resid_pre needs the CPU.
            int64_t j = i + 1;
            while (j < L && !gpu_side_hooked(hooks, j) && !(hooks && hooks->matches(layer_prefix(j) + "hook_resid_pre")))
                ++j;
            hook(hooks, layer_prefix(i) + "hook_resid_pre", x);
            s.reserve(kResid1, R * d);
            std::copy(x.ptr(), x.ptr() + R * d, s.reserve(kResid0, R * d));
            int cur = kResid0;
            gb.before_attention(i, inference_slots(layer_cfgs_[i], i, cur, kResid1));
            s.run(batch.fp16);
            for (int64_t l = i; l < j; ++l) {
                const ModelConfig& c = layer_cfgs_[l];
                const int64_t Hd = c.n_heads * c.head_dim(), KVd = c.kv_heads() * c.head_dim();
                auto get = [&](int slot, int64_t width) {
                    Tensor t({R, width});
                    std::copy(s.data(slot), s.data(slot) + R * width, t.ptr());
                    return t;
                };
                const Tensor z = attend(l, get(kQ, Hd), get(kK, KVd), get(kV, KVd), hooks, &batch);
                std::copy(z.ptr(), z.ptr() + R * Hd, s.reserve(kZ, R * Hd));
                const int next = cur == kResid0 ? kResid1 : kResid0;
                gb.after_attention(l, inference_slots(c, l, cur, next));
                cur = next;
                if (l + 1 < j) gb.before_attention(l + 1, inference_slots(layer_cfgs_[l + 1], l + 1, cur, cur == kResid0 ? kResid1 : kResid0));
                s.run(batch.fp16);
            }
            x = Tensor({R, d});
            std::copy(s.data(cur), s.data(cur) + R * d, x.ptr());
            i = j;
        }
        return x;
    }
#endif
    for (int64_t i = layer; i < L; ++i) x = block(i, std::move(x), hooks, &batch);
    return x;
}

// Forward: one submission per block as in run_rows, keeping each block's activations on the GPU
// (saved_slots) and each segment's attention tensors on the CPU. Backward: from the last layer
// down, one submission per block (GpuBlocks::backward_after of one block, backward_before of the
// next), with attention's backward on the CPU in between, per segment (detail::attention_backward).
std::vector<ActivationCache> Model::gradients_rows(const Tensor& x0, const detail::RowBatch& batch,
                                                   const std::function<Tensor(int64_t, const Tensor&)>& dmetric,
                                                   const NameFilter& filter) const {
#ifdef TFI_USE_ACCELERATE
    const int64_t L = cfg_.n_layers, d = cfg_.d_model, R = batch.rows(), S = static_cast<int64_t>(batch.start.size());
    if (!batch.gpu) throw std::logic_error("gradients_rows needs a GPU");
    if (x0.dim() != 2 || x0.shape[0] != R || x0.shape[1] != d)
        throw std::invalid_argument("gradients_rows: input must be (rows, d_model), got " + shape_str(x0.shape));
    for (int64_t start : batch.start)
        if (start != 0) throw std::invalid_argument("gradients_rows: every segment must start at position 0");
    auto wanted = [&](const std::string& n) { return !filter || filter(n); };

    detail::Gpu::Seq s(*batch.gpu);
    GpuBlocks gb{*this, s, R, d};
    // Each segment's attention inputs (after qk_norm and RoPE), pattern, and projections before qk_norm.
    struct SegAttn {
        Tensor q, k, v, pattern, q0, k0;
    };
    std::vector<std::vector<SegAttn>> saved(L, std::vector<SegAttn>(S));
    auto rows = [&](int slot, int64_t seg, int64_t width) {
        Tensor t({batch.len[seg], width});
        const float* src = s.data(slot) + batch.offset[seg] * width;
        std::copy(src, src + t.numel(), t.ptr());
        return t;
    };

    std::copy(x0.ptr(), x0.ptr() + R * d, s.reserve(saved_slots(0).x, R * d));
    gb.before_attention(0, saved_slots(0));
    s.run(false);
    for (int64_t l = 0; l < L; ++l) {
        const ModelConfig& c = layer_cfgs_[l];
        const std::string p = layer_prefix(l) + "attn.";
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        float* z = s.reserve(kZ, R * H * D);
        for (int64_t g = 0; g < S; ++g) {
            SegAttn& a = saved[l][g];
            Tensor q2 = rows(kQ, g, H * D), k2 = rows(kK, g, Hkv * D), v2 = rows(kV, g, Hkv * D);
            if (c.qk_norm) {
                a.q0 = detail::split_heads(q2, H, D);
                a.k0 = detail::split_heads(k2, Hkv, D);
            }
            Hooks keep;
            keep.add(p + "hook_q", [&](const std::string&, Tensor& t) { a.q = t; });
            keep.add(p + "hook_k", [&](const std::string&, Tensor& t) { a.k = t; });
            keep.add(p + "hook_v", [&](const std::string&, Tensor& t) { a.v = t; });
            keep.add(p + "hook_pattern", [&](const std::string&, Tensor& t) { a.pattern = t; });
            const Tensor zg = attend(l, std::move(q2), std::move(k2), std::move(v2), &keep, nullptr);
            std::copy(zg.ptr(), zg.ptr() + zg.numel(), z + batch.offset[g] * H * D);
        }
        gb.after_attention(l, saved_slots(l));
        if (l + 1 < L) gb.before_attention(l + 1, saved_slots(l + 1));
        s.run(false);
    }

    // dF/d(hook_resid_post) of the last block: only each segment's last row, as in run_with_gradients.
    const int final_resid = saved_slots(L - 1).out;
    float* top = s.reserve(kTop, R * d);
    std::fill(top, top + R * d, 0.f);
    for (int64_t g = 0; g < S; ++g) {
        const int64_t r = batch.offset[g] + batch.len[g] - 1;
        Tensor x({1, d});
        std::copy(s.data(final_resid) + r * d, s.data(final_resid) + (r + 1) * d, x.ptr());
        const Tensor logits = logits_from_final(final_norm(x));
        Tensor dl = dmetric(g, logits);
        if (dl.shape != logits.shape)
            throw std::invalid_argument("dlogits must have the logits' shape " + shape_str(logits.shape) + ", got " +
                                        shape_str(dl.shape));
        if (cfg_.final_logit_softcap) {
            const float cap = static_cast<float>(*cfg_.final_logit_softcap);
            for (int64_t i = 0; i < dl.numel(); ++i) {
                const float th = logits.data[i] / cap;
                dl.data[i] *= 1.f - th * th;
            }
        }
        Tensor gr = matmul(dl, unembed());
        if (cfg_.norm_position != "post") gr = detail::norm_backward(*this, "ln_final", cfg_, x, gr);
        std::copy(gr.ptr(), gr.ptr() + d, top + r * d);
    }

    auto g_post = [&](int64_t l) { return l == L - 1 ? static_cast<int>(kTop) : grad_slots(l + 1).resid_pre; };
    std::vector<GpuBlocks::Reported> rep(L);
    std::vector<int> base(L);
    std::vector<Tensor> dz(L);  // dF/dz, (rows, H*D)
    base[L - 1] = gb.backward_before(L - 1, saved_slots(L - 1), grad_slots(L - 1), g_post(L - 1), rep[L - 1]);
    s.run(false);
    for (int64_t l = L - 1; l >= 0; --l) {
        const ModelConfig& c = layer_cfgs_[l];
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        dz[l] = Tensor({R, H * D});
        std::copy(s.data(kDZ), s.data(kDZ) + R * H * D, dz[l].ptr());
        float* dq = s.reserve(kDQ, R * H * D);
        float* dk = s.reserve(kDK, R * Hkv * D);
        float* dv = s.reserve(kDV, R * Hkv * D);
        for (int64_t g = 0; g < S; ++g) {
            const SegAttn& a = saved[l][g];
            const int64_t off = batch.offset[g], n = batch.len[g];
            const Tensor dzg({n, H * D}, std::vector<float>(dz[l].ptr() + off * H * D, dz[l].ptr() + (off + n) * H * D));
            const detail::AttentionGrads ag = detail::attention_backward(
                *this, l, dzg, a.q, a.k, a.v, a.pattern, c.qk_norm ? &a.q0 : nullptr, c.qk_norm ? &a.k0 : nullptr);
            std::copy(ag.q_proj.ptr(), ag.q_proj.ptr() + n * H * D, dq + off * H * D);
            std::copy(ag.k_proj.ptr(), ag.k_proj.ptr() + n * Hkv * D, dk + off * Hkv * D);
            std::copy(ag.v_proj.ptr(), ag.v_proj.ptr() + n * Hkv * D, dv + off * Hkv * D);
        }
        gb.backward_after(l, saved_slots(l), grad_slots(l), base[l]);
        if (l > 0) base[l - 1] = gb.backward_before(l - 1, saved_slots(l - 1), grad_slots(l - 1), g_post(l - 1), rep[l - 1]);
        s.run(false);
    }

    std::vector<ActivationCache> out(S);
    auto put = [&](const std::string& name, int slot) {
        if (slot < 0 || !wanted(name)) return;
        for (int64_t g = 0; g < S; ++g) out[g][name] = rows(slot, g, d);
    };
    for (int64_t l = 0; l < L; ++l) {
        const ModelConfig& c = layer_cfgs_[l];
        const std::string p = layer_prefix(l);
        put(p + "hook_resid_pre", grad_slots(l).resid_pre);
        put(p + "hook_resid_post", g_post(l));
        put(p + "hook_resid_mid", rep[l].mid);
        put(p + "hook_attn_out", rep[l].attn_out);
        put(p + "hook_mlp_out", rep[l].mlp_out);
        if (wanted(p + "attn.hook_z"))
            for (int64_t g = 0; g < S; ++g) {
                const int64_t off = batch.offset[g], n = batch.len[g], Hd = c.n_heads * c.head_dim();
                out[g][p + "attn.hook_z"] =
                    Tensor({n, c.n_heads, c.head_dim()}, std::vector<float>(dz[l].ptr() + off * Hd, dz[l].ptr() + (off + n) * Hd));
            }
    }
    put("hook_embed", grad_slots(0).resid_pre);
    return out;
#else
    (void)x0;
    (void)batch;
    (void)dmetric;
    (void)filter;
    throw std::logic_error("gradients_rows needs a GPU (macOS)");
#endif
}

Tensor Model::forward_rows_final(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks) const {
    check_rows(*this, layer, x, batch);
    return final_norm(run_rows(layer, std::move(x), batch, hooks));
}

Tensor Model::forward_rows(int64_t layer, Tensor x, const detail::RowBatch& batch, const Hooks* hooks) const {
    check_rows(*this, layer, x, batch);
    x = run_rows(layer, std::move(x), batch, hooks);
    const int64_t S = static_cast<int64_t>(batch.start.size()), d = cfg_.d_model;
    Tensor last({S, d});
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r = batch.offset[s] + batch.len[s] - 1;
        std::copy(x.ptr() + r * d, x.ptr() + (r + 1) * d, last.ptr() + s * d);
    }
    return logits_from_final(final_norm(last));
}

Tensor Model::run_with_cache(const Tokens& tokens, ActivationCache& cache, NameFilter filter, const Hooks* hooks) const {
    Hooks all;
    if (hooks) all.extend(*hooks);
    all.add_filter(std::move(filter), [&cache](const std::string& name, Tensor& act) { cache[name] = act; });
    return forward(tokens, &all);
}

}  // namespace tfi
