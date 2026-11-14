// Self-contained unit tests for tfi: invariants of the analysis tools on random models.
// The forward pass itself is checked against PyTorch by tests/test_cpp.py (golden tests).
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "tfi/analysis.h"
#include "tfi/json.h"
#include "tfi/model.h"
#include "../src/vmath.h"  // internal: vectorizable exp / erf / tanh
#include "../src/gpu.h"    // internal (macOS): GPU matrix multiplies
#include "../src/ops.h"    // internal: patching batch sizes

using namespace tfi;

static int g_failures = 0, g_checks = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_failures;                                                           \
            std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs(double(a) - double(b)) <= (tol))

#define CHECK_THROWS(expr)                     \
    do {                                       \
        bool threw = false;                    \
        try { (void)(expr); } catch (...) { threw = true; } \
        CHECK(threw);                          \
    } while (0)

// Random weights with exactly the tensor names and shapes indranet's state_dict uses.
static Model random_model(ModelConfig cfg, uint64_t seed = 0) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::map<std::string, Tensor> w;
    auto randn = [&](const std::string& name, Shape s, float std) {
        Tensor t(s);
        for (float& v : t.data) v = normal(rng) * std;
        w[name] = t;
    };
    auto norm = [&](const std::string& prefix, const ModelConfig& c, int64_t dim) {
        if (c.norm == "none") return;
        Tensor g({dim});
        for (float& v : g.data) v = 1.f + 0.2f * normal(rng);
        w[prefix + ".weight"] = g;
        if (c.norm == "layernorm" && c.bias) randn(prefix + ".bias", {dim}, 0.1f);
    };
    auto lin = [&](const std::string& prefix, const ModelConfig& c, int64_t out, int64_t in, float std) {
        randn(prefix + ".weight", {out, in}, std);
        if (c.bias) randn(prefix + ".bias", {out}, 0.1f);
    };
    const int64_t d = cfg.d_model;
    randn("embed.weight", {cfg.vocab_size, d}, 1.0f);
    if (cfg.positional == "learned") randn("pos_embed.weight", {cfg.max_seq_len, d}, 0.5f);
    if (!cfg.tie_embeddings) randn("unembed.weight", {cfg.vocab_size, d}, 0.2f);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        const ModelConfig c = cfg.layer_config(i);
        const std::string p = "blocks." + std::to_string(i) + ".";
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim();
        norm(p + "ln1", c, d);
        lin(p + "attn.q_proj", c, H * D, d, 0.5f);
        lin(p + "attn.k_proj", c, Hkv * D, d, 0.5f);
        lin(p + "attn.v_proj", c, Hkv * D, d, 0.3f);
        lin(p + "attn.o_proj", c, d, H * D, 0.3f);
        if (c.qk_norm) {
            ModelConfig qc = c;
            if (qc.norm == "none") qc.norm = "rmsnorm";
            norm(p + "attn.q_norm", qc, D);
            norm(p + "attn.k_norm", qc, D);
        }
        if (c.norm_position == "sandwich") norm(p + "ln1_post", c, d);
        if (c.has_mlp()) {
            const bool gated = c.mlp != "mlp";
            int64_t d_mlp = c.d_mlp ? *c.d_mlp : 4 * d;
            if (gated && !c.d_mlp) {
                const int64_t base = static_cast<int64_t>(8.0 * d / 3.0), mult = c.mlp_multiple_of;
                d_mlp = mult * ((base + mult - 1) / mult);
            }
            if (gated) lin(p + "mlp.gate", c, d_mlp, d, 0.3f);
            lin(p + "mlp.up", c, d_mlp, d, 0.3f);
            lin(p + "mlp.down", c, d, d_mlp, 0.2f);
            norm(p + "ln2", c, d);
            if (c.norm_position == "sandwich") norm(p + "ln2_post", c, d);
        }
    }
    if (cfg.norm_position != "post") norm("ln_final", cfg, d);
    return Model(cfg, w);
}

static ModelConfig tiny(std::function<void(ModelConfig&)> edit = nullptr) {
    ModelConfig c;
    c.vocab_size = 50;
    c.d_model = 16;
    c.n_layers = 2;
    c.n_heads = 4;
    c.max_seq_len = 32;
    c.mlp_multiple_of = 8;
    if (edit) edit(c);
    c.validate();
    return c;
}

static Tokens seq(int64_t T, uint64_t seed, int64_t V = 50) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, V - 1);
    Tokens t(T);
    for (auto& x : t) x = dist(rng);
    return t;
}

static std::vector<std::pair<std::string, ModelConfig>> configs() {
    return {
        {"default (rmsnorm, rope, swiglu)", tiny()},
        {"layernorm + learned + gelu mlp + bias", tiny([](ModelConfig& c) {
             c.norm = "layernorm"; c.positional = "learned"; c.mlp = "mlp"; c.bias = true; c.tie_embeddings = false;
         })},
        {"alibi + GQA + window", tiny([](ModelConfig& c) { c.positional = "alibi"; c.n_kv_heads = 2; c.sliding_window = 3; })},
        {"parallel residual", tiny([](ModelConfig& c) { c.parallel_residual = true; c.norm = "layernorm"; c.bias = true; })},
        {"sandwich + softcaps", tiny([](ModelConfig& c) {
             c.norm_position = "sandwich"; c.attn_logit_softcap = 5.0; c.final_logit_softcap = 10.0;
         })},
        {"attention-only, no norm", tiny([](ModelConfig& c) { c.mlp = "none"; c.norm = "none"; })},
    };
}

static void test_json() {
    const json::Value v = json::parse(R"({"a": [1, 2.5, -3e2], "b": {"c": null, "d": "x\"é"}, "e": true})");
    CHECK(v.at("a").array.size() == 3);
    CHECK_NEAR(v.at("a").array[2].as_number(), -300, 1e-9);
    CHECK(v.at("b").at("c").is_null());
    CHECK(v.at("b").at("d").as_string() == "x\"\xc3\xa9");
    CHECK(v.at("e").as_bool());
    CHECK_THROWS(json::parse("{\"a\": }"));
}

static void test_reshape_and_linear() {
    // reshape on an rvalue takes over the buffer; on an lvalue it copies and leaves the original alone.
    Tensor t({2, 3}, {0, 1, 2, 3, 4, 5});
    const float* buf = t.ptr();
    Tensor moved = std::move(t).reshape({3, -1});
    CHECK(moved.ptr() == buf);
    CHECK(moved.shape == (Shape{3, 2}));
    const Tensor copy = moved.reshape({6});
    CHECK(copy.ptr() != moved.ptr());
    CHECK(moved.shape == (Shape{3, 2}) && moved.data[5] == 5.f);
    CHECK(copy.shape == (Shape{6}) && copy.data[5] == 5.f);
    CHECK_THROWS(moved.reshape({4}));

    // linear on a 3-D input with bias, against a direct computation.
    std::mt19937_64 rng(2);
    std::normal_distribution<float> n(0, 1);
    Tensor x({2, 3, 4}), W({5, 4}), b({5});
    for (Tensor* p : {&x, &W, &b})
        for (float& v : p->data) v = n(rng);
    const Tensor y = linear(x, W, &b);
    CHECK(y.shape == (Shape{2, 3, 5}));
    for (int64_t i = 0; i < 2; ++i)
        for (int64_t t2 = 0; t2 < 3; ++t2)
            for (int64_t o = 0; o < 5; ++o) {
                float s = b.data[o];
                for (int64_t k = 0; k < 4; ++k) s += x.at({i, t2, k}) * W.at({o, k});
                CHECK_NEAR(y.at({i, t2, o}), s, 1e-4);
            }
    CHECK(linear(x, W).shape == (Shape{2, 3, 5}));
    CHECK_THROWS(linear(x, Tensor({5, 3})));
}

static void test_matmul() {
    std::mt19937_64 rng(1);
    std::normal_distribution<float> n(0, 1);
    Tensor A({5, 7}), B({7, 3});
    for (float& v : A.data) v = n(rng);
    for (float& v : B.data) v = n(rng);
    const Tensor C = matmul(A, B), Ct = matmul(A, transpose2d(B), true);
    for (int64_t i = 0; i < 5; ++i)
        for (int64_t j = 0; j < 3; ++j) {
            float s = 0;
            for (int64_t k = 0; k < 7; ++k) s += A.at({i, k}) * B.at({k, j});
            CHECK_NEAR(C.at({i, j}), s, 1e-4);
            CHECK_NEAR(Ct.at({i, j}), s, 1e-4);
        }
    CHECK_THROWS(matmul(A, A));
}

static void test_cache_and_hooks(const std::string& name, const Model& m) {
    const Tokens t = seq(10, 3);
    ActivationCache cache;
    const Tensor logits = m.run_with_cache(t, cache);
    CHECK(cache.size() == m.hook_names().size());
    for (const auto& n : m.hook_names()) CHECK(cache.count(n));
    CHECK(max_abs_diff(logits, m.forward(t)) == 0.f);

    // Causality: changing the last token leaves earlier logits untouched.
    Tokens t2 = t;
    t2.back() = (t2.back() + 1) % m.cfg().vocab_size;
    const Tensor l2 = m.forward(t2);
    const int64_t V = m.cfg().vocab_size;
    CHECK(max_abs_diff(rows(logits, 0, 9), rows(l2, 0, 9)) == 0.f);
    CHECK(max_abs_diff(rows(logits, 9, 10), rows(l2, 9, 10)) > 0.f);

    // A zero-ablation hook changes the output.
    Hooks zero;
    zero.add("blocks.1.hook_attn_out", [](const std::string&, Tensor& a) { std::fill(a.data.begin(), a.data.end(), 0.f); });
    CHECK(max_abs_diff(m.forward(t, &zero), logits) > 0.f);
    (void)V;
    (void)name;
}

static void test_logit_lens(const Model& m) {
    const Tokens t = seq(9, 4);
    const auto lens = logit_lens(m, t, 8, 7, 3);
    CHECK(lens.front().stage == "embed");
    CHECK(lens.size() == 1 + 2 * 2 - (m.cfg().has_mlp() ? 0 : 2));
    // The last stage is the model's real output distribution.
    Tensor logits = m.forward(t);
    Tensor last = rows(logits, 8, 9);
    softmax_(last.ptr(), m.cfg().vocab_size);
    CHECK_NEAR(lens.back().target_prob, last.data[7], 1e-5);
    CHECK(lens.back().top_tokens[0] == topk(last.ptr(), m.cfg().vocab_size, 1)[0]);
}

static void test_dla(const Model& m) {
    const Tokens t = seq(11, 5);
    const DLAResult r = direct_logit_attribution(m, t, 10, 3, 4);
    CHECK_NEAR(r.sum, r.actual, 1e-3 * std::max(1.f, std::fabs(r.actual)));
    // The actual value is the pre-softcap logit difference; without softcap it equals the logits.
    if (!m.cfg().final_logit_softcap) {
        const Tensor logits = m.forward(t);
        CHECK_NEAR(r.actual, logit_metric(logits, 10, 3, 4), 1e-4);
    }
    const DLAResult single = direct_logit_attribution(m, t, 10, 3);
    CHECK_NEAR(single.sum, single.actual, 1e-3 * std::max(1.f, std::fabs(single.actual)));
}

static void test_patching(const Model& m) {
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const PatchResult r = patch_residual(m, clean, corrupt, "resid_pre", 7, 1, 2);
    // At layer 0 only position 3 differs, so patching it restores the clean run exactly.
    CHECK_NEAR(r.values.at({0, 3}), 1.0, 1e-4);
    for (int64_t p = 0; p < 8; ++p)
        if (p != 3) CHECK_NEAR(r.values.at({0, p}), 0.0, 1e-4);
    // Positions before the change are identical in both runs at every layer.
    for (int64_t l = 0; l < 2; ++l)
        for (int64_t p = 0; p < 3; ++p) CHECK_NEAR(r.values.at({l, p}), 0.0, 1e-4);
    const PatchResult heads = patch_heads(m, clean, corrupt, 7, 1, 2);
    CHECK(heads.values.shape == (Shape{2, 4}));
    CHECK_THROWS(patch_residual(m, clean, Tokens(7, 0), "resid_pre", 6, 1, 2));
    CHECK_THROWS(patch_residual(m, clean, corrupt, "nope", 7, 1, 2));

    // values = effect / gap wherever a site exists.
    CHECK(r.normalized());
    for (int64_t i = 0; i < r.values.numel(); ++i)
        if (!std::isnan(r.effect.data[i])) CHECK_NEAR(r.values.data[i], r.effect.data[i] / r.gap(), 1e-5);
}

static void test_patching_edge_cases(const Model& m) {
    const bool has_mlp = m.layer_cfg(0).has_mlp();
    // Identical inputs: zero gap. No NaN effects, values undefined (all NaN), nothing claims to restore.
    const Tokens t = seq(8, 11);
    const std::vector<std::string> sites = {"resid_pre", "mlp_out"};
    for (const std::string& site : sites) {
        const PatchResult r = patch_residual(m, t, t, site, 7, 1, 2);
        CHECK(!r.normalized());
        CHECK(r.gap() == 0.f);
        for (float v : r.values.data) CHECK(std::isnan(v));
        for (float e : r.effect.data) {
            if (site == "mlp_out" && !has_mlp) CHECK(std::isnan(e));  // no such site: NaN effect
            else CHECK(e == 0.f);
        }
    }
    const PatchResult heads = patch_heads(m, t, t, 7, 1, 2);
    CHECK(!heads.normalized());
    for (float e : heads.effect.data) CHECK(e == 0.f);

    // A change after `pos` cannot affect it (causality): also a zero gap, not NaN effects.
    Tokens later = t;
    later[7] = (later[7] + 1) % m.cfg().vocab_size;
    const PatchResult causal = patch_residual(m, t, later, "resid_pre", 5, 1, 2);
    CHECK(!causal.normalized());
    for (int64_t l = 0; l < m.cfg().n_layers; ++l) CHECK(causal.effect.at({l, 0}) == 0.f);

    // Missing sites stay NaN in both tensors when normalized.
    if (!has_mlp) {
        Tokens corrupt = t;
        corrupt[2] = (corrupt[2] + 3) % m.cfg().vocab_size;
        const PatchResult r = patch_residual(m, t, corrupt, "mlp_out", 7, 1, 2);
        for (int64_t i = 0; i < r.values.numel(); ++i) CHECK(std::isnan(r.values.data[i]) && std::isnan(r.effect.data[i]));
    }
}

// The straightforward versions the optimized tools must agree with: full sequences, full forward
// passes, one at a time.
static Tensor naive_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                          int64_t pos, int64_t target, int64_t other) {
    const int64_t L = m.cfg().n_layers, T = static_cast<int64_t>(clean.size()), d = m.cfg().d_model;
    ActivationCache cc;
    m.run_with_cache(clean, cc);
    const float base = logit_metric(m.forward(corrupt), pos, target, other);
    Tensor out({L, T}, NAN);
    for (int64_t i = 0; i < L; ++i) {
        const std::string name = "blocks." + std::to_string(i) + ".hook_" + site;
        if (!cc.count(name)) continue;
        const Tensor& src = cc.at(name);
        for (int64_t t = 0; t < T; ++t) {
            Hooks h;
            h.add(name, [&src, t, d](const std::string&, Tensor& a) { std::copy(src.ptr() + t * d, src.ptr() + (t + 1) * d, a.ptr() + t * d); });
            out.data[i * T + t] = logit_metric(m.forward(corrupt, &h), pos, target, other) - base;
        }
    }
    return out;
}

static Tensor naive_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                                int64_t other) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    ActivationCache cc;
    m.run_with_cache(clean, cc);
    const float base = logit_metric(m.forward(corrupt), pos, target, other);
    Tensor out({L, H}, NAN);
    for (int64_t i = 0; i < L; ++i) {
        const std::string name = "blocks." + std::to_string(i) + ".attn.hook_z";
        const Tensor& src = cc.at(name);
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        for (int64_t hd = 0; hd < Hl; ++hd) {
            Hooks h;
            h.add(name, [&src, hd, Hl, D](const std::string&, Tensor& a) {
                for (int64_t t = 0; t < a.shape[0]; ++t)
                    std::copy(src.ptr() + (t * Hl + hd) * D, src.ptr() + (t * Hl + hd + 1) * D, a.ptr() + (t * Hl + hd) * D);
            });
            out.data[i * H + hd] = logit_metric(m.forward(corrupt, &h), pos, target, other) - base;
        }
    }
    return out;
}

static void check_close(const Tensor& got, const Tensor& want) {
    CHECK(got.shape == want.shape);
    for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i) {
        if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
        else CHECK_NEAR(got.data[i], want.data[i], 1e-4 * std::max(1.f, std::fabs(want.data[i])));
    }
}

// Patching runs batched passes (src/analysis.cpp): stacked rows, recomputing only the rows from the
// patched position on, and on macOS GPU linear layers. Every batch size, with the GPU off and on,
// must agree with naive patching, and the thread count must never change a result.
// Batched reference means and ablation agree with one plain forward pass per reference and per head.
static void check_ablation_matches_naive(const Model& m, const Tokens& t) {
    const int64_t T = static_cast<int64_t>(t.size());
    for (const std::vector<Tokens>& refs : std::vector<std::vector<Tokens>>{
             {seq(T, 61), seq(T, 62), seq(T, 63)},  // one mean per position
             {seq(T, 61), seq(4, 62), seq(T + 2, 63)}}) {  // other lengths: one pooled mean
        const HeadMeans means = head_means(m, refs, T);
        double count = 0;
        std::vector<Tensor> naive;
        for (int64_t l = 0; l < m.cfg().n_layers; ++l) naive.emplace_back(means.layers[l].shape);
        for (const Tokens& r : refs) {
            ActivationCache c;
            m.run_with_cache(r, c, [](const std::string& n) { return n.find("attn.hook_z") != std::string::npos; });
            for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
                const Tensor& z = c.at("blocks." + std::to_string(l) + ".attn.hook_z");
                const int64_t row = z.numel() / z.shape[0];
                for (int64_t p = 0; p < z.shape[0]; ++p)
                    for (int64_t e = 0; e < row; ++e)
                        naive[l].data[(means.per_position() ? p : 0) * row + e] += z.data[p * row + e];
            }
            count += means.per_position() ? 1.0 : static_cast<double>(r.size());
        }
        for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
            for (float& v : naive[l].data) v = static_cast<float>(v / count);
            CHECK(max_abs_diff(means.layers[l], naive[l]) <= 1e-5f * std::max(1.f, max_abs(naive[l])));
        }
        if (!means.per_position()) continue;
        const float base = next_token_loss(m.forward(t), t);
        for (Ablation mode : {Ablation::Zero, Ablation::Mean}) {
            const Tensor got = head_ablation(m, t, mode, refs);
            for (int64_t l = 0; l < m.cfg().n_layers; ++l)
                for (int64_t h = 0; h < m.layer_cfg(l).n_heads; ++h) {
                    const Hooks hk = ablate_head(m, l, h, mode, mode == Ablation::Mean ? &means : nullptr);
                    CHECK_NEAR(got.at({l, h}), next_token_loss(m.forward(t, &hk), t) - base, 1e-5);
                }
        }
    }
}

static void test_optimized_patching_matches_naive(const Model& m) {
    // forward_from reproduces forward exactly from any layer's cached input.
    const Tokens t = seq(9, 51);
    ActivationCache cache;
    const Tensor full = m.run_with_cache(t, cache);
    for (int64_t l = 0; l <= m.cfg().n_layers; ++l) {
        const Tensor start = l < m.cfg().n_layers ? cache.at("blocks." + std::to_string(l) + ".hook_resid_pre")
                                                  : cache.at("blocks." + std::to_string(l - 1) + ".hook_resid_post");
        CHECK(max_abs_diff(m.forward_from(l, start), full) == 0.f);
    }
    CHECK_THROWS(m.forward_from(-1, cache.at("blocks.0.hook_resid_pre")));
    CHECK_THROWS(m.forward_from(0, Tensor({9, m.cfg().d_model + 1})));

    Tokens corrupt = t;
    corrupt[2] = (corrupt[2] + 5) % m.cfg().vocab_size;
    corrupt[6] = (corrupt[6] + 9) % m.cfg().vocab_size;
    const int64_t last = m.cfg().n_layers - 1;
    std::vector<bool> gpu_modes = {false};
#ifdef TFI_USE_ACCELERATE
    if (m.gpu()) gpu_modes.push_back(true);
#endif
    for (const bool gpu : gpu_modes) {
#ifdef TFI_USE_ACCELERATE
        detail::set_gpu_min_rows(gpu ? 1 : std::numeric_limits<int64_t>::max());
#else
        (void)gpu;
#endif
        for (const auto& rows : std::vector<std::pair<int64_t, int64_t>>{{1, 1}, {5, 9}, {0, 0}}) {
            detail::set_patch_batch_rows(rows.first, rows.second);
            for (int64_t pos : {8, 3}) {
                for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"}) {
                    const PatchResult r = patch_residual(m, t, corrupt, site, pos, 1, 2);
                    check_close(r.effect, naive_patch(m, t, corrupt, site, pos, 1, 2));
                    // After the last layer's attention, earlier positions cannot reach `pos`: exactly 0.
                    if (site != "resid_pre" && !std::isnan(r.effect.at({last, 0})))
                        for (int64_t q = 0; q < pos; ++q) CHECK(r.effect.at({last, q}) == 0.f);
                }
                check_close(patch_heads(m, t, corrupt, pos, 1, 2).effect, naive_patch_heads(m, t, corrupt, pos, 1, 2));
                check_close(patch_heads(m, t, corrupt, pos, 1, -1).effect, naive_patch_heads(m, t, corrupt, pos, 1, -1));
            }
            check_ablation_matches_naive(m, t);
        }
        // The thread count never changes a result.
        detail::set_patch_batch_rows(5, 9);
        set_num_threads(1);
        const Tensor p1 = patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect;
        const Tensor h1 = patch_heads(m, t, corrupt, 8, 1, 2).effect;
        set_num_threads(4);
        CHECK(num_threads() == 4);
        CHECK(max_abs_diff(patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect, p1) == 0.f);
        CHECK(max_abs_diff(patch_heads(m, t, corrupt, 8, 1, 2).effect, h1) == 0.f);
        set_num_threads(1);
        const std::vector<Tokens> refs = {seq(9, 61), seq(9, 62), seq(9, 63), seq(5, 64)};
        const Tensor a1 = head_ablation(m, t, Ablation::Mean, {refs.begin(), refs.begin() + 3});
        const Tensor pooled1 = head_means(m, refs, 9).layers.back();
        set_num_threads(4);
        CHECK(max_abs_diff(head_ablation(m, t, Ablation::Mean, {refs.begin(), refs.begin() + 3}), a1) == 0.f);
        CHECK(max_abs_diff(head_means(m, refs, 9).layers.back(), pooled1) == 0.f);
        set_num_threads(0);
    }
    detail::set_patch_batch_rows(0, 0);
#ifdef TFI_USE_ACCELERATE
    // A copy of the model gets its own GPU context (the context caches weights by address) and the
    // same results.
    if (m.gpu()) {
        detail::set_gpu_min_rows(1);
        const Model copy = m;
        CHECK(copy.gpu() != nullptr && copy.gpu() != m.gpu());
        check_close(patch_residual(copy, t, corrupt, "resid_pre", 8, 1, 2).effect,
                    patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect);
        // set_use_gpu(false) gives the CPU result bit for bit, even where the GPU would be used.
        detail::set_gpu_min_rows(std::numeric_limits<int64_t>::max());
        const Tensor cpu = patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect;
        const Tensor cpu_heads = patch_heads(m, t, corrupt, 8, 1, 2).effect;
        detail::set_gpu_min_rows(1);
        set_use_gpu(false);
        CHECK(max_abs_diff(patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect, cpu) == 0.f);
        CHECK(max_abs_diff(patch_heads(m, t, corrupt, 8, 1, 2).effect, cpu_heads) == 0.f);
        set_use_gpu(true);

        // fp16 GPU multiplies: really used, close to naive patching (errors around 1e-3 relative),
        // the same for any thread count, and no effect when the GPU is off.
        const Tensor gpu32 = patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect;
        set_gpu_fp16(true);
        CHECK(max_abs_diff(patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect, gpu32) > 0.f);
        auto check_fp16 = [](const Tensor& got, const Tensor& want) {
            CHECK(got.shape == want.shape);
            float scale = 1.f;
            for (float v : want.data) if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
            for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
                if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
                else CHECK_NEAR(got.data[i], want.data[i], 2e-2 * scale);
        };
        for (int64_t pos : {8, 3}) {
            for (const std::string site : {"resid_pre", "attn_out", "mlp_out"})
                check_fp16(patch_residual(m, t, corrupt, site, pos, 1, 2).effect, naive_patch(m, t, corrupt, site, pos, 1, 2));
            check_fp16(patch_heads(m, t, corrupt, pos, 1, 2).effect, naive_patch_heads(m, t, corrupt, pos, 1, 2));
        }
        set_num_threads(1);
        const Tensor f1 = patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect;
        set_num_threads(4);
        CHECK(max_abs_diff(patch_residual(m, t, corrupt, "attn_out", 8, 1, 2).effect, f1) == 0.f);
        set_num_threads(0);
        set_use_gpu(false);
        CHECK(max_abs_diff(patch_residual(m, t, corrupt, "resid_pre", 8, 1, 2).effect, cpu) == 0.f);
        set_use_gpu(true);
        set_gpu_fp16(false);
    }
    detail::set_gpu_min_rows(0);
#endif
    set_use_gpu(false);
    CHECK(!use_gpu());
    set_use_gpu(true);
    CHECK(use_gpu());
    CHECK(!gpu_fp16());
    set_gpu_fp16(true);
    CHECK(gpu_fp16());
    set_gpu_fp16(false);

}

static void test_patch_gap_threshold() {
    PatchResult r;
    r.clean_metric = 5.f, r.corrupt_metric = 5.f;
    CHECK(!r.normalized());
    r.corrupt_metric = 5.f - 1e-7f;  // float noise around a real metric
    CHECK(!r.normalized());
    r.corrupt_metric = 4.9f;           // small but genuine
    CHECK(r.normalized());
    r.clean_metric = 0.f, r.corrupt_metric = 1e-3f;
    CHECK(r.normalized());
    r.clean_metric = NAN;
    CHECK(!r.normalized());
}

static void test_mean_ablation(const Model& m) {
    const int64_t V = m.cfg().vocab_size;
    const Tokens t = seq(10, 21);
    const std::vector<Tokens> refs = {seq(10, 31), seq(10, 32), seq(10, 33)};
    const HeadMeans means = head_means(m, refs, 10);
    CHECK(means.per_position());

    // No future leak: editing only the input's last token never moves earlier logits.
    Tokens t2 = t;
    t2.back() = (t2.back() + 1) % V;
    for (int64_t l = 0; l < m.cfg().n_layers; ++l) {
        const Hooks hk = ablate_head(m, l, 0, Ablation::Mean, &means);
        CHECK(max_abs_diff(rows(m.forward(t, &hk), 0, 9), rows(m.forward(t2, &hk), 0, 9)) == 0.f);
    }

    // Means come from the references: an input that is its own reference is left unchanged, exactly
    // (every head already equals its mean, so no pass runs).
    for (float v : head_ablation(m, t, Ablation::Mean, {t}).data)
        if (!std::isnan(v)) CHECK(v == 0.f);

    // References of other lengths give one pooled mean, written at every position.
    const std::vector<Tokens> other = {seq(6, 41), seq(9, 42)};
    const HeadMeans pooled = head_means(m, other, 10);
    CHECK(!pooled.per_position());
    const int64_t H = m.layer_cfg(0).n_heads, D = m.layer_cfg(0).head_dim();
    std::vector<double> expect(D, 0.0);
    for (const Tokens& r : other) {
        ActivationCache c;
        m.run_with_cache(r, c, [](const std::string& n) { return n == "blocks.0.attn.hook_z"; });
        const Tensor& z = c.at("blocks.0.attn.hook_z");
        for (int64_t p = 0; p < z.shape[0]; ++p)
            for (int64_t e = 0; e < D; ++e) expect[e] += z.data[(p * H + 1) * D + e] / 15.0;  // 6 + 9 positions
    }
    const Hooks hk = ablate_head(m, 0, 1, Ablation::Mean, &pooled);
    ActivationCache after;
    m.run_with_cache(t, after, [](const std::string& n) { return n == "blocks.0.attn.hook_z"; }, &hk);
    const Tensor& z = after.at("blocks.0.attn.hook_z");
    for (int64_t p = 0; p < 10; p += 3)
        for (int64_t e = 0; e < D; ++e) CHECK_NEAR(z.data[(p * H + 1) * D + e], expect[e], 1e-5);

    // Refusals: no references, or per-position means used on an input of another length.
    CHECK_THROWS(head_ablation(m, t, Ablation::Mean));
    CHECK_THROWS(ablate_head(m, 0, 0, Ablation::Mean));
    const Hooks wrong = ablate_head(m, 0, 0, Ablation::Mean, &means);
    CHECK_THROWS(m.forward(seq(7, 1), &wrong));
    CHECK_THROWS(ablate_head(m, 0, 99, Ablation::Zero));
}

static void test_ablation_and_scores(const Model& m) {
    const Tensor abl = head_ablation(m, seq(12, 7), Ablation::Mean, {seq(12, 8), seq(12, 9)});
    CHECK(abl.shape == (Shape{2, 4}));
    for (float v : abl.data) CHECK(std::isfinite(v));
    const HeadScores s = head_scores(m, 6, 2, 0);
    CHECK(s.induction.shape == (Shape{2, 4}));
    for (float v : s.previous_token.data) CHECK(v >= 0.f && v <= 1.f + 1e-5f);
    CHECK(s.top_two_hop.size() == 5);
}

static void test_composition_matches_direct(const Model& m) {
    // The low-rank formula must equal the direct d x d computation.
    const Composition c = composition_scores(m);
    const int64_t H = m.cfg().n_heads;
    for (int64_t a = 0; a < H; a += 3)
        for (int64_t b = 0; b < H; b += 2) {
            const HeadWeights w = head_weights(m, 0, a), r = head_weights(m, 1, b);
            const Tensor ov = ov_matrix(w), qk = qk_matrix(r), ov2 = ov_matrix(r);
            const float k = frobenius(matmul(qk, ov)) / (frobenius(qk) * frobenius(ov));
            const float q = frobenius(matmul(transpose2d(qk), ov)) / (frobenius(qk) * frobenius(ov));
            const float v = frobenius(matmul(ov2, ov)) / (frobenius(ov2) * frobenius(ov));
            const int64_t N = 2 * H, i = a, j = H + b;
            CHECK_NEAR(c.k.data[i * N + j], k, 1e-4);
            CHECK_NEAR(c.q.data[i * N + j], q, 1e-4);
            CHECK_NEAR(c.v.data[i * N + j], v, 1e-4);
        }
    CHECK(std::isnan(c.k.data[0]));  // same layer
    const float base = composition_random_baseline(m, 16);
    CHECK(base > 0.f && base < 1.f);
}

// The OV path through d_model, as before the factored form: embed -> W_V^T W_O^T per head -> unembed.
static float direct_ov_path(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& path, int64_t max_tokens) {
    const int64_t V = m.cfg().vocab_size, d = m.cfg().d_model, n = std::min(V, max_tokens);
    Tensor X({n, d});
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < d; ++j) X.data[i * d + j] = m.weight("embed.weight").data[(i * V / n) * d + j];
    for (const auto& lh : path) {
        const HeadWeights w = head_weights(m, lh.first, lh.second);
        X = matmul(matmul(X, w.W_V, true), w.W_O, true);
    }
    Tensor U = m.unembed();
    if (m.cfg().norm != "none" && m.cfg().norm_position != "post")
        for (int64_t v = 0; v < V; ++v)
            for (int64_t j = 0; j < d; ++j) U.data[v * d + j] *= m.weight("ln_final.weight").data[j];
    const Tensor logits = matmul(X, U, true);
    int64_t hits = 0;
    for (int64_t i = 0; i < n; ++i) hits += topk(logits.ptr() + i * V, V, 1)[0] == i * V / n;
    return float(hits) / float(n);
}

// circuits: factored OV paths = the direct product (up to rounding at near-ties); the all-pairs
// scores = one path at a time, bit for bit; nothing depends on the thread count.
static void test_circuits_fast_paths(const Model& m) {
    const int64_t n = 40, H = m.cfg().n_heads, N = 2 * H;
    const Tensor single = ov_copying_scores(m, n), pair = ov_pair_copying_scores(m, n);
    CHECK(pair.shape == (Shape{N, N}));
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b) {
            const int64_t la = a / H, ha = a % H, lb = b / H, hb = b % H;
            if (la >= lb) {
                CHECK(std::isnan(pair.data[a * N + b]));
                continue;
            }
            CHECK(pair.data[a * N + b] == ov_path_copying(m, {{la, ha}, {lb, hb}}, n));
            CHECK_NEAR(pair.data[a * N + b], direct_ov_path(m, {{la, ha}, {lb, hb}}, n), 1.5 / n);  // a near-tie may flip
        }
    for (int64_t l = 0; l < 2; ++l)
        for (int64_t h = 0; h < H; ++h) {
            CHECK(single.at({l, h}) == ov_path_copying(m, {{l, h}}, n));
            CHECK_NEAR(single.at({l, h}), direct_ov_path(m, {{l, h}}, n), 1.5 / n);
        }
    auto same = [](const Tensor& x, const Tensor& y) {  // bit for bit, NaN where the other is NaN
        if (x.shape != y.shape) return false;
        for (size_t i = 0; i < x.data.size(); ++i)
            if (!(x.data[i] == y.data[i] || (std::isnan(x.data[i]) && std::isnan(y.data[i])))) return false;
        return true;
    };
    set_num_threads(1);
    const Composition c1 = composition_scores(m);
    const float base1 = composition_random_baseline(m, 8);
    const Tensor single1 = ov_copying_scores(m, n), pair1 = ov_pair_copying_scores(m, n);
    set_num_threads(4);
    const Composition c4 = composition_scores(m);
    CHECK(same(c1.q, c4.q) && same(c1.k, c4.k) && same(c1.v, c4.v));
    CHECK(composition_random_baseline(m, 8) == base1);
    CHECK(same(single1, single) && same(pair1, pair));
    set_num_threads(0);
    CHECK(composition_random_baseline(m, 8, 1) != base1);  // the seed matters
}

static void test_copying_and_positional() {
    // A head whose OV circuit is the identity copies every token.
    Model m = random_model(tiny([](ModelConfig& c) { c.mlp = "none"; c.norm = "none"; c.positional = "learned"; c.n_heads = 1; c.d_head = 16; }));
    auto w = m.weights();
    for (int64_t l = 0; l < 2; ++l) {
        const std::string p = "blocks." + std::to_string(l) + ".attn.";
        Tensor I({16, 16});
        for (int64_t i = 0; i < 16; ++i) I.at({i, i}) = 1.f;
        w[p + "v_proj.weight"] = I;
        w[p + "o_proj.weight"] = I;
    }
    Model id(m.cfg(), w);
    const Tensor copy = ov_copying_scores(id);
    CHECK(copy.data[0] > 0.9f);
    CHECK_NEAR(ov_path_copying(id, {{0, 0}, {1, 0}}), copy.data[0], 1e-6);  // identity twice is still identity
    CHECK_THROWS(ov_path_copying(id, {{1, 0}, {0, 0}}));
    CHECK(ov_path_copying(random_model(tiny()), {{0, 1}}) < 0.5f);
    const Tensor pqk = positional_qk(id, 0, 0, 8);
    CHECK(pqk.shape == (Shape{8, 8}));
    CHECK_NEAR(pqk.at({0, 0}), 1.0, 1e-6);
    CHECK(pqk.at({2, 5}) == 0.f);
    CHECK_THROWS(positional_qk(random_model(tiny()), 0, 0, 8));
}

static void test_post_norm_dla_refused() {
    Model m = random_model(tiny([](ModelConfig& c) { c.norm_position = "post"; c.norm = "layernorm"; }));
    CHECK_THROWS(direct_logit_attribution(m, seq(5, 1), 4, 1));
    // A single post-norm layer in a pre-norm model breaks additivity just the same.
    Model mixed = random_model(tiny([](ModelConfig& c) {
        c.norm = "layernorm";
        c.layer_overrides[1] = json::parse(R"({"norm_position": "post"})");
    }));
    CHECK_THROWS(direct_logit_attribution(mixed, seq(5, 1), 4, 1));
}

// The vectorizable exp, erf and tanh (src/vmath.h) against double precision, at every 16th float
// of their working range, plus special values.
static void test_vector_math() {
    auto ulp_error = [](float got, double want) {
        const float w = static_cast<float>(std::fabs(want));
        const double ulp = w < std::numeric_limits<float>::min() ? std::numeric_limits<float>::denorm_min()
                                                                  : double(std::nextafter(w, INFINITY)) - w;
        return std::fabs(double(got) - want) / ulp;
    };
    auto sweep = [&](float lo, float hi, auto f, auto ref, double max_ulp) {
        double worst = 0;
        int64_t step = 0;
        for (float x = lo; x <= hi; x = std::nextafter(x, INFINITY))
            if (step++ % 16 == 0) worst = std::max(worst, ulp_error(f(x), ref(double(x))));
        CHECK(worst <= max_ulp);
    };
    sweep(-87.3f, 88.7f, [](float x) { return vm::exp(x); }, [](double x) { return std::exp(x); }, 1.5);
    sweep(-6.f, 6.f, [](float x) { return vm::erf(x); }, [](double x) { return std::erf(x); }, 3.0);
    sweep(-12.f, 12.f, [](float x) { return vm::tanh(x); }, [](double x) { return std::tanh(x); }, 2.0);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(vm::exp(-inf) == 0.f && vm::exp(-200.f) == 0.f && vm::exp(inf) == inf && vm::exp(100.f) == inf);
    CHECK(vm::exp(0.f) == 1.f && vm::erf(0.f) == 0.f && vm::tanh(0.f) == 0.f);
    CHECK(vm::erf(inf) == 1.f && vm::erf(-inf) == -1.f && vm::tanh(inf) == 1.f && vm::tanh(-inf) == -1.f);
    CHECK(std::isnan(vm::exp(nan)) && std::isnan(vm::erf(nan)) && std::isnan(vm::tanh(nan)));
    // Softmax: masked (-inf) entries are exactly 0, the rest sum to 1.
    float row[11] = {0.5f, -inf, 3.f, -1.f, -inf, 2.f, 0.f, 1.f, -2.f, 4.f, -inf};
    softmax_(row, 11);
    CHECK(row[1] == 0.f && row[4] == 0.f && row[10] == 0.f);
    double total = 0;
    for (float v : row) total += v;
    CHECK_NEAR(total, 1.0, 1e-6);
    CHECK_NEAR(row[9] / row[2], std::exp(1.0), 1e-5);
}

#ifdef TFI_USE_ACCELERATE
// The GPU linear layer (macOS) against the CPU one, on awkward sizes, with and without a bias.
static void test_gpu_linear() {
    std::unique_ptr<detail::Gpu> gpu = detail::Gpu::create();
    if (!gpu) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd;
    for (const auto& dims : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {7, 13, 5}, {300, 67, 129}, {64, 512, 2048}}) {
        const int64_t m = dims[0], k = dims[1], n = dims[2];
        Tensor x({m, k}), W({n, k}), b({n});
        for (float& v : x.data) v = nd(rng);
        for (float& v : W.data) v = nd(rng);
        for (float& v : b.data) v = nd(rng);
        // A Gpu caches weights by address, so each must outlive the Gpu that saw it: a fresh one per W.
        const std::unique_ptr<detail::Gpu> g = detail::Gpu::create();
        for (const Tensor* bias : {static_cast<const Tensor*>(nullptr), static_cast<const Tensor*>(&b)}) {
            const Tensor want = linear(x, W, bias), got = g->linear(x, W, bias);
            CHECK(got.shape == want.shape);
            CHECK(max_abs_diff(got, want) <= 1e-4f * std::max(1.f, max_abs(want)));
        }
    }
    CHECK_THROWS(gpu->linear(Tensor({2, 3}), Tensor({4, 5}), nullptr));

    // fp16 inputs: close to fp32. A value outside fp16's range (in x or in W) falls back to fp32.
    Tensor x({300, 512}), W({257, 512}), b({257});
    for (float& v : x.data) v = nd(rng);
    for (float& v : W.data) v = nd(rng);
    for (float& v : b.data) v = nd(rng);
    const Tensor want = linear(x, W, &b);
    CHECK(max_abs_diff(gpu->linear(x, W, &b, /*fp16=*/true), want) <= 1e-2f * std::max(1.f, max_abs(want)));
    CHECK(max_abs_diff(gpu->linear(x, W, &b, true), want) > 0.f);  // really fp16
    Tensor xbig = x, Wbig = W;
    xbig.data[17] = 1e5f;
    Wbig.data[33] = -7e4f;
    for (const auto& [xi, Wi] : std::vector<std::pair<const Tensor*, const Tensor*>>{{&xbig, &W}, {&x, &Wbig}}) {
        const Tensor exact = linear(*xi, *Wi, &b), got = gpu->linear(*xi, *Wi, &b, true);
        CHECK(max_abs(got) < INFINITY);
        CHECK(max_abs_diff(got, exact) <= 1e-4f * std::max(1.f, max_abs(exact)));
    }

    // linear_many (one submission) = separate linear calls, bit for bit: weights of different widths
    // (as Q/K/V with grouped-query attention), with and without bias, an empty weight, and in fp16 a
    // weight outside fp16's range that alone falls back to fp32.
    Tensor Wk({64, 512}), Wv({64, 512}), Wempty({0, 512});
    for (float& v : Wk.data) v = nd(rng);
    for (float& v : Wv.data) v = nd(rng);
    const std::vector<const Tensor*> Ws = {&W, &Wk, &Wbig, &Wempty, &Wv};
    const std::vector<const Tensor*> bs = {&b, nullptr, &b, nullptr, nullptr};
    for (const bool fp16 : {false, true})
        for (const Tensor* xi : {&x, &xbig}) {
            const std::vector<Tensor> got = gpu->linear_many(*xi, Ws, bs, fp16);
            CHECK(got.size() == Ws.size());
            for (size_t i = 0; i < Ws.size(); ++i) {
                const Tensor want_i = gpu->linear(*xi, *Ws[i], bs[i], fp16);
                CHECK(got[i].shape == want_i.shape);
                CHECK(got[i].data == want_i.data);
            }
        }
    const std::vector<Tensor> nobias = gpu->linear_many(x, {&W, &Wk}, {});
    CHECK(nobias[0].data == gpu->linear(x, W, nullptr).data);
    CHECK(nobias[1].data == gpu->linear(x, Wk, nullptr).data);
    CHECK(gpu->linear_many(Tensor({0, 512}), {&W, &Wk}, {})[1].shape == Shape({0, 64}));
    CHECK_THROWS(gpu->linear_many(x, {&W, &Wk}, {&b}));  // one bias per weight
    const Tensor narrow({4, 5});
    CHECK_THROWS(gpu->linear_many(x, {&W, &narrow}, {}));  // wrong input width
}

// Gpu::Seq's operations against the CPU's. Each block creates its Gpu after its tensors, so the Gpu
// (which caches weights by address) is gone before they are.
static void test_gpu_seq() {
    if (!detail::Gpu::create()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd;
    auto near = [](float got, float want, float tol) {
        if (std::isnan(want)) CHECK(std::isnan(got));
        else CHECK_NEAR(got, want, tol * std::max(1.f, std::fabs(want)));
    };
    {   // Activations: the same polynomials as the CPU (vmath.h), over a wide range and NaN.
        Tensor xs({4002});
        for (int64_t i = 0; i < 4001; ++i) xs.data[i] = -20.f + 0.01f * static_cast<float>(i);
        xs.data[4001] = NAN;
        const auto g = detail::Gpu::create();
        detail::Gpu::Seq s(*g);
        const char* names[] = {"relu", "gelu", "gelu_tanh", "silu", "tanh", "relu2"};
        for (int a = 0; a < 6; ++a) {
            Tensor want = xs;
            detail::activate_(want, names[a]);
            std::copy(xs.data.begin(), xs.data.end(), s.reserve(0, xs.numel()));
            s.activate(0, xs.numel(), static_cast<detail::GpuAct>(a));
            s.run(false);
            for (int64_t i = 0; i < xs.numel(); ++i) near(s.data(0)[i], want.data[i], 1e-6f);
        }
        // glu, add, copy.
        const int64_t n = 1000;
        Tensor gate({n}), up({n});
        for (float& v : gate.data) v = 3.f * nd(rng);
        for (float& v : up.data) v = nd(rng);
        std::copy(gate.data.begin(), gate.data.end(), s.reserve(1, n));
        std::copy(up.data.begin(), up.data.end(), s.reserve(2, n));
        s.reserve(3, n);
        s.reserve(4, n);
        s.glu(1, 2, n, detail::GpuAct::Silu);
        s.add(1, 2, 3, n);
        s.copy(3, 4, n);
        s.run(false);
        Tensor want = gate;
        detail::activate_(want, "silu");
        for (int64_t i = 0; i < n; ++i) {
            near(s.data(1)[i], want.data[i] * up.data[i], 1e-6f);
            CHECK(s.data(3)[i] == s.data(1)[i] + up.data[i]);
            CHECK(s.data(4)[i] == s.data(3)[i]);
        }
    }
    {   // layernorm (with bias) and rmsnorm; rows wider than a threadgroup.
        const int64_t rows = 37, d = 300;
        Tensor x({rows, d}), w({d}), b({d});
        for (float& v : x.data) v = 3.f + 2.f * nd(rng);
        for (float& v : w.data) v = 1.f + 0.2f * nd(rng);
        for (float& v : b.data) v = 0.1f * nd(rng);
        const auto g = detail::Gpu::create();
        for (const bool layernorm : {true, false}) {
            detail::Gpu::Seq s(*g);
            std::copy(x.data.begin(), x.data.end(), s.reserve(0, rows * d));
            s.reserve(1, rows * d);
            s.norm(0, 1, rows, w, layernorm ? &b : nullptr, layernorm, 1e-5f);
            s.run(false);
            for (int64_t r = 0; r < rows; ++r) {
                double mean = 0, var = 0;
                for (int64_t j = 0; j < d; ++j) mean += layernorm ? x.data[r * d + j] / double(d) : 0.0;
                for (int64_t j = 0; j < d; ++j) var += (x.data[r * d + j] - mean) * (x.data[r * d + j] - mean) / double(d);
                for (int64_t j = 0; j < d; ++j) {
                    const double y = (x.data[r * d + j] - mean) / std::sqrt(var + 1e-5) * w.data[j] + (layernorm ? b.data[j] : 0.f);
                    near(s.data(1)[r * d + j], static_cast<float>(y), 1e-5f);
                }
            }
            CHECK_THROWS(s.norm(0, 1, rows, w, &b, false, 1e-5f));  // rmsnorm has no bias
        }
    }
    {   // linear (+ bias) -> activation -> linear -> add, in fp32 and fp16. An input outside fp16's
        // range, first or in the middle (after an in-place activation), reruns the whole thing in fp32.
        const int64_t R = 300, k = 64, h = 96;
        Tensor x({R, k}), W1({h, k}), b1({h}), W2({k, h});
        for (float& v : x.data) v = nd(rng);
        for (float& v : W1.data) v = 0.3f * nd(rng);
        for (float& v : b1.data) v = 0.1f * nd(rng);
        for (float& v : W2.data) v = 0.3f * nd(rng);
        Tensor b1big = b1;
        b1big.data[3] = 1e5f;  // hidden column 3 overflows fp16, only as the second multiply's input
        Tensor xbig = x;
        xbig.data[5] = 1e5f;
        const auto g = detail::Gpu::create();
        auto run_seq = [&](const Tensor& xin, const Tensor& bias, bool fp16) {
            detail::Gpu::Seq s(*g);
            std::copy(xin.data.begin(), xin.data.end(), s.reserve(0, R * k));
            s.reserve(1, R * h);
            s.reserve(2, R * k);
            s.reserve(3, R * k);
            s.linear(0, 1, R, W1, &bias);
            s.activate(1, R * h, detail::GpuAct::Gelu);
            s.linear(1, 2, R, W2, nullptr);
            s.add(2, 0, 3, R * k);
            s.run(fp16);
            return Tensor({R, k}, std::vector<float>(s.data(3), s.data(3) + R * k));
        };
        auto cpu = [&](const Tensor& xin, const Tensor& bias) {
            Tensor hid = linear(xin, W1, &bias);
            detail::activate_(hid, "gelu");
            Tensor y = linear(hid, W2, nullptr);
            for (int64_t i = 0; i < y.numel(); ++i) y.data[i] += xin.data[i];
            return y;
        };
        const Tensor want = cpu(x, b1), f32 = run_seq(x, b1, false), f16 = run_seq(x, b1, true);
        CHECK(max_abs_diff(f32, want) <= 1e-4f * std::max(1.f, max_abs(want)));
        CHECK(max_abs_diff(f16, want) <= 2e-2f * std::max(1.f, max_abs(want)));
        CHECK(max_abs_diff(f16, f32) > 0.f);  // really fp16
        for (const auto& [xi, bi] : std::vector<std::pair<const Tensor*, const Tensor*>>{{&xbig, &b1}, {&x, &b1big}}) {
            const Tensor over = run_seq(*xi, *bi, true);
            CHECK(over.data == run_seq(*xi, *bi, false).data);
            const Tensor exact = cpu(*xi, *bi);
            CHECK(max_abs_diff(over, exact) <= 1e-4f * std::max(1.f, max_abs(exact)));
        }
    }
    {   // Slots must be reserved and large enough; a failed run records nothing further.
        Tensor W({4, 5});
        const auto g = detail::Gpu::create();
        detail::Gpu::Seq s(*g);
        s.reserve(0, 10);
        s.linear(0, 1, 2, W, nullptr);
        CHECK_THROWS(s.run(false));  // slot 1 not reserved
        s.reserve(1, 7);
        s.linear(0, 1, 2, W, nullptr);
        CHECK_THROWS(s.run(false));  // slot 1 holds 7 < 2 x 4
        s.reserve(1, 8);
        s.run(false);  // nothing left to run
        s.linear(0, 1, 2, W, nullptr);
        s.run(false);
        for (int64_t i = 0; i < 8; ++i) CHECK(s.data(1)[i] == 0.f);
        CHECK_THROWS(s.data(9));
    }
}

// Model types the GPU blocks must handle beyond configs(): post-norm (with and without the
// parallel residual), every activation, GLU variants, qk_norm, sandwich without a norm, and layers
// of different sizes.
static std::vector<std::pair<std::string, ModelConfig>> gpu_block_configs() {
    std::vector<std::pair<std::string, ModelConfig>> v = configs();
    v.push_back({"post-norm layernorm, gelu_tanh mlp", tiny([](ModelConfig& c) {
                     c.norm_position = "post"; c.norm = "layernorm"; c.bias = true; c.mlp = "mlp"; c.activation = "gelu_tanh";
                 })});
    v.push_back({"post-norm parallel, relu", tiny([](ModelConfig& c) {
                     c.norm_position = "post"; c.parallel_residual = true; c.mlp = "mlp"; c.activation = "relu";
                 })});
    v.push_back({"post-norm, no norm, attention-only", tiny([](ModelConfig& c) {
                     c.norm_position = "post"; c.norm = "none"; c.mlp = "none";
                 })});
    v.push_back({"geglu tanh + qk_norm", tiny([](ModelConfig& c) { c.mlp = "geglu"; c.activation = "gelu_tanh"; c.qk_norm = true; })});
    v.push_back({"glu relu2", tiny([](ModelConfig& c) { c.mlp = "glu"; c.activation = "relu2"; })});
    v.push_back({"sandwich layernorm, tanh mlp", tiny([](ModelConfig& c) {
                     c.norm_position = "sandwich"; c.norm = "layernorm"; c.bias = true; c.mlp = "mlp"; c.activation = "tanh";
                 })});
    v.push_back({"sandwich, no norm", tiny([](ModelConfig& c) { c.norm_position = "sandwich"; c.norm = "none"; })});
    v.push_back({"per-layer sizes", tiny([](ModelConfig& c) {
                     c.n_layers = 3;
                     c.layer_overrides[0] = json::parse(R"({"n_heads": 2, "d_mlp": 24})");
                     c.layer_overrides[2] = json::parse(R"({"n_heads": 8, "d_head": 2, "d_mlp": 72})");
                 })});
    return v;
}

// Batched passes with the residual stream kept on the GPU (Model::run_rows): unpatched rows equal
// the plain forward pass, and every pass equals the one-linear-layer-at-a-time GPU path, with hooks
// on CPU-side activations (q, z) and on GPU-side ones (which bring that block back to the CPU).
static void test_gpu_resident(const Model& m) {
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    detail::set_gpu_min_rows(1);
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model, T = 9;
    const Tokens t = seq(T, 7);
    ActivationCache cache;
    m.run_with_cache(t, cache);
    auto lp = [](int64_t i) { return "blocks." + std::to_string(i) + "."; };
    const Tensor fin = m.final_norm(cache.at(lp(L - 1) + "hook_resid_post"));
    // A whole sequence and two that start later, reading earlier keys and values from the cache.
    detail::RowBatch rb;
    for (int64_t i = 0; i < L; ++i) {
        rb.prefix_k.push_back(&cache.at(lp(i) + "attn.hook_k"));
        rb.prefix_v.push_back(&cache.at(lp(i) + "attn.hook_v"));
    }
    rb.gpu = m.gpu();
    for (const auto& [s0, n] : std::vector<std::pair<int64_t, int64_t>>{{0, T}, {4, T - 4}, {2, 5}}) rb.add(s0, n);
    auto stack = [&](const Tensor& full) {
        Tensor x({rb.rows(), d});
        for (size_t s = 0; s < rb.start.size(); ++s)
            std::copy(full.ptr() + rb.start[s] * d, full.ptr() + (rb.start[s] + rb.len[s]) * d, x.ptr() + rb.offset[s] * d);
        return x;
    };
    std::map<std::string, int> calls;
    Hooks hooks;
    const std::string edited = m.cfg().has_mlp() ? "hook_mlp_out" : "hook_attn_out";
    hooks.add_filter([](const std::string& n) { return n.find("attn.hook_q") != std::string::npos || n.find("attn.hook_z") != std::string::npos; },
                     [&](const std::string& n, Tensor& a) { ++calls[n]; a.data[0] += 1.f; });
    for (int64_t start = 0; start < L; ++start) {
        const Tensor x = stack(cache.at(lp(start) + "hook_resid_pre"));
        auto pass = [&](bool resident, bool fp16, const Hooks* h) {
            detail::set_gpu_resident(resident);
            rb.fp16 = fp16;
            const Tensor r = m.forward_rows_final(start, x, rb, h);
            detail::set_gpu_resident(true);
            rb.fp16 = false;
            return r;
        };
        const Tensor got = pass(true, false, nullptr);
        check_close(got, stack(fin));
        check_close(got, pass(false, false, nullptr));
        Hooks h = hooks;
        const int64_t mid = std::min(start + 1, L - 1);
        h.add(lp(mid) + edited, [&](const std::string& n, Tensor& a) {
            ++calls[n];
            for (float& v : a.data) v += 0.25f;
        });
        h.add(lp(L - 1) + "hook_resid_pre", [&](const std::string& n, Tensor& a) {
            ++calls[n];
            for (float& v : a.data) v *= 1.1f;
        });
        calls.clear();
        const Tensor a = pass(true, false, &h);
        const std::map<std::string, int> seen = calls;
        calls.clear();
        check_close(a, pass(false, false, &h));
        CHECK(seen == calls);
        CHECK(seen.count(lp(mid) + edited) == 1 && seen.count(lp(L - 1) + "attn.hook_z") == 1);
        CHECK(max_abs_diff(a, got) > 0.f);
        // fp16: close, and really used.
        const Tensor h16 = pass(true, true, nullptr);
        CHECK(max_abs_diff(h16, got) <= 3e-2f * std::max(1.f, max_abs(got)));
        CHECK(max_abs_diff(h16, got) > 0.f);
    }
    detail::set_gpu_min_rows(0);
}

// Batched gradient passes on the GPU (Model::gradients_rows) = one run_with_gradients per sequence,
// at every hook point they support, for log p(target) and a logit difference at each sequence's end.
static void test_gpu_gradients(const Model& m) {
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const int64_t L = m.cfg().n_layers, V = m.cfg().vocab_size, d = m.cfg().d_model;
    const std::vector<Tokens> seqs = {seq(9, 21), seq(5, 22), seq(12, 23)};
    detail::RowBatch rb;
    rb.gpu = m.gpu();
    for (const Tokens& t : seqs) rb.add(0, static_cast<int64_t>(t.size()));
    Tensor x0({rb.rows(), d});
    for (size_t g = 0; g < seqs.size(); ++g) {
        const Tensor e = m.embed(seqs[g]);
        std::copy(e.ptr(), e.ptr() + e.numel(), x0.ptr() + rb.offset[g] * d);
    }
    std::vector<std::string> names = {"hook_embed"};
    for (int64_t i = 0; i < L; ++i)
        for (const char* n : {"hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_attn_out", "hook_mlp_out", "attn.hook_z"})
            names.push_back("blocks." + std::to_string(i) + "." + n);
    for (const int64_t other : {int64_t(-1), int64_t(7)}) {
        // dF/dlogits of one row: log softmax at `target`, or logit[target] - logit[other].
        auto row_grad = [&](const float* logits, float* out) {
            if (other >= 0) {
                out[3] += 1.f;
                out[other] -= 1.f;
                return;
            }
            float mx = -INFINITY, sum = 0.f;
            for (int64_t j = 0; j < V; ++j) mx = std::max(mx, logits[j]);
            for (int64_t j = 0; j < V; ++j) sum += std::exp(logits[j] - mx);
            for (int64_t j = 0; j < V; ++j) out[j] = -std::exp(logits[j] - mx) / sum;
            out[3] += 1.f;
        };
        const std::vector<ActivationCache> got = m.gradients_rows(
            x0, rb,
            [&](int64_t, const Tensor& logits) {
                Tensor dl(logits.shape, 0.f);
                row_grad(logits.ptr(), dl.ptr());
                return dl;
            },
            nullptr);
        CHECK(got.size() == seqs.size());
        for (size_t g = 0; g < seqs.size() && g < got.size(); ++g) {
            ActivationCache want;
            m.run_with_gradients(
                seqs[g],
                [&](const Tensor& logits) {
                    Tensor dl(logits.shape, 0.f);
                    const int64_t last = logits.shape[0] - 1;
                    row_grad(logits.ptr() + last * V, dl.ptr() + last * V);
                    return dl;
                },
                want);
            for (const std::string& n : names) {
                CHECK(got[g].count(n) == want.count(n));
                if (!got[g].count(n) || !want.count(n)) continue;
                const Tensor &a = got[g].at(n), &b = want.at(n);
                CHECK(a.shape == b.shape);
                if (a.shape == b.shape) CHECK(max_abs_diff(a, b) <= 1e-4f * std::max(1.f, max_abs(b)));
            }
        }
    }
    // Only the filter's hook points, and every segment must start at 0.
    const auto only_embed = m.gradients_rows(x0, rb, [&](int64_t, const Tensor& l) { return Tensor(l.shape, 0.f); },
                                             [](const std::string& n) { return n == "hook_embed"; });
    CHECK(only_embed[0].size() == 1 && only_embed[0].count("hook_embed") == 1);
    detail::RowBatch later = rb;
    later.start[1] = 2;
    CHECK_THROWS(m.gradients_rows(x0, later, [&](int64_t, const Tensor& l) { return Tensor(l.shape, 0.f); }, nullptr));
}
#endif

static void test_nan_never_matches() {
    Tensor a({3}, {1.f, 2.f, 3.f}), b = a;
    b.data[1] = NAN;
    CHECK(std::isnan(max_abs_diff(a, b)));
    CHECK(std::isnan(max_abs_diff(b, a)));
    CHECK(std::isnan(max_abs_diff(b, b)));
    CHECK(!(max_abs_diff(a, b) <= 1e-4f));  // the golden test's comparison must fail
    Tensor inf = a;
    inf.data[0] = INFINITY;
    CHECK(max_abs_diff(inf, inf) == 0.f);
    CHECK(std::isinf(max_abs_diff(inf, a)));
}

// Layers with different head counts and head sizes: every per-head tool must use each layer's own count.
static void test_heterogeneous_heads() {
    const ModelConfig c = tiny([](ModelConfig& c) {
        c.n_layers = 3;
        c.layer_overrides[0] = json::parse(R"({"n_heads": 2})");                // 2 heads of size 8
        c.layer_overrides[2] = json::parse(R"({"n_heads": 8, "d_head": 2})");   // 8 heads of size 2
    });
    const Model m = random_model(c, 3);
    CHECK(m.max_heads() == 8);
    const Tokens t = seq(10, 2);
    Tokens corrupt = t;
    corrupt[4] = (corrupt[4] + 1) % 50;
    auto check_layout = [](const Tensor& x) {
        CHECK(x.shape == (Shape{3, 8}));
        CHECK(std::isfinite(x.at({0, 1})));
        CHECK(std::isnan(x.at({0, 2})));  // layer 0 has only 2 heads
        CHECK(std::isnan(x.at({1, 4})));  // layer 1 has 4
        CHECK(std::isfinite(x.at({2, 7})));
    };
    const HeadScores s = head_scores(m, 5, 2, 0);
    check_layout(s.induction);
    check_layout(s.previous_token);
    CHECK(s.top_two_hop.size() == 5);
    check_layout(head_ablation(m, t, Ablation::Mean, {seq(10, 5), seq(10, 6)}));
    check_layout(head_ablation(m, t, Ablation::Zero));
    check_layout(patch_heads(m, t, corrupt, 9, 1, 2).values);
    check_layout(attribution_patch_heads(m, t, corrupt, 9, 1, 2).values);
    const PatchResult mixed = mixed_patch_heads(m, t, corrupt, 9, 1, 2, 1000);
    check_layout(mixed.values);
    for (int64_t i = 0; i < mixed.effect.numel(); ++i)  // every existing head measured, no missing one
        CHECK(mixed.exact[i] == (std::isnan(mixed.estimate.data[i]) ? 0 : 1));
    check_layout(ov_copying_scores(m, 32));

    const Composition comp = composition_scores(m);
    CHECK(comp.labels.size() == 14);
    CHECK(comp.labels[13] == "L2H7");
    CHECK(std::isnan(comp.k.at({0, 1})));  // same layer
    // L0H1 -> L2H3 (indices 1 and 9) matches the direct d x d computation despite different head sizes.
    const HeadWeights w = head_weights(m, 0, 1), r = head_weights(m, 2, 3);
    const Tensor ov = ov_matrix(w), qk = qk_matrix(r);
    CHECK_NEAR(comp.k.at({1, 9}), frobenius(matmul(qk, ov)) / (frobenius(qk) * frobenius(ov)), 1e-4);
    // Two-head OV paths use the same head order and handle different head sizes.
    const Tensor pair = ov_pair_copying_scores(m, 32);
    CHECK(pair.shape == (Shape{14, 14}));
    CHECK(std::isnan(pair.at({0, 1})) && std::isnan(pair.at({9, 2})));
    CHECK(pair.at({1, 9}) == ov_path_copying(m, {{0, 1}, {2, 3}}, 32));
    CHECK(pair.at({3, 13}) == ov_path_copying(m, {{1, 1}, {2, 7}}, 32));

    const DLAResult dla = direct_logit_attribution(m, t, 9, 1, 2);
    CHECK_NEAR(dla.sum, dla.actual, 1e-3 * std::max(1.f, std::fabs(dla.actual)));
    CHECK(std::find(dla.labels.begin(), dla.labels.end(), "L2H7") != dla.labels.end());
    CHECK(std::find(dla.labels.begin(), dla.labels.end(), "L0H2") == dla.labels.end());
}

// The backward pass against central finite differences of the metric, at every hook point. The
// direction mixes the computed gradient with noise, so a wrong gradient can't agree by accident.
static void test_gradients(const Model& m) {
    const Tokens t = seq(7, 3);
    const int64_t pos = 5;
    for (const int64_t other : {int64_t(3), int64_t(-1)}) {  // logit difference, then log-probability
        ActivationCache grads, acts;
        const float metric = metric_gradients(m, t, pos, 7, other, grads, nullptr, &acts);
        CHECK(metric == logit_metric(m.forward(t), pos, 7, other));  // the same forward pass, bit for bit
        std::mt19937_64 rng(11);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (const std::string& n : m.hook_names()) {
            CHECK(grads.count(n) == 1);
            if (!grads.count(n)) continue;
            const Tensor& g = grads.at(n);
            CHECK(g.shape == acts.at(n).shape);
            Tensor r(g.shape);
            for (float& v : r.data) v = normal(rng);
            const float gn = frobenius(g), rn = frobenius(r);
            Tensor v(g.shape);
            for (int64_t i = 0; i < v.numel(); ++i) v.data[i] = (gn > 0 ? g.data[i] / gn : 0.f) + 0.5f * r.data[i] / rn;
            auto f = [&](double e) {
                Hooks h;
                h.add(n, [&](const std::string&, Tensor& a) {
                    for (int64_t i = 0; i < a.numel(); ++i) a.data[i] += static_cast<float>(e) * v.data[i];
                });
                return double(logit_metric(m.forward(t, &h), pos, 7, other));
            };
            // Central differences at h and h/2, Richardson-extrapolated: error O(h^4) instead of O(h^2).
            const double h = 1e-2, fd1 = (f(h) - f(-h)) / (2 * h), fd2 = (f(h / 2) - f(-h / 2)) / h;
            const double fd = (4 * fd2 - fd1) / 3;
            const double an = dot(g.ptr(), v.ptr(), g.numel());
            const bool ok = std::fabs(fd - an) <= 2e-3 * std::max({1.0, std::fabs(fd), std::fabs(an)});
            CHECK(ok);
            if (!ok) std::printf("    %s (other %lld): backward %.6g, finite differences %.6g\n", n.c_str(),
                                 static_cast<long long>(other), an, fd);
        }
    }
}

static void test_attribution_patching(const Model& m) {
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t L = m.cfg().n_layers;
    for (const std::string site : {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"}) {
        const PatchResult exact = patch_residual(m, clean, corrupt, site, 6, 1, 2);
        const PatchResult atp = attribution_patch_residual(m, clean, corrupt, site, 6, 1, 2);
        CHECK(atp.effect.shape == exact.effect.shape);
        CHECK(atp.clean_metric == exact.clean_metric);
        CHECK(atp.corrupt_metric == exact.corrupt_metric);
        for (int64_t i = 0; i < L; ++i)
            for (int64_t p = 0; p < 8; ++p) {
                const float e = atp.effect.at({i, p});
                CHECK(std::isnan(e) == std::isnan(exact.effect.at({i, p})));  // same missing sites
                if ((p < 3 || p == 7) && !std::isnan(e)) CHECK(e == 0.f);  // unchanged positions, and one after `pos`
            }
    }
    const PatchResult heads = attribution_patch_heads(m, clean, corrupt, 6, 1, 2);
    const PatchResult exact_heads = patch_heads(m, clean, corrupt, 6, 1, 2);
    CHECK(heads.effect.shape == exact_heads.effect.shape);
    for (int64_t i = 0; i < heads.effect.numel(); ++i)
        CHECK(std::isnan(heads.effect.data[i]) == std::isnan(exact_heads.effect.data[i]));
    const PatchResult same = attribution_patch_residual(m, clean, clean, "resid_pre", 6, 1, 2);
    CHECK(!same.normalized());
    for (float e : same.effect.data) CHECK(e == 0.f);
    CHECK_THROWS(attribution_patch_residual(m, clean, corrupt, "nope", 6, 1, 2));
    CHECK_THROWS(attribution_patch_heads(m, clean, Tokens(7, 0), 6, 1, 2));
    CHECK_THROWS(attribution_patch_residual(m, clean, corrupt, "resid_pre", 6, 1, 2, 0));

    // Integrated-gradient steps run in parallel; the result must not depend on the thread count.
    set_num_threads(1);
    const PatchResult one = attribution_patch_heads(m, clean, corrupt, 6, 1, -1, 5);
    set_num_threads(4);
    const PatchResult four = attribution_patch_heads(m, clean, corrupt, 6, 1, -1, 5);
    set_num_threads(0);
    for (int64_t i = 0; i < one.effect.numel(); ++i)
        CHECK(std::isnan(one.effect.data[i]) ? std::isnan(four.effect.data[i]) : one.effect.data[i] == four.effect.data[i]);
}

// With no final norm and no MLP, the logits are linear in the last layer's outputs, so attribution
// patching there must equal exact patching.
static void test_attribution_exact_when_linear() {
    const Model m = random_model(tiny([](ModelConfig& c) { c.mlp = "none"; c.norm = "none"; c.n_layers = 3; }), 5);
    const Tokens clean = seq(9, 2);
    Tokens corrupt = clean;
    corrupt[2] = (corrupt[2] + 11) % 50;
    corrupt[6] = (corrupt[6] + 3) % 50;
    const PatchResult exact = patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9);
    const PatchResult atp = attribution_patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9);
    for (int64_t p = 0; p < 9; ++p)
        CHECK_NEAR(atp.effect.at({2, p}), exact.effect.at({2, p}), 1e-4 * std::max(1.f, std::fabs(exact.effect.at({2, p}))));
    const PatchResult eh = patch_heads(m, clean, corrupt, 8, 4, 9), ah = attribution_patch_heads(m, clean, corrupt, 8, 4, 9);
    for (int64_t h = 0; h < 4; ++h)
        CHECK_NEAR(ah.effect.at({2, h}), eh.effect.at({2, h}), 1e-4 * std::max(1.f, std::fabs(eh.effect.at({2, h}))));
    // There the gradient is constant, so averaging it over interpolated inputs changes nothing.
    const PatchResult ig = attribution_patch_residual(m, clean, corrupt, "resid_post", 8, 4, 9, 4);
    for (int64_t p = 0; p < 9; ++p)
        CHECK_NEAR(ig.effect.at({2, p}), exact.effect.at({2, p}), 1e-4 * std::max(1.f, std::fabs(exact.effect.at({2, p}))));
    // Earlier layers feed attention patterns, which are nonlinear: there the estimate is only approximate.
    bool differs = false;
    for (int64_t p = 0; p < 9; ++p) differs |= std::fabs(atp.effect.at({0, p}) - exact.effect.at({0, p})) > 1e-3f;
    CHECK(differs);
}

// Mixed patching: exact values at the top-K estimated sites, attribution estimates elsewhere.
static void check_mixed(const PatchResult& mixed, const PatchResult& atp, const PatchResult& exact, int64_t verify,
                        int64_t usable_after_pos_cols /* residual: first column past pos, else -1 */) {
    const int64_t cols = mixed.effect.shape[1];
    CHECK(mixed.estimate.shape == atp.effect.shape);
    CHECK(mixed.exact.size() == static_cast<size_t>(mixed.effect.numel()));
    int64_t flagged = 0, usable = 0;
    float min_flagged = INFINITY, max_unflagged = 0;
    for (int64_t i = 0; i < mixed.effect.numel(); ++i) {
        const float est = atp.effect.data[i];
        CHECK(std::isnan(est) ? std::isnan(mixed.estimate.data[i]) : mixed.estimate.data[i] == est);
        const bool can = !std::isnan(est) && (usable_after_pos_cols < 0 || i % cols < usable_after_pos_cols);
        usable += can;
        if (mixed.exact[i]) {
            ++flagged;
            CHECK(can);
            // Measured: the same up to float rounding (batched passes group different sites).
            CHECK_NEAR(mixed.effect.data[i], exact.effect.data[i], 1e-5 * std::max(1.f, std::fabs(exact.effect.data[i])));
            min_flagged = std::min(min_flagged, std::fabs(est));
        } else {
            CHECK(std::isnan(est) ? std::isnan(mixed.effect.data[i]) : mixed.effect.data[i] == est);
            if (can) max_unflagged = std::max(max_unflagged, std::fabs(est));
        }
    }
    CHECK(flagged == std::min(verify, usable));
    if (flagged > 0 && flagged < usable) CHECK(min_flagged >= max_unflagged);  // the largest estimates were measured
    if (!std::isnan(mixed.gap()) && mixed.normalized())
        for (int64_t i = 0; i < mixed.effect.numel(); ++i)
            if (!std::isnan(mixed.effect.data[i])) CHECK_NEAR(mixed.values.data[i], mixed.effect.data[i] / mixed.gap(), 1e-5);
}

static void test_mixed_patching(const Model& m) {
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t pos = 6;
    for (const std::string site : {"resid_pre", "attn_out", "mlp_out"}) {
        const PatchResult atp = attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, 2);
        const PatchResult exact = patch_residual(m, clean, corrupt, site, pos, 1, 2);
        for (const int64_t verify : {int64_t(0), int64_t(3), int64_t(1000)})
            check_mixed(mixed_patch_residual(m, clean, corrupt, site, pos, 1, 2, verify, 2), atp, exact, verify, pos + 1);
        // Verifying every site reproduces exact patching.
        const PatchResult all = mixed_patch_residual(m, clean, corrupt, site, pos, 1, 2, 1000, 2);
        for (int64_t i = 0; i < all.effect.numel(); ++i)
            CHECK(std::isnan(exact.effect.data[i]) ? std::isnan(all.effect.data[i]) : all.effect.data[i] == exact.effect.data[i]);
    }
    const PatchResult atp = attribution_patch_heads(m, clean, corrupt, pos, 1, -1);
    const PatchResult exact = patch_heads(m, clean, corrupt, pos, 1, -1);
    for (const int64_t verify : {int64_t(0), int64_t(2), int64_t(1000)})
        check_mixed(mixed_patch_heads(m, clean, corrupt, pos, 1, -1, verify), atp, exact, verify, -1);
    CHECK_THROWS(mixed_patch_heads(m, clean, corrupt, pos, 1, -1, -1));
}

#ifdef TFI_USE_ACCELERATE
// Attribution and mixed patching with their gradient runs batched on the GPU: close to the CPU's
// results for any batch size, the same for any thread count, and the CPU tests' invariants hold.
static void test_gpu_attribution(const Model& m) {
    if (!m.gpu()) {
        std::printf("     (no Metal device: GPU path not tested)\n");
        return;
    }
    const Tokens clean = seq(8, 6);
    Tokens corrupt = clean;
    corrupt[3] = (corrupt[3] + 7) % m.cfg().vocab_size;
    const int64_t pos = 6;
    auto close = [](const Tensor& got, const Tensor& want) {
        CHECK(got.shape == want.shape);
        float scale = 1.f;
        for (float v : want.data) if (!std::isnan(v)) scale = std::max(scale, std::fabs(v));
        for (int64_t i = 0; i < got.numel() && got.shape == want.shape; ++i)
            if (std::isnan(want.data[i])) CHECK(std::isnan(got.data[i]));
            else CHECK_NEAR(got.data[i], want.data[i], 1e-4 * scale);
    };
    bool differs = false;  // the GPU really ran
    for (const int64_t steps : {1, 3}) {
        set_use_gpu(false);
        std::vector<Tensor> cpu;
        for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"})
            cpu.push_back(attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, steps).effect);
        cpu.push_back(attribution_patch_heads(m, clean, corrupt, pos, 1, -1, steps).effect);
        set_use_gpu(true);
        detail::set_gpu_min_rows(1);
        for (const int64_t rows : {0, 7, 1000}) {  // default, one sequence per batch, all in one
            detail::set_patch_batch_rows(0, rows);
            size_t i = 0;
            for (const std::string site : {"resid_pre", "resid_mid", "attn_out", "mlp_out", "resid_post"}) {
                const Tensor g = attribution_patch_residual(m, clean, corrupt, site, pos, 1, 2, steps).effect;
                close(g, cpu[i]);
                differs |= max_abs_diff(g, cpu[i++]) > 0.f;
            }
            close(attribution_patch_heads(m, clean, corrupt, pos, 1, -1, steps).effect, cpu[i]);
        }
        detail::set_patch_batch_rows(0, 0);
        detail::set_gpu_min_rows(0);
    }
    CHECK(differs);
    detail::set_gpu_min_rows(1);
    set_num_threads(1);
    const Tensor a1 = attribution_patch_residual(m, clean, corrupt, "attn_out", pos, 1, 2, 5).effect;
    const Tensor h1 = mixed_patch_heads(m, clean, corrupt, pos, 1, -1, 3, 5).effect;
    set_num_threads(4);
    auto same = [](const Tensor& a, const Tensor& b) {  // bit for bit, NaN matching NaN (missing heads)
        CHECK(a.shape == b.shape);
        for (int64_t i = 0; i < a.numel() && a.shape == b.shape; ++i)
            CHECK(std::isnan(b.data[i]) ? std::isnan(a.data[i]) : a.data[i] == b.data[i]);
    };
    same(attribution_patch_residual(m, clean, corrupt, "attn_out", pos, 1, 2, 5).effect, a1);
    same(mixed_patch_heads(m, clean, corrupt, pos, 1, -1, 3, 5).effect, h1);
    set_num_threads(0);
    test_attribution_patching(m);
    test_mixed_patching(m);
    detail::set_gpu_min_rows(0);
}
#endif

int main() {
    auto run = [](const std::string& name, const std::function<void()>& fn) {
        const int before = g_failures;
        try {
            fn();
        } catch (const std::exception& e) {
            ++g_failures;
            std::printf("  EXCEPTION: %s\n", e.what());
        }
        std::printf("%s %s\n", g_failures == before ? "ok  " : "FAIL", name.c_str());
    };
    run("json", test_json);
    run("reshape + linear", test_reshape_and_linear);
    run("matmul", test_matmul);
    for (const auto& kv : configs()) {
        const Model m = random_model(kv.second, 42);
        run("cache/hooks/causality: " + kv.first, [&] { test_cache_and_hooks(kv.first, m); });
        run("logit lens: " + kv.first, [&] { test_logit_lens(m); });
        run("direct logit attribution: " + kv.first, [&] { test_dla(m); });
        run("activation patching: " + kv.first, [&] { test_patching(m); });
        run("patching edge cases (zero gap, missing sites): " + kv.first, [&] { test_patching_edge_cases(m); });
        run("optimized patching (batched, GPU) and ablation = naive: " + kv.first, [&] { test_optimized_patching_matches_naive(m); });
        run("ablation + head scores: " + kv.first, [&] { test_ablation_and_scores(m); });
        run("mean ablation has no future leak: " + kv.first, [&] { test_mean_ablation(m); });
        run("gradients = finite differences: " + kv.first, [&] { test_gradients(m); });
        run("attribution patching: " + kv.first, [&] { test_attribution_patching(m); });
        run("mixed patching: " + kv.first, [&] { test_mixed_patching(m); });
        if (kv.second.kv_heads() == kv.second.n_heads)
            run("composition = direct: " + kv.first, [&] { test_composition_matches_direct(m); });
        run("circuits: factored, parallel = direct: " + kv.first, [&] { test_circuits_fast_paths(m); });
    }
    run("OV copying + positional QK", test_copying_and_positional);
    run("post-norm DLA refused", test_post_norm_dla_refused);
    run("vector math", test_vector_math);
#ifdef TFI_USE_ACCELERATE
    run("GPU linear layer (macOS)", test_gpu_linear);
    run("GPU block operations (macOS)", test_gpu_seq);
    for (const auto& kv : gpu_block_configs())
        run("GPU-resident batched passes = per-layer and plain (macOS): " + kv.first,
            [&] { test_gpu_resident(random_model(kv.second, 43)); });
    for (const auto& kv : gpu_block_configs())
        run("GPU batched gradients = run_with_gradients (macOS): " + kv.first,
            [&] { test_gpu_gradients(random_model(kv.second, 44)); });
    for (const auto& kv : gpu_block_configs())
        run("attribution + mixed patching on the GPU = CPU (macOS): " + kv.first,
            [&] { test_gpu_attribution(random_model(kv.second, 45)); });
#endif
    run("NaN never matches", test_nan_never_matches);
    run("patch gap threshold", test_patch_gap_threshold);
    run("per-layer head counts", test_heterogeneous_heads);
    run("attribution patching exact when linear", test_attribution_exact_when_linear);
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
