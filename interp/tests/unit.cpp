// Self-contained unit tests for tfi: invariants of the analysis tools on random models.
// The forward pass itself is checked against PyTorch by tests/test_cpp.py (golden tests).
#include <algorithm>
#include <cmath>
#include <map>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>

#include "tfi/analysis.h"
#include "tfi/json.h"
#include "tfi/model.h"

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

// Random weights with exactly the tensor names and shapes tforge's state_dict uses.
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
}

static void test_ablation_and_scores(const Model& m) {
    const Tensor abl = head_ablation(m, seq(12, 7), Ablation::Mean);
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
    check_layout(head_ablation(m, t, Ablation::Mean));
    check_layout(patch_heads(m, t, corrupt, 9, 1, 2).values);
    check_layout(ov_copying_scores(m, 32));

    const Composition comp = composition_scores(m);
    CHECK(comp.labels.size() == 14);
    CHECK(comp.labels[13] == "L2H7");
    CHECK(std::isnan(comp.k.at({0, 1})));  // same layer
    // L0H1 -> L2H3 (indices 1 and 9) matches the direct d x d computation despite different head sizes.
    const HeadWeights w = head_weights(m, 0, 1), r = head_weights(m, 2, 3);
    const Tensor ov = ov_matrix(w), qk = qk_matrix(r);
    CHECK_NEAR(comp.k.at({1, 9}), frobenius(matmul(qk, ov)) / (frobenius(qk) * frobenius(ov)), 1e-4);

    const DLAResult dla = direct_logit_attribution(m, t, 9, 1, 2);
    CHECK_NEAR(dla.sum, dla.actual, 1e-3 * std::max(1.f, std::fabs(dla.actual)));
    CHECK(std::find(dla.labels.begin(), dla.labels.end(), "L2H7") != dla.labels.end());
    CHECK(std::find(dla.labels.begin(), dla.labels.end(), "L0H2") == dla.labels.end());
}

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
    run("matmul", test_matmul);
    for (const auto& kv : configs()) {
        const Model m = random_model(kv.second, 42);
        run("cache/hooks/causality: " + kv.first, [&] { test_cache_and_hooks(kv.first, m); });
        run("logit lens: " + kv.first, [&] { test_logit_lens(m); });
        run("direct logit attribution: " + kv.first, [&] { test_dla(m); });
        run("activation patching: " + kv.first, [&] { test_patching(m); });
        run("ablation + head scores: " + kv.first, [&] { test_ablation_and_scores(m); });
        if (kv.second.kv_heads() == kv.second.n_heads)
            run("composition = direct: " + kv.first, [&] { test_composition_matches_direct(m); });
    }
    run("OV copying + positional QK", test_copying_and_positional);
    run("post-norm DLA refused", test_post_norm_dla_refused);
    run("NaN never matches", test_nan_never_matches);
    run("per-layer head counts", test_heterogeneous_heads);
    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
