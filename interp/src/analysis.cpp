#include "tfi/analysis.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

namespace tfi {

namespace {

std::atomic<int> g_threads{0};  // 0 = TFI_THREADS, else every core

// Run fn(0..n-1) on up to num_threads() threads. Jobs are independent; the first exception any job
// throws is re-thrown here after the workers stop.
template <typename F>
void parallel_for(int64_t n, F&& fn) {
    const int64_t workers = std::min<int64_t>(num_threads(), n);
    if (workers <= 1) {
        for (int64_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto work = [&] {
        for (int64_t i; (i = next.fetch_add(1)) < n;) {
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
                next = n;  // stop handing out jobs
            }
        }
    };
    std::vector<std::thread> threads;
    for (int64_t w = 1; w < workers; ++w) threads.emplace_back(work);
    work();
    for (auto& t : threads) t.join();
    if (error) std::rethrow_exception(error);
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();

std::string head_label(int64_t layer, int64_t head) { return "L" + std::to_string(layer) + "H" + std::to_string(head); }

// (n_layers, widest layer's head count), `fill` where a head exists and NaN where a layer has
// fewer heads (layer_overrides may change n_heads per layer).
Tensor per_head_matrix(const Model& m, float fill) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    Tensor t({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) t.data[i * H + h] = fill;
    return t;
}
std::string lp(int64_t layer) { return "blocks." + std::to_string(layer) + "."; }

// A cache filter accepting exactly these hook names, so a tool copies only what it reads.
NameFilter only(std::set<std::string> names) {
    return [names = std::move(names)](const std::string& n) { return names.count(n) > 0; };
}

void check_pos(const Tokens& tokens, int64_t pos) {
    if (pos < 0 || pos >= static_cast<int64_t>(tokens.size()))
        throw std::out_of_range("position " + std::to_string(pos) + " is outside the sequence");
}

void check_token(const Model& m, int64_t t, const char* what) {
    if (t < 0 || t >= m.cfg().vocab_size) throw std::out_of_range(std::string(what) + " token is outside the vocabulary");
}

// Unembedding direction for the metric: W_U[target] - W_U[other] (or W_U[target]).
std::vector<float> logit_direction(const Model& m, int64_t target, int64_t other) {
    const Tensor& U = m.unembed();
    const int64_t d = m.cfg().d_model;
    std::vector<float> u(U.ptr() + target * d, U.ptr() + (target + 1) * d);
    if (other >= 0)
        for (int64_t j = 0; j < d; ++j) u[j] -= U.data[other * d + j];
    return u;
}

// tr(A) for a square matrix.
float trace(const Tensor& A) {
    double s = 0;
    for (int64_t i = 0; i < A.shape[0]; ++i) s += A.data[i * A.shape[1] + i];
    return static_cast<float>(s);
}

// || L^T C R ||_F for L, R of shape (r, d) and C (r, r), computed through r x r Gram matrices:
// tr(C^T (L L^T) C (R R^T)).
float frob_lcr(const Tensor& GL, const Tensor& C, const Tensor& GR) {
    const Tensor M = matmul(matmul(transpose2d(C), GL), matmul(C, GR));
    return std::sqrt(std::max(0.f, trace(M)));
}

// Composition of writer (L0 index i) into reader j, given each head's weights and Gram matrices.
struct HeadCache {
    HeadWeights w;
    Tensor GQ, GK, GV, GOt;  // W_Q W_Q^T, W_K W_K^T, W_V W_V^T, W_O^T W_O, all (r, r)
    float qk_norm, ov_norm;  // ||W_Q^T W_K||_F, ||W_O W_V||_F
};

HeadCache cache_head(HeadWeights w) {
    HeadCache c;
    c.GQ = matmul(w.W_Q, w.W_Q, true);
    c.GK = matmul(w.W_K, w.W_K, true);
    c.GV = matmul(w.W_V, w.W_V, true);
    c.GOt = matmul(transpose2d(w.W_O), w.W_O);
    c.qk_norm = std::sqrt(std::max(0.f, trace(matmul(c.GQ, c.GK))));
    c.ov_norm = std::sqrt(std::max(0.f, trace(matmul(c.GOt, c.GV))));
    c.w = std::move(w);
    return c;
}

struct CompScores {
    float q, k, v;
};

CompScores compose(const HeadCache& writer, const HeadCache& reader) {
    // Q: W_QK^T W_OV = W_K^T (W_Q W_O) W_V;  K: W_QK W_OV = W_Q^T (W_K W_O) W_V;
    // V: W_OV' W_OV = W_O' (W_V' W_O) W_V, with W_O' = (W_O'^T)^T.
    const Tensor Cq = matmul(reader.w.W_Q, writer.w.W_O);
    const Tensor Ck = matmul(reader.w.W_K, writer.w.W_O);
    const Tensor Cv = matmul(reader.w.W_V, writer.w.W_O);
    const float denom_qk = reader.qk_norm * writer.ov_norm, denom_ov = reader.ov_norm * writer.ov_norm;
    return {frob_lcr(reader.GK, Cq, writer.GV) / denom_qk, frob_lcr(reader.GQ, Ck, writer.GV) / denom_qk,
            frob_lcr(reader.GOt, Cv, writer.GV) / denom_ov};
}

}  // namespace

void set_num_threads(int n) { g_threads = n < 0 ? 0 : n; }

int num_threads() {
    if (const int n = g_threads.load(); n > 0) return n;
    if (const char* env = std::getenv("TFI_THREADS")) {
        const int n = std::atoi(env);
        if (n > 0) return n;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw ? static_cast<int>(hw) : 1;
}

float logit_metric(const Tensor& logits, int64_t pos, int64_t target, int64_t other) {
    const int64_t V = logits.shape[1];
    const float* l = logits.ptr() + pos * V;
    return other >= 0 ? l[target] - l[other] : log_softmax_at(l, V, target);
}

float next_token_loss(const Tensor& logits, const Tokens& tokens) {
    const int64_t T = static_cast<int64_t>(tokens.size()), V = logits.shape[1];
    if (T < 2) throw std::invalid_argument("loss needs at least two tokens");
    double s = 0;
    for (int64_t t = 0; t + 1 < T; ++t) s -= log_softmax_at(logits.ptr() + t * V, V, tokens[t + 1]);
    return static_cast<float>(s / (T - 1));
}

// --- logit lens ---------------------------------------------------------------------

std::vector<LensEntry> logit_lens(const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t k) {
    check_pos(tokens, pos);
    if (target >= 0) check_token(m, target, "target");
    std::vector<std::pair<std::string, std::string>> stages = {{"embed", "blocks.0.hook_resid_pre"}};
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) {
        const std::string L = "L" + std::to_string(i);
        if (m.layer_cfg(i).has_mlp()) {
            stages.push_back({L + ".attn", lp(i) + "hook_resid_mid"});
            stages.push_back({L + ".mlp", lp(i) + "hook_resid_post"});
        } else {
            stages.push_back({L + ".attn", lp(i) + "hook_resid_post"});
        }
    }
    // Only the stages decoded: later layers' resid_pre would just duplicate the previous resid_post.
    std::set<std::string> needed;
    for (const auto& st : stages) needed.insert(st.second);
    ActivationCache cache;
    m.run_with_cache(tokens, cache, only(std::move(needed)));
    std::vector<LensEntry> out;
    const int64_t V = m.cfg().vocab_size;
    for (const auto& st : stages) {
        Tensor r = rows(cache.at(st.second), pos, pos + 1);
        Tensor probs = m.logits_from_final(m.final_norm(r));
        softmax_(probs.ptr(), V);
        LensEntry e;
        e.stage = st.first;
        e.top_tokens = topk(probs.ptr(), V, k);
        for (int64_t t : e.top_tokens) e.top_probs.push_back(probs.data[t]);
        if (target >= 0) {
            e.target_prob = probs.data[target];
            e.target_rank = rank_of(probs.ptr(), V, target);
        }
        out.push_back(std::move(e));
    }
    return out;
}

// --- direct logit attribution --------------------------------------------------------

DLAResult direct_logit_attribution(const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t other) {
    const ModelConfig& cfg = m.cfg();
    if (cfg.norm_position == "post")
        throw std::runtime_error("direct logit attribution needs an additive residual stream; post-norm models have none");
    for (int64_t i = 0; i < cfg.n_layers; ++i)  // a single post-norm layer (via layer_overrides) breaks it too
        if (m.layer_cfg(i).norm_position == "post")
            throw std::runtime_error("direct logit attribution needs an additive residual stream; layer " +
                                     std::to_string(i) + " is post-norm");
    check_pos(tokens, pos);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");

    const int64_t d = cfg.d_model, L = cfg.n_layers;
    std::set<std::string> needed = {"hook_embed", "hook_pos_embed", "hook_final", lp(L - 1) + "hook_resid_post"};
    for (int64_t i = 0; i < L; ++i) {
        const ModelConfig& c = m.layer_cfg(i);
        needed.insert(lp(i) + (c.norm_position == "sandwich" ? "hook_attn_out" : "attn.hook_z"));
        if (c.has_mlp()) needed.insert(lp(i) + "hook_mlp_out");
    }
    ActivationCache cache;
    m.run_with_cache(tokens, cache, only(std::move(needed)));
    const std::vector<float> u = logit_direction(m, target, other);

    // Final norm as a fixed linear map: f(c) = gamma * (c - mean(c)) / sigma for LayerNorm,
    // gamma * c / rms for RMSNorm, with sigma / rms taken from the real final residual.
    const Tensor resid = row(cache.at(lp(L - 1) + "hook_resid_post"), pos);
    std::vector<float> w(u);  // u * gamma / scale
    bool center = false;
    float bias_term = 0;
    if (cfg.norm != "none") {
        const Tensor& gamma = m.weight("ln_final.weight");
        double mean = 0, sq = 0;
        for (float v : resid.data) mean += v;
        mean /= d;
        if (cfg.norm == "layernorm") {
            for (float v : resid.data) sq += (v - mean) * (v - mean);
            center = true;
        } else {
            for (float v : resid.data) sq += double(v) * v;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sq / d) + static_cast<float>(cfg.norm_eps));
        for (int64_t j = 0; j < d; ++j) w[j] = u[j] * gamma.data[j] * inv;
        if (const Tensor* beta = m.weight_or_null("ln_final.bias")) bias_term = dot(u.data(), beta->ptr(), d);
    }
    auto contribution = [&](const float* c) {
        float mean = 0;
        if (center) {
            for (int64_t j = 0; j < d; ++j) mean += c[j];
            mean /= d;
        }
        float s = 0;
        for (int64_t j = 0; j < d; ++j) s += w[j] * (c[j] - mean);
        return s;
    };

    DLAResult res;
    auto push = [&](const std::string& label, float v) {
        res.labels.push_back(label);
        res.values.push_back(v);
        res.sum += v;
    };
    push("embed", contribution(cache.at("hook_embed").ptr() + pos * d));
    if (cache.count("hook_pos_embed")) push("pos_embed", contribution(cache.at("hook_pos_embed").ptr() + pos * d));
    for (int64_t i = 0; i < L; ++i) {
        const ModelConfig& c = m.layer_cfg(i);
        if (c.norm_position == "sandwich") {  // per-head outputs pass through a norm; attribute the layer
            push("L" + std::to_string(i) + ".attn", contribution(cache.at(lp(i) + "hook_attn_out").ptr() + pos * d));
        } else {
            const Tensor& z = cache.at(lp(i) + "attn.hook_z");  // (T, H, D)
            const Tensor& Wo = m.weight(lp(i) + "attn.o_proj.weight");  // (d, H*D)
            const int64_t H = c.n_heads, D = c.head_dim();
            std::vector<float> out(d);
            for (int64_t h = 0; h < H; ++h) {
                const float* zh = z.ptr() + (pos * H + h) * D;
                for (int64_t j = 0; j < d; ++j) out[j] = dot(Wo.ptr() + j * H * D + h * D, zh, D);
                push(head_label(i, h), contribution(out.data()));
            }
            if (const Tensor* bo = m.weight_or_null(lp(i) + "attn.o_proj.bias"))
                push("L" + std::to_string(i) + ".attn_bias", contribution(bo->ptr()));
        }
        if (c.has_mlp()) push("L" + std::to_string(i) + ".mlp", contribution(cache.at(lp(i) + "hook_mlp_out").ptr() + pos * d));
    }
    if (bias_term != 0) push("final_norm_bias", bias_term);
    res.actual = dot(u.data(), cache.at("hook_final").ptr() + pos * d, d);
    return res;
}

// --- head pattern scores --------------------------------------------------------------

HeadScores head_scores(const Model& m, int64_t seg, int64_t n_samples, uint64_t seed) {
    const ModelConfig& cfg = m.cfg();
    if (cfg.positional == "learned") seg = std::min(seg, cfg.max_seq_len / 2);
    if (seg < 2) throw std::invalid_argument("segment length must be at least 2");
    const int64_t L = cfg.n_layers, H = m.max_heads(), T = 2 * seg;  // H: row stride; layer i has heads(i)
    auto heads = [&m](int64_t layer) { return m.layer_cfg(layer).n_heads; };
    HeadScores s;
    for (Tensor* t : {&s.previous_token, &s.induction, &s.duplicate_token, &s.first_token, &s.self})
        *t = per_head_matrix(m, 0.f);
    std::vector<float> two_hop(static_cast<size_t>(L * H * L * H), 0.f);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> tok(0, cfg.vocab_size - 1);
    for (int64_t n = 0; n < n_samples; ++n) {
        Tokens tokens(T);
        for (int64_t t = 0; t < seg; ++t) tokens[t] = tokens[t + seg] = tok(rng);
        ActivationCache cache;
        m.run_with_cache(tokens, cache, [](const std::string& name) { return name.find("hook_pattern") != std::string::npos; });
        std::vector<const Tensor*> P(L);
        for (int64_t i = 0; i < L; ++i) P[i] = &cache.at(lp(i) + "attn.hook_pattern");
        for (int64_t i = 0; i < L; ++i)
            for (int64_t h = 0; h < heads(i); ++h) {
                const float* p = P[i]->ptr() + h * T * T;
                double prev = 0, ind = 0, dup = 0, first = 0, self = 0;
                for (int64_t t = seg; t < T; ++t) {
                    prev += p[t * T + t - 1];
                    ind += p[t * T + t - seg + 1];
                    dup += p[t * T + t - seg];
                    first += p[t * T];
                    self += p[t * T + t];
                }
                const double norm = double(seg) * n_samples;
                s.previous_token.data[i * H + h] += float(prev / norm);
                s.induction.data[i * H + h] += float(ind / norm);
                s.duplicate_token.data[i * H + h] += float(dup / norm);
                s.first_token.data[i * H + h] += float(first / norm);
                s.self.data[i * H + h] += float(self / norm);
            }
        // Two-hop virtual attention reader(t -> r) * writer(r -> t-seg+1), summed over r.
        for (int64_t a = 0; a < L; ++a)
            for (int64_t b = a + 1; b < L; ++b)
                for (int64_t ha = 0; ha < heads(a); ++ha)
                    for (int64_t hb = 0; hb < heads(b); ++hb) {
                        const float* pa = P[a]->ptr() + ha * T * T;
                        const float* pb = P[b]->ptr() + hb * T * T;
                        double acc = 0;
                        for (int64_t t = seg; t < T; ++t) {
                            const int64_t src = t - seg + 1;
                            for (int64_t r = src; r <= t; ++r) acc += double(pb[t * T + r]) * pa[r * T + src];
                        }
                        two_hop[((a * H + ha) * L + b) * H + hb] += float(acc / (double(seg) * n_samples));
                    }
    }
    for (int64_t a = 0; a < L; ++a)
        for (int64_t b = a + 1; b < L; ++b)
            for (int64_t ha = 0; ha < heads(a); ++ha)
                for (int64_t hb = 0; hb < heads(b); ++hb)
                    s.top_two_hop.push_back({head_label(a, ha), head_label(b, hb), two_hop[((a * H + ha) * L + b) * H + hb]});
    std::sort(s.top_two_hop.begin(), s.top_two_hop.end(), [](const TwoHop& x, const TwoHop& y) { return x.score > y.score; });
    if (s.top_two_hop.size() > 5) s.top_two_hop.resize(5);
    return s;
}

// --- activation patching ----------------------------------------------------------------

bool PatchResult::normalized() const {
    const float scale = std::max({1.f, std::fabs(clean_metric), std::fabs(corrupt_metric)});
    return std::isfinite(gap()) && std::fabs(gap()) > 1e-5f * scale;
}

// Fill `values` from `effect`, or leave it all NaN when the gap is (numerically) zero.
static void normalize_patch(PatchResult& r) {
    r.values = Tensor(r.effect.shape, kNaN);
    if (!r.normalized()) return;
    for (int64_t i = 0; i < r.effect.numel(); ++i) r.values.data[i] = r.effect.data[i] / r.gap();
}

static void check_patch_inputs(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                               int64_t other) {
    if (clean.size() != corrupt.size()) throw std::invalid_argument("clean and corrupted inputs must have the same length");
    check_pos(clean, pos);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");
}

// Patching runs one forward pass per site, made cheap in three ways that don't change any result:
// - the metric at `pos` depends only on tokens 0..pos, so both inputs are cut to that prefix and
//   patches at later positions (which cannot reach `pos`) are not run: their effect is exactly 0;
// - a patch at layer i leaves layers 0..i-1 as in the corrupted run, so each pass starts at layer i
//   from the cached corrupted residual stream (Model::forward_from);
// - the passes are independent and run in parallel (set_num_threads).
namespace {

struct PatchRuns {
    Tokens clean, corrupt;                 // cut to positions 0..pos
    ActivationCache clean_cache, corrupt_cache;
    PatchResult result;
};

PatchRuns prepare_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other, std::set<std::string> clean_hooks) {
    PatchRuns p;
    p.clean.assign(clean.begin(), clean.begin() + pos + 1);
    p.corrupt.assign(corrupt.begin(), corrupt.begin() + pos + 1);
    std::set<std::string> starts;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) starts.insert(lp(i) + "hook_resid_pre");
    p.result.clean_metric = logit_metric(m.run_with_cache(p.clean, p.clean_cache, only(std::move(clean_hooks))),
                                         pos, target, other);
    p.result.corrupt_metric = logit_metric(m.run_with_cache(p.corrupt, p.corrupt_cache, only(std::move(starts))),
                                           pos, target, other);
    return p;
}

}  // namespace

namespace {

using Site = std::pair<int64_t, int64_t>;  // (layer, position) or (layer, head)

// Exact patching of every site, or with `selected` only those sites: the others are left NaN
// (except positions after `pos`, whose effect is exactly 0 without running anything).
PatchResult exact_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                                 int64_t pos, int64_t target, int64_t other, const std::vector<Site>* selected) {
    static const std::vector<std::string> sites = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"};
    if (std::find(sites.begin(), sites.end(), site) == sites.end())
        throw std::invalid_argument("unknown patch site '" + site + "'");
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    const int64_t L = m.cfg().n_layers, T = static_cast<int64_t>(clean.size()), d = m.cfg().d_model;

    std::set<std::string> needed;
    for (int64_t i = 0; i < L; ++i) needed.insert(lp(i) + "hook_" + site);
    PatchRuns p = prepare_patch(m, clean, corrupt, pos, target, other, std::move(needed));
    PatchResult& r = p.result;
    r.effect = Tensor({L, T}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t t = 0; t < T; ++t) r.col_labels.push_back(std::to_string(t));

    std::vector<const Tensor*> src(L, nullptr), start(L);
    std::vector<std::pair<int64_t, int64_t>> jobs;  // (layer, position)
    for (int64_t i = 0; i < L; ++i) {
        start[i] = &p.corrupt_cache.at(lp(i) + "hook_resid_pre");
        auto it = p.clean_cache.find(lp(i) + "hook_" + site);
        if (it == p.clean_cache.end()) continue;  // e.g. mlp_out on an attention-only layer: stays NaN
        src[i] = &it->second;
        for (int64_t t = pos + 1; t < T; ++t) r.effect.data[i * T + t] = 0.f;  // cannot reach `pos`
        if (!selected)
            for (int64_t t = 0; t <= pos; ++t) jobs.emplace_back(i, t);
    }
    if (selected)
        for (const Site& s : *selected)
            if (s.first >= 0 && s.first < L && src[s.first] && s.second >= 0 && s.second <= pos) jobs.push_back(s);
    parallel_for(static_cast<int64_t>(jobs.size()), [&](int64_t j) {
        const int64_t i = jobs[j].first, t = jobs[j].second;
        const Tensor* s = src[i];
        Hooks hooks;
        hooks.add(lp(i) + "hook_" + site, [s, t, d](const std::string&, Tensor& act) {
            std::copy(s->ptr() + t * d, s->ptr() + (t + 1) * d, act.ptr() + t * d);
        });
        const float metric = logit_metric(m.forward_from(i, *start[i], &hooks), pos, target, other);
        r.effect.data[i * T + t] = metric - r.corrupt_metric;
    });
    normalize_patch(r);
    return r;
}

PatchResult exact_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                              int64_t other, const std::vector<Site>* selected) {
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    std::set<std::string> needed;
    for (int64_t i = 0; i < L; ++i) needed.insert(lp(i) + "attn.hook_z");
    PatchRuns p = prepare_patch(m, clean, corrupt, pos, target, other, std::move(needed));
    PatchResult& r = p.result;
    r.effect = Tensor({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t h = 0; h < H; ++h) r.col_labels.push_back("H" + std::to_string(h));

    std::vector<const Tensor*> src(L), start(L);
    std::vector<std::pair<int64_t, int64_t>> jobs;  // (layer, head)
    for (int64_t i = 0; i < L; ++i) {
        src[i] = &p.clean_cache.at(lp(i) + "attn.hook_z");  // (pos+1, H, D)
        start[i] = &p.corrupt_cache.at(lp(i) + "hook_resid_pre");
        if (!selected)
            for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) jobs.emplace_back(i, h);
    }
    if (selected)
        for (const Site& s : *selected)
            if (s.first >= 0 && s.first < L && s.second >= 0 && s.second < m.layer_cfg(s.first).n_heads) jobs.push_back(s);
    parallel_for(static_cast<int64_t>(jobs.size()), [&](int64_t j) {
        const int64_t i = jobs[j].first, h = jobs[j].second;
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        const Tensor* s = src[i];
        Hooks hooks;
        hooks.add(lp(i) + "attn.hook_z", [s, h, Hl, D](const std::string&, Tensor& act) {
            for (int64_t t = 0; t < act.shape[0]; ++t)
                std::copy(s->ptr() + (t * Hl + h) * D, s->ptr() + (t * Hl + h + 1) * D, act.ptr() + (t * Hl + h) * D);
        });
        const float metric = logit_metric(m.forward_from(i, *start[i], &hooks), pos, target, other);
        r.effect.data[i * H + h] = metric - r.corrupt_metric;
    });
    normalize_patch(r);
    return r;
}

}  // namespace

PatchResult patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                           int64_t pos, int64_t target, int64_t other) {
    return exact_patch_residual(m, clean, corrupt, site, pos, target, other, nullptr);
}

PatchResult patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other) {
    return exact_patch_heads(m, clean, corrupt, pos, target, other, nullptr);
}

// --- attribution patching ----------------------------------------------------------------

namespace {

// dF/dlogits for F = logit_metric(pos, target, other).
Tensor metric_dlogits(const Tensor& logits, int64_t pos, int64_t target, int64_t other) {
    const int64_t V = logits.shape[1];
    Tensor dl(logits.shape, 0.f);
    float* row = dl.ptr() + pos * V;
    if (other >= 0) {
        row[target] += 1.f;
        row[other] -= 1.f;
    } else {  // d log_softmax(l)[target] / dl = onehot(target) - softmax(l)
        std::copy(logits.ptr() + pos * V, logits.ptr() + (pos + 1) * V, row);
        softmax_(row, V);
        for (int64_t j = 0; j < V; ++j) row[j] = -row[j];
        row[target] += 1.f;
    }
    return dl;
}

}  // namespace

float metric_gradients(const Model& m, const Tokens& tokens, int64_t pos, int64_t target, int64_t other,
                       ActivationCache& grads, NameFilter filter, ActivationCache* acts) {
    check_pos(tokens, pos);
    check_token(m, target, "target");
    if (other >= 0) check_token(m, other, "other");
    float metric = 0;
    m.run_with_gradients(
        tokens,
        [&](const Tensor& logits) {
            metric = logit_metric(logits, pos, target, other);
            return metric_dlogits(logits, pos, target, other);
        },
        grads, std::move(filter), acts);
    return metric;
}

namespace {

// Clean activations, corrupted activations and the metric's gradient at `names`, on the inputs cut
// to positions 0..pos (later positions cannot affect the metric). With steps = 1 the gradient is
// the corrupted run's; with more, it is averaged over runs whose token embeddings are interpolated
// from corrupted to clean, at alpha = 0, 1/steps, ..., (steps-1)/steps.
struct AttributionRuns {
    ActivationCache clean, corrupt, grads;
    PatchResult result;
};

AttributionRuns prepare_attribution(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other, std::set<std::string> names, int64_t steps) {
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    if (steps < 1) throw std::invalid_argument("attribution patching needs at least one step");
    AttributionRuns a;
    const Tokens c(clean.begin(), clean.begin() + pos + 1), x(corrupt.begin(), corrupt.begin() + pos + 1);
    std::set<std::string> with_embed = names;
    with_embed.insert("hook_embed");
    a.result.clean_metric = logit_metric(m.run_with_cache(c, a.clean, only(with_embed)), pos, target, other);
    ActivationCache all;
    a.result.corrupt_metric = metric_gradients(m, x, pos, target, other, a.grads, only(names), &all);
    for (const auto& n : with_embed)
        if (all.count(n)) a.corrupt[n] = std::move(all[n]);
    if (steps == 1) return a;

    const Tensor &e_clean = a.clean.at("hook_embed"), &e_corrupt = a.corrupt.at("hook_embed");
    std::vector<ActivationCache> step_grads(steps - 1);
    parallel_for(steps - 1, [&](int64_t k) {
        const float alpha = static_cast<float>(k + 1) / static_cast<float>(steps);
        Hooks hooks;
        hooks.add("hook_embed", [&, alpha](const std::string&, Tensor& e) {
            for (int64_t i = 0; i < e.numel(); ++i)
                e.data[i] = e_corrupt.data[i] + alpha * (e_clean.data[i] - e_corrupt.data[i]);
        });
        m.run_with_gradients(
            x, [&](const Tensor& logits) { return metric_dlogits(logits, pos, target, other); }, step_grads[k],
            only(names), nullptr, &hooks);
    });
    for (auto& kv : a.grads) {  // average, summing in step order: deterministic
        for (const ActivationCache& g : step_grads) add_(kv.second, g.at(kv.first));
        kv.second = scale(kv.second, 1.f / static_cast<float>(steps));
    }
    return a;
}

}  // namespace

PatchResult attribution_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt,
                                       const std::string& site, int64_t pos, int64_t target, int64_t other,
                                       int64_t steps) {
    static const std::vector<std::string> sites = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"};
    if (std::find(sites.begin(), sites.end(), site) == sites.end())
        throw std::invalid_argument("unknown patch site '" + site + "'");
    const int64_t L = m.cfg().n_layers, T = static_cast<int64_t>(clean.size()), d = m.cfg().d_model;
    std::set<std::string> names;
    for (int64_t i = 0; i < L; ++i) names.insert(lp(i) + "hook_" + site);
    AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, names, steps);
    PatchResult& r = a.result;
    r.effect = Tensor({L, T}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t t = 0; t < T; ++t) r.col_labels.push_back(std::to_string(t));
    for (int64_t i = 0; i < L; ++i) {
        const std::string n = lp(i) + "hook_" + site;
        if (!a.clean.count(n)) continue;  // site doesn't exist in this layer: stays NaN
        const Tensor &cl = a.clean.at(n), &co = a.corrupt.at(n), &g = a.grads.at(n);
        for (int64_t t = 0; t < T; ++t) {
            double s = 0;
            if (t <= pos)
                for (int64_t j = 0; j < d; ++j) s += double(cl.data[t * d + j] - co.data[t * d + j]) * g.data[t * d + j];
            r.effect.data[i * T + t] = static_cast<float>(s);
        }
    }
    normalize_patch(r);
    return r;
}

PatchResult attribution_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other, int64_t steps) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    std::set<std::string> names;
    for (int64_t i = 0; i < L; ++i) names.insert(lp(i) + "attn.hook_z");
    AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, names, steps);
    PatchResult& r = a.result;
    r.effect = Tensor({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t h = 0; h < H; ++h) r.col_labels.push_back("H" + std::to_string(h));
    for (int64_t i = 0; i < L; ++i) {
        const std::string n = lp(i) + "attn.hook_z";  // (pos+1, H, D)
        const Tensor &cl = a.clean.at(n), &co = a.corrupt.at(n), &g = a.grads.at(n);
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim(), T = cl.shape[0];
        for (int64_t h = 0; h < Hl; ++h) {
            double s = 0;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t j = 0; j < D; ++j) {
                    const int64_t idx = (t * Hl + h) * D + j;
                    s += double(cl.data[idx] - co.data[idx]) * g.data[idx];
                }
            r.effect.data[i * H + h] = static_cast<float>(s);
        }
    }
    normalize_patch(r);
    return r;
}

// --- mixed patching ---------------------------------------------------------------------------

namespace {

// Rows x cols site of each of the `verify` largest finite |estimates|, ties broken by position in the
// table so the choice is deterministic. `usable(row, col)` excludes sites that need no measurement.
template <typename Usable>
std::vector<Site> top_sites(const Tensor& estimate, int64_t verify, Usable usable) {
    if (verify < 0) throw std::invalid_argument("mixed patching: verify must be >= 0");
    const int64_t cols = estimate.shape[1];
    std::vector<int64_t> idx;
    for (int64_t i = 0; i < estimate.numel(); ++i)
        if (!std::isnan(estimate.data[i]) && usable(i / cols, i % cols)) idx.push_back(i);
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        return std::fabs(estimate.data[a]) > std::fabs(estimate.data[b]);
    });
    if (static_cast<int64_t>(idx.size()) > verify) idx.resize(static_cast<size_t>(verify));
    std::vector<Site> sites;
    for (int64_t i : idx) sites.emplace_back(i / cols, i % cols);
    return sites;
}

// Start from the estimates and overwrite the measured sites.
PatchResult merge_mixed(PatchResult atp, const PatchResult& exact, const std::vector<Site>& sites) {
    const int64_t cols = atp.effect.shape[1];
    atp.estimate = atp.effect;
    atp.exact.assign(static_cast<size_t>(atp.effect.numel()), 0);
    for (const Site& s : sites) {
        const int64_t i = s.first * cols + s.second;
        atp.effect.data[i] = exact.effect.data[i];
        atp.exact[i] = 1;
    }
    normalize_patch(atp);
    return atp;
}

}  // namespace

PatchResult mixed_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                                 int64_t pos, int64_t target, int64_t other, int64_t verify, int64_t steps) {
    PatchResult atp = attribution_patch_residual(m, clean, corrupt, site, pos, target, other, steps);
    const std::vector<Site> sites = top_sites(atp.effect, verify, [pos](int64_t, int64_t t) { return t <= pos; });
    const PatchResult exact = exact_patch_residual(m, clean, corrupt, site, pos, target, other, &sites);
    return merge_mixed(std::move(atp), exact, sites);
}

PatchResult mixed_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                              int64_t other, int64_t verify, int64_t steps) {
    PatchResult atp = attribution_patch_heads(m, clean, corrupt, pos, target, other, steps);
    const std::vector<Site> sites = top_sites(atp.effect, verify, [](int64_t, int64_t) { return true; });
    const PatchResult exact = exact_patch_heads(m, clean, corrupt, pos, target, other, &sites);
    return merge_mixed(std::move(atp), exact, sites);
}

// --- ablation -----------------------------------------------------------------------------

HeadMeans head_means(const Model& m, const std::vector<Tokens>& reference, int64_t T) {
    if (reference.empty()) throw std::invalid_argument("mean ablation needs at least one reference sequence");
    const int64_t L = m.cfg().n_layers;
    const bool per_position = std::all_of(reference.begin(), reference.end(),
                                          [T](const Tokens& r) { return static_cast<int64_t>(r.size()) == T; });
    HeadMeans means;
    std::vector<double> counts(L, 0.0);
    for (int64_t i = 0; i < L; ++i)
        means.layers.emplace_back(Shape{per_position ? T : 1, m.layer_cfg(i).n_heads, m.layer_cfg(i).head_dim()});
    std::vector<ActivationCache> caches(reference.size());
    parallel_for(static_cast<int64_t>(reference.size()), [&](int64_t n) {
        m.run_with_cache(reference[n], caches[n],
                         [](const std::string& name) { return name.find("attn.hook_z") != std::string::npos; });
    });
    for (const ActivationCache& cache : caches) {  // reduce in reference order: deterministic sums
        for (int64_t i = 0; i < L; ++i) {
            const Tensor& z = cache.at(lp(i) + "attn.hook_z");  // (len, H, D)
            Tensor& acc = means.layers[i];
            const int64_t row = acc.numel() / acc.shape[0];
            for (int64_t t = 0; t < z.shape[0]; ++t) {
                float* dst = acc.ptr() + (per_position ? t : 0) * row;
                for (int64_t j = 0; j < row; ++j) dst[j] += z.data[t * row + j];
            }
            counts[i] += per_position ? 1.0 : static_cast<double>(z.shape[0]);
        }
    }
    for (int64_t i = 0; i < L; ++i)
        for (float& v : means.layers[i].data) v = static_cast<float>(v / counts[i]);
    return means;
}

Hooks ablate_head(const Model& m, int64_t layer, int64_t head, Ablation mode, const HeadMeans* means) {
    if (layer < 0 || layer >= m.cfg().n_layers) throw std::out_of_range("layer index out of range");
    const int64_t Hl = m.layer_cfg(layer).n_heads, D = m.layer_cfg(layer).head_dim();
    if (head < 0 || head >= Hl) throw std::out_of_range("head index out of range");
    const Tensor* mean = nullptr;
    if (mode == Ablation::Mean) {
        if (!means || static_cast<int64_t>(means->layers.size()) != m.cfg().n_layers)
            throw std::invalid_argument("mean ablation needs reference means (see head_means)");
        mean = &means->layers[layer];
    }
    Hooks hooks;
    hooks.add(lp(layer) + "attn.hook_z", [head, Hl, D, mean](const std::string&, Tensor& z) {
        const int64_t T = z.shape[0];
        if (mean && mean->shape[0] != 1 && mean->shape[0] != T)
            throw std::invalid_argument("per-position reference means were computed for length " +
                                        std::to_string(mean->shape[0]) + ", input has length " + std::to_string(T));
        for (int64_t t = 0; t < T; ++t) {
            float* dst = z.ptr() + (t * Hl + head) * D;
            if (!mean) std::fill(dst, dst + D, 0.f);
            else {
                const float* src = mean->ptr() + ((mean->shape[0] == 1 ? 0 : t) * Hl + head) * D;
                std::copy(src, src + D, dst);
            }
        }
    });
    return hooks;
}

Tensor head_ablation(const Model& m, const Tokens& tokens, Ablation mode, const std::vector<Tokens>& reference) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    const HeadMeans means = mode == Ablation::Mean ? head_means(m, reference, static_cast<int64_t>(tokens.size()))
                                                   : HeadMeans{};
    // Ablating a head in layer i leaves layers 0..i-1 untouched: start each pass at layer i from the
    // cached residual stream, and run the passes in parallel.
    std::set<std::string> starts;
    for (int64_t i = 0; i < L; ++i) starts.insert(lp(i) + "hook_resid_pre");
    ActivationCache cache;
    const float base = next_token_loss(m.run_with_cache(tokens, cache, only(std::move(starts))), tokens);
    Tensor out({L, H}, kNaN);
    std::vector<std::pair<int64_t, int64_t>> jobs;
    for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) jobs.emplace_back(i, h);
    std::vector<const Tensor*> start(L);
    for (int64_t i = 0; i < L; ++i) start[i] = &cache.at(lp(i) + "hook_resid_pre");
    parallel_for(static_cast<int64_t>(jobs.size()), [&](int64_t j) {
        const int64_t i = jobs[j].first, h = jobs[j].second;
        const Hooks hooks = ablate_head(m, i, h, mode, mode == Ablation::Mean ? &means : nullptr);
        out.data[i * H + h] = next_token_loss(m.forward_from(i, *start[i], &hooks), tokens) - base;
    });
    return out;
}

// --- weight-based circuits -------------------------------------------------------------------

HeadWeights head_weights(const Model& m, int64_t layer, int64_t head) {
    const ModelConfig& c = m.layer_cfg(layer);
    const int64_t H = c.n_heads, D = c.head_dim(), rep = H / c.kv_heads(), g = head / rep;
    if (head < 0 || head >= H) throw std::out_of_range("head index out of range");
    const std::string p = lp(layer) + "attn.";
    HeadWeights w;
    w.W_Q = rows(m.weight(p + "q_proj.weight"), head * D, (head + 1) * D);
    w.W_K = rows(m.weight(p + "k_proj.weight"), g * D, (g + 1) * D);
    w.W_V = rows(m.weight(p + "v_proj.weight"), g * D, (g + 1) * D);
    w.W_O = cols(m.weight(p + "o_proj.weight"), head * D, (head + 1) * D);
    // The attention input is gamma * normalized(x) for pre/sandwich norm; fold gamma in.
    if (c.norm != "none" && c.norm_position != "post") {
        const Tensor& gamma = m.weight(lp(layer) + "ln1.weight");
        for (Tensor* W : {&w.W_Q, &w.W_K, &w.W_V})
            for (int64_t r = 0; r < D; ++r)
                for (int64_t j = 0; j < c.d_model; ++j) W->data[r * c.d_model + j] *= gamma.data[j];
    }
    return w;
}

Tensor ov_matrix(const HeadWeights& h) { return matmul(h.W_O, h.W_V); }
Tensor qk_matrix(const HeadWeights& h) { return matmul(transpose2d(h.W_Q), h.W_K); }

Composition composition_scores(const Model& m) {
    std::vector<HeadCache> heads;
    std::vector<int64_t> layer_of;
    Composition c;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) {
            heads.push_back(cache_head(head_weights(m, i, h)));
            layer_of.push_back(i);
            c.labels.push_back(head_label(i, h));
        }
    const int64_t N = static_cast<int64_t>(heads.size());
    c.q = Tensor({N, N}, kNaN);
    c.k = Tensor({N, N}, kNaN);
    c.v = Tensor({N, N}, kNaN);
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b) {
            if (layer_of[a] >= layer_of[b]) continue;
            const CompScores s = compose(heads[a], heads[b]);
            c.q.data[a * N + b] = s.q;
            c.k.data[a * N + b] = s.k;
            c.v.data[a * N + b] = s.v;
        }
    return c;
}

float composition_random_baseline(const Model& m, int64_t n_pairs, uint64_t seed) {
    const int64_t d = m.cfg().d_model, D = m.cfg().head_dim();
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    auto random = [&](Shape s) {
        Tensor t(s);
        for (float& v : t.data) v = normal(rng);
        return t;
    };
    auto random_head = [&]() { return cache_head({random({D, d}), random({D, d}), random({D, d}), random({d, D})}); };
    double total = 0;
    for (int64_t n = 0; n < n_pairs; ++n) {
        const CompScores s = compose(random_head(), random_head());
        total += (s.q + s.k + s.v) / 3.0;
    }
    return static_cast<float>(total / n_pairs);
}

float ov_path_copying(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& path, int64_t max_tokens) {
    const ModelConfig& cfg = m.cfg();
    const int64_t V = cfg.vocab_size, d = cfg.d_model, n = std::min(V, max_tokens);
    if (path.empty()) throw std::invalid_argument("OV path needs at least one head");
    for (size_t i = 1; i < path.size(); ++i)
        if (path[i].first <= path[i - 1].first) throw std::invalid_argument("OV path heads must be in increasing layers");
    Tokens toks(n);
    for (int64_t i = 0; i < n; ++i) toks[i] = i * V / n;

    const Tensor& E = m.weight("embed.weight");
    Tensor X({n, d});
    for (int64_t i = 0; i < n; ++i) std::copy(E.ptr() + toks[i] * d, E.ptr() + (toks[i] + 1) * d, X.ptr() + i * d);
    for (const auto& lh : path) {  // push each token's vector through the head's OV circuit
        const HeadWeights w = head_weights(m, lh.first, lh.second);
        X = matmul(matmul(X, w.W_V, true), w.W_O, true);
    }
    // Fold the final norm's weight into the unembedding.
    Tensor U = m.unembed();
    if (cfg.norm != "none" && cfg.norm_position != "post") {
        const Tensor& gamma = m.weight("ln_final.weight");
        for (int64_t v = 0; v < V; ++v)
            for (int64_t j = 0; j < d; ++j) U.data[v * d + j] *= gamma.data[j];
    }
    int64_t hits = 0;
    const int64_t chunk = 256;
    for (int64_t s = 0; s < n; s += chunk) {
        const int64_t e = std::min(n, s + chunk);
        const Tensor logits = matmul(rows(X, s, e), U, true);  // (chunk, V)
        for (int64_t r = 0; r < e - s; ++r) hits += topk(logits.ptr() + r * V, V, 1)[0] == toks[s + r];
    }
    return float(hits) / float(n);
}

Tensor ov_copying_scores(const Model& m, int64_t max_tokens) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    Tensor out({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h)
            out.data[i * H + h] = ov_path_copying(m, {{i, h}}, max_tokens);
    return out;
}

Tensor positional_qk(const Model& m, int64_t layer, int64_t head, int64_t T) {
    if (m.cfg().positional != "learned") throw std::runtime_error("positional QK needs learned positional embeddings");
    if (T > m.cfg().max_seq_len) throw std::out_of_range("T exceeds max_seq_len");
    const HeadWeights w = head_weights(m, layer, head);
    const Tensor P = rows(m.weight("pos_embed.weight"), 0, T);
    const Tensor S = matmul(matmul(P, w.W_Q, true), matmul(P, w.W_K, true), true);  // (T, T)
    const ModelConfig& c = m.layer_cfg(layer);
    const float scale = static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(c.head_dim())));
    Tensor pattern({T, T});
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t j = 0; j < T; ++j) pattern.data[i * T + j] = j <= i ? S.data[i * T + j] * scale : -INFINITY;
        softmax_(pattern.ptr() + i * T, T);
    }
    return pattern;
}

}  // namespace tfi
