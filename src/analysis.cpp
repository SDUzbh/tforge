#include "tfi/analysis.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

#include "gpu.h"
#include "ops.h"

namespace tfi {

namespace {

std::atomic<int> g_threads{0};  // 0 = TFI_THREADS, else every core
std::atomic<int> g_use_gpu{-1};  // -1 = TFI_NO_GPU decides
std::atomic<bool> g_gpu_fp16{false};

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
    const Weight& U = m.unembed();
    const Tensor ut = U.row(target);
    std::vector<float> u(ut.data);
    if (other >= 0) {
        const Tensor uo = U.row(other);
        for (size_t j = 0; j < u.size(); ++j) u[j] -= uo.data[j];
    }
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

void set_use_gpu(bool use) { g_use_gpu = use ? 1 : 0; }

void set_gpu_fp16(bool fp16) { g_gpu_fp16 = fp16; }
bool gpu_fp16() { return g_gpu_fp16; }

bool use_gpu() {
    if (const int u = g_use_gpu.load(); u >= 0) return u == 1;
    const char* env = std::getenv("TFI_NO_GPU");
    return !env || !*env || std::string(env) == "0";
}

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
        const Tensor& gamma = m.weight("ln_final.weight").f32();
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
        if (const Weight* beta = m.weight_or_null("ln_final.bias")) bias_term = dot(u.data(), beta->f32().ptr(), d);
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
            const Weight& Wo = m.weight(lp(i) + "attn.o_proj.weight");  // (d, H*D)
            const int64_t H = c.n_heads, D = c.head_dim();
            std::vector<float> out(d);
            for (int64_t h = 0; h < H; ++h) {
                const float* zh = z.ptr() + (pos * H + h) * D;
                const Tensor Woh = Wo.cols(h * D, (h + 1) * D);  // (d, D)
                for (int64_t j = 0; j < d; ++j) out[j] = dot(Woh.ptr() + j * D, zh, D);
                push(head_label(i, h), contribution(out.data()));
            }
            if (const Weight* bo = m.weight_or_null(lp(i) + "attn.o_proj.bias"))
                push("L" + std::to_string(i) + ".attn_bias", contribution(bo->f32().ptr()));
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

// Exact patching needs one patched forward pass per site. It is made cheap in ways that change
// results by float rounding at most (and are the same for any thread count):
// - The metric at `pos` depends only on tokens 0..pos, so both inputs are cut to that prefix.
// - Sites that can't matter are not run; their effect is exactly 0: positions after `pos`, patches
//   that change nothing (clean and corrupted activations identical), and positions before `pos`
//   after the last layer's attention (nothing carries them to `pos`).
// - A patch at layer i leaves layers 0..i-1 as in the corrupted run, so its pass starts at layer i
//   from the cached corrupted residual stream.
// - Passes that start at the same layer run together through Model::forward_rows: their rows are
//   stacked, so every linear layer is one large matrix multiply instead of one small one per pass.
//   On macOS large multiplies run faster on the matrix units, and go to the GPU (gpu.h).
// - A residual-stream patch at position t recomputes only positions t..pos: earlier positions are
//   as in the corrupted run at every layer, and attention reads their keys and values from it.
// - The batches run in parallel (set_num_threads). They are fixed by the sites alone.
namespace {

using Site = std::pair<int64_t, int64_t>;  // (layer, position) or (layer, head)

struct PatchRuns {
    ActivationCache clean_cache, corrupt_cache;  // inputs cut to positions 0..pos
    PatchResult result;
};

// The clean and corrupted runs, caching `clean_hooks` and `corrupt_hooks` (plus every layer's input
// for the corrupted run).
PatchRuns prepare_patch(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                        int64_t other, std::set<std::string> clean_hooks, std::set<std::string> corrupt_hooks) {
    PatchRuns p;
    const Tokens c(clean.begin(), clean.begin() + pos + 1), x(corrupt.begin(), corrupt.begin() + pos + 1);
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) corrupt_hooks.insert(lp(i) + "hook_resid_pre");
    p.result.clean_metric = logit_metric(m.run_with_cache(c, p.clean_cache, only(std::move(clean_hooks))), pos, target, other);
    p.result.corrupt_metric =
        logit_metric(m.run_with_cache(x, p.corrupt_cache, only(std::move(corrupt_hooks))), pos, target, other);
    return p;
}

#ifdef TFI_USE_ACCELERATE
// Stacked rows per batch, tuned on an M5 Pro: larger multiplies run faster, but head patching has
// few passes per layer and needs smaller batches to keep every thread busy. Ablation and reference
// means have fewer passes still: 256 rows is the GPU threshold and leaves several batches.
constexpr int64_t kBatchRows = 1024, kHeadBatchRows = 512, kAblationBatchRows = 256;
#else
// The portable matrix multiply is no faster on larger inputs, so one pass per batch keeps every
// thread busy.
constexpr int64_t kBatchRows = 1, kHeadBatchRows = 1, kAblationBatchRows = 1;
#endif
std::atomic<int64_t> g_batch_rows{0}, g_head_batch_rows{0};  // test overrides
int64_t batch_rows() { return g_batch_rows ? g_batch_rows.load() : kBatchRows; }
int64_t head_batch_rows() { return g_head_batch_rows ? g_head_batch_rows.load() : kHeadBatchRows; }
int64_t ablation_batch_rows() { return g_head_batch_rows ? g_head_batch_rows.load() : kAblationBatchRows; }
#ifdef TFI_USE_ACCELERATE
// Attribution patching's gradient runs (steps sequences of P rows), tuned on an M5 Pro: aim for 8
// batches running in parallel, at most 256 rows each. When the GPU is used at all (steps x P >=
// gpu_min_rows()), a batch goes to it from a quarter of gpu_min_rows() rows (default 64): one
// such sequence already runs faster there than on the CPU.
int64_t gradient_batch_rows(int64_t steps, int64_t P) {
    if (g_head_batch_rows) return g_head_batch_rows.load();
    const int64_t min_rows = std::max<int64_t>(1, detail::gpu_min_rows() / 4);
    const int64_t per = std::min(std::max((steps + 7) / 8, (min_rows + P - 1) / P), std::max<int64_t>(1, 256 / P));
    return per * P;
}
#endif

struct Batch {
    int64_t layer;               // where its passes start
    std::vector<int64_t> items;  // positions (residual stream) or heads
    int64_t rows, cost;          // stacked rows; rows x layers run
};

// Each layer's items in batches of at most max_rows rows (rows(item) each), most expensive first
// so the threads finish together.
std::vector<Batch> make_batches(std::vector<std::vector<int64_t>> items_per_layer,
                                const std::function<int64_t(int64_t)>& rows, int64_t max_rows) {
    const int64_t L = static_cast<int64_t>(items_per_layer.size());
    std::vector<Batch> batches;
    for (int64_t i = 0; i < L; ++i) {
        std::sort(items_per_layer[i].begin(), items_per_layer[i].end());  // batches depend only on the set of sites
        Batch b{i, {}, 0, 0};
        auto flush = [&] {
            b.cost = b.rows * (L - i);
            batches.push_back(std::move(b));
            b = Batch{i, {}, 0, 0};
        };
        for (int64_t item : items_per_layer[i]) {
            if (!b.items.empty() && b.rows + rows(item) > max_rows) flush();
            b.items.push_back(item);
            b.rows += rows(item);
        }
        if (!b.items.empty()) flush();
    }
    std::stable_sort(batches.begin(), batches.end(), [](const Batch& a, const Batch& b) { return a.cost > b.cost; });
    return batches;
}

// The model's GPU context (macOS), only when some batch is large enough to use it: creating one takes time.
detail::Gpu* gpu_for(const Model& m, const std::vector<Batch>& batches) {
#ifdef TFI_USE_ACCELERATE
    if (!use_gpu()) return nullptr;
    for (const Batch& b : batches)
        if (b.rows >= detail::gpu_min_rows()) return m.gpu();
#else
    (void)m;
    (void)batches;
#endif
    return nullptr;
}

bool same_floats(const float* a, const float* b, int64_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

// Exact patching of every site, or with `selected` only those sites: the others are left NaN
// (except positions after `pos`, whose effect is exactly 0 without running anything).
PatchResult exact_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                                 int64_t pos, int64_t target, int64_t other, const std::vector<Site>* selected,
                                 const PatchRuns* runs = nullptr) {
    static const std::vector<std::string> sites = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"};
    if (std::find(sites.begin(), sites.end(), site) == sites.end())
        throw std::invalid_argument("unknown patch site '" + site + "'");
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    const int64_t L = m.cfg().n_layers, T = static_cast<int64_t>(clean.size()), d = m.cfg().d_model, P = pos + 1;
    auto hook_name = [&](int64_t i) { return lp(i) + "hook_" + site; };

    std::set<std::string> needed, corrupt_needed;
    for (int64_t i = 0; i < L; ++i) {
        needed.insert(hook_name(i));
        corrupt_needed.insert({hook_name(i), lp(i) + "attn.hook_k", lp(i) + "attn.hook_v"});
    }
    PatchRuns own;
    if (!runs) own = prepare_patch(m, clean, corrupt, pos, target, other, needed, corrupt_needed);
    const PatchRuns& p = runs ? *runs : own;
    PatchResult r = p.result;
    r.effect = Tensor({L, T}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t t = 0; t < T; ++t) r.col_labels.push_back(std::to_string(t));

    // Sites to run, per layer. A layer without the site (e.g. mlp_out on an attention-only layer) stays NaN.
    std::vector<const Tensor*> src(L, nullptr), start(L), prefix_k(L), prefix_v(L);
    std::vector<std::vector<int64_t>> todo(L);
    auto consider = [&](int64_t i, int64_t t) {
        const Tensor& cor = p.corrupt_cache.at(hook_name(i));
        if (same_floats(src[i]->ptr() + t * d, cor.ptr() + t * d, d) || (i == L - 1 && site != "resid_pre" && t < pos))
            r.effect.data[i * T + t] = 0.f;
        else todo[i].push_back(t);
    };
    for (int64_t i = 0; i < L; ++i) {
        start[i] = &p.corrupt_cache.at(lp(i) + "hook_resid_pre");
        prefix_k[i] = &p.corrupt_cache.at(lp(i) + "attn.hook_k");
        prefix_v[i] = &p.corrupt_cache.at(lp(i) + "attn.hook_v");
        auto it = p.clean_cache.find(hook_name(i));
        if (it == p.clean_cache.end()) continue;
        src[i] = &it->second;
        for (int64_t t = P; t < T; ++t) r.effect.data[i * T + t] = 0.f;  // cannot reach `pos`
        if (!selected)
            for (int64_t t = 0; t < P; ++t) consider(i, t);
    }
    if (selected)
        for (const Site& s : *selected)
            if (s.first >= 0 && s.first < L && src[s.first] && s.second >= 0 && s.second < P) consider(s.first, s.second);

    const std::vector<Batch> batches = make_batches(todo, [P](int64_t t) { return P - t; }, batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = gpu_fp16();  // read once: the same for every batch
    parallel_for(static_cast<int64_t>(batches.size()), [&](int64_t j) {
        const Batch& b = batches[j];
        const int64_t i = b.layer;
        detail::RowBatch rb;
        rb.prefix_k = prefix_k;
        rb.prefix_v = prefix_v;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        for (int64_t t : b.items) rb.add(t, P - t);
        Tensor x({rb.rows(), d});  // the corrupted residual stream, positions t..pos of each pass
        for (size_t s = 0; s < b.items.size(); ++s)
            std::copy(start[i]->ptr() + b.items[s] * d, start[i]->ptr() + P * d, x.ptr() + rb.offset[s] * d);
        const Tensor* sp = src[i];
        Hooks hooks;
        hooks.add(hook_name(i), [&](const std::string&, Tensor& act) {  // patch each pass's first row
            for (size_t s = 0; s < b.items.size(); ++s) {
                const int64_t t = b.items[s];
                std::copy(sp->ptr() + t * d, sp->ptr() + (t + 1) * d, act.ptr() + rb.offset[s] * d);
            }
        });
        const Tensor logits = m.forward_rows(i, std::move(x), rb, &hooks);
        for (size_t s = 0; s < b.items.size(); ++s)
            r.effect.data[i * T + b.items[s]] = logit_metric(logits, static_cast<int64_t>(s), target, other) - r.corrupt_metric;
    });
    normalize_patch(r);
    return r;
}

PatchResult exact_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                              int64_t other, const std::vector<Site>* selected, const PatchRuns* runs = nullptr) {
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    const int64_t L = m.cfg().n_layers, H = m.max_heads(), d = m.cfg().d_model, P = pos + 1;
    std::set<std::string> needed;
    for (int64_t i = 0; i < L; ++i) needed.insert(lp(i) + "attn.hook_z");
    PatchRuns own;
    if (!runs) own = prepare_patch(m, clean, corrupt, pos, target, other, needed, needed);
    const PatchRuns& p = runs ? *runs : own;
    PatchResult r = p.result;
    r.effect = Tensor({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t h = 0; h < H; ++h) r.col_labels.push_back("H" + std::to_string(h));

    std::vector<const Tensor*> src(L), start(L);
    std::vector<std::vector<int64_t>> todo(L);
    auto consider = [&](int64_t i, int64_t h) {  // run unless the head's output is the same in both runs
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        const Tensor& cor = p.corrupt_cache.at(lp(i) + "attn.hook_z");
        bool same = true;
        for (int64_t t = 0; t < P && same; ++t)
            same = same_floats(src[i]->ptr() + (t * Hl + h) * D, cor.ptr() + (t * Hl + h) * D, D);
        if (same) r.effect.data[i * H + h] = 0.f;
        else todo[i].push_back(h);
    };
    for (int64_t i = 0; i < L; ++i) {
        src[i] = &p.clean_cache.at(lp(i) + "attn.hook_z");  // (pos+1, H, D)
        start[i] = &p.corrupt_cache.at(lp(i) + "hook_resid_pre");
        if (!selected)
            for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) consider(i, h);
    }
    if (selected)
        for (const Site& s : *selected)
            if (s.first >= 0 && s.first < L && s.second >= 0 && s.second < m.layer_cfg(s.first).n_heads)
                consider(s.first, s.second);

    const std::vector<Batch> batches = make_batches(todo, [P](int64_t) { return P; }, head_batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = gpu_fp16();  // read once: the same for every batch
    parallel_for(static_cast<int64_t>(batches.size()), [&](int64_t j) {
        const Batch& b = batches[j];
        const int64_t i = b.layer, Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        detail::RowBatch rb;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        for (size_t s = 0; s < b.items.size(); ++s) rb.add(0, P);
        Tensor x({rb.rows(), d});
        for (size_t s = 0; s < b.items.size(); ++s) std::copy(start[i]->ptr(), start[i]->ptr() + P * d, x.ptr() + rb.offset[s] * d);
        const Tensor* sp = src[i];
        Hooks hooks;
        hooks.add(lp(i) + "attn.hook_z", [&](const std::string&, Tensor& act) {  // each pass's head, every position
            for (size_t s = 0; s < b.items.size(); ++s) {
                const int64_t h = b.items[s], off = rb.offset[s];
                for (int64_t t = 0; t < P; ++t)
                    std::copy(sp->ptr() + (t * Hl + h) * D, sp->ptr() + (t * Hl + h + 1) * D, act.ptr() + ((off + t) * Hl + h) * D);
            }
        });
        const Tensor logits = m.forward_rows(i, std::move(x), rb, &hooks);
        for (size_t s = 0; s < b.items.size(); ++s)
            r.effect.data[i * H + b.items[s]] = logit_metric(logits, static_cast<int64_t>(s), target, other) - r.corrupt_metric;
    });
    normalize_patch(r);
    return r;
}

}  // namespace

void detail::set_patch_batch_rows(int64_t residual, int64_t heads) {
    g_batch_rows = residual;
    g_head_batch_rows = heads;
}

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
// from corrupted to clean, at alpha = 0, 1/steps, ..., (steps-1)/steps. All the runs are
// independent, so they run in parallel. `clean_extra` and `corrupt_extra` name more activations to
// cache, so that mixed patching can reuse the runs for exact patching.
struct AttributionRuns {
    PatchRuns runs;  // the clean and corrupted caches and metrics, as prepare_patch gives them
    ActivationCache grads;
};

AttributionRuns prepare_attribution(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other, const std::set<std::string>& names, int64_t steps,
                                    std::set<std::string> clean_extra = {}, std::set<std::string> corrupt_extra = {}) {
    check_patch_inputs(m, clean, corrupt, pos, target, other);
    if (steps < 1) throw std::invalid_argument("attribution patching needs at least one step");
    AttributionRuns a;
    const Tokens c(clean.begin(), clean.begin() + pos + 1), x(corrupt.begin(), corrupt.begin() + pos + 1);
    clean_extra.insert(names.begin(), names.end());
    corrupt_extra.insert(names.begin(), names.end());
    // Token embeddings (hook_embed) of both inputs, for the interpolated runs.
    Tensor e_clean, e_corrupt;
    if (steps > 1) {
        Hooks grab;
        grab.add("hook_embed", [&](const std::string&, Tensor& e) { (e_clean.numel() ? e_corrupt : e_clean) = e; });
        m.embed(c, &grab);
        m.embed(x, &grab);
    }
    // Step k's run: the corrupted input with its token embeddings moved k / steps of the way to the
    // clean ones (step 0 is the corrupted run itself).
    auto interpolate = [&](int64_t k) {
        Hooks hooks;
        if (k == 0) return hooks;
        const float alpha = static_cast<float>(k) / static_cast<float>(steps);
        hooks.add("hook_embed", [&, alpha](const std::string&, Tensor& e) {
            for (int64_t i = 0; i < e.numel(); ++i)
                e.data[i] = e_corrupt.data[i] + alpha * (e_clean.data[i] - e_corrupt.data[i]);
        });
        return hooks;
    };
    // On the GPU (macOS), the steps' gradient runs go in batches of stacked sequences
    // (Model::gradients_rows); batches too small for the GPU, and other systems, run one sequence
    // at a time on the CPU. The batches depend only on the number of steps and the length.
    const int64_t P = pos + 1;
    std::vector<int64_t> all_steps(steps);
    for (int64_t k = 0; k < steps; ++k) all_steps[k] = k;
    std::vector<Batch> gpu_batches;
    std::vector<int64_t> cpu_steps;
    detail::Gpu* gpu = nullptr;
#ifdef TFI_USE_ACCELERATE
    // Only with enough work in all to pay for starting the GPU (about 60 ms in a fresh process).
    if (use_gpu() && steps * P >= detail::gpu_min_rows())
        for (Batch& b : make_batches({all_steps}, [P](int64_t) { return P; }, gradient_batch_rows(steps, P))) {
            if (b.rows >= std::max<int64_t>(1, detail::gpu_min_rows() / 4) && (gpu || (gpu = m.gpu())))
                gpu_batches.push_back(std::move(b));
            else cpu_steps.insert(cpu_steps.end(), b.items.begin(), b.items.end());
        }
    else cpu_steps = all_steps;
#else
    cpu_steps = all_steps;
#endif
    const bool corrupt_on_cpu = std::find(cpu_steps.begin(), cpu_steps.end(), 0) != cpu_steps.end();
    std::vector<ActivationCache> step_grads(steps);
    // Jobs: the clean run, the corrupted run (with step 0's gradients when that is on the CPU), the
    // GPU batches, then the other CPU steps.
    const int64_t n_cpu_jobs = static_cast<int64_t>(cpu_steps.size()) - (corrupt_on_cpu ? 1 : 0);
    const int64_t n_gpu = static_cast<int64_t>(gpu_batches.size());
    std::vector<int64_t> later_cpu;
    for (int64_t k : cpu_steps)
        if (k != 0) later_cpu.push_back(k);
    parallel_for(2 + n_gpu + n_cpu_jobs, [&](int64_t j) {
        if (j == 0) {
            a.runs.result.clean_metric =
                logit_metric(m.run_with_cache(c, a.runs.clean_cache, only(clean_extra)), pos, target, other);
        } else if (j == 1) {
            if (corrupt_on_cpu) {
                ActivationCache all;
                a.runs.result.corrupt_metric = metric_gradients(m, x, pos, target, other, step_grads[0], only(names), &all);
                for (const auto& n : corrupt_extra)
                    if (all.count(n)) a.runs.corrupt_cache[n] = std::move(all[n]);
            } else {
                a.runs.result.corrupt_metric =
                    logit_metric(m.run_with_cache(x, a.runs.corrupt_cache, only(corrupt_extra)), pos, target, other);
            }
        } else if (j < 2 + n_gpu) {
            const Batch& b = gpu_batches[j - 2];
            detail::RowBatch rb;
            rb.gpu = gpu;
            Tensor x0({b.rows, m.cfg().d_model});
            for (int64_t k : b.items) {
                const Hooks h = interpolate(k);
                const Tensor e = m.embed(x, &h);
                std::copy(e.ptr(), e.ptr() + e.numel(), x0.ptr() + rb.rows() * m.cfg().d_model);
                rb.add(0, P);
            }
            std::vector<ActivationCache> g = m.gradients_rows(
                x0, rb, [&](int64_t, const Tensor& logits) { return metric_dlogits(logits, 0, target, other); },
                only(names));
            for (size_t i = 0; i < b.items.size(); ++i) step_grads[b.items[i]] = std::move(g[i]);
        } else {
            const int64_t k = later_cpu[j - 2 - n_gpu];
            const Hooks hooks = interpolate(k);
            m.run_with_gradients(
                x, [&](const Tensor& logits) { return metric_dlogits(logits, pos, target, other); }, step_grads[k],
                only(names), nullptr, &hooks);
        }
    });
    a.grads = std::move(step_grads[0]);
    if (steps > 1)
        for (auto& kv : a.grads) {  // average, summing in step order: deterministic
            for (int64_t k = 1; k < steps; ++k) add_(kv.second, step_grads[k].at(kv.first));
            kv.second = scale(kv.second, 1.f / static_cast<float>(steps));
        }
    return a;
}

std::set<std::string> residual_names(const Model& m, const std::string& site) {
    static const std::vector<std::string> sites = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"};
    if (std::find(sites.begin(), sites.end(), site) == sites.end())
        throw std::invalid_argument("unknown patch site '" + site + "'");
    std::set<std::string> names;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) names.insert(lp(i) + "hook_" + site);
    return names;
}

std::set<std::string> head_names(const Model& m) {
    std::set<std::string> names;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) names.insert(lp(i) + "attn.hook_z");
    return names;
}

// What exact patching caches beyond the patched activations: every layer's input (and for residual
// sites, attention's keys and values) in the corrupted run.
std::set<std::string> exact_corrupt_extra(const Model& m, bool residual) {
    std::set<std::string> names;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) {
        names.insert(lp(i) + "hook_resid_pre");
        if (residual) names.insert({lp(i) + "attn.hook_k", lp(i) + "attn.hook_v"});
    }
    return names;
}

PatchResult attribution_residual_result(const Model& m, const AttributionRuns& a, const std::string& site,
                                        int64_t pos, int64_t T) {
    const int64_t L = m.cfg().n_layers, d = m.cfg().d_model;
    PatchResult r = a.runs.result;
    r.effect = Tensor({L, T}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t t = 0; t < T; ++t) r.col_labels.push_back(std::to_string(t));
    for (int64_t i = 0; i < L; ++i) {
        const std::string n = lp(i) + "hook_" + site;
        if (!a.runs.clean_cache.count(n)) continue;  // site doesn't exist in this layer: stays NaN
        const Tensor &cl = a.runs.clean_cache.at(n), &co = a.runs.corrupt_cache.at(n), &g = a.grads.at(n);
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

PatchResult attribution_heads_result(const Model& m, const AttributionRuns& a) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    PatchResult r = a.runs.result;
    r.effect = Tensor({L, H}, kNaN);
    for (int64_t i = 0; i < L; ++i) r.row_labels.push_back("L" + std::to_string(i));
    for (int64_t h = 0; h < H; ++h) r.col_labels.push_back("H" + std::to_string(h));
    for (int64_t i = 0; i < L; ++i) {
        const std::string n = lp(i) + "attn.hook_z";  // (pos+1, H, D)
        const Tensor &cl = a.runs.clean_cache.at(n), &co = a.runs.corrupt_cache.at(n), &g = a.grads.at(n);
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

}  // namespace

PatchResult attribution_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt,
                                       const std::string& site, int64_t pos, int64_t target, int64_t other,
                                       int64_t steps) {
    const std::set<std::string> names = residual_names(m, site);
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, names, steps);
    return attribution_residual_result(m, a, site, pos, static_cast<int64_t>(clean.size()));
}

PatchResult attribution_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos,
                                    int64_t target, int64_t other, int64_t steps) {
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, head_names(m), steps);
    return attribution_heads_result(m, a);
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

// Mixed patching reuses the attribution's clean and corrupted runs for exact patching (they are the
// same runs prepare_patch would do, so the exact values are the same bit for bit).
PatchResult mixed_patch_residual(const Model& m, const Tokens& clean, const Tokens& corrupt, const std::string& site,
                                 int64_t pos, int64_t target, int64_t other, int64_t verify, int64_t steps) {
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, residual_names(m, site), steps,
                                                  {}, exact_corrupt_extra(m, true));
    PatchResult atp = attribution_residual_result(m, a, site, pos, static_cast<int64_t>(clean.size()));
    const std::vector<Site> sites = top_sites(atp.effect, verify, [pos](int64_t, int64_t t) { return t <= pos; });
    const PatchResult exact = exact_patch_residual(m, clean, corrupt, site, pos, target, other, &sites, &a.runs);
    return merge_mixed(std::move(atp), exact, sites);
}

PatchResult mixed_patch_heads(const Model& m, const Tokens& clean, const Tokens& corrupt, int64_t pos, int64_t target,
                              int64_t other, int64_t verify, int64_t steps) {
    const AttributionRuns a = prepare_attribution(m, clean, corrupt, pos, target, other, head_names(m), steps, {},
                                                  exact_corrupt_extra(m, false));
    PatchResult atp = attribution_heads_result(m, a);
    const std::vector<Site> sites = top_sites(atp.effect, verify, [](int64_t, int64_t) { return true; });
    const PatchResult exact = exact_patch_heads(m, clean, corrupt, pos, target, other, &sites, &a.runs);
    return merge_mixed(std::move(atp), exact, sites);
}

// --- ablation -----------------------------------------------------------------------------

// Reference sequences run like patching passes (see exact patching above): stacked into batches
// of at most head_batch_rows() rows, on the GPU on macOS when large enough, batches in parallel.
// The batches depend only on the references, and the sums run in reference order, so the means are
// the same for any thread count.
HeadMeans head_means(const Model& m, const std::vector<Tokens>& reference, int64_t T) {
    if (reference.empty()) throw std::invalid_argument("mean ablation needs at least one reference sequence");
    const int64_t L = m.cfg().n_layers, R = static_cast<int64_t>(reference.size());
    const bool per_position = std::all_of(reference.begin(), reference.end(),
                                          [T](const Tokens& r) { return static_cast<int64_t>(r.size()) == T; });
    HeadMeans means;
    std::vector<double> counts(L, 0.0);
    for (int64_t i = 0; i < L; ++i)
        means.layers.emplace_back(Shape{per_position ? T : 1, m.layer_cfg(i).n_heads, m.layer_cfg(i).head_dim()});
    std::vector<int64_t> all(R);
    for (int64_t n = 0; n < R; ++n) all[n] = n;
    const std::vector<Batch> batches = make_batches(
        {all}, [&](int64_t n) { return static_cast<int64_t>(reference[n].size()); }, ablation_batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = gpu_fp16();
    std::vector<std::vector<Tensor>> zs(R, std::vector<Tensor>(L));  // reference n, layer i: (len, H, D)
    parallel_for(static_cast<int64_t>(batches.size()), [&](int64_t j) {
        const Batch& b = batches[j];
        detail::RowBatch rb;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        std::vector<Tensor> inputs;
        for (int64_t n : b.items) {
            inputs.push_back(m.embed(reference[n]));
            rb.add(0, static_cast<int64_t>(reference[n].size()));
        }
        Tensor x({rb.rows(), m.cfg().d_model});
        for (size_t s = 0; s < inputs.size(); ++s)
            std::copy(inputs[s].data.begin(), inputs[s].data.end(), x.ptr() + rb.offset[s] * m.cfg().d_model);
        Hooks hooks;
        for (int64_t i = 0; i < L; ++i)
            hooks.add(lp(i) + "attn.hook_z", [&, i](const std::string&, Tensor& z) {  // (rows, H, D)
                const int64_t row = z.numel() / z.shape[0];
                for (size_t s = 0; s < b.items.size(); ++s) {
                    Tensor& dst = zs[b.items[s]][i];
                    dst = Tensor({rb.len[s], z.shape[1], z.shape[2]});
                    std::copy(z.ptr() + rb.offset[s] * row, z.ptr() + (rb.offset[s] + rb.len[s]) * row, dst.ptr());
                }
            });
        m.forward_rows(0, std::move(x), rb, &hooks);
    });
    for (int64_t n = 0; n < R; ++n) {  // reduce in reference order: deterministic sums
        for (int64_t i = 0; i < L; ++i) {
            const Tensor& z = zs[n][i];
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

// Ablation passes run like head patching passes (see exact patching above): a head ablated in layer
// i starts at layer i from the cached residual stream, passes starting at the same layer are
// stacked into one batch, and the batches run in parallel. A head whose output already equals its
// replacement at every position (zeros, or its mean) changes nothing: exactly 0, without a pass.
Tensor head_ablation(const Model& m, const Tokens& tokens, Ablation mode, const std::vector<Tokens>& reference) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads(), T = static_cast<int64_t>(tokens.size());
    const int64_t d = m.cfg().d_model, V = m.cfg().vocab_size;
    if (T < 2) throw std::invalid_argument("loss needs at least two tokens");
    const HeadMeans means = mode == Ablation::Mean ? head_means(m, reference, T) : HeadMeans{};
    std::set<std::string> needed;
    for (int64_t i = 0; i < L; ++i) {
        needed.insert(lp(i) + "hook_resid_pre");
        needed.insert(lp(i) + "attn.hook_z");
    }
    ActivationCache cache;
    const float base = next_token_loss(m.run_with_cache(tokens, cache, only(std::move(needed))), tokens);
    Tensor out({L, H}, kNaN);

    // The replacement for head h of layer i at position t (D floats).
    int64_t max_d = 0;
    for (int64_t i = 0; i < L; ++i) max_d = std::max(max_d, m.layer_cfg(i).head_dim());
    const std::vector<float> zeros(static_cast<size_t>(max_d), 0.f);
    auto replacement = [&](int64_t i, int64_t h, int64_t t) -> const float* {
        if (mode == Ablation::Zero) return zeros.data();
        const Tensor& mean = means.layers[i];
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        return mean.ptr() + ((mean.shape[0] == 1 ? 0 : t) * Hl + h) * D;
    };
    std::vector<std::vector<int64_t>> todo(L);
    std::vector<const Tensor*> start(L);
    for (int64_t i = 0; i < L; ++i) {
        start[i] = &cache.at(lp(i) + "hook_resid_pre");
        const Tensor& z = cache.at(lp(i) + "attn.hook_z");
        const int64_t Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        for (int64_t h = 0; h < Hl; ++h) {
            bool same = true;
            for (int64_t t = 0; t < T && same; ++t) same = same_floats(z.ptr() + (t * Hl + h) * D, replacement(i, h, t), D);
            if (same) out.data[i * H + h] = 0.f;
            else todo[i].push_back(h);
        }
    }

    const std::vector<Batch> batches = make_batches(todo, [T](int64_t) { return T; }, ablation_batch_rows());
    detail::Gpu* const gpu = gpu_for(m, batches);
    const bool fp16 = gpu_fp16();  // read once: the same for every batch
    // Logits are unembedded a few passes at a time: at most this many floats at once.
    constexpr int64_t kMaxLogits = int64_t(1) << 23;
    parallel_for(static_cast<int64_t>(batches.size()), [&](int64_t j) {
        const Batch& b = batches[j];
        const int64_t i = b.layer, Hl = m.layer_cfg(i).n_heads, D = m.layer_cfg(i).head_dim();
        const int64_t S = static_cast<int64_t>(b.items.size());
        detail::RowBatch rb;
        rb.gpu = gpu;
        rb.fp16 = fp16;
        for (int64_t s = 0; s < S; ++s) rb.add(0, T);
        Tensor x({rb.rows(), d});
        for (int64_t s = 0; s < S; ++s) std::copy(start[i]->ptr(), start[i]->ptr() + T * d, x.ptr() + rb.offset[s] * d);
        Hooks hooks;
        hooks.add(lp(i) + "attn.hook_z", [&](const std::string&, Tensor& z) {  // each pass's head, every position
            for (int64_t s = 0; s < S; ++s) {
                const int64_t h = b.items[s];
                for (int64_t t = 0; t < T; ++t) {
                    const float* src = replacement(i, h, t);
                    std::copy(src, src + D, z.ptr() + ((rb.offset[s] + t) * Hl + h) * D);
                }
            }
        });
        const Tensor fin = m.forward_rows_final(i, std::move(x), rb, &hooks);
        // Next-token loss of each pass: rows 0..T-2 predict tokens 1..T-1.
        const int64_t per = std::max<int64_t>(1, kMaxLogits / ((T - 1) * V));
        for (int64_t s0 = 0; s0 < S; s0 += per) {
            const int64_t n = std::min(per, S - s0);
            Tensor rows_in({n * (T - 1), d});
            for (int64_t s = 0; s < n; ++s)
                std::copy(fin.ptr() + rb.offset[s0 + s] * d, fin.ptr() + (rb.offset[s0 + s] + T - 1) * d,
                          rows_in.ptr() + s * (T - 1) * d);
            const Tensor logits = m.logits_from_final(rows_in, &rb);
            for (int64_t s = 0; s < n; ++s) {
                double loss = 0;
                for (int64_t t = 0; t + 1 < T; ++t)
                    loss -= log_softmax_at(logits.ptr() + (s * (T - 1) + t) * V, V, tokens[t + 1]);
                out.data[i * H + b.items[s0 + s]] = static_cast<float>(loss / (T - 1)) - base;
            }
        }
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
    w.W_Q = m.weight(p + "q_proj.weight").rows(head * D, (head + 1) * D);
    w.W_K = m.weight(p + "k_proj.weight").rows(g * D, (g + 1) * D);
    w.W_V = m.weight(p + "v_proj.weight").rows(g * D, (g + 1) * D);
    w.W_O = m.weight(p + "o_proj.weight").cols(head * D, (head + 1) * D);
    // The attention input is gamma * normalized(x) for pre/sandwich norm; fold gamma in.
    if (c.norm != "none" && c.norm_position != "post") {
        const Tensor& gamma = m.weight(lp(layer) + "ln1.weight").f32();
        for (Tensor* W : {&w.W_Q, &w.W_K, &w.W_V})
            for (int64_t r = 0; r < D; ++r)
                for (int64_t j = 0; j < c.d_model; ++j) W->data[r * c.d_model + j] *= gamma.data[j];
    }
    return w;
}

Tensor ov_matrix(const HeadWeights& h) { return matmul(h.W_O, h.W_V); }
Tensor qk_matrix(const HeadWeights& h) { return matmul(transpose2d(h.W_Q), h.W_K); }

// The heads' Gram matrices are computed in parallel, then the pairs. Each pair's arithmetic is the
// same as on one thread, so the scores don't depend on the thread count.
Composition composition_scores(const Model& m) {
    std::vector<std::pair<int64_t, int64_t>> ids;  // (layer, head)
    Composition c;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) {
            ids.emplace_back(i, h);
            c.labels.push_back(head_label(i, h));
        }
    const int64_t N = static_cast<int64_t>(ids.size());
    std::vector<HeadCache> heads(N);
    parallel_for(N, [&](int64_t a) { heads[a] = cache_head(head_weights(m, ids[a].first, ids[a].second)); });
    c.q = Tensor({N, N}, kNaN);
    c.k = Tensor({N, N}, kNaN);
    c.v = Tensor({N, N}, kNaN);
    std::vector<std::pair<int64_t, int64_t>> pairs;  // writer in an earlier layer than the reader
    for (int64_t a = 0; a < N; ++a)
        for (int64_t b = 0; b < N; ++b)
            if (ids[a].first < ids[b].first) pairs.emplace_back(a, b);
    parallel_for(static_cast<int64_t>(pairs.size()), [&](int64_t n) {
        const auto [a, b] = pairs[n];
        const CompScores s = compose(heads[a], heads[b]);
        c.q.data[a * N + b] = s.q;
        c.k.data[a * N + b] = s.k;
        c.v.data[a * N + b] = s.v;
    });
    return c;
}

// Each random pair draws from its own stream (seeded by seed and the pair's index), so the pairs run
// in parallel and the result doesn't depend on the thread count.
float composition_random_baseline(const Model& m, int64_t n_pairs, uint64_t seed) {
    const int64_t d = m.cfg().d_model, D = m.cfg().head_dim();
    std::vector<double> score(n_pairs);
    parallel_for(n_pairs, [&](int64_t n) {
        std::seed_seq seq{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), static_cast<uint32_t>(n),
                          static_cast<uint32_t>(static_cast<uint64_t>(n) >> 32)};
        std::mt19937_64 rng(seq);
        std::normal_distribution<float> normal(0.f, 1.f);
        auto random = [&](Shape sh) {
            Tensor t(sh);
            for (float& v : t.data) v = normal(rng);
            return t;
        };
        auto random_head = [&]() { return cache_head({random({D, d}), random({D, d}), random({D, d}), random({d, D})}); };
        const HeadCache writer = random_head(), reader = random_head();
        const CompScores s = compose(writer, reader);
        score[n] = (s.q + s.k + s.v) / 3.0;
    });
    double total = 0;
    for (double v : score) total += v;  // in pair order: deterministic
    return static_cast<float>(total / n_pairs);
}

namespace {

// OV paths through the sampled tokens. A head's OV circuit W_O W_V has rank D (head size) < d_model,
// so embed -> OV(a) -> ... -> OV(z) -> unembed is evaluated in factored form:
//   (X W_V(a)^T) (W_V(b) W_O(a))^T ... (U W_O(z))^T
// which is the same product as X W_V(a)^T W_O(a)^T ... W_O(z)^T U^T, with far fewer operations.
struct OvPaths {
    const Model& m;
    Tokens toks;  // the sampled tokens
    Tensor X;     // their embeddings (n, d)
    Tensor U;     // unembedding with the final norm's weight folded in (V, d)

    OvPaths(const Model& model, int64_t max_tokens) : m(model) {
        const ModelConfig& cfg = m.cfg();
        const int64_t V = cfg.vocab_size, d = cfg.d_model, n = std::min(V, max_tokens);
        toks.resize(n);
        for (int64_t i = 0; i < n; ++i) toks[i] = i * V / n;
        const Weight& E = m.weight("embed.weight");
        X = Tensor({n, d});
        for (int64_t i = 0; i < n; ++i) E.read_rows(toks[i], toks[i] + 1, X.ptr() + i * d);
        U = m.unembed().to_tensor();
        if (cfg.norm != "none" && cfg.norm_position != "post") {
            const Tensor& gamma = m.weight("ln_final.weight").f32();
            for (int64_t v = 0; v < V; ++v)
                for (int64_t j = 0; j < d; ++j) U.data[v * d + j] *= gamma.data[j];
        }
    }
    // The sampled tokens after the first head's W_V: (n, D).
    Tensor start(const HeadWeights& w) const { return matmul(X, w.W_V, true); }
    // Carry A (after `from`'s W_V) through from's W_O and `to`'s W_V: (n, D).
    static Tensor step(const Tensor& A, const HeadWeights& from, const HeadWeights& to) {
        return matmul(A, matmul(to.W_V, from.W_O), true);
    }
    // The unembedding seen through the last head's W_O: (V, D).
    Tensor finish(const HeadWeights& w) const { return matmul(U, w.W_O); }
    // Fraction of sampled tokens whose logits A UO^T rank the token itself first.
    float hits(const Tensor& A, const Tensor& UO) const {
        const int64_t n = A.shape[0], V = UO.shape[0], chunk = 256;
        int64_t hit = 0;
        for (int64_t s = 0; s < n; s += chunk) {
            const int64_t e = std::min(n, s + chunk);
            const Tensor logits = matmul(rows(A, s, e), UO, true);  // (chunk, V)
            for (int64_t r = 0; r < e - s; ++r) hit += topk(logits.ptr() + r * V, V, 1)[0] == toks[s + r];
        }
        return float(hit) / float(n);
    }
};

}  // namespace

float ov_path_copying(const Model& m, const std::vector<std::pair<int64_t, int64_t>>& path, int64_t max_tokens) {
    if (path.empty()) throw std::invalid_argument("OV path needs at least one head");
    for (size_t i = 1; i < path.size(); ++i)
        if (path[i].first <= path[i - 1].first) throw std::invalid_argument("OV path heads must be in increasing layers");
    const OvPaths ov(m, max_tokens);
    std::vector<HeadWeights> w;
    for (const auto& lh : path) w.push_back(head_weights(m, lh.first, lh.second));
    Tensor A = ov.start(w[0]);
    for (size_t i = 1; i < w.size(); ++i) A = OvPaths::step(A, w[i - 1], w[i]);
    return ov.hits(A, ov.finish(w.back()));
}

Tensor ov_copying_scores(const Model& m, int64_t max_tokens) {
    const int64_t L = m.cfg().n_layers, H = m.max_heads();
    const OvPaths ov(m, max_tokens);
    std::vector<std::pair<int64_t, int64_t>> ids;
    for (int64_t i = 0; i < L; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) ids.emplace_back(i, h);
    Tensor out({L, H}, kNaN);
    parallel_for(static_cast<int64_t>(ids.size()), [&](int64_t j) {
        const auto [i, h] = ids[j];
        const HeadWeights w = head_weights(m, i, h);
        out.data[i * H + h] = ov.hits(ov.start(w), ov.finish(w));
    });
    return out;
}

Tensor ov_pair_copying_scores(const Model& m, int64_t max_tokens) {
    std::vector<std::pair<int64_t, int64_t>> ids;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i)
        for (int64_t h = 0; h < m.layer_cfg(i).n_heads; ++h) ids.emplace_back(i, h);
    const int64_t N = static_cast<int64_t>(ids.size());
    const OvPaths ov(m, max_tokens);
    std::vector<HeadWeights> w(N);
    std::vector<Tensor> A(N);  // each head as the writer: (n, D)
    parallel_for(N, [&](int64_t a) {
        w[a] = head_weights(m, ids[a].first, ids[a].second);
        A[a] = ov.start(w[a]);
    });
    Tensor out({N, N}, kNaN);
    // One reader per job: its unembedding side (V, D) is computed once for all its writers.
    parallel_for(N, [&](int64_t b) {
        Tensor UO;
        for (int64_t a = 0; a < N; ++a) {
            if (ids[a].first >= ids[b].first) continue;
            if (UO.data.empty()) UO = ov.finish(w[b]);
            out.data[a * N + b] = ov.hits(OvPaths::step(A[a], w[a], w[b]), UO);
        }
    });
    return out;
}

Tensor positional_qk(const Model& m, int64_t layer, int64_t head, int64_t T) {
    if (m.cfg().positional != "learned") throw std::runtime_error("positional QK needs learned positional embeddings");
    if (T > m.cfg().max_seq_len) throw std::out_of_range("T exceeds max_seq_len");
    const HeadWeights w = head_weights(m, layer, head);
    const Tensor P = m.weight("pos_embed.weight").rows(0, T);
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
