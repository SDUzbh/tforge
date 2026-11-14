// indra: IndraNet's command line. Reverse engineers models exported with indranet.export.export_model,
// using the tfi engine (include/tfi).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tfi/analysis.h"
#include "tfi/model.h"

using namespace tfi;

namespace {

const char* kUsage = R"(usage: indra <command> <model.safetensors> [options]

Commands
  info       config, parameter count and hook point names
  run        next-token predictions                           --tokens
  lens       logit lens over the residual stream               --tokens [--pos] [--target] [--top]
  dla        direct logit attribution per head / MLP           --tokens --target [--other] [--pos] [--top]
  heads      attention-head pattern scores on repeated text    [--seg 24] [--samples 8] [--seed 0]
  patch      activation patching, clean -> corrupted           --clean --corrupt --target [--other] [--pos]
                                                               [--site resid_pre|resid_mid|resid_post|attn_out|mlp_out|heads]
                                                               [--raw: show patched - corrupted instead of normalizing]
                                                               [--method exact (default)|attribution|mixed] [--steps 1]
                                                               [--verify 10] [--compare]
             --method attribution estimates every site at once from 2 forward + 1 backward passes
             (a linear approximation); --steps N averages gradients along N interpolated inputs
             (integrated gradients, better for large changes); --method mixed then measures the
             --verify K largest estimates exactly; --compare also runs full exact patching
  ablate     loss increase when each head is ablated           --tokens [--mode mean|zero]
             mean ablation averages each head over reference sequences, never the input itself:
             --reference-file F (one per line) | --reference-text "a;b" | --reference "1,2;3,4"
             | --reference-random N (random tokens; only for models trained on random tokens)
  circuits   weight-only: composition scores, OV copying       [--top 10] [--tokens-sampled 1024]
  attn       where each head attends, query by query           --tokens --layer L [--head H] [--top K] [--pos P]

Token input
  --tokens 5,17,3,...          explicit token ids
  --text "00123+00045="        text, if the model was exported with a vocabulary
                               (patch: --clean / --corrupt or --clean-text / --corrupt-text)
  --target / --other           a token id, or a token's text when the model has a vocabulary
  --repeat-random L [--seed S] a random segment of length L, repeated twice (for induction)
  --pos P                      position to analyse (default: last; negative counts from the end)

Performance
  --threads N                  threads for patch / ablate (default: TFI_THREADS, else all cores)
  --no-gpu                     patch / ablate: keep everything on the CPU (default: the
                               GPU when there is one, macOS only; TFI_NO_GPU=1 does the same)

Precision
  --fp32                       weights in float32 (the default)
  --fp16                       weights in fp16: half the memory, each weight rounded to about 3
                               significant digits (relative error up to 5e-4); patch / ablate also
                               give the GPU's multiplies fp16 inputs, which is faster. Errors around
                               1e-3 on the patching metric (fine for gaps of 1 or more) and around
                               1e-4 on ablation's loss increases. Arithmetic stays float32.
)";

struct Args {
    std::string command, model;
    std::map<std::string, std::string> opts;

    bool has(const std::string& k) const { return opts.count(k) > 0; }
    std::string get(const std::string& k, const std::string& def = "") const {
        auto it = opts.find(k);
        return it == opts.end() ? def : it->second;
    }
    int64_t get_int(const std::string& k, int64_t def) const { return has(k) ? std::stoll(get(k)) : def; }
    std::string require(const std::string& k) const {
        if (!has(k)) throw std::invalid_argument("missing --" + k);
        return get(k);
    }
};

Args parse_args(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("expected a command and a model file");
    Args a;
    a.command = argv[1];
    a.model = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument '" + k + "'");
        k = k.substr(2);
        // A flag with no value (e.g. --raw) is followed by another --option or nothing.
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) a.opts[k] = "1";
        else a.opts[k] = argv[++i];
    }
    return a;
}

Tokens parse_tokens(const std::string& s) {
    Tokens t;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) t.push_back(std::stoll(item));
    if (t.empty()) throw std::invalid_argument("no tokens given");
    return t;
}

Tokens repeat_random(const Model& m, int64_t len, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, m.cfg().vocab_size - 1);
    Tokens t(2 * len);
    for (int64_t i = 0; i < len; ++i) t[i] = t[i + len] = dist(rng);
    return t;
}

Tokens input_tokens(const Args& a, const Model& m, const std::string& key = "tokens") {
    const std::string text_key = key == "tokens" ? "text" : key + "-text";
    if (a.has(text_key)) return m.tokenize(a.get(text_key));
    if (a.has(key)) return parse_tokens(a.get(key));
    if (key == "tokens" && a.has("repeat-random")) return repeat_random(m, a.get_int("repeat-random", 8), a.get_int("seed", 0));
    throw std::invalid_argument("give --" + key + " or --" + text_key + (key == "tokens" ? " or --repeat-random" : ""));
}

// A token named on the command line: its text when the model has a vocabulary, otherwise an id.
int64_t parse_token(const Model& m, const std::string& s) {
    const auto& v = m.vocab();
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == s) return static_cast<int64_t>(i);
    return std::stoll(s);
}

int64_t token_opt(const Args& a, const Model& m, const std::string& key, int64_t def) {
    return a.has(key) ? parse_token(m, a.get(key)) : def;
}

// For display: 'text' when the model has a vocabulary, the id otherwise.
std::string tok(const Model& m, int64_t id) { return m.vocab().empty() ? std::to_string(id) : "'" + m.token_str(id) + "'"; }

int64_t resolve_pos(const Args& a, const Tokens& t) {
    int64_t p = a.get_int("pos", -1);
    if (p < 0) p += static_cast<int64_t>(t.size());
    return p;
}

std::string opt_str(const std::optional<int64_t>& v) { return v ? std::to_string(*v) : "-"; }
std::string opt_str(const std::optional<double>& v) {
    if (!v) return "-";
    std::ostringstream o;
    o << *v;
    return o.str();
}

// `marks` (optional, one per entry): entries flagged 1 are printed with a trailing '*'.
void print_matrix(const Tensor& t, const std::vector<std::string>& rows, const std::vector<std::string>& cols,
                  const char* fmt = "%7.3f", const std::vector<uint8_t>* marks = nullptr) {
    const char* col = marks ? "%7s " : "%7s";  // a mark takes one more character per entry
    std::printf("%8s", "");
    for (const auto& c : cols) std::printf(col, c.c_str());
    std::printf("\n");
    for (size_t r = 0; r < rows.size(); ++r) {
        std::printf("%8s", rows[r].c_str());
        for (size_t c = 0; c < cols.size(); ++c) {
            const size_t i = r * cols.size() + c;
            const float v = t.data[i];
            if (std::isnan(v)) std::printf(col, ".");
            else if (marks) std::printf((*marks)[i] ? "%7.3f*" : "%7.3f ", v);
            else std::printf(fmt, v);
        }
        std::printf("\n");
    }
}

std::vector<std::string> layer_labels(const Model& m) {
    std::vector<std::string> l;
    for (int64_t i = 0; i < m.cfg().n_layers; ++i) l.push_back("L" + std::to_string(i));
    return l;
}

std::vector<std::string> head_labels(const Model& m) {
    std::vector<std::string> l;
    for (int64_t h = 0; h < m.max_heads(); ++h) l.push_back("H" + std::to_string(h));
    return l;
}

// --- commands -------------------------------------------------------------------------

int cmd_info(const Args&, const Model& m) {
    const ModelConfig& c = m.cfg();
    std::printf("parameters    %lld\n", static_cast<long long>(m.num_parameters()));
    std::printf("weights       %s, %.2f GB in memory\n", m.weight_format() == WeightFormat::fp16 ? "fp16" : "float32",
                double(m.weight_bytes()) / 1e9);
    std::printf("vocab/d_model %lld / %lld   layers %lld   heads %lld (kv %lld, d_head %lld)   max_seq_len %lld\n",
                (long long)c.vocab_size, (long long)c.d_model, (long long)c.n_layers, (long long)c.n_heads,
                (long long)c.kv_heads(), (long long)c.head_dim(), (long long)c.max_seq_len);
    std::printf("norm          %s (%s)%s\n", c.norm.c_str(), c.norm_position.c_str(), c.qk_norm ? ", qk-norm" : "");
    std::printf("positional    %s\n", c.positional.c_str());
    std::printf("mlp           %s (%s)\n", c.mlp.c_str(), c.has_mlp() ? c.mlp_activation().c_str() : "attention-only");
    std::printf("attention     window %s, scale %s, softcap %s%s\n", opt_str(c.sliding_window).c_str(),
                opt_str(c.attn_scale).c_str(), opt_str(c.attn_logit_softcap).c_str(),
                c.parallel_residual ? ", parallel residual" : "");
    std::printf("embeddings    %s%s, final softcap %s\n", c.tie_embeddings ? "tied" : "untied",
                c.embed_scale ? ", scaled by sqrt(d_model)" : "", opt_str(c.final_logit_softcap).c_str());
    for (const auto& kv : c.layer_overrides) {
        std::printf("layer %lld      overrides:", static_cast<long long>(kv.first));
        for (size_t i = 0; i < kv.second.keys.size(); ++i) std::printf(" %s", kv.second.keys[i].c_str());
        std::printf("\n");
    }
    const auto names = m.hook_names();
    std::printf("\n%zu hook points (layer 0 shown):\n", names.size());
    for (const auto& n : names)
        if (n.find("blocks.") != 0 || n.find("blocks.0.") == 0) std::printf("  %s\n", n.c_str());
    return 0;
}

int cmd_run(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const Tensor logits = m.forward(t);
    const int64_t V = m.cfg().vocab_size, k = a.get_int("top", 3);
    std::printf("%5s %7s   top-%lld next tokens (prob)\n", "pos", "token", static_cast<long long>(k));
    for (int64_t p = 0; p < static_cast<int64_t>(t.size()); ++p) {
        Tensor probs = rows(logits, p, p + 1);
        softmax_(probs.ptr(), V);
        std::printf("%5lld %7s  ", (long long)p, tok(m, t[p]).c_str());
        for (int64_t id : topk(probs.ptr(), V, k)) std::printf("  %6s (%.3f)", tok(m, id).c_str(), probs.data[id]);
        if (p + 1 < static_cast<int64_t>(t.size()))
            std::printf("   | actual next %s: p=%.3f", tok(m, t[p + 1]).c_str(), probs.data[t[p + 1]]);
        std::printf("\n");
    }
    if (t.size() > 1) std::printf("mean next-token loss %.4f\n", next_token_loss(logits, t));
    return 0;
}

int cmd_lens(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const int64_t pos = resolve_pos(a, t);
    const int64_t target = token_opt(a, m, "target", pos + 1 < (int64_t)t.size() ? t[pos + 1] : -1);
    const auto lens = logit_lens(m, t, pos, target, a.get_int("top", 5));
    std::printf("logit lens at position %lld (token %s)", (long long)pos, tok(m, t[pos]).c_str());
    if (target >= 0) std::printf(", tracking token %s", tok(m, target).c_str());
    std::printf("\n\n%-9s %-10s %-8s  top predictions (prob)\n", "stage", target >= 0 ? "p(target)" : "", target >= 0 ? "rank" : "");
    for (const auto& e : lens) {
        std::printf("%-9s ", e.stage.c_str());
        if (target >= 0) std::printf("%-10.4f %-8lld ", e.target_prob, (long long)e.target_rank);
        else std::printf("%-10s %-8s ", "", "");
        for (size_t i = 0; i < e.top_tokens.size(); ++i)
            std::printf(" %s (%.3f)", tok(m, e.top_tokens[i]).c_str(), e.top_probs[i]);
        std::printf("\n");
    }
    return 0;
}

int cmd_dla(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const int64_t pos = resolve_pos(a, t);
    const int64_t target = parse_token(m, a.require("target")), other = token_opt(a, m, "other", -1);
    const DLAResult r = direct_logit_attribution(m, t, pos, target, other);
    std::vector<size_t> order(r.values.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return std::fabs(r.values[x]) > std::fabs(r.values[y]); });
    const size_t top = std::min<size_t>(order.size(), static_cast<size_t>(a.get_int("top", 15)));
    std::printf("direct logit attribution at position %lld: logit[%s]%s\n", (long long)pos, tok(m, target).c_str(),
                other >= 0 ? (" - logit[" + tok(m, other) + "]").c_str() : "");
    std::printf("actual %.4f   sum of parts %.4f   (final norm scale frozen)\n\n", r.actual, r.sum);
    for (size_t i = 0; i < top; ++i) {
        const float v = r.values[order[i]];
        const int bar = static_cast<int>(std::min(40.f, std::fabs(v) / std::max(1e-6f, std::fabs(r.values[order[0]])) * 40.f));
        std::printf("%-16s %9.4f  %s%s\n", r.labels[order[i]].c_str(), v, v < 0 ? "-" : "+", std::string(bar, '#').c_str());
    }
    if (top < order.size()) std::printf("... %zu smaller components (--top)\n", order.size() - top);
    return 0;
}

int cmd_heads(const Args& a, const Model& m) {
    const int64_t seg = a.get_int("seg", 24);
    const HeadScores s = head_scores(m, seg, a.get_int("samples", 8), a.get_int("seed", 0));
    const auto rows = layer_labels(m), cols = head_labels(m);
    std::printf("attention-head scores on random segments repeated twice (queries in the repeat)\n");
    std::printf("\nprevious token (t-1):\n");
    print_matrix(s.previous_token, rows, cols);
    std::printf("\nduplicate token (earlier copy of the current token):\n");
    print_matrix(s.duplicate_token, rows, cols);
    std::printf("\ninduction (token after the earlier copy):\n");
    print_matrix(s.induction, rows, cols);
    std::printf("\nfirst token (position 0):\n");
    print_matrix(s.first_token, rows, cols);
    std::printf("\nself (t):\n");
    print_matrix(s.self, rows, cols);
    if (!s.top_two_hop.empty()) {
        std::printf("\ntwo-hop induction (reader attends to r, writer at r attends to the token to copy; any lag):\n");
        for (const auto& th : s.top_two_hop)
            std::printf("  %s -> %s   %.3f\n", th.writer.c_str(), th.reader.c_str(), th.score);
    }
    return 0;
}

int cmd_patch(const Args& a, const Model& m) {
    const Tokens clean = input_tokens(a, m, "clean"), corrupt = input_tokens(a, m, "corrupt");
    const int64_t pos = resolve_pos(a, clean);
    const int64_t target = parse_token(m, a.require("target")), other = token_opt(a, m, "other", -1);
    const std::string site = a.get("site", "resid_pre"), method = a.get("method", "exact");
    if (method != "exact" && method != "attribution" && method != "mixed")
        throw std::invalid_argument("--method must be exact, attribution or mixed, not '" + method + "'");
    const int64_t steps = a.get_int("steps", 1), verify = a.get_int("verify", 10);
    const bool heads = site == "heads";
    auto run = [&](const std::string& how) {
        if (how == "attribution")
            return heads ? attribution_patch_heads(m, clean, corrupt, pos, target, other, steps)
                         : attribution_patch_residual(m, clean, corrupt, site, pos, target, other, steps);
        if (how == "mixed")
            return heads ? mixed_patch_heads(m, clean, corrupt, pos, target, other, verify, steps)
                         : mixed_patch_residual(m, clean, corrupt, site, pos, target, other, verify, steps);
        return heads ? patch_heads(m, clean, corrupt, pos, target, other)
                     : patch_residual(m, clean, corrupt, site, pos, target, other);
    };
    const PatchResult r = run(method);
    std::printf("%s patching (%s), metric at position %lld: %s\n", method == "exact" ? "activation" : method.c_str(),
                site.c_str(), (long long)pos,
                other >= 0 ? ("logit[" + tok(m, target) + "] - logit[" + tok(m, other) + "]").c_str()
                           : ("log p(" + tok(m, target) + ")").c_str());
    std::printf("clean %.4f   corrupted %.4f   gap %.4f\n", r.clean_metric, r.corrupt_metric, r.gap());
    const Tensor* shown = &r.values;
    if (!r.normalized()) {
        std::printf("note: the corruption doesn't change this metric (gap 0), so there is nothing to restore and\n"
                    "      normalized values are undefined. Showing each patch's raw effect (patched - corrupted).\n");
        shown = &r.effect;
    } else if (a.has("raw")) {
        std::printf("raw effect of each patch on the metric (patched - corrupted)\n");
        shown = &r.effect;
    } else {
        if (std::fabs(r.gap()) < 1.0f)
            std::printf("warning: the gap is only %.3g in size, so normalized values amplify small effects and may be noise.\n"
                        "         Choose a corruption that moves the metric more, or pass --raw.\n", std::fabs(r.gap()));
        std::printf("normalized: 1 = clean behaviour restored, 0 = no effect\n");
    }
    if (a.has("fp16")) {
        const double err = 1e-3 / std::max(1e-12, double(std::fabs(r.gap())));
        std::printf("note: --fp16: weights are stored in fp16%s, which adds errors of up to about\n"
                    "      1e-3 to the metric: up to about %.2g in normalized values for this gap\n",
                    use_gpu() && m.gpu() && (method != "attribution" || a.has("compare")) ? " and the GPU's multiplies take fp16 inputs" : "",
                    err);
        if (err > 0.01) std::printf("warning: that is large for this gap; run with --fp32 for exact values\n");
    }
    if (method == "attribution")
        std::printf("linear estimate (2 forward + 1 backward passes): can be far off where a change saturates\n"
                    "attention or crosses a nonlinearity; exact patching (--method exact, the default) measures every site\n");
    const int64_t n_exact = std::count(r.exact.begin(), r.exact.end(), uint8_t(1));
    if (method == "mixed")
        std::printf("* = measured by exact patching (the %lld largest estimates); other entries are attribution estimates\n",
                    (long long)n_exact);
    std::printf("\n");
    print_matrix(*shown, r.row_labels, r.col_labels, "%7.3f", method == "mixed" ? &r.exact : nullptr);
    if (std::any_of(shown->data.begin(), shown->data.end(), [](float v) { return std::isnan(v); }))
        std::printf("(. = no such site in that layer)\n");
    if (method == "mixed" && n_exact > 0) {
        // How far off were the estimates where we checked, and how large is the biggest one left unchecked?
        // In the shown table's units: normalized values divide by the signed gap, so sizes divide by |gap|.
        const float unit = shown == &r.values ? r.gap() : 1.f, size_unit = std::fabs(unit);
        const int64_t cols = static_cast<int64_t>(r.col_labels.size());
        float worst = 0, unverified = 0;
        int64_t worst_i = -1, unverified_i = -1;
        for (int64_t i = 0; i < r.effect.numel(); ++i) {
            if (std::isnan(r.estimate.data[i])) continue;
            if (r.exact[i]) {
                const float err = std::fabs(r.effect.data[i] - r.estimate.data[i]) / size_unit;
                if (err > worst) worst = err, worst_i = i;
            } else if (std::fabs(r.estimate.data[i]) / size_unit > unverified) {
                unverified = std::fabs(r.estimate.data[i]) / size_unit, unverified_i = i;
            }
        }
        auto where = [&](int64_t i) { return r.row_labels[i / cols] + " " + r.col_labels[i % cols]; };
        std::printf("\nthe exactly measured sites' estimates were off by up to %.3f", worst);
        if (worst_i >= 0)
            std::printf(" (%s: estimate %.3f, exact %.3f)", where(worst_i).c_str(), r.estimate.data[worst_i] / unit,
                        r.effect.data[worst_i] / unit);
        std::printf("\nlargest estimate left unverified: %.3f", unverified);
        if (unverified_i >= 0) std::printf(" (%s)", where(unverified_i).c_str());
        std::printf("\nif estimates were far off, unverified sites may be too; exact patching (the default) measures every site\n");
    }
    if (method != "exact" && a.has("compare")) {
        const PatchResult exact = run("exact");
        const Tensor& want = shown == &r.values ? exact.values : exact.effect;
        // Pearson correlation and worst error over the sites that exist (and can reach `pos`).
        double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, worst = 0;
        int64_t worst_i = -1;
        for (int64_t i = 0; i < want.numel(); ++i) {
            const double x = shown->data[i], y = want.data[i];
            if (std::isnan(x) || std::isnan(y)) continue;
            n += 1, sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y;
            if (std::fabs(x - y) > worst) worst = std::fabs(x - y), worst_i = i;
        }
        const double cov = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
        std::printf("\nexact patching for comparison:\n\n");
        print_matrix(want, exact.row_labels, exact.col_labels);
        std::printf("\nagreement over %.0f sites: correlation %s, largest error %.3f", n,
                    vx > 0 && vy > 0 ? std::to_string(cov / std::sqrt(vx * vy)).substr(0, 5).c_str() : "n/a", worst);
        if (worst_i >= 0) {
            const int64_t cols = static_cast<int64_t>(r.col_labels.size());
            std::printf(" at %s %s (estimate %.3f, exact %.3f)", r.row_labels[worst_i / cols].c_str(),
                        r.col_labels[worst_i % cols].c_str(), shown->data[worst_i], want.data[worst_i]);
        }
        std::printf("\n");
    }
    return 0;
}

int cmd_attn(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const int64_t T = static_cast<int64_t>(t.size()), layer = a.get_int("layer", 0);
    if (layer < 0 || layer >= m.cfg().n_layers) throw std::invalid_argument("--layer is out of range");
    const int64_t H = m.layer_cfg(layer).n_heads, head = a.get_int("head", -1);
    if (head >= H) throw std::invalid_argument("--head is out of range");
    const int64_t k = a.get_int("top", head >= 0 ? 3 : 1);
    const std::string name = "blocks." + std::to_string(layer) + ".attn.hook_pattern";
    ActivationCache cache;
    m.run_with_cache(t, cache, [&name](const std::string& n) { return n == name; });
    const Tensor& P = cache.at(name);  // (H, T, T)

    std::printf("layer %lld attention: for each query, the top-%lld key(s) per head as position:token weight\n\n",
                (long long)layer, (long long)k);
    const int64_t first = a.has("pos") ? resolve_pos(a, t) : 0, last = a.has("pos") ? first + 1 : T;
    for (int64_t q = first; q < last; ++q) {
        std::printf("%3lld %5s |", (long long)q, tok(m, t[q]).c_str());
        for (int64_t h = head >= 0 ? head : 0; h < (head >= 0 ? head + 1 : H); ++h) {
            if (head < 0) std::printf(" H%lld", (long long)h);
            for (int64_t key : topk(P.ptr() + (h * T + q) * T, T, k))
                std::printf(" %lld:%s %.2f", (long long)key, m.vocab().empty() ? std::to_string(t[key]).c_str()
                                                                                : m.token_str(t[key]).c_str(),
                            P.data[(h * T + q) * T + key]);
            std::printf(" |");
        }
        std::printf("\n");
    }
    return 0;
}

// Reference sequences for mean ablation, from any mix of the --reference* options.
std::vector<Tokens> reference_tokens(const Args& a, const Model& m, int64_t T) {
    std::vector<Tokens> refs;
    auto split = [](const std::string& s) {
        std::vector<std::string> parts;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ';'))
            if (!item.empty()) parts.push_back(item);
        return parts;
    };
    if (a.has("reference"))
        for (const auto& s : split(a.get("reference"))) refs.push_back(parse_tokens(s));
    if (a.has("reference-text"))
        for (const auto& s : split(a.get("reference-text"))) refs.push_back(m.tokenize(s));
    if (a.has("reference-file")) {
        std::ifstream f(a.get("reference-file"));
        if (!f) throw std::runtime_error("cannot open " + a.get("reference-file"));
        std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            refs.push_back(m.vocab().empty() ? parse_tokens(line) : m.tokenize(line));
        }
    }
    if (a.has("reference-random")) {
        std::mt19937_64 rng(static_cast<uint64_t>(a.get_int("seed", 0)) + 1);
        std::uniform_int_distribution<int64_t> dist(0, m.cfg().vocab_size - 1);
        for (int64_t n = 0; n < a.get_int("reference-random", 0); ++n) {
            Tokens r(T);
            for (auto& x : r) x = dist(rng);
            refs.push_back(std::move(r));
        }
    }
    return refs;
}

int cmd_ablate(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const std::string mode = a.get("mode", "mean");
    if (mode != "mean" && mode != "zero") throw std::invalid_argument("--mode must be mean or zero");
    const int64_t T = static_cast<int64_t>(t.size());
    const std::vector<Tokens> refs = mode == "mean" ? reference_tokens(a, m, T) : std::vector<Tokens>();
    if (mode == "mean" && refs.empty())
        throw std::invalid_argument(
            "mean ablation averages each head over reference sequences, not over the input itself (that would "
            "copy later tokens into earlier positions); give --reference-file, --reference-text, --reference or "
            "--reference-random N, or use --mode zero");
    const Tensor d = head_ablation(m, t, mode == "mean" ? Ablation::Mean : Ablation::Zero, refs);
    std::printf("increase in next-token loss when each head is %s-ablated (base loss %.4f)\n", mode.c_str(),
                next_token_loss(m.forward(t), t));
    if (a.has("fp16"))
        std::printf("note: --fp16: weights are stored in fp16%s, which adds errors of around 1e-4 to each value\n",
                    use_gpu() && m.gpu() ? " and the GPU's multiplies take fp16 inputs" : "");
    if (mode == "mean") {
        const bool per_pos = std::all_of(refs.begin(), refs.end(), [T](const Tokens& r) { return (int64_t)r.size() == T; });
        std::printf("means over %zu reference sequences, %s\n", refs.size(),
                    per_pos ? "one per position" : "one over all positions (reference lengths differ from the input)");
    }
    std::printf("\n");
    print_matrix(d, layer_labels(m), head_labels(m));
    return 0;
}

int cmd_circuits(const Args& a, const Model& m) {
    const int64_t top = a.get_int("top", 10);
    const Composition c = composition_scores(m);
    const float base = composition_random_baseline(m);
    struct Entry { float score; std::string kind; size_t i, j; };
    std::vector<Entry> all;
    const size_t N = c.labels.size();
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            if (std::isnan(c.k.data[i * N + j])) continue;
            all.push_back({c.q.data[i * N + j], "Q", i, j});
            all.push_back({c.k.data[i * N + j], "K", i, j});
            all.push_back({c.v.data[i * N + j], "V", i, j});
        }
    std::sort(all.begin(), all.end(), [](const Entry& x, const Entry& y) { return x.score > y.score; });
    std::printf("composition scores (weights only; random-head baseline %.3f)\n", base);
    if (m.cfg().positional == "rope")
        std::printf("note: RoPE rotates q/k by position, so Q/K scores ignore the rotation\n");
    for (size_t n = 0; n < std::min<size_t>(all.size(), static_cast<size_t>(top)); ++n)
        std::printf("  %s-composition  %s -> %s   %.3f  (%.1fx baseline)\n", all[n].kind.c_str(),
                    c.labels[all[n].i].c_str(), c.labels[all[n].j].c_str(), all[n].score, all[n].score / base);
    if (all.empty()) std::printf("  (needs at least two layers)\n");

    const int64_t sampled = a.get_int("tokens-sampled", 1024);
    const Tensor copy = ov_copying_scores(m, sampled);
    std::printf("\nOV copying: fraction of tokens a head's direct path maps back to themselves "
                "(chance %.4f)\n", 1.0 / m.cfg().vocab_size);
    print_matrix(copy, layer_labels(m), head_labels(m));

    // Two-head OV paths (V-composition): writer OV then reader OV, for every earlier/later pair.
    const int64_t L = m.cfg().n_layers;
    if (L >= 2 && c.labels.size() <= 64) {
        const Tensor pair = ov_pair_copying_scores(m, sampled);
        std::vector<int64_t> first(L + 1, 0);  // index of each layer's first head in c.labels
        for (int64_t l = 0; l < L; ++l) first[l + 1] = first[l] + m.layer_cfg(l).n_heads;
        std::vector<std::pair<float, std::string>> paths;
        for (int64_t la = 0; la < L; ++la)
            for (int64_t lb = la + 1; lb < L; ++lb)
                for (int64_t i = first[la]; i < first[la + 1]; ++i)
                    for (int64_t j = first[lb]; j < first[lb + 1]; ++j)
                        paths.push_back({pair.data[i * N + j], c.labels[i] + " -> " + c.labels[j]});
        std::sort(paths.begin(), paths.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        std::printf("\ntwo-head OV path copying (writer OV, then reader OV), top %lld:\n", (long long)top);
        for (size_t n = 0; n < std::min<size_t>(paths.size(), static_cast<size_t>(top)); ++n)
            std::printf("  %s   %.3f\n", paths[n].second.c_str(), paths[n].first);
    } else if (L >= 2) {
        std::printf("\n(two-head OV paths skipped for models with more than 64 heads)\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        std::fputs(kUsage, argc < 2 ? stderr : stdout);
        return argc < 2 ? 2 : 0;
    }
    try {
        const Args a = parse_args(argc, argv);
        if (a.has("threads")) set_num_threads(static_cast<int>(a.get_int("threads", 0)));
        if (a.has("no-gpu")) set_use_gpu(false);
        if (a.has("fp16") && a.has("fp32")) throw std::invalid_argument("give --fp16 or --fp32, not both");
        const bool fp16 = a.has("fp16");
        set_gpu_fp16(fp16);
        const Model m = Model::load(a.model, fp16 ? WeightFormat::fp16 : WeightFormat::fp32);
        if (fp16) {
            const std::vector<std::string> kept = m.fp32_weights();
            if (!kept.empty()) {
                std::printf("note: %zu weight%s kept in float32 (values outside fp16's range):", kept.size(), kept.size() == 1 ? "" : "s");
                for (size_t i = 0; i < std::min<size_t>(kept.size(), 4); ++i) std::printf(" %s", kept[i].c_str());
                std::printf("%s\n", kept.size() > 4 ? " ..." : "");
            }
            if (a.command != "patch" && a.command != "ablate" && a.command != "info")
                std::printf("note: --fp16: weights are stored in fp16, so values can differ from float32 weights by about 1e-3 relative\n");
        }
        static const std::map<std::string, int (*)(const Args&, const Model&)> commands = {
            {"info", cmd_info}, {"run", cmd_run}, {"lens", cmd_lens}, {"dla", cmd_dla}, {"heads", cmd_heads},
            {"patch", cmd_patch}, {"ablate", cmd_ablate}, {"circuits", cmd_circuits}, {"attn", cmd_attn},
        };
        auto it = commands.find(a.command);
        if (it == commands.end()) throw std::invalid_argument("unknown command '" + a.command + "'");
        return it->second(a, m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "indra: %s\n\n%s", e.what(), kUsage);
        return 1;
    }
}
