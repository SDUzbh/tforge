// tfi: command-line reverse engineering for models exported with tforge.export.export_model.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

const char* kUsage = R"(usage: tfi <command> <model.safetensors> [options]

Commands
  info       config, parameter count and hook point names
  run        next-token predictions                           --tokens
  lens       logit lens over the residual stream               --tokens [--pos] [--target] [--top]
  dla        direct logit attribution per head / MLP           --tokens --target [--other] [--pos] [--top]
  heads      attention-head pattern scores on repeated text    [--seg 24] [--samples 8] [--seed 0]
  patch      activation patching, clean -> corrupted           --clean --corrupt --target [--other] [--pos]
                                                               [--site resid_pre|resid_mid|resid_post|attn_out|mlp_out|heads]
  ablate     loss increase when each head is ablated           --tokens [--mode mean|zero]
  circuits   weight-only: composition scores, OV copying       [--top 10] [--tokens-sampled 1024]

Token input
  --tokens 5,17,3,...          explicit token ids
  --repeat-random L [--seed S] a random segment of length L, repeated twice (for induction)
  --pos P                      position to analyse (default: last; negative counts from the end)
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
        if (i + 1 >= argc) throw std::invalid_argument("--" + k + " needs a value");
        a.opts[k] = argv[++i];
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
    if (a.has(key)) return parse_tokens(a.get(key));
    if (key == "tokens" && a.has("repeat-random")) return repeat_random(m, a.get_int("repeat-random", 8), a.get_int("seed", 0));
    throw std::invalid_argument("give --" + key + (key == "tokens" ? " or --repeat-random" : ""));
}

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

void print_matrix(const Tensor& t, const std::vector<std::string>& rows, const std::vector<std::string>& cols,
                  const char* fmt = "%7.3f") {
    std::printf("%8s", "");
    for (const auto& c : cols) std::printf("%7s", c.c_str());
    std::printf("\n");
    for (size_t r = 0; r < rows.size(); ++r) {
        std::printf("%8s", rows[r].c_str());
        for (size_t c = 0; c < cols.size(); ++c) {
            const float v = t.data[r * cols.size() + c];
            if (std::isnan(v)) std::printf("%7s", ".");
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
        std::printf("%5lld %7lld  ", (long long)p, (long long)t[p]);
        for (int64_t id : topk(probs.ptr(), V, k)) std::printf("  %6lld (%.3f)", (long long)id, probs.data[id]);
        if (p + 1 < static_cast<int64_t>(t.size()))
            std::printf("   | actual next %lld: p=%.3f", (long long)t[p + 1], probs.data[t[p + 1]]);
        std::printf("\n");
    }
    if (t.size() > 1) std::printf("mean next-token loss %.4f\n", next_token_loss(logits, t));
    return 0;
}

int cmd_lens(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const int64_t pos = resolve_pos(a, t);
    const int64_t target = a.get_int("target", pos + 1 < (int64_t)t.size() ? t[pos + 1] : -1);
    const auto lens = logit_lens(m, t, pos, target, a.get_int("top", 5));
    std::printf("logit lens at position %lld (token %lld)", (long long)pos, (long long)t[pos]);
    if (target >= 0) std::printf(", tracking token %lld", (long long)target);
    std::printf("\n\n%-9s %-10s %-8s  top predictions (prob)\n", "stage", target >= 0 ? "p(target)" : "", target >= 0 ? "rank" : "");
    for (const auto& e : lens) {
        std::printf("%-9s ", e.stage.c_str());
        if (target >= 0) std::printf("%-10.4f %-8lld ", e.target_prob, (long long)e.target_rank);
        else std::printf("%-10s %-8s ", "", "");
        for (size_t i = 0; i < e.top_tokens.size(); ++i)
            std::printf(" %lld (%.3f)", (long long)e.top_tokens[i], e.top_probs[i]);
        std::printf("\n");
    }
    return 0;
}

int cmd_dla(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const int64_t pos = resolve_pos(a, t);
    const int64_t target = std::stoll(a.require("target")), other = a.get_int("other", -1);
    const DLAResult r = direct_logit_attribution(m, t, pos, target, other);
    std::vector<size_t> order(r.values.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return std::fabs(r.values[x]) > std::fabs(r.values[y]); });
    const size_t top = std::min<size_t>(order.size(), static_cast<size_t>(a.get_int("top", 15)));
    std::printf("direct logit attribution at position %lld: logit[%lld]%s\n", (long long)pos, (long long)target,
                other >= 0 ? (" - logit[" + std::to_string(other) + "]").c_str() : "");
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
    const int64_t target = std::stoll(a.require("target")), other = a.get_int("other", -1);
    const std::string site = a.get("site", "resid_pre");
    const PatchResult r = site == "heads" ? patch_heads(m, clean, corrupt, pos, target, other)
                                          : patch_residual(m, clean, corrupt, site, pos, target, other);
    std::printf("activation patching (%s), metric at position %lld: %s\n", site.c_str(), (long long)pos,
                other >= 0 ? "logit difference" : "log p(target)");
    std::printf("clean %.4f   corrupted %.4f   (1 = clean behaviour restored, 0 = no effect)\n\n", r.clean_metric,
                r.corrupt_metric);
    print_matrix(r.values, r.row_labels, r.col_labels);
    return 0;
}

int cmd_ablate(const Args& a, const Model& m) {
    const Tokens t = input_tokens(a, m);
    const std::string mode = a.get("mode", "mean");
    if (mode != "mean" && mode != "zero") throw std::invalid_argument("--mode must be mean or zero");
    const Tensor d = head_ablation(m, t, mode == "mean" ? Ablation::Mean : Ablation::Zero);
    std::printf("increase in next-token loss when each head is %s-ablated (base loss %.4f)\n\n", mode.c_str(),
                next_token_loss(m.forward(t), t));
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
    auto heads = [&m](int64_t layer) { return m.layer_cfg(layer).n_heads; };
    auto label = [](int64_t layer, int64_t head) { return "L" + std::to_string(layer) + "H" + std::to_string(head); };
    if (L >= 2 && c.labels.size() <= 64) {
        std::vector<std::pair<float, std::string>> paths;
        for (int64_t la = 0; la < L; ++la)
            for (int64_t lb = la + 1; lb < L; ++lb)
                for (int64_t ha = 0; ha < heads(la); ++ha)
                    for (int64_t hb = 0; hb < heads(lb); ++hb)
                        paths.push_back({ov_path_copying(m, {{la, ha}, {lb, hb}}, sampled),
                                         label(la, ha) + " -> " + label(lb, hb)});
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
        const Model m = Model::load(a.model);
        static const std::map<std::string, int (*)(const Args&, const Model&)> commands = {
            {"info", cmd_info}, {"run", cmd_run}, {"lens", cmd_lens}, {"dla", cmd_dla}, {"heads", cmd_heads},
            {"patch", cmd_patch}, {"ablate", cmd_ablate}, {"circuits", cmd_circuits},
        };
        auto it = commands.find(a.command);
        if (it == commands.end()) throw std::invalid_argument("unknown command '" + a.command + "'");
        return it->second(a, m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tfi: %s\n\n%s", e.what(), kUsage);
        return 1;
    }
}
