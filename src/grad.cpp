// Backward pass: gradients of a scalar function of the logits with respect to every hook point.
//
// The forward pass is Model::run_with_cache, so activations are exactly forward()'s. The backward
// pass reads them from the cache and recomputes the few intermediates that are not hook points
// (pre-norm values, pre-softcap scores, the MLP's up projection). Only activation gradients are
// computed, no weight gradients, so it costs about as much as one forward pass.
#include <cmath>
#include <stdexcept>

#include "ops.h"
#include "vmath.h"
#include "tfi/model.h"

namespace tfi {

using detail::apply_rope_;
using detail::merge_heads;

namespace {

std::string lp(int64_t layer) { return "blocks." + std::to_string(layer) + "."; }

// dL/dx for y = x W^T (+ b), W (out, in): dy (..., out) -> (..., in).
Tensor linear_backward(const Tensor& dy, const Tensor& W) {
    const int64_t out = W.shape[0], in = W.shape[1];
    Shape s = dy.shape;
    s.back() = in;
    Tensor dx = matmul(dy.reshape({-1, out}), W);
    return std::move(dx).reshape(s);
}

}  // namespace

namespace detail {

Tensor norm_backward(const Model& m, const std::string& prefix, const ModelConfig& c, const Tensor& x, const Tensor& dy) {
    if (c.norm == "none") return dy;
    const Tensor& w = m.weight(prefix + ".weight");
    const int64_t d = x.size(-1), n = x.numel() / d;
    const float eps = static_cast<float>(c.norm_eps);
    Tensor dx(x.shape);
    std::vector<float> g(d);
    for (int64_t r = 0; r < n; ++r) {
        const float* xi = x.ptr() + r * d;
        const float* dyi = dy.ptr() + r * d;
        float* dxi = dx.ptr() + r * d;
        for (int64_t j = 0; j < d; ++j) g[j] = dyi[j] * w.data[j];
        if (c.norm == "layernorm") {
            // y = xhat * w + b, xhat = (x - mean) * inv:  dx = inv * (g - mean(g) - xhat * mean(g * xhat))
            double mean = 0, var = 0;
            for (int64_t j = 0; j < d; ++j) mean += xi[j];
            mean /= d;
            for (int64_t j = 0; j < d; ++j) var += (xi[j] - mean) * (xi[j] - mean);
            var /= d;
            const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
            double gm = 0, gx = 0;
            for (int64_t j = 0; j < d; ++j) {
                const double xhat = (xi[j] - mean) * inv;
                gm += g[j];
                gx += g[j] * xhat;
            }
            gm /= d;
            gx /= d;
            for (int64_t j = 0; j < d; ++j) {
                const double xhat = (xi[j] - mean) * inv;
                dxi[j] = static_cast<float>(inv * (g[j] - gm - xhat * gx));
            }
        } else {  // rmsnorm: y = x * inv * w, inv = (mean(x^2) + eps)^-1/2:  dx = inv * g - x * inv^3 * mean(g * x)
            double ms = 0, gx = 0;
            for (int64_t j = 0; j < d; ++j) {
                ms += double(xi[j]) * xi[j];
                gx += double(g[j]) * xi[j];
            }
            const double inv = 1.0 / std::sqrt(static_cast<double>(static_cast<float>(ms / d) + eps));
            gx /= d;
            for (int64_t j = 0; j < d; ++j) dxi[j] = static_cast<float>(inv * g[j] - xi[j] * inv * inv * inv * gx);
        }
    }
    return dx;
}

// Add a (T, D) block into head g of an (Hg, T, D) tensor.
static void add_from(Tensor& dst, int64_t g, const Tensor& src) {
    float* d = dst.ptr() + g * src.numel();
    for (int64_t i = 0; i < src.numel(); ++i) d[i] += src.data[i];
}

AttentionGrads attention_backward(const Model& m, int64_t layer, const Tensor& dz, const Tensor& q, const Tensor& k,
                                  const Tensor& v, const Tensor& P, const Tensor* q0, const Tensor* k0) {
    const ModelConfig& c = m.layer_cfg(layer);
    const std::string p = lp(layer) + "attn.";
    const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim(), rep = H / Hkv, T = q.shape[1];
    const float scale = static_cast<float>(c.attn_scale ? *c.attn_scale : 1.0 / std::sqrt(double(D)));
    Tensor dq({H, T, D}), dk({Hkv, T, D}), dv({Hkv, T, D}), dP({H, T, T});
    for (int64_t h = 0; h < H; ++h) {
        const int64_t g = h / rep;
        Tensor dzh({T, D});
        for (int64_t t = 0; t < T; ++t)
            std::copy(dz.ptr() + t * H * D + h * D, dz.ptr() + t * H * D + (h + 1) * D, dzh.ptr() + t * D);
        Tensor qh({T, D}, std::vector<float>(q.data.begin() + h * T * D, q.data.begin() + (h + 1) * T * D));
        Tensor kh({T, D}, std::vector<float>(k.data.begin() + g * T * D, k.data.begin() + (g + 1) * T * D));
        Tensor vh({T, D}, std::vector<float>(v.data.begin() + g * T * D, v.data.begin() + (g + 1) * T * D));
        Tensor Ph({T, T}, std::vector<float>(P.data.begin() + h * T * T, P.data.begin() + (h + 1) * T * T));

        Tensor dPh = matmul(dzh, vh, /*transpose_b=*/true);  // dz_h v^T
        std::copy(dPh.data.begin(), dPh.data.end(), dP.ptr() + h * T * T);
        add_from(dv, g, matmul(transpose2d(Ph), dzh));  // P^T dz_h

        // Softmax backward, then the softcap's tanh and the scale. Masked entries have P = 0.
        Tensor dS({T, T});
        Tensor raw;  // q k^T, needed only for the softcap derivative
        if (c.attn_logit_softcap) raw = matmul(qh, kh, true);
        for (int64_t i = 0; i < T; ++i) {
            const float* pr = Ph.ptr() + i * T;
            const float* dpr = dPh.ptr() + i * T;
            double s = 0;
            for (int64_t j = 0; j < T; ++j) s += double(pr[j]) * dpr[j];
            for (int64_t j = 0; j < T; ++j) {
                float ds = pr[j] * static_cast<float>(dpr[j] - s);
                if (c.attn_logit_softcap && pr[j] != 0.f) {
                    const float cap = static_cast<float>(*c.attn_logit_softcap);
                    const float th = vm::tanh(raw.data[i * T + j] * scale / cap);
                    ds *= 1.f - th * th;
                }
                dS.data[i * T + j] = ds * scale;
            }
        }
        Tensor dqh = matmul(dS, kh);               // dS k
        Tensor dkh = matmul(transpose2d(dS), qh);  // dS^T q
        std::copy(dqh.data.begin(), dqh.data.end(), dq.ptr() + h * T * D);
        add_from(dk, g, dkh);
    }
    AttentionGrads r;
    r.pattern = std::move(dP);
    r.q = dq;
    r.k = dk;
    r.v = dv;
    if (c.positional == "rope") {
        apply_rope_(dq, c, /*inverse=*/true);
        apply_rope_(dk, c, /*inverse=*/true);
    }
    if (c.qk_norm) {
        if (!q0 || !k0) throw std::logic_error("attention_backward: qk_norm needs the projections before the norm");
        ModelConfig qc = c;
        if (qc.norm == "none") qc.norm = "rmsnorm";
        dq = norm_backward(m, p + "q_norm", qc, *q0, dq);
        dk = norm_backward(m, p + "k_norm", qc, *k0, dk);
    }
    r.q_proj = merge_heads(dq);
    r.k_proj = merge_heads(dk);
    r.v_proj = merge_heads(dv);
    return r;
}

}  // namespace detail

using detail::norm_backward;

namespace {

struct Backward {
    const Model& m;
    const ActivationCache& a;
    ActivationCache& grads;
    const NameFilter& filter;

    const Tensor& act(const std::string& name) const { return a.at(name); }
    void report(const std::string& name, const Tensor& g) const {
        if (!filter || filter(name)) grads[name] = g;
    }

    // dL/d(attention input) from dL/d(attention output before any sandwich norm), (T, d_model).
    Tensor attention(int64_t layer, const Tensor& dout) const {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = lp(layer) + "attn.";
        const Tensor& q = act(p + "hook_q");  // (H, T, D)
        const int64_t H = c.n_heads, Hkv = c.kv_heads(), D = c.head_dim(), T = q.shape[1];

        Tensor dz = linear_backward(dout, m.weight(p + "o_proj.weight"));  // (T, H*D)
        report(p + "hook_z", dz.reshape({T, H, D}));
        Tensor q0, k0;  // the projections before qk_norm
        const Tensor& x = act(lp(layer) + "hook_attn_in");
        if (c.qk_norm) {
            q0 = detail::split_heads(linear(x, m.weight(p + "q_proj.weight"), m.weight_or_null(p + "q_proj.bias")), H, D);
            k0 = detail::split_heads(linear(x, m.weight(p + "k_proj.weight"), m.weight_or_null(p + "k_proj.bias")), Hkv, D);
        }
        const detail::AttentionGrads g = detail::attention_backward(
            m, layer, dz, q, act(p + "hook_k"), act(p + "hook_v"), act(p + "hook_pattern"), &q0, &k0);
        report(p + "hook_pattern", g.pattern);
        report(p + "hook_q", g.q);
        report(p + "hook_k", g.k);
        report(p + "hook_v", g.v);
        Tensor dx = linear_backward(g.q_proj, m.weight(p + "q_proj.weight"));
        add_(dx, linear_backward(g.k_proj, m.weight(p + "k_proj.weight")));
        add_(dx, linear_backward(g.v_proj, m.weight(p + "v_proj.weight")));
        return dx;
    }

    // dL/d(MLP input) from dL/d(MLP output before any sandwich norm).
    Tensor mlp(int64_t layer, const Tensor& dout) const {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = lp(layer) + "mlp.";
        const std::string fn = c.mlp_activation();
        Tensor dpost = linear_backward(dout, m.weight(p + "down.weight"));
        report(p + "hook_post", dpost);
        const Tensor& pre = act(p + "hook_pre");
        Tensor dact = pre;
        detail::activation_grad_(dact, fn);
        if (c.mlp == "mlp") {
            Tensor dpre = mul(dpost, dact);
            report(p + "hook_pre", dpre);
            return linear_backward(dpre, m.weight(p + "up.weight"));
        }
        // post = act(gate(x)) * up(x)
        const Tensor& x = act(lp(layer) + "hook_mlp_in");
        const Tensor up = linear(x, m.weight(p + "up.weight"), m.weight_or_null(p + "up.bias"));
        Tensor activated = pre;
        detail::activate_(activated, fn);
        Tensor dpre = mul(mul(dpost, up), dact);
        report(p + "hook_pre", dpre);
        Tensor dx = linear_backward(dpre, m.weight(p + "gate.weight"));
        add_(dx, linear_backward(mul(dpost, activated), m.weight(p + "up.weight")));
        return dx;
    }

    // Sandwich norms sit after each sublayer: map dL/d(normed output) back to the raw output.
    Tensor sandwich_attn(int64_t layer, const Tensor& g) const {
        const ModelConfig& c = m.layer_cfg(layer);
        if (c.norm_position != "sandwich") return g;
        const std::string p = lp(layer) + "attn.";
        const Tensor& z = act(p + "hook_z");
        const int64_t T = z.shape[0];
        const Tensor raw = linear(z.reshape({T, -1}), m.weight(p + "o_proj.weight"), m.weight_or_null(p + "o_proj.bias"));
        return norm_backward(m, lp(layer) + "ln1_post", c, raw, g);
    }
    Tensor sandwich_mlp(int64_t layer, const Tensor& g) const {
        const ModelConfig& c = m.layer_cfg(layer);
        if (c.norm_position != "sandwich") return g;
        const std::string p = lp(layer) + "mlp.";
        const Tensor raw = linear(act(p + "hook_post"), m.weight(p + "down.weight"), m.weight_or_null(p + "down.bias"));
        return norm_backward(m, lp(layer) + "ln2_post", c, raw, g);
    }

    // dL/d(hook_resid_pre) from dL/d(hook_resid_post), mirroring Model::block.
    Tensor block(int64_t layer, const Tensor& g_post) const {
        const ModelConfig& c = m.layer_cfg(layer);
        const std::string p = lp(layer);
        const bool post = c.norm_position == "post";
        const Tensor& x = act(p + "hook_resid_pre");
        report(p + "hook_resid_post", g_post);

        auto mlp_branch = [&](const Tensor& g_out) {  // dL/d(hook_mlp_in) from dL/d(hook_mlp_out)
            report(p + "hook_mlp_out", g_out);
            Tensor d_in = mlp(layer, sandwich_mlp(layer, g_out));
            report(p + "hook_mlp_in", d_in);
            return d_in;
        };
        auto attn_branch = [&](const Tensor& g_out) {  // dL/d(hook_attn_in) from dL/d(hook_attn_out)
            report(p + "hook_attn_out", g_out);
            Tensor d_in = attention(layer, sandwich_attn(layer, g_out));
            report(p + "hook_attn_in", d_in);
            return d_in;
        };

        if (c.parallel_residual) {  // mid = x + attn(ln1 x); out = mid + mlp(ln2 x)
            Tensor g_x = g_post;  // copy: g_mid is g_post, and g_x accumulates every path
            if (c.has_mlp()) add_(g_x, norm_backward(m, p + "ln2", c, x, mlp_branch(g_post)));
            report(p + "hook_resid_mid", g_post);
            const Tensor d_attn_in = attn_branch(g_post);
            add_(g_x, post ? d_attn_in : norm_backward(m, p + "ln1", c, x, d_attn_in));
            report(p + "hook_resid_pre", g_x);
            return g_x;
        }

        const Tensor& mid = act(p + "hook_resid_mid");
        Tensor g_mid = g_post;
        if (c.has_mlp()) {
            Tensor g_raw = g_post;  // gradient of mid + mlp_out
            if (post) g_raw = norm_backward(m, p + "ln2", c, add(mid, act(p + "hook_mlp_out")), g_post);
            const Tensor d_mlp_in = mlp_branch(g_raw);
            g_mid = g_raw;
            add_(g_mid, post ? d_mlp_in : norm_backward(m, p + "ln2", c, mid, d_mlp_in));
        }
        report(p + "hook_resid_mid", g_mid);
        Tensor g_x = g_mid;  // gradient of x + attn_out
        if (post) g_x = norm_backward(m, p + "ln1", c, add(x, act(p + "hook_attn_out")), g_mid);
        const Tensor d_attn_in = attn_branch(g_x);
        add_(g_x, post ? d_attn_in : norm_backward(m, p + "ln1", c, x, d_attn_in));
        report(p + "hook_resid_pre", g_x);
        return g_x;
    }
};

}  // namespace

Tensor Model::run_with_gradients(const Tokens& tokens, const std::function<Tensor(const Tensor& logits)>& dlogits_fn,
                                 ActivationCache& grads, NameFilter filter, ActivationCache* acts,
                                 const Hooks* hooks) const {
    ActivationCache cache;
    Tensor logits = run_with_cache(tokens, cache, nullptr, hooks);
    Tensor dl = dlogits_fn(logits);
    if (dl.shape != logits.shape)
        throw std::invalid_argument("dlogits must have the logits' shape " + shape_str(logits.shape) + ", got " +
                                    shape_str(dl.shape));
    if (cfg_.final_logit_softcap) {  // logits = cap * tanh(raw / cap): d/draw = 1 - (logits / cap)^2
        const float cap = static_cast<float>(*cfg_.final_logit_softcap);
        for (int64_t i = 0; i < dl.numel(); ++i) {
            const float th = logits.data[i] / cap;
            dl.data[i] *= 1.f - th * th;
        }
    }
    grads.clear();
    const Backward bw{*this, cache, grads, filter};
    Tensor g = matmul(dl, unembed());  // d hook_final
    bw.report("hook_final", g);
    const int64_t L = cfg_.n_layers;
    if (cfg_.norm_position != "post") g = norm_backward(*this, "ln_final", cfg_, cache.at(lp(L - 1) + "hook_resid_post"), g);
    for (int64_t i = L - 1; i >= 0; --i) g = bw.block(i, g);
    bw.report("hook_embed", g);  // resid_pre of block 0 = embed (+ pos_embed)
    if (cfg_.embedding_positional()) bw.report("hook_pos_embed", g);
    if (acts) *acts = std::move(cache);
    return logits;
}

}  // namespace tfi
