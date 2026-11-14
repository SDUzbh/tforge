// Internal building blocks shared by the forward pass (model.cpp) and the backward pass (grad.cpp).
// Not part of the public API.
#pragma once

#include <string>
#include <vector>

#include "tfi/config.h"
#include "tfi/tensor.h"

namespace tfi {

class Model;

namespace detail {

class Gpu;  // gpu.h (macOS)

void activate_(Tensor& t, const std::string& act);
// d act(x) / dx, elementwise, for the pre-activation x.
void activation_grad_(Tensor& x, const std::string& act);
Tensor split_heads(const Tensor& x, int64_t H, int64_t D);  // (T, H*D) -> (H, T, D)
Tensor merge_heads(const Tensor& x);                        // (H, T, D) -> (T, H*D)
// Rotate (H, T, D) in place at positions 0..T-1 (tforge's "rotate half" RoPE), or at positions[t]
// for row t when given. `inverse` rotates back, which is also the transpose: the backward pass.
void apply_rope_(Tensor& x, const ModelConfig& c, bool inverse = false, const std::vector<int64_t>* positions = nullptr);
std::vector<float> alibi_slopes(int64_t n);
// C(m,n) = A(m,k) @ op(B) with row strides lda/ldb/ldc; op(B) = B (k,n) or, with trans_b, B(n,k)^T.
// With accumulate, C += A @ op(B).
void gemm(bool trans_b, int64_t m, int64_t n, int64_t k, const float* A, int64_t lda, const float* B, int64_t ldb,
          float* C, int64_t ldc, bool accumulate = false);

// Backward pass of Model::norm (prefix's weight, config c) over the last dimension: dL/dx given its
// input x and dL/dy (grad.cpp).
Tensor norm_backward(const Model& m, const std::string& prefix, const ModelConfig& c, const Tensor& x, const Tensor& dy);

// The part of attention's backward pass between o_proj and the q/k/v projections, for one sequence
// at positions 0..T-1 (grad.cpp). From dz (T, H*D) and the forward pass's hook_q, hook_k, hook_v
// ((H or Hkv, T, D), after qk_norm and RoPE) and hook_pattern (H, T, T): the gradients at those hook
// points, and at the outputs of the q, k and v projections, (T, H*D) and (T, Hkv*D). q0 and k0 are
// the projections split into heads before qk_norm, needed only with qk_norm.
struct AttentionGrads {
    Tensor pattern, q, k, v;        // at the hook points
    Tensor q_proj, k_proj, v_proj;  // at the projections' outputs
};
AttentionGrads attention_backward(const Model& m, int64_t layer, const Tensor& dz, const Tensor& q, const Tensor& k,
                                  const Tensor& v, const Tensor& P, const Tensor* q0, const Tensor* k0);

// Rows of several sequences stacked into one (R, d) activation, so that many patched passes run
// through the blocks together and every linear layer is one large matrix multiply
// (Model::forward_rows). Segment s is rows [offset[s], offset[s] + len[s]) of the stack: positions
// [start[s], start[s] + len[s]) of its sequence. Positions before start[s] are not recomputed:
// attention takes their keys and values from prefix_k / prefix_v[layer] ((Hkv, T, D), e.g. a cached
// run's hook_k / hook_v), which must hold what those positions would have produced.
// For tests: rows per batched pass (residual-stream patching; head patching, ablation and
// reference means); 0 restores the defaults.
void set_patch_batch_rows(int64_t residual, int64_t heads);
// For tests (macOS): false runs a batched pass's GPU work one linear layer at a time, the residual
// stream returning to the CPU after each, instead of keeping it on the GPU (Model::run_rows).
void set_gpu_resident(bool on);

struct RowBatch {
    std::vector<int64_t> start, len, offset;
    std::vector<const Tensor*> prefix_k, prefix_v;  // per layer; unused when every start is 0
    Gpu* gpu = nullptr;  // macOS: if set and rows() >= gpu_min_rows(), the blocks run on the GPU (Model::run_rows)
    bool fp16 = false;   // ... with fp16 multiply inputs (Gpu::linear, Gpu::Seq)
    int64_t rows() const { return offset.empty() ? 0 : offset.back() + len.back(); }
    void add(int64_t start_pos, int64_t n) {
        offset.push_back(rows());
        start.push_back(start_pos);
        len.push_back(n);
    }
};

}  // namespace detail
}  // namespace tfi
