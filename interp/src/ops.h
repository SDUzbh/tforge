// Internal building blocks shared by the forward pass (model.cpp) and the backward pass (grad.cpp).
// Not part of the public API.
#pragma once

#include <string>
#include <vector>

#include "tfi/config.h"
#include "tfi/tensor.h"

namespace tfi {
namespace detail {

void activate_(Tensor& t, const std::string& act);
// d act(x) / dx, elementwise, for the pre-activation x.
void activation_grad_(Tensor& x, const std::string& act);
Tensor split_heads(const Tensor& x, int64_t H, int64_t D);  // (T, H*D) -> (H, T, D)
Tensor merge_heads(const Tensor& x);                        // (H, T, D) -> (T, H*D)
// Rotate (H, T, D) in place at positions 0..T-1 (tforge's "rotate half" RoPE). `inverse` rotates
// back, which is also the transpose: the backward pass of RoPE.
void apply_rope_(Tensor& x, const ModelConfig& c, bool inverse = false);
std::vector<float> alibi_slopes(int64_t n);

}  // namespace detail
}  // namespace tfi
