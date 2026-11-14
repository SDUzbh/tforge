// Model weights, kept in float32 or, to halve a model's memory, in IEEE half precision (fp16).
#pragma once

#include <cstdint>
#include <memory>

#include "tfi/tensor.h"

namespace tfi {

// How a model keeps its weights in memory (Model::load). fp16 stores every 2-D weight (embeddings,
// attention and MLP projections, unembedding) in half precision, halving the model's memory. Each
// value is rounded to the nearest fp16 value, a relative error of at most 2^-11 (about 5e-4); a
// weight with a value outside fp16's range (|v| > 65504, or not finite) stays float32. 1-D weights
// (norm scales, biases) always stay float32, and all arithmetic is float32 either way.
enum class WeightFormat { fp32, fp16 };

// One weight: float32 (a Tensor), or half precision. Half-precision values are read through rows(),
// cols(), to_tensor() and the linear() / matmul() overloads below, which convert them to float32.
class Weight {
  public:
    Shape shape;

    Weight() = default;
    explicit Weight(Tensor t);  // float32
    // Half precision, rounded to nearest even. Throws if a value does not fit (fits_fp16).
    static Weight fp16(const Tensor& t);
    static bool fits_fp16(const Tensor& t);  // every value finite and within +-65504

    int64_t numel() const;
    int64_t dim() const { return static_cast<int64_t>(shape.size()); }
    bool is_fp16() const { return static_cast<bool>(half_); }
    int64_t bytes() const { return numel() * (is_fp16() ? 2 : 4); }  // memory the values take

    // The float32 values. Throws std::logic_error for a half-precision weight.
    const Tensor& f32() const;
    // The half-precision values (row-major, as IEEE fp16 bit patterns), or nullptr for float32.
    // The allocation starts on a page boundary and is f16_capacity() bytes, a whole number of pages,
    // so that the GPU can use it in place. f16_owner() keeps it alive.
    const uint16_t* f16() const { return half_.get(); }
    size_t f16_capacity() const { return half_capacity_; }
    std::shared_ptr<const uint16_t> f16_owner() const { return half_; }

    Tensor to_tensor() const;                       // every value, as float32
    Tensor rows(int64_t begin, int64_t end) const;  // W[begin:end] of a 2-D weight, as float32
    Tensor cols(int64_t begin, int64_t end) const;  // W[:, begin:end] of a 2-D weight, as float32
    Tensor row(int64_t i) const;                    // W[i] of a 2-D weight, as a 1-D float32 tensor
    // W[begin:end] of a 2-D weight into dst ((end - begin) * columns floats), as float32.
    void read_rows(int64_t begin, int64_t end, float* dst) const;

  private:
    Tensor f32_;
    std::shared_ptr<const uint16_t> half_;
    size_t half_capacity_ = 0;
};

// y = x @ W^T + b (W (out, in), as linear() above) with a model weight. A half-precision W is
// converted to float32 in blocks of rows as the multiply goes, so the full float32 copy never exists.
Tensor linear(const Tensor& x, const Weight& W, const Weight* b = nullptr);
// A(m, n) @ W(n, k) for a 2-D weight W: e.g. linear()'s input gradient dy @ W.
Tensor matmul(const Tensor& A, const Weight& W);

}  // namespace tfi
