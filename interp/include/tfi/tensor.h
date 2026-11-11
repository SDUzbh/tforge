// Minimal dense float32 tensor and the linear algebra the interpretability tools need.
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace tfi {

using Shape = std::vector<int64_t>;

std::string shape_str(const Shape& s);
int64_t shape_numel(const Shape& s);

// Row-major float32 tensor that owns its data. Copies are deep.
struct Tensor {
    Shape shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(Shape s, float fill = 0.f);
    Tensor(Shape s, std::vector<float> d);

    int64_t numel() const { return static_cast<int64_t>(data.size()); }
    int64_t dim() const { return static_cast<int64_t>(shape.size()); }
    int64_t size(int64_t i) const;  // negative i counts from the end
    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }

    float& at(std::initializer_list<int64_t> idx);
    float at(std::initializer_list<int64_t> idx) const;
    Tensor reshape(Shape s) const;
};

// C(m,n) = A(m,k) @ B(k,n), or A(m,k) @ B(n,k)^T when transpose_b is set. 2-D only.
Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_b = false);
// y = x @ W^T + b with PyTorch Linear layout W(out, in); x is (..., in).
Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b = nullptr);
Tensor transpose2d(const Tensor& A);

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float s);
void add_(Tensor& a, const Tensor& b);

Tensor rows(const Tensor& A, int64_t begin, int64_t end);  // A[begin:end] of a 2-D tensor
Tensor cols(const Tensor& A, int64_t begin, int64_t end);  // A[:, begin:end] of a 2-D tensor
Tensor row(const Tensor& A, int64_t i);                     // A[i] as 1-D

float dot(const float* a, const float* b, int64_t n);
float frobenius(const Tensor& a);
float max_abs_diff(const Tensor& a, const Tensor& b);
float max_abs(const Tensor& a);

void softmax_(float* v, int64_t n);
float log_softmax_at(const float* v, int64_t n, int64_t idx);
std::vector<int64_t> topk(const float* v, int64_t n, int64_t k);  // indices, largest first
int64_t rank_of(const float* v, int64_t n, int64_t idx);          // 0 = largest

}  // namespace tfi
