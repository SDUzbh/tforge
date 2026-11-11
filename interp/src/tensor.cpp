#include "tfi/tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef TFI_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace tfi {

std::string shape_str(const Shape& s) {
    std::ostringstream o;
    o << "(";
    for (size_t i = 0; i < s.size(); ++i) o << (i ? ", " : "") << s[i];
    o << ")";
    return o.str();
}

int64_t shape_numel(const Shape& s) {
    return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>());
}

Tensor::Tensor(Shape s, float fill) : shape(std::move(s)), data(shape_numel(shape), fill) {}

Tensor::Tensor(Shape s, std::vector<float> d) : shape(std::move(s)), data(std::move(d)) {
    if (shape_numel(shape) != numel())
        throw std::invalid_argument("tensor data size " + std::to_string(numel()) + " does not match shape " +
                                    shape_str(shape));
}

int64_t Tensor::size(int64_t i) const {
    if (i < 0) i += dim();
    if (i < 0 || i >= dim()) throw std::out_of_range("dim " + std::to_string(i) + " for shape " + shape_str(shape));
    return shape[i];
}

static int64_t flat_index(const Shape& shape, std::initializer_list<int64_t> idx) {
    if (static_cast<int64_t>(idx.size()) != static_cast<int64_t>(shape.size()))
        throw std::out_of_range("index rank does not match shape " + shape_str(shape));
    int64_t flat = 0, d = 0;
    for (int64_t i : idx) {
        if (i < 0 || i >= shape[d]) throw std::out_of_range("index out of range for shape " + shape_str(shape));
        flat = flat * shape[d] + i;
        ++d;
    }
    return flat;
}

float& Tensor::at(std::initializer_list<int64_t> idx) { return data[flat_index(shape, idx)]; }
float Tensor::at(std::initializer_list<int64_t> idx) const { return data[flat_index(shape, idx)]; }

// Resolve a -1 in `to` and check it holds the same number of elements as `from`.
static Shape reshape_target(const Shape& from, Shape to) {
    const int64_t n = shape_numel(from);
    int64_t known = 1, infer = -1;
    for (size_t i = 0; i < to.size(); ++i) {
        if (to[i] == -1) infer = static_cast<int64_t>(i);
        else known *= to[i];
    }
    if (infer >= 0) to[infer] = known ? n / known : 0;
    if (shape_numel(to) != n) throw std::invalid_argument("cannot reshape " + shape_str(from) + " to " + shape_str(to));
    return to;
}

Tensor Tensor::reshape(Shape s) const& { return Tensor(reshape_target(shape, std::move(s)), data); }

Tensor Tensor::reshape(Shape s) && {
    Shape target = reshape_target(shape, std::move(s));  // before `data` is moved out
    return Tensor(std::move(target), std::move(data));
}

// C(m,n) = A(m,k) @ op(B), op(B) = B(k,n) or B(n,k)^T.
static void gemm(bool trans_b, int64_t m, int64_t n, int64_t k, const float* A, const float* B, float* C) {
#ifdef TFI_USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans, (int)m, (int)n, (int)k, 1.0f, A,
                (int)k, B, trans_b ? (int)k : (int)n, 0.0f, C, (int)n);
#else
    if (trans_b) {
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j) C[i * n + j] = dot(A + i * k, B + j * k, k);
    } else {
        std::fill(C, C + m * n, 0.f);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t p = 0; p < k; ++p) {
                const float a = A[i * k + p];
                const float* b = B + p * n;
                float* c = C + i * n;
                for (int64_t j = 0; j < n; ++j) c[j] += a * b[j];
            }
    }
#endif
}

Tensor matmul(const Tensor& A, const Tensor& B, bool transpose_b) {
    if (A.dim() != 2 || B.dim() != 2) throw std::invalid_argument("matmul expects 2-D tensors");
    const int64_t m = A.shape[0], k = A.shape[1];
    const int64_t kb = transpose_b ? B.shape[1] : B.shape[0];
    const int64_t n = transpose_b ? B.shape[0] : B.shape[1];
    if (k != kb) throw std::invalid_argument("matmul shape mismatch " + shape_str(A.shape) + " @ " + shape_str(B.shape) +
                                             (transpose_b ? "^T" : ""));
    Tensor C({m, n});
    if (m && n) gemm(transpose_b, m, n, k, A.ptr(), B.ptr(), C.ptr());
    return C;
}

Tensor linear(const Tensor& x, const Tensor& W, const Tensor* b) {
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    const int64_t in = W.shape[1], out = W.shape[0];
    if (x.size(-1) != in) throw std::invalid_argument("linear: input " + shape_str(x.shape) + " vs weight " + shape_str(W.shape));
    // Treat x as (rows, in) and write straight into a tensor of the final shape: no copies.
    Shape s(x.shape.begin(), x.shape.end() - 1);  // x has at least one dim: x.size(-1) checked it
    s.push_back(out);
    Tensor y(std::move(s));
    const int64_t n_rows = in ? x.numel() / in : 0;
    if (n_rows && out) gemm(/*trans_b=*/true, n_rows, out, in, x.ptr(), W.ptr(), y.ptr());
    if (b) {
        for (int64_t r = 0; r < n_rows; ++r)
            for (int64_t j = 0; j < out; ++j) y.data[r * out + j] += b->data[j];
    }
    return y;
}

Tensor transpose2d(const Tensor& A) {
    if (A.dim() != 2) throw std::invalid_argument("transpose2d expects 2-D");
    const int64_t m = A.shape[0], n = A.shape[1];
    Tensor T({n, m});
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j) T.data[j * m + i] = A.data[i * n + j];
    return T;
}

static void check_same(const Tensor& a, const Tensor& b, const char* op) {
    if (a.shape != b.shape) throw std::invalid_argument(std::string(op) + ": shape " + shape_str(a.shape) + " vs " + shape_str(b.shape));
}

Tensor add(const Tensor& a, const Tensor& b) { Tensor r = a; add_(r, b); return r; }

Tensor sub(const Tensor& a, const Tensor& b) {
    check_same(a, b, "sub");
    Tensor r = a;
    for (int64_t i = 0; i < r.numel(); ++i) r.data[i] -= b.data[i];
    return r;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    check_same(a, b, "mul");
    Tensor r = a;
    for (int64_t i = 0; i < r.numel(); ++i) r.data[i] *= b.data[i];
    return r;
}

Tensor scale(const Tensor& a, float s) {
    Tensor r = a;
    for (float& v : r.data) v *= s;
    return r;
}

void add_(Tensor& a, const Tensor& b) {
    check_same(a, b, "add");
    for (int64_t i = 0; i < a.numel(); ++i) a.data[i] += b.data[i];
}

Tensor rows(const Tensor& A, int64_t begin, int64_t end) {
    if (A.dim() != 2 || begin < 0 || end > A.shape[0] || begin > end) throw std::out_of_range("rows: bad range");
    const int64_t n = A.shape[1];
    return Tensor({end - begin, n}, std::vector<float>(A.data.begin() + begin * n, A.data.begin() + end * n));
}

Tensor cols(const Tensor& A, int64_t begin, int64_t end) {
    if (A.dim() != 2 || begin < 0 || end > A.shape[1] || begin > end) throw std::out_of_range("cols: bad range");
    const int64_t m = A.shape[0], n = A.shape[1], w = end - begin;
    Tensor C({m, w});
    for (int64_t i = 0; i < m; ++i) std::copy(A.ptr() + i * n + begin, A.ptr() + i * n + end, C.ptr() + i * w);
    return C;
}

Tensor row(const Tensor& A, int64_t i) {
    Tensor r = rows(A, i, i + 1);
    r.shape = {r.shape[1]};
    return r;
}

float dot(const float* a, const float* b, int64_t n) {
    float s = 0.f;
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

float frobenius(const Tensor& a) {
    double s = 0;
    for (float v : a.data) s += double(v) * v;
    return static_cast<float>(std::sqrt(s));
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    check_same(a, b, "max_abs_diff");
    float m = 0.f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        const float x = a.data[i], y = b.data[i];
        // NaN never matches anything (std::max would silently drop it); equal infinities match.
        if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<float>::quiet_NaN();
        if (std::isinf(x) && x == y) continue;
        m = std::max(m, std::fabs(x - y));
    }
    return m;
}

float max_abs(const Tensor& a) {
    float m = 0.f;
    for (float v : a.data) m = std::max(m, std::fabs(v));
    return m;
}

void softmax_(float* v, int64_t n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, v[i]);
    float sum = 0.f;
    for (int64_t i = 0; i < n; ++i) sum += (v[i] = std::exp(v[i] - mx));
    for (int64_t i = 0; i < n; ++i) v[i] /= sum;
}

float log_softmax_at(const float* v, int64_t n, int64_t idx) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, v[i]);
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) sum += std::exp(double(v[i] - mx));
    return v[idx] - mx - static_cast<float>(std::log(sum));
}

std::vector<int64_t> topk(const float* v, int64_t n, int64_t k) {
    k = std::min(k, n);
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int64_t a, int64_t b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

int64_t rank_of(const float* v, int64_t n, int64_t idx) {
    int64_t r = 0;
    for (int64_t i = 0; i < n; ++i) r += v[i] > v[idx];
    return r;
}

}  // namespace tfi
