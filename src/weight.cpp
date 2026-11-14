#include "tfi/weight.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

#include <unistd.h>

#include "half.h"
#include "ops.h"

namespace tfi {

namespace {

// Converted half-precision blocks hold about this many floats (1 MB), per thread.
constexpr int64_t kBlockFloats = int64_t(1) << 18;

std::vector<float>& block_buffer(int64_t floats) {
    thread_local std::vector<float> buf;
    if (static_cast<int64_t>(buf.size()) < floats) buf.resize(static_cast<size_t>(floats));
    return buf;
}

}  // namespace

Weight::Weight(Tensor t) : shape(t.shape), f32_(std::move(t)) {}

int64_t Weight::numel() const { return shape_numel(shape); }

bool Weight::fits_fp16(const Tensor& t) {
    for (float v : t.data)
        if (!detail::fits_half(v)) return false;
    return true;
}

Weight Weight::fp16(const Tensor& t) {
    const int64_t n = t.numel();
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t bytes = std::max<size_t>(static_cast<size_t>(n) * sizeof(uint16_t), 1);
    const size_t capacity = (bytes + page - 1) / page * page;
    void* p = nullptr;
    if (posix_memalign(&p, page, capacity) != 0) throw std::bad_alloc();
    std::shared_ptr<uint16_t> half(static_cast<uint16_t*>(p), [](uint16_t* q) { std::free(q); });
    if (detail::floats_to_halves(t.ptr(), half.get(), n) != 0)
        throw std::invalid_argument("weight has values outside fp16's range (|v| > 65504 or not finite)");
    std::fill(reinterpret_cast<char*>(half.get()) + static_cast<size_t>(n) * sizeof(uint16_t),
              reinterpret_cast<char*>(half.get()) + capacity, char(0));
    Weight w;
    w.shape = t.shape;
    w.half_ = std::move(half);
    w.half_capacity_ = capacity;
    return w;
}

const Tensor& Weight::f32() const {
    if (is_fp16()) throw std::logic_error("this weight is stored in fp16; read it with rows(), cols() or to_tensor()");
    return f32_;
}

void Weight::read_rows(int64_t begin, int64_t end, float* dst) const {
    if (dim() != 2) throw std::invalid_argument("read_rows needs a 2-D weight, got " + shape_str(shape));
    if (begin < 0 || end < begin || end > shape[0])
        throw std::out_of_range("rows [" + std::to_string(begin) + ", " + std::to_string(end) + ") of " + shape_str(shape));
    const int64_t c = shape[1];
    if (is_fp16()) detail::halves_to_floats(half_.get() + begin * c, dst, (end - begin) * c);
    else std::copy(f32_.ptr() + begin * c, f32_.ptr() + end * c, dst);
}

Tensor Weight::rows(int64_t begin, int64_t end) const {
    if (!is_fp16()) return tfi::rows(f32_, begin, end);
    if (dim() != 2) throw std::invalid_argument("rows needs a 2-D weight, got " + shape_str(shape));
    Tensor out({std::max<int64_t>(end - begin, 0), shape[1]});
    read_rows(begin, end, out.ptr());
    return out;
}

Tensor Weight::row(int64_t i) const {
    Tensor r = rows(i, i + 1);
    return std::move(r).reshape({shape[1]});
}

Tensor Weight::cols(int64_t begin, int64_t end) const {
    if (!is_fp16()) return tfi::cols(f32_, begin, end);
    if (dim() != 2) throw std::invalid_argument("cols needs a 2-D weight, got " + shape_str(shape));
    if (begin < 0 || end < begin || end > shape[1])
        throw std::out_of_range("cols [" + std::to_string(begin) + ", " + std::to_string(end) + ") of " + shape_str(shape));
    const int64_t R = shape[0], C = shape[1], n = end - begin;
    Tensor out({R, n});
    for (int64_t r = 0; r < R; ++r) detail::halves_to_floats(half_.get() + r * C + begin, out.ptr() + r * n, n);
    return out;
}

Tensor Weight::to_tensor() const {
    if (!is_fp16()) return f32_;
    Tensor out(shape);
    detail::halves_to_floats(half_.get(), out.ptr(), out.numel());
    return out;
}

Tensor linear(const Tensor& x, const Weight& W, const Weight* b) {
    if (!W.is_fp16() && (!b || !b->is_fp16())) return linear(x, W.f32(), b ? &b->f32() : nullptr);
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    const int64_t in = W.shape[1], out = W.shape[0];
    if (x.size(-1) != in) throw std::invalid_argument("linear: input " + shape_str(x.shape) + " vs weight " + shape_str(W.shape));
    Shape s(x.shape.begin(), x.shape.end() - 1);
    s.push_back(out);
    Tensor y(std::move(s));
    const int64_t n_rows = in ? x.numel() / in : 0;
    if (n_rows && out) {
        if (!W.is_fp16()) {
            detail::gemm(/*trans_b=*/true, n_rows, out, in, x.ptr(), in, W.f32().ptr(), in, y.ptr(), out);
        } else {
            const int64_t block = std::min(out, std::max<int64_t>(1, kBlockFloats / in));
            std::vector<float>& buf = block_buffer(block * in);
            for (int64_t j = 0; j < out; j += block) {
                const int64_t nb = std::min(block, out - j);
                W.read_rows(j, j + nb, buf.data());
                detail::gemm(/*trans_b=*/true, n_rows, nb, in, x.ptr(), in, buf.data(), in, y.ptr() + j, out);
            }
        }
    }
    if (b) {
        const Tensor bias = b->to_tensor();
        if (bias.numel() != out) throw std::invalid_argument("linear: bias does not match the weight");
        for (int64_t r = 0; r < n_rows; ++r)
            for (int64_t j = 0; j < out; ++j) y.data[r * out + j] += bias.data[j];
    }
    return y;
}

Tensor matmul(const Tensor& A, const Weight& W) {
    if (!W.is_fp16()) return matmul(A, W.f32());
    if (A.dim() != 2 || W.dim() != 2) throw std::invalid_argument("matmul expects 2-D tensors");
    const int64_t m = A.shape[0], n = A.shape[1], k = W.shape[1];
    if (n != W.shape[0]) throw std::invalid_argument("matmul shape mismatch " + shape_str(A.shape) + " @ " + shape_str(W.shape));
    Tensor C({m, k});
    if (!m || !k || !n) return C;
    const int64_t block = std::min(n, std::max<int64_t>(1, kBlockFloats / k));
    std::vector<float>& buf = block_buffer(block * k);
    for (int64_t j = 0; j < n; j += block) {
        const int64_t nb = std::min(block, n - j);
        W.read_rows(j, j + nb, buf.data());
        detail::gemm(/*trans_b=*/false, m, k, nb, A.ptr() + j, n, buf.data(), k, C.ptr(), k, /*accumulate=*/j > 0);
    }
    return C;
}

}  // namespace tfi
