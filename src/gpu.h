// The GPU side of batched patching passes: matrix multiplies through Metal Performance Shaders, and
// (Gpu::Seq) whole layers but for attention, with the residual stream kept on the GPU.
// macOS only (TFI_USE_ACCELERATE); internal, not part of the public API.
#pragma once

#ifdef TFI_USE_ACCELERATE

#include <cstdint>
#include <memory>
#include <vector>

#include "tfi/tensor.h"
#include "tfi/weight.h"

namespace tfi {
namespace detail {

// Activations of Gpu::Seq, the same functions as detail::activate_.
enum class GpuAct : int32_t { Relu, Gelu, GeluTanh, Silu, Tanh, Relu2 };

class Gpu {
  public:
    class Seq;

    // nullptr when there is no Metal device (e.g. some virtual machines): callers then use the CPU.
    static std::unique_ptr<Gpu> create();
    ~Gpu();

    // linear(x, W, b) on the GPU: x (..., k), W (n, k). A float32 W is copied to the GPU the first
    // time it is used; a half-precision W (WeightFormat::fp16) is used where it is, without a copy.
    // Weights are identified by their address, so a Gpu must only see weights that outlive it and
    // never change (Model::gpu() keeps one per model). Thread-safe.
    // With fp16, x and W go in half precision (accumulation and output stay fp32): the multiply runs
    // several times faster, with relative errors around 1e-3. An x or a float32 W with a value outside
    // fp16's range (|v| > 65504) is multiplied in fp32 instead. Without fp16, x stays fp32; a
    // half-precision W is then multiplied as it is, exactly (each fp16 value is an fp32 value).
    Tensor linear(const Tensor& x, const Weight& W, const Weight* b, bool fp16 = false);

    // linear(x, Ws[i], bs[i]) for every i in one GPU submission: x is copied to the GPU once and
    // the thread waits once, instead of once per weight. Results are identical to separate
    // linear() calls. bs[i] may be null; bs may be empty (no biases).
    std::vector<Tensor> linear_many(const Tensor& x, const std::vector<const Weight*>& Ws,
                                    const std::vector<const Weight*>& bs, bool fp16 = false);

  private:
    Gpu();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Operations on GPU-resident fp32 buffers ("slots"), recorded and then run as one GPU submission:
// how Model::forward_rows keeps a batched pass's residual stream on the GPU, with one round trip per
// layer (q, k, v come back for attention on the CPU, z goes in). Slots are numbered by the caller;
// the CPU reads and writes them through data() between runs. One per thread; the Gpu must outlive
// it. Weights are cached as in Gpu::linear.
class Gpu::Seq {
  public:
    explicit Seq(Gpu& gpu);
    ~Seq();
    Seq(const Seq&) = delete;
    Seq& operator=(const Seq&) = delete;

    // Make `slot` hold at least `floats` values and return it. It only grows; growing loses the contents.
    float* reserve(int slot, int64_t floats);
    float* data(int slot);  // a reserved slot's values

    // Record operations; sizes are checked against the slots when run() encodes them.
    void linear(int x, int y, int64_t rows, const Weight& W, const Weight* b);  // y = x W^T (+ b)
    // Over rows of w.numel() values: (x - mean) / sqrt(var + eps) * w (+ b), as Model::norm.
    // Without `layernorm` (rmsnorm), mean is 0 and b must be null.
    void norm(int x, int y, int64_t rows, const Tensor& w, const Tensor* b, bool layernorm, float eps);
    void activate(int x, int64_t n, GpuAct act);         // in place
    void glu(int gate, int up, int64_t n, GpuAct act);   // gate = act(gate) * up
    void add(int a, int b, int out, int64_t n);          // out = a + b; out may be a or b
    void copy(int x, int y, int64_t n);
    // Backward passes (fp32 inputs, whatever run() is given): dx = dy W, linear()'s input gradient for W (out, in);
    // norm()'s input gradient given its input x and output gradient dy (as grad.cpp's norm_backward);
    // dpre = dpost * act'(pre); and for a GLU, post = act(pre) * up: dpre = dpost * up * act'(pre),
    // dup = dpost * act(pre).
    void linear_t(int dy, int dx, int64_t rows, const Weight& W);
    void norm_backward(int x, int dy, int dx, int64_t rows, const Tensor& w, bool layernorm, float eps);
    void act_backward(int dpost, int pre, int dpre, int64_t n, GpuAct act);
    void glu_backward(int dpost, int pre, int up, int dpre, int dup, int64_t n, GpuAct act);

    // Run everything recorded since the last run in one submission and wait. With fp16, linear()
    // inputs and weights go in half precision, as in Gpu::linear; if any input is outside fp16's
    // range the whole run is repeated in fp32. So the recorded operations must not overwrite a slot
    // that they read before writing it (anything the CPU or an earlier run left there).
    void run(bool fp16);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Batched passes and linear layers with at least this many rows use the GPU (smaller ones are
// faster on the CPU).
int64_t gpu_min_rows();
// For tests: 0 restores the default; a very large value turns the GPU off.
void set_gpu_min_rows(int64_t rows);

}  // namespace detail
}  // namespace tfi

#endif
