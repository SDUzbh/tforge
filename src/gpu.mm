// GPU matrix multiplies through Metal Performance Shaders (see gpu.h).
#include "gpu.h"

#ifdef TFI_USE_ACCELERATE

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tfi {
namespace detail {

namespace {
constexpr int64_t kDefaultMinRows = 256;  // tuned on an M5 Pro
std::atomic<int64_t> g_min_rows{0};
}  // namespace

int64_t gpu_min_rows() {
    const int64_t r = g_min_rows.load();
    return r > 0 ? r : kDefaultMinRows;
}
void set_gpu_min_rows(int64_t rows) { g_min_rows = rows; }

// Kernels of Gpu::Seq, compiled when first needed. The math is src/vmath.h's (same polynomials),
// so GPU and CPU agree to float rounding.
const char* const kKernelSource = R"(
#include <metal_stdlib>
using namespace metal;

static float pow2i(int n) { return as_type<float>((n + 127) << 23); }

static float vexp(float x) {
    const float xc = x > -104.f ? (x < 88.8f ? x : 88.8f) : -104.f;
    const float n = (xc * 1.44269504088896341f + 12582912.f) - 12582912.f;
    const float r = (xc - n * 0.693359375f) - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.f;
    const int ni = int(n), h = ni / 2;
    float y = p * pow2i(h) * pow2i(ni - h);
    y = x > 88.7228394f ? INFINITY : y;
    y = x < -103.98f ? 0.f : y;
    return x != x ? x : y;
}

static float verf(float x) {
    const float a = fabs(x);
    const float s = x * x;
    float p = -6.37348276e-4f;
    p = p * s + 5.06145647e-3f;
    p = p * s + -2.68091038e-2f;
    p = p * s + 1.12828925e-1f;
    p = p * s + -3.76125902e-1f;
    p = p * s + 1.12837911f;
    const float small = x * p;
    const float ac = a < 4.f ? a : 4.f;
    const float t = 1.f / (1.f + ac);
    float q = -0.514288127f;
    q = q * t + 1.09394825f;
    q = q * t + -0.410534561f;
    q = q * t + -0.74236536f;
    q = q * t + 0.463124543f;
    q = q * t + 0.528218627f;
    q = q * t + 0.567896247f;
    q = q * t + -0.000157813804f;
    const float big = 1.f - vexp(-ac * ac) * q;
    const float y = a < 0.84375f ? small : (x < 0.f ? -big : big);
    return x != x ? x : y;
}

static float vtanh(float x) {
    const float a = fabs(x);
    const float s = x * x;
    float p = -5.69752231e-3f;
    p = p * s + 2.06303075e-2f;
    p = p * s + -5.37358634e-2f;
    p = p * s + 1.33313656e-1f;
    p = p * s + -3.33332747e-1f;
    p = p * s + 1.f;
    const float small = x * p;
    const float ac = a < 10.f ? a : 10.f;
    const float big = 1.f - 2.f / (vexp(2.f * ac) + 1.f);
    const float y = a < 0.625f ? small : (x < 0.f ? -big : big);
    return x != x ? x : y;
}

// GpuAct order: relu, gelu, gelu_tanh, silu, tanh, relu2.
static float act(float v, int kind, float k) {
    switch (kind) {
        case 0: return v > 0 ? v : 0.f;
        case 1: return 0.5f * v * (1.f + verf(v * 0.70710678118654752440f));
        case 2: return 0.5f * v * (1.f + vtanh(k * (v + 0.044715f * v * v * v)));
        case 3: return v / (1.f + vexp(-v));
        case 4: return vtanh(v);
        default: return v > 0 ? v * v : 0.f;
    }
}

// d act(v) / dv, as detail::activation_grad_.
static float act_grad(float v, int kind, float k) {
    switch (kind) {
        case 0: return v > 0 ? 1.f : 0.f;
        case 1: return 0.5f * (1.f + verf(v * 0.70710678118654752440f)) + v * 0.398942280401432678f * vexp(-0.5f * v * v);
        case 2: {
            const float th = vtanh(k * (v + 0.044715f * v * v * v));
            return 0.5f * (1.f + th) + 0.5f * v * (1.f - th * th) * k * (1.f + 3.f * 0.044715f * v * v);
        }
        case 3: {
            const float s = 1.f / (1.f + vexp(-v));
            return s * (1.f + v * (1.f - s));
        }
        case 4: {
            const float th = vtanh(v);
            return 1.f - th * th;
        }
        default: return v > 0 ? 2.f * v : 0.f;
    }
}

struct ActArgs { uint n; int kind; float k; };

kernel void act_bwd_k(device const float* dpost [[buffer(0)]], device const float* pre [[buffer(1)]],
                      device float* dpre [[buffer(2)]], constant ActArgs& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) dpre[i] = dpost[i] * act_grad(pre[i], a.kind, a.k);
}

kernel void glu_bwd_k(device const float* dpost [[buffer(0)]], device const float* pre [[buffer(1)]],
                      device const float* up [[buffer(2)]], device float* dpre [[buffer(3)]], device float* dup [[buffer(4)]],
                      constant ActArgs& a [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.n) return;
    const float p = pre[i], g = dpost[i];
    dpre[i] = (g * up[i]) * act_grad(p, a.kind, a.k);
    dup[i] = g * act(p, a.kind, a.k);
}

kernel void act_k(device float* x [[buffer(0)]], constant ActArgs& a [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < a.n) x[i] = act(x[i], a.kind, a.k);
}

kernel void glu_k(device float* g [[buffer(0)]], device const float* u [[buffer(1)]], constant ActArgs& a [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < a.n) g[i] = act(g[i], a.kind, a.k) * u[i];
}

kernel void add_k(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                  constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

kernel void copy_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i];
}

struct BiasArgs { uint n, cols; };

kernel void bias_k(device float* y [[buffer(0)]], device const float* b [[buffer(1)]], constant BiasArgs& a [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < a.n) y[i] += b[i % a.cols];
}

// fp16 copy of x for a half-precision multiply; flags any value outside fp16's range (or NaN).
kernel void half_k(device const float* x [[buffer(0)]], device half* h [[buffer(1)]], device atomic_uint* flag [[buffer(2)]],
                   constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = x[i];
    h[i] = half(v);
    if (!(fabs(v) <= 65504.f)) atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

// Sum over the threadgroup; every thread gets the same value (summed in the same order).
static float tg_sum(float v, threadgroup float* part, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.f;
    for (uint i = 0; i < nsg; ++i) t += part[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

struct NormArgs { uint d; float eps; uint layernorm; uint has_b; };

// One threadgroup per row.
kernel void norm_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const float* w [[buffer(2)]],
                   device const float* b [[buffer(3)]], constant NormArgs& a [[buffer(4)]],
                   uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint d = a.d;
    device const float* xi = x + ulong(row) * d;
    device float* yi = y + ulong(row) * d;
    float mean = 0.f;
    if (a.layernorm) {
        float s = 0.f;
        for (uint j = tid; j < d; j += tg) s += xi[j];
        mean = tg_sum(s, part, lane, sg, nsg) / float(d);
    }
    float q = 0.f;
    for (uint j = tid; j < d; j += tg) {
        const float t = xi[j] - mean;
        q += t * t;
    }
    const float inv = 1.0f / precise::sqrt(tg_sum(q, part, lane, sg, nsg) / float(d) + a.eps);
    for (uint j = tid; j < d; j += tg) {
        const float v = (xi[j] - mean) * inv * w[j];
        yi[j] = a.has_b ? v + b[j] : v;
    }
}

// Backward of norm_k (without its bias): dx from x and dy, one threadgroup per row.
kernel void norm_bwd_k(device const float* x [[buffer(0)]], device const float* dy [[buffer(1)]],
                       device const float* w [[buffer(2)]], device float* dx [[buffer(3)]], constant NormArgs& a [[buffer(4)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[32];
    const uint d = a.d;
    device const float* xi = x + ulong(row) * d;
    device const float* dyi = dy + ulong(row) * d;
    device float* dxi = dx + ulong(row) * d;
    if (a.layernorm) {  // dx = inv * (g - mean(g) - xhat * mean(g * xhat)), g = dy * w
        float s = 0.f;
        for (uint j = tid; j < d; j += tg) s += xi[j];
        const float mean = tg_sum(s, part, lane, sg, nsg) / float(d);
        float q = 0.f;
        for (uint j = tid; j < d; j += tg) {
            const float t = xi[j] - mean;
            q += t * t;
        }
        const float inv = 1.0f / precise::sqrt(tg_sum(q, part, lane, sg, nsg) / float(d) + a.eps);
        float gs = 0.f, gx = 0.f;
        for (uint j = tid; j < d; j += tg) {
            const float g = dyi[j] * w[j];
            gs += g;
            gx += g * (xi[j] - mean) * inv;
        }
        const float gm = tg_sum(gs, part, lane, sg, nsg) / float(d);
        const float gxm = tg_sum(gx, part, lane, sg, nsg) / float(d);
        for (uint j = tid; j < d; j += tg) dxi[j] = inv * (dyi[j] * w[j] - gm - (xi[j] - mean) * inv * gxm);
    } else {  // rmsnorm: dx = inv * g - x * inv^3 * mean(g * x)
        float ms = 0.f, gx = 0.f;
        for (uint j = tid; j < d; j += tg) {
            ms += xi[j] * xi[j];
            gx += dyi[j] * w[j] * xi[j];
        }
        const float inv = 1.0f / precise::sqrt(tg_sum(ms, part, lane, sg, nsg) / float(d) + a.eps);
        const float gxm = tg_sum(gx, part, lane, sg, nsg) / float(d);
        for (uint j = tid; j < d; j += tg) dxi[j] = inv * dyi[j] * w[j] - xi[j] * inv * inv * inv * gxm;
    }
}
)";

struct ActArgs { uint32_t n; int32_t kind; float k; };
struct BiasArgs { uint32_t n, cols; };
struct NormArgs { uint32_t d; float eps; uint32_t layernorm, has_b; };

// Write src as fp16; false if a value is outside fp16's range (or not finite).
bool to_half(const float* src, __fp16* dst, int64_t n) {
    int64_t out_of_range = 0;
    for (int64_t i = 0; i < n; ++i) {
        out_of_range += !(std::fabs(src[i]) <= 65504.f);
        dst[i] = static_cast<__fp16>(src[i]);
    }
    return out_of_range == 0;
}

struct Gpu::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::mutex mutex;  // guards the weight caches and pool
    // By host address; the fp16 cache holds nil for a weight outside fp16's range.
    std::unordered_map<const float*, id<MTLBuffer>> weights, weights_fp16;
    std::vector<id<MTLBuffer>> pool;  // free scratch buffers

    id<MTLBuffer> weight_buffer(const Tensor& W, bool fp16) {
        std::lock_guard<std::mutex> lock(mutex);
        auto& cache = fp16 ? weights_fp16 : weights;
        auto it = cache.find(W.ptr());
        if (it != cache.end()) return it->second;
        const NSUInteger bytes = static_cast<NSUInteger>(W.numel()) * (fp16 ? sizeof(__fp16) : sizeof(float));
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("GPU: could not allocate a weight buffer");
        if (!fp16) std::memcpy(b.contents, W.ptr(), bytes);
        else if (!to_half(W.ptr(), static_cast<__fp16*>(b.contents), W.numel())) b = nil;
        cache.emplace(W.ptr(), b);
        return b;
    }

    id<MTLBuffer> acquire(size_t bytes) {
        {
            // The smallest free buffer that fits, so small requests don't take large buffers.
            std::lock_guard<std::mutex> lock(mutex);
            size_t best = pool.size();
            for (size_t i = 0; i < pool.size(); ++i)
                if (pool[i].length >= bytes && (best == pool.size() || pool[i].length < pool[best].length)) best = i;
            if (best < pool.size()) {
                id<MTLBuffer> b = pool[best];
                pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(best));
                return b;
            }
        }
        id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) throw std::runtime_error("GPU: could not allocate a buffer");
        return b;
    }

    void release(id<MTLBuffer> b) {
        std::lock_guard<std::mutex> lock(mutex);
        pool.push_back(b);
    }

    // Gpu::Seq's kernels, compiled on first use.
    std::once_flag kernels_once;
    id<MTLComputePipelineState> act_k, glu_k, add_k, copy_k, bias_k, half_k, norm_k, norm_bwd_k, act_bwd_k, glu_bwd_k;

    void load_kernels() {
        std::call_once(kernels_once, [this] {
            @autoreleasepool {
                MTLCompileOptions* opt = [MTLCompileOptions new];
                opt.mathMode = MTLMathModeSafe;  // no fast-math rewrites: keep the CPU's arithmetic
                NSError* err = nil;
                id<MTLLibrary> lib = [device newLibraryWithSource:@(kKernelSource) options:opt error:&err];
                if (!lib)
                    throw std::runtime_error(std::string("GPU: kernels did not compile: ") +
                                             (err ? err.localizedDescription.UTF8String : "unknown error"));
                auto make = [&](const char* name) {
                    id<MTLFunction> f = [lib newFunctionWithName:@(name)];
                    id<MTLComputePipelineState> p = f ? [device newComputePipelineStateWithFunction:f error:&err] : nil;
                    if (!p) throw std::runtime_error(std::string("GPU: could not load kernel ") + name);
                    return p;
                };
                act_k = make("act_k");
                glu_k = make("glu_k");
                add_k = make("add_k");
                copy_k = make("copy_k");
                bias_k = make("bias_k");
                half_k = make("half_k");
                norm_k = make("norm_k");
                norm_bwd_k = make("norm_bwd_k");
                act_bwd_k = make("act_bwd_k");
                glu_bwd_k = make("glu_bwd_k");
            }
        });
    }
};

Gpu::Gpu() : impl_(new Impl) {}
Gpu::~Gpu() = default;

std::unique_ptr<Gpu> Gpu::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !MPSSupportsMTLDevice(device)) return nullptr;
        std::unique_ptr<Gpu> gpu(new Gpu);
        gpu->impl_->device = device;
        gpu->impl_->queue = [device newCommandQueue];
        if (!gpu->impl_->queue) return nullptr;
        return gpu;
    }
}

Tensor Gpu::linear(const Tensor& x, const Tensor& W, const Tensor* b, bool fp16) {
    return std::move(linear_many(x, {&W}, {b}, fp16)[0]);
}

std::vector<Tensor> Gpu::linear_many(const Tensor& x, const std::vector<const Tensor*>& Ws,
                                     const std::vector<const Tensor*>& bs, bool fp16) {
    if (!bs.empty() && bs.size() != Ws.size()) throw std::invalid_argument("linear_many: one bias per weight");
    const int64_t k = x.size(-1), m = k ? x.numel() / k : 0;
    std::vector<Tensor> ys;
    ys.reserve(Ws.size());
    for (const Tensor* W : Ws) {
        if (W->dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
        if (W->shape[1] != k)
            throw std::invalid_argument("linear: input " + shape_str(x.shape) + " vs weight " + shape_str(W->shape));
        Shape shape(x.shape.begin(), x.shape.end() - 1);
        shape.push_back(W->shape[0]);
        ys.emplace_back(std::move(shape));
    }
    if (m == 0) return ys;
    Impl& g = *impl_;
    const size_t nw = Ws.size();
    @autoreleasepool {
        // Each weight runs in fp16 if asked and both it and x fit, else in fp32 (as linear() would).
        std::vector<id<MTLBuffer>> wb(nw, nil);
        std::vector<bool> half(nw, false);
        id<MTLBuffer> xh = nil, xf = nil;
        if (fp16) {
            bool any = false;
            for (size_t i = 0; i < nw; ++i)
                if (Ws[i]->numel()) any |= (wb[i] = g.weight_buffer(*Ws[i], true)) != nil;
            if (any) {
                xh = g.acquire(static_cast<size_t>(m * k) * sizeof(__fp16));
                if (!to_half(x.ptr(), static_cast<__fp16*>(xh.contents), m * k)) {
                    g.release(xh);
                    xh = nil;
                }
            }
            for (size_t i = 0; i < nw; ++i) half[i] = xh && wb[i];
        }
        for (size_t i = 0; i < nw; ++i) {
            if (half[i] || Ws[i]->numel() == 0) continue;
            wb[i] = g.weight_buffer(*Ws[i], false);
            if (!xf) {
                xf = g.acquire(static_cast<size_t>(m * k) * sizeof(float));
                std::memcpy(xf.contents, x.ptr(), static_cast<size_t>(m * k) * sizeof(float));
            }
        }
        std::vector<id<MTLBuffer>> yb(nw, nil);
        id<MTLCommandBuffer> cmd = [g.queue commandBuffer];
        for (size_t i = 0; i < nw; ++i) {
            const int64_t n = Ws[i]->shape[0];
            if (n == 0) continue;
            yb[i] = g.acquire(static_cast<size_t>(m * n) * sizeof(float));
            const MPSDataType in_type = half[i] ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            const NSUInteger in_size = half[i] ? sizeof(__fp16) : sizeof(float);
            MPSMatrixDescriptor* dx = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * in_size
                                                                           dataType:in_type];
            MPSMatrixDescriptor* dw = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * in_size
                                                                           dataType:in_type];
            MPSMatrixDescriptor* dy = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
                                                                           rowBytes:n * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
            MPSMatrix* mx = [[MPSMatrix alloc] initWithBuffer:(half[i] ? xh : xf) descriptor:dx];
            MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:wb[i] descriptor:dw];
            MPSMatrix* my = [[MPSMatrix alloc] initWithBuffer:yb[i] descriptor:dy];
            // A kernel object per call: MPS kernels must not be shared between threads.
            MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:g.device
                                                                            transposeLeft:NO
                                                                           transposeRight:YES
                                                                               resultRows:m
                                                                            resultColumns:n
                                                                          interiorColumns:k
                                                                                    alpha:1.0
                                                                                     beta:0.0];
            [mm encodeToCommandBuffer:cmd leftMatrix:mx rightMatrix:mw resultMatrix:my];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        const bool ok = cmd.status == MTLCommandBufferStatusCompleted;
        for (size_t i = 0; i < nw; ++i) {
            if (!yb[i]) continue;
            if (ok) std::memcpy(ys[i].ptr(), yb[i].contents, static_cast<size_t>(ys[i].numel()) * sizeof(float));
            g.release(yb[i]);
        }
        if (xh) g.release(xh);
        if (xf) g.release(xf);
        if (!ok) throw std::runtime_error("GPU: matrix multiply failed");
    }
    for (size_t i = 0; i < nw; ++i) {
        const Tensor* b = bs.empty() ? nullptr : bs[i];
        if (!b) continue;
        const int64_t n = Ws[i]->shape[0];
        float* y = ys[i].ptr();
        for (int64_t r = 0; r < m; ++r)
            for (int64_t j = 0; j < n; ++j) y[r * n + j] += b->data[j];
    }
    return ys;
}

// --- Gpu::Seq ---------------------------------------------------------------------------

struct Gpu::Seq::Impl {
    Gpu::Impl& g;
    struct Slot {
        id<MTLBuffer> buf = nil, half = nil;  // half: fp16 copy for half-precision multiplies
        int64_t floats = 0;
    };
    struct Op {
        enum Kind { Linear, LinearT, Norm, NormBwd, Act, Glu, Add, Copy, ActBwd, GluBwd } kind;
        int a = 0, b = 0, c = 0, d = 0, e = 0;
        int64_t rows = 0, n = 0;
        const Tensor* W = nullptr;
        const Tensor* bias = nullptr;
        GpuAct act = GpuAct::Relu;
        float eps = 0.f;
        bool layernorm = false;
    };
    std::vector<Slot> slots;
    std::vector<Op> ops;
    id<MTLBuffer> flag = nil;  // set by half_k when an input does not fit fp16
    std::map<std::tuple<int64_t, int64_t, int64_t, bool>, MPSMatrixMultiplication*> mms;  // by (m, n, k, W^T)

    // Encoding state of one attempt.
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::vector<bool> half_valid;

    explicit Impl(Gpu::Impl& gpu) : g(gpu) {}

    Slot& slot(int i) {
        if (i < 0 || static_cast<size_t>(i) >= slots.size() || !slots[i].buf)
            throw std::logic_error("Gpu::Seq: slot " + std::to_string(i) + " was not reserved");
        return slots[i];
    }
    void need(int i, int64_t floats) {
        if (slot(i).floats < floats) throw std::logic_error("Gpu::Seq: slot " + std::to_string(i) + " is too small");
    }
    id<MTLComputeCommandEncoder> compute() {
        if (!enc) enc = [cmd computeCommandEncoder];
        return enc;
    }
    void end_compute() {
        if (enc) [enc endEncoding];
        enc = nil;
    }
    void dispatch(id<MTLComputePipelineState> p, int64_t n) {
        const NSUInteger tg = std::min<NSUInteger>(256, p.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }
    static uint32_t u32(int64_t n) {
        if (n < 0 || n > static_cast<int64_t>(UINT32_MAX)) throw std::length_error("Gpu::Seq: operation too large");
        return static_cast<uint32_t>(n);
    }

    // y (m, n) = x (m, k) W^T, W (n, k); or without trans_w, y = x W, W (k, n).
    MPSMatrixMultiplication* mm(int64_t m, int64_t n, int64_t k, bool trans_w = true) {
        const auto key = std::make_tuple(m, n, k, trans_w);
        auto it = mms.find(key);
        if (it != mms.end()) return it->second;
        MPSMatrixMultiplication* p = [[MPSMatrixMultiplication alloc] initWithDevice:g.device
                                                                       transposeLeft:NO
                                                                      transposeRight:trans_w ? YES : NO
                                                                          resultRows:m
                                                                       resultColumns:n
                                                                     interiorColumns:k
                                                                               alpha:1.0
                                                                                beta:0.0];
        mms.emplace(key, p);
        return p;
    }

    void encode_linear(const Op& op, bool fp16) {
        const Tensor& W = *op.W;
        const int64_t m = op.rows, n = W.shape[0], k = W.shape[1];
        need(op.a, m * k);
        need(op.c, m * n);
        if (m == 0 || n == 0) return;
        if (k == 0) throw std::logic_error("Gpu::Seq: linear with an empty input");
        id<MTLBuffer> wh = fp16 ? g.weight_buffer(W, true) : nil;
        const bool half = wh != nil;
        id<MTLBuffer> in = slot(op.a).buf, wb = wh;
        if (half) {
            Slot& s = slot(op.a);
            if (!half_valid[op.a]) {
                const size_t bytes = static_cast<size_t>(m * k) * sizeof(__fp16);
                if (s.half && s.half.length < bytes) {
                    g.release(s.half);
                    s.half = nil;
                }
                if (!s.half) s.half = g.acquire(bytes);
                const uint32_t cnt = u32(m * k);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.half_k];
                [e setBuffer:s.buf offset:0 atIndex:0];
                [e setBuffer:s.half offset:0 atIndex:1];
                [e setBuffer:flag offset:0 atIndex:2];
                [e setBytes:&cnt length:sizeof cnt atIndex:3];
                dispatch(g.half_k, m * k);
                half_valid[op.a] = true;
            }
            in = s.half;
        } else {
            wb = g.weight_buffer(W, false);
        }
        end_compute();
        const MPSDataType type = half ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        const NSUInteger size = half ? sizeof(__fp16) : sizeof(float);
        MPSMatrixDescriptor* dx = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * size dataType:type];
        MPSMatrixDescriptor* dw = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * size dataType:type];
        MPSMatrixDescriptor* dy = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrix* mx = [[MPSMatrix alloc] initWithBuffer:in descriptor:dx];
        MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:wb descriptor:dw];
        MPSMatrix* my = [[MPSMatrix alloc] initWithBuffer:slot(op.c).buf descriptor:dy];
        [mm(m, n, k) encodeToCommandBuffer:cmd leftMatrix:mx rightMatrix:mw resultMatrix:my];
        if (op.bias) {
            if (op.bias->numel() != n) throw std::invalid_argument("Gpu::Seq: bias does not match the weight");
            const BiasArgs args{u32(m * n), u32(n)};
            id<MTLComputeCommandEncoder> e = compute();
            [e setComputePipelineState:g.bias_k];
            [e setBuffer:slot(op.c).buf offset:0 atIndex:0];
            [e setBuffer:g.weight_buffer(*op.bias, false) offset:0 atIndex:1];
            [e setBytes:&args length:sizeof args atIndex:2];
            dispatch(g.bias_k, m * n);
        }
    }

    // dx (m, k) = dy (m, n) W, W (n, k).
    void encode_linear_t(const Op& op) {
        const Tensor& W = *op.W;
        const int64_t m = op.rows, n = W.shape[0], k = W.shape[1];
        need(op.a, m * n);
        need(op.c, m * k);
        if (m == 0 || k == 0) return;
        if (n == 0) throw std::logic_error("Gpu::Seq: linear_t with an empty input");
        end_compute();
        MPSMatrixDescriptor* ddy = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float)
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dw = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* ddx = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(float)
                                                                        dataType:MPSDataTypeFloat32];
        MPSMatrix* mdy = [[MPSMatrix alloc] initWithBuffer:slot(op.a).buf descriptor:ddy];
        MPSMatrix* mw = [[MPSMatrix alloc] initWithBuffer:g.weight_buffer(W, false) descriptor:dw];
        MPSMatrix* mdx = [[MPSMatrix alloc] initWithBuffer:slot(op.c).buf descriptor:ddx];
        [mm(m, k, n, false) encodeToCommandBuffer:cmd leftMatrix:mdy rightMatrix:mw resultMatrix:mdx];
    }

    void encode(const Op& op, bool fp16) {
        switch (op.kind) {
            case Op::Linear:
                encode_linear(op, fp16);
                break;
            case Op::LinearT:
                encode_linear_t(op);
                break;
            case Op::NormBwd: {
                const int64_t d = op.W->numel();
                need(op.a, op.rows * d);
                need(op.b, op.rows * d);
                need(op.c, op.rows * d);
                if (op.rows == 0 || d == 0) break;
                const NormArgs args{u32(d), op.eps, op.layernorm ? 1u : 0u, 0u};
                id<MTLBuffer> wb = g.weight_buffer(*op.W, false);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.norm_bwd_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBuffer:wb offset:0 atIndex:2];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:3];
                [e setBytes:&args length:sizeof args atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(256, g.norm_bwd_k.maxTotalThreadsPerThreadgroup);
                [e dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(op.rows), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                break;
            }
            case Op::ActBwd:
            case Op::GluBwd: {
                const bool glu = op.kind == Op::GluBwd;
                need(op.a, op.n);
                need(op.b, op.n);
                need(op.c, op.n);
                if (glu) {
                    need(op.d, op.n);
                    need(op.e, op.n);
                }
                if (op.n == 0) break;
                const ActArgs args{u32(op.n), static_cast<int32_t>(op.act), std::sqrt(2.f / 3.14159265358979323846f)};
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:glu ? g.glu_bwd_k : g.act_bwd_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:2];
                if (glu) {
                    [e setBuffer:slot(op.d).buf offset:0 atIndex:3];
                    [e setBuffer:slot(op.e).buf offset:0 atIndex:4];
                }
                [e setBytes:&args length:sizeof args atIndex:glu ? 5 : 3];
                dispatch(glu ? g.glu_bwd_k : g.act_bwd_k, op.n);
                break;
            }
            case Op::Norm: {
                const int64_t d = op.W->numel();
                need(op.a, op.rows * d);
                need(op.c, op.rows * d);
                if (op.rows == 0 || d == 0) break;
                if (op.bias && op.bias->numel() != d) throw std::invalid_argument("Gpu::Seq: norm bias does not match");
                const NormArgs args{u32(d), op.eps, op.layernorm ? 1u : 0u, op.bias ? 1u : 0u};
                id<MTLBuffer> wb = g.weight_buffer(*op.W, false);
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:g.norm_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                [e setBuffer:wb offset:0 atIndex:2];
                [e setBuffer:op.bias ? g.weight_buffer(*op.bias, false) : wb offset:0 atIndex:3];
                [e setBytes:&args length:sizeof args atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(256, g.norm_k.maxTotalThreadsPerThreadgroup);
                [e dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(op.rows), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                break;
            }
            case Op::Act:
            case Op::Glu: {
                need(op.a, op.n);
                if (op.kind == Op::Glu) need(op.b, op.n);
                if (op.n == 0) break;
                const ActArgs args{u32(op.n), static_cast<int32_t>(op.act), std::sqrt(2.f / 3.14159265358979323846f)};
                id<MTLComputeCommandEncoder> e = compute();
                [e setComputePipelineState:op.kind == Op::Act ? g.act_k : g.glu_k];
                [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                if (op.kind == Op::Glu) [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                [e setBytes:&args length:sizeof args atIndex:op.kind == Op::Act ? 1 : 2];
                dispatch(op.kind == Op::Act ? g.act_k : g.glu_k, op.n);
                break;
            }
            case Op::Add:
            case Op::Copy: {
                need(op.a, op.n);
                need(op.c, op.n);
                if (op.kind == Op::Add) need(op.b, op.n);
                if (op.n == 0) break;
                const uint32_t cnt = u32(op.n);
                id<MTLComputeCommandEncoder> e = compute();
                if (op.kind == Op::Add) {
                    [e setComputePipelineState:g.add_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.b).buf offset:0 atIndex:1];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:2];
                    [e setBytes:&cnt length:sizeof cnt atIndex:3];
                    dispatch(g.add_k, op.n);
                } else {
                    [e setComputePipelineState:g.copy_k];
                    [e setBuffer:slot(op.a).buf offset:0 atIndex:0];
                    [e setBuffer:slot(op.c).buf offset:0 atIndex:1];
                    [e setBytes:&cnt length:sizeof cnt atIndex:2];
                    dispatch(g.copy_k, op.n);
                }
                break;
            }
        }
        // Whatever the operation wrote no longer matches its fp16 copy.
        if (op.kind == Op::GluBwd) {
            half_valid[op.d] = false;
            half_valid[op.e] = false;
        } else {
            half_valid[op.kind == Op::Act || op.kind == Op::Glu ? op.a : op.c] = false;
        }
    }

    // One attempt; true if fp16 was asked for and an input did not fit.
    bool attempt(bool fp16) {
        @autoreleasepool {
            *static_cast<uint32_t*>(flag.contents) = 0;
            cmd = [g.queue commandBuffer];
            half_valid.assign(slots.size(), false);
            try {
                for (const Op& op : ops) encode(op, fp16);
                end_compute();
            } catch (...) {
                end_compute();
                cmd = nil;
                throw;
            }
            [cmd commit];
            [cmd waitUntilCompleted];
            const bool ok = cmd.status == MTLCommandBufferStatusCompleted;
            cmd = nil;
            if (!ok) throw std::runtime_error("GPU: a batched layer failed");
            return fp16 && *static_cast<uint32_t*>(flag.contents) != 0;
        }
    }
};

Gpu::Seq::Seq(Gpu& gpu) : impl_(new Impl(*gpu.impl_)) {
    gpu.impl_->load_kernels();
    impl_->flag = gpu.impl_->acquire(sizeof(uint32_t));
}

Gpu::Seq::~Seq() {
    Impl& s = *impl_;
    for (Impl::Slot& slot : s.slots) {
        if (slot.buf) s.g.release(slot.buf);
        if (slot.half) s.g.release(slot.half);
    }
    if (s.flag) s.g.release(s.flag);
}

float* Gpu::Seq::reserve(int slot, int64_t floats) {
    if (slot < 0) throw std::invalid_argument("Gpu::Seq: negative slot");
    Impl& s = *impl_;
    if (static_cast<size_t>(slot) >= s.slots.size()) s.slots.resize(static_cast<size_t>(slot) + 1);
    Impl::Slot& sl = s.slots[slot];
    if (!sl.buf || sl.floats < floats) {
        if (sl.buf) s.g.release(sl.buf);
        sl.buf = s.g.acquire(static_cast<size_t>(std::max<int64_t>(floats, 1)) * sizeof(float));
        sl.floats = floats;
    }
    return static_cast<float*>(sl.buf.contents);
}

float* Gpu::Seq::data(int slot) { return static_cast<float*>(impl_->slot(slot).buf.contents); }

void Gpu::Seq::linear(int x, int y, int64_t rows, const Tensor& W, const Tensor* b) {
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    Impl::Op op{Impl::Op::Linear};
    op.a = x;
    op.c = y;
    op.rows = rows;
    op.W = &W;
    op.bias = b;
    impl_->ops.push_back(op);
}

void Gpu::Seq::norm(int x, int y, int64_t rows, const Tensor& w, const Tensor* b, bool layernorm, float eps) {
    if (b && !layernorm) throw std::invalid_argument("Gpu::Seq: rmsnorm has no bias");
    Impl::Op op{Impl::Op::Norm};
    op.a = x;
    op.c = y;
    op.rows = rows;
    op.W = &w;
    op.bias = b;
    op.layernorm = layernorm;
    op.eps = eps;
    impl_->ops.push_back(op);
}

void Gpu::Seq::activate(int x, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::Act};
    op.a = x;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::glu(int gate, int up, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::Glu};
    op.a = gate;
    op.b = up;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::add(int a, int b, int out, int64_t n) {
    Impl::Op op{Impl::Op::Add};
    op.a = a;
    op.b = b;
    op.c = out;
    op.n = n;
    impl_->ops.push_back(op);
}

void Gpu::Seq::copy(int x, int y, int64_t n) {
    Impl::Op op{Impl::Op::Copy};
    op.a = x;
    op.c = y;
    op.n = n;
    impl_->ops.push_back(op);
}

void Gpu::Seq::linear_t(int dy, int dx, int64_t rows, const Tensor& W) {
    if (W.dim() != 2) throw std::invalid_argument("linear weight must be 2-D");
    Impl::Op op{Impl::Op::LinearT};
    op.a = dy;
    op.c = dx;
    op.rows = rows;
    op.W = &W;
    impl_->ops.push_back(op);
}

void Gpu::Seq::norm_backward(int x, int dy, int dx, int64_t rows, const Tensor& w, bool layernorm, float eps) {
    Impl::Op op{Impl::Op::NormBwd};
    op.a = x;
    op.b = dy;
    op.c = dx;
    op.rows = rows;
    op.W = &w;
    op.layernorm = layernorm;
    op.eps = eps;
    impl_->ops.push_back(op);
}

void Gpu::Seq::act_backward(int dpost, int pre, int dpre, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::ActBwd};
    op.a = dpost;
    op.b = pre;
    op.c = dpre;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::glu_backward(int dpost, int pre, int up, int dpre, int dup, int64_t n, GpuAct act) {
    Impl::Op op{Impl::Op::GluBwd};
    op.a = dpost;
    op.b = pre;
    op.c = up;
    op.d = dpre;
    op.e = dup;
    op.n = n;
    op.act = act;
    impl_->ops.push_back(op);
}

void Gpu::Seq::run(bool fp16) {
    Impl& s = *impl_;
    if (s.ops.empty()) return;
    struct Clear {
        std::vector<Impl::Op>& ops;
        ~Clear() { ops.clear(); }
    } clear{s.ops};
    if (s.attempt(fp16)) s.attempt(false);
}

}  // namespace detail
}  // namespace tfi

#endif
