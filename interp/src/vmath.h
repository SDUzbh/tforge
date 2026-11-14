// Vectorizable float math for the hot elementwise loops (activations, softmax, norms).
// Internal; not part of the public API.
//
// std::exp, std::erf and std::tanh are scalar library calls, so a loop over them runs one element
// at a time. These are branch-free polynomial versions that the compiler turns into SIMD code when
// they're called in a loop. Worst error against double precision over every float in range:
// exp 1.02 ulp, erf 2.5 ulp, tanh 1.33 ulp (unit test "vector math" checks a sample).
// The reductions use 8 independent accumulators so they vectorize without -ffast-math.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tfi {
namespace vm {

inline float from_bits(int32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof f);
    return f;
}

// 2^n as a float, for integer n in [-126, 127].
inline float pow2i(int32_t n) { return from_bits((n + 127) << 23); }

// e^x, with exp(-inf) = 0, exp(inf) = inf, NaN in = NaN out. Results that would be subnormal
// are, to within an ulp of the smallest normal float.
inline float exp(float x) {
    const float xc = x > -104.f ? (x < 88.8f ? x : 88.8f) : -104.f;  // NaN goes to -104 here
    // n = round(x / ln 2) via the 1.5 * 2^23 trick (valid without -ffast-math), r = x - n ln 2.
    const float n = (xc * 1.44269504088896341f + 12582912.f) - 12582912.f;
    const float r = (xc - n * 0.693359375f) - n * -2.12194440e-4f;
    // e^r on |r| <= ln(2)/2 (Cephes expf).
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.f;
    // Scale by 2^n in two halves so every n in [-150, 128] stays representable.
    const int32_t ni = static_cast<int32_t>(n), h = ni / 2;
    float y = p * pow2i(h) * pow2i(ni - h);
    y = x > 88.7228394f ? std::numeric_limits<float>::infinity() : y;
    y = x < -103.98f ? 0.f : y;
    return x != x ? x : y;
}

// erf(x). |x| < 0.84375: x P(x^2); beyond: 1 - e^{-x^2} R(1/(1+|x|)), which rounds to +-1 from |x| = 4.
inline float erf(float x) {
    const float a = std::fabs(x);
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
    const float big = 1.f - vm::exp(-ac * ac) * q;
    const float y = a < 0.84375f ? small : (x < 0.f ? -big : big);
    return x != x ? x : y;
}

// tanh(x). |x| < 0.625: x P(x^2); beyond: 1 - 2 / (e^{2|x|} + 1), which rounds to +-1 from |x| = 10.
inline float tanh(float x) {
    const float a = std::fabs(x);
    const float s = x * x;
    float p = -5.69752231e-3f;
    p = p * s + 2.06303075e-2f;
    p = p * s + -5.37358634e-2f;
    p = p * s + 1.33313656e-1f;
    p = p * s + -3.33332747e-1f;
    p = p * s + 1.f;
    const float small = x * p;
    const float ac = a < 10.f ? a : 10.f;
    const float big = 1.f - 2.f / (vm::exp(2.f * ac) + 1.f);
    const float y = a < 0.625f ? small : (x < 0.f ? -big : big);
    return x != x ? x : y;
}

// Sum of v[0..n), accumulated in 8 lanes.
inline float sum(const float* v, int64_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) acc[k] += v[i + k];
    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; ++i) s += v[i];
    return s;
}

// Sum of (v[i] - c)^2 over v[0..n), accumulated in 8 lanes.
inline float sum_sq_dev(const float* v, int64_t n, float c) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) {
            const float d = v[i + k] - c;
            acc[k] += d * d;
        }
    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; ++i) s += (v[i] - c) * (v[i] - c);
    return s;
}

// Largest of v[0..n), skipping NaN (like a std::max loop); -inf for n = 0.
inline float max(const float* v, int64_t n) {
    const float ninf = -std::numeric_limits<float>::infinity();
    float acc[8] = {ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf};
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) acc[k] = v[i + k] > acc[k] ? v[i + k] : acc[k];
    float m = ninf;
    for (int k = 0; k < 8; ++k) m = acc[k] > m ? acc[k] : m;
    for (; i < n; ++i) m = v[i] > m ? v[i] : m;
    return m;
}

// In-place softmax of v[0..n).
inline void softmax(float* v, int64_t n) {
    if (n <= 0) return;
    const float mx = vm::max(v, n);
    for (int64_t i = 0; i < n; ++i) v[i] = vm::exp(v[i] - mx);
    const float total = vm::sum(v, n);
    for (int64_t i = 0; i < n; ++i) v[i] /= total;
}

}  // namespace vm
}  // namespace tfi
