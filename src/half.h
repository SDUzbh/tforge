// IEEE half precision (fp16) <-> float32 conversion, scalar and in bulk. Internal.
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tfi {
namespace detail {

inline float half_to_float(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --exp;
            }
            bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round to nearest, ties to even (as the hardware conversion); beyond fp16's range gives infinity.
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, ax = x & 0x7FFFFFFF;
    if (ax >= 0x7F800000) return static_cast<uint16_t>(sign | 0x7C00 | (ax > 0x7F800000 ? 0x200 : 0));  // inf, NaN
    if (ax >= 0x477FF000) return static_cast<uint16_t>(sign | 0x7C00);  // >= 65520 rounds to infinity
    if (ax < 0x38800000) {                                               // below 2^-14: subnormal or zero
        if (ax < 0x33000000) return static_cast<uint16_t>(sign);         // below 2^-25 rounds to zero
        const uint32_t e = ax >> 23, m = (ax & 0x7FFFFF) | 0x800000, shift = 126 - e;
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (r & 1))) ++r;
        return static_cast<uint16_t>(sign | r);
    }
    uint32_t r = (ax - 0x38000000) >> 13;  // rebias the exponent from 127 to 15
    const uint32_t rem = ax & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;  // a carry into the exponent is still right
    return static_cast<uint16_t>(sign | r);
}

// Whether v survives the round trip through fp16 without overflowing: finite and |v| <= 65504.
inline bool fits_half(float v) { return v >= -65504.f && v <= 65504.f; }

inline void halves_to_floats(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_high_f32_f16(h));
    }
#endif
    for (; i < n; ++i) dst[i] = half_to_float(src[i]);
}

// Returns how many values did not fit fp16 (fits_half); those become infinity or NaN.
inline int64_t floats_to_halves(const float* src, uint16_t* dst, int64_t n) {
    int64_t bad = 0, i = 0;
#if defined(__aarch64__)
    for (; i + 8 <= n; i += 8) {
        const float32x4_t a = vld1q_f32(src + i), b = vld1q_f32(src + i + 4);
        const float16x8_t h = vcvt_high_f16_f32(vcvt_f16_f32(a), b);
        vst1q_u16(dst + i, vreinterpretq_u16_f16(h));
        const float32x4_t lim = vdupq_n_f32(65504.f);
        // |v| <= 65504 is false for NaN too
        const uint32x4_t oka = vcleq_f32(vabsq_f32(a), lim), okb = vcleq_f32(vabsq_f32(b), lim);
        bad += 8 - static_cast<int64_t>(vaddvq_u32(vshrq_n_u32(oka, 31)) + vaddvq_u32(vshrq_n_u32(okb, 31)));
    }
#endif
    for (; i < n; ++i) {
        bad += !fits_half(src[i]);
        dst[i] = float_to_half(src[i]);
    }
    return bad;
}

}  // namespace detail
}  // namespace tfi
