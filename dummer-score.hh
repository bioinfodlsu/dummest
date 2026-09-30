// SPDX-License-Identifier: BSD-3-Clause

// Score layer: overflow-proof ExpScore values plus the scalar arithmetic,
// rescaling bookkeeping, and packing used by the DP and traceback.
//
// An ExpScore represents a non-negative real as a normalized pair (m, e):
// value = m * 2^e, where m is in [0.5, 1) or m == 0 (the zero sentinel).
// The exponent is a full int, so values far outside double range stay
// finite and exactly ordered; the float mantissa fixes ~2^-24 relative
// precision. Arithmetic is carried out in double and narrowed on store.
//
// Rescaling: to keep the Float DP lanes finite, any row/lane whose magnitude
// crosses RESCALE_THRESH is divided by 2^64 and its cumulative count is
// incremented; promoteRescaledScore/rowRescaleVector undo that exactly.
//
// Requires Float, simd_t, and simdWidth (defined by dummer-core.hh from the
// DOUBLE switch and the Kokkos SIMD type).
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Static downscale applied to the DP: none (1.0), combined scores use
// mulScoresStaticShifted with zero shift (plain product).
constexpr int STATIC_SHIFT = 0;

// A non-negative score as (mantissa, exponent): value = m * 2^e with
// m in [0.5, 1) or m == 0 meaning zero. Eight bytes so matrices stay small.
struct ExpScore {
    float m = 0.0f;
    int32_t e = 0;
};
static_assert(sizeof(ExpScore) == 8, "ExpScore must stay 8 bytes");

// Normalize a non-negative real x into an ExpScore: split x into a mantissa
// in [0.5, 1) and a power of two, folded with e. x == 0 gives the zero
// sentinel.
inline ExpScore normalizeScore(double m, int e) {
    if (m == 0.0) return ExpScore{0.0f, 0};
    int ex = 0;
    m = std::frexp(m, &ex);
    return ExpScore{(float)m, e + ex};
}

// Order two scores by their real value; the zero sentinel sorts below any
// positive score.
inline bool scoreLess(const ExpScore &a, const ExpScore &b) {
    if (a.m == 0.0f) return b.m != 0.0f;
    if (b.m == 0.0f) return false;
    if (a.e != b.e) return a.e < b.e;
    return a.m < b.m;
}

// One dynamic rescale step, exactly 2^-64 (written via ldexp rather than a
// product of 0.5 literals so no rounding can creep in).
const Float RESCALE_STEP = std::ldexp((Float)1.0, -64); // 2^-64
#ifdef DOUBLE
const Float RESCALE_THRESH = (Float)1e150; // well below 1e308, headroom for intra-row growth
#else
const Float RESCALE_THRESH = (Float)1e30; // well below 3.4e38, headroom for intra-row growth
#endif

// Bound a cumulative rescale counter so the exponent arithmetic cannot
// overflow int. The bound is far past any real DP size.
inline int clampRescaleCount(int cum) {
    if (cum > 1000000) return 1000000;
    if (cum < -1000000) return -1000000;
    return cum;
}

// Convert a stored (rescaled) Float lane back to static-scale units:
// value = v * 2^(64*cum), where cum is the number of 2^-64 rescalings that
// row/lane has accumulated. Non-positive input (including NaN) becomes zero;
// a +inf lane (one that overflowed before a rescale could fire) saturates to
// the largest finite score at that lane's exponent, preserving its ordering
// without letting +inf enter pair arithmetic.
inline ExpScore promoteRescaledScore(Float v, int cum) {
    if (!(v > 0)) return ExpScore{0.0f, 0};
    cum = clampRescaleCount(cum);
    // Largest float strictly below 1 (0.9999999999999999 rounds to 1.0f).
    if (!(v < (Float)INFINITY)) return ExpScore{std::nextafterf(1.0f, 0.0f), 1023 + 64 * cum};
    return normalizeScore((double)v, 64 * cum);
}

// Multiply one DP row in place by a per-lane factor. Each factor lane is
// 1.0 or an exact power of two, so the operation is exact in binary FP.
inline void rescaleRow(simd_t *base, int lo, int hi, simd_t factor) {
    for (int j = lo; j < hi; ++j) base[j] *= factor;
}

// Decide which lanes of DP row i must be rescaled. A lane whose row maximum
// exceeds RESCALE_THRESH gets factor 2^-64 and its cumulative counter is
// bumped; the others get 1.0. Returns the per-lane factor and whether any
// lane triggered. A NaN maximum triggers nothing.
inline std::pair<simd_t, bool> makeRowRescaleFactor(std::vector<std::array<int, simdWidth>> &cum, int i,
                                                  const Float *maxArr, int activeCount) {
    alignas(64) Float fbuf[simdWidth];
    bool any = false;
    for (int k = 0; k < activeCount; ++k) {
        if (maxArr[k] > RESCALE_THRESH) {
            cum[i][k]++;
            fbuf[k] = RESCALE_STEP;
            any = true;
        } else {
            fbuf[k] = (Float)1.0;
        }
    }
    for (int k = activeCount; k < (int)simdWidth; ++k) fbuf[k] = (Float)1.0;
    return {Kokkos::Experimental::simd_unchecked_load<simd_t>(fbuf), any};
}

// Per-lane factor 2^(-64*cum) that maps row-scale values back to static
// scale, for the effective static scale of a DP row.
inline simd_t rowRescaleVector(const std::array<int, simdWidth> &cum) {
    alignas(64) Float a[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) {
        a[k] = std::ldexp((Float)1.0, -64 * clampRescaleCount(cum[k]));
    }
    return Kokkos::Experimental::simd_unchecked_load<simd_t>(a);
}

// Multiply a score by a finite non-negative scalar factor (an O(1) constant
// such as a background or emission probability). A non-positive or
// non-finite factor produces the zero sentinel.
inline ExpScore scaleScore(const ExpScore &a, double f) {
    if (a.m == 0.0f) return ExpScore{0.0f, 0};
    if (!(f > 0.0) || !std::isfinite(f)) return ExpScore{0.0f, 0};
    return normalizeScore((double)a.m * f, a.e);
}

// Product of two scores with the fixed DP downscale undone:
// (a * b) * 2^STATIC_SHIFT. This is the combined-score formation used for
// comparison and selection.
inline ExpScore mulScoresStaticShifted(const ExpScore &a, const ExpScore &b) {
    return normalizeScore((double)a.m * (double)b.m, a.e + b.e + STATIC_SHIFT);
}

// Sum of two scores. An addend more than 60 binary exponents below the other
// is dropped: its relative contribution is under 2^-60, below the result's
// representable precision.
inline ExpScore addScores(ExpScore a, ExpScore b) {
    if (a.m == 0.0f) return b;
    if (b.m == 0.0f) return a;
    if (a.e >= b.e) {
        int d = a.e - b.e;
        if (d > 60) return a;
        return normalizeScore((double)a.m + std::ldexp((double)b.m, -d), a.e);
    }
    int d = b.e - a.e;
    if (d > 60) return b;
    return normalizeScore((double)b.m + std::ldexp((double)a.m, -d), b.e);
}

// Plain product of two scores, with no scale adjustment; used by standalone
// probability recursions such as the null model.
inline ExpScore mulScores(const ExpScore &a, const ExpScore &b) {
    if (a.m == 0.0f || b.m == 0.0f) return ExpScore{0.0f, 0};
    return normalizeScore((double)a.m * (double)b.m, a.e + b.e);
}

// Arithmetic operators so the DP body can use *, +, <, > uniformly on scores
// without special-casing the representation.
inline ExpScore operator*(const ExpScore &a, const ExpScore &b) { return mulScores(a, b); }
inline ExpScore operator+(const ExpScore &a, const ExpScore &b) { return addScores(a, b); }
inline bool operator<(const ExpScore &a, const ExpScore &b) { return scoreLess(a, b); }
inline bool operator>(const ExpScore &a, const ExpScore &b) { return scoreLess(b, a); }

// The exact power of two 2^e2 as a score (mantissa 0.5); e2 == 0 is 1.0.
inline ExpScore pow2Score(int e2) {
    return ExpScore{0.5f, e2 + 1};
}

// Natural logarithm of the value; -inf for the zero sentinel.
inline double scoreToLog(const ExpScore &a) {
    if (a.m == 0.0) return -INFINITY;
    static const double LN2 = 0.6931471805599453;
    return std::log(a.m) + (double)a.e * LN2;
}

// Base-2 logarithm of the value; -inf for the zero sentinel.
inline double scoreToLog2(const ExpScore &a) {
    if (a.m == 0.0) return -INFINITY;
    return std::log2((double)a.m) + (double)a.e;
}

#ifdef DOUBLE
const Float EXP2_HI = (Float)1023; // exp2 overflow boundary for double (1024 is already +inf)
#else
const Float EXP2_HI = (Float)120; // exp2 overflow boundary for float
#endif

// Materialize a score as Float, clamping the exponent at the exp2 overflow
// boundary so degenerate input cannot produce +inf.
inline Float scoreToFloatClamped(const ExpScore &a) {
    if (a.m == 0.0f) return (Float)0.0;
    int e = a.e;
    if (e > (int)EXP2_HI) e = (int)EXP2_HI;
    return std::ldexp((Float)a.m, e);
}
