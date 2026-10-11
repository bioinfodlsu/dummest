// SPDX-License-Identifier: BSD-3-Clause

// Score layer: scores that never overflow, plus row-halving helpers.
// A score is m * 2^e with m in [0.5, 1) or 0 for zero.
// Rows that grow too large are halved and counted;
// the helpers below undo that count exactly.
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

#include "dummest-poison.hh"

// Static downscale applied to the DP: none (1.0), combined scores use
// mulScoresStaticShifted with zero shift (plain product).
constexpr int STATIC_SHIFT = 0;

// A score as (mantissa, exponent). Eight bytes.
struct ExpScore {
    float m = 0.0f;
    int32_t e = 0;
};
static_assert(sizeof(ExpScore) == 8, "ExpScore must stay 8 bytes");

// Bad-value marker. Helpers fail loudly on it instead of hiding it as zero.
inline ExpScore poisonScore() {
#ifdef DUMMEST_POISON
    return ExpScore{dummestPoisonMantissa(), dummestPoisonExponent()};
#else
    return ExpScore{0.0f, 0};
#endif
}

inline bool isPoisonScore(const ExpScore &a) {
#ifdef DUMMEST_POISON
    return dummestIsPoisonMantissa(a.m) || dummestIsPoisonExp(a.e);
#else
    (void)a;
    return false;
#endif
}

// Poison for a Float DP lane (quiet NaN). Helpers that consume Float lanes
// assert on NaN under DUMMEST_POISON; without it this is a no-op (zero).
template <typename T> inline T poisonFloatLane() { return dummestPoisonFloat<T>(); }

inline bool isPoisonFloatLane(Float v) { return dummestIsPoisonFloat(v); }

// Normalize a non-negative real x into an ExpScore: split x into a mantissa
// in [0.5, 1) and a power of two, folded with e. x == 0 gives the zero
// sentinel.
inline ExpScore normalizeScore(double m, int e) {
    if (m == 0.0) return ExpScore{0.0f, 0};
    int ex = 0;
    m = std::frexp(m, &ex);
    return ExpScore{(float)m, e + ex};
}

// Order two scores. Zero sorts below anything positive.
inline bool scoreLess(const ExpScore &a, const ExpScore &b) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a) && !isPoisonScore(b));
    if (a.m == 0.0f) return b.m != 0.0f;
    if (b.m == 0.0f) return false;
    if (a.e != b.e) return a.e < b.e;
    return a.m < b.m;
}

// Halve step, always an exact power of two.
// Audit builds use a small step so tiny tests trigger halves.
#ifdef AUDIT_FORCE_RESCALE
constexpr int RESCALE_BITS = 8;
#else
constexpr int RESCALE_BITS = 64;
#endif
// One dynamic rescale step, exactly 2^-RESCALE_BITS (written via ldexp rather
// than a product of 0.5 literals so no rounding can creep in).
const Float RESCALE_STEP = std::ldexp((Float)1.0, -RESCALE_BITS);
#ifdef AUDIT_FORCE_RESCALE
const Float RESCALE_THRESH = (Float)0.5; // fires a few times on tiny-model magnitudes
#elif defined(DOUBLE)
const Float RESCALE_THRESH = (Float)1e150; // well below 1e308, headroom for intra-row growth
#else
const Float RESCALE_THRESH = (Float)1e30; // well below 3.4e38, headroom for intra-row growth
#endif

// Bound a cumulative rescale counter so the exponent arithmetic cannot
// overflow int. The bound is far past any real DP size.
inline int clampRescaleCount(int cum) {
    DUMMEST_POISON_ASSERT(!dummestIsPoisonInt(cum));
    if (cum > 1000000) return 1000000;
    if (cum < -1000000) return -1000000;
    return cum;
}

// Convert a stored halved lane back. Bad input gives zero;
// overflow gives the largest finite score.
inline ExpScore promoteRescaledScore(Float v, int cum) {
    // Bad values must fail here, not hide as zero.
    DUMMEST_POISON_ASSERT(!dummestIsPoisonFloat(v));
    DUMMEST_POISON_ASSERT(!dummestIsPoisonInt(cum));
    if (!(v > 0)) return ExpScore{0.0f, 0};
    cum = clampRescaleCount(cum);
    // Largest float strictly below 1 (0.9999999999999999 rounds to 1.0f).
    if (!(v < (Float)INFINITY)) return ExpScore{std::nextafterf(1.0f, 0.0f), 1023 + RESCALE_BITS * cum};
    return normalizeScore((double)v, RESCALE_BITS * cum);
}

// Multiply one row in place. Each factor is 1 or an exact halve, so exact.
inline void rescaleRow(simd_t *base, int lo, int hi, simd_t factor) {
    for (int j = lo; j < hi; ++j) base[j] *= factor;
}

// Pick rows to halve. Returns the halve factors and whether any fired.
inline std::pair<simd_t, bool> makeRowRescaleFactor(std::vector<std::array<int, simdWidth>> &cum, int i,
                                                  const Float *maxArr, int activeCount) {
    alignas(64) Float fbuf[simdWidth];
    bool any = false;
    for (int k = 0; k < activeCount; ++k) {
        // Bad maxima must not hide as "no halve".
        DUMMEST_POISON_ASSERT(!dummestIsPoisonFloat(maxArr[k]));
        DUMMEST_POISON_ASSERT(!dummestIsPoisonInt(cum[i][k]));
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

// Halve factor that maps stored rows back.
inline simd_t rowRescaleVector(const std::array<int, simdWidth> &cum) {
    alignas(64) Float a[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) {
        DUMMEST_POISON_ASSERT(!dummestIsPoisonInt(cum[k]));
        a[k] = std::ldexp((Float)1.0, -RESCALE_BITS * clampRescaleCount(cum[k]));
    }
    return Kokkos::Experimental::simd_unchecked_load<simd_t>(a);
}

// Multiply by a plain number. Bad factor gives zero.
inline ExpScore scaleScore(const ExpScore &a, double f) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a));
    if (a.m == 0.0f) return ExpScore{0.0f, 0};
    if (!(f > 0.0) || !std::isfinite(f)) return ExpScore{0.0f, 0};
    return normalizeScore((double)a.m * f, a.e);
}

// Product of two scores. Used for match scores.
inline ExpScore mulScoresStaticShifted(const ExpScore &a, const ExpScore &b) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a) && !isPoisonScore(b));
    return normalizeScore((double)a.m * (double)b.m, a.e + b.e + STATIC_SHIFT);
}

// Sum of two scores. A piece far below the other is dropped.
inline ExpScore addScores(ExpScore a, ExpScore b) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a) && !isPoisonScore(b));
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

// Plain product, no extra shift. Used for background odds.
inline ExpScore mulScores(const ExpScore &a, const ExpScore &b) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a) && !isPoisonScore(b));
    if (a.m == 0.0f || b.m == 0.0f) return ExpScore{0.0f, 0};
    return normalizeScore((double)a.m * (double)b.m, a.e + b.e);
}

// Lets the DP write a * b, a + b, a < b directly.
inline ExpScore operator*(const ExpScore &a, const ExpScore &b) { return mulScores(a, b); }
inline ExpScore operator+(const ExpScore &a, const ExpScore &b) { return addScores(a, b); }
inline bool operator<(const ExpScore &a, const ExpScore &b) { return scoreLess(a, b); }
inline bool operator>(const ExpScore &a, const ExpScore &b) { return scoreLess(b, a); }

// The exact power 2^e2. e2 == 0 is 1.0.
inline ExpScore pow2Score(int e2) {
    return ExpScore{0.5f, e2 + 1};
}

// Natural logarithm of the value; -inf for the zero sentinel.
inline double scoreToLog(const ExpScore &a) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a));
    if (a.m == 0.0) return -INFINITY;
    static const double LN2 = 0.6931471805599453;
    return std::log(a.m) + (double)a.e * LN2;
}

// Base-2 logarithm of the value; -inf for the zero sentinel.
inline double scoreToLog2(const ExpScore &a) {
    DUMMEST_POISON_ASSERT(!isPoisonScore(a));
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
    DUMMEST_POISON_ASSERT(!isPoisonScore(a));
    if (a.m == 0.0f) return (Float)0.0;
    int e = a.e;
    if (e > (int)EXP2_HI) e = (int)EXP2_HI;
    return std::ldexp((Float)a.m, e);
}
