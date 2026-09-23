// SPDX-License-Identifier: BSD-3-Clause

// Overflow-proof DP support for dummer.cc.
//
// Include AFTER Float, simd_t, and simdWidth are defined (dummer.cc defines
// them from the DOUBLE switch + Kokkos SIMD), since everything here is
// typed in terms of them.
//
// Design recap: bulk DP buffers (W0/W1/Y/Z, left/right_side) stay in SIMD
// Float, with one cumulative power-of-two offset per (profile row, SIMD
// lane). Combined scores (wMid = W0*W1*invScale, X, X_pfx) additionally
// exceed Float range, so they are computed/stored as scalar ExpScore pairs
// (double mantissa + int binary exponent): value = m * 2^e, m in [0.5,1),
// with m == 0 representing zero (which also serves as the -inf sentinel for
// max-tracking, since every valid score is non-negative).
// All formation arithmetic reproduces the static-scale formulas exactly,
// re-bracketed with exact powers of two (invScale folded in at formation).
// Downstream /scale, shift, and calibration code is untouched.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// Packed to 8 bytes: float mantissa + int32 exponent. The mantissa keeps a
// ~2^-24 relative precision, which is the only numeric compromise of the
// packing; all arithmetic below is done in double and narrowed on store.
struct ExpScore {
    float m = 0.0f;
    int32_t e = 0;
};
static_assert(sizeof(ExpScore) == 8, "ExpScore must stay 8 bytes");

inline ExpScore expNormalize(double m, int e) {
    if (m == 0.0) return ExpScore{0.0f, 0};
    int ex = 0;
    m = std::frexp(m, &ex);
    return ExpScore{(float)m, e + ex};
}

// Promote a stored (rescaled) Float lane to static-scale units.
// cum = cumulative 2^-64 rescalings applied to that row/lane, so
// true static-scale value = (double)v * 2^(64*cum).
// NaN/Inf fence: NaN or non-positive maps to zero; a +inf lane (bulk
// overflowed before the rescale trigger could fire) saturates to the
// largest finite pair at the lane's own exponent, so true ordering is
// kept and no inf ever enters pair arithmetic.
inline int clampCum(int cum) {
    if (cum > 1000000) return 1000000;
    if (cum < -1000000) return -1000000;
    return cum;
}

inline ExpScore expFromScaled(Float v, int cum) {
    if (!(v > 0)) return ExpScore{0.0f, 0};
    cum = clampCum(cum);
    // Largest float strictly below 1 (0.9999999999999999 rounds to 1.0f).
    if (!(v < (Float)INFINITY)) return ExpScore{std::nextafterf(1.0f, 0.0f), 1023 + 64 * cum};
    return expNormalize((double)v, 64 * cum);
}

inline bool expLess(const ExpScore &a, const ExpScore &b) {
    if (a.m == 0.0f) return b.m != 0.0f;
    if (b.m == 0.0f) return false;
    if (a.e != b.e) return a.e < b.e;
    return a.m < b.m;
}

// Scale a pair value by a finite non-negative double factor (O(1) constants
// such as bg*null_emit or null_model). Non-positive/non-finite factors map
// to zero so no inf ever enters pair arithmetic.
inline ExpScore expMulFloat(const ExpScore &a, double f) {
    if (a.m == 0.0f) return ExpScore{0.0f, 0};
    if (!(f > 0.0) || !std::isfinite(f)) return ExpScore{0.0f, 0};
    return expNormalize((double)a.m * f, a.e);
}

// Combined score (a * b) * invScale with invScale = 2^63 (scale = 2^-63),
// folded into the exponent as +63.
inline ExpScore expMulInvScale(const ExpScore &a, const ExpScore &b) {
    return expNormalize((double)a.m * (double)b.m, a.e + b.e + 63);
}

// Sum of two non-negative pair values. Distant exponents (>60 apart) return
// the larger operand; the dropped relative contribution is below 2^-60.
inline ExpScore expAdd(ExpScore a, ExpScore b) {
    if (a.m == 0.0f) return b;
    if (b.m == 0.0f) return a;
    if (a.e >= b.e) {
        int d = a.e - b.e;
        if (d > 60) return a;
        return expNormalize((double)a.m + std::ldexp((double)b.m, -d), a.e);
    }
    int d = b.e - a.e;
    if (d > 60) return b;
    return expNormalize((double)b.m + std::ldexp((double)a.m, -d), b.e);
}

inline double expToLog(const ExpScore &a) {
    if (a.m == 0.0) return -INFINITY;
    static const double LN2 = 0.6931471805599453;
    return std::log(a.m) + (double)a.e * LN2;
}

// Materialize in static-scale units as double. Only a true ratio above ~1e308
// yields +inf here (by design); everything below double range stays finite.
inline double expToDouble(const ExpScore &a) {
    if (a.m == 0.0) return 0.0;
    double d = std::ldexp(a.m, a.e);
    if (std::isfinite(d)) return d;
    return INFINITY;
}

// One dynamic rescale step: exact in binary FP for any finite input
// (written as ldexp, not a product of 0.5 literals, so it is exactly 2^-64).
const Float RESCALE_STEP = std::ldexp((Float)1.0, -64); // 2^-64
#ifdef DOUBLE
const Float RESCALE_THRESH = (Float)1e150; // well below 1e308, headroom for intra-row growth
#else
const Float RESCALE_THRESH = (Float)1e30; // well below 3.4e38, headroom for intra-row growth
#endif
#ifdef DOUBLE
const Float EXP2_HI = (Float)1023; // exp2 overflow boundary for double (1024 is already +inf)
#else
const Float EXP2_HI = (Float)120; // exp2 overflow boundary for float
#endif

// Scalar per-lane ExpScore vector for EV side buffers (left_side_EV,
// right_side_EV). PrePad supports negative indexing (right_side_EV writes
// j-3); operator[] exposes the per-lane array used by the fixup loops.
template <int PrePad = 0>
class ExpVector {
    std::vector<std::array<ExpScore, simdWidth>> data_;
public:
    ExpVector() = default;

    void assign(int n) {
        data_.assign((size_t)n + PrePad, std::array<ExpScore, simdWidth>{});
    }

    std::array<ExpScore, simdWidth> &operator[](int j) { return data_[(size_t)PrePad + j]; }
    const std::array<ExpScore, simdWidth> &operator[](int j) const { return data_[(size_t)PrePad + j]; }
};

// Scalar per-lane ExpScore matrix for combined scores (X, X_pfx, Wopt).
// Mirrors the SimdMatrix indexing API (get_lane/set_lane/assign) so call
// sites translate directly; PrePad supports negative-column reads (j-3).
template <int PrePad = 0>
class ExpMatrix {
    std::vector<std::array<ExpScore, simdWidth>> rows_;
    int logical_cols_ = 0;
public:
    ExpMatrix() = default;

    void assign(size_t nrows, int ncols) {
        logical_cols_ = ncols;
        rows_.assign(nrows * (ncols + PrePad), std::array<ExpScore, simdWidth>{});
    }

    const ExpScore &get_lane(size_t i, int j, int k) const {
        return rows_[i * (logical_cols_ + PrePad) + (j + PrePad)][k];
    }
    void set_lane(const ExpScore &v, size_t i, int j, int k) {
        rows_[i * (logical_cols_ + PrePad) + (j + PrePad)][k] = v;
    }
    int cols() const { return logical_cols_; }
    size_t rows() const {
        size_t stride = (size_t)logical_cols_ + PrePad;
        return stride == 0 ? 0 : rows_.size() / stride;
    }
};

// Multiply one SIMD row by a prebuilt per-lane factor in place (vectorized).
// The factor lanes are 1.0 or an exact power of two, so this is exact.
inline void rescaleVec(simd_t *base, int lo, int hi, simd_t factor) {
    for (int j = lo; j < hi; ++j) base[j] *= factor;
}

// Build the per-lane rescale factor for one DP row: triggered lanes get
// 2^-64 (and their cumulative counter increments), the rest 1.0.
// NaN row-max never triggers (NaN > THRESH is false), matching the
// historic if (!(max > THRESH)) continue idiom.
inline std::pair<simd_t, bool> buildRescaleFactor(std::vector<std::array<int, simdWidth>> &cum, int i,
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

// Per-lane 2^(-64*cum) factor vector for the effective static scale.
inline simd_t simdRowScale(const std::array<int, simdWidth> &cum) {
    alignas(64) Float a[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) {
        a[k] = std::ldexp((Float)1.0, -64 * clampCum(cum[k]));
    }
    return Kokkos::Experimental::simd_unchecked_load<simd_t>(a);
}
