// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Batch pair kernels: lane-wise replacements for the scalar ExpScore ops
// that avoid libm calls (no frexp/ldexp) so the DP can run them over a whole
// SIMD batch cheaply.
//
// Every kernel takes arrays indexed by lane and acts on the first n lanes
// (n <= simdWidth); lanes at or beyond n are either left untouched or
// written as the zero sentinel, as documented per kernel. Inputs must be
// normalized-or-zero scores (m in [0.5,1) or m == 0) and non-negative; the
// build runs with flush-to-zero/flush-to-denormal so subnormal values cannot
// arise from arithmetic. Under those conditions each kernel is bit-identical
// to calling the corresponding scalar op lane by lane.
//
// Requires ExpScore (dummer-score.hh) and Float/simdWidth.

#include "dummer-score.hh"

#include <bit>
#include <cmath>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Struct kernels: arrays of ExpScore, one entry per lane.
//
// Each loop is plain integer/double arithmetic. Products of normalized
// mantissas land in [0.25, 1) and sums in [0.5, 2), so renormalizing after a
// multiply or add is a single conditional halving -- exactly what frexp
// would compute over those ranges, without the libm call. Zero and infinity
// inputs follow the same branches as the scalar versions; subnormals map to
// zero (the flush-to-zero policy).
// ---------------------------------------------------------------------------

// Split a positive finite normal double into m in [0.5,1) and an exponent
// with x == (double)m * 2^eAdd, replicating frexp-then-narrow-to-float.
// Input 0 maps to (0,0) and subnormal input flushes to (0,0).
inline void splitNormalizedDouble(double x, float &mOut, int &eAdd) {
    if (x == 0.0) { mOut = 0.0f; eAdd = 0; return; }
    uint64_t u = std::bit_cast<uint64_t>(x);
    int ef = (int)((u >> 52) & 0x7FFULL);
    if (ef == 0x7FF) { mOut = (float)x; eAdd = 0; return; } // inf (NaN impossible here)
    if (ef == 0) { mOut = 0.0f; eAdd = 0; return; } // subnormal: flush (FTZ semantics)
    eAdd = ef - 1022;
    mOut = (float)std::bit_cast<double>((u & 0xFFFFFFFFFFFFFULL) | (1022ULL << 52));
}

// out[k] = A[k] * B[k], optionally times 2^eFold (0 = plain product,
// STATIC_SHIFT = the combined-score formation). Zero inputs yield zero.
// Equal to mulScores on every lane.
inline void batchMulScores(const ExpScore *A, const ExpScore *B, ExpScore *out, int n, int eFold) {
#pragma GCC ivdep
    for (int k = 0; k < n; ++k) {
        const bool nz = (A[k].m != 0.0f) & (B[k].m != 0.0f);
        const double p = (double)A[k].m * (double)B[k].m;
        const int e = A[k].e + B[k].e + eFold;
        const bool lo = (p < 0.5);
        const double ps = lo ? p * 2.0 : p;
        const int es = lo ? e - 1 : e;
        out[k] = nz ? ExpScore{(float)ps, es} : ExpScore{0.0f, 0};
    }
}

// out[k] = A[k] + B[k]. Reproduces the scalar addScores branches, including
// the "drop addends more than 60 exponents down" rule (implemented by capping
// the exponent gap, since the dropped part is below double rounding). Equal
// to addScores on every lane.
inline void batchAddScores(const ExpScore *A, const ExpScore *B, ExpScore *out, int n) {
#pragma GCC ivdep
    for (int k = 0; k < n; ++k) {
        const float am = A[k].m, bm = B[k].m;
        const int ae = A[k].e, be = B[k].e;
        const bool a0 = (am == 0.0f), b0 = (bm == 0.0f);
        // Larger-exponent side first. The d > 60 early-out is reproduced by
        // capping d: 2^-60 shifts below double rounding for [0.5,1) heads
        // (2^-60 < ulp/2), so the capped sum rounds back to the head exactly.
        const bool agebe = (ae >= be);
        const float mh = agebe ? am : bm;
        const float ml = agebe ? bm : am;
        const int eh = agebe ? ae : be;
        const int el = agebe ? be : ae;
        int d = eh - el;
        d = d > 60 ? 60 : d;
        // Small-term scaling by an exact power of two (no rounding when
        // the result stays normal; d <= 60 keeps float-range inputs
        // normal in double).
        double s = (double)mh + (double)ml * (1.0 / (double)(1ULL << (unsigned)d));
        int e = eh;
        const bool hi = (s >= 1.0);
        s = hi ? s * 0.5 : s; // exact, mirrors frexp on [0.5,2)
        e = hi ? e + 1 : e;
        const ExpScore arith{(float)s, e};
        out[k] = a0 ? B[k] : (b0 ? A[k] : arith);
    }
}

// out[k] = A[k] * f[k], where the factors may have any magnitude (unlike
// the constrained kernels above), so the product is normalized generally.
// Non-positive or non-finite factors, and zero inputs, yield zero. Equal to
// scaleScore on every lane.
inline void batchScaleScores(const ExpScore *A, const double *f, ExpScore *out, int n) {
#pragma GCC ivdep
    for (int k = 0; k < n; ++k) {
        const bool a0 = (A[k].m == 0.0f);
        const double fv = f[k];
        const bool fok = (fv > 0.0) && std::isfinite(fv);
        float m;
        int ex;
        // Dummy factor on dead lanes keeps the call total (no UB: finite).
        splitNormalizedDouble(a0 ? 0.0 : (double)A[k].m * (fok ? fv : 1.0), m, ex);
        const ExpScore arith{m, A[k].e + ex};
        out[k] = (a0 || !fok) ? ExpScore{0.0f, 0} : arith;
    }
}

// ---------------------------------------------------------------------------
// Plane (SoA) kernels: the mantissa and exponent streams of a cell are
// passed as separate unit-stride arrays, which vectorizes better than the
// struct kernels. They are bit-identical to the struct/scalar versions on
// normalized-or-zero inputs; subnormals flush to zero.
// ---------------------------------------------------------------------------

static_assert(sizeof(int) == 4, "plane kernels assume 32-bit int lanes");

// Promote Float lanes to (mantissa, exponent) pairs, folding in a per-lane
// 2^64*cum rescale factor so the result is at static scale. This is the
// lane-wise equivalent of promoteRescaledScore: zero/NaN/non-positive lanes
// become zero, +inf saturates, subnormals flush. A full float batch uses an
// AVX2 bit-trick path; everything else falls back to the scalar form.
inline void batchPromoteRescaled(const Float *in, const int *cum, float *mOut, int *eOut, int n) {
    if constexpr (sizeof(Float) == 4) {
        if (n == (int)simdWidth) {
#if defined(__AVX2__)
        // Dead at runtime unless Float is 32-bit (see outer guard); the cast
        // keeps the branch compilable in DOUBLE builds.
        __m256 x = _mm256_loadu_ps((const float *)in);
        __m256i ux = _mm256_castps_si256(x);
        // Zero lanes: non-positive (sign bit covers negatives and -0; +0
        // via ax==0), NaN (inf exponent, dirty significand), subnormals.
        __m256i ax = _mm256_and_si256(ux, _mm256_set1_epi32(0x7FFFFFFF));
        __m256i isZero = _mm256_cmpeq_epi32(ax, _mm256_setzero_si256());
        __m256i isNeg = _mm256_srai_epi32(ux, 31);
        __m256i ef = _mm256_and_si256(_mm256_srli_epi32(ux, 23), _mm256_set1_epi32(0xFF));
        __m256i isSub = _mm256_cmpeq_epi32(ef, _mm256_setzero_si256());
        __m256i efFF = _mm256_cmpeq_epi32(ef, _mm256_set1_epi32(0xFF));
        __m256i sig = _mm256_and_si256(ux, _mm256_set1_epi32(0x7FFFFF));
        __m256i sigZero = _mm256_cmpeq_epi32(sig, _mm256_setzero_si256());
        __m256i isInf = _mm256_and_si256(efFF, sigZero);
        __m256i isNan = _mm256_andnot_si256(sigZero, efFF);
        __m256i zero = _mm256_or_si256(_mm256_or_si256(isZero, isNeg),
                                       _mm256_or_si256(isSub, isNan));
        __m256i c = _mm256_loadu_si256((const __m256i *)cum);
        __m256i cCl = _mm256_min_epi32(_mm256_max_epi32(c, _mm256_set1_epi32(-1000000)),
                                       _mm256_set1_epi32(1000000));
        __m256i e = _mm256_add_epi32(_mm256_sub_epi32(ef, _mm256_set1_epi32(126)),
                                     _mm256_slli_epi32(cCl, 6));
        __m256i mbits = _mm256_or_si256(_mm256_and_si256(ux, _mm256_set1_epi32(0x807FFFFF)),
                                        _mm256_set1_epi32(126 << 23));
        __m256 m = _mm256_castsi256_ps(mbits);
        // +inf saturates to the largest finite pair at its own exponent.
        __m256i eSat = _mm256_add_epi32(_mm256_set1_epi32(1023), _mm256_slli_epi32(cCl, 6));
        m = _mm256_blendv_ps(m, _mm256_set1_ps(std::nextafterf(1.0f, 0.0f)),
                             _mm256_castsi256_ps(isInf));
        e = _mm256_blendv_epi8(e, eSat, isInf);
        m = _mm256_blendv_ps(m, _mm256_setzero_ps(), _mm256_castsi256_ps(zero));
        e = _mm256_blendv_epi8(e, _mm256_setzero_si256(), zero);
        _mm256_storeu_ps(mOut, m);
        _mm256_storeu_si256((__m256i *)eOut, e);
        return;
#endif
        }
    }
    for (int k = 0; k < n; ++k) {
        // Subnormal inputs flush (FTZ policy); everything else replicates
        // promoteRescaledScore bit-for-bit (incl. +inf saturation narrowing).
        if (std::fpclassify(in[k]) == FP_SUBNORMAL) { mOut[k] = 0.0f; eOut[k] = 0; }
        else {
            ExpScore s = promoteRescaledScore(in[k], cum[k]);
            mOut[k] = s.m;
            eOut[k] = s.e;
        }
    }
}

// Plane product: out = (am*2^ae) * (bm*2^be) * 2^eFold, written as planes.
// eFold == STATIC_SHIFT forms the combined wMid score; eFold == 0 is a plain
// product. Zero lanes yield zero. Equal to batchMulScores on every lane.
inline void batchMulScorePlanes(const float *am, const int *ae, const float *bm, const int *be,
                       int eFold, float *mo, int *eo, int n) {
#pragma GCC ivdep
    for (int k = 0; k < n; ++k) {
        const bool nz = (am[k] != 0.0f) & (bm[k] != 0.0f);
        const double p = (double)am[k] * (double)bm[k];
        const int e = ae[k] + be[k] + eFold;
        const bool lo = (p < 0.5);
        const double ps = lo ? p * 2.0 : p;
        const int es = lo ? e - 1 : e;
        mo[k] = nz ? (float)ps : 0.0f;
        eo[k] = nz ? es : 0;
    }
}

