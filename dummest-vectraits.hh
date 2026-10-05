// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// VecTraits<simd_t>: the small per-lane operation surface the DP body uses,
// so DP code can stay free of SIMD intrinsics. Only the simd_t (Float lane)
// specialization is instantiated.

#include "dummest-score.hh"
#include "can_i_haz_simd.hh" // simdLookup

template <typename Vec> struct VecTraits;

template <> struct VecTraits<simd_t> {
    using lane_type = Float;
    using buf_type = Float; // plain-lane staging buffer for fused extraction

    // An all-zero lane vector.
    static inline simd_t zero() { return simd_t((Float)0.0); }

    // Splat one scalar value across every lane.
    static inline simd_t broadcast(Float v) { return simd_t(v); }

    // Materialize scores as Float lanes, each exponent clamped so degenerate
    // input stays finite. Lanes at or beyond activeCount are zero.
    static inline simd_t packScores(const ExpScore *arr, int activeCount) {
        alignas(64) Float buf[simdWidth];
        for (size_t k = 0; k < (size_t)simdWidth; k++)
            buf[k] = ((int)k < activeCount) ? scoreToFloatClamped(arr[k]) : (Float)0.0;
        return Kokkos::Experimental::simd_unchecked_load<simd_t>(buf);
    }

    // Elementwise arithmetic on lane vectors.
    static inline simd_t add(simd_t a, simd_t b) { return a + b; }
    static inline simd_t mul(simd_t a, simd_t b) { return a * b; }
    static inline simd_t fma(simd_t a, simd_t b, simd_t c) { return Kokkos::fma(a, b, c); }
    static inline simd_t vecMax(simd_t a, simd_t b) { return Kokkos::max(a, b); }

    // Gather probabilities by per-lane alphabet index (ptr[0] is the first
    // real symbol, the caller offsets past punctuation).
    static inline simd_t lookupEmit(const Float *ptr, const char *idx) {
        return simd_t(simdLookup(ptr, idx));
    }
    static inline simd_t lookupBg(const Float *ptr, const char *idx) {
        return simd_t(simdLookup(ptr, idx));
    }

    // Per-lane mask: 1.0 where column j lies within that lane's sequence
    // length, 0.0 otherwise (sequences shorter than the batch maximum).
    static inline simd_t activeMask(int j, const Float *seqLenF) {
        simd_t lenVec = Kokkos::Experimental::simd_unchecked_load<simd_t>(seqLenF);
        simd_t jVec((Float)j);
        return Kokkos::Experimental::condition(jVec < lenVec, simd_t((Float)1), simd_t((Float)0));
    }
};
