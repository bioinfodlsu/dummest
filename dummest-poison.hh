// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Debug "poison" mode for stale-scratch detection.
//
// Build with -DDUMMEST_POISON (CMake: -DDUMMEST_POISON=ON) to fill
// uninitialized DP scratch with loud sentinels instead of zeros:
//
//   * Float/simd lanes  -> quiet NaN
//   * ExpScore (m, e)    -> { NaN, INT_MAX }  (never a valid normalized score)
//   * int fields         -> INT_MAX (kPoisonInt)
//   * uint8 pads         -> 0xFF
//
// Any read of a poisoned cell must abort via DUMMEST_POISON_ASSERT in the
// score/batch/row-scale helpers (see dummest-score.hh, dummest-batch.hh,
// dummest-scalerow.hh). A poison build that finishes with output identical
// to a normal build proves every scratch read was preceded by a write.
//
// Design notes:
//  * Quiet NaN survives the FTZ/DAZ flush set in ThreadPool/dummest.cc.
//  * Without DUMMEST_POISON every helper below compiles to a no-op / zero
//    so release performance is unaffected.
//  * This header has NO dependency on ExpScore/Float/simd_t: it provides
//    primitives; dummest-score.hh builds poisonScore()/isPoisonScore() on top.
//
// Include AFTER the standard headers; BEFORE any poison-aware logic.
// Requires <limits>, <cmath>, <cstdint>, <cassert> (included here).

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef DUMMEST_POISON
#define DUMMEST_POISON_ENABLED 1
#define DUMMEST_POISON_ASSERT(cond) assert(cond)
#else
#define DUMMEST_POISON_ASSERT(cond) ((void)0)
#endif

// --- ExpScore poison primitives (mantissa NaN + impossible exponent) ---
inline float dummestPoisonMantissa() {
#ifdef DUMMEST_POISON
    return std::numeric_limits<float>::quiet_NaN();
#else
    return 0.0f;
#endif
}

constexpr int32_t dummestPoisonExponent() { return INT32_MAX; }

inline bool dummestIsPoisonMantissa(float m) {
#ifdef DUMMEST_POISON
    return std::isnan(m);
#else
    (void)m;
    return false;
#endif
}

inline bool dummestIsPoisonExp(int32_t e) {
#ifdef DUMMEST_POISON
    return e == dummestPoisonExponent();
#else
    (void)e;
    return false;
#endif
}

// --- Float-lane poison (works for float and double) ---
template <typename T> inline T dummestPoisonFloat() {
#ifdef DUMMEST_POISON
    return std::numeric_limits<T>::quiet_NaN();
#else
    return T(0);
#endif
}

template <typename T> inline bool dummestIsPoisonFloat(T v) {
#ifdef DUMMEST_POISON
    return std::isnan(v);
#else
    (void)v;
    return false;
#endif
}

// --- int-field poison (rescale counters, row indices, exponents) ---
// Valid rescale counts are small (>=0); valid row indices are >= -1.
// INT_MAX is never valid and ++ on it would overflow, so poison-aware code
// must assert before incrementing (see makeRowRescaleFactor).
constexpr int dummestPoisonInt() { return INT_MAX; }

inline bool dummestIsPoisonInt(int v) {
#ifdef DUMMEST_POISON
    return v == dummestPoisonInt();
#else
    (void)v;
    return false;
#endif
}

// --- uint8 pad poison (transposed bases use zero_idx; 0xFF is out of range) ---
constexpr uint8_t dummestPoisonU8() { return 0xFF; }
