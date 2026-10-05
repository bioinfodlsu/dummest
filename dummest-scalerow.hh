// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Row-scaled float compute for the deferred prefix- and suffix-optimal
// rebuilds. The rebuilds take a maximum over candidate scores, so instead of
// carrying each candidate as an ExpScore through the max chain, every
// candidate is shifted to one shared row exponent E, compared as plain
// floats, and only the winner is packed back to an ExpScore.
//
// Why this is sound: multiplying a score by 2^-E is exact (it only changes
// the exponent field), so comparing row-scaled values preserves ordering.
// The additions inside the chain round at float precision and terms far
// below the row maximum flush to zero; this is the accepted fast-compute
// contract -- the maximum itself is exact, and stored values stay ExpScore.

#include "dummest-score.hh"

#include <cstdint>
#include <cstring>

// Shift a normalized score (m, e) to row scale E: returns m * 2^(e-E) as a
// float. Inputs must satisfy e <= E (E is chosen as the row maximum plus one
// exponent of headroom). A zero mantissa, or a result below float range,
// returns 0. The shift changes only the exponent field, so nonzero results
// are exact (no rounding).
inline float scoreToRowScale(float m, int e, int E) {
    uint32_t u;
    std::memcpy(&u, &m, 4);
    int d = e - E;
    d = d > 0 ? 0 : (d < -1000 ? -1000 : d); // legitimate terms never exceed E
    int newExp = 126 + d;
    uint32_t mb = (u & 0x807FFFFFu) | ((uint32_t)(newExp > 0 ? newExp : 0) << 23);
    float s;
    std::memcpy(&s, &mb, 4);
    return (m == 0.0f || newExp <= 0) ? 0.0f : s;
}

// Inverse of scoreToRowScale: pack a row-scaled float back into an ExpScore,
// folding the row exponent E back in. This is an exact frexp replication.
// A zero or subnormal result maps to the zero sentinel.
inline ExpScore scoreFromRowScale(float s, int E) {
    uint32_t u;
    std::memcpy(&u, &s, 4);
    uint32_t ef = (u >> 23) & 0xFFu;
    const bool live = (s > 0.0f) & (ef != 0);
    uint32_t mb = (u & 0x807FFFFFu) | (126u << 23);
    float m;
    std::memcpy(&m, &mb, 4);
    ExpScore o{m, (int)ef - 126 + E};
    return live ? o : ExpScore{0.0f, 0};
}
