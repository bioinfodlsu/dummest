// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummest-core.hh"
#include "dummest-batch.hh"

struct DPScratchCommon {
    // Combined-score matrices in scalar ExpScore (never overflow).
    ExpMatrix<0> X;
    ExpMatrix<3> prefixOptimal;
    // Rebuilt suffix-optimal values for forward traceback (pair domain)
    ExpMatrix<0> suffixOptimal;

    // Null-model prefix/suffix probabilities in the pair domain (value =
    // m*2^e), so the recursion is plain products/sums with no exp2/log2.
    ExpVector<0> nullProbsPrefix, nullProbsSuffix;
    // EV side buffers in scalar ExpScore (pair domain, never overflows):
    // reverseSideEv needs front padding (writes j-3), forwardSideEv needs
    // back padding (writes j+3).
    ExpVector<3> reverseSideEv;
    ExpVector<0> forwardSideEv;
    std::array<std::vector<AlignedSimilarity>, simdWidth> optProfilePosition;
    std::array<std::vector<bool>, simdWidth> aligned;
    std::vector<uint8_t> transposedDecoded;

    // Per-lane anchor tracking — indexed by sequence position j.
    // bestMidScore holds static-scale-unit pairs (never overflows).
    std::array<std::vector<ExpScore>, simdWidth> bestMidScore;
    std::array<std::vector<int>, simdWidth> bestMidRow;

    // Per-column best trackers. Updated per lane in the hot loop;
    // extracted into the per-lane arrays after the pass.
    std::vector<std::array<ExpScore, simdWidth>> bestMidScorePhys;
    std::vector<std::array<int, simdWidth>> bestMidRowPhys;

    // Current DP column count (= max sequence length in batch)
    int dpWidth = 0;
};

// Bulk DP buffers use simd_t lanes throughout (single instantiation).
struct DPScratch : public DPScratchCommon {
    // Full W1 matrix: simd_t per cell, one value per lane
    Matrix<simd_t, 0, 0> W1;
    // 2-row rolling buffer for the lean score-only filter
    // (findSimilaritiesBackwardOnly): cache-resident, no matrix store.
    std::array<PaddedVec<simd_t, 0, 0>, 2> filterRollingW1;
    // Log2-domain null prefix/suffix for the lean filter (rebuilt per
    // batch; the pair-domain nullProbs* in DPScratchCommon are untouched).
    PaddedVec<simd_t, 0, 0> filterNullProbsPrefix, filterNullProbsSuffix;
    // Cumulative 2^-64 dynamic rescalings per (profile row, lane).
    // Bulk buffers stay single-scale by inheritance: forwardRescaleCount[i] starts as
    // forwardRescaleCount[i-1] (forward ascends), backwardRescaleCount[i] starts as backwardRescaleCount[i+1]
    // (backward descends); triggers only increment the current row.
    std::vector<std::array<int, simdWidth>> forwardRescaleCount, backwardRescaleCount;
    // Row-scale side products for the Float-compute rebuilds (step 5):
    // per-row max X exponent (tracked during formation) and per-batch max
    // side-EV exponents (scanned after each fixup). Write-only until then.
    std::vector<std::array<int, simdWidth>> combinedRowMaxExp;
    std::array<int, simdWidth> reverseSideEvMaxExp, forwardSideEvMaxExp;

    // Reusable temporary buffers for findSimilarities
    PaddedVec<simd_t, 0, 0> W0_curr, W0_next;
    PaddedVec<simd_t, 0, 0> Y0_next, Y0_curr;
    PaddedVec<simd_t, 0, 0> nullModelPrefix, nullModelSuffix;
    // Side accumulators (see PaddedSide): static scale only.
    PaddedSide<simd_t, 3, 0> reverseSide;
    PaddedSide<simd_t, 0, 3> forwardSide;
    PaddedVec<simd_t, 4, 4> bgCodonProbs;
};


uint8_t* buildTransposedDecoded(DPScratchCommon& scratch, int dpWidth, int activeCount,
                                 int zero_idx, const std::array<std::vector<uint8_t>*, simdWidth>& decoded) {
    scratch.transposedDecoded.assign((dpWidth + 8) * simdWidth, zero_idx);
    uint8_t* transposed_base = scratch.transposedDecoded.data() + 4 * simdWidth;
    for (int idx = 0; idx < activeCount; idx++) {
        for (size_t j = 0; j < decoded[idx]->size(); j++) {
            transposed_base[j * simdWidth + idx] = (*decoded[idx])[j];
        }
    }
    return transposed_base;
}

#ifdef ALIGN
// Unified right/left-side EV fixup over simd_t lanes.
// `side` accumulates in simd_t arithmetic at static scale; `ev`
// accumulates per lane in ExpScore so the nullModel multiply and
// prefix/suffix sum can never inf before promotion.
// step=-1 walks j descending with targets j-1/-2/-3 and an ascending prefix
// sum; step=+1 mirrors it (targets j+1/+2/+3, descending suffix sum).
// bgShift is the transposed_base codon offset (-2 right, +1 left).
// seems to be a clean way of getting expected value of null-sided junctions
// TODO: verify logic
// TODO: the ev[j] *= nullModel term might be off by 1 idk

template <int SPad, int SPadPost, int EPad>
void fixupSide(PaddedSide<simd_t, SPad, SPadPost> &side, ExpVector<EPad> &ev,
               const PaddedVec<simd_t, 0, 0> &nullModel,
               simd_t nullEmit1, simd_t nullEmit2, simd_t nullEmit3,
               int step, int bgShift, const uint8_t *transposed_base,
               const Float *bg_probs_ptr, int width, int activeCount,
               const int *cumArr = nullptr) {
    // Per-lane null_emit triples as exact doubles (shared by the side
    // factors below and the EV factors). Stored once per simd_t (not once per
    // lane as before).
    double nullEmit1Dbl[simdWidth], nullEmit2Dbl[simdWidth], nullEmit3Dbl[simdWidth];
    {
        alignas(64) Float b1[simdWidth], b2[simdWidth], b3[simdWidth];
        simd_unchecked_store(nullEmit1, b1, Kokkos::Experimental::simd_flag_default);
        simd_unchecked_store(nullEmit2, b2, Kokkos::Experimental::simd_flag_default);
        simd_unchecked_store(nullEmit3, b3, Kokkos::Experimental::simd_flag_default);
        for (int k = 0; k < activeCount; k++) {
            nullEmit1Dbl[k] = (double)b1[k];
            nullEmit2Dbl[k] = (double)b2[k];
            nullEmit3Dbl[k] = (double)b3[k];
        }
    }
    const double c0base = (double)(Float)(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2);
    const double c1base = (double)(Float)(BACKGROUND_FRAMESHIFT_RATE * 0.25);
    const double c2base = (double)(Float)(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625);
    // Per-lane junction fractions for the ±1/±2 targets (j-invariant).
    double f1A[simdWidth], f2A[simdWidth];
    for (int k = 0; k < activeCount; k++) {
        f1A[k] = c1base * nullEmit1Dbl[k] * (1.0 / 3.0);
        f2A[k] = c2base * nullEmit2Dbl[k] * (2.0 / 3.0);
    }
    simd_t c0v = VecTraits<simd_t>::broadcast((Float)(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2));
    simd_t c1v = VecTraits<simd_t>::broadcast((Float)(BACKGROUND_FRAMESHIFT_RATE * 0.25));
    simd_t c2v = VecTraits<simd_t>::broadcast((Float)(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625));
    simd_t vzero = VecTraits<simd_t>::zero();
    for (int j = step < 0 ? width - 1 : 0; (step < 0) ? (j >= 0) : (j < width); j += step) {
        const char *indices = (const char *)&transposed_base[(j + bgShift) * simdWidth];
        simd_t bgCodonEmitProbs = VecTraits<simd_t>::lookupBg(bg_probs_ptr, indices);

        side[j + 3 * step] = side[j + 3 * step] + c0v * bgCodonEmitProbs * nullEmit3 * side[j];
        side[j + 1 * step] = side[j + 1 * step] + c1v * nullEmit1 * side[j];
        side[j + 2 * step] = side[j + 2 * step] + c2v * nullEmit2 * side[j];

        // EV side in the pair domain, batched (no per-lane stores/frexp):
        // one store per simd_t, libm-free kernels, same op order. Exact.
        // side[j] sits at the row scale recorded in cumArr (kept there by
        // rescaling on every trigger); fold 2^(64*cum) into the exponent.
        ExpScore S[simdWidth];
        double f0[simdWidth], nmD[simdWidth];
        {
            alignas(64) Float sb[simdWidth], bb[simdWidth], nb[simdWidth];
            simd_unchecked_store(side[j], sb, Kokkos::Experimental::simd_flag_default);
            simd_unchecked_store(bgCodonEmitProbs, bb, Kokkos::Experimental::simd_flag_default);
            simd_unchecked_store(nullModel[j], nb, Kokkos::Experimental::simd_flag_default);
            // Promote side[j] to the pair domain, folding the row scale
            // recorded in cumArr (2^(64*cum)) into the exponent.
            alignas(64) float sm[simdWidth];
            alignas(64) int se[simdWidth];
            static const int zeroCum[simdWidth] = {};
            batchPromoteRescaled(sb, cumArr ? cumArr : zeroCum, sm, se, activeCount);
            for (int k = 0; k < activeCount; k++) {
                S[k] = ExpScore{sm[k], se[k]};
                f0[k] = c0base * (double)bb[k] * nullEmit3Dbl[k];
                nmD[k] = (double)nb[k];
            }
        }
        ExpScore m0[simdWidth], m1[simdWidth], m2[simdWidth];
        batchScaleScores(S, f0, m0, activeCount);
        batchScaleScores(S, f1A, m1, activeCount);
        batchScaleScores(S, f2A, m2, activeCount);
        // In-place adds read-then-write the same lane: safe lane-by-lane.
        batchAddScores(ev[j + 3 * step].data(), m0, ev[j + 3 * step].data(), activeCount);
        batchAddScores(ev[j + 1 * step].data(), m1, ev[j + 1 * step].data(), activeCount);
        batchAddScores(ev[j + 2 * step].data(), m2, ev[j + 2 * step].data(), activeCount);
        batchScaleScores(ev[j].data(), nmD, ev[j].data(), activeCount);
        side[j] = VecTraits<simd_t>::vecMax(side[j], vzero);
    }
    // Prefix/suffix sum: ev[j] += ev[j+step], iterating so the source side
    // is already final (opposite direction from the accumulation loop).
    // Batched per row (in-place lane-by-lane: safe). Exact.
    for (int j = step < 0 ? 1 : width - 2;
         (step < 0) ? (j < width) : (j >= 0);
         j -= step) {
        batchAddScores(ev[j + step].data(), ev[j].data(), ev[j].data(), activeCount);
    }
}
#endif

// Row-scale side product: max pair exponent over an EV side buffer.
// Scanned once per fixup (O(width)); feeds the Float-compute rebuild scales.
template <int EPad>
void sideEVMax(const ExpVector<EPad> &ev, int width, int activeCount, int *outMax) {
    for (int k = 0; k < activeCount; k++) outMax[k] = INT_MIN;
    for (int j = 0; j < width; j++) {
        const std::array<ExpScore, simdWidth> &row = ev[j];
        for (int k = 0; k < activeCount; k++) {
            if (row[k].m != 0.0f && row[k].e > outMax[k]) outMax[k] = row[k].e;
        }
    }
}

// Handoff from the backward half to the forward half (split halves): small
// simd_t values. Bulk buffers (W1, reverseSide, null models,
// bg_codon, transposed) stay in scratch.

struct BackwardCtx {
    simd_t nullEmit1, nullEmit2, nullEmit3;
};

Float log2_sum_exp(Float a, Float b) {
    if (a == -INFINITY)
        return b;
    if (b == -INFINITY)
        return a;
    Float m = std::max(a, b);
    return m + log2(exp2(a - m) + exp2(b - m));
}

inline simd_t log2_sum_exp(simd_t a, simd_t b) {
    simd_t m = Kokkos::max(a, b);
    // When both a and b are -inf, a - b = NaN. Clamp to avoid NaN propagation:
    // min(abs(NaN), huge) would still be NaN, so use the fact that
    // m is -inf in that case and -inf + anything finite = -inf.
    simd_t diff = a - b;
    // Replace NaN lanes (from -inf - -inf) with 0: exp2(-0) = 1, harmless.
    // NaN comparison: NaN == NaN is false, so (diff == diff) is false for NaN lanes.
    Kokkos::Experimental::simd_mask<Float> valid = (diff == diff); // false for NaN
    simd_t x = Kokkos::Experimental::condition(valid, Kokkos::abs(diff), simd_t(0));
    return m + Kokkos::log2(simd_t(1.0) + Kokkos::exp2(-x));
}

