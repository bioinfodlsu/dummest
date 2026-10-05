// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummest-dp.hh"


// Backward half of the unified DP: setup + null models + backward pass over
// W1 (+ reverseSide accumulate) + optional per-lane best
// (filter / calibration score). The forward half (forwardPass) consumes W1
// read-only.
// Batch setup shared by backwardPass: transposed decode, null pair models,
// null_emit triple, buffer sizing, nullModel pair formation, bg precompute.
// Fills ctx.nullEmit1/2/3. W1 bordered-zeroed; reverseSide assigned when
// needAlign. Bulk DP buffers live in scratch.
void setupBatch(const Profile &profile,
             const std::array<std::vector<uint8_t>*, simdWidth> &decoded,
             DPScratch &scratch, int activeCount,
             BackwardCtx &ctx, bool needAlign = true) {
    assert(0 < activeCount && activeCount <= simdWidth);

    int maxSequenceLength = 0;
    for (int idx = 0; idx < activeCount; idx++) {
        maxSequenceLength = std::max(maxSequenceLength, (int)decoded[idx]->size());
    }

    scratch.dpWidth = maxSequenceLength;

    // Instrumentation (D): max DP width across all batches (CAS-max).
    {
        size_t w = (size_t)maxSequenceLength;
        size_t cur = g_maxDpWidth.load(std::memory_order_relaxed);
        while (w > cur && !g_maxDpWidth.compare_exchange_weak(cur, w, std::memory_order_relaxed))
            ;
    }

    int alphabetSize = profile.width - nonLetterWidth;
    int zero_idx = alphabetSize + 4; // new padded index that maps to 0.0

    uint8_t* transposed_base;
    auto &nullProbsPrefix = scratch.nullProbsPrefix;
    auto &nullProbsSuffix = scratch.nullProbsSuffix;

    transposed_base = buildTransposedDecoded(scratch, scratch.dpWidth, activeCount, zero_idx, decoded);

    alignas(64) Float seqLengthArr[simdWidth] = {};
    for (int idx = 0; idx < activeCount; idx++) {
        seqLengthArr[idx] = (Float)decoded[idx]->size();
    }
    // Inactive lanes: length 1 keeps per-lane divisions finite.
    for (int k = activeCount; k < (int)simdWidth; k++) {
        seqLengthArr[k] = 1;
    }

    nullProbsPrefix.assign(scratch.dpWidth + 4);
    nullProbsSuffix.assign(scratch.dpWidth + 4);

    // Null model in the pair domain: value = m * 2^e, so the log-sum-exp
    // recursion becomes ordinary products/sums of non-negative pairs. This
    // removes every per-column exp2/log2 from the null model.
    const double c_full = 1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2;
    const double c_fs1 = BACKGROUND_FRAMESHIFT_RATE * 0.25;
    const double c_fs2 = BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625;
    const Float *bg_lin_ptr = profile.bg_probs.data() + 4;

    // suffix[i] = P(sequence[i..end]): backward recursion, seed 1 at i=width.
    // Batched pair kernels (no frexp/ldexp): per-column factor/row gathers +
    // batchScaleScores/batchAddScores; the partial-codon branch is an exact power
    // of two, so its per-lane value is pure integer arithmetic. Exact.
    float mCfull;
    int eCfull;
    splitNormalizedDouble(0.5 * c_full, mCfull, eCfull);
    double cfs1arr[simdWidth], cfs2arr[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) { cfs1arr[k] = c_fs1; cfs2arr[k] = c_fs2; }
    for (int k = 0; k < activeCount; k++)
        nullProbsSuffix[scratch.dpWidth][k] = pow2Score(0);
    for (int i = scratch.dpWidth - 1; i >= 0; i--) {
        const char* indices = (const char*)&transposed_base[i * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_lin_ptr, indices);
        alignas(64) Float bgA[simdWidth];
        simd_unchecked_store(simd_t(bg_raw), bgA, Kokkos::Experimental::simd_flag_default);
        const ExpScore *suf3 = nullProbsSuffix[i + 3].data();
        const ExpScore *suf1 = nullProbsSuffix[i + 1].data();
        const ExpScore *suf2 = nullProbsSuffix[i + 2].data();
        double f1[simdWidth];
        ExpScore m1[simdWidth], m2[simdWidth];
        for (int k = 0; k < activeCount; k++) {
            int len = (int)seqLengthArr[k];
            f1[k] = (double)bgA[k] * c_full;
            // Masked inputs: batchScaleScores maps zero to zero, so dead
            // branches contribute nothing to the final adds.
            m1[k] = (i < len) ? suf1[k] : ExpScore{0.0f, 0};
            m2[k] = (i + 1 < len) ? suf2[k] : ExpScore{0.0f, 0};
        }
        ExpScore t1a[simdWidth], t1[simdWidth], tfs1[simdWidth], tfs2[simdWidth], tmp[simdWidth];
        batchScaleScores(suf3, f1, t1a, activeCount);
        batchScaleScores(m1, cfs1arr, tfs1, activeCount);
        batchScaleScores(m2, cfs2arr, tfs2, activeCount);
        for (int k = 0; k < activeCount; k++) {
            int len = (int)seqLengthArr[k];
            // scaleScore(pow2Score(-2*(len-i)), c_full): the pow2 part is
            // {0.5, -2*(len-i)+1}, so this is the precomputed constant with
            // a per-lane exponent shift. Matches the scalar bit-for-bit.
            t1[k] = (i + 2 < len) ? t1a[k]
                                  : ExpScore{mCfull, eCfull - 2 * (len - i) + 1};
        }
        auto &dst = nullProbsSuffix[i];
        batchAddScores(tfs1, tfs2, tmp, activeCount);
        batchAddScores(t1, tmp, dst.data(), activeCount);
    }
    // prefix[i] = P(sequence[0..i]): forward recursion, seed 1 at i=width.
    // Same batching as suffix. t2/t3 boundary rows (i==0/1) select a
    // precomputed constant / zero-sentinel row instead of branching; only the
    // t1 boundary (i<3, 3 columns per batch) keeps the scalar original.
    const ExpScore one0 = pow2Score(0);
    ExpScore one0arr[simdWidth], zeroarr[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) { one0arr[k] = one0; zeroarr[k] = ExpScore{0.0f, 0}; }
    for (int k = 0; k < activeCount; k++)
        nullProbsPrefix[scratch.dpWidth][k] = pow2Score(0);
    for (int i = 0; i < scratch.dpWidth; i++) {
        // Vectorized bg emission lookup via transposed_base
        const char* indices = (const char*)&transposed_base[(i - 2) * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_lin_ptr, indices);
        alignas(64) Float bgA[simdWidth];
        simd_unchecked_store(simd_t(bg_raw), bgA, Kokkos::Experimental::simd_flag_default);
        double f1[simdWidth];
        for (int k = 0; k < activeCount; k++) f1[k] = (double)bgA[k] * c_full;
        const ExpScore *r2 = (i > 0) ? nullProbsPrefix[i - 1].data() : one0arr;
        const ExpScore *r3 = (i >= 2) ? nullProbsPrefix[i - 2].data()
                                      : (i == 1 ? one0arr : zeroarr);
        ExpScore t1[simdWidth], t2[simdWidth], t3[simdWidth], tmp[simdWidth], res[simdWidth];
        if (i >= 3) {
            batchScaleScores(nullProbsPrefix[i - 3].data(), f1, t1, activeCount);
        } else {
            // Boundary columns (3 per batch): original scalar expressions.
            for (int k = 0; k < activeCount; k++) {
                if (i == 2) {
                    t1[k] = scaleScore(normalizeScore((double)bgA[k], 0), c_full);
                } else {
                    t1[k] = scaleScore(pow2Score(-2 * (i + 1)), c_full);
                }
            }
        }
        batchScaleScores(r2, cfs1arr, t2, activeCount);
        batchScaleScores(r3, cfs2arr, t3, activeCount);
        batchAddScores(t2, t3, tmp, activeCount);
        batchAddScores(t1, tmp, res, activeCount);
        auto &dst = nullProbsPrefix[i];
        for (int k = 0; k < activeCount; k++) {
            int len = (int)seqLengthArr[k];
            dst[k] = (i < len) ? res[k] : ExpScore{0.0f, 0};
        }
    }

    alignas(64) Float nullSeqLogProb[simdWidth] = {0};
    // Per-lane null_emit triple in the pair domain: ne1 = ne*0.25,
    // ne2 = ne^2*0.0625, ne3 = ne^3.
    ExpScore nullEmitArr[simdWidth]{}, nullEmitInvArr[simdWidth]{};
    ExpScore nullEmit1Arr[simdWidth]{}, nullEmit2Arr[simdWidth]{}, nullEmit3Arr[simdWidth]{};

    for (int idx = 0; idx < activeCount; idx++) {
        int len = (int)seqLengthArr[idx];
        if (len >= 3) {
            ExpScore total = addScores(
                addScores(nullProbsPrefix[len - 1][idx], nullProbsPrefix[len - 2][idx]),
                nullProbsPrefix[len - 3][idx]);
            nullSeqLogProb[idx] = (Float)scoreToLog2(total);
        } else {
            // For very short sequences, use a neutral finite null model.
            // (The old -INFINITY fed exp2(+inf)=+inf into every C_*
            // constant and poisoned the whole lane.)
            nullSeqLogProb[idx] = 0;
        }

        // Clamp: the true value is O(1); anything past the overflow boundary
        // is degenerate input, and must stay finite, never +inf.
        double e = -(double)nullSeqLogProb[idx] / (double)seqLengthArr[idx];
        if (!(e < (double)EXP2_HI)) e = (double)EXP2_HI;
        double ne_d = exp2(e);
        ExpScore ne = normalizeScore(ne_d, 0);
        nullEmitArr[idx] = ne;
        nullEmitInvArr[idx] = normalizeScore(1.0 / ne_d, 0);
        ExpScore q1 = normalizeScore(0.25, 0);
        ExpScore q2 = normalizeScore(0.0625, 0);
        nullEmit1Arr[idx] = ne * q1;
        nullEmit2Arr[idx] = ne * ne * q2;
        nullEmit3Arr[idx] = ne * ne * ne;
    }

    simd_t nullEmit1 = VecTraits<simd_t>::packScores(nullEmit1Arr, activeCount);
    simd_t nullEmit2 = VecTraits<simd_t>::packScores(nullEmit2Arr, activeCount);
    simd_t nullEmit3 = VecTraits<simd_t>::packScores(nullEmit3Arr, activeCount);
    simd_t vzero = VecTraits<simd_t>::zero();

    scratch.W0_curr.resize(scratch.dpWidth + 4, vzero);
    scratch.W0_next.resize(scratch.dpWidth + 4, vzero);

    scratch.W1.resizeNoFill(profile.length + 2, scratch.dpWidth + 4);
    // Bordered zeros: row plen+1 is never written (loop runs i=plen..0) but is
    // read as the i+1 seed row; tail columns are read as shift-register seeds
    // before the owning row writes them. Zero only these (O(W+P)).
    scratch.W1.fillRow(profile.length + 1, vzero);
    for (int bi = 0; bi <= profile.length; bi++) {
        simd_t *wrp = scratch.W1.row_ptr(bi);
        for (int bj = scratch.dpWidth; bj < scratch.dpWidth + 4; bj++)
            wrp[bj] = vzero;
    }

    // Cumulative dynamic-rescale counters, zeroed per batch. Inheritance
    // (backwardRescaleCount[i] starts as backwardRescaleCount[i+1]) happens in backwardPass below.
    scratch.forwardRescaleCount.assign(profile.length + 2, std::array<int, simdWidth>{});
    scratch.backwardRescaleCount.assign(profile.length + 2, std::array<int, simdWidth>{});
    // Per-row max X exponent, seeded to INT_MIN per batch (tracked during
    // forwardPass formation; consumed by the Float-compute rebuilds).
    {
        std::array<int, simdWidth> neg;
        neg.fill(INT_MIN);
        scratch.combinedRowMaxExp.assign(profile.length + 2, neg);
    }

    const size_t padded_seq_len = scratch.dpWidth + 4;
    auto &Y0_next = scratch.Y0_next; Y0_next.resize(padded_seq_len, vzero);
    auto &Y0_curr = scratch.Y0_curr; Y0_curr.resize(padded_seq_len, vzero);

    auto &nullModelPrefix = scratch.nullModelPrefix; nullModelPrefix.resize(padded_seq_len, vzero);
    auto &nullModelSuffix = scratch.nullModelSuffix; nullModelSuffix.resize(padded_seq_len, vzero);

    // nullModel = P * null_emit^k, formed in the pair domain and materialized
    // as simd_t lanes. null_emit^k grows
    // with k while P decays, so the product stays O(1); the pair domain keeps
    // both factors in range. Only the per-lane exp2 above remains.
    {
        // suffix side: exponent = len-1-j (decreasing in j)
        ExpScore nullEmitPow[simdWidth]{};
        for (int k = 0; k < activeCount; k++) nullEmitPow[k] = pow2Score(0);
        // Batched running product (same op order per lane as the scalar
        // loop, so bit-exact): lanes whose length is exhausted keep their
        // value via save/restore. No frexp: batchMulScores is libm-free.
        int maxLen = 0;
        for (int k = 0; k < activeCount; k++)
            maxLen = std::max(maxLen, (int)seqLengthArr[k]);
        ExpScore savePow[simdWidth];
        for (int t = 0; t < maxLen - 1; t++) {
            for (int k = 0; k < activeCount; k++) savePow[k] = nullEmitPow[k];
            batchMulScores(nullEmitPow, nullEmitArr, nullEmitPow, activeCount, 0);
            for (int k = 0; k < activeCount; k++)
                if (t >= (int)seqLengthArr[k] - 1) nullEmitPow[k] = savePow[k];
        }
        for (int j = 0; j < scratch.dpWidth; j++) {
            ExpScore col[simdWidth]{};
            batchMulScores(nullProbsSuffix[j + 1].data(), nullEmitPow, col, activeCount, 0);
            nullModelSuffix[j] = VecTraits<simd_t>::packScores(col, activeCount);
            batchMulScores(nullEmitPow, nullEmitInvArr, nullEmitPow, activeCount, 0);
        }
        // prefix side: exponent = j+1 (increasing in j); null_emit^1 at j=0.
        for (int k = 0; k < activeCount; k++)
            nullEmitPow[k] = nullEmitArr[k]; // null_emit^1
        for (int k = activeCount; k < (int)simdWidth; k++)
            nullEmitPow[k] = ExpScore{0.0f, 0};
        for (int j = 0; j < scratch.dpWidth; j++) {
            ExpScore col[simdWidth]{};
            batchMulScores(nullProbsPrefix[j].data(), nullEmitPow, col, activeCount, 0);
            nullModelPrefix[j] = VecTraits<simd_t>::packScores(col, activeCount);
            batchMulScores(nullEmitPow, nullEmitArr, nullEmitPow, activeCount, 0);
        }
    }
    auto &reverseSide = scratch.reverseSide;
#ifdef ALIGN
    // Scores-only callers (calibration) never consume the alignment buffers:
    // skip allocating the side vectors entirely.
    if (needAlign) {
        // reverseSide accumulates in the backward loop below; forwardSide
        // belongs to the forward half (forwardPass assigns it).
        reverseSide.assign(nullModelPrefix.size(), vzero);
    }
#endif
        const Float *bg_probs_ptr = profile.bg_probs.data() + 4;
        scratch.bgCodonProbs.resize(scratch.dpWidth);
        simd_t* bg_codon_probs_base = scratch.bgCodonProbs.data();
        for (int j = -4; j < scratch.dpWidth + 4; j++) {
            const char* indices = (const char*)&transposed_base[j * simdWidth];
            bg_codon_probs_base[j] = VecTraits<simd_t>::lookupBg(bg_probs_ptr, indices);
        }

    ctx.nullEmit1 = nullEmit1;
    ctx.nullEmit2 = nullEmit2;
    ctx.nullEmit3 = nullEmit3;
}

// Lean score-only filter: float log2-domain backward, 2-row rolling W1
// (cache-resident, no matrix store), no alignment, no ExpScore. Survivors
// are rescored by the slow wave; inf scores always survive (firewall).
void findSimilaritiesBackwardOnly(
    std::array<Float, simdWidth> &bestScores,
    const Profile &profile,
    const std::array<std::vector<uint8_t>*, simdWidth> &decoded,
    DPScratch &scratch, int activeCount) {
    assert(0 < activeCount && activeCount <= simdWidth);

    int maxSequenceLength = 0;
    for (int idx = 0; idx < activeCount; idx++) {
        maxSequenceLength = std::max(maxSequenceLength, (int)decoded[idx]->size());
    }
    scratch.dpWidth = maxSequenceLength;

    int alphabetSize = profile.width - nonLetterWidth;
    int zero_idx = alphabetSize + 4;

    uint8_t* transposed_base = buildTransposedDecoded(scratch, scratch.dpWidth, activeCount, zero_idx, decoded);

    alignas(64) Float seqLengthArr[simdWidth] = {};
    for (int idx = 0; idx < activeCount; idx++)
        seqLengthArr[idx] = (Float)decoded[idx]->size();
    simd_t seqLengths = Kokkos::Experimental::simd_unchecked_load<simd_t>(seqLengthArr);

    auto &nullProbsPrefix = scratch.filterNullProbsPrefix;
    auto &nullProbsSuffix = scratch.filterNullProbsSuffix;
    nullProbsPrefix.resize(scratch.dpWidth + 4);
    nullProbsSuffix.resize(scratch.dpWidth + 4);

    nullProbsSuffix[scratch.dpWidth] = 0;
    for (int i = scratch.dpWidth - 1; i >= 0; i--) {
        const Float *bg_probs_ptr = profile.log2_bg_probs.data() + 4;
        const char* indices = (const char*)&transposed_base[i * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_probs_ptr, indices);
        simd_t bgCodonEmitProbs(bg_raw);

        Kokkos::Experimental::simd_mask<Float> msk = i + 2 < seqLengths;
        Kokkos::Experimental::simd_mask<Float> msk_fs2 = i + 1 < seqLengths;
        Kokkos::Experimental::simd_mask<Float> msk_fs1 = i < seqLengths;
        simd_t full_codon = (Float)log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) + bgCodonEmitProbs + nullProbsSuffix[i + 3];
        simd_t partial_codon = (Float)log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) + (Float)log2(0.25) * (seqLengths - (Float)i);
        simd_t t1 = Kokkos::Experimental::condition(msk, full_codon, partial_codon);

        simd_t fs1 = (Float)log2(BACKGROUND_FRAMESHIFT_RATE * 0.25) + nullProbsSuffix[i + 1];
        simd_t fs2 = (Float)log2(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) + nullProbsSuffix[i + 2];
        simd_t neg_inf_vec((Float)-INFINITY);
        simd_t t_fs1 = Kokkos::Experimental::condition(msk_fs1, fs1, neg_inf_vec);
        simd_t t_fs2 = Kokkos::Experimental::condition(msk_fs2, fs2, neg_inf_vec);

        nullProbsSuffix[i] = log2_sum_exp(t1, log2_sum_exp(t_fs1, t_fs2));
    }

    nullProbsPrefix[scratch.dpWidth] = 0;
    const Float *log2_bg_probs_ptr = profile.log2_bg_probs.data() + 4;
    const Float log2_1_bg_fs = log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2);
    const Float log2_bg_fs_025 = log2(BACKGROUND_FRAMESHIFT_RATE * 0.25);
    const Float log2_bg_fs2_00625 = log2(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625);
    const Float log2_025 = log2(0.25);
    const simd_t simd_log2_1_bg_fs(log2_1_bg_fs);
    const simd_t simd_log2_bg_fs_025(log2_bg_fs_025);
    const simd_t simd_log2_bg_fs2_00625(log2_bg_fs2_00625);
    const simd_t simd_log2_025(log2_025);
    const simd_t simd_neg_inf(-INFINITY);

    for (int i = 0; i < scratch.dpWidth; i++) {
        const char* indices = (const char*)&transposed_base[(i - 2) * simdWidth];
        SimdFloat bg_raw = simdLookup(log2_bg_probs_ptr, indices);
        simd_t bgCodonEmitProbs(bg_raw);

        Kokkos::Experimental::simd_mask<Float> msk_valid = (Float)i < seqLengths;

        simd_t t1;
        if (i >= 3) {
            t1 = simd_log2_1_bg_fs + bgCodonEmitProbs + nullProbsPrefix[i - 3];
        } else if (i == 2) {
            t1 = simd_log2_1_bg_fs + bgCodonEmitProbs;
        } else {
            t1 = simd_log2_1_bg_fs + simd_log2_025 * (Float)(i + 1);
        }

        simd_t t2 = simd_log2_bg_fs_025 + (i > 0 ? nullProbsPrefix[i - 1] : simd_t(0));

        simd_t t3;
        if (i >= 2) {
            t3 = simd_log2_bg_fs2_00625 + nullProbsPrefix[i - 2];
        } else if (i == 1) {
            t3 = simd_log2_bg_fs2_00625 + simd_t(0);
        } else {
            t3 = simd_neg_inf;
        }

        simd_t result = log2_sum_exp(t1, log2_sum_exp(t2, t3));
        nullProbsPrefix[i] = Kokkos::Experimental::condition(msk_valid, result, simd_neg_inf);
    }

    alignas(64) Float nullEmitTmp[simdWidth] = {0}, nullSeqLogProb[simdWidth] = {0};

    for (int idx = 0; idx < activeCount; idx++) {
        if (seqLengthArr[idx] >= 3) {
            nullSeqLogProb[idx] =
            log2_sum_exp(log2_sum_exp(nullProbsPrefix[seqLengthArr[idx] - 1][idx], nullProbsPrefix[seqLengthArr[idx] - 2][idx]),
                nullProbsPrefix[seqLengthArr[idx] - 3][idx]
            );
        } else {
            nullSeqLogProb[idx] = -INFINITY;
        }

        Float null_emit_raw = exp2(-(nullSeqLogProb[idx] / seqLengthArr[idx]));
        nullEmitTmp[idx] = null_emit_raw;
    }

    auto nullEmit1 = Kokkos::Experimental::simd_unchecked_load<simd_t>(nullEmitTmp);
    auto nullEmit2 = nullEmit1 * nullEmit1;
    auto nullEmit3 = nullEmit2 * nullEmit1;

    nullEmit2 *= (Float)(0.25 * 0.25);
    nullEmit1 *= (Float)0.25;

    auto null_seq_log_prob_simd = Kokkos::Experimental::simd_unchecked_load<simd_t>(nullSeqLogProb);
    simd_t inv_actual_seq_len = (Float)1.0 / seqLengths;
    simd_t nullProbPerPos = null_seq_log_prob_simd * inv_actual_seq_len;

    const size_t padded_seq_len = scratch.dpWidth + 4;
    scratch.filterRollingW1[0].resize(padded_seq_len, simd_t(0.0));
    scratch.filterRollingW1[1].resize(padded_seq_len, simd_t(0.0));
    auto &Y0_next = scratch.Y0_next; Y0_next.resize(padded_seq_len, 0.0);
    auto &Y0_curr = scratch.Y0_curr; Y0_curr.resize(padded_seq_len, 0.0);

    auto &nullModelPrefix = scratch.nullModelPrefix; nullModelPrefix.resize(padded_seq_len, 0.0);
    auto &nullModelSuffix = scratch.nullModelSuffix; nullModelSuffix.resize(padded_seq_len, 0.0);

    for (int j = 0; j < scratch.dpWidth; j++) {
        simd_t suffix_exponent = -nullProbPerPos * (seqLengths - (Float)1.0 - (Float)j) + nullProbsSuffix[j + 1];
        nullModelSuffix[j] = Kokkos::exp2(suffix_exponent);

        simd_t prefix_exponent = -nullProbPerPos * (Float)(j + 1) + nullProbsPrefix[j];
        nullModelPrefix[j] = Kokkos::exp2(prefix_exponent);
    }

    const Float *bg_probs_ptr = profile.bg_probs.data() + 4;
    scratch.bgCodonProbs.resize(scratch.dpWidth);
    simd_t* bg_codon_probs_base = scratch.bgCodonProbs.data();
    for (int j = -4; j < scratch.dpWidth + 4; j++) {
        const char* indices = (const char*)&transposed_base[j * simdWidth];
        bg_codon_probs_base[j] = simd_t(simdLookup(bg_probs_ptr, indices));
    }

    simd_t globalBest = simd_t(0.0);

    for (int i = profile.length; i >= 0; i--) {
        const Params &params = profile.values_v2[i];
        const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

        const simd_t enterCoeff = params.enter_match_probability * nullEmit3;
        const simd_t delta0Coeff = params.delta_prime[0];
        const simd_t delta1Coeff = params.delta_prime[1] * nullEmit2;
        const simd_t delta2Coeff = params.delta_prime[2] * nullEmit1;
        const simd_t alpha0Coeff = params.alpha_prime[0] * nullEmit3;
        const simd_t alpha1Coeff = params.alpha_prime[1] * nullEmit1;
        const simd_t alpha2Coeff = params.alpha_prime[2] * nullEmit2;
        const simd_t beta0Coeff = params.beta_prime[0] * nullEmit3;
        simd_t Z0_ring[4] = {0, 0, 0, 0};
        const simd_t epsCoeff = params.epsilon_prime;
        const simd_t rowScaleCoeff = scale;

        simd_t w1NextJ3 = scratch.filterRollingW1[(i + 1) & 1][scratch.dpWidth + 3];
        simd_t w1NextJ2 = scratch.filterRollingW1[(i + 1) & 1][scratch.dpWidth + 2];
        simd_t w1NextJ1 = scratch.filterRollingW1[(i + 1) & 1][scratch.dpWidth + 1];
        simd_t w1CurJ2 = scratch.filterRollingW1[i & 1][scratch.dpWidth + 2];
        simd_t w1CurJ1 = scratch.filterRollingW1[i & 1][scratch.dpWidth + 1];

        for (int j = scratch.dpWidth - 1; j >= 0; j--) {
            int r_0 = j & 3;
            int r_3 = (j + 3) & 3;

            const char* indices = (const char*)&transposed_base[(j + 1) * simdWidth];
            SimdFloat codon_raw = simdLookup(params_emission_probabilities, indices);
            simd_t codonEmitProbs(codon_raw);
            simd_t bgCodonEmitProbs = bg_codon_probs_base[j + 1];

            simd_t w1NextAtJ = scratch.filterRollingW1[(i + 1) & 1][j];

            simd_t wVal =
                w1NextJ3 * codonEmitProbs * enterCoeff +
                Y0_next[j + 0] * delta0Coeff +
                w1NextJ2 * delta1Coeff +
                w1NextJ1 * delta2Coeff +
                Z0_ring[r_3] * bgCodonEmitProbs * alpha0Coeff +
                w1CurJ2 * alpha2Coeff +
                w1CurJ1 * alpha1Coeff
             + nullModelSuffix[j] * rowScaleCoeff;

            simd_t mid_score = wVal * nullModelPrefix[j];
            globalBest = Kokkos::max(globalBest, mid_score);

            scratch.filterRollingW1[i & 1][j] = wVal;

            Y0_curr[j] = Kokkos::fma(epsCoeff, Y0_next[j], wVal);
            simd_t z0_future = Z0_ring[r_3] * bgCodonEmitProbs;
            Z0_ring[r_0] = Kokkos::fma(beta0Coeff, z0_future, wVal);

            w1NextJ3 = w1NextJ2;
            w1NextJ2 = w1NextJ1;
            w1NextJ1 = w1NextAtJ;
            w1CurJ2 = w1CurJ1;
            w1CurJ1 = wVal;
        }

        std::swap(Y0_curr, Y0_next);
    }

    alignas(64) Float best_arr[simdWidth];
    simd_unchecked_store(globalBest, best_arr, Kokkos::Experimental::simd_flag_default);
    for (int k = 0; k < activeCount; k++)
        bestScores[k] = best_arr[k];
    for (int k = activeCount; k < simdWidth; k++)
        bestScores[k] = 0;
}

// Backward half: i-descending DP over W1 (+ reverseSide accumulate) +
// overflow gate + optional per-lane best. W1 bordered-zeroed by setupBatch;
// this loop overwrites rows 0..plen (row plen+1 stays the zero seed).

void backwardPass(const Profile &profile,
             const std::array<std::vector<uint8_t>*, simdWidth> &decoded,
             DPScratch &scratch, int activeCount,
             std::array<ExpScore, simdWidth> *bestOut, // filter/calibration best (null to skip)
             BackwardCtx &ctx, bool needAlign = true) {
    setupBatch(profile, decoded, scratch, activeCount, ctx, needAlign);
    simd_t vzero = VecTraits<simd_t>::zero();
    const simd_t nullEmit1 = ctx.nullEmit1;
    const simd_t nullEmit2 = ctx.nullEmit2;
    const simd_t nullEmit3 = ctx.nullEmit3;
    uint8_t *transposed_base = scratch.transposedDecoded.data() + 4 * simdWidth;
    simd_t *bg_codon_probs_base = scratch.bgCodonProbs.data();
    auto &nullModelPrefix = scratch.nullModelPrefix;
    auto &nullModelSuffix = scratch.nullModelSuffix;
    auto &reverseSide = scratch.reverseSide;
    auto &Y0_next = scratch.Y0_next;
    auto &Y0_curr = scratch.Y0_curr;
    const size_t padded_seq_len = scratch.dpWidth + 4;

        {
        auto gather_w1 = [&](int i_row, int col) -> simd_t {
            return scratch.W1.get(i_row, col);
        };
        auto scatter_w1 = [&](simd_t val, int i_row, int col) {
            scratch.W1.set(i_row, col, val);
        };

        for (int i = profile.length; i >= 0; i--) {
            // Dynamic-scale inheritance: this row starts at the next row's
            // scale, so all W1/Y0 reads below are single-scale by construction.
            if (i < profile.length)
                scratch.backwardRescaleCount[i] = scratch.backwardRescaleCount[i + 1];
            const Params &params = profile.values_v2[i];
            const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

            const simd_t enterCoeff = VecTraits<simd_t>::broadcast(params.enter_match_probability) * nullEmit3;
            const simd_t delta0Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[0]);
            const simd_t delta1Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[1]) * nullEmit2;
            const simd_t delta2Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[2]) * nullEmit1;
            const simd_t alpha0Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[0]) * nullEmit3;
            const simd_t alpha1Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[1]) * nullEmit1;
            const simd_t alpha2Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[2]) * nullEmit2;
            const simd_t beta0Coeff = VecTraits<simd_t>::broadcast(params.beta_prime[0]) * nullEmit3;
            simd_t Z0_ring[4] = {vzero, vzero, vzero, vzero};

            const simd_t epsCoeff = VecTraits<simd_t>::broadcast(params.epsilon_prime);
            // Effective static scale for this row/lane: the seed term is
            // stored in row scale.
            simd_t rowScaleCoeff = VecTraits<simd_t>::broadcast(scale) * rowRescaleVector(scratch.backwardRescaleCount[i]);
            simd_t backwardRowMax = vzero;

            int rev_lo = 0, rev_hi = scratch.dpWidth - 1;

            // Shift registers to eliminate redundant W1 gathers.
            // Between consecutive j iterations, 3/5 W1 reads overlap:
            //   iter j:   reads (i+1,j+1),(i+1,j+2),(i+1,j+3),(i,j+1),(i,j+2)
            //   iter j-1: reads (i+1,j),  (i+1,j+1),(i+1,j+2),(i,j),  (i,j+1)
            // We keep the last 5 columns as shift registers and only gather
            // the one truly new value W1[i+1][j] each iteration.
            // W1[i][j] is wVal itself — already in register, no gather needed.
            simd_t w1NextJ3 = gather_w1(i + 1, rev_hi + 3);
            simd_t w1NextJ2 = gather_w1(i + 1, rev_hi + 2);
            simd_t w1NextJ1 = gather_w1(i + 1, rev_hi + 1);
            simd_t w1CurJ2 = gather_w1(i, rev_hi + 2);
            simd_t w1CurJ1 = gather_w1(i, rev_hi + 1);

            for (int j = rev_hi; j >= rev_lo; j--) {
                int r_0 = j & 3;
                int r_3 = (j + 3) & 3;

                const char* indices = (const char*)&transposed_base[(j + 1) * simdWidth];
                simd_t codonEmitProbs = VecTraits<simd_t>::lookupEmit(params_emission_probabilities, indices);
                simd_t bgCodonEmitProbs = bg_codon_probs_base[j + 1];

                // Only gather the one new column needed for next iteration
                simd_t w1NextAtJ = gather_w1(i + 1, j);

                simd_t wVal =
                    w1NextJ3 * codonEmitProbs * enterCoeff +
                    Y0_next[j + 0] * delta0Coeff +
                    w1NextJ2 * delta1Coeff +
                    w1NextJ1 * delta2Coeff +
                    Z0_ring[r_3] * bgCodonEmitProbs * alpha0Coeff +
                    w1CurJ2 * alpha2Coeff +
                    w1CurJ1 * alpha1Coeff
                 + nullModelSuffix[j] * rowScaleCoeff;

                scatter_w1(wVal, i, j);
#ifdef ALIGN
                // Cheap simd_t accumulate only; the expensive per-lane ExpScore
                // fixupSide is deferred until after the score gate (hit
                // batches only), since reverseSideEv is only read there.
                if (needAlign) {
                    reverseSide[j] = VecTraits<simd_t>::fma(VecTraits<simd_t>::broadcast((Float)1.0), wVal, reverseSide[j]);
                }
#endif
                backwardRowMax = VecTraits<simd_t>::vecMax(backwardRowMax, wVal);

                Y0_curr[j] = VecTraits<simd_t>::fma(epsCoeff, Y0_next[j], wVal);
                backwardRowMax = VecTraits<simd_t>::vecMax(backwardRowMax, Y0_curr[j]);
                simd_t z0_future = Z0_ring[r_3] * bgCodonEmitProbs;
                Z0_ring[r_0] = VecTraits<simd_t>::fma(beta0Coeff, z0_future, wVal);

                // Rotate shift registers for next iteration (j-1)
                w1NextJ3 = w1NextJ2;
                w1NextJ2 = w1NextJ1;
                w1NextJ1 = w1NextAtJ;
                w1CurJ2 = w1CurJ1;
                w1CurJ1 = wVal;
            }

            std::swap(Y0_curr, Y0_next);

            // Per-row dynamic rescaling (backward): if any lane's row max
            // exceeds threshold, divide that lane's row-carried state by
            // 2^64 (exact) and record it. Y0 buffers feed row i-1, whose
            // inherited scale picks this up; stored rows i+1.. keep theirs.
                alignas(64) Float maxArr[simdWidth];
                simd_unchecked_store(backwardRowMax, maxArr, Kokkos::Experimental::simd_flag_default);
                auto [factor, any] = makeRowRescaleFactor(scratch.backwardRescaleCount, i, maxArr, activeCount);
                if (any) {
                    simd_t *rp = scratch.W1.row_ptr(i);
                    for (int j = 0; j < scratch.dpWidth + 4; ++j) rp[j] *= factor;
                    rescaleRow(Y0_next, 0, (int)padded_seq_len, factor);
                    rescaleRow(Y0_curr, 0, (int)padded_seq_len, factor);
#ifdef ALIGN
                    // Include front padding: the EV fixup writes j-3.
                    if (needAlign) rescaleRow(reverseSide.m, -3, (int)reverseSide.size(), factor);
#endif
                }
        }
        }

    std::fill(Y0_next.begin(), Y0_next.begin() + padded_seq_len, vzero);

    if (bestOut) {
        // Backward-accumulator best (calibration filter score) without a
        // second DP: W1 holds raw backward values here.
        // Promote-then-multiply in pair domain: rp is in row scale backwardRescaleCount[i],
        // np (nullModel) is static-scale (cum 0). No Float product, so a
        // capped rp (~1e30/1e150) times a large nullModel can never inf
        // before promotion. np varies only with j, so the j loop is outer.
        std::array<ExpScore, simdWidth> bestMid{};
        {
            const simd_t *__restrict__ np = nullModelPrefix.data();
            for (int j = 0; j < scratch.dpWidth; j++) {
                alignas(64) Float narr[simdWidth];
                simd_unchecked_store(np[j], narr, Kokkos::Experimental::simd_flag_default);
                double narrD[simdWidth];
#pragma GCC ivdep
                for (int k = 0; k < activeCount; k++) narrD[k] = (double)narr[k];
                for (int i = profile.length; i >= 0; i--) {
                    const simd_t *__restrict__ rp = scratch.W1.row_ptr(i);
                    alignas(64) Float rarr[simdWidth];
                    simd_unchecked_store(rp[j], rarr, Kokkos::Experimental::simd_flag_default);
                    // Libm-free promote (exact replicate of promoteRescaledScore)
                    // + multiply (exact replicate of scaleScore).
                    alignas(64) float rm[simdWidth];
                    alignas(64) int re[simdWidth];
                    batchPromoteRescaled(rarr, scratch.backwardRescaleCount[i].data(), rm, re, activeCount);
                    ExpScore R[simdWidth], cand[simdWidth];
                    for (int k = 0; k < activeCount; k++) R[k] = ExpScore{rm[k], re[k]};
                    batchScaleScores(R, narrD, cand, activeCount);
                    for (int k = 0; k < activeCount; k++) {
                        if (scoreLess(bestMid[k], cand[k])) bestMid[k] = cand[k];
                    }
                }
            }
        }
        *bestOut = bestMid;
    }

    // Hand off small values; bulk buffers stay in scratch for forwardPass.
    ctx.nullEmit1 = nullEmit1;
    ctx.nullEmit2 = nullEmit2;
    ctx.nullEmit3 = nullEmit3;
}

