// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummest-traceback.hh"
#include "dummest-scalerow.hh"

// Forward half of the unified DP: forward pass over W0 (X fill, wMid gate)
// + deferred alignment work (fixups, prefixOptimal/suffixOptimal, traceback).
// Consumes backwardPass outputs in scratch read-only (W1, reverseSide, null
// models, bg_codon, transposed) plus the small ctx handoff. NOTE:
// reverseSide fixupSide runs after the score gate (hit batches only), since
// reverseSideEv is only read there.
void forwardPass(std::array<std::vector<AlignedSimilarity>, simdWidth> &similarities, const Profile &profile,
             const std::array<std::vector<uint8_t>*, simdWidth> &decoded, std::array<Float, simdWidth> minProbRatio,
             DPScratch &scratch, int activeCount, const BackwardCtx &ctx,
             bool needAlign = true) {
    simd_t vzero = VecTraits<simd_t>::zero();
    auto &forwardSide = scratch.forwardSide;
    auto &reverseSide = scratch.reverseSide;
    auto &forwardSideEv = scratch.forwardSideEv;
    auto &reverseSideEv = scratch.reverseSideEv;
    auto &nullModelPrefix = scratch.nullModelPrefix;
    auto &nullModelSuffix = scratch.nullModelSuffix;
    const Float *bg_probs_ptr = profile.bg_probs.data() + 4;
    uint8_t *transposed_base = scratch.transposedDecoded.data() + 4 * simdWidth;
    const simd_t nullEmit1 = ctx.nullEmit1;
    const simd_t nullEmit2 = ctx.nullEmit2;
    const simd_t nullEmit3 = ctx.nullEmit3;
    simd_t *bg_codon_probs_base = scratch.bgCodonProbs.data();
    auto &Y0_next = scratch.Y0_next;
    auto &Y0_curr = scratch.Y0_curr;
    // Forward-side inits: Y0_next zero seed, per-lane best trackers and
    // their physical mirrors.
    const size_t fwd_padded = scratch.dpWidth + 4;
    std::fill(Y0_next.begin(), Y0_next.begin() + fwd_padded, vzero);
    for (int k = 0; k < activeCount; k++) {
        int len = (int)decoded[k]->size();
        scratch.bestMidScore[k].assign(len, ExpScore{0.0, 0});
        scratch.bestMidRow[k].assign(len, -1);
    }
    scratch.bestMidScorePhys.assign(fwd_padded, std::array<ExpScore, simdWidth>{});
    scratch.bestMidRowPhys.assign(fwd_padded, std::array<int, simdWidth>{});
#ifdef ALIGN
    if (needAlign) {
        forwardSide.assign(nullModelPrefix.size(), vzero);
        scratch.X.assign(profile.length + 2, scratch.dpWidth + 4);
    }
#endif

    std::array<ExpScore, simdWidth> globalBestPair{};
    // Per-lane sequence lengths (row-invariant): Float copy for the simd_t
    // mask, int copy for the per-lane scalar guards.
    alignas(64) Float seq_len_f[simdWidth] = {};
    int seq_len_i[simdWidth] = {};
    for (int k = 0; k < activeCount; k++) {
        seq_len_i[k] = (int)decoded[k]->size();
        seq_len_f[k] = (Float)seq_len_i[k];
    }
    for (int k = activeCount; k < (int)simdWidth; k++) seq_len_f[k] = (Float)1;
    for (int i = 0; i <= profile.length; i++) {
        // Dynamic-scale inheritance: this row starts at the previous row's
        // scale, so all W0/Y0 reads below are single-scale by construction.
        if (i > 0)
            scratch.forwardRescaleCount[i] = scratch.forwardRescaleCount[i - 1];
        const Params &params = profile.values_v2[i];
        const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

        // Pre-calculate constants
        const simd_t enterCoeff = VecTraits<simd_t>::broadcast(params.enter_match_probability) * nullEmit3;
        const simd_t alpha0Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[0]);
        const simd_t alpha1Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[1]) * nullEmit1;
        const simd_t alpha2Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[2]) * nullEmit2;
        const simd_t beta0Coeff = VecTraits<simd_t>::broadcast(params.beta_prime[0]);
        const simd_t delta0Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[0]);
        const simd_t delta1Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[1]) * nullEmit2;
        const simd_t delta2Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[2]) * nullEmit1;
        const simd_t epsCoeff = VecTraits<simd_t>::broadcast(params.epsilon_prime);
        // Effective static scale for this row/lane: the seed term is
        // stored in row scale.
        simd_t rowScaleCoeff = VecTraits<simd_t>::broadcast(scale) * rowRescaleVector(scratch.forwardRescaleCount[i]);
        simd_t forwardRowMax = vzero;

        simd_t Z0_ring[4] = {vzero, vzero, vzero, vzero};
        simd_t Z1_ring[4] = {vzero, vzero, vzero, vzero};
        simd_t Z2_ring[4] = {vzero, vzero, vzero, vzero};

        // Raw row pointers — avoid repeated i*cols in inner loop
        simd_t *__restrict__ w0_row_i = scratch.W0_curr.data();
        simd_t *__restrict__ w0_row_ip1 = (i + 1 <= profile.length) ? scratch.W0_next.data() : nullptr;

        auto gather_w1 = [&](int i_row, int col) -> simd_t {
            return scratch.W1.get(i_row, col);
        };

        bool w1_ip1_avail = (i + 1 <= profile.length);
        auto gather_w1_ip1 = [&](int col) -> simd_t {
            if (!w1_ip1_avail) return vzero;
            return gather_w1(i + 1, col);
        };
        auto gather_w1_i = [&](int col) -> simd_t {
            return gather_w1(i, col);
        };

        int lo = 0, hi = scratch.dpWidth - 1;

        int seq_start = lo;

        // Shift register for w[1..3] — avoids 3 matrix reads per iteration
        simd_t w_shift[3] = {vzero, vzero, vzero}; // w_shift[0]=w0(i,j-1), [1]=w0(i,j-2), [2]=w0(i,j-3)

        for (int j = seq_start; j <= hi; j++) {

            simd_t w1 = w_shift[0], w2 = w_shift[1], w3 = w_shift[2];

            int r_0 = j & 3;
            int r_3 = (j - 3) & 3;

            const char* indices = (const char*)&transposed_base[(j - 2) * simdWidth];
            simd_t codonEmitProbs = VecTraits<simd_t>::lookupEmit(params_emission_probabilities, indices);
            simd_t bgCodonEmitProbs = bg_codon_probs_base[j - 2];

            simd_t X_ij = enterCoeff * codonEmitProbs * w3;

#ifdef ALIGN
            // Combined X = X_ij * W1[i+1][j] * 2^STATIC_SHIFT, formed in
            // ExpScore: X_ij is in forward row scale forwardRescaleCount[i],
            // W1 in backward row scale backwardRescaleCount[i+1] — promote
            // each side first (a raw product could overflow before promotion).
            if (needAlign) {
                simd_t w1n = gather_w1_ip1(j); // zero when row i+1 is unavailable
                {
                    alignas(64) Float xfwdA[simdWidth], w1nA[simdWidth];
                    simd_unchecked_store(X_ij, xfwdA, Kokkos::Experimental::simd_flag_default);
                    simd_unchecked_store(w1n, w1nA, Kokkos::Experimental::simd_flag_default);
                    // Vectorized promote-then-multiply on SoA planes, stored
                    // straight into the matrix planes (no ExpScore temps).
                    alignas(64) float xfM[simdWidth], w1M[simdWidth];
                    alignas(64) int xfE[simdWidth], w1E[simdWidth];
                    batchPromoteRescaled(xfwdA, scratch.forwardRescaleCount[i].data(), xfM, xfE, activeCount);
                    batchPromoteRescaled(w1nA, scratch.backwardRescaleCount[i + 1].data(), w1M, w1E, activeCount);
                    alignas(64) float xM[simdWidth];
                    alignas(64) int xE[simdWidth];
                    batchMulScorePlanes(xfM, xfE, w1M, w1E, STATIC_SHIFT, xM, xE, activeCount);
                    float *mp = scratch.X.m_ptr(i, j);
                    int32_t *ep = scratch.X.e_ptr(i, j);
                    std::array<int, simdWidth> &xrm = scratch.combinedRowMaxExp[i];
#pragma GCC ivdep
                    for (int k = 0; k < activeCount; k++) {
                        mp[k] = xM[k];
                        ep[k] = (int32_t)xE[k];
                        // Side product only: per-row max X exponent.
                        if (xM[k] != 0.0f && xE[k] > xrm[k]) xrm[k] = xE[k];
                    }
                }
            }
#endif
            Z0_ring[r_0] =
                bgCodonEmitProbs * nullEmit3 *
                (alpha0Coeff * w3 + beta0Coeff * Z0_ring[r_3]);
            Z1_ring[r_0] = alpha1Coeff * w1;
            Z2_ring[r_0] = alpha2Coeff * w2;

            simd_t w0 = w0_row_i[j];
            w0 = w0 + (Z0_ring[r_0] + Z1_ring[r_0] + Z2_ring[r_0] + nullModelPrefix[j] * rowScaleCoeff);
#ifdef ALIGN
            // Cheap simd_t accumulate only; forwardSideEv fixup is already after
            // the gate (hit batches only).
            if (needAlign) {
                forwardSide[j] = VecTraits<simd_t>::fma(VecTraits<simd_t>::broadcast((Float)1.0), w0, forwardSide[j]);
            }
#endif
            forwardRowMax = VecTraits<simd_t>::vecMax(forwardRowMax, w0);

            // Combined wMid = w0 * B(i,j) * 2^STATIC_SHIFT per lane, in
            // ExpScore: w0 is in forward row scale forwardRescaleCount[i], B
            // in backward row scale backwardRescaleCount[i]; promote each side
            // first, since a raw product could overflow before promotion.
            simd_t w1i = gather_w1_i(j);
            // Per-lane variable-length mask (batches pack different lengths)
            simd_t isActive = VecTraits<simd_t>::activeMask(j, seq_len_f);
            ExpScore wMidArr[simdWidth];
            {
                alignas(64) Float w0A[simdWidth], w1iA[simdWidth];
                simd_unchecked_store(w0, w0A, Kokkos::Experimental::simd_flag_default);
                simd_unchecked_store(w1i, w1iA, Kokkos::Experimental::simd_flag_default);
                // Vectorized promote-then-multiply on SoA planes. Past-end
                // lanes are zeroed before the multiply so they stay zero.
                alignas(64) float fM[simdWidth], bM[simdWidth];
                alignas(64) int fE[simdWidth], bE[simdWidth];
                batchPromoteRescaled(w0A, scratch.forwardRescaleCount[i].data(), fM, fE, activeCount);
                batchPromoteRescaled(w1iA, scratch.backwardRescaleCount[i].data(), bM, bE, activeCount);
#pragma GCC ivdep
                for (int k = 0; k < activeCount; k++) {
                    if (j >= seq_len_i[k]) { fM[k] = 0.0f; bM[k] = 0.0f; }
                }
                alignas(64) float wM[simdWidth];
                alignas(64) int wE[simdWidth];
                batchMulScorePlanes(fM, fE, bM, bE, STATIC_SHIFT, wM, wE, activeCount);
                for (int k = 0; k < activeCount; k++) wMidArr[k] = ExpScore{wM[k], wE[k]};
            }
            for (int k = 0; k < activeCount; k++) {
                // Past-end lanes stay zero so their best trackers stay zero
                // and they emit no hits below.
                ExpScore wMid{0.0, 0};
                if (j < seq_len_i[k]) wMid = wMidArr[k];
                if (scoreLess(globalBestPair[k], wMid)) globalBestPair[k] = wMid;
                if (scoreLess(scratch.bestMidScorePhys[j][k], wMid)) {
                    scratch.bestMidScorePhys[j][k] = wMid;
                    scratch.bestMidRowPhys[j][k] = i;
                }
            }
            w0 = w0 * isActive;
            w0_row_i[j] = w0;

            Z0_ring[r_0] = Z0_ring[r_0] * isActive;
            Z1_ring[r_0] = Z1_ring[r_0] * isActive;
            Z2_ring[r_0] = Z2_ring[r_0] * isActive;

            w_shift[2] = w_shift[1];
            w_shift[1] = w_shift[0];
            w_shift[0] = w0;

            Y0_curr[j] = delta0Coeff * w0 + epsCoeff * Y0_next[j];
            Y0_curr[j] = Y0_curr[j] * isActive;
            forwardRowMax = VecTraits<simd_t>::vecMax(forwardRowMax, Y0_curr[j]);
            if (w0_row_ip1)
                w0_row_ip1[j] = w0_row_ip1[j] + (X_ij + Y0_curr[j] + delta1Coeff * w2 + delta2Coeff * w1);
        }

        // Per-row dynamic rescaling (forward): divide triggered lanes'
        // row-carried state by 2^64 (exact). W0_next holds row i+1's
        // incoming sums; Y0 buffers feed row i+1 after the swap below.
        {
            alignas(64) Float maxArr[simdWidth];
            simd_unchecked_store(forwardRowMax, maxArr, Kokkos::Experimental::simd_flag_default);
            auto [factor, any] = makeRowRescaleFactor(scratch.forwardRescaleCount, i, maxArr, activeCount);
            if (any) {
                if (w0_row_ip1) {
                    rescaleRow(scratch.W0_next, 0, scratch.dpWidth + 4, factor);
                }
                rescaleRow(Y0_next, 0, (int)fwd_padded, factor);
                rescaleRow(Y0_curr, 0, (int)fwd_padded, factor);
#ifdef ALIGN
                if (needAlign) rescaleRow(forwardSide.m, 0, (int)forwardSide.size(), factor);
#endif
            }
        }

        std::swap(Y0_curr, Y0_next);
        // No fill of Y0_curr: it is overwrite-assigned for every j each row.
        // (W0_next accumulates via += so its fill below is required.)

        std::swap(scratch.W0_curr, scratch.W0_next);
        std::fill(scratch.W0_next.begin(), scratch.W0_next.end(), vzero);
    }

    // Per-lane threshold as an exact pair (minProbRatio >= 0); zero when the
    // threshold is 0. All selection comparisons below work in this domain.
    std::array<ExpScore, simdWidth> minExp{};
    for (int k = 0; k < activeCount; k++) {
        if (minProbRatio[k] >= 0)
            minExp[k] = normalizeScore((double)minProbRatio[k], 0);
    }
    {
        bool any_fwd_above = false;
        for (int k = 0; k < activeCount; k++) {
            if (minProbRatio[k] < 0 || !scoreLess(globalBestPair[k], minExp[k])) {
                any_fwd_above = true;
                break;
            }
        }
        if (!any_fwd_above) return;
    }

#ifdef ALIGN
    if (needAlign) {
        // Deferred right-side fixup (was before the score gate): reverseSideEv
        // is only read by the prefixOptimal rebuild below, so no-hit batches skip it.
        // reverseSide sits at row scale backwardRescaleCount[0] (kept there by rescaling on
        // every trigger); promote with it.
        reverseSideEv.assign(nullModelPrefix.size());
        fixupSide(reverseSide, reverseSideEv, nullModelPrefix,
                    nullEmit1, nullEmit2, nullEmit3,
                    -1, -2, transposed_base, bg_probs_ptr,
                    scratch.dpWidth, activeCount,
                    scratch.backwardRescaleCount[0].data());
        // Side product only: per-batch max right-side EV exponent.
        sideEVMax(reverseSideEv, scratch.dpWidth, activeCount, scratch.reverseSideEvMaxExp.data());

        // Deferred prefixOptimal prefix-max chain (bit-exact replica of the former
        // in-loop chain): max(prefixOptimal(i-1,j), prefixOptimal(i,j-1), reverseSideEv[j],
        // X(i,j) + prefixOptimal(i-1,j-3)). Runs i ascending / j ascending so every
        // predecessor is final on read. Only reached when at least one lane
        // passed the score gate, so no-hit batches skip it entirely.
        // The X+opt_succ add is batched (no frexp); the max-selects stay
        // scalar (cheap compares). Exact.
        scratch.prefixOptimal.assign(profile.length + 2, scratch.dpWidth + 4);
        scratch.suffixOptimal.assign(profile.length + 2, scratch.dpWidth + 4);
        // Float-compute prefix-max chain: same recurrence (up, left, X+succ,
        // reverseSideEv) and tie order, but candidates scale once to a
        // hoisted row exponent Erow (running prefix max over X rows + side
        // max, +1 headroom) and combine with plain float add/max; results
        // pack back to pairs. Max-selects ordering-exact; adds round at
        // 24 bits and far-below terms flush (fixtures decide).
        {
            const int S = (int)simdWidth;
            const int W = scratch.dpWidth;
            int runMax[simdWidth];
            for (int k = 0; k < activeCount; k++) runMax[k] = INT_MIN;
            auto rowScale = [&](int i, const int *sideMax, int *rowExp) {
                for (int k = 0; k < activeCount; k++) {
                    if (scratch.combinedRowMaxExp[i][k] > runMax[k]) runMax[k] = scratch.combinedRowMaxExp[i][k];
                    const int mx = runMax[k] > sideMax[k] ? runMax[k] : sideMax[k];
                    rowExp[k] = mx + 1;
                }
            };
            // i == 0: opt_succ/up predecessors don't exist (all zero).
            {
                int rowExp[simdWidth];
                rowScale(0, scratch.reverseSideEvMaxExp.data(), rowExp);
                float prefixMaxLeft[simdWidth] = {0.0f};
                const float *combinedM = scratch.X.m_ptr(0, 0);
                const int32_t *combinedE = scratch.X.e_ptr(0, 0);
                float *prefixMaxM = scratch.prefixOptimal.m_ptr(0, 0);
                int32_t *prefixMaxE = scratch.prefixOptimal.e_ptr(0, 0);
                for (int j = 0; j < W; j++) {
                    const std::array<ExpScore, simdWidth> &rs = scratch.reverseSideEv[j];
#pragma GCC ivdep
                    for (int k = 0; k < activeCount; k++) {
                        const int E = rowExp[k];
                        const float combinedRow = scoreToRowScale(combinedM[j * S + k], combinedE[j * S + k], E);
                        const float reverseSideRow = scoreToRowScale(rs[k].m, rs[k].e, E);
                        float prefixMax = 0.0f;
                        if (prefixMaxLeft[k] > prefixMax) prefixMax = prefixMaxLeft[k];
                        if (combinedRow > prefixMax) prefixMax = combinedRow;
                        if (reverseSideRow > prefixMax) prefixMax = reverseSideRow;
                        prefixMaxLeft[k] = prefixMax;
                        const ExpScore out = scoreFromRowScale(prefixMax, E);
                        prefixMaxM[j * S + k] = out.m;
                        prefixMaxE[j * S + k] = out.e;
                    }
                }
            }
            for (int i = 1; i <= profile.length; i++) {
                int rowExp[simdWidth];
                rowScale(i, scratch.reverseSideEvMaxExp.data(), rowExp);
                // Rolling prefixOptimal(i, j-1) per lane, already in row scale.
                float prefixMaxLeft[simdWidth] = {0.0f};
                const float *combinedM = scratch.X.m_ptr(i, 0);
                const int32_t *combinedE = scratch.X.e_ptr(i, 0);
                const float *prefixPrevM = scratch.prefixOptimal.m_ptr(i - 1, 0);
                const int32_t *prefixPrevE = scratch.prefixOptimal.e_ptr(i - 1, 0);
                float *prefixMaxM = scratch.prefixOptimal.m_ptr(i, 0);
                int32_t *prefixMaxE = scratch.prefixOptimal.e_ptr(i, 0);
                for (int j = 0; j < W; j++) {
                    const std::array<ExpScore, simdWidth> &rs = scratch.reverseSideEv[j];
#pragma GCC ivdep
                    for (int k = 0; k < activeCount; k++) {
                        const int E = rowExp[k];
                        const float combinedRow = scoreToRowScale(combinedM[j * S + k], combinedE[j * S + k], E);
                        const float prefixPrevRow = scoreToRowScale(prefixPrevM[(j - 3) * S + k], prefixPrevE[(j - 3) * S + k], E);
                        const float combinedPlusSucc = combinedRow + prefixPrevRow;
                        const float upRow = scoreToRowScale(prefixPrevM[j * S + k], prefixPrevE[j * S + k], E);
                        const float reverseSideRow = scoreToRowScale(rs[k].m, rs[k].e, E);
                        float prefixMax = upRow;
                        if (prefixMaxLeft[k] > prefixMax) prefixMax = prefixMaxLeft[k];
                        if (combinedPlusSucc > prefixMax) prefixMax = combinedPlusSucc;
                        if (reverseSideRow > prefixMax) prefixMax = reverseSideRow;
                        prefixMaxLeft[k] = prefixMax;
                        const ExpScore out = scoreFromRowScale(prefixMax, E);
                        prefixMaxM[j * S + k] = out.m;
                        prefixMaxE[j * S + k] = out.e;
                    }
                }
            }
        }

        {
        // forwardSide sits at row scale forwardRescaleCount[plen] (kept there by rescaling
        // on every trigger); promote with it.
        forwardSideEv.assign(nullModelPrefix.size());
        fixupSide(forwardSide, forwardSideEv, nullModelSuffix,
                    nullEmit1, nullEmit2, nullEmit3,
                    +1, +1, transposed_base, bg_probs_ptr,
                    scratch.dpWidth, activeCount,
                    scratch.forwardRescaleCount[profile.length].data());
        // Side product only: per-batch max left-side EV exponent.
        sideEVMax(forwardSideEv, scratch.dpWidth, activeCount, scratch.forwardSideEvMaxExp.data());
        }

        // Float-compute suffix-optimal rebuild: same recurrence (down, right,
        // forwardSideEv, X+succ) and tie order, scaled once per row to Erow
        // (running suffix max over X rows + side max, +1 headroom) with plain
        // float add/max; results pack back to pairs. Max-selects
        // ordering-exact.
        // MUST read the already-rebuilt row i+1 (suffixOptimal), exactly as the original
        // in-place W1 rebuild did. W1 keeps its raw backward values and is never
        // consulted by the traceback.
        {
            const int S = (int)simdWidth;
            const int W = scratch.dpWidth;
            const int P = profile.length;
            int runMax[simdWidth];
            for (int k = 0; k < activeCount; k++) runMax[k] = INT_MIN;
            auto rowScaleW = [&](int i, int *rowExp) {
                for (int k = 0; k < activeCount; k++) {
                    if (scratch.combinedRowMaxExp[i][k] > runMax[k]) runMax[k] = scratch.combinedRowMaxExp[i][k];
                    const int mx = runMax[k] > scratch.forwardSideEvMaxExp[k] ? runMax[k] : scratch.forwardSideEvMaxExp[k];
                    rowExp[k] = mx + 1;
                }
            };
            // i == P: no successor row (downRow/succ terms are zero).
            {
                int rowExp[simdWidth];
                rowScaleW(P, rowExp);
                float suffixMaxRight[simdWidth] = {0.0f};
                float *suffixMaxM = scratch.suffixOptimal.m_ptr(P, 0);
                int32_t *suffixMaxE = scratch.suffixOptimal.e_ptr(P, 0);
                for (int j = W - 1; j >= 0; j--) {
                    const std::array<ExpScore, simdWidth> &lv = scratch.forwardSideEv[j];
#pragma GCC ivdep
                    for (int k = 0; k < activeCount; k++) {
                        const int E = rowExp[k];
                        const float forwardSideRow = scoreToRowScale(lv[k].m, lv[k].e, E);
                        float suffixMax = 0.0f;
                        if (suffixMaxRight[k] > suffixMax) suffixMax = suffixMaxRight[k];
                        if (forwardSideRow > suffixMax) suffixMax = forwardSideRow;
                        suffixMaxRight[k] = suffixMax;
                        const ExpScore out = scoreFromRowScale(suffixMax, E);
                        suffixMaxM[j * S + k] = out.m;
                        suffixMaxE[j * S + k] = out.e;
                    }
                }
            }
            for (int i = P - 1; i >= 0; i--) {
                int rowExp[simdWidth];
                rowScaleW(i, rowExp);
                float suffixMaxRight[simdWidth] = {0.0f};
                const float *combinedM = scratch.X.m_ptr(i, 0);
                const int32_t *combinedE = scratch.X.e_ptr(i, 0);
                const float *suffixNextM = scratch.suffixOptimal.m_ptr(i + 1, 0);
                const int32_t *suffixNextE = scratch.suffixOptimal.e_ptr(i + 1, 0);
                float *suffixMaxM = scratch.suffixOptimal.m_ptr(i, 0);
                int32_t *suffixMaxE = scratch.suffixOptimal.e_ptr(i, 0);
                for (int j = W - 1; j >= 0; j--) {
                    const std::array<ExpScore, simdWidth> &lv = scratch.forwardSideEv[j];
#pragma GCC ivdep
                    for (int k = 0; k < activeCount; k++) {
                        const int E = rowExp[k];
                        const float combinedRow = scoreToRowScale(combinedM[j * S + k], combinedE[j * S + k], E);
                        const float suffixNextRow = scoreToRowScale(suffixNextM[(j + 3) * S + k], suffixNextE[(j + 3) * S + k], E);
                        const float combinedPlusSucc = combinedRow + suffixNextRow;
                        const float downRow = scoreToRowScale(suffixNextM[j * S + k], suffixNextE[j * S + k], E);
                        const float forwardSideRow = scoreToRowScale(lv[k].m, lv[k].e, E);
                        float suffixMax = downRow;
                        if (suffixMaxRight[k] > suffixMax) suffixMax = suffixMaxRight[k];
                        if (forwardSideRow > suffixMax) suffixMax = forwardSideRow;
                        if (combinedPlusSucc > suffixMax) suffixMax = combinedPlusSucc;
                        suffixMaxRight[k] = suffixMax;
                        const ExpScore out = scoreFromRowScale(suffixMax, E);
                        suffixMaxM[j * S + k] = out.m;
                        suffixMaxE[j * S + k] = out.e;
                    }
                }
            }
        }
    }
#endif

    // Extract per-lane anchor trackers from the physical layout.
    // bestMidScorePhys holds static-scale-unit pairs: compare exactly, then
    // materialize doubles only for reporting.
    for (int idx = 0; idx < activeCount; idx++) {
        int len = (int)decoded[idx]->size();
        for (int j = 0; j < scratch.dpWidth && j < len; j++) {
            const ExpScore &wMid_k = scratch.bestMidScorePhys[j][idx];
            if (scoreLess(scratch.bestMidScore[idx][j], wMid_k)) {
                scratch.bestMidScore[idx][j] = wMid_k;
                scratch.bestMidRow[idx][j] = scratch.bestMidRowPhys[j][idx];
            }
        }
    }

    for (int idx = 0; idx < activeCount; idx++) {
        int len = (int)decoded[idx]->size();

        scratch.optProfilePosition[idx].assign(len, AlignedSimilarity{ExpScore{}, -INFINITY, 0, 0, {}});

        ExpScore bestOverall{};
        for (int j = 0; j < len; j++) {
            const ExpScore &best_pair = scratch.bestMidScore[idx][j];
            if (best_pair.m != 0.0) {
                if (scoreLess(bestOverall, best_pair)) bestOverall = best_pair;
                scratch.optProfilePosition[idx][j] = {
                    best_pair,
                    -INFINITY,
                    scratch.bestMidRow[idx][j],
                    j
                };
            }
        }

        if (minProbRatio[idx] >= 0) {
            if (bestOverall.m != 0.0 && !scoreLess(bestOverall, minExp[idx])) {
                if (verbosity > 0)
                    std::cerr << "has" << std::endl;
                std::ranges::sort(scratch.optProfilePosition[idx], std::greater<>());
                auto &aligned = scratch.aligned;
                aligned[idx].assign(decoded[idx]->size() + 0, false);
                // Placeholders have pair.m == 0; the guard keeps them out even
                // when minProbRatio == 0.
                for (auto &aligned_similarity : scratch.optProfilePosition[idx]) {
                    int logical_j = aligned_similarity.anchor2;

                    if (aligned_similarity.pair.m != 0.0 &&
                        !scoreLess(aligned_similarity.pair, minExp[idx]) &&
                        !aligned[idx][logical_j]) {
                        addMidAnchored(idx, profile.length, similarities[idx], aligned_similarity.anchor1,
                                       aligned_similarity.anchor2,
                                       aligned_similarity.pair, scratch);
                        auto &x = similarities[idx].back();
                        finishMidAnchored(idx, x, scratch);

                        int seqBeg = simBeg2(x);
                        int seqEnd = simEnd2(x);
                        int clampedBeg = std::max(seqBeg, 0);
                        int clampedEnd = std::min(seqEnd, (int)decoded[idx]->size());
                        bool overlaps = false;
                        for (int p = clampedBeg; p < clampedEnd; ++p) {
                            if (aligned[idx][p]) {
                                overlaps = true;
                                break;
                            }
                        }
                        if (overlaps) {
                            similarities[idx].pop_back();
                        } else {
                            std::fill(aligned[idx].begin() + clampedBeg, aligned[idx].begin() + clampedEnd, true);
                        }
                        }
                    }
            }
        } else {
            auto sel = *std::max_element(scratch.optProfilePosition[idx].begin(), scratch.optProfilePosition[idx].end());
            AlignedSimilarity b = sel;
            b.logProbRatio = -INFINITY;
            sel.logProbRatio = scoreToLog(sel.pair);
            similarities[idx].push_back(b);
            similarities[idx].push_back(b);
            similarities[idx].push_back(sel);
        }
    }
    // No overflow: lazy rescaling keeps every lane finite (see triggers).
}

