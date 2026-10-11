// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummest-dp.hh"

inline void buildNullSuffix(const Float *backgroundPerBase, const uint8_t *transposedSequenceBases,
        const Float *sequenceLengths, int activeCount, int dpWidth, ExpVector<0> &suffixProbabilities,
        double codonKeepProbability, double singleNucleotideEmission, double doubleNucleotideEmission) {
    float trailingMantissa;
    int trailingExponent;
    splitNormalizedDouble(0.5 * codonKeepProbability, trailingMantissa, trailingExponent);
    double singleRow[simdWidth], doubleRow[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) { singleRow[k] = singleNucleotideEmission; doubleRow[k] = doubleNucleotideEmission; }
    // Empty rest counts as 1, set before use below.
    for (int k = 0; k < activeCount; k++)
        suffixProbabilities[(int)sequenceLengths[k]][k] = pow2Score(0);
    for (int i = dpWidth - 1; i >= 0; i--) {
        const char* indices = (const char*)&transposedSequenceBases[i * simdWidth];
        SimdFloat backgroundRaw = simdLookup(backgroundPerBase, indices);
        alignas(64) Float backgroundCodon[simdWidth];
        simd_unchecked_store(simd_t(backgroundRaw), backgroundCodon, Kokkos::Experimental::simd_flag_default);
        const ExpScore *suffixPlusThree = suffixProbabilities[i + 3].data();
        const ExpScore *suffixPlusOne = suffixProbabilities[i + 1].data();
        const ExpScore *suffixPlusTwo = suffixProbabilities[i + 2].data();
        double codonScale[simdWidth];
        ExpScore singleInput[simdWidth], doubleInput[simdWidth];
        for (int k = 0; k < activeCount; k++) {
            int length = (int)sequenceLengths[k];
            codonScale[k] = (double)backgroundCodon[k] * codonKeepProbability;
            singleInput[k] = (i < length) ? suffixPlusOne[k] : ExpScore{0.0f, 0};
            // Past the end the 2-base piece has no rest:
            // count it as one 0.25 piece.
            doubleInput[k] = (i + 1 < length) ? suffixPlusTwo[k]
                  : ((i + 1 == length) ? pow2Score(2) : ExpScore{0.0f, 0});
        }
        ExpScore codonRaw[simdWidth], codonBranch[simdWidth], singleBranch[simdWidth], doubleBranch[simdWidth], shiftBranchSum[simdWidth];
        batchScaleScores(suffixPlusThree, codonScale, codonRaw, activeCount);
        batchScaleScores(singleInput, singleRow, singleBranch, activeCount);
        batchScaleScores(doubleInput, doubleRow, doubleBranch, activeCount);
        for (int k = 0; k < activeCount; k++) {
            int length = (int)sequenceLengths[k];
            // Leftover end codon: one 0.25 per missing base.
            codonBranch[k] = (i + 2 < length) ? codonRaw[k]
                                  : ExpScore{trailingMantissa, trailingExponent - 2 * (length - i) + 1};
        }
        auto &suffixOutput = suffixProbabilities[i];
        batchAddScores(singleBranch, doubleBranch, shiftBranchSum, activeCount);
        batchAddScores(codonBranch, shiftBranchSum, suffixOutput.data(), activeCount);
        // Past the end only the boundary cell keeps its 1.
        for (int k = 0; k < activeCount; k++) {
            int length = (int)sequenceLengths[k];
            if (i >= length) suffixOutput[k] = (i == length) ? pow2Score(0) : ExpScore{0.0f, 0};
        }
    }
}

// Background odds of each prefix. First piece uses 1/3 (S4).
inline void buildNullPrefixFirstCodon(const Float *backgroundPerBase, const uint8_t *transposedSequenceBases,
        const Float *sequenceLengths, int activeCount, int dpWidth, ExpVector<0> &prefixProbabilities,
        double codonKeepProbability, double singleNucleotideEmission, double doubleNucleotideEmission, double frameChoiceProbability) {
    const ExpScore zeroScore{0.0f, 0};
    ExpScore zeroRow[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) zeroRow[k] = zeroScore;
    double singleRow[simdWidth], doubleRow[simdWidth];
    for (int k = 0; k < (int)simdWidth; k++) { singleRow[k] = singleNucleotideEmission; doubleRow[k] = doubleNucleotideEmission; }
    for (int i = 0; i < dpWidth; i++) {
        const char* indices = (const char*)&transposedSequenceBases[(i - 2) * simdWidth];
        SimdFloat backgroundRaw = simdLookup(backgroundPerBase, indices);
        alignas(64) Float backgroundCodon[simdWidth];
        simd_unchecked_store(simd_t(backgroundRaw), backgroundCodon, Kokkos::Experimental::simd_flag_default);
        double codonScale[simdWidth];
        for (int k = 0; k < activeCount; k++) codonScale[k] = (double)backgroundCodon[k] * codonKeepProbability;
        const ExpScore *prefixSingle = (i > 0) ? prefixProbabilities[i - 1].data() : zeroRow;
        const ExpScore *prefixDouble = (i >= 2) ? prefixProbabilities[i - 2].data() : zeroRow;
        ExpScore codonBranch[simdWidth], singleBranch[simdWidth], doubleBranch[simdWidth], shiftBranchSum[simdWidth], prefixOutput[simdWidth];
        if (i >= 3) {
            batchScaleScores(prefixProbabilities[i - 3].data(), codonScale, codonBranch, activeCount);
        } else if (i == 2) {
            for (int k = 0; k < activeCount; k++)
                codonBranch[k] = scaleScore(normalizeScore((double)backgroundCodon[k], 0), frameChoiceProbability);
        } else if (i == 1) {
            // 1/3 times 0.25*0.25.
            for (int k = 0; k < activeCount; k++)
                codonBranch[k] = scaleScore(pow2Score(-4), frameChoiceProbability);
        } else {
            // 1/3 times 0.25.
            for (int k = 0; k < activeCount; k++)
                codonBranch[k] = scaleScore(pow2Score(-2), frameChoiceProbability);
        }
        batchScaleScores(prefixSingle, singleRow, singleBranch, activeCount);
        batchScaleScores(prefixDouble, doubleRow, doubleBranch, activeCount);
        batchAddScores(singleBranch, doubleBranch, shiftBranchSum, activeCount);
        batchAddScores(codonBranch, shiftBranchSum, prefixOutput, activeCount);
        auto &prefixCell = prefixProbabilities[i];
        for (int k = 0; k < activeCount; k++) {
            int length = (int)sequenceLengths[k];
            prefixCell[k] = (i < length) ? prefixOutput[k] : ExpScore{0.0f, 0};
        }
    }
}

// Whole-sequence background odds plus per-base scales for the DP.
inline void buildNullTotalAndScales(const Float *sequenceLengths, int activeCount,
        ExpVector<0> &prefixProbabilities, double singleShiftRate, double doubleShiftRate,
        ExpScore *scalePerBase, ExpScore *scalePerBaseInverse,
        ExpScore *scaleOneNucleotide, ExpScore *scaleTwoNucleotide, ExpScore *scaleCodon) {
    alignas(64) Float nullSequenceLog[simdWidth] = {0};
    const double keepSingle = 1.0 - (double)singleShiftRate;
    const double keepBoth = keepSingle - (double)doubleShiftRate;
    for (int idx = 0; idx < activeCount; idx++) {
        int length = (int)sequenceLengths[idx];
        assert(length >= 3);
        ExpScore trailingSingle = scaleScore(prefixProbabilities[length - 2][idx], keepSingle * 0.25);
        ExpScore trailingDouble = scaleScore(prefixProbabilities[length - 3][idx], keepBoth * 0.0625);
        ExpScore total = addScores(addScores(prefixProbabilities[length - 1][idx], trailingSingle), trailingDouble);
        nullSequenceLog[idx] = (Float)scoreToLog2(total);
        double perBaseLogProb = -(double)nullSequenceLog[idx] / (double)sequenceLengths[idx];
        if (!(perBaseLogProb < (double)EXP2_HI)) perBaseLogProb = (double)EXP2_HI;
        double perBaseRaw = exp2(perBaseLogProb);
        ExpScore perBaseValue = normalizeScore(perBaseRaw, 0);
        scalePerBase[idx] = perBaseValue;
        scalePerBaseInverse[idx] = normalizeScore(1.0 / perBaseRaw, 0);
        ExpScore quarter = normalizeScore(0.25, 0);
        ExpScore sixteenth = normalizeScore(0.0625, 0);
        scaleOneNucleotide[idx] = perBaseValue * quarter;
        scaleTwoNucleotide[idx] = perBaseValue * perBaseValue * sixteenth;
        scaleCodon[idx] = perBaseValue * perBaseValue * perBaseValue;
    }
}

// Background odds with scaling folded in, one per base.
// The suffix seed at j starts after j. Short lanes freeze early.
inline void buildNullSeedModels(int dpWidth, int activeCount, const Float *sequenceLengths,
        ExpVector<0> &prefixProbabilities, ExpVector<0> &suffixProbabilities,
        ExpScore *scalePerBase, ExpScore *scalePerBaseInverse,
        PaddedVec<simd_t, 0, 0> &prefixSeed, PaddedVec<simd_t, 0, 0> &suffixSeedFromNext) {
    ExpScore runningScale[simdWidth]{};
    for (int k = 0; k < activeCount; k++) runningScale[k] = pow2Score(0);
    int maxLength = 0;
    for (int k = 0; k < activeCount; k++)
        maxLength = std::max(maxLength, (int)sequenceLengths[k]);
    ExpScore savedScale[simdWidth];
    for (int t = 0; t < maxLength - 1; t++) {
        for (int k = 0; k < activeCount; k++) savedScale[k] = runningScale[k];
        batchMulScores(runningScale, scalePerBase, runningScale, activeCount, 0);
        for (int k = 0; k < activeCount; k++)
            if (t >= (int)sequenceLengths[k] - 1) runningScale[k] = savedScale[k];
    }
    for (int j = 0; j < dpWidth; j++) {
        ExpScore column[simdWidth]{};
        batchMulScores(suffixProbabilities[j + 1].data(), runningScale, column, activeCount, 0);
        suffixSeedFromNext[j] = VecTraits<simd_t>::packScores(column, activeCount);
        batchMulScores(runningScale, scalePerBaseInverse, runningScale, activeCount, 0);
    }
    for (int k = 0; k < activeCount; k++)
        runningScale[k] = scalePerBase[k];
    for (int k = activeCount; k < (int)simdWidth; k++)
        runningScale[k] = ExpScore{0.0f, 0};
    for (int j = 0; j < dpWidth; j++) {
        ExpScore column[simdWidth]{};
        batchMulScores(prefixProbabilities[j].data(), runningScale, column, activeCount, 0);
        prefixSeed[j] = VecTraits<simd_t>::packScores(column, activeCount);
        batchMulScores(runningScale, scalePerBase, runningScale, activeCount, 0);
    }
}

// Shared background setup. Both the full and filter-only paths
// run it so their scores match.
inline void buildBatchNullModels(const Profile &profile, const uint8_t *transposedSequenceBases,
        const Float *sequenceLengths, int activeCount, int dpWidth,
        ExpVector<0> &prefixProbabilities, ExpVector<0> &suffixProbabilities,
        ExpScore *scalePerBase, ExpScore *scalePerBaseInverse,
        ExpScore *scaleOneNucleotide, ExpScore *scaleTwoNucleotide, ExpScore *scaleCodon) {
    const double singleShiftRate = BACKGROUND_FRAMESHIFT_RATE;
    const double doubleShiftRate = BACKGROUND_FRAMESHIFT_RATE_2;
    const double codonKeepProbability = 1 - singleShiftRate - doubleShiftRate;
    const double singleNucleotideEmission = singleShiftRate * 0.25;
    const double doubleNucleotideEmission = doubleShiftRate * 0.0625;
    const double frameChoiceProbability = 1.0 / 3.0;
    const Float *backgroundPerBase = profile.bg_probs.data() + 4;
    buildNullSuffix(backgroundPerBase, transposedSequenceBases, sequenceLengths, activeCount,
        dpWidth, suffixProbabilities, codonKeepProbability, singleNucleotideEmission, doubleNucleotideEmission);
    buildNullPrefixFirstCodon(backgroundPerBase, transposedSequenceBases, sequenceLengths, activeCount,
        dpWidth, prefixProbabilities, codonKeepProbability, singleNucleotideEmission,
        doubleNucleotideEmission, frameChoiceProbability);
    buildNullTotalAndScales(sequenceLengths, activeCount, prefixProbabilities, singleShiftRate,
        doubleShiftRate, scalePerBase, scalePerBaseInverse, scaleOneNucleotide, scaleTwoNucleotide, scaleCodon);
}

// Setup for the backward half. Fills the background scales in ctx.
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

    // Track widest batch seen.
    {
        size_t w = (size_t)maxSequenceLength;
        size_t cur = g_maxDpWidth.load(std::memory_order_relaxed);
        while (w > cur && !g_maxDpWidth.compare_exchange_weak(cur, w, std::memory_order_relaxed))
            ;
    }

    auto &prefixProbabilities = scratch.nullPrefixProbabilities;
    auto &suffixProbabilities = scratch.nullSuffixProbabilities;

    uint8_t* transposedSequenceBases = buildBatchTransposed(scratch, profile, activeCount, scratch.dpWidth, decoded);

    alignas(64) Float sequenceLengths[simdWidth] = {};
    fillSequenceLengths(activeCount, decoded, sequenceLengths);

    prefixProbabilities.assign(scratch.dpWidth + 4);
    suffixProbabilities.assign(scratch.dpWidth + 4);

    ExpScore scalePerBase[simdWidth]{}, scalePerBaseInverse[simdWidth]{};
    ExpScore scaleOneNucleotide[simdWidth]{}, scaleTwoNucleotide[simdWidth]{}, scaleCodon[simdWidth]{};
    buildBatchNullModels(profile, transposedSequenceBases, sequenceLengths, activeCount,
        scratch.dpWidth, prefixProbabilities, suffixProbabilities,
        scalePerBase, scalePerBaseInverse, scaleOneNucleotide, scaleTwoNucleotide, scaleCodon);

    // Tail cells are read but never written; zero them, keeping
    // each lane's end cell at 1.
    for (int j = scratch.dpWidth; j < scratch.dpWidth + 4; j++) {
        for (int k = 0; k < activeCount; k++) {
            if (j != (int)sequenceLengths[k]) {
                prefixProbabilities[j][k] = ExpScore{0.0f, 0};
                suffixProbabilities[j][k] = ExpScore{0.0f, 0};
            }
        }
        for (int k = activeCount; k < (int)simdWidth; k++) {
            prefixProbabilities[j][k] = ExpScore{0.0f, 0};
            suffixProbabilities[j][k] = ExpScore{0.0f, 0};
        }
    }

    simd_t nullScaleOneNucleotide = VecTraits<simd_t>::packScores(scaleOneNucleotide, activeCount);
    simd_t nullScaleTwoNucleotide = VecTraits<simd_t>::packScores(scaleTwoNucleotide, activeCount);
    simd_t nullScaleCodon = VecTraits<simd_t>::packScores(scaleCodon, activeCount);
    simd_t vzero = VecTraits<simd_t>::zero();

    scratch.W0_curr.resize(scratch.dpWidth + 4, vzero);
    scratch.W0_next.resize(scratch.dpWidth + 4, vzero);

    scratch.W1.resizeNoFill(profile.length + 2, scratch.dpWidth + 4);
    // Borders are read before written; zero only them.
    scratch.W1.fillRow(profile.length + 1, vzero);
    for (int bi = 0; bi <= profile.length; bi++) {
        simd_t *wrp = scratch.W1.row_ptr(bi);
        for (int bj = scratch.dpWidth; bj < scratch.dpWidth + 4; bj++)
            wrp[bj] = vzero;
    }

    // Halve counts, zeroed per batch. Carried over row to row below.
    scratch.forwardRescaleCount.assign(profile.length + 2, std::array<int, simdWidth>{});
    scratch.backwardRescaleCount.assign(profile.length + 2, std::array<int, simdWidth>{});
    // Largest match scores per row, for the rebuild scale.
    {
        std::array<int, simdWidth> neg;
        neg.fill(INT_MIN);
        scratch.combinedRowMaxExp.assign(profile.length + 2, neg);
    }

    const size_t padded_seq_len = scratch.dpWidth + 4;
    auto &Y0_next = scratch.Y0_next; Y0_next.resize(padded_seq_len, vzero);
    auto &Y0_curr = scratch.Y0_curr; Y0_curr.resize(padded_seq_len, vzero);
#ifdef DUMMEST_POISON
    // Mark for overwrite; a missing write fails loudly below.
    std::fill(Y0_curr.begin(), Y0_curr.end(), VecTraits<simd_t>::poison());
#endif

    auto &prefixSeed = scratch.nullPrefixSeed; prefixSeed.resize(padded_seq_len, vzero);
    auto &suffixSeedFromNext = scratch.nullSuffixSeedFromNext; suffixSeedFromNext.resize(padded_seq_len, vzero);

    buildNullSeedModels(scratch.dpWidth, activeCount, sequenceLengths, prefixProbabilities, suffixProbabilities,
        scalePerBase, scalePerBaseInverse, prefixSeed, suffixSeedFromNext);
    auto &reverseSide = scratch.reverseSide;
#ifdef ALIGN
    // Scores-only path skips these buffers.
    if (needAlign) {
        // backward half fills reverseSide; forward half fills forwardSide.
        reverseSide.assign(prefixSeed.size(), vzero);
    }
#endif
        fillBackgroundCodon(scratch, profile, transposedSequenceBases, scratch.dpWidth);

    ctx.nullScaleOneNucleotide = nullScaleOneNucleotide;
    ctx.nullScaleTwoNucleotide = nullScaleTwoNucleotide;
    ctx.nullScaleCodon = nullScaleCodon;
}

// Fast filter: 2 rolling rows, no alignment. Survivors go to full DP.
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

    uint8_t* transposedSequenceBases = buildBatchTransposed(scratch, profile, activeCount, scratch.dpWidth, decoded);

    alignas(64) Float sequenceLengths[simdWidth] = {};
    fillSequenceLengths(activeCount, decoded, sequenceLengths);

    auto &prefixProbabilities = scratch.nullPrefixProbabilities;
    auto &suffixProbabilities = scratch.nullSuffixProbabilities;
    prefixProbabilities.assign(scratch.dpWidth + 4);
    suffixProbabilities.assign(scratch.dpWidth + 4);

    ExpScore scalePerBase[simdWidth]{}, scalePerBaseInverse[simdWidth]{};
    ExpScore scaleOneNucleotide[simdWidth]{}, scaleTwoNucleotide[simdWidth]{}, scaleCodon[simdWidth]{};
    buildBatchNullModels(profile, transposedSequenceBases, sequenceLengths, activeCount,
        scratch.dpWidth, prefixProbabilities, suffixProbabilities,
        scalePerBase, scalePerBaseInverse, scaleOneNucleotide, scaleTwoNucleotide, scaleCodon);

    // Same tail zeroing as above.
    for (int j = scratch.dpWidth; j < scratch.dpWidth + 4; j++) {
        for (int k = 0; k < activeCount; k++) {
            if (j != (int)sequenceLengths[k]) {
                prefixProbabilities[j][k] = ExpScore{0.0f, 0};
                suffixProbabilities[j][k] = ExpScore{0.0f, 0};
            }
        }
        for (int k = activeCount; k < (int)simdWidth; k++) {
            prefixProbabilities[j][k] = ExpScore{0.0f, 0};
            suffixProbabilities[j][k] = ExpScore{0.0f, 0};
        }
    }

    simd_t nullScaleOneNucleotide = VecTraits<simd_t>::packScores(scaleOneNucleotide, activeCount);
    simd_t nullScaleTwoNucleotide = VecTraits<simd_t>::packScores(scaleTwoNucleotide, activeCount);
    simd_t nullScaleCodon = VecTraits<simd_t>::packScores(scaleCodon, activeCount);

    const size_t padded_seq_len = scratch.dpWidth + 4;
    scratch.filterRollingW1[0].resize(padded_seq_len, simd_t(0.0));
    scratch.filterRollingW1[1].resize(padded_seq_len, simd_t(0.0));
    auto &Y0_next = scratch.Y0_next; Y0_next.resize(padded_seq_len, 0.0);
    auto &Y0_curr = scratch.Y0_curr; Y0_curr.resize(padded_seq_len, 0.0);
#ifdef DUMMEST_POISON
    // Must be overwritten each row; fail loudly otherwise.
    std::fill(Y0_curr.begin(), Y0_curr.end(), VecTraits<simd_t>::poison());
#endif

    auto &prefixSeed = scratch.nullPrefixSeed; prefixSeed.resize(padded_seq_len, 0.0);
    auto &suffixSeedFromNext = scratch.nullSuffixSeedFromNext; suffixSeedFromNext.resize(padded_seq_len, 0.0);
    buildNullSeedModels(scratch.dpWidth, activeCount, sequenceLengths, prefixProbabilities, suffixProbabilities,
        scalePerBase, scalePerBaseInverse, prefixSeed, suffixSeedFromNext);

    fillBackgroundCodon(scratch, profile, transposedSequenceBases, scratch.dpWidth);
    simd_t* backgroundCodonBase = scratch.backgroundCodonProbabilities.data();

    simd_t globalBest = simd_t(0.0);

    for (int i = profile.length; i >= 0; i--) {
        const Params &params = profile.values_v2[i];
        const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

        const simd_t enterCoeff = params.enter_match_probability * nullScaleCodon;
        const simd_t delta0Coeff = params.delta_prime[0];
        const simd_t delta1Coeff = params.delta_prime[1] * nullScaleTwoNucleotide;
        const simd_t delta2Coeff = params.delta_prime[2] * nullScaleOneNucleotide;
        const simd_t alpha0Coeff = params.alpha_prime[0] * nullScaleCodon;
        const simd_t alpha1Coeff = params.alpha_prime[1] * nullScaleOneNucleotide;
        const simd_t alpha2Coeff = params.alpha_prime[2] * nullScaleTwoNucleotide;
        const simd_t beta0Coeff = params.beta_prime[0] * nullScaleCodon;
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

            const char* indices = (const char*)&transposedSequenceBases[(j + 1) * simdWidth];
            SimdFloat codon_raw = simdLookup(params_emission_probabilities, indices);
            simd_t codonEmitProbs(codon_raw);
            simd_t backgroundCodonEmission = backgroundCodonBase[j + 1];

            simd_t w1NextAtJ = scratch.filterRollingW1[(i + 1) & 1][j];

            simd_t wVal =
                w1NextJ3 * codonEmitProbs * enterCoeff +
                Y0_next[j + 0] * delta0Coeff +
                w1NextJ2 * delta1Coeff +
                w1NextJ1 * delta2Coeff +
                Z0_ring[r_3] * backgroundCodonEmission * alpha0Coeff +
                w1CurJ2 * alpha2Coeff +
                w1CurJ1 * alpha1Coeff
             + suffixSeedFromNext[j] * rowScaleCoeff;

            // Past-end lanes carry no mass (same masking as the forward pass).
            wVal = wVal * VecTraits<simd_t>::activeMask(j, sequenceLengths);

            simd_t mid_score = wVal * prefixSeed[j];
            globalBest = Kokkos::max(globalBest, mid_score);

            scratch.filterRollingW1[i & 1][j] = wVal;

            Y0_curr[j] = Kokkos::fma(epsCoeff, Y0_next[j], wVal) * VecTraits<simd_t>::activeMask(j, sequenceLengths);
            simd_t z0_future = Z0_ring[r_3] * backgroundCodonEmission;
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
    const simd_t nullScaleOneNucleotide = ctx.nullScaleOneNucleotide;
    const simd_t nullScaleTwoNucleotide = ctx.nullScaleTwoNucleotide;
    const simd_t nullScaleCodon = ctx.nullScaleCodon;
    uint8_t *transposedSequenceBases = scratch.transposedDecoded.data() + 4 * simdWidth;
    simd_t *backgroundCodonBase = scratch.backgroundCodonProbabilities.data();
    auto &prefixSeed = scratch.nullPrefixSeed;
    auto &suffixSeedFromNext = scratch.nullSuffixSeedFromNext;
    auto &reverseSide = scratch.reverseSide;
    auto &Y0_next = scratch.Y0_next;
    auto &Y0_curr = scratch.Y0_curr;
    const size_t padded_seq_len = scratch.dpWidth + 4;
    alignas(64) Float sequenceLengths[simdWidth] = {};
    fillSequenceLengths(activeCount, decoded, sequenceLengths);

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

            const simd_t enterCoeff = VecTraits<simd_t>::broadcast(params.enter_match_probability) * nullScaleCodon;
            const simd_t delta0Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[0]);
            const simd_t delta1Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[1]) * nullScaleTwoNucleotide;
            const simd_t delta2Coeff = VecTraits<simd_t>::broadcast(params.delta_prime[2]) * nullScaleOneNucleotide;
            const simd_t alpha0Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[0]) * nullScaleCodon;
            const simd_t alpha1Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[1]) * nullScaleOneNucleotide;
            const simd_t alpha2Coeff = VecTraits<simd_t>::broadcast(params.alpha_prime[2]) * nullScaleTwoNucleotide;
            const simd_t beta0Coeff = VecTraits<simd_t>::broadcast(params.beta_prime[0]) * nullScaleCodon;
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

                const char* indices = (const char*)&transposedSequenceBases[(j + 1) * simdWidth];
                simd_t codonEmitProbs = VecTraits<simd_t>::lookupEmit(params_emission_probabilities, indices);
                simd_t backgroundCodonEmission = backgroundCodonBase[j + 1];

                // Only gather the one new column needed for next iteration
                simd_t w1NextAtJ = gather_w1(i + 1, j);

                simd_t wVal =
                    w1NextJ3 * codonEmitProbs * enterCoeff +
                Y0_next[j] * delta0Coeff +
                    w1NextJ2 * delta1Coeff +
                    w1NextJ1 * delta2Coeff +
                    Z0_ring[r_3] * backgroundCodonEmission * alpha0Coeff +
                    w1CurJ2 * alpha2Coeff +
                    w1CurJ1 * alpha1Coeff
                 + suffixSeedFromNext[j] * rowScaleCoeff;

                // Past-end lanes carry no mass.
                wVal = wVal * VecTraits<simd_t>::activeMask(j, sequenceLengths);

                scatter_w1(wVal, i, j);
#ifdef ALIGN
                // Add now; per-base totals are built after the score gate.
                if (needAlign) {
                    reverseSide[j] = VecTraits<simd_t>::fma(VecTraits<simd_t>::broadcast((Float)1.0), wVal, reverseSide[j]);
                }
#endif
                backwardRowMax = VecTraits<simd_t>::vecMax(backwardRowMax, wVal);

                Y0_curr[j] = VecTraits<simd_t>::fma(epsCoeff, Y0_next[j], wVal) * VecTraits<simd_t>::activeMask(j, sequenceLengths);
                backwardRowMax = VecTraits<simd_t>::vecMax(backwardRowMax, Y0_curr[j]);
                simd_t z0_future = Z0_ring[r_3] * backgroundCodonEmission;
                Z0_ring[r_0] = VecTraits<simd_t>::fma(beta0Coeff, z0_future, wVal);

                // Move to the next base.
                w1NextJ3 = w1NextJ2;
                w1NextJ2 = w1NextJ1;
                w1NextJ1 = w1NextAtJ;
                w1CurJ2 = w1CurJ1;
                w1CurJ1 = wVal;
            }

            std::swap(Y0_curr, Y0_next);

            // Halve rows that grew too large; count the halves.
                alignas(64) Float maxArr[simdWidth];
                simd_unchecked_store(backwardRowMax, maxArr, Kokkos::Experimental::simd_flag_default);
                auto [factor, any] = makeRowRescaleFactor(scratch.backwardRescaleCount, i, maxArr, activeCount);
                if (any) {
                    simd_t *rp = scratch.W1.row_ptr(i);
                    for (int j = 0; j < scratch.dpWidth + 4; ++j) rp[j] *= factor;
                    rescaleRow(Y0_next, 0, (int)padded_seq_len, factor);
                    rescaleRow(Y0_curr, 0, (int)padded_seq_len, factor);
#ifdef ALIGN
                    // Keep padding in the halve too.
                    if (needAlign) rescaleRow(reverseSide.m, -3, (int)reverseSide.size(), factor);
#endif
                }
        }
        }

    std::fill(Y0_next.begin(), Y0_next.begin() + padded_seq_len, vzero);

    if (bestOut) {
        // Best score without a second DP: convert before multiplying.
        std::array<ExpScore, simdWidth> bestMid{};
        {
            const simd_t *__restrict__ np = prefixSeed.data();
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
    ctx.nullScaleOneNucleotide = nullScaleOneNucleotide;
    ctx.nullScaleTwoNucleotide = nullScaleTwoNucleotide;
    ctx.nullScaleCodon = nullScaleCodon;
}

