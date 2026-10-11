// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummest-core.hh"
#include "dummest-batch.hh"

struct DPScratchCommon {
    // Match scores that never overflow.
    ExpMatrix<0> X;
    ExpMatrix<3> prefixOptimal;
    // Best suffix scores, rebuilt for forward traceback.
    ExpMatrix<0> suffixOptimal;

    // Background odds of each prefix/suffix under no alignment.
    ExpVector<0> nullPrefixProbabilities, nullSuffixProbabilities;
    // Flank totals: entry sums run forward (~ L_j / 3Pn),
    // exit sums run backward (~ R_j / 3Pn).
    ExpVector<3> reverseSideEv;
    ExpVector<0> forwardSideEv;
    std::array<std::vector<AlignedSimilarity>, simdWidth> optProfilePosition;
    std::array<std::vector<bool>, simdWidth> aligned;
    std::vector<uint8_t> transposedDecoded;

    // Anchor of best match per query position.
    // bestMidScore holds static-scale-unit pairs (never overflows).
    std::array<std::vector<ExpScore>, simdWidth> bestMidScore;
    std::array<std::vector<int>, simdWidth> bestMidRow;

    // Best score seen per column. Updated in the hot loop,
    // copied out after the pass.
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
    // How many times each row was halved to stay finite.
    // Counts carry over: forward rows inherit from the row above,
    // backward rows from the row below.
    std::vector<std::array<int, simdWidth>> forwardRescaleCount, backwardRescaleCount;
    // Largest scores per row and per flank total. Used once
    // to pick a safe scale for the plain-float rebuild.
    std::vector<std::array<int, simdWidth>> combinedRowMaxExp;
    std::array<int, simdWidth> reverseSideEvMaxExp, forwardSideEvMaxExp;

    // Reusable scratch for findSimilarities
    PaddedVec<simd_t, 0, 0> W0_curr, W0_next;
    PaddedVec<simd_t, 0, 0> Y0_next, Y0_curr;
    // Background odds with per-base scaling folded in.
    // The suffix seed at j covers bases after j.
    PaddedVec<simd_t, 0, 0> nullPrefixSeed, nullSuffixSeedFromNext;
    // Running cut totals at one scale.
    PaddedSide<simd_t, 3, 0> reverseSide;
    PaddedSide<simd_t, 0, 3> forwardSide;
    PaddedVec<simd_t, 4, 4> backgroundCodonProbabilities;
};


uint8_t* buildTransposedDecoded(DPScratchCommon& scratch, int dpWidth, int activeCount,
                                 int zero_idx, const std::array<std::vector<uint8_t>*, simdWidth>& decoded) {
    scratch.transposedDecoded.assign((dpWidth + 8) * simdWidth, zero_idx);
    uint8_t* transposedSequenceBases = scratch.transposedDecoded.data() + 4 * simdWidth;
    for (int idx = 0; idx < activeCount; idx++) {
        for (size_t j = 0; j < decoded[idx]->size(); j++) {
            transposedSequenceBases[j * simdWidth + idx] = (*decoded[idx])[j];
        }
    }
    return transposedSequenceBases;
}

// Shared setup: lengths, sideways bases, background row.
// Empty lanes get length 0 so masks zero them.
inline void fillSequenceLengths(int activeCount,
        const std::array<std::vector<uint8_t>*, simdWidth> &decoded, Float *sequenceLengths) {
    for (int k = 0; k < activeCount; k++) {
        assert((int)decoded[k]->size() >= 3);
        sequenceLengths[k] = (Float)decoded[k]->size();
    }
    for (int k = activeCount; k < (int)simdWidth; k++) sequenceLengths[k] = 0;
}

inline uint8_t *buildBatchTransposed(DPScratchCommon &scratch, const Profile &profile,
        int activeCount, int dpWidth, const std::array<std::vector<uint8_t>*, simdWidth> &decoded) {
    int alphabetSize = profile.width - nonLetterWidth;
    int zero_idx = alphabetSize + 4; // padded index that maps to 0.0
    return buildTransposedDecoded(scratch, dpWidth, activeCount, zero_idx, decoded);
}

inline void fillBackgroundCodon(DPScratch &scratch, const Profile &profile,
        const uint8_t *transposedSequenceBases, int dpWidth) {
    const Float *backgroundProbs = profile.bg_probs.data() + 4;
    scratch.backgroundCodonProbabilities.resize(dpWidth);
    // Covers [-4, dpWidth+4).
    simd_t* backgroundCodonBase = scratch.backgroundCodonProbabilities.data();
    for (int j = -4; j < dpWidth + 4; j++) {
        const char* indices = (const char*)&transposedSequenceBases[j * simdWidth];
        backgroundCodonBase[j] = VecTraits<simd_t>::lookupBg(backgroundProbs, indices);
    }
}

#ifdef ALIGN
// Flank totals (S4: L_j / R_j for picking the shown alignment).
// Adds up every way to cover each outer piece with 1/2/3-base
// tiles, starting from the cut masses. Longer tiles count
// more (x3 for codons); the sum is divided by 3 to keep
// match scores comparable. Entry totals add up forward,
// exit totals backward, over the same per-base amounts.
// The first piece at the very start uses 1/3 (S4).
// Leftover end pieces with 1-2 bases use one 0.25 per base.

// Wrap one background-scaled value as a score.
inline ExpScore pairScoreFromFloat(Float value) { return normalizeScore((double)value, 0); }

// Convert one stored cut total back using its row's halve count.
inline ExpScore promoteColumnMassToPair(const simd_t &columnMass, int laneIndex, const int *rescaleCounts) {
    alignas(64) Float buffer[simdWidth];
    simd_unchecked_store(columnMass, buffer, Kokkos::Experimental::simd_flag_default);
    return promoteRescaledScore(buffer[laneIndex], rescaleCounts[laneIndex]);
}

// Background odds of the codon starting here.
// Past the end reads as 0.
inline ExpScore backgroundCodonProbabilityAt(const uint8_t *transposedSequenceBases, const Float *backgroundPerBase,
                               int position, int laneIndex) {
    uint8_t byte = transposedSequenceBases[position * simdWidth + laneIndex];
    return pairScoreFromFloat(backgroundPerBase[byte]);
}

template <int SPad, int SPadPost>
void buildFlankEVs(PaddedSide<simd_t, SPad, SPadPost> &forwardColumnAccumulator,
                   PaddedSide<simd_t, 3, 0> &backwardColumnAccumulator,
                   ExpVector<0> &exitFlankSums, ExpVector<3> &entryFlankSums,
                   const PaddedVec<simd_t, 0, 0> &nullPrefixSeed,
                   simd_t scaleOneNucleotide, simd_t scaleTwoNucleotide, simd_t scaleCodon,
                   const uint8_t *transposedSequenceBases, const Float *backgroundPerBase,
                    int dpWidth, int activeCount, const int *sequenceLengths,
                    const int *forwardRescaleCounts, const int *backwardRescaleCounts,
                    double singleShiftRate, double doubleShiftRate,
                    const bool *laneHit) {
    const double codonKeepProbability = 1.0 - singleShiftRate - doubleShiftRate;
    const double frameChoiceProbability = 1.0 / 3.0;
    const double oneThirdWeight = 1.0 / 3.0;
    const ExpScore zeroScore{0.0f, 0};
    alignas(64) Float singleScaleBuffer[simdWidth], doubleScaleBuffer[simdWidth], codonScaleBuffer[simdWidth];
    simd_unchecked_store(scaleOneNucleotide, singleScaleBuffer, Kokkos::Experimental::simd_flag_default);
    simd_unchecked_store(scaleTwoNucleotide, doubleScaleBuffer, Kokkos::Experimental::simd_flag_default);
    simd_unchecked_store(scaleCodon, codonScaleBuffer, Kokkos::Experimental::simd_flag_default);
    for (int laneIndex = 0; laneIndex < activeCount; laneIndex++) {
        if (!laneHit[laneIndex]) continue; // miss lane: EVs unread downstream
        const ExpScore absorbedSingleScale = pairScoreFromFloat(singleScaleBuffer[laneIndex]);
        const ExpScore absorbedDoubleScale = pairScoreFromFloat(doubleScaleBuffer[laneIndex]);
        const ExpScore absorbedCodonScale = pairScoreFromFloat(codonScaleBuffer[laneIndex]);
        const ExpScore bareSingleScale = scaleScore(absorbedSingleScale, 4.0); // drop the 0.25
        const ExpScore bareDoubleScale = scaleScore(absorbedDoubleScale, 16.0); // drop 0.25*0.25
        const int sequenceLength = sequenceLengths[laneIndex];
        auto backgroundCodonProbability = [&](int position) {
            return backgroundCodonProbabilityAt(transposedSequenceBases, backgroundPerBase, position, laneIndex);
        };
        // Start odds for each lane.
        std::vector<ExpScore> entryPrefixMass((size_t)dpWidth, zeroScore);
        for (int sequencePosition = 0; sequencePosition < dpWidth; sequencePosition++) entryPrefixMass[sequencePosition] = pairScoreFromFloat([&] {
            alignas(64) Float prefixBuffer[simdWidth];
            simd_unchecked_store(nullPrefixSeed[sequencePosition], prefixBuffer, Kokkos::Experimental::simd_flag_default);
            return prefixBuffer[laneIndex];
        }());
        // Pieces ending here. The first piece uses 1/3 (S4).
        std::vector<ExpScore> entryTileSingleMass((size_t)dpWidth, zeroScore), entryTileDoubleMass((size_t)dpWidth, zeroScore), entryTileCodonMass((size_t)dpWidth, zeroScore);
        if (dpWidth > 0) entryTileSingleMass[0] = entryPrefixMass[0];
        if (dpWidth > 1) entryTileDoubleMass[1] = scaleScore(absorbedDoubleScale, frameChoiceProbability);
        if (dpWidth > 2) entryTileCodonMass[2] = mulScores(scaleScore(absorbedCodonScale, frameChoiceProbability), backgroundCodonProbability(0));
        for (int sequencePosition = 1; sequencePosition < dpWidth; sequencePosition++) entryTileSingleMass[sequencePosition] = mulScores(scaleScore(entryPrefixMass[sequencePosition - 1], singleShiftRate), absorbedSingleScale);
        for (int sequencePosition = 2; sequencePosition < dpWidth; sequencePosition++) entryTileDoubleMass[sequencePosition] = mulScores(scaleScore(entryPrefixMass[sequencePosition - 2], doubleShiftRate), absorbedDoubleScale);
        for (int sequencePosition = 3; sequencePosition < dpWidth; sequencePosition++)
            entryTileCodonMass[sequencePosition] = mulScores(scaleScore(mulScores(entryPrefixMass[sequencePosition - 3], absorbedCodonScale), codonKeepProbability), backgroundCodonProbability(sequencePosition - 2));
        // Middle cover plus fresh cut mass, run backward.
        std::vector<ExpScore> middleEntryMass((size_t)dpWidth + 3, zeroScore);
        {
            const int lastPosition = std::min(dpWidth, sequenceLength);
            for (int sequencePosition = lastPosition - 1; sequencePosition >= 0; sequencePosition--) {
                ExpScore continuation = mulScores(scaleScore(middleEntryMass[sequencePosition + 1], singleShiftRate), absorbedSingleScale);
                continuation = addScores(continuation, mulScores(scaleScore(middleEntryMass[sequencePosition + 2], doubleShiftRate), absorbedDoubleScale));
                continuation = addScores(continuation, mulScores(scaleScore(mulScores(middleEntryMass[sequencePosition + 3], absorbedCodonScale), codonKeepProbability),
                                                backgroundCodonProbability(sequencePosition + 1)));
                middleEntryMass[sequencePosition] = addScores(promoteColumnMassToPair(backwardColumnAccumulator[sequencePosition], laneIndex, backwardRescaleCounts), continuation);
            }
        }
        // Cover of [j, end). Empty rest at end counts as 1.
        std::vector<ExpScore> suffixRestMass((size_t)dpWidth + 4, zeroScore);
        if (sequenceLength >= 0) suffixRestMass[sequenceLength] = pow2Score(0);
        for (int sequencePosition = sequenceLength - 1; sequencePosition >= 0; sequencePosition--) {
            ExpScore singleTile = mulScores(scaleScore(suffixRestMass[sequencePosition + 1], singleShiftRate), absorbedSingleScale);
            ExpScore doubleTile = (sequencePosition + 1 <= sequenceLength - 1) ? mulScores(scaleScore(suffixRestMass[sequencePosition + 2], doubleShiftRate), absorbedDoubleScale)
                                              : scaleScore(bareSingleScale, doubleShiftRate * 0.25);
            ExpScore codonTile;
            if (sequencePosition + 2 <= sequenceLength - 1) {
                codonTile = mulScores(scaleScore(mulScores(suffixRestMass[sequencePosition + 3], absorbedCodonScale), codonKeepProbability), backgroundCodonProbability(sequencePosition));
            } else if (sequencePosition + 1 <= sequenceLength - 1) {
                codonTile = scaleScore(bareDoubleScale, codonKeepProbability * 0.0625);
            } else {
                codonTile = scaleScore(bareSingleScale, codonKeepProbability * 0.25);
            }
            suffixRestMass[sequencePosition] = addScores(singleTile, addScores(doubleTile, codonTile));
        }
        // Cover ending here, grown forward from cut mass.
        std::vector<ExpScore> forwardArrivalMass((size_t)dpWidth, zeroScore);
        for (int sequencePosition = 0; sequencePosition < dpWidth; sequencePosition++) {
            ExpScore arrival = promoteColumnMassToPair(forwardColumnAccumulator[sequencePosition], laneIndex, forwardRescaleCounts);
            if (sequencePosition - 1 >= 0) arrival = addScores(arrival, mulScores(scaleScore(forwardArrivalMass[sequencePosition - 1], singleShiftRate), absorbedSingleScale));
            if (sequencePosition - 2 >= 0) arrival = addScores(arrival, mulScores(scaleScore(forwardArrivalMass[sequencePosition - 2], doubleShiftRate), absorbedDoubleScale));
            if (sequencePosition - 3 >= 0)
                arrival = addScores(arrival, mulScores(scaleScore(mulScores(forwardArrivalMass[sequencePosition - 3], absorbedCodonScale), codonKeepProbability), backgroundCodonProbability(sequencePosition - 2)));
            forwardArrivalMass[sequencePosition] = arrival;
        }
        // Per-base amounts: start pieces plus end pieces.
        // End pieces are cover before the tile, times the tile,
        // times cover after. Leftover end pieces at the last base
        // have empty rest.
        {
            std::vector<ExpScore> unifiedColumnMass((size_t)dpWidth, zeroScore);
            for (int sequencePosition = 0; sequencePosition < dpWidth; sequencePosition++) {
                ExpScore weightedEntryMass = addScores(entryTileSingleMass[sequencePosition], addScores(scaleScore(entryTileDoubleMass[sequencePosition], 2.0), scaleScore(entryTileCodonMass[sequencePosition], 3.0)));
                ExpScore columnTotal = mulScores(middleEntryMass[sequencePosition], weightedEntryMass);
                if (sequencePosition >= 1 && sequencePosition < sequenceLength) {
                    ExpScore exitSingleTile = mulScores(scaleScore(suffixRestMass[sequencePosition + 1], singleShiftRate), absorbedSingleScale);
                    columnTotal = addScores(columnTotal, mulScores(forwardArrivalMass[sequencePosition - 1], exitSingleTile));
                }
                if (sequencePosition >= 2 && sequencePosition < sequenceLength) {
                    ExpScore exitDoubleTile = mulScores(scaleScore(suffixRestMass[sequencePosition + 1], doubleShiftRate), absorbedDoubleScale);
                    columnTotal = addScores(columnTotal, scaleScore(mulScores(forwardArrivalMass[sequencePosition - 2], exitDoubleTile), 2.0));
                }
                if (sequencePosition >= 3 && sequencePosition < sequenceLength) {
                    ExpScore exitCodonTile = mulScores(scaleScore(mulScores(suffixRestMass[sequencePosition + 1], absorbedCodonScale), codonKeepProbability), backgroundCodonProbability(sequencePosition - 2));
                    columnTotal = addScores(columnTotal, scaleScore(mulScores(forwardArrivalMass[sequencePosition - 3], exitCodonTile), 3.0));
                }
                if (sequencePosition == sequenceLength - 1) {
                    // Leftover endings at the last base.
                    ExpScore truncatedDoubleTile = scaleScore(bareSingleScale, doubleShiftRate * 0.25);
                    columnTotal = addScores(columnTotal, scaleScore(mulScores(forwardArrivalMass[sequencePosition - 1], truncatedDoubleTile), 2.0));
                    ExpScore truncatedCodonSingleBase = scaleScore(bareSingleScale, codonKeepProbability * 0.25);
                    columnTotal = addScores(columnTotal, scaleScore(mulScores(forwardArrivalMass[sequencePosition - 1], truncatedCodonSingleBase), 3.0));
                    if (sequencePosition - 2 >= 0) {
                        ExpScore truncatedCodonDoubleBase = scaleScore(bareDoubleScale, codonKeepProbability * 0.0625);
                        columnTotal = addScores(columnTotal, scaleScore(mulScores(forwardArrivalMass[sequencePosition - 2], truncatedCodonDoubleBase), 3.0));
                    }
                }
                unifiedColumnMass[sequencePosition] = scaleScore(columnTotal, oneThirdWeight);
            }
            ExpScore cumulativeSum = zeroScore;
            for (int sequencePosition = 0; sequencePosition < dpWidth; sequencePosition++) {
                cumulativeSum = addScores(cumulativeSum, unifiedColumnMass[sequencePosition]);
                entryFlankSums[sequencePosition][laneIndex] = cumulativeSum;
            }
            cumulativeSum = zeroScore;
            for (int sequencePosition = dpWidth - 1; sequencePosition >= 0; sequencePosition--) {
                cumulativeSum = addScores(cumulativeSum, unifiedColumnMass[sequencePosition]);
                exitFlankSums[sequencePosition][laneIndex] = cumulativeSum;
            }
        }
    }
}

#endif

// Largest score in a flank total. Used once to pick a safe scale.
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

// Background scales handed to the forward half. Bulk arrays stay in scratch.
struct BackwardCtx {
    simd_t nullScaleOneNucleotide, nullScaleTwoNucleotide, nullScaleCodon;
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

