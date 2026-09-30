// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummer-dp.hh"

void addForwardMatch(std::vector<SegmentPair> &alignment, int pos1, int pos2) {
    if (!alignment.empty()) {
        SegmentPair &x = alignment.back();
        if (x.start1 + x.length == pos1 && x.start2 + x.length == pos2) {
            ++x.length;
        }
        if (x.start1 + x.length > pos1 || x.start2 + x.length > pos2)
            return;
    }
    SegmentPair sp = {pos1, pos2, 1};
    alignment.push_back(sp);
}

void addReverseMatch(std::vector<SegmentPair> &alignment, int pos1, int pos2) {
    if (!alignment.empty()) {
        SegmentPair &x = alignment.back();
        if (x.start1 - 1 == pos1 && x.start2 - 1 == pos2) {
            --x.start1;
            --x.start2;
            ++x.length;
        }
        if (x.start1 <= pos1 || x.start2 <= pos2)
            return;
    }
    SegmentPair sp = {pos1, pos2, 1};
    alignment.push_back(sp);
}


void addForwardAlignment(int idx, size_t profileLength, std::vector<SegmentPair> &alignment, int iBeg, int jBeg,
                          DPScratchCommon &scratch) {
    int cols = (int)scratch.suffixOptimal.cols();
    int dp_bound = cols - 4;

    // All candidates compared as ExpScore in static-scale units (same
    // formulas as before, but with double-mantissa range so overflowed
    // regimes keep their true ordering). suffixOptimal holds the rebuilt
    // suffix-optimal values; forwardSideEv is already in pair domain.
    // An all-zero tie leaves bi/bj at the sentinel and exits the walk.
    int i = iBeg, abs_pos = jBeg;
    while (i <= profileLength && abs_pos < dp_bound) {
        ExpScore best{0.0, 0};
        int bi = INT_MAX, bj = INT_MAX;
        bool bemit = false;
        auto consider = [&](const ExpScore &cand, int ni, int nj, bool emit) {
            if (scoreLess(best, cand)) { best = cand; bi = ni; bj = nj; bemit = emit; }
        };
        if (abs_pos + 3 < cols)
            consider(addScores(scratch.X.get_lane(i, abs_pos + 3, idx),
                            scratch.suffixOptimal.get_lane(i + 1, abs_pos + 3, idx)),
                     i + 1, abs_pos + 3, true);
        consider(scratch.suffixOptimal.get_lane(i + 1, abs_pos, idx), i + 1, abs_pos, false);
        consider(scratch.suffixOptimal.get_lane(i, abs_pos + 1, idx), i, abs_pos + 1, false);
        consider(scratch.forwardSideEv[abs_pos][idx], INT_MAX, INT_MAX, false);

        if (bemit) {
            addForwardMatch(alignment, i, abs_pos + 1);
        }
        i = bi, abs_pos = bj;
    }
}

void addReverseAlignment(int idx, std::vector<SegmentPair> &alignment, int iEnd, int jEnd,
                         DPScratchCommon &scratch) {
    int i = iEnd - 1, abs_pos = jEnd;
    while (i >= 0 && abs_pos >= 0) {
        ExpScore best{0.0, 0};
        int bi = -1, bj = -1;
        bool bemit = false;
        auto consider = [&](const ExpScore &cand, int ni, int nj, bool emit) {
            if (scoreLess(best, cand)) { best = cand; bi = ni; bj = nj; bemit = emit; }
        };
        if (abs_pos >= 3) {
            ExpScore opt_succ = (i >= 1) ? scratch.prefixOptimal.get_lane(i - 1, abs_pos - 3, idx)
                                         : ExpScore{0.0, 0};
            consider(addScores(scratch.X.get_lane(i, abs_pos, idx), opt_succ),
                     i - 1, abs_pos - 3, true);
        }
        if (i >= 1)
            consider(scratch.prefixOptimal.get_lane(i - 1, abs_pos, idx), i - 1, abs_pos, false);
        if (abs_pos > 0)
            consider(scratch.prefixOptimal.get_lane(i, abs_pos - 1, idx), i, abs_pos - 1, false);
        consider(scratch.reverseSideEv[abs_pos][idx], -1, -1, false);

        if (bemit) {
            addReverseMatch(alignment, i, abs_pos - 2);
        }
        i = bi, abs_pos = bj;
    }
}

void addMidAnchored(int idx, size_t profileLength, std::vector<AlignedSimilarity> &similarities, int anchor1, int anchor2,
                    const ExpScore &pair, DPScratchCommon &scratch) {
    AlignedSimilarity s;
    s.pair = pair;
    s.logProbRatio = scoreToLog(pair);
    s.anchor1 = anchor1;
    s.anchor2 = anchor2;
    // Traceback walks exact pair-domain matrices, so it runs unconditionally:
    // even a beyond-double report gets its full alignment.
#ifdef ALIGN
    addForwardAlignment(idx, profileLength, s.alignment, anchor1, anchor2, scratch);
#endif
    similarities.push_back(s);
}

void finishMidAnchored(int idx, AlignedSimilarity &s, DPScratchCommon &scratch) {
    reverse(s.alignment.begin(), s.alignment.end());
#ifdef ALIGN
    addReverseAlignment(idx, s.alignment, s.anchor1, s.anchor2, scratch);
#endif
    reverse(s.alignment.begin(), s.alignment.end());
}


