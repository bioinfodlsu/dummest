// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummer-core.hh"

char profileLetter(const char *alphabet, char letterCode) {
    return alphabet[letterCode & 31] + (letterCode & 32); // upper/lowercase
}

void addAlignedProfile(std::vector<char> &gappedSeq, const std::vector<SegmentPair> &alignment,
                       const char *alphabet, const char *consensusSequence) {
    int pos1 = alignment[0].start1;
    int pos2 = alignment[0].start2;
    for (auto a : alignment) {
        for (; pos1 < a.start1; ++pos1) {
            gappedSeq.push_back(profileLetter(alphabet, consensusSequence[pos1]));
        }
        gappedSeq.insert(gappedSeq.end(), a.start2 - pos2, '-');
        for (; pos1 < a.start1 + a.length; ++pos1) {
            gappedSeq.push_back(profileLetter(alphabet, consensusSequence[pos1]));
        }
        pos2 = a.start2 + a.length;
    }
}

char seqLetter(const char *alphabet, const char *sequence, const char *maskedSequence,
               int position) {
    char c = sequence[position];
    return alphabet[c] + (maskedSequence[position] > c) * 32; // upper/lowercase
}

void addAlignedSequence(std::vector<char> &gappedSeq, const std::vector<SegmentPair> &alignment,
                        const char *alphabet, const char *sequence, const char *maskedSequence) {
    int pos1 = alignment[0].start1;
    int pos2 = alignment[0].start2;
    for (auto a : alignment) {
        gappedSeq.insert(gappedSeq.end(), a.start1 - pos1, '-');
        for (; pos2 < a.start2; ++pos2) {
            gappedSeq.push_back(seqLetter(alphabet, sequence, maskedSequence, pos2));
        }
        for (; pos2 < a.start2 + a.length; ++pos2) {
            gappedSeq.push_back(seqLetter(alphabet, sequence, maskedSequence, pos2));
        }
        pos1 = a.start1 + a.length;
    }
}

int strandPosition(size_t strandNum, int seqLength, int position) {
    return (strandNum % 2) ? seqLength - position : position;
}
// Decimal display for log-scores: fixed notation (never scientific) with
// trailing zeros trimmed; non-finite stays "-inf" so test_scores.sh
// awk '$5!="-inf"' keeps working. Computation stays in double.
inline std::string scoreStr(double v) {
    if (v == 0.0) v = 0.0; // normalize -0.0
    if (!std::isfinite(v)) {
        std::ostringstream os;
        os << v;
        return os.str();
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << v;
    std::string s = os.str();
    auto dot = s.find('.');
    if (dot != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
    }
    return s;
}
void printSimilarity(std::ostream &os, const char *names, const Profile &p, Sequence s,
                     const FinalSimilarity &sim, double evalue) {
    if (std::isnan(evalue)) {
        return;
    }
    char strand = "+-"[!s.is_plus];
    const char *seq = sim.alignedSequences.data();
    int length = sim.alignedSequences.size() / 2;
    int span1 = length - std::count(seq, seq + length, '-');
    int span2 = length - std::count(seq + length, seq + length * 2, '-');
    int start2 = strandPosition(sim.strandNum, s.length, sim.start2);
    if (s.has_pipeline_fields) {
        if (s.is_plus) {
            start2 = s.w_start - 1 + start2;
        } else {
            start2 = s.true_length - s.w_end + start2;
        }
    }
    int reportSeqLength = s.has_pipeline_fields ? s.true_length : s.length;
    int anchor2 = strandPosition(sim.strandNum, reportSeqLength, sim.anchor2);
    int w1 = std::max(strlen(names + p.nameIdx), strlen(names + s.nameIdx));
    int w2 = std::max(numOfDigits(sim.start1), numOfDigits(start2));
    int w3 = std::max(numOfDigits(span1), numOfDigits(span2));
    int w4 = std::max(numOfDigits(p.length), numOfDigits(reportSeqLength));
    os << "a score=" << scoreStr(sim.logProbRatio + shift) << " E=" << evalue
       << " anchor=" << sim.anchor1 << "," << anchor2 << "\n";
    os << "s " << std::left << std::setw(w1) << names + p.nameIdx << " " << std::right
       << std::setw(w2) << sim.start1 << " " << std::setw(w3) << span1 << " " << '+' << " "
       << std::setw(w4) << p.length << " ";
    os.write(seq, length);
    os << "\n";
    os << "s " << std::left << std::setw(w1) << names + s.nameIdx << " " << std::right
       << std::setw(w2) << start2 << " " << std::setw(w3) << span2 << " " << strand << " "
       << std::setw(w4) << reportSeqLength << " ";
    os.write(seq + length, length);
    os << "\n\n";
}

// Lockless record emission: worker threads format records into a per-call
// string (evalue + formatting OUTSIDE any lock), then emit the whole chunk
// with ONE fwrite call. POSIX requires stdio ops on the same FILE to behave
// as if under flockfile, so concurrent single-call fwrites never interleave
// or split: no explicit mutex needed, no extra syscalls per record.
// (An earlier O_APPEND-dup design was removed: mixing raw O_APPEND writes
// with stdio-buffered writes to the same file lets a stdio flush land at a
// stale offset and clobber appended records.)
inline void emitChunk(const std::string &chunk) {
    if (chunk.empty()) return;
    std::fwrite(chunk.data(), 1, chunk.size(), stdout);
}


int contigToSequencePos(Contig contig, size_t strandNum, int posInContig) {
    return contig.start + strandPosition(strandNum, contig.length, posInContig);
}


