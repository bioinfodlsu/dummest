// Author: Martin C. Frith 2025
// SPDX-License-Identifier: BSD-3-Clause

// See [Frith2025]: "Simple and thorough detection of related
// sequences with position-varying probabilities of substitutions,
// insertions, and deletions", MC Frith 2025

#include "dummer-util.hh"
#include "tantan-wrapper.hh"
// clang-format off
#include "can_i_haz_simd.hh"
// clang-format on

#include <algorithm>
#include <array>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <utility>

#include <assert.h>
#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <getopt.h>
#include <map>
#include <memory>
#include <optional>
#include <queue>

#include <Kokkos_SIMD.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>

#define XXH_INLINE_ALL
#include <xxhash.h>

#define OPT_e 10
#define OPT_s 2
#define OPT_m 3
#define OPT_t 1000
#define OPT_l 5000
#define OPT_b 100
#define OPT_insert1 0.0171
#define OPT_insert2 0.0018
#define OPT_delete1 0.0328
#define OPT_delete2 0.0083
#define OPT_stop 0.0005
#define OPT_bg_stop 0.046875 // 3/64
#define OPT_tantan 0.5
#define EVALUE
#define ALIGN

// using through BATH heuristic pipeline
#define PIPELINE_MODE

// uncomment to use codon probabilities instead of base probabilities to generate random sequences
#define ESTIMATOR_USE_RANDOM_CODONS

// Frameshift / stop-codon / masking parameters (user-configurable via CLI,
// defaults are the OPT_* values above).  BACKGROUND_* are derived as:
//   BACKGROUND_FRAMESHIFT_RATE = INSERT1 + DELETE2 (1-bp branch)
//   BACKGROUND_FRAMESHIFT_RATE_2 = INSERT2 + DELETE1 (2-bp branch)
enum {
    OPT_INSERT1_CODE = 1000,
    OPT_INSERT2_CODE,
    OPT_DELETE1_CODE,
    OPT_DELETE2_CODE,
    OPT_STOP_CODE,
    OPT_BG_STOP_CODE,
    OPT_TANTAN_CODE,
    OPT_MAX_CODE
};

#ifdef DOUBLE
typedef double Float;
typedef SimdDbl SimdFloat;
const int simdLen = simdDblLen;
#else
typedef float Float;
typedef SimdFlt SimdFloat;
const int simdLen = simdFltLen;
#endif
using simd_t = Kokkos::Experimental::simd<Float>;
constexpr auto simdWidth = simd_t::size();

Float STOP_CODON_PROB = OPT_stop;
Float BG_STOP_CODON_PROB = OPT_bg_stop; // 3/64

Float INSERT1 = OPT_insert1;
Float INSERT2 = OPT_insert2;
Float DELETE1 = OPT_delete1;
Float DELETE2 = OPT_delete2;

Float BACKGROUND_FRAMESHIFT_RATE = INSERT1 + DELETE2;
Float BACKGROUND_FRAMESHIFT_RATE_2 = INSERT2 + DELETE1;

Float TANTAN_MASK_THRESHOLD = OPT_tantan;

static void updateBackgroundFrameshiftRates() {
    BACKGROUND_FRAMESHIFT_RATE = INSERT1 + DELETE2;
    BACKGROUND_FRAMESHIFT_RATE_2 = INSERT2 + DELETE1;
}

int simdRoundUp(int x) { // lowest multiple of simdLen that is >= x
    return x - 1 - (x - 1) % simdLen + simdLen;
}

class ThreadPool {
public:
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this, i] {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
                _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
                while (true) {
                    std::function<void(int)> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task(i);
                }
            });
        }
    }

    void enqueue(std::function<void(int)> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void(int)>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;
};

// Only consider similarities that are local maxima.  If 2
// similarities have identical 1st anchor coordinates, and their 2nd
// anchor coordinates are closer than this, omit the lower-scoring one.
const int minSeparation = 32; // xxx ???

// down-scale probabilities by this amount, to delay overflow:
const Float scale = 1.0 / (1 << 30) / (1 << 30) / (1 << 3); // sqrt[min normal float]
const Float invScale = 1.0 / scale;
const int shift = 63; // add this to scores, to undo the scaling

int verbosity = 0;

const int nonLetterWidth = 9; // number of non-letter values per position

struct Params { // TODO: maybe SIMD order
    Float alpha_prime[3];
    Float beta_prime[3];
    Float delta_prime[3];
    Float epsilon_prime;
    Float enter_match_probability;
};

struct Profile {   // position-specific (insert, delete, letter) probabilities
    Float *values; // probabilities or probability ratios
    std::vector<Params> values_v2;
    std::vector<Float> bg_probs, log2_bg_probs;
    int width;  // number of values per position
    int length; // number of positions
    size_t nameIdx;
    size_t consensusSequenceIdx;
    double gumbelKendAnchored, gumbelKbegAnchored, gumbelKmidAnchored, lambda;
#ifdef FORWARD_ONLY_FILTER
    double gumbel_k_forward_only, lambda_forward_only;
#endif
    std::string name;
};

struct Sequence {
    size_t nameIdx;
    int length;

#ifdef PIPELINE_MODE
    size_t w_start, w_end, true_length;
    std::string target_profile;
    bool is_plus;
    bool has_pipeline_fields = false;
#endif
};

struct Contig {
    int start;
    int length;
};

struct SegmentPair {
    int start1, start2, length;
};

struct InitialSimilarity {
    Float probRatio;
    int anchor2; // 2nd anchor coordinate (don't need to store the 1st one)
};

struct SequenceData {
    std::vector<uint8_t> decoded;
    std::string sequence;
    std::string maskedSequence;
    Contig contig;
    size_t strandNum;
};

struct SequenceRequest {
    std::shared_ptr<SequenceData> seqData;
    Float minProbRatio;

    bool operator<(const SequenceRequest &other) const {
        return seqData->decoded.size() < other.seqData->decoded.size();
    }

    bool operator>(const SequenceRequest &other) const {
        return seqData->decoded.size() > other.seqData->decoded.size();
    }
};

struct AlignedSimilarity {
    double probRatio;
    int anchor1, anchor2;
    Float wEndAnchored;
    std::vector<SegmentPair> alignment;

    bool operator<(const AlignedSimilarity &other) const {
        return this->probRatio < other.probRatio;
    }
    bool operator>(const AlignedSimilarity &other) const {
        return this->probRatio > other.probRatio;
    }
};

struct FinalSimilarity {
    double probRatio;
    size_t profileNum;
    size_t strandNum;
    int anchor1, anchor2;
    int start1, start2;
    std::vector<char> alignedSequences;
};

int simBeg2(const AlignedSimilarity &x) {
    return x.alignment.empty() ? x.anchor2 : x.alignment[0].start2;
}

int simEnd2(const AlignedSimilarity &x) {
    return x.alignment.empty() ? x.anchor2 : x.alignment.back().start2 + x.alignment.back().length;
}

double mean(const double *x, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i)
        s += x[i];
    return s / n;
}

int numOfDigits(int x) {
    int n = 0;
    do
        ++n;
    while (x /= 10);
    return n;
}

const char *getAlphabet(int alphabetSize) {
    assert(alphabetSize == 20 || alphabetSize == 4);
    return alphabetSize == 20  ? "ACDEFGHIKLMNPQRSTVWYUO?*" // 20 + 2 amino acids
           : alphabetSize == 4 ? "ACGT"
                               : 0;
}

char complement(char c) {
    // Map DNA bases correctly using the protein alphabet indices
    // "ACDEFGHIKLMNPQRSTVWYUO?*" -> A=0, C=1, G=5, T=16
    switch (c) {
    case 0:
        return 16; // A -> T
    case 16:
        return 0; // T -> A
    case 1:
        return 5; // C -> G
    case 5:
        return 1; // G -> C
    default:
        return c; // Fallback for masked '?' or unrecognized chars
    }
}

void reverseComplement(char *beg, char *end) {
    while (beg < end) {
        char c = *--end;
        *end = complement(*beg);
        *beg++ = complement(c);
    }
}

std::istream &readContig(std::istream &in, Sequence &sequence, Contig &contig,
                         std::vector<char> &vec, const char *charToNumber) {
    if (contig.length == 0) {
        char x;
        if (!(in >> x))
            return in;
        if (x != '>')
            return fail(in, "bad sequence data: no '>'");
        std::string line, word;
        getline(in, line);
        std::istringstream iss(line);
        if (!(iss >> word))
            return fail(in, "bad sequence data: no name");
#ifdef PIPELINE_MODE
        size_t slash = word.find('/');
        if (slash != std::string::npos) {
            std::string chr = word.substr(0, slash);
            std::string range = word.substr(slash + 1);
            size_t dash = range.find('-');
            if (dash != std::string::npos) {
                sequence.w_start = std::stoi(range.substr(0, dash));
                sequence.w_end = std::stoi(range.substr(dash + 1));
            }
            std::string length, profile, strand;
            if (iss >> length >> profile >> strand) {
                sequence.true_length = stoll(length.substr(length.find('=') + 1));
                sequence.target_profile = profile.substr(profile.find('=') + 1);
                sequence.is_plus = strand.find("plus_strand") != std::string::npos;
            }
            sequence.has_pipeline_fields = true;
            word = chr;
        }
#endif
        sequence.nameIdx = vec.size();
        sequence.length = 0;
        const char *name = word.c_str();
        vec.insert(vec.end(), name, name + word.size() + 1);
        if (verbosity > 0)
            std::cerr << "Sequence: " << name << "\n";
    }

    size_t seqIdx = vec.size();
    std::streambuf *buf = in.rdbuf();
    int c = buf->sgetc();

    while (c != std::streambuf::traits_type::eof() && c != '>') {
        if (charToNumber[c] < 125)
            break; // found a contig symbol
        if (c > ' ')
            ++sequence.length; // skip over non-contig symbols
        c = buf->snextc();
    }

    while (c != std::streambuf::traits_type::eof() && charToNumber[c] < 126) {
        if (c > ' ') {
            vec.push_back(charToNumber[c]);
        }
        c = buf->snextc();
    }

    size_t seqLen = vec.size() - seqIdx;
    if (seqLen > INT_MAX - 2 * simdLen)
        return fail(in, "sequence is too long!");
    contig.start = sequence.length;
    contig.length = seqLen;
    sequence.length += seqLen;
    return in;
}

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
void printSimilarity(const char *names, Profile &p, Sequence s, const FinalSimilarity &sim,
                     double evalue) {
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
    std::cout << "a score=" << (log(sim.probRatio) + shift) << " E=" << evalue
              << " anchor=" << sim.anchor1 << "," << anchor2 << "\n";
    std::cout << "s " << std::left << std::setw(w1) << names + p.nameIdx << " " << std::right
              << std::setw(w2) << sim.start1 << " " << std::setw(w3) << span1 << " " << '+' << " "
              << std::setw(w4) << p.length << " ";
    std::cout.write(seq, length);
    std::cout << "\n";
    std::cout << "s " << std::left << std::setw(w1) << names + s.nameIdx << " " << std::right
              << std::setw(w2) << start2 << " " << std::setw(w3) << span2 << " " << strand << " "
              << std::setw(w4) << reportSeqLength << " ";
    std::cout.write(seq + length, length);
    std::cout << "\n\n";
}

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

using DP_Cell = Float;
template <typename T, bool Rolling = false, int PrePad = 0, int PostPad = 0>
class FlatMatrix {
    std::vector<T> data;
    size_t logical_cols;
    size_t logical_rows;
    int offset_ = 0;

    size_t phys_cols() const { return PrePad + logical_cols + PostPad; }

    int phys_index(int j) const { return (j - offset_) + PrePad; }

public:
    FlatMatrix() : logical_cols(0), logical_rows(0) {}

    int offset() const { return offset_; }
    void set_offset(int off) { offset_ = off; }

    T get(size_t i, int j) const {
        int p = phys_index(j);
        if (p < 0 || p >= (int)phys_cols()) return T{};
        if constexpr (Rolling)
            return data[(i & 1) * phys_cols() + p];
        else
            return data[i * phys_cols() + p];
    }

    void set(size_t i, int j, T val) {
        int p = phys_index(j);
        if (p < 0 || p >= (int)phys_cols()) return;
        if constexpr (Rolling)
            data[(i & 1) * phys_cols() + p] = val;
        else
            data[i * phys_cols() + p] = val;
    }

    void resize(size_t r, size_t c) {
        offset_ = 0;
        logical_rows = r;
        logical_cols = c;
        size_t n = Rolling ? 2 : r;
        size_t total = n * phys_cols();
        data.resize(total);
    }

    void assign(size_t r, size_t c, T init = T()) {
        offset_ = 0;
        logical_rows = r;
        logical_cols = c;
        size_t n = Rolling ? 2 : r;
        size_t total = n * phys_cols();
        data.assign(total, init);
        for (size_t i = 0; i < n; ++i) {
            T *row = data.data() + i * phys_cols();
            std::fill_n(row, PrePad, T{});
            std::fill_n(row + PrePad + logical_cols, PostPad, T{});
        }
    }

    inline T &operator()(size_t i, int j) {
        if constexpr (Rolling) {
            return data[(i & 1) * phys_cols() + phys_index(j)];
        } else {
            return data[i * phys_cols() + phys_index(j)];
        }
    }

    inline const T &operator()(size_t i, int j) const {
        if constexpr (Rolling) {
            return data[(i & 1) * phys_cols() + phys_index(j)];
        } else {
            return data[i * phys_cols() + phys_index(j)];
        }
    }

    inline void clear_row(size_t i, T init_val = T()) {
        size_t actual_row = Rolling ? (i & 1) : i;
        auto row_start = data.begin() + actual_row * phys_cols() + PrePad;
        std::fill(row_start, row_start + logical_cols, init_val);
    }

    inline T *row_ptr(size_t i) {
        if constexpr (Rolling) {
            return data.data() + (i & 1) * phys_cols() + PrePad;
        } else {
            return data.data() + i * phys_cols() + PrePad;
        }
    }

    inline const T *row_ptr(size_t i) const {
        if constexpr (Rolling) {
            return data.data() + (i & 1) * phys_cols() + PrePad;
        } else {
            return data.data() + i * phys_cols() + PrePad;
        }
    }

    size_t cols() const { return logical_cols; }

    void set_cols(size_t c) {
        offset_ = 0;
        logical_cols = c;
        size_t n = Rolling ? 2 : logical_rows;
        size_t total = n * phys_cols();
        data.resize(total);
    }

    size_t rows() const { return Rolling ? 2 : logical_rows; }
};

template <typename T, int PrePad = 0, int PostPad = 0>
class PaddedVec {
    std::vector<T> data_;
    int logical_size_ = 0;
public:
    PaddedVec() = default;

    T &operator[](int i) { return data_[PrePad + i]; }

    const T &operator[](int i) const { return data_[PrePad + i]; }

    T *data() { return data_.data() + PrePad; }

    const T *data() const { return data_.data() + PrePad; }

    int size() const { return logical_size_; }
    bool empty() const { return logical_size_ == 0; }

    auto begin() { return data(); }
    auto end() { return data() + logical_size_; }
    auto begin() const { return data(); }
    auto end() const { return data() + logical_size_; }

    void resize(int n, T val = T{}) {
        logical_size_ = n;
        data_.assign(n + PrePad + PostPad, val);
        std::fill_n(data_.data(), PrePad, T{});
        std::fill_n(data_.data() + PrePad + n, PostPad, T{});
    }

    void assign(int n, T val = T{}) { resize(n, val); }

    void fill_logical(T val) {
        std::fill_n(data_.data() + PrePad, logical_size_, val);
    }

    void set_logical_size(int n) {
        logical_size_ = n;
        data_.resize(n + PrePad + PostPad, T{});
        std::fill_n(data_.data(), PrePad, T{});
        std::fill_n(data_.data() + PrePad + n, PostPad, T{});
    }

    int logical_size() const { return logical_size_; }
};

// SIMD-native 2D matrix: rows of PaddedVec<simd_t>.  Lane k of element [i][j]
// holds the value for sequence k at (row i, column j).  This eliminates the
// per-lane gather/scatter transpose that FlatMatrix required.
template <int PrePad = 0, int PostPad = 0>
class SimdMatrix {
    std::vector<PaddedVec<simd_t, PrePad, PostPad>> rows_;
    int logical_cols_ = 0;
public:
    SimdMatrix() = default;

    void assign(size_t nrows, int ncols, simd_t init = simd_t(0.0)) {
        logical_cols_ = ncols;
        rows_.resize(nrows);
        for (auto& r : rows_) r.resize(ncols, init);
    }

    simd_t get(size_t i, int j) const { return rows_[i][j]; }
    void set(size_t i, int j, simd_t v) { rows_[i][j] = v; }
    Float get_lane(size_t i, int j, int k) const { return rows_[i][j][k]; }
    simd_t *row_ptr(size_t i) { return rows_[i].data(); }
    const simd_t *row_ptr(size_t i) const { return rows_[i].data(); }

    int cols() const { return logical_cols_; }
    size_t rows() const { return rows_.size(); }
};

struct DPScratch {
    // SIMD-native W1 matrix: simd_t per cell, one value per lane
    SimdMatrix<0, 0> W1;
    // 2-row rolling buffer for findSimilaritiesBackwardOnly
    std::array<PaddedVec<simd_t, 0, 0>, 2> W1_rolling;
    SimdMatrix<0, 0> X;
    SimdMatrix<3, 0> X_pfx;

    // Reusable temporary buffers for findSimilarities
    PaddedVec<simd_t, 0, 0> W0_curr, W0_next;
    PaddedVec<simd_t, 0, 0> null_probs_prefix, null_probs_suffix;
    PaddedVec<simd_t, 0, 0> Y0_next, Y0_curr;
    PaddedVec<simd_t, 0, 0> null_model_prefix, null_model_suffix;
    PaddedVec<simd_t, 3, 0> right_side, right_side_EV;
    PaddedVec<simd_t, 0, 3> left_side, left_side_EV;
    std::array<std::vector<AlignedSimilarity>, simdWidth> opt_profile_position;
    std::array<std::vector<bool>, simdWidth> aligned;
    std::vector<uint8_t> transposed_decoded;
    PaddedVec<simd_t, 4, 4> bg_codon_probs;

    // Per-lane anchor tracking — indexed by sequence position j
    std::array<std::vector<Float>, simdWidth> best_wMid;
    std::array<std::vector<Float>, simdWidth> best_wEnd;
    std::array<std::vector<int>, simdWidth> best_i;

    // SIMD-native best trackers indexed by DP column. Updated branchlessly
    // in the hot loop; extracted into the per-lane arrays after the pass.
    std::vector<simd_t> best_wMid_phys, best_wEnd_phys, best_i_phys;

    // Current DP column count (= max sequence length in batch)
    int active_dp_width = 0;
};
struct DP_Cell_v2 {
    Float metric;
    int i, j;
    bool emit = false;

    constexpr bool operator< (const DP_Cell_v2 &other) const {
        return metric < other.metric;
    }
};

void addForwardAlignment(int idx, size_t profileLength, std::vector<SegmentPair> &alignment, int iBeg, int jBeg,
                         DPScratch &scratch) {
    int cols = (int)scratch.W1.cols();
    int dp_bound = cols - 4;

    int i = iBeg, abs_pos = jBeg;
    while (i <= profileLength && abs_pos < dp_bound) {
        auto choice = std::max({
            DP_Cell_v2{.metric=(abs_pos + 3 < cols ? scratch.X.get_lane(i, abs_pos + 3, idx) + scratch.W1.get_lane(i + 1, abs_pos + 3, idx) : -INFINITY), .i=i + 1, .j=abs_pos + 3, .emit=true},
            DP_Cell_v2{.metric=scratch.W1.get_lane(i + 1, abs_pos, idx), .i=i + 1, .j=abs_pos, .emit=false},
            DP_Cell_v2{.metric=scratch.W1.get_lane(i, abs_pos + 1, idx), .i=i, .j=abs_pos + 1, .emit=false},
            DP_Cell_v2{.metric=scratch.left_side_EV[abs_pos][idx], .i=INT_MAX, .j=INT_MAX, .emit=false},
        });

        if (choice.emit) {
            addForwardMatch(alignment, i, abs_pos + 1);
        }
        i = choice.i, abs_pos = choice.j;
    }
}

void addReverseAlignment(int idx, std::vector<SegmentPair> &alignment, int iEnd, int jEnd,
                         DPScratch &scratch) {
    int i = iEnd - 1, abs_pos = jEnd;
    while (i >= 0 && abs_pos >= 0) {
        DP_Cell opt_succ = 0;
        if (i >= 1 && abs_pos >= 3) {
            opt_succ = scratch.X_pfx.get_lane(i - 1, abs_pos - 3, idx);
        }
        auto choice = std::max({
            DP_Cell_v2{.metric=(abs_pos >= 3 ? scratch.X.get_lane(i, abs_pos, idx) + opt_succ : -INFINITY), .i=i - 1, .j=abs_pos - 3, .emit=true},
            DP_Cell_v2{.metric=(i >= 1 ? scratch.X_pfx.get_lane(i - 1, abs_pos, idx) : -INFINITY), .i=i - 1, .j=abs_pos, .emit=false},
            DP_Cell_v2{.metric=(abs_pos > 0 ? scratch.X_pfx.get_lane(i, abs_pos - 1, idx) : -INFINITY), .i=i, .j=abs_pos - 1, .emit=false},
            DP_Cell_v2{.metric=scratch.right_side_EV[abs_pos][idx], .i=-1, .j=-1, .emit=false},
        });

        if (choice.emit) {
            addReverseMatch(alignment, i, abs_pos - 2);
        }
        i = choice.i, abs_pos = choice.j;
    }
}

void addMidAnchored(int idx, size_t profileLength, std::vector<AlignedSimilarity> &similarities, int anchor1, int anchor2,
                    Float wBegAnchored, Float wEndAnchored, DPScratch &scratch) {
    Float wMidAnchored = wEndAnchored * wBegAnchored;
    AlignedSimilarity s = {wMidAnchored / scale, anchor1, anchor2, wEndAnchored};
#ifdef ALIGN
    if (!isinf(s.probRatio))
        addForwardAlignment(idx, profileLength, s.alignment, anchor1, anchor2, scratch);
#endif
    similarities.push_back(s);
}

void finishMidAnchored(int idx, AlignedSimilarity &s, DPScratch &scratch) {
    reverse(s.alignment.begin(), s.alignment.end());
#ifdef ALIGN
    if (!isinf(s.probRatio))
        addReverseAlignment(idx, s.alignment, s.anchor1, s.anchor2, scratch);
#endif
    reverse(s.alignment.begin(), s.alignment.end());
}

bool isLess(const AlignedSimilarity &a, const AlignedSimilarity &b) {
    return simBeg2(a) < simBeg2(b);
}

bool isOverlapping(const std::vector<SegmentPair> &alignment1,
                   const std::vector<SegmentPair> &alignment2) {
    for (const auto &i : alignment1) {
        for (const auto &j : alignment2) {
            if (i.start1 - i.start2 == j.start1 - j.start2 && i.start1 + i.length > j.start1 &&
                i.start1 < j.start1 + j.length)
                return true;
        }
    }
    return false;
}

void setCharToNumber(char *charToNumber, const char *alphabet) {
    for (int i = 0; alphabet[i]; ++i) {
        int c = alphabet[i];
        charToNumber[toupper(c)] = charToNumber[tolower(c)] = i;
    }
}

std::unordered_map<char, std::vector<std::string>> aa2codons;
std::unordered_map<char, std::vector<std::string>> &build_standard_genetic_code() {
    if (aa2codons.size() > 0) {
        return aa2codons;
    }

    aa2codons['A'] = {"GCT", "GCC", "GCA", "GCG"};               // Ala
    aa2codons['R'] = {"CGT", "CGC", "CGA", "CGG", "AGA", "AGG"}; // Arg
    aa2codons['N'] = {"AAT", "AAC"};                             // Asn
    aa2codons['D'] = {"GAT", "GAC"};                             // Asp
    aa2codons['C'] = aa2codons['U'] = {"TGT", "TGC"};            // Cys
    aa2codons['Q'] = {"CAA", "CAG"};                             // Gln
    aa2codons['E'] = {"GAA", "GAG"};                             // Glu
    aa2codons['G'] = {"GGT", "GGC", "GGA", "GGG"};               // Gly
    aa2codons['H'] = {"CAT", "CAC"};                             // His
    aa2codons['I'] = {"ATT", "ATC", "ATA"};                      // Ile
    aa2codons['L'] = {"TTA", "TTG", "CTT", "CTC", "CTA", "CTG"}; // Leu
    aa2codons['K'] = aa2codons['O'] = {"AAA", "AAG"};            // Lys
    aa2codons['M'] = {"ATG"};                                    // Met
    aa2codons['F'] = {"TTT", "TTC"};                             // Phe
    aa2codons['P'] = {"CCT", "CCC", "CCA", "CCG"};               // Pro
    aa2codons['S'] = {"TCT", "TCC", "TCA", "TCG", "AGT", "AGC"}; // Ser
    aa2codons['T'] = {"ACT", "ACC", "ACA", "ACG"};               // Thr
    aa2codons['W'] = {"TGG"};                                    // Trp
    aa2codons['Y'] = {"TAT", "TAC"};                             // Tyr
    aa2codons['V'] = {"GTT", "GTC", "GTA", "GTG"};               // Val
    aa2codons['*'] = {"TAA", "TAG", "TGA"};
    aa2codons['?'] = {"???"}; // masked

    return aa2codons;
}

// Fast codon translation via flat lookup table (no heap alloc, no hashing)
// Build a flat 32768-entry codon table indexed by packed 5-bit-per-base key
static char codonTableFlat[32768];
static bool codonTableBuilt = false;

static void buildCodonTable() {
    if (codonTableBuilt) return;
    memset(codonTableFlat, '?', sizeof(codonTableFlat));
    static const char *codons[] = {
        "TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG",
        "ATT", "ATC", "ATA", "ATG", "GTT", "GTC", "GTA", "GTG",
        "TCT", "TCC", "TCA", "TCG", "CCT", "CCC", "CCA", "CCG",
        "ACT", "ACC", "ACA", "ACG", "GCT", "GCC", "GCA", "GCG",
        "TAT", "TAC", "TAA", "TAG", "CAT", "CAC", "CAA", "CAG",
        "AAT", "AAC", "AAA", "AAG", "GAT", "GAC", "GAA", "GAG",
        "TGT", "TGC", "TGA", "TGG", "CGT", "CGC", "CGA", "CGG",
        "AGT", "AGC", "AGA", "AGG", "GGT", "GGC", "GGA", "GGG"
    };

    static const char aaMap[] = {
        'F','F','L','L','L','L','L','L','I','I','I','M','V','V','V','V',
        'S','S','S','S','P','P','P','P','T','T','T','T','A','A','A','A',
        'Y','Y','*','*','H','H','Q','Q','N','N','K','K','D','D','E','E',
        'C','C','*','W','R','R','R','R','S','S','R','R','G','G','G','G'
    };
    for (int i = 0; i < 64; i++) {
        unsigned key = ((unsigned)(unsigned char)codons[i][0] & 0x1f)
                     | (((unsigned)(unsigned char)codons[i][1] & 0x1f) << 5)
                     | (((unsigned)(unsigned char)codons[i][2] & 0x1f) << 10);
        codonTableFlat[key] = aaMap[i];
    }
    // Also handle lowercase
    for (int i = 0; i < 64; i++) {
        char lc[3] = { (char)(codons[i][0] | 0x20), (char)(codons[i][1] | 0x20), (char)(codons[i][2] | 0x20) };
        unsigned key = ((unsigned)(unsigned char)lc[0] & 0x1f)
                     | (((unsigned)(unsigned char)lc[1] & 0x1f) << 5)
                     | (((unsigned)(unsigned char)lc[2] & 0x1f) << 10);
        codonTableFlat[key] = aaMap[i];
    }
    codonTableBuilt = true;
}

inline char translateFast(const char *dna, int i) {
    if (dna[i] == '?' || dna[i + 1] == '?' || dna[i + 2] == '?') return '?';

    unsigned key = ((unsigned)(unsigned char)dna[i] & 0x1f)
                 | (((unsigned)(unsigned char)dna[i+1] & 0x1f) << 5)
                 | (((unsigned)(unsigned char)dna[i+2] & 0x1f) << 10);
    return codonTableFlat[key];
}

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

// Single-pass decode: decompress + translate + charToNumber in one loop
std::vector<uint8_t> decodeSequence(const char *sequence, int sequenceLength, const char *alphabet,
                                    const char *charToNumber) {
    buildCodonTable();
    int n = sequenceLength;

    // Decompress sequence in-place to a stack buffer for the codon window
    // We only need a sliding window of 3 decompressed chars at a time
    std::vector<uint8_t> decoded(n, (uint8_t)INT_MIN);

    // Pre-decompress into a flat buffer (avoids per-char string append)
    // Use a local buffer instead of std::string for cache efficiency
    std::vector<char> decompressed(n);
    for (int i = 0; i < n; i++) {
        assert(sequence[i] <= 22);
        decompressed[i] = alphabet[(unsigned char)sequence[i]];
    }

    // Single pass: translate codons and map to numbers
    const char *dec = decompressed.data();
    for (int i = 0; i < n - 2; i++) {
        decoded[i] = charToNumber[(unsigned char)translateFast(dec, i)];
    }
    return decoded;
}
template <typename MatrixType>
simd_t simdGather(const MatrixType* matrices, Float* buf, int activeCount, int i_row, int col) {
    for (int k = 0; k < activeCount; k++)
        buf[k] = matrices[k].get(i_row, col);
    for (int k = activeCount; k < simdWidth; k++)
        buf[k] = Float(0);
    return Kokkos::Experimental::simd_unchecked_load<simd_t>(buf);
}

template <typename MatrixType>
void simdScatter(simd_t val, MatrixType* matrices, Float* buf, int activeCount, int i_row, int col) {
    simd_unchecked_store(val, buf, Kokkos::Experimental::simd_flag_default);
    for (int k = 0; k < activeCount; k++)
        matrices[k].set(i_row, col, buf[k]);
}

uint8_t* buildTransposedDecoded(DPScratch& scratch, int active_dp_width, int activeCount,
                                 int zero_idx, const std::array<std::vector<uint8_t>*, simdWidth>& decoded) {
    scratch.transposed_decoded.assign((active_dp_width + 8) * simdWidth, zero_idx);
    uint8_t* transposed_base = scratch.transposed_decoded.data() + 4 * simdWidth;
    for (int idx = 0; idx < activeCount; idx++) {
        for (size_t j = 0; j < decoded[idx]->size(); j++) {
            transposed_base[j * simdWidth + idx] = (*decoded[idx])[j];
        }
    }
    return transposed_base;
}

void findSimilarities(std::array<std::vector<AlignedSimilarity>, simdWidth> &similarities, const Profile &profile,
             const std::array<std::vector<uint8_t>*, simdWidth> &decoded, std::array<Float, simdWidth> minProbRatio,
             DPScratch &scratch, int activeCount, simd_t *bwdBestOut = nullptr) {
    assert(0 < activeCount && activeCount <= simdWidth);

    int maxSequenceLength = 0;
    for (int idx = 0; idx < activeCount; idx++) {
        maxSequenceLength = std::max(maxSequenceLength, (int)decoded[idx]->size());
    }

    scratch.active_dp_width = maxSequenceLength;

    int alphabetSize = profile.width - nonLetterWidth;
    int zero_idx = alphabetSize + 4; // new padded index that maps to 0.0

    simd_t actual_seq_len;
    simd_t null_prob_per_pos;
    simd_t null_emit_1, null_emit_2, null_emit_3;
    uint8_t* transposed_base;
    auto &null_probs_prefix = scratch.null_probs_prefix;
    auto &null_probs_suffix = scratch.null_probs_suffix;

    transposed_base = buildTransposedDecoded(scratch, scratch.active_dp_width, activeCount, zero_idx, decoded);

    alignas(64) Float actual_sequence_length[simdWidth] = {};
    for (int idx = 0; idx < activeCount; idx++) {
        actual_sequence_length[idx] = (Float)decoded[idx]->size();
    }
    actual_seq_len = Kokkos::Experimental::simd_unchecked_load<simd_t>(actual_sequence_length);

    null_probs_prefix.resize(scratch.active_dp_width + 4); null_probs_suffix.resize(scratch.active_dp_width + 4);

    null_probs_suffix[scratch.active_dp_width] = 0;
    for (int i = scratch.active_dp_width - 1; i >= 0; i--) {
        const Float *bg_probs_ptr = profile.log2_bg_probs.data() + 4;
        const char* indices = (const char*)&transposed_base[i * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_probs_ptr, indices);
        simd_t bg_codon_emit_probs(bg_raw);

        Kokkos::Experimental::simd_mask<Float> msk = i + 2 < actual_seq_len;
        Kokkos::Experimental::simd_mask<Float> msk_fs2 = i + 1 < actual_seq_len;
        Kokkos::Experimental::simd_mask<Float> msk_fs1 = i < actual_seq_len;
        simd_t full_codon = (Float)log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) + bg_codon_emit_probs + null_probs_suffix[i + 3];
        simd_t partial_codon = (Float)log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) + (Float)log2(0.25) * (actual_seq_len - (Float)i);
        simd_t t1 = Kokkos::Experimental::condition(msk, full_codon, partial_codon);

        simd_t fs1 = (Float)log2(BACKGROUND_FRAMESHIFT_RATE * 0.25) + null_probs_suffix[i + 1];
        simd_t fs2 = (Float)log2(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) + null_probs_suffix[i + 2];
        simd_t neg_inf_vec((Float)-INFINITY);
        simd_t t_fs1 = Kokkos::Experimental::condition(msk_fs1, fs1, neg_inf_vec);
        simd_t t_fs2 = Kokkos::Experimental::condition(msk_fs2, fs2, neg_inf_vec);

        null_probs_suffix[i] = log2_sum_exp(t1, log2_sum_exp(t_fs1, t_fs2));
    }
    null_probs_prefix[scratch.active_dp_width] = 0;
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

    for (int i = 0; i < scratch.active_dp_width; i++) {
        // Vectorized bg emission lookup via transposed_base
        const char* indices = (const char*)&transposed_base[(i - 2) * simdWidth];
        SimdFloat bg_raw = simdLookup(log2_bg_probs_ptr, indices);
        simd_t bg_codon_emit_probs(bg_raw);

        // Mask for variable-length sequences (per-lane)
        Kokkos::Experimental::simd_mask<Float> msk_valid = (Float)i < actual_seq_len;

        // t1: codon branch — i is scalar, so use plain if/else
        simd_t t1;
        if (i >= 3) {
            t1 = simd_log2_1_bg_fs + bg_codon_emit_probs + null_probs_prefix[i - 3];
        } else if (i == 2) {
            t1 = simd_log2_1_bg_fs + bg_codon_emit_probs;
        } else {
            t1 = simd_log2_1_bg_fs + simd_log2_025 * (Float)(i + 1);
        }

        // t2: 1-bp frameshift branch
        simd_t t2 = simd_log2_bg_fs_025 + (i > 0 ? null_probs_prefix[i - 1] : simd_t(0));

        // t3: 2-bp frameshift branch
        simd_t t3;
        if (i >= 2) {
            t3 = simd_log2_bg_fs2_00625 + null_probs_prefix[i - 2];
        } else if (i == 1) {
            t3 = simd_log2_bg_fs2_00625 + simd_t(0);
        } else {
            t3 = simd_neg_inf;
        }

        // log2_sum_exp and mask out-of-bounds lanes
        simd_t result = log2_sum_exp(t1, log2_sum_exp(t2, t3));
        null_probs_prefix[i] = Kokkos::Experimental::condition(msk_valid, result, simd_neg_inf);
    }

    alignas(64) Float null_emit_tmp[simdWidth] = {0}, null_seq_log_prob[simdWidth] = {0};

    for (int idx = 0; idx < activeCount; idx++) {
        if (actual_sequence_length[idx] >= 3) {
            null_seq_log_prob[idx] =
            log2_sum_exp(log2_sum_exp(null_probs_prefix[actual_sequence_length[idx] - 1][idx],null_probs_prefix[actual_sequence_length[idx] - 2][idx]),
                null_probs_prefix[actual_sequence_length[idx] - 3][idx]
            );
        } else {
            // For very short sequences, treat as non‑alignable (e.g., -INF or 0)
            null_seq_log_prob[idx] = -INFINITY;
        }

        Float null_emit_raw = exp2(-(null_seq_log_prob[idx] / actual_sequence_length[idx]));
        null_emit_tmp[idx] = null_emit_raw;
    }

    auto null_emit_raw_vec = Kokkos::Experimental::simd_unchecked_load<simd_t>(null_emit_tmp);
    null_emit_1 = null_emit_raw_vec;
    null_emit_2 = null_emit_1 * null_emit_1;
    null_emit_3 = null_emit_2 * null_emit_1;

    null_emit_2 *= (Float)(0.25 * 0.25);
    null_emit_1 *= (Float)0.25;

    auto null_seq_log_prob_simd = Kokkos::Experimental::simd_unchecked_load<simd_t>(null_seq_log_prob);
    simd_t inv_actual_seq_len = (Float)1.0 / actual_seq_len;
    null_prob_per_pos = null_seq_log_prob_simd * inv_actual_seq_len;

    scratch.W0_curr.resize(scratch.active_dp_width + 4, simd_t(0.0));
    scratch.W0_next.resize(scratch.active_dp_width + 4, simd_t(0.0));

    scratch.W1.assign(profile.length + 2, scratch.active_dp_width + 4);

    const size_t padded_seq_len = scratch.active_dp_width + 4;
    auto &Y0_next = scratch.Y0_next; Y0_next.resize(padded_seq_len, 0.0);
    auto &Y0_curr = scratch.Y0_curr; Y0_curr.resize(padded_seq_len, 0.0);

    auto &null_model_prefix = scratch.null_model_prefix; null_model_prefix.resize(padded_seq_len, 0.0);

    for (int j = 0; j < scratch.active_dp_width; j++) {
        simd_t exponent = -null_prob_per_pos * (actual_seq_len - (Float)1.0 - (Float)j) + null_probs_suffix[j + 1];
        null_model_prefix[j] = Kokkos::exp2(exponent);
    }
    auto &null_model_suffix = scratch.null_model_suffix; null_model_suffix = null_model_prefix;
    auto &left_side = scratch.left_side;
    auto &right_side = scratch.right_side;
    auto &left_side_EV = scratch.left_side_EV;
    auto &right_side_EV = scratch.right_side_EV;
#ifdef ALIGN
    left_side.assign(null_model_prefix.size(), 0.0);
    right_side.assign(null_model_prefix.size(), 0.0);
    left_side_EV.assign(null_model_prefix.size(), 0.0);
    right_side_EV.assign(null_model_prefix.size(), 0.0);
    scratch.X.assign(profile.length + 2, scratch.active_dp_width + 4);
    scratch.X_pfx.assign(profile.length + 2, scratch.active_dp_width + 4);
#endif
        const Float *bg_probs_ptr = profile.bg_probs.data() + 4;
        scratch.bg_codon_probs.resize(scratch.active_dp_width);
        simd_t* bg_codon_probs_base = scratch.bg_codon_probs.data();
        for (int j = -4; j < scratch.active_dp_width + 4; j++) {
            const char* indices = (const char*)&transposed_base[j * simdWidth];
            bg_codon_probs_base[j] = simd_t(simdLookup(bg_probs_ptr, indices));
        }

        {
        auto gather_w1 = [&](int i_row, int col) -> simd_t {
            return scratch.W1.get(i_row, col);
        };
        auto scatter_w1 = [&](simd_t val, int i_row, int col) {
            scratch.W1.set(i_row, col, val);
        };

        for (int i = profile.length; i >= 0; i--) {
            const Params &params_cur = profile.values_v2[i];
            const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

            const simd_t C_enter = params_cur.enter_match_probability * null_emit_3;
            const simd_t C_delta0 = params_cur.delta_prime[0];
            const simd_t C_delta1 = params_cur.delta_prime[1] * null_emit_2;
            const simd_t C_delta2 = params_cur.delta_prime[2] * null_emit_1;
            const simd_t C_alpha0 = params_cur.alpha_prime[0] * null_emit_3;
            const simd_t C_alpha1 = params_cur.alpha_prime[1] * null_emit_1;
            const simd_t C_alpha2 = params_cur.alpha_prime[2] * null_emit_2;
            const simd_t C_beta0 = params_cur.beta_prime[0] * null_emit_3;
            simd_t Z0_ring[4] = {0, 0, 0, 0};

            const simd_t C_eps0 = params_cur.epsilon_prime;
            const simd_t C_scale = scale;

            int rev_lo = 0, rev_hi = scratch.active_dp_width - 1;

            // Shift registers to eliminate redundant W1 gathers.
            // Between consecutive j iterations, 3/5 W1 reads overlap:
            //   iter j:   reads (i+1,j+1),(i+1,j+2),(i+1,j+3),(i,j+1),(i,j+2)
            //   iter j-1: reads (i+1,j),  (i+1,j+1),(i+1,j+2),(i,j),  (i,j+1)
            // We keep the last 5 columns as shift registers and only gather
            // the one truly new value W1[i+1][j] each iteration.
            // W1[i][j] is w_val itself — already in register, no gather needed.
            simd_t sr_n3 = gather_w1(i + 1, rev_hi + 3);
            simd_t sr_n2 = gather_w1(i + 1, rev_hi + 2);
            simd_t sr_n1 = gather_w1(i + 1, rev_hi + 1);
            simd_t sr_c2 = gather_w1(i, rev_hi + 2);
            simd_t sr_c1 = gather_w1(i, rev_hi + 1);

            for (int j = rev_hi; j >= rev_lo; j--) {
                int r_0 = j & 3;
                int r_1 = (j + 1) & 3;
                int r_2 = (j + 2) & 3;
                int r_3 = (j + 3) & 3;

                const char* indices = (const char*)&transposed_base[(j + 1) * simdWidth];
                SimdFloat codon_raw = simdLookup(params_emission_probabilities, indices);
                simd_t codon_emit_probs(codon_raw);
                simd_t bg_codon_emit_probs = bg_codon_probs_base[j + 1];

                // Only gather the one new column needed for next iteration
                simd_t w1_next_at_j = gather_w1(i + 1, j);

                simd_t w_val =
                    sr_n3 * codon_emit_probs * C_enter +
                    Y0_next[j + 0] * C_delta0 +
                    sr_n2 * C_delta1 +
                    sr_n1 * C_delta2 +
                    Z0_ring[r_3] * bg_codon_emit_probs * C_alpha0 +
                    sr_c2 * C_alpha2 +
                    sr_c1 * C_alpha1
                 + null_model_prefix[j] * C_scale;

                scatter_w1(w_val, i, j);
#ifdef ALIGN
                right_side[j] += w_val;
#endif

                Y0_curr[j] = Kokkos::fma(C_eps0, Y0_next[j], w_val);
                simd_t z0_future = Z0_ring[r_3] * bg_codon_emit_probs;
                Z0_ring[r_0] = Kokkos::fma(C_beta0, z0_future, w_val);

                // Rotate shift registers for next iteration (j-1)
                sr_n3 = sr_n2;
                sr_n2 = sr_n1;
                sr_n1 = w1_next_at_j;
                sr_c2 = sr_c1;
                sr_c1 = w_val;
            }

            std::swap(Y0_curr, Y0_next);
        }
        }

    std::fill(Y0_next.begin(), Y0_next.begin() + padded_seq_len, simd_t(0.0));

    for (int k = 0; k < activeCount; k++) {
        int len = (int)decoded[k]->size();
        scratch.best_wMid[k].assign(len, Float(-INFINITY));
        scratch.best_wEnd[k].assign(len, Float(0.0));
        scratch.best_i[k].assign(len, -1);
    }
    size_t nphys = scratch.active_dp_width + 4;
    scratch.best_wMid_phys.assign(nphys, simd_t(-INFINITY));
    scratch.best_wEnd_phys.assign(nphys, simd_t(0.0));
    scratch.best_i_phys.assign(nphys, simd_t(0.0));
    for (int j = 0; j < scratch.active_dp_width; j++) {
        simd_t exponent = -null_prob_per_pos * (Float)(j + 1) + null_probs_prefix[j];

        null_model_prefix[j] = Kokkos::exp2(exponent);
    }

    if (bwdBestOut) {
        // Backward-accumulator best reproduces findSimilaritiesBackwardOnly's
        // global best without a second DP: W1 holds raw backward values here.
        simd_t best_mid = simd_t(0.0);
        const simd_t *__restrict__ np = null_model_prefix.data();
        for (int i = profile.length; i >= 0; i--) {
            const simd_t *__restrict__ rp = scratch.W1.row_ptr(i);
            for (int j = 0; j < scratch.active_dp_width; j++) {
                best_mid = Kokkos::max(best_mid, rp[j] * np[j]);
            }
        }
        *bwdBestOut = best_mid;
    }

#ifdef ALIGN
    // seems to be a clean way of getting expected value of null-sided junctions
    // TODO: verify logic
    for (int j = scratch.active_dp_width - 1; j >= 0; j--) {
        const char* indices = (const char*)&transposed_base[(j - 2) * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_probs_ptr, indices);
        simd_t bg_codon_emit_probs(bg_raw);

        right_side[j - 3] += (Float)(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) * bg_codon_emit_probs *
                             null_emit_3 * right_side[j];
        right_side[j - 1] += (Float)(BACKGROUND_FRAMESHIFT_RATE * 0.25) * null_emit_1 * right_side[j];
        right_side[j - 2] += (Float)(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) * null_emit_2 * right_side[j];

        right_side_EV[j - 3] += (Float)(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) * bg_codon_emit_probs *
                             null_emit_3 * right_side[j];
        right_side_EV[j - 1] += (Float)(BACKGROUND_FRAMESHIFT_RATE * 0.25) * null_emit_1 * (Float)(1.0 / 3.0) * right_side[j];
        right_side_EV[j - 2] += (Float)(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) * null_emit_2 * (Float)(2.0 / 3.0) * right_side[j];

        right_side_EV[j] *= null_model_prefix[j]; // TODO: this one specifically (might be off by 1 idk)
        right_side[j] = Kokkos::max(right_side[j], Float(0.0));
    }
    for (int j = 1; j < scratch.active_dp_width; j++) {
        right_side_EV[j] += right_side_EV[j - 1];
    }
#endif

    simd_t global_best = simd_t(0.0);
    for (int i = 0; i <= profile.length; i++) {
        const Params &params_cur = profile.values_v2[i];
        const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

        // Pre-calculate constants
        const simd_t C_enter = params_cur.enter_match_probability * null_emit_3;
        const simd_t C_alpha0 = params_cur.alpha_prime[0];
        const simd_t C_alpha1 = params_cur.alpha_prime[1] * null_emit_1;
        const simd_t C_alpha2 = params_cur.alpha_prime[2] * null_emit_2;
        const simd_t C_beta0 = params_cur.beta_prime[0];
        const simd_t C_delta0 = params_cur.delta_prime[0];
        const simd_t C_delta1 = params_cur.delta_prime[1] * null_emit_2;
        const simd_t C_delta2 = params_cur.delta_prime[2] * null_emit_1;
        const simd_t C_eps0 = params_cur.epsilon_prime;
        const simd_t C_scale = scale;

        simd_t Z0_ring[4] = {0, 0, 0, 0};
        simd_t Z1_ring[4] = {0, 0, 0, 0};
        simd_t Z2_ring[4] = {0, 0, 0, 0};

        // Raw row pointers — avoid repeated i*cols in inner loop
        simd_t *__restrict__ w0_row_i = scratch.W0_curr.data();
        simd_t *__restrict__ w0_row_ip1 = (i + 1 <= profile.length) ? scratch.W0_next.data() : nullptr;

        auto gather_w1 = [&](int i_row, int col) -> simd_t {
            return scratch.W1.get(i_row, col);
        };

        bool w1_ip1_avail = (i + 1 <= profile.length);
        auto gather_w1_ip1 = [&](int col) -> simd_t {
            if (!w1_ip1_avail) return simd_t(0);
            return gather_w1(i + 1, col);
        };
        auto gather_w1_i = [&](int col) -> simd_t {
            return gather_w1(i, col);
        };

#ifdef ALIGN
        auto gather_xpfx = [&](int i_row, int col) -> simd_t {
            return scratch.X_pfx.get(i_row, col);
        };
        auto scatter_x = [&](simd_t val, int i_row, int col) {
            scratch.X.set(i_row, col, val);
        };
        auto scatter_xpfx = [&](simd_t val, int i_row, int col) {
            scratch.X_pfx.set(i_row, col, val);
        };
#endif

        int lo = 0, hi = scratch.active_dp_width - 1;

        // Per-lane sequence lengths for variable-length masking
        alignas(64) Float seq_len_f[simdWidth] = {};
        for (int k = 0; k < activeCount; k++) {
            seq_len_f[k] = (Float)decoded[k]->size();
        }
        simd_t seq_len_vec = Kokkos::Experimental::simd_unchecked_load<simd_t>(seq_len_f);

        int seq_start = lo;

        // Shift register for w[1..3] — avoids 3 matrix reads per iteration
        simd_t w_shift[3] = {0,0, 0}; // w_shift[0]=w0(i,j-1), [1]=w0(i,j-2), [2]=w0(i,j-3)
        simd_t pfx_prev = simd_t(0.0); // X_pfx(i, j-1) rolling value

        const simd_t simd_invScale(invScale);

        for (int j = seq_start; j <= hi; j++) {

            simd_t w1 = w_shift[0], w2 = w_shift[1], w3 = w_shift[2];

            int r_0 = j & 3;
            int r_3 = (j - 3) & 3;

            const char* indices = (const char*)&transposed_base[(j - 2) * simdWidth];
            SimdFloat codon_raw = simdLookup(params_emission_probabilities, indices);

            simd_t codon_emit_probs(codon_raw);
            simd_t bg_codon_emit_probs = bg_codon_probs_base[j - 2];

            simd_t X_ij = C_enter * codon_emit_probs * w3;

#ifdef ALIGN
            {
                simd_t X_ij_EV = X_ij * gather_w1_ip1(j) * simd_invScale;
                scatter_x(X_ij_EV, i, j);

                simd_t opt_succ = (i - 1 >= 0) ? gather_xpfx(i - 1, j - 3) : simd_t(0);

                simd_t pfx_mx = (i - 1 >= 0) ? gather_xpfx(i - 1, j) : simd_t(0);
                pfx_mx = Kokkos::max(pfx_mx, pfx_prev);

                pfx_mx = Kokkos::max(pfx_mx, X_ij_EV + opt_succ);
                pfx_mx = Kokkos::max(pfx_mx, right_side_EV[j]);
                scatter_xpfx(pfx_mx, i, j);
                pfx_prev = pfx_mx;
            }
#endif
            Z0_ring[r_0] =
                bg_codon_emit_probs * null_emit_3 *
                (C_alpha0 * w3 + C_beta0 * Z0_ring[r_3]);
            Z1_ring[r_0] = C_alpha1 * w1;
            Z2_ring[r_0] = C_alpha2 * w2;

            simd_t w0 = w0_row_i[j];
            w0 += Z0_ring[r_0] + Z1_ring[r_0] + Z2_ring[r_0] + null_model_prefix[j] * C_scale;
#ifdef ALIGN
            left_side[j] += w0;
#endif

            simd_t wMid = w0 * gather_w1_i(j) * simd_invScale;

            global_best = Kokkos::max(global_best, wMid);
            // Per-lane variable-length mask (batches pack different lengths)
            simd_t j_vec((Float)j);
            simd_t is_active = Kokkos::Experimental::condition(j_vec < seq_len_vec, simd_t(1), simd_t(0));
            w0 *= is_active;
            w0_row_i[j] = w0;

            Z0_ring[r_0] *= is_active;
            Z1_ring[r_0] *= is_active;
            Z2_ring[r_0] *= is_active;

            w_shift[2] = w_shift[1];
            w_shift[1] = w_shift[0];
            w_shift[0] = w0;

            Y0_curr[j] = C_delta0 * w0 + C_eps0 * Y0_next[j];
            Y0_curr[j] *= is_active;
            if (w0_row_ip1)
                w0_row_ip1[j] += X_ij + Y0_curr[j] + C_delta1 * w2 + C_delta2 * w1;

            // Per-lane anchor tracking — branchless SIMD update per DP column
            {
                auto better_mask = wMid > scratch.best_wMid_phys[j];
                if (Kokkos::Experimental::any_of(better_mask)) {
                    scratch.best_wMid_phys[j] = Kokkos::max(scratch.best_wMid_phys[j], wMid);
                    scratch.best_wEnd_phys[j] = Kokkos::Experimental::condition(better_mask, w0, scratch.best_wEnd_phys[j]);
                    scratch.best_i_phys[j] = Kokkos::Experimental::condition(better_mask, simd_t((Float)i), scratch.best_i_phys[j]);
                }
            }
        }

        std::swap(Y0_curr, Y0_next);
        std::fill(Y0_curr.begin(), Y0_curr.end(), simd_t(0.0));

        std::swap(scratch.W0_curr, scratch.W0_next);
        std::fill(scratch.W0_next.begin(), scratch.W0_next.end(), simd_t(0.0));
    }

    {
        alignas(64) Float fwd_best_arr[simdWidth];
        simd_unchecked_store(global_best, fwd_best_arr, Kokkos::Experimental::simd_flag_default);

        bool any_fwd_above = false;
        for (int k = 0; k < activeCount; k++) {
            if (minProbRatio[k] < 0 || fwd_best_arr[k] >= minProbRatio[k]) {
                any_fwd_above = true;
                break;
            }
        }
        if (!any_fwd_above) {
            return;
        }
    }

#ifdef ALIGN
    for (int j = 0; j < scratch.active_dp_width; j++) {
        const char* indices = (const char*)&transposed_base[(j + 1) * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_probs_ptr, indices);

        simd_t bg_codon_emit_probs(bg_raw);

        left_side[j + 3] += (Float)(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) * bg_codon_emit_probs * null_emit_3 * left_side[j];
        left_side[j + 1] += (Float)(BACKGROUND_FRAMESHIFT_RATE * 0.25) * null_emit_1 * left_side[j];
        left_side[j + 2] += (Float)(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) * null_emit_2 * left_side[j];

        left_side_EV[j + 3] += (Float)(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) * bg_codon_emit_probs * null_emit_3 * left_side[j];
        left_side_EV[j + 1] += (Float)(BACKGROUND_FRAMESHIFT_RATE * 0.25) * null_emit_1 * (Float)(1.0 / 3.0) * left_side[j];
        left_side_EV[j + 2] += (Float)(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) * null_emit_2 * (Float)(2.0 / 3.0) * left_side[j];

        left_side_EV[j] *= null_model_suffix[j];
        left_side[j] = Kokkos::max(left_side[j], Float(0.0));
    }
    for (int j = scratch.active_dp_width - 2; j >= 0; j--) {
        left_side_EV[j] += left_side_EV[j + 1];
    }

    {
    auto gather_w1 = [&](int i_row, int col) -> simd_t {
        return scratch.W1.get(i_row, col);
    };
    auto gather_x = [&](int i_row, int col) -> simd_t {
        return scratch.X.get(i_row, col);
    };
    auto scatter_w1 = [&](simd_t val, int i_row, int col) {
        scratch.W1.set(i_row, col, val);
    };

    for (int i = profile.length; i >= 0; i--) {
        simd_t opt_right_rolling = simd_t(0.0);

        bool ip1_avail = (i + 1 <= profile.length);
        for (int j = scratch.active_dp_width - 1; j >= 0; j--) {
            simd_t opt_succ = ip1_avail ? gather_w1(i + 1, j + 3) : simd_t(0);
            simd_t opt_down = ip1_avail ? gather_w1(i + 1, j) : simd_t(0);

            auto opt = Kokkos::max(opt_down, opt_right_rolling);
            opt = Kokkos::max(opt, left_side_EV[j]);
            opt = Kokkos::max(opt, gather_x(i, j) + opt_succ);
            scatter_w1(opt, i, j);
            opt_right_rolling = opt;
        }
    }
    }
#endif

    // Extract per-lane anchor trackers from the physical SIMD layout.
    for (int idx = 0; idx < activeCount; idx++) {
        int len = (int)decoded[idx]->size();
        for (int j = 0; j < scratch.active_dp_width && j < len; j++) {
            Float wMid_k = scratch.best_wMid_phys[j][idx];
            if (wMid_k > scratch.best_wMid[idx][j]) {
                scratch.best_wMid[idx][j] = wMid_k;
                scratch.best_wEnd[idx][j] = scratch.best_wEnd_phys[j][idx];
                scratch.best_i[idx][j] = (int)scratch.best_i_phys[j][idx];
            }
        }
    }

    for (int idx = 0; idx < activeCount; idx++) {
        int len = (int)decoded[idx]->size();

        scratch.opt_profile_position[idx].assign(len, AlignedSimilarity(-INFINITY));

        Float best_overall = -INFINITY;
        for (int j = 0; j < len; j++) {
            Float best_prob = scratch.best_wMid[idx][j];
            if (best_prob > -INFINITY) {
                best_overall = std::max(best_overall, best_prob);
                scratch.opt_profile_position[idx][j] = {
                    best_prob,
                    scratch.best_i[idx][j],
                    j,
                    (Float)scratch.best_wEnd[idx][j]
                };
            }
        }

        if (minProbRatio[idx] >= 0) {
            if (best_overall >= minProbRatio[idx]) {
                if (verbosity > 0)
                    std::cerr << "has" << std::endl;
                std::ranges::sort(scratch.opt_profile_position[idx], std::greater<>());
                auto &aligned = scratch.aligned;
                aligned[idx].assign(decoded[idx]->size() + 0, false);

                for (auto &aligned_similarity : scratch.opt_profile_position[idx]) {
                    int logical_j = aligned_similarity.anchor2;

                    if (aligned_similarity.probRatio >= minProbRatio[idx] &&
                        !aligned[idx][logical_j]) {

                        addMidAnchored(idx, profile.length, similarities[idx], aligned_similarity.anchor1,
                                       aligned_similarity.anchor2,
                                       aligned_similarity.probRatio * scale /
                                           aligned_similarity.wEndAnchored,
                                       aligned_similarity.wEndAnchored, scratch);
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
            auto sel = *std::max_element(scratch.opt_profile_position[idx].begin(), scratch.opt_profile_position[idx].end());
            AlignedSimilarity b = sel;
            b.probRatio = 0;
            similarities[idx].push_back(b);
            b.probRatio = 0;
            similarities[idx].push_back(b);
            similarities[idx].push_back(sel);
        }
    }
}

#ifdef FORWARD_ONLY_FILTER

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
    scratch.active_dp_width = maxSequenceLength;

    int alphabetSize = profile.width - nonLetterWidth;
    int zero_idx = alphabetSize + 4;

    uint8_t* transposed_base = buildTransposedDecoded(scratch, scratch.active_dp_width, activeCount, zero_idx, decoded);

    alignas(64) Float actual_sequence_length[simdWidth] = {};
    for (int idx = 0; idx < activeCount; idx++)
        actual_sequence_length[idx] = (Float)decoded[idx]->size();
    simd_t actual_seq_len = Kokkos::Experimental::simd_unchecked_load<simd_t>(actual_sequence_length);

    auto &null_probs_prefix = scratch.null_probs_prefix;
    auto &null_probs_suffix = scratch.null_probs_suffix;
    null_probs_prefix.resize(scratch.active_dp_width + 4);
    null_probs_suffix.resize(scratch.active_dp_width + 4);

    null_probs_suffix[scratch.active_dp_width] = 0;
    for (int i = scratch.active_dp_width - 1; i >= 0; i--) {
        const Float *bg_probs_ptr = profile.log2_bg_probs.data() + 4;
        const char* indices = (const char*)&transposed_base[i * simdWidth];
        SimdFloat bg_raw = simdLookup(bg_probs_ptr, indices);
        simd_t bg_codon_emit_probs(bg_raw);

        Kokkos::Experimental::simd_mask<Float> msk = i + 2 < actual_seq_len;
        Kokkos::Experimental::simd_mask<Float> msk_fs2 = i + 1 < actual_seq_len;
        Kokkos::Experimental::simd_mask<Float> msk_fs1 = i < actual_seq_len;
        simd_t full_codon = (Float)log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) + bg_codon_emit_probs + null_probs_suffix[i + 3];
        simd_t partial_codon = (Float)log2(1 - BACKGROUND_FRAMESHIFT_RATE - BACKGROUND_FRAMESHIFT_RATE_2) + (Float)log2(0.25) * (actual_seq_len - (Float)i);
        simd_t t1 = Kokkos::Experimental::condition(msk, full_codon, partial_codon);

        simd_t fs1 = (Float)log2(BACKGROUND_FRAMESHIFT_RATE * 0.25) + null_probs_suffix[i + 1];
        simd_t fs2 = (Float)log2(BACKGROUND_FRAMESHIFT_RATE_2 * 0.0625) + null_probs_suffix[i + 2];
        simd_t neg_inf_vec((Float)-INFINITY);
        simd_t t_fs1 = Kokkos::Experimental::condition(msk_fs1, fs1, neg_inf_vec);
        simd_t t_fs2 = Kokkos::Experimental::condition(msk_fs2, fs2, neg_inf_vec);

        null_probs_suffix[i] = log2_sum_exp(t1, log2_sum_exp(t_fs1, t_fs2));
    }

    null_probs_prefix[scratch.active_dp_width] = 0;
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

    for (int i = 0; i < scratch.active_dp_width; i++) {
        const char* indices = (const char*)&transposed_base[(i - 2) * simdWidth];
        SimdFloat bg_raw = simdLookup(log2_bg_probs_ptr, indices);
        simd_t bg_codon_emit_probs(bg_raw);

        Kokkos::Experimental::simd_mask<Float> msk_valid = (Float)i < actual_seq_len;

        simd_t t1;
        if (i >= 3) {
            t1 = simd_log2_1_bg_fs + bg_codon_emit_probs + null_probs_prefix[i - 3];
        } else if (i == 2) {
            t1 = simd_log2_1_bg_fs + bg_codon_emit_probs;
        } else {
            t1 = simd_log2_1_bg_fs + simd_log2_025 * (Float)(i + 1);
        }

        simd_t t2 = simd_log2_bg_fs_025 + (i > 0 ? null_probs_prefix[i - 1] : simd_t(0));

        simd_t t3;
        if (i >= 2) {
            t3 = simd_log2_bg_fs2_00625 + null_probs_prefix[i - 2];
        } else if (i == 1) {
            t3 = simd_log2_bg_fs2_00625 + simd_t(0);
        } else {
            t3 = simd_neg_inf;
        }

        simd_t result = log2_sum_exp(t1, log2_sum_exp(t2, t3));
        null_probs_prefix[i] = Kokkos::Experimental::condition(msk_valid, result, simd_neg_inf);
    }

    alignas(64) Float null_emit_tmp[simdWidth] = {0}, null_seq_log_prob[simdWidth] = {0};

    for (int idx = 0; idx < activeCount; idx++) {
        if (actual_sequence_length[idx] >= 3) {
            null_seq_log_prob[idx] =
            log2_sum_exp(log2_sum_exp(null_probs_prefix[actual_sequence_length[idx] - 1][idx], null_probs_prefix[actual_sequence_length[idx] - 2][idx]),
                null_probs_prefix[actual_sequence_length[idx] - 3][idx]
            );
        } else {
            null_seq_log_prob[idx] = -INFINITY;
        }

        Float null_emit_raw = exp2(-(null_seq_log_prob[idx] / actual_sequence_length[idx]));
        null_emit_tmp[idx] = null_emit_raw;
    }

    auto null_emit_1 = Kokkos::Experimental::simd_unchecked_load<simd_t>(null_emit_tmp);
    auto null_emit_2 = null_emit_1 * null_emit_1;
    auto null_emit_3 = null_emit_2 * null_emit_1;

    null_emit_2 *= (Float)(0.25 * 0.25);
    null_emit_1 *= (Float)0.25;

    auto null_seq_log_prob_simd = Kokkos::Experimental::simd_unchecked_load<simd_t>(null_seq_log_prob);
    simd_t inv_actual_seq_len = (Float)1.0 / actual_seq_len;
    simd_t null_prob_per_pos = null_seq_log_prob_simd * inv_actual_seq_len;

    const size_t padded_seq_len = scratch.active_dp_width + 4;
    scratch.W1_rolling[0].resize(padded_seq_len, simd_t(0.0));
    scratch.W1_rolling[1].resize(padded_seq_len, simd_t(0.0));
    auto &Y0_next = scratch.Y0_next; Y0_next.resize(padded_seq_len, 0.0);
    auto &Y0_curr = scratch.Y0_curr; Y0_curr.resize(padded_seq_len, 0.0);

    auto &null_model_prefix = scratch.null_model_prefix; null_model_prefix.resize(padded_seq_len, 0.0);
    auto &null_model_suffix = scratch.null_model_suffix; null_model_suffix.resize(padded_seq_len, 0.0);

    for (int j = 0; j < scratch.active_dp_width; j++) {
        simd_t suffix_exponent = -null_prob_per_pos * (actual_seq_len - (Float)1.0 - (Float)j) + null_probs_suffix[j + 1];
        null_model_suffix[j] = Kokkos::exp2(suffix_exponent);

        simd_t prefix_exponent = -null_prob_per_pos * (Float)(j + 1) + null_probs_prefix[j];
        null_model_prefix[j] = Kokkos::exp2(prefix_exponent);
    }

    const Float *bg_probs_ptr = profile.bg_probs.data() + 4;
    scratch.bg_codon_probs.resize(scratch.active_dp_width);
    simd_t* bg_codon_probs_base = scratch.bg_codon_probs.data();
    for (int j = -4; j < scratch.active_dp_width + 4; j++) {
        const char* indices = (const char*)&transposed_base[j * simdWidth];
        bg_codon_probs_base[j] = simd_t(simdLookup(bg_probs_ptr, indices));
    }

    simd_t global_best = simd_t(0.0);

    for (int i = profile.length; i >= 0; i--) {
        const Params &params_cur = profile.values_v2[i];
        const Float *params_emission_probabilities = profile.values + (i)*profile.width + 4;

        const simd_t C_enter = params_cur.enter_match_probability * null_emit_3;
        const simd_t C_delta0 = params_cur.delta_prime[0];
        const simd_t C_delta1 = params_cur.delta_prime[1] * null_emit_2;
        const simd_t C_delta2 = params_cur.delta_prime[2] * null_emit_1;
        const simd_t C_alpha0 = params_cur.alpha_prime[0] * null_emit_3;
        const simd_t C_alpha1 = params_cur.alpha_prime[1] * null_emit_1;
        const simd_t C_alpha2 = params_cur.alpha_prime[2] * null_emit_2;
        const simd_t C_beta0 = params_cur.beta_prime[0] * null_emit_3;
        simd_t Z0_ring[4] = {0, 0, 0, 0};
        const simd_t C_eps0 = params_cur.epsilon_prime;
        const simd_t C_scale = scale;

        simd_t sr_n3 = scratch.W1_rolling[(i + 1) & 1][scratch.active_dp_width + 3];
        simd_t sr_n2 = scratch.W1_rolling[(i + 1) & 1][scratch.active_dp_width + 2];
        simd_t sr_n1 = scratch.W1_rolling[(i + 1) & 1][scratch.active_dp_width + 1];
        simd_t sr_c2 = scratch.W1_rolling[i & 1][scratch.active_dp_width + 2];
        simd_t sr_c1 = scratch.W1_rolling[i & 1][scratch.active_dp_width + 1];

        for (int j = scratch.active_dp_width - 1; j >= 0; j--) {
            int r_0 = j & 3;
            int r_3 = (j + 3) & 3;

            const char* indices = (const char*)&transposed_base[(j + 1) * simdWidth];
            SimdFloat codon_raw = simdLookup(params_emission_probabilities, indices);
            simd_t codon_emit_probs(codon_raw);
            simd_t bg_codon_emit_probs = bg_codon_probs_base[j + 1];

            simd_t w1_next_at_j = scratch.W1_rolling[(i + 1) & 1][j];

            simd_t w_val =
                sr_n3 * codon_emit_probs * C_enter +
                Y0_next[j + 0] * C_delta0 +
                sr_n2 * C_delta1 +
                sr_n1 * C_delta2 +
                Z0_ring[r_3] * bg_codon_emit_probs * C_alpha0 +
                sr_c2 * C_alpha2 +
                sr_c1 * C_alpha1
             + null_model_suffix[j] * C_scale;

            simd_t mid_score = w_val * null_model_prefix[j];
            global_best = Kokkos::max(global_best, mid_score);

            scratch.W1_rolling[i & 1][j] = w_val;

            Y0_curr[j] = Kokkos::fma(C_eps0, Y0_next[j], w_val);
            simd_t z0_future = Z0_ring[r_3] * bg_codon_emit_probs;
            Z0_ring[r_0] = Kokkos::fma(C_beta0, z0_future, w_val);

            sr_n3 = sr_n2;
            sr_n2 = sr_n1;
            sr_n1 = w1_next_at_j;
            sr_c2 = sr_c1;
            sr_c1 = w_val;
        }

        std::swap(Y0_curr, Y0_next);
    }

    alignas(64) Float best_arr[simdWidth];
    simd_unchecked_store(global_best, best_arr, Kokkos::Experimental::simd_flag_default);
    for (int k = 0; k < activeCount; k++)
        bestScores[k] = best_arr[k];
    for (int k = activeCount; k < simdWidth; k++)
        bestScores[k] = 0;
}

#endif // FORWARD_ONLY_FILTER

int contigToSequencePos(Contig contig, size_t strandNum, int posInContig) {
    return contig.start + strandPosition(strandNum, contig.length, posInContig);
}

void findFinalSimilarities(std::vector<FinalSimilarity> &similarities, std::array<SequenceRequest, simdWidth> &req,
                           const Profile &profile, size_t profileNum, const char *charVec,
                           DPScratch &scratch, int activeCount) {
    const char *alphabet = getAlphabet(profile.width - nonLetterWidth);
    const char *profileSeq = charVec + profile.consensusSequenceIdx;

    std::array<std::vector<AlignedSimilarity>, simdWidth> sims;
    std::array<std::vector<uint8_t>*, simdWidth> decoded;
    std::array<Float, simdWidth> minProbRatio;
    for (int i = 0; i < activeCount; i++) {
        decoded[i] = &req[i].seqData->decoded;
        minProbRatio[i] = req[i].minProbRatio;
    }
    findSimilarities(sims, profile, decoded, minProbRatio, scratch, activeCount);

    for (int idx = 0; idx < activeCount; idx++) {
        const char *sequence = req[idx].seqData->sequence.c_str();
        const char *maskedSequence = req[idx].seqData->maskedSequence.c_str();
        for (const auto &x : sims[idx]) {
            if (x.alignment.empty()) {
                continue;  // anchor passed the score threshold but traceback
                           // emitted no segments (boundary leak, footprint-edge
                           // remnant, weak-ridge anchor); a zero-column block
                           // is not a valid MAF record
            }
            int anchor2 = contigToSequencePos(req[idx].seqData->contig, req[idx].seqData->strandNum, x.anchor2);
            FinalSimilarity s = {x.probRatio, profileNum, req[idx].seqData->strandNum, x.anchor1,
                                 anchor2,     x.anchor1,  anchor2};
            if (!x.alignment.empty()) {
                s.start1 = x.alignment[0].start1;
                s.start2 =
                    contigToSequencePos(req[idx].seqData->contig, req[idx].seqData->strandNum, x.alignment[0].start2);
                addAlignedProfile(s.alignedSequences, x.alignment, alphabet, profileSeq);
                addAlignedSequence(s.alignedSequences, x.alignment, alphabet, sequence, maskedSequence);
            }
            similarities.push_back(s);
        }
    }

}

void findFinalSimilaritiesBatched(std::vector<FinalSimilarity> &similarities,
                                  std::vector<std::vector<SequenceRequest>> &allRequests,
                                  const std::vector<Profile> &profiles, const char *charVec,
                                  ThreadPool &threadPool, std::vector<DPScratch> &threadScratches
#ifdef FORWARD_ONLY_FILTER
                                  , double forward_only_evalue, double totSequenceLength, bool skipPrefilter
#endif
                                  ) {
#ifdef FORWARD_ONLY_FILTER
    // Forward-only pre-filter: same E-value semantics as forward-backward.
    // No enable threshold gate: it always runs unless bypassed via --max.
    std::vector<std::vector<SequenceRequest>> filteredRequests(profiles.size());
    if (!skipPrefilter) {
        struct forward_only_job {
            size_t profileIdx;
            size_t startRequestIdx;
            int activeCount;
        };
        std::vector<forward_only_job> forward_only_jobs;
        for (size_t i = 0; i < profiles.size(); ++i) {
            const auto &requests = allRequests[i];
            size_t n = requests.size();
            for (size_t start = 0; start < n; start += simdWidth) {
                int count = (int)std::min((size_t)simdWidth, n - start);
                forward_only_jobs.push_back({i, start, count});
            }
        }

        if (!forward_only_jobs.empty()) {
            std::vector<std::vector<SequenceRequest>> forward_only_job_results(forward_only_jobs.size());
            std::atomic<size_t> completed_forward_only(0);
            std::mutex forward_only_mtx;
            std::condition_variable forward_only_cv;

            for (size_t jobIdx = 0; jobIdx < forward_only_jobs.size(); ++jobIdx) {
                threadPool.enqueue([&, jobIdx](int threadId) {
                    DPScratch &ts = threadScratches[threadId];
                    const auto &job = forward_only_jobs[jobIdx];
                    const auto &p = profiles[job.profileIdx];
                    const auto &requests = allRequests[job.profileIdx];

                    std::array<std::vector<uint8_t>*, simdWidth> decodedPtrs = {};
                    for (int k = 0; k < job.activeCount; k++)
                        decodedPtrs[k] = &requests[job.startRequestIdx + k].seqData->decoded;

                    std::array<Float, simdWidth> bestScores;
                    findSimilaritiesBackwardOnly(bestScores, p, decodedPtrs, ts, job.activeCount);

                    for (int k = 0; k < job.activeCount; k++) {
                        if (bestScores[k] > 0) {
                            double log_probRatio = log((double)bestScores[k]);
                            double probRatio = exp(log_probRatio * p.lambda_forward_only);
                            double evalue = p.gumbel_k_forward_only * totSequenceLength / probRatio;

                            if (verbosity > 0) {
                                std::ostringstream s;
                                s << "# DBG " << p.name << " log: " << log_probRatio << " E: " << evalue << std::endl;

                                std::cout << s.str();
                            }

                            if (evalue <= forward_only_evalue) {
                                forward_only_job_results[jobIdx].push_back(requests[job.startRequestIdx + k]);
                            }
                        }
                    }

                    if (++completed_forward_only == forward_only_jobs.size()) {
                        std::lock_guard<std::mutex> lock(forward_only_mtx);
                        forward_only_cv.notify_one();
                    }
                });
            }

            {
                std::unique_lock<std::mutex> lock(forward_only_mtx);
                forward_only_cv.wait(lock, [&]{ return completed_forward_only == forward_only_jobs.size(); });
            }

            for (size_t jobIdx = 0; jobIdx < forward_only_jobs.size(); ++jobIdx) {
                size_t pi = forward_only_jobs[jobIdx].profileIdx;
                for (auto &req : forward_only_job_results[jobIdx])
                    filteredRequests[pi].push_back(req);
            }
        }
        // Sort filtered requests by size
        for (size_t i = 0; i < profiles.size(); ++i)
            std::sort(filteredRequests[i].begin(), filteredRequests[i].end(), std::greater<>());

        allRequests = std::move(filteredRequests);
    } else {
        for (size_t i = 0; i < profiles.size(); ++i)
            std::sort(allRequests[i].begin(), allRequests[i].end(), std::greater<>());
    }
#else
    for (size_t i = 0; i < profiles.size(); ++i)
        std::sort(allRequests[i].begin(), allRequests[i].end(), std::greater<>());
#endif

    struct BatchJob {
        size_t profileIdx;
        size_t startRequestIdx;
        int activeCount;
    };

    int batchStride = simdWidth;
    static const char* lanes_env = getenv("DUMMER_MAX_LANES");
    if (lanes_env) batchStride = std::clamp(std::atoi(lanes_env), 1, simdWidth);
    std::vector<BatchJob> jobs;
    for (size_t i = 0; i < profiles.size(); ++i) {
        const auto &requests = allRequests[i];
        size_t n = requests.size();
        for (size_t start = 0; start < n; start += batchStride) {
            int count = std::min(static_cast<size_t>(batchStride), n - start);
            jobs.push_back({i, start, count});
        }
    }

    if (jobs.empty()) return;

    std::vector<std::vector<FinalSimilarity>> jobSimilarities(jobs.size());
    std::atomic<size_t> completedJobs(0);
    std::mutex mtx;
    std::condition_variable cv;

    for (size_t jobIdx = 0; jobIdx < jobs.size(); ++jobIdx) {
        threadPool.enqueue([&, jobIdx](int threadId) {
            DPScratch &threadScratch = threadScratches[threadId];
            const auto &job = jobs[jobIdx];
            std::array<SequenceRequest, simdWidth> curBatch;
            const auto &requests = allRequests[job.profileIdx];
            for (int k = 0; k < job.activeCount; ++k) {
                curBatch[k] = requests[job.startRequestIdx + k];
            }

            findFinalSimilarities(jobSimilarities[jobIdx], curBatch,
                                  profiles[job.profileIdx], job.profileIdx,
                                  charVec, threadScratch, job.activeCount);

            if (++completedJobs == jobs.size()) {
                std::lock_guard<std::mutex> lock(mtx);
                cv.notify_one();
            }
        });
    }

    if (!jobs.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return completedJobs == jobs.size(); });
    }

    size_t totalSimilarities = 0;
    for (const auto &jobSims : jobSimilarities) {
        totalSimilarities += jobSims.size();
    }
    similarities.reserve(totalSimilarities);
    for (const auto &jobSims : jobSimilarities) {
        similarities.insert(similarities.end(), jobSims.begin(), jobSims.end());
    }
}

double methodOfMomentsLambda(const double *scores, int n, double meanScore) {
    double pi = 3.1415926535897932;
    double s = 0;
    for (int i = 0; i < n; ++i) {
        s += (scores[i] - meanScore) * (scores[i] - meanScore);
    }
    double variance = s / n; // apparently, method of moments doesn't use n-1
    return pi / sqrt(6 * variance);
}

double methodOfMomentsK(double meanScore, double lambda, double seqLength) {
    double euler = 0.57721566490153286;
    return exp(lambda * meanScore - euler) / seqLength;
}

double methodOfLmomentsLambda(const double *sortedScores, int n, double meanScore) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        s += i * sortedScores[i];
    }
    double d = 0.5 * n * (n - 1); // !!! avoids int overflow
    return log(2.0) / (s / d - meanScore);
}

double shouldBe0(const double *scores, int scoreCount, double lambda) {
    double x = 0;
    double y = 0;
    double z = 0;
    for (int i = 0; i < scoreCount; ++i) {
        x += scores[i];
        y += exp(-lambda * scores[i]);
        z += scores[i] * exp(-lambda * scores[i]);
    }
    return 1 / lambda - x / scoreCount + z / y;
}

double maximumLikelihoodLambda(const double *scores, int n) {
    double lo = 1;
    double hi = 1;
    double x, y;
    int iters = 0;
    do {
        if (++iters > 1000) return 1.0;
        lo /= 2;
        hi *= 2;
        x = shouldBe0(scores, n, lo);
        y = shouldBe0(scores, n, hi);
    } while ((x < 0 && y < 0) || (x > 0 && y > 0));
    double gap = hi - lo;
    while (1) { // bisection method to find lambda that makes shouldBe0 = 0
        gap /= 2;
        double mid = lo + gap;
        if (mid <= lo)
            return lo;
        double z = shouldBe0(scores, n, mid);
        if ((x < 0 && z <= 0) || (x > 0 && z >= 0))
            lo = mid;
    }
}

double maximumLikelihoodK(const double *scores, int n, double lambda, double seqLength) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        s += exp(-lambda * scores[i]);
    }
    return n / (s * seqLength);
}

void methodOfMomentsGumbel(double &lambda, double &k, double &kSimple, const double *scores, int n,
                           double seqLength) {
    double meanScore = mean(scores, n);
    lambda = methodOfMomentsLambda(scores, n, meanScore);
    k = methodOfMomentsK(meanScore, lambda, seqLength);
    kSimple = methodOfMomentsK(meanScore, 1, seqLength);
}

void methodOfLmomentsGumbel(double &lambda, double &k, const double *scores, int n,
                            double seqLength) {
    double meanScore = mean(scores, n);
    lambda = methodOfLmomentsLambda(scores, n, meanScore);
    k = methodOfMomentsK(meanScore, lambda, seqLength);
}

void maximumLikelihoodGumbel(double &lambda, double &k, double &kSimple, const double *scores,
                             int n, double seqLength) {
    lambda = maximumLikelihoodLambda(scores, n);
    k = maximumLikelihoodK(scores, n, lambda, seqLength);
    kSimple = maximumLikelihoodK(scores, n, 1, seqLength);
}

void estimateGumbel(double &mmLambda, double &mmK, double &mmKsimple, double &mlLambda, double &mlK,
                    double &mlKsimple, double &lmLambda, double &lmK, double *scores, int n,
                    double seqLength) {
    std::sort(scores, scores + n);
    methodOfMomentsGumbel(mmLambda, mmK, mmKsimple, scores, n, seqLength);
    maximumLikelihoodGumbel(mlLambda, mlK, mlKsimple, scores, n, seqLength);
    methodOfLmomentsGumbel(lmLambda, lmK, scores, n, seqLength);
}

static std::mutex g_cout_mutex;

class Hash128 {
public:
    Hash128() {
        XXH3_128bits_reset(state);
    }
    ~Hash128() {
        XXH3_freeState(state);
    }

    void add(const void* data, size_t size) {
        XXH3_128bits_update(state, data, size);
    }

    void add(const std::string& str) {
        add(str.data(), str.size());
    }

    template <typename T>
    requires std::integral<T> || std::floating_point<T>
    void add(const T& val) {
        add(&val, sizeof(T));
    }

    XXH128_hash_t hash() {
        return XXH3_128bits_digest(state);
    }

    std::string to_string() {
        auto res = hash();

        std::stringstream ss;
        // Format high and low 64-bit parts as 16-character padded hex strings
        ss << std::hex << std::setfill('0')
           << std::setw(16) << res.high64
           << std::setw(16) << res.low64;
        return ss.str();
    }

private:
    XXH3_state_t* const state = XXH3_createState();
};

std::string getBinaryHash() {
    std::ifstream file("/proc/self/exe", std::ios::binary);
    assert(file);

    Hash128 h;
    char buffer[65536];
    while (file.read(buffer, sizeof(buffer))) {
        h.add(buffer, file.gcount());
    }
    h.add(buffer, file.gcount());

    auto hash = h.to_string();
    return hash;
}

struct CacheEntry {
    double MMendL, MMbegL, MMmidL;
    double MMendK, MMbegK, MMmidK;
    double MMendKsimple, MMbegKsimple, MMmidKsimple;
    double MLendL, MLbegL, MLmidL;
    double MLendK, MLbegK, MLmidK;
    double MLendKsimple, MLbegKsimple, MLmidKsimple;
    double LMendL, LMbegL, LMmidL;
    double LMendK, LMbegK, LMmidK;
#ifdef FORWARD_ONLY_FILTER
    double fmm_mid_l, fmm_mid_k;
#endif

    template<class Archive>
    void serialize(Archive& archive) {
        archive(
            MMendL, MMbegL, MMmidL,
            MMendK, MMbegK, MMmidK,
            MMendKsimple, MMbegKsimple, MMmidKsimple,
            MLendL, MLbegL, MLmidL,
            MLendK, MLbegK, MLmidK,
            MLendKsimple, MLbegKsimple, MLmidKsimple,
            LMendL, LMbegL, LMmidL,
            LMendK, LMbegK, LMmidK
#ifdef FORWARD_ONLY_FILTER
            , fmm_mid_l, fmm_mid_k
#endif
        );
    }
};

// vibe-coded cache
class ProfileCache {
public:
    ~ProfileCache() {
        save();
    }

    std::string computeCacheKey(const Profile &profile, const Float *letterFreqs,
                                    int sequenceLength, int border, int numOfSequences) {
        // DUMMER_CACHE_IGNORE_BINARY_HASH=1 keeps calibration results valid
        // across rebuilds (at the risk of stale entries if FP behavior changes).
        static const bool ignoreBinaryHash = [] {
            const char *e = getenv("DUMMER_CACHE_IGNORE_BINARY_HASH");
            return e && *e && strcmp(e, "0") != 0;
        }();
        Hash128 h;
        if (!ignoreBinaryHash)
            h.add(binaryHash);
        h.add(profile.name);
        h.add(profile.width);
        h.add(profile.length);
        h.add(profile.values, profile.width * (profile.length + 1) * sizeof(Float));
        h.add(letterFreqs, (profile.width - nonLetterWidth) * sizeof(Float));
        h.add(sequenceLength);
        h.add(border);
        h.add(numOfSequences);
        h.add(INSERT1);
        h.add(INSERT2);
        h.add(DELETE1);
        h.add(DELETE2);
        h.add(BACKGROUND_FRAMESHIFT_RATE);
        h.add(BACKGROUND_FRAMESHIFT_RATE_2);
        h.add(STOP_CODON_PROB);
        h.add(BG_STOP_CODON_PROB);
        h.add(TANTAN_MASK_THRESHOLD);

        return h.to_string();
    }
private:
    std::unordered_map<std::string, CacheEntry> entries;
    //std::unordered_set<std::string> read_entries;
    std::filesystem::path cacheFilePath;
    std::string binaryHash = getBinaryHash();
    std::mutex cacheMutex;
    bool loaded = false;

    static std::filesystem::path getCacheFilePath() {
        std::filesystem::path cacheDir;

        if (const char* xdgCache = std::getenv("XDG_CACHE_HOME"); xdgCache && *xdgCache) {
            cacheDir = std::filesystem::path(xdgCache);
        } else if (const char* home = std::getenv("HOME")) {
            cacheDir = std::filesystem::path(home) / ".cache";
        } else {
            cacheDir = std::filesystem::current_path();
        }

        std::filesystem::path appCacheDir = cacheDir / "dummer";
        if (!std::filesystem::exists(appCacheDir)) {
            std::error_code ec;
            std::filesystem::create_directories(appCacheDir, ec);
            if (ec) {
                std::cerr << "# Warning: Could not create cache directory: " << ec.message() << "\n";
                appCacheDir = std::filesystem::current_path();
            }
        }

        return appCacheDir / "cache.bin";
    }

    void load() {
        if (loaded) return;

        cacheFilePath = getCacheFilePath();
        std::cout << "# Cache file: " << cacheFilePath << std::endl;

        for (int tries = 0; tries < 5; tries++) {
            if (std::filesystem::exists(cacheFilePath)) {
                try {
                    std::ifstream in(cacheFilePath, std::ios::binary);
                    if (in.is_open() && in.peek() != std::ifstream::traits_type::eof()) {
                        cereal::BinaryInputArchive archive(in);
                        archive(entries);
                        break;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "# Cache Load Error: " << e.what() << ".\n";
                    std::error_code delete_ec;
                    std::filesystem::remove(cacheFilePath, delete_ec);
                    entries.clear();
                }
            }
        }

        loaded = true;
    }

public:
    bool lookup(const Profile &profile, const Float *letterFreqs, int sequenceLength,
                int border, int numOfSequences, CacheEntry &outEntry) {
        std::scoped_lock lock(cacheMutex);
        load();

        std::string key = computeCacheKey(profile, letterFreqs, sequenceLength, border, numOfSequences);
        if (auto it = entries.find(key); it != entries.end()) {
            outEntry = it->second;
            return true;
        }
        return false;
    }

    void store(const Profile &profile, const Float *letterFreqs, int sequenceLength,
               int border, int numOfSequences, const CacheEntry &entry) {
        std::scoped_lock lock(cacheMutex);
        load();

        std::string key = computeCacheKey(profile, letterFreqs, sequenceLength, border, numOfSequences);
        entries[key] = entry;
    }

    void save() {
        std::scoped_lock lock(cacheMutex);
        load();

        // std::erase_if(entries, [&](const auto& pair) {
        //     return read_entries.find(pair.first) == read_entries.end();
        // });

        try {
            std::ofstream out(cacheFilePath, std::ios::binary | std::ios::trunc);
            if (out) {
                cereal::BinaryOutputArchive archive(out);
                archive(entries);
            }
        } catch (const std::exception& e) {
            std::cerr << "# Cache Save Error: " << e.what() << std::endl;
        }
    }
} cache;

void estimateK(Profile &profile, const Float *letterFreqs, char *sequence, int sequenceLength,
               int border, int numOfSequences, int printVerbosity, ThreadPool &threadPool, std::vector<DPScratch> &threadScratches,
               std::ofstream* scoresFile = nullptr) {
    // One DP pass yields both the forward-backward (end/beg/mid) scores and
    // the forward-only scores (backward accumulator), stored in one entry.
    CacheEntry entry;
    if (scoresFile == nullptr &&
        cache.lookup(profile, letterFreqs, sequenceLength, border, numOfSequences, entry)) {
        if (printVerbosity > 1) {
            std::cout << "# Warning: using cached results\n";
        }

        profile.gumbelKendAnchored = entry.MMendK;
        profile.gumbelKbegAnchored = entry.MMbegK;
        profile.gumbelKmidAnchored = entry.MMmidK;
        profile.lambda = entry.MMmidL;
#ifdef FORWARD_ONLY_FILTER
        profile.gumbel_k_forward_only = entry.fmm_mid_k;
        profile.lambda_forward_only = entry.fmm_mid_l;
#endif
    } else {
        int alphabetSize = profile.width - nonLetterWidth;

        std::vector<double> aaFreqs;
        Float sum = 0;
        if (alphabetSize > 4) {
            aaFreqs.resize(alphabetSize + 1);
            for (int k = 0; k < alphabetSize; ++k) {
                int n = aa2codons.at(getAlphabet(alphabetSize)[k]).size();
                aaFreqs[k] = letterFreqs[k] * n;
                sum += aaFreqs[k];
            }
            aaFreqs[alphabetSize] = BG_STOP_CODON_PROB;
            sum += aaFreqs[alphabetSize];
        } else {
            aaFreqs.assign(letterFreqs, letterFreqs + alphabetSize);
        }

        std::cout << "# sum is " << sum << std::endl;
        std::discrete_distribution<> dist(aaFreqs.begin(), aaFreqs.end());
        std::vector<double> scores(numOfSequences * 3);
        double *endScores = scores.data();
        double *begScores = endScores + numOfSequences;
        double *midScores = begScores + numOfSequences;
#ifdef FORWARD_ONLY_FILTER
        std::vector<double> fwdOnlyScores(numOfSequences);
#endif

        if (printVerbosity > 1) {
            std::cout << "#trial\tend-anchored\t\tstart-anchored\t\tmid-anchored\n\
#\tprofPos\tseqPos\tscore\tprofPos\tseqPos\tscore\tprofPos\tseqPos\tscore"
                      << std::endl;
        }

        auto alphabet = getAlphabet(20);
        char charToNumber[256];
        setCharToNumber(charToNumber, alphabet);

        int batchStride;
#ifdef FORWARD_ONLY_FILTER
        batchStride = simdWidth;
#else
        batchStride = simdWidth;
#endif
        static const char* lanes_env = getenv("DUMMER_MAX_LANES");
        if (lanes_env) batchStride = std::clamp(std::atoi(lanes_env), 1, batchStride);
        int numBatches = (numOfSequences + batchStride - 1) / batchStride;

#ifdef FORWARD_ONLY_FILTER
        int maxLanes = simdWidth;
#else
        int maxLanes = simdWidth;
#endif
        std::vector<std::vector<std::vector<char>>> threadLocalSeqs(threadScratches.size());
        for (auto &localSeqs : threadLocalSeqs) {
            localSeqs.resize(maxLanes);
            for (int lane = 0; lane < maxLanes; ++lane) {
                localSeqs[lane].resize(sequenceLength + border + 16);
            }
        }

        std::atomic<int> completedBatches(0);
        std::mutex mtx;
        std::condition_variable cv;

        for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            threadPool.enqueue([&, batchIdx](int threadId) {
                DPScratch &threadScratch = threadScratches[threadId];
                auto &localSeqs = threadLocalSeqs[threadId];

                int start = batchIdx * batchStride;
                int activeCount = std::min(static_cast<int>(batchStride), numOfSequences - start);

                // Generate random sequences; use maxLanes for array size
                std::vector<std::vector<uint8_t>> decoded(maxLanes);

                for (int lane = 0; lane < activeCount; ++lane) {
                    int trialIdx = start + lane;
                    std::mt19937_64 trialRandGen(5489 + trialIdx);
                    char *seqBuf = localSeqs[lane].data();

                    const char bases[] = {'A', 'C', 'G', 'T'};
                    std::uniform_int_distribution<int> distDNA(0, 3);
                    std::uniform_int_distribution<int> distOffset(0, 2);

                    int offset = distOffset(trialRandGen);
                    for (int j = 0; j < offset; j++) {
                        seqBuf[j] = charToNumber[bases[distDNA(trialRandGen)]];
                    }
                    for (int j = offset; j <= sequenceLength; j += 3) {
                        double r = std::generate_canonical<double, 10>(trialRandGen);
                        if (r < BACKGROUND_FRAMESHIFT_RATE) {
                            if (j <= sequenceLength) {
                                seqBuf[j] = charToNumber[bases[distDNA(trialRandGen)]];
                            }
                            j -= 2;
                            continue;
                        } else if (r < BACKGROUND_FRAMESHIFT_RATE + BACKGROUND_FRAMESHIFT_RATE_2) {
                            if (j <= sequenceLength) {
                                seqBuf[j] = charToNumber[bases[distDNA(trialRandGen)]];
                            }
                            if (j + 1 <= sequenceLength) {
                                seqBuf[j + 1] = charToNumber[bases[distDNA(trialRandGen)]];
                            }
                            j -= 1;
                            continue;
                        }
                        int x = dist(trialRandGen);
                        char aa = (x < alphabetSize) ? alphabet[x] : '*';
                        const auto &codons = aa2codons.at(aa);
                        std::uniform_int_distribution<int> dist2(0, (int)codons.size() - 1);
                        const auto &xx = codons[dist2(trialRandGen)];
                        for (int k = 0; k < 3; k++) {
                            if (j + k <= sequenceLength) {
                                seqBuf[j + k] = charToNumber[xx[k]];
                            }
                        }
                    }

                    for (int j = 0; j < border; ++j)
                        seqBuf[sequenceLength + j] = seqBuf[j];

                    decoded[lane] = decodeSequence(seqBuf, sequenceLength + border, alphabet, charToNumber);
                }

                {
                    std::array<std::vector<uint8_t>*, simdWidth> decodedPtrs = {};
                    for (int k = 0; k < activeCount; ++k)
                        decodedPtrs[k] = &decoded[k];
                    std::array<Float, simdWidth> minProbRatio;
                    minProbRatio.fill(-2.0f);
                    std::array<std::vector<AlignedSimilarity>, simdWidth> simsSIMD;
                    simd_t bwdBest = simd_t(0.0);
                    findSimilarities(simsSIMD, profile, decodedPtrs, minProbRatio, threadScratch, activeCount,
                                     &bwdBest);
                    alignas(64) Float bwdBestArr[simdWidth];
                    simd_unchecked_store(bwdBest, bwdBestArr, Kokkos::Experimental::simd_flag_default);

                    for (int lane = 0; lane < activeCount; ++lane) {
                        int trialIdx = start + lane;
                        const auto &sims = simsSIMD[lane];
                        endScores[trialIdx] = log(sims[0].probRatio);
                        begScores[trialIdx] = log(sims[1].probRatio);
                        midScores[trialIdx] = log(sims[2].probRatio);
#ifdef FORWARD_ONLY_FILTER
                        fwdOnlyScores[trialIdx] = log((double)bwdBestArr[lane]);
#endif

                        if (printVerbosity > 1) {
                            std::lock_guard<std::mutex> lock(g_cout_mutex);
                            std::cout << (trialIdx + 1) << "\t" << sims[0].anchor1 << "\t" << sims[0].anchor2 << "\t"
                                      << log(sims[0].probRatio) + shift << "\t" << sims[1].anchor1 << "\t"
                                      << sims[1].anchor2 << "\t" << log(sims[1].probRatio) + shift << "\t"
                                      << sims[2].anchor1 << "\t" << sims[2].anchor2 << "\t"
                                      << log(sims[2].probRatio) + shift << std::endl;
                        }
                    }
                }

                if (++completedBatches == numBatches) {
                    std::lock_guard<std::mutex> lock(mtx);
                    cv.notify_one();
                }
            });
        }

        if (numBatches > 0) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]{ return completedBatches == numBatches; });
        }

        if (scoresFile && scoresFile->is_open()) {
            for (int trial = 0; trial < numOfSequences; ++trial) {
                (*scoresFile) << "forward-backward" << "\t" << profile.name << "\t" << trial + 1 << "\tend\t"
                              << endScores[trial] + shift << "\n";
                (*scoresFile) << "forward-backward" << "\t" << profile.name << "\t" << trial + 1 << "\tstart\t"
                              << begScores[trial] + shift << "\n";
                (*scoresFile) << "forward-backward" << "\t" << profile.name << "\t" << trial + 1 << "\tmid\t"
                              << midScores[trial] + shift << "\n";
#ifdef FORWARD_ONLY_FILTER
                (*scoresFile) << "forward-only" << "\t" << profile.name << "\t" << trial + 1 << "\tall\t"
                              << fwdOnlyScores[trial] + shift << "\n";
#endif
            }
        }

    double MMendL, MMendK, MMendKsimple, MLendL, MLendK, MLendKsimple;
    double LMendL, LMendK;
    estimateGumbel(MMendL, MMendK, MMendKsimple, MLendL, MLendK, MLendKsimple, LMendL, LMendK,
                   endScores, numOfSequences, sequenceLength);

    double MMbegL, MMbegK, MMbegKsimple, MLbegL, MLbegK, MLbegKsimple;
    double LMbegL, LMbegK;
    estimateGumbel(MMbegL, MMbegK, MMbegKsimple, MLbegL, MLbegK, MLbegKsimple, LMbegL, LMbegK,
                   begScores, numOfSequences, sequenceLength);

    double MMmidL, MMmidK, MMmidKsimple, MLmidL, MLmidK, MLmidKsimple;
    double LMmidL, LMmidK;
    estimateGumbel(MMmidL, MMmidK, MMmidKsimple, MLmidL, MLmidK, MLmidKsimple, LMmidL, LMmidK,
                   midScores, numOfSequences, sequenceLength);

#ifdef FORWARD_ONLY_FILTER
    double fMMmidL, fMMmidK, fMMmidKsimple, fMLmidL, fMLmidK, fMLmidKsimple;
    double fLMmidL, fLMmidK;
    estimateGumbel(fMMmidL, fMMmidK, fMMmidKsimple, fMLmidL, fMLmidK, fMLmidKsimple, fLMmidL, fLMmidK,
                   fwdOnlyScores.data(), numOfSequences, sequenceLength);
    {
        double mn = *std::min_element(fwdOnlyScores.begin(), fwdOnlyScores.end());
        double mx = *std::max_element(fwdOnlyScores.begin(), fwdOnlyScores.end());
        double mean_s = 0; for (int z = 0; z < numOfSequences; z++) mean_s += fwdOnlyScores[z]; mean_s /= numOfSequences;
        if (verbosity > 0)
            std::cout << "# DBG forward-only scores: min=" << mn << " max=" << mx << " mean=" << mean_s << " range=" << (mx-mn) << "\n";
    }
    profile.gumbel_k_forward_only = fMMmidK;
    profile.lambda_forward_only = fMMmidL;
#endif
        profile.gumbelKendAnchored = MMendK;
        profile.gumbelKbegAnchored = MMbegK;
        profile.gumbelKmidAnchored = MMmidK;
        profile.lambda = MMmidL;

        entry = {
            MMendL, MMbegL, MMmidL,
            MMendK, MMbegK, MMmidK,
            MMendKsimple, MMbegKsimple, MMmidKsimple,
            MLendL, MLbegL, MLmidL,
            MLendK, MLbegK, MLmidK,
            MLendKsimple, MLbegKsimple, MLmidKsimple,
            LMendL, LMbegL, LMmidL,
            LMendK, LMbegK, LMmidK
#ifdef FORWARD_ONLY_FILTER
            , fMMmidL, fMMmidK
#endif
        };
        cache.store(profile, letterFreqs, sequenceLength, border, numOfSequences, entry);
    }

    double s = scale;

    auto printEntryStats = [&](const CacheEntry &e) {
        if (printVerbosity > 1) {
            std::cout << "#\tend-\tstart-\tmid-anchored\n";

            std::cout << "#lamMM\t" << e.MMendL << "\t" << e.MMbegL << "\t" << e.MMmidL << "\n"

                      << "#kMM\t" << e.MMendK / pow(s, e.MMendL) << "\t" << e.MMbegK / pow(s, e.MMbegL) << "\t"
                      << e.MMmidK / pow(s, e.MMmidL) << "\n"

                      << "#kMM1\t" << e.MMendKsimple / scale << "\t" << e.MMbegKsimple / scale << "\t"
                      << e.MMmidKsimple / scale << "\n";

            std::cout << "#lamML\t" << e.MLendL << "\t" << e.MLbegL << "\t" << e.MLmidL << "\n"

                      << "#kML\t" << e.MLendK / pow(s, e.MLendL) << "\t" << e.MLbegK / pow(s, e.MLbegL) << "\t"
                      << e.MLmidK / pow(s, e.MLmidL) << "\n"

                      << "#kML1\t" << e.MLendKsimple / scale << "\t" << e.MLbegKsimple / scale << "\t"
                      << e.MLmidKsimple / scale << "\n";

            std::cout << "#lamLM\t" << e.LMendL << "\t" << e.LMbegL << "\t" << e.LMmidL << "\n"

                      << "#kLM\t" << e.LMendK / pow(s, e.LMendL) << "\t" << e.LMbegK / pow(s, e.LMbegL) << "\t"
                      << e.LMmidK / pow(s, e.LMmidL) << "\n";
        } else if (printVerbosity > 0) {
            std::cout << "# K: " << e.MMendKsimple / scale << " " << e.MMbegKsimple / scale << " "
                      << e.MMmidKsimple / scale << "\n";
        } else {
            std::cout << "# K: " << e.MMmidKsimple / scale << "\n";
        }

        std::cout << "# Lambda: " << e.MMmidL << "\n";
    };

    printEntryStats(entry);
#ifdef FORWARD_ONLY_FILTER
    {
        CacheEntry fwd = entry;
        fwd.MMendL = fwd.MMbegL = fwd.MMmidL = entry.fmm_mid_l;
        fwd.MMendK = fwd.MMbegK = fwd.MMmidK = entry.fmm_mid_k;
        printEntryStats(fwd);
    }
#endif
}

int intFromText(const char *text) {
    long x = strtol(text, 0, 0);
    if (x > INT_MAX || x < INT_MIN)
        return -1;
    return x;
}

double probFromText(const char *text) {
    if (*text == '*')
        return 0;
    double d = strtod(text, 0);
    return exp(-d);
}

void normalize(Float *x, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i)
        sum += x[i];
    assert(sum > 0);
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}

double meanOfLogs(const Float *x, int n) {
    double m = 1;
    for (int i = 0; i < n; ++i)
        m *= x[i];
    return log(m) / n;
}

double myMean(const Float *values, int length, int step, int meanType, Float *valuesForMedian,
              const float *tantanProbs) {
    double mean = 0;
    int n = 0;
    for (int i = 0; i < length; ++i) {
        if (tantanProbs[i] >= TANTAN_MASK_THRESHOLD)
            continue;
        double v = values[i * step];
        // Geometric mean is bad for zero (or very low) probabilities
        // All letter probs in Dfam-curated_only 3.9 and Pfam-A 38.0 are > 1e-6
        if (meanType == 'G')
            mean += log(std::max(v, 1e-6)); // geometric mean
        if (meanType == 'A')
            mean += v; // arithmetic mean
        if (meanType == 'M')
            valuesForMedian[n] = v; // median
        ++n;
    }
    assert(n > 0);
    if (meanType == 'G')
        return exp(mean / n);
    if (meanType == 'A')
        return mean / n;
    std::sort(valuesForMedian, valuesForMedian + n);
    return valuesForMedian[n / 2];
}

void filterLetterProbabilities(Float *letterProbs, int length, int step, double stdDev,
                               bool keepNonvaryingTerm) {
    const double sqrt2pi = 2.5066282746310005;
    const double inv2var = 0.5 / (stdDev * stdDev);
    const int gaussianLimit = ceil(stdDev * 8); // truncate Gaussian tails
    const int alphabetSize = step - nonLetterWidth;
    std::vector<double> meanLogProbs(length);
    std::vector<double> values(length);

    for (int i = 0; i < length; ++i) {
        meanLogProbs[i] = meanOfLogs(letterProbs + i * step, alphabetSize);
    } // at each position in the profile, calculate: mean(log(letter prob))

    for (int k = 0; k < alphabetSize; ++k) {
        for (int i = 0; i < length; ++i) {
            double prob = letterProbs[i * step + k];
            values[i] = log(prob) - meanLogProbs[i]; // apply filter to this
        }

        double addItBack = keepNonvaryingTerm ? mean(values.data(), length) : 0.0;
        for (int i = 0; i < length; ++i) {
            double sum = 0;
            for (int j = -gaussianLimit; j <= gaussianLimit; ++j) {
                // xxx this treats the profile as circular (wrapping around at
                // the edges), which is rarely appropriate, but ensures no
                // change in average value:
                int x = (i + j) % length;
                if (x < 0)
                    x += length;
                sum += values[x] * exp(-1.0 * j * j * inv2var);
            }
            sum /= stdDev * sqrt2pi;
            letterProbs[i * step + k] = exp(values[i] - sum + addItBack);
        }
    }

    for (int i = 0; i < length; ++i) {
        normalize(letterProbs + i * step, alphabetSize);
    }
}

int finalizeProfile(Profile &p, char *consensusSequence, int backgroundProbsType, bool isMask,
                    double filterStdDev, bool keepNonvaryingTerm) {
    int alphabetSize = p.width - nonLetterWidth;
    std::vector<float> tantanProbs(p.length);
    std::vector<Float> valuesForMedian(p.length);
    Float *end = p.values + p.width * p.length;

    const char *alphabet = getAlphabet(alphabetSize);
    if (end[3] <= 0) {
        // set the final epsilon to the geometric mean of the other epsilons
        end[3] = myMean(p.values + p.width + 3, p.length - 1, p.width, 'G', valuesForMedian.data(),
                        tantanProbs.data());
    }

    if (filterStdDev > 0) {
        filterLetterProbabilities(p.values + 4, p.length, p.width, filterStdDev,
                                  keepNonvaryingTerm);
    } else if (isMask && (alphabetSize == 4 || alphabetSize == 20)) {
        calcTantanProbabilities((const unsigned char *)consensusSequence, p.length,
                                alphabetSize > 4, tantanProbs.data());
    }

    double sumOfMeans = 0;
    for (int k = 4; k < 4 + alphabetSize; ++k) {
        double mean = myMean(p.values + k, p.length, p.width, backgroundProbsType,
                             valuesForMedian.data(), tantanProbs.data());
        end[k] = mean;
        sumOfMeans += mean;
    }

    p.bg_probs.push_back(0);
    p.bg_probs.push_back(0);
    p.bg_probs.push_back(0);
    p.bg_probs.push_back(0);
    for (int k = 4; k < 4 + alphabetSize; ++k) {
        end[k] /= sumOfMeans;
        end[k] *= (1 - BG_STOP_CODON_PROB) / aa2codons.at(alphabet[k - 4]).size();
        p.bg_probs.push_back(end[k]);
    }
    p.bg_probs.push_back(0);
    p.bg_probs.push_back(0);
    p.bg_probs.push_back(1.0 / 64.0);
    p.bg_probs.push_back(BG_STOP_CODON_PROB / 3.0);
    p.bg_probs.push_back(0); // For zero_idx padding

    for (int k = 0; k < alphabetSize + 9; ++k) {
        p.log2_bg_probs.push_back(p.bg_probs[k] > 0 ? log2(p.bg_probs[k]) : -INFINITY);
    }

    std::unordered_map<char, double> dist;
    char charToNumber[256];
    setCharToNumber(charToNumber, alphabet);
    for (auto c : std::string(alphabet)) {
        if (c == 'O')
            c = 'K';
        if (c == 'U')
            c = 'C';
        dist[c] = end[4 + charToNumber[c]] * (1 - STOP_CODON_PROB);
    }
    dist['*'] = STOP_CODON_PROB;

    p.values_v2.reserve(p.length + 5);
    for (int i = 0;; ++i) {
        p.values_v2.push_back({0});

        Float *probs = p.values + i * p.width;
        double alpha = probs[0] * (1 - INSERT1 - INSERT2 - DELETE1 - DELETE2);
        double beta = probs[1];

        double alphaFS1 = INSERT1;
        double alphaFS2 = INSERT2;
        p.values_v2.rbegin()->alpha_prime[0] = alpha * (1 - beta);
        p.values_v2.rbegin()->alpha_prime[1] = alphaFS1 * (1 - beta);
        p.values_v2.rbegin()->alpha_prime[2] = alphaFS2 * (1 - beta);

        p.values_v2.rbegin()->beta_prime[0] = beta;
        // all beta, are equal for now
        for (int j = 1; j <= 2; j++) { // perform actual insertion after frameshift
            p.values_v2.rbegin()->beta_prime[j] = 0;
        }

        double delta = probs[2] * (1 - INSERT1 - INSERT2 - DELETE1 - DELETE2);;
        double epsilon = probs[3];
        if (i == p.length)
            break;

        double delta1 = probs[p.width + 2];
        double epsilon1 = probs[p.width + 3];
        double deltaFS1 = DELETE1; // simulate delete
        double deltaFS2 = DELETE2;
        p.values_v2.rbegin()->delta_prime[0] = delta * (1 - epsilon1);
        p.values_v2.rbegin()->delta_prime[1] = deltaFS1 * (1 - epsilon1);
        p.values_v2.rbegin()->delta_prime[2] = deltaFS2 * (1 - epsilon1);
        p.values_v2.rbegin()->epsilon_prime = epsilon * (1 - epsilon1) / (1 - epsilon);
        p.values_v2.rbegin()->enter_match_probability = (1 - alpha - alphaFS1 - alphaFS2 - delta - deltaFS1 - deltaFS2);

        double c = (1 - alpha - delta);
        if (epsilon >= 1)
            return 0;
        probs[2] = delta;
        probs[3] = epsilon;
        for (int k = 4; k < 4 + alphabetSize; ++k) {
            assert(alphabet[k - 4] != '*');
            if (tantanProbs[i] >= TANTAN_MASK_THRESHOLD) {
                probs[k] = end[k];
            } else {
                double p = probs[k];
                int codonCount = (alphabetSize > 4) ? aa2codons.at(alphabet[k - 4]).size() : 1;
                probs[k] = ((1 - STOP_CODON_PROB) /* minus stop codon */ * p / codonCount);
            }
        }
        if (alphabetSize == 20) {
            probs[4 + 20] = probs[4 + 1]; // selenocysteine = cysteine
            probs[4 + 21] = probs[4 + 8]; // pyrrolysine = lysine
        }
        probs[4 + alphabetSize + 2] = 1.0 / 64.0; // for masked sequence letters
        probs[4 + alphabetSize + 3] = STOP_CODON_PROB / 3.0;
        probs[4 + alphabetSize + 4] = 0.0; // zero_idx padding
        if (tantanProbs[i] >= TANTAN_MASK_THRESHOLD)
            consensusSequence[i] |= 32;
    }

    // extra padding
    p.values_v2.push_back({0});

    return 1;
}

int readProfiles(std::istream &in, std::vector<Profile> &profiles, std::vector<Float> &values,
                 std::vector<char> &charVec, int backgroundProbsType, bool isMask,
                 double filterStdDev, bool keepNonvaryingTerm) {
    Profile profile = {0};
    int state = 0;
    std::string line, word;
    while (getline(in, line)) {
        std::istringstream iss(line);
        iss >> word;
        switch (state) {
        case 0:
            if (word == "NAME") {
                profile.nameIdx = charVec.size();
                iss >> word;
                profile.name = word;
                const char *name = word.c_str();
                charVec.insert(charVec.end(), name, name + word.size() + 1);
                profile.consensusSequenceIdx = charVec.size();
            } else if (word == "HMM") {
                ++state;
            }
            break;
        case 1:
            ++state;
            break;
        case 2:
            if (word != "COMPO")
                ++state;
            break;
        case 3: {
            iss >> word;
            double MtoI = probFromText(word.c_str());
            iss >> word;
            double MtoD = probFromText(word.c_str());
            iss >> word >> word;
            double ItoI = probFromText(word.c_str());
            iss >> word >> word;
            double DtoD = probFromText(word.c_str());
            if (!iss)
                return 0;
            if (MtoI > 1 || MtoD > 1 || ItoI > 1 || DtoD > 1)
                return 0;
            values.push_back(MtoI);
            values.push_back(ItoI);
            values.push_back(MtoD);
            values.push_back(DtoD);
        }
            ++state;
            break;
        case 4:
            if (word == "//") {
                if (profile.length < 2)
                    return 0;
                values.insert(values.end(), profile.width - 4, 0.0);
                profiles.push_back(profile);
                profile.width = profile.length = 0;
                state = 0;
            } else {
                int k = 0;
                while (iss >> word && strchr(word.c_str(), '.')) { // xxx "*"?
                    double prob = probFromText(word.c_str());
                    if (prob > 1)
                        return 0;
                    values.push_back(prob);
                    ++k;
                }
                values.insert(values.end(), nonLetterWidth - 4, 0.0); // extra letters
                if (k == 0)
                    return 0;
                if (profile.width > 0 && k + nonLetterWidth != profile.width)
                    return 0;
                profile.width = k + nonLetterWidth;
                profile.length += 1;
                if (profile.length + 1 > INT_MAX / profile.width)
                    return 0;
                const Float *letterProbs = &values[values.size() - profile.width + 4];
                const Float *m = std::max_element(letterProbs, letterProbs + k);
                charVec.push_back(m - letterProbs); // consensus sequence
                state = 2;
            }
        }
    }

    Float *v = &values[0];
    for (auto &p : profiles) {
        p.values = v;
        char *consensus = &charVec[p.consensusSequenceIdx];
        if (!finalizeProfile(p, consensus, backgroundProbsType, isMask, filterStdDev,
                             keepNonvaryingTerm))
            return 0;
        v += p.width * (p.length + 1);
    }

    return state == 0;
}
void makeMaskedSequence(char *sequence, int length, int alphabetSize) {
    std::vector<float> tantanProbs(length);
    std::string seq2;
    for (int i = 0; i < length; i++) {
        char val = '\0';
        switch (sequence[i]) {
        case 0:
            val = 0;
            break;
        case 1:
            val = 1;
            break;
        case 5:
            val = 2;
            break;
        case 16:
            val = 3;
            break;
        default:
            val = 0;
            break;
        }
        seq2 += val;
    }

    calcTantanProbabilities((const unsigned char *)seq2.c_str(), length, false, tantanProbs.data());
    int mask = alphabetSize + 2;
    for (int i = 0; i < length; ++i) {
        sequence[length + i] = (tantanProbs[i] < TANTAN_MASK_THRESHOLD) ? sequence[i] : mask;
    }
}

int main(int argc, char *argv[]) {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    build_standard_genetic_code(); // hack

    double evalueOpt = OPT_e;
    int strandOpt = OPT_s;
    int maskOpt = OPT_m;
    bool maxModeOpt = false;
#ifdef FORWARD_ONLY_FILTER
    double forward_only_evalue_opt = -1;
    bool forward_only_evalue_explicit = false;
#endif
    long long totSequenceLengthOverride = -1;
    double filterStdDev = 0;
    bool keepNonvaryingTerm = false;
    int randomSeqNum = OPT_t;
    int randomSeqLen = OPT_l;
    int border = OPT_b;
    int backgroundProbsType = 'G';
    char* scoresFilename = nullptr;
    int numThreadsOpt = std::thread::hardware_concurrency();
    if (numThreadsOpt == 0) {
        numThreadsOpt = 1;
    }

    const char help[] = "\
usage: dummer profiles.hmm [sequences.fa]\n\
\n\
Find similarities between sequences and profiles.   A profile is a set of\n\
position-specific letter, deletion, and insertion probabilities.\n\
\n\
Options:\n\
  -h, --help        show this help message and exit\n\
  -V, --version     show version and exit\n\
  -v, --verbose     show progress messages\n\
  -T N, --threads N number of threads to use (default: CPU cores)\n\
  -e E, --evalue E  find similarities with E-value <= this (default: " STR(OPT_e) ")\n\
  -N N, --total-length N  override total search space size (default: sum of input lengths)\n\
  -s S, --strand S  DNA strand: 0=reverse, 1=forward, 2=both (default: " STR(OPT_s) ")\n\
  -m M, --mask M    mask simple regions of:\n\
                    0=neither, 1=profile, 2=sequence, 3=both (default: " STR(OPT_m) ")\n\
\n\
Options for low-cut/high-pass filter on position-specific letter probabilities:\n\
  -d D, --dev D     standard deviation for Gaussian filter\n\
  -D D, --Dev D     same as above, but keep the non-varying component\n\
\n\
Options for random sequences:\n\
  -t T, --trials T  generate this many random sequences (default: " STR(OPT_t) ")\n\
  -l L, --length L  length of each random sequence (default: " STR(OPT_l) ")\n\
     -b B, --border B  add this size border to each random sequence (default: " STR(OPT_b) ")\n\
   -S F, --scores-file F  write adjusted bit scores of random sequences to file F\n\
                              and exit (skips sequence search)\n\
\n\
Options for background letter probabilities:\n\
  --barithmetic     arithmetic mean of position-specific probabilities\n\
  --bgeometric      geometric mean of position-specific probabilities (default)\n\
  --bmedian         median of position-specific probabilities\n\
\n\
Options for frameshifts, stop codons, and masking:\n\
  --insert1 F       1-base insertion rate per base (default: " STR(OPT_insert1) ")\n\
  --insert2 F       2-base insertion rate per base (default: " STR(OPT_insert2) ")\n\
  --delete1 F       1-base deletion rate per base (default: " STR(OPT_delete1) ")\n\
  --delete2 F       2-base deletion rate per base (default: " STR(OPT_delete2) ")\n\
  --stop-codon-prob F  stop codon probability (default: " STR(OPT_stop) ")\n\
  --bg-stop-codon-prob F  background stop codon probability (default: " STR(OPT_bg_stop) ")\n\
  --tantan-threshold F  tantan masking threshold (default: " STR(OPT_tantan) ")\n\
                    1-bp frameshift branch = insert1 + delete2; 2-bp branch = insert2 + delete1\n\
\n\
Environment:\n\
  DUMMER_CACHE_IGNORE_BINARY_HASH=1  reuse the E-value calibration cache across rebuilds\n\
\n\
Max sensitivity:\n\
  --max             skip/bypass forward-only pre-filter (overrides -W/--forward-only-evalue)\n"
#ifdef FORWARD_ONLY_FILTER
"\n\
Forward-only pre-filter options:\n\
  -W E, --forward-only-evalue E  Forward-only pre-filter E-value threshold (default: value of -e)\n"
#endif
;

    const char sOpts[] = "hVve:N:s:m:d:D:t:l:b:T:S:"
#ifdef FORWARD_ONLY_FILTER
        "W:"
#endif
        ;

    static struct option lOpts[] = {{"help", no_argument, 0, 'h'},
                                    {"version", no_argument, 0, 'V'},
                                    {"verbose", no_argument, 0, 'v'},
                                    {"threads", required_argument, 0, 'T'},
                                    {"evalue", required_argument, 0, 'e'},
                                    {"total-length", required_argument, 0, 'N'},
                                    {"strand", required_argument, 0, 's'},
                                    {"mask", required_argument, 0, 'm'},
                                    {"dev", required_argument, 0, 'd'},
                                    {"Dev", required_argument, 0, 'D'},
                                    {"trials", required_argument, 0, 't'},
                                    {"length", required_argument, 0, 'l'},
                                    {"border", required_argument, 0, 'b'},
                                    {"barithmetic", no_argument, 0, 'A'},
                                    {"bgeometric", no_argument, 0, 'G'},
                                    {"bmedian", no_argument, 0, 'M'},
                                    {"insert1", required_argument, 0, OPT_INSERT1_CODE},
                                    {"insert2", required_argument, 0, OPT_INSERT2_CODE},
                                    {"delete1", required_argument, 0, OPT_DELETE1_CODE},
                                    {"delete2", required_argument, 0, OPT_DELETE2_CODE},
                                    {"stop-codon-prob", required_argument, 0, OPT_STOP_CODE},
                                    {"bg-stop-codon-prob", required_argument, 0, OPT_BG_STOP_CODE},
                                    {"tantan-threshold", required_argument, 0, OPT_TANTAN_CODE},
                                    {"max", no_argument, 0, OPT_MAX_CODE},
#ifdef FORWARD_ONLY_FILTER
                                     {"forward-only-evalue", required_argument, 0, 'W'},
#endif
                                     {"scores-file", required_argument, 0, 'S'},
                                     {0, 0, 0, 0}};

    int c;
    while ((c = getopt_long(argc, argv, sOpts, lOpts, &c)) != -1) {
        switch (c) {
        case 'h':
            std::cout << help;
            return 0;
        case 'V':
            std::cout << "DUMMER "
#include "version.hh"
                         "\n";
            return 0;
        case 'v':
            ++verbosity;
            break;
        case 'T':
            numThreadsOpt = intFromText(optarg);
            if (numThreadsOpt < 1)
                return badOpt();
            break;
        case 'e':
            evalueOpt = strtod(optarg, 0);
            if (evalueOpt < 0)
                return badOpt();
            break;
        case 'N':
            totSequenceLengthOverride = atoll(optarg);
            if (totSequenceLengthOverride < 0)
                return badOpt();
            break;
        case 's':
            strandOpt = intFromText(optarg);
            if (strandOpt < 0 || strandOpt > 2)
                return badOpt();
            break;
        case 'm':
            maskOpt = intFromText(optarg);
            if (maskOpt < 0 || maskOpt > 3)
                return badOpt();
            break;
        case 'd':
            filterStdDev = strtod(optarg, 0);
            // too low: discretized Gaussian problems; too high: overflow or slow
            if (filterStdDev < 2 || filterStdDev > 1000)
                return badOpt();
            break;
        case 'D':
            filterStdDev = strtod(optarg, 0);
            if (filterStdDev < 2 || filterStdDev > 1000)
                return badOpt();
            keepNonvaryingTerm = true;
            break;
        case 't':
            randomSeqNum = intFromText(optarg);
            if (randomSeqNum < 1)
                return badOpt();
            break;
        case 'l':
            randomSeqLen = intFromText(optarg);
            if (randomSeqLen < 1 || randomSeqLen > INT_MAX - 2 * simdLen)
                return badOpt();
            break;
        case 'b':
            border = intFromText(optarg);
            if (border < 0)
                return badOpt();
            break;
        case 'S':
            scoresFilename = optarg;
            break;
        case 'A':
            backgroundProbsType = 'A';
            break;
        case 'G':
            backgroundProbsType = 'G';
            break;
        case 'M':
            backgroundProbsType = 'M';
            break;
        case OPT_INSERT1_CODE:
            INSERT1 = strtod(optarg, 0);
            if (!(INSERT1 >= 0 && INSERT1 < 1))
                return badOpt();
            break;
        case OPT_INSERT2_CODE:
            INSERT2 = strtod(optarg, 0);
            if (!(INSERT2 >= 0 && INSERT2 < 1))
                return badOpt();
            break;
        case OPT_DELETE1_CODE:
            DELETE1 = strtod(optarg, 0);
            if (!(DELETE1 >= 0 && DELETE1 < 1))
                return badOpt();
            break;
        case OPT_DELETE2_CODE:
            DELETE2 = strtod(optarg, 0);
            if (!(DELETE2 >= 0 && DELETE2 < 1))
                return badOpt();
            break;
        case OPT_STOP_CODE:
            STOP_CODON_PROB = strtod(optarg, 0);
            if (!(STOP_CODON_PROB >= 0 && STOP_CODON_PROB < 1))
                return badOpt();
            break;
        case OPT_BG_STOP_CODE:
            BG_STOP_CODON_PROB = strtod(optarg, 0);
            if (!(BG_STOP_CODON_PROB >= 0 && BG_STOP_CODON_PROB < 1))
                return badOpt();
            break;
        case OPT_TANTAN_CODE:
            TANTAN_MASK_THRESHOLD = strtod(optarg, 0);
            if (!(TANTAN_MASK_THRESHOLD >= 0 && TANTAN_MASK_THRESHOLD <= 1))
                return badOpt();
            break;
#ifdef FORWARD_ONLY_FILTER
        case 'W':
            forward_only_evalue_opt = strtod(optarg, 0);
            if (forward_only_evalue_opt < 0)
                return badOpt();
            forward_only_evalue_explicit = true;
            break;
#endif
        case OPT_MAX_CODE:
            maxModeOpt = true;
            break;
        case '?':
            std::cerr << help;
            return 1;
        }
    }

    if (filterStdDev > 0)
        maskOpt &= 2; // filtering turns off profile-masking

    updateBackgroundFrameshiftRates();
    if (!(INSERT1 + INSERT2 + DELETE1 + DELETE2 < 1))
        return badOpt();
    if (!(BACKGROUND_FRAMESHIFT_RATE + BACKGROUND_FRAMESHIFT_RATE_2 < 1))
        return badOpt();

#ifdef FORWARD_ONLY_FILTER
    // -W defaults to the -e value; --max bypasses the prefilter separately
    // via skipPrefilter (no enable-threshold gate). Note: with -e0 the
    // prefilter passes nothing unless --max is given.
    if (!maxModeOpt && !forward_only_evalue_explicit)
        forward_only_evalue_opt = evalueOpt;
#endif

    if (argc - optind < 1 || argc - optind > 2) {
        std::cerr << help;
        return 1;
    }

    if (border > INT_MAX - 2 * simdLen - randomSeqLen) {
        return err("sequence + border is too big");
    }

    std::vector<char> charVec;
    std::vector<Float> profileValues;
    std::vector<Profile> profiles;

    {
        std::ifstream file;
        std::istream &in = openFile(file, argv[optind]);
        if (!file)
            return 1;
        if (!readProfiles(in, profiles, profileValues, charVec, backgroundProbsType, maskOpt & 1,
                          filterStdDev, keepNonvaryingTerm)) {
            return err("can't read the profile data");
        }
    }

    size_t numOfProfiles = profiles.size();

    int maxProfileLength = 0;
    for (size_t i = 0; i < numOfProfiles; ++i) {
        maxProfileLength = std::max(maxProfileLength, profiles[i].length);
    }

    size_t seqIdx = charVec.size();
    charVec.resize(seqIdx + simdRoundUp(randomSeqLen + border + 1));

    std::cout << "# DUMMER "
#include "version.hh"
                 "\n";
    std::cout << "# Bytes per floating-point number: " << sizeof(Float) << "\n";
    if (filterStdDev > 0)
        std::cout << "# Filtering position-specific letter probabilities: std dev " << filterStdDev
                  << "\n";
    std::cout << "# Background letter probabilities: "
              << (backgroundProbsType == 'A'   ? "arithmetic mean"
                  : backgroundProbsType == 'G' ? "geometric mean"
                                               : "median")
              << " of foreground probabilities\n";
    std::cout << "# Random sequences: trials=" << randomSeqNum << " length=" << randomSeqLen
              << " border=" << border << "\n";
    std::cout << "# Frameshift rates: insert1=" << INSERT1 << " insert2=" << INSERT2
              << " delete1=" << DELETE1 << " delete2=" << DELETE2 << "\n";
    std::cout << "# Stop codon probs: stop=" << STOP_CODON_PROB << " bg-stop=" << BG_STOP_CODON_PROB
               << " tantan-threshold=" << TANTAN_MASK_THRESHOLD << "\n";
    if (maxModeOpt)
        std::cout << "# Max mode: forward-only pre-filter disabled\n";
#ifdef FORWARD_ONLY_FILTER
    else if (argc - optind > 1)
        std::cout << "# Forward-only pre-filter E-value <= " << forward_only_evalue_opt << "\n";
#endif
    if (maskOpt & 1)
        std::cout << "# Masking simple regions in profiles\n";
    if (argc - optind > 1) {
        if (maskOpt & 2)
            std::cout << "# Masking simple regions in sequences\n";
        if (evalueOpt > 0)
            std::cout << "# E-value <= " << evalueOpt << "\n";
        if (strandOpt < 2)
            std::cout << "# Strand: " << (strandOpt ? "forward" : "reverse") << "\n";
    }

    int printVerbosity = (argc - optind < 2) * 2 + (evalueOpt <= 0);

    ThreadPool threadPool(numThreadsOpt);
    std::vector<DPScratch> threadScratches(numThreadsOpt);

    std::ofstream scoresFile;
    if (scoresFilename) {
        scoresFile.open(scoresFilename);
        if (!scoresFile) {
            return err("can't open scores file");
        }
        scoresFile << "mode\tprofile_name\ttrial\tanchor_type\tln_score\n";
    }

    for (auto &p : profiles) {
        std::cout << "\n";
        std::cout << "# Profile name: " << &charVec[p.nameIdx] << "\n";
        std::cout << "# Profile length: " << p.length << "\n";
        if (maskOpt & 1) {
            int maskCount = 0;
            const char *consensus = &charVec[p.consensusSequenceIdx];
            for (int i = 0; i < p.length; ++i)
                maskCount += (consensus[i] > 31);
            std::cout << "# Positions masked by tantan: " << maskCount << "\n";
        }
        const Float *bgProbs = p.values + p.width * p.length + 4;
        std::cout << "# Background letter probabilities:";
        for (int j = 0; j < p.width - nonLetterWidth; ++j)
            std::cout << " " << bgProbs[j];
        std::cout << std::endl;

        char charToNumber[256];
        setCharToNumber(charToNumber, getAlphabet(p.width - nonLetterWidth));

#ifdef EVALUE
        // one pass calibrates both the forward-backward and forward-only models
        estimateK(p, bgProbs, &charVec[seqIdx], randomSeqLen, border, randomSeqNum, printVerbosity, threadPool, threadScratches, scoresFilename ? &scoresFile : nullptr);
#endif
    }

    if (scoresFilename) {
        scoresFile.close();
        return 0;
    }

    if (argc - optind < 2 || numOfProfiles < 1)
        return 0;
    std::cout << std::endl;

    int width = profiles[0].width;
    for (size_t i = 1; i < numOfProfiles; ++i) {
        if (profiles[i].width != width)
            width = 0;
    }
    int alphabetSize = width - nonLetterWidth;
    const char *alphabet = getAlphabet(alphabetSize);
    if (!alphabet) {
        return err("the profiles should be all protein, or all nucleotide");
    }
    char charToNumber[256];
    memset(charToNumber, 127, 256);
    memset(charToNumber, 125, ' ' + 1); // map "space characters" (<= ' ') to 125
    charToNumber['>'] = 126;            // record separator for FASTA-format sequences
    setCharToNumber(charToNumber, alphabet);

    // IUPAC ambiguity
    char charToNumberIUPAC[256];
    std::copy(charToNumber, charToNumber + 256, charToNumberIUPAC);
    for (auto c : "RYSWKMBDHVN") {
        if (c != '\0') charToNumberIUPAC[c] = charToNumberIUPAC[tolower(c)] = 22;
    }

#ifdef PIPELINE_MODE
    strandOpt = 1;
#endif

    charVec.resize(seqIdx);
    std::vector<Sequence> sequences;
    std::vector<FinalSimilarity> similarities;
    size_t totSequenceLength = 0;
    if (totSequenceLengthOverride >= 0)
        totSequenceLength = (size_t)totSequenceLengthOverride;

    std::ifstream file;
    std::istream &in = openFile(file, argv[optind + 1]);
    if (!file)
        return 1;
    Sequence sequence;
    Contig contig = {0, 0};
    std::vector<std::vector<SequenceRequest>> allRequests(numOfProfiles);
    while (readContig(in, sequence, contig, charVec, charToNumberIUPAC)) {
        if (contig.length == 0) {
            sequences.push_back(sequence);
            continue;
        }
        seqIdx = charVec.size() - contig.length;
        size_t maskedSeqIdx = (maskOpt & 2) ? charVec.size() : seqIdx;
        // The algorithms need one arbitrary letter past the end
        // Then round up to a multiple of the SIMD length
        charVec.resize(maskedSeqIdx + simdRoundUp(contig.length + 1));
        if (totSequenceLengthOverride < 0) {
            totSequenceLength += contig.length;
            if (strandOpt == 2)
                totSequenceLength += contig.length;
        }
        char *seq = &charVec[seqIdx];
        for (int s = 0; s < 2; ++s) {
            if (s != strandOpt) {
                if (maskOpt & 2)
                    makeMaskedSequence(seq, contig.length, alphabetSize);
                size_t strandNum = sequences.size() * 2 + s;
                std::vector<uint8_t> decoded =
                    decodeSequence(&charVec[maskedSeqIdx], contig.length, alphabet, charToNumber);
                std::shared_ptr<SequenceData> sd = std::make_shared<SequenceData>(
                    std::move(decoded), std::string(&charVec[seqIdx], contig.length),
                    std::string(&charVec[maskedSeqIdx], contig.length), contig, strandNum);

                for (size_t j = 0; j < numOfProfiles; ++j) {
                    const Profile &p = profiles[j];
#ifdef PIPELINE_MODE
                    if (sequence.target_profile.empty() || !strcmp(&charVec[p.nameIdx], sequence.target_profile.c_str())) {
#endif
                        Float minProbRatio =
                            (evalueOpt > 0)
                                ? (std::pow(p.gumbelKmidAnchored * totSequenceLength / evalueOpt,
                                            1.0 / 1.0 /* p.lambda */))
                                : -1;
                        if (verbosity > 1)
                            std::cerr << "Profile: " << &charVec[p.nameIdx] << "\n";

                        allRequests[j].push_back({sd, minProbRatio});
#ifdef PIPELINE_MODE
                    }
#endif
                }
            }
        }
        charVec.resize(seqIdx);
    }

    findFinalSimilaritiesBatched(similarities, allRequests, profiles, charVec.data(), threadPool, threadScratches
#ifdef FORWARD_ONLY_FILTER
        , forward_only_evalue_opt, totSequenceLength, maxModeOpt
#endif
    );

    std::cout << "# Total sequence length: " << totSequenceLength << "\n";

    std::cout.precision(3);
    for (size_t i = 0; i < similarities.size(); ++i) {
        Profile p = profiles[similarities[i].profileNum];
        Sequence s = sequences[similarities[i].strandNum / 2];
        double k = (evalueOpt > 0) ? p.gumbelKmidAnchored
                   : (i % 3 == 0)  ? p.gumbelKendAnchored
                   : (i % 3 == 1)  ? p.gumbelKbegAnchored
                                   : p.gumbelKmidAnchored;
        double evalue = k * totSequenceLength / pow(similarities[i].probRatio, p.lambda);
        if (evalueOpt <= 0 && i % 3 == 0)
            std::cout << "\n";
        if (evalueOpt > 0 && evalue > evalueOpt)
            continue;
        printSimilarity(charVec.data(), p, s, similarities[i], evalue);
    }

    return 0;
}
