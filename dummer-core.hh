#pragma once

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
#include <type_traits>
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

#include <cstdio>

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
    OPT_MAX_CODE,
    OPT_BATCH_CODE,
    OPT_WARMUP_CODE
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

// Padded DP containers + overflow-proof score arithmetic. Included here
// because both are typed on Float/simd_t/simdWidth above.
#include "dummer-padded.hh"

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
const int shift = STATIC_SHIFT; // add this to scores, to undo the scaling

int verbosity = 0;

static std::mutex g_cout_mutex;

// Always-on accounting (negligible cost: one fetch_add per batch).
static std::atomic<size_t> g_fastBatches{0};
static std::atomic<size_t> g_calibBatches{0};
// Lean-filter accounting: filter batches run, empty batches dropped.
static std::atomic<size_t> g_filterBatches{0};
static std::atomic<size_t> g_filterDiscarded{0};
static std::atomic<size_t> g_maxDpWidth{0};

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
#ifdef FILTER_PASS
    double gumbel_k_filter, lambda_filter;
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
    // Exact pair-domain score (ExpScore): the comparison/sort key, so ordering
    // never materializes a double or calls log(). m == 0 iff there is no hit
    // (or the placeholder).
    ExpScore pair;
    // log(probRatio) in static-scale units, materialized only for reporting
    // (and calibration). For beyond-double hits the double would saturate to
    // +inf, but the exact pair stays finite. -INFINITY iff no hit/placeholder.
    double logProbRatio;
    int anchor1, anchor2;
    std::vector<SegmentPair> alignment;

    bool operator<(const AlignedSimilarity &other) const {
        return scoreLess(this->pair, other.pair);
    }
    bool operator>(const AlignedSimilarity &other) const {
        return scoreLess(other.pair, this->pair);
    }
};

struct FinalSimilarity {
    double logProbRatio;
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


