// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummer-search.hh"
#include "dummer-sequence.hh"

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
#ifdef FILTER_PASS
    double flt_mid_l, flt_mid_k, flt_mid_k_simple;
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
#ifdef FILTER_PASS
            , flt_mid_l, flt_mid_k, flt_mid_k_simple
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
               std::ofstream* scoresFile = nullptr, bool* wasCached = nullptr) {
    // One DP pass yields both the full (end/beg/mid) scores and
    // the filter scores (backward accumulator), stored in one entry.
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
#ifdef FILTER_PASS
        profile.gumbel_k_filter = entry.flt_mid_k;
        profile.lambda_filter = entry.flt_mid_l;
#endif
        if (wasCached)
            *wasCached = true;
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
#ifdef FILTER_PASS
        std::vector<double> fltOnlyScores(numOfSequences);
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
#ifdef FILTER_PASS
        batchStride = simdWidth;
#else
        batchStride = simdWidth;
#endif
        static const char* lanes_env = getenv("DUMMER_MAX_LANES");
        if (lanes_env) batchStride = std::clamp(std::atoi(lanes_env), 1, batchStride);
        int numBatches = (numOfSequences + batchStride - 1) / batchStride;

#ifdef FILTER_PASS
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
                    std::array<ExpScore, simdWidth> filterBest{};
                    // Calibration only needs scores (end/beg/mid + filter);
                    // alignments are never consumed here, so skip the ExpMatrix.
                    // Lazy rescaling keeps every lane finite: no overflow.
                    findSimilarities(simsSIMD, profile, decodedPtrs, minProbRatio, threadScratch, activeCount,
                                     &filterBest, /*needAlign=*/false);
                    g_calibBatches.fetch_add(1, std::memory_order_relaxed);

                    for (int lane = 0; lane < activeCount; ++lane) {
                        int trialIdx = start + lane;
                        const auto &sims = simsSIMD[lane];
                        if (sims.size() < 3) continue; // empty guard
                        endScores[trialIdx] = sims[0].logProbRatio;
                        begScores[trialIdx] = sims[1].logProbRatio;
                        midScores[trialIdx] = sims[2].logProbRatio;
#ifdef FILTER_PASS
                        fltOnlyScores[trialIdx] = scoreToLog(filterBest[lane]);
#endif

                        if (printVerbosity > 1) {
                            std::lock_guard<std::mutex> lock(g_cout_mutex);
                            std::cout << (trialIdx + 1) << "\t" << sims[0].anchor1 << "\t" << sims[0].anchor2 << "\t"
                                      << scoreStr(sims[0].logProbRatio + shift) << "\t"
                                      << sims[1].anchor1 << "\t" << sims[1].anchor2 << "\t"
                                      << scoreStr(sims[1].logProbRatio + shift) << "\t"
                                      << sims[2].anchor1 << "\t" << sims[2].anchor2 << "\t"
                                      << scoreStr(sims[2].logProbRatio + shift) << std::endl;
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
                (*scoresFile) << "full" << "\t" << profile.name << "\t" << trial + 1 << "\tend\t"
                              << scoreStr(endScores[trial] + shift) << "\n";
                (*scoresFile) << "full" << "\t" << profile.name << "\t" << trial + 1 << "\tstart\t"
                              << scoreStr(begScores[trial] + shift) << "\n";
                (*scoresFile) << "full" << "\t" << profile.name << "\t" << trial + 1 << "\tmid\t"
                              << scoreStr(midScores[trial] + shift) << "\n";
#ifdef FILTER_PASS
                (*scoresFile) << "filter" << "\t" << profile.name << "\t" << trial + 1 << "\tall\t"
                              << scoreStr(fltOnlyScores[trial] + shift) << "\n";
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

#ifdef FILTER_PASS
    double fltMMmidL, fltMMmidK, fltMMmidKsimple, fltMLmidL, fltMLmidK, fltMLmidKsimple;
    double fltLMmidL, fltLMmidK;
    estimateGumbel(fltMMmidL, fltMMmidK, fltMMmidKsimple, fltMLmidL, fltMLmidK, fltMLmidKsimple, fltLMmidL, fltLMmidK,
                   fltOnlyScores.data(), numOfSequences, sequenceLength);
    {
        double mn = *std::min_element(fltOnlyScores.begin(), fltOnlyScores.end());
        double mx = *std::max_element(fltOnlyScores.begin(), fltOnlyScores.end());
        double mean_s = 0; for (int z = 0; z < numOfSequences; z++) mean_s += fltOnlyScores[z]; mean_s /= numOfSequences;
        if (verbosity > 0)
            std::cout << "# DBG filter scores: min=" << mn << " max=" << mx << " mean=" << mean_s << " range=" << (mx-mn) << "\n";
    }
    profile.gumbel_k_filter = fltMMmidK;
    profile.lambda_filter = fltMMmidL;
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
#ifdef FILTER_PASS
            , fltMMmidL, fltMMmidK, fltMMmidKsimple
#endif
        };
        cache.store(profile, letterFreqs, sequenceLength, border, numOfSequences, entry);
        if (wasCached)
            *wasCached = false;
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
#ifdef FILTER_PASS
    {
        CacheEntry flt = entry;
        flt.MMendL = flt.MMbegL = flt.MMmidL = entry.flt_mid_l;
        flt.MMendK = flt.MMbegK = flt.MMmidK = entry.flt_mid_k;
        flt.MMendKsimple = flt.MMbegKsimple = flt.MMmidKsimple = entry.flt_mid_k_simple;
        printEntryStats(flt);
    }
#endif
}

