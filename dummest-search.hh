// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "dummest-dp-backward.hh"
#include "dummest-dp-forward.hh"
#include "dummest-output.hh"

// Combined entry (max mode + calibration): backward half then forward half.
void findSimilarities(std::array<std::vector<AlignedSimilarity>, simdWidth> &similarities, const Profile &profile,
             const std::array<std::vector<uint8_t>*, simdWidth> &decoded, std::array<Float, simdWidth> minProbRatio,
             DPScratch &scratch, int activeCount, std::array<ExpScore, simdWidth> *filterBestOut = nullptr,
             bool needAlign = true) {
    BackwardCtx ctx;
    backwardPass(profile, decoded, scratch, activeCount, filterBestOut, ctx, needAlign);
    forwardPass(similarities, profile, decoded, minProbRatio, scratch, activeCount, ctx,
                       needAlign);
}

void findFinalSimilarities(std::vector<FinalSimilarity> &similarities, std::array<SequenceRequest, simdWidth> &req,
                           const Profile &profile, size_t profileNum, const char *charVec,
                           DPScratch &scratch, int activeCount);

// Shared record emission: AlignedSimilarity hits -> FinalSimilarity records
// (traceback-guard + coordinate mapping + gapped sequences).
void emitFinalSimilarities(std::vector<FinalSimilarity> &similarities,
                           const std::array<std::vector<AlignedSimilarity>, simdWidth> &sims,
                           std::array<SequenceRequest, simdWidth> &req,
                           const Profile &profile, size_t profileNum, const char *charVec,
                           int activeCount) {
    const char *alphabet = getAlphabet(profile.width - nonLetterWidth);
    const char *profileSeq = charVec + profile.consensusSequenceIdx;
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
            FinalSimilarity s = {x.logProbRatio, profileNum, req[idx].seqData->strandNum, x.anchor1,
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

void findFinalSimilarities(std::vector<FinalSimilarity> &similarities, std::array<SequenceRequest, simdWidth> &req,
                           const Profile &profile, size_t profileNum, const char *charVec,
                           DPScratch &scratch, int activeCount) {
    std::array<std::vector<AlignedSimilarity>, simdWidth> sims;
    std::array<std::vector<uint8_t>*, simdWidth> decoded{};
    std::array<Float, simdWidth> minProbRatio;
    for (int i = 0; i < activeCount; i++) {
        decoded[i] = &req[i].seqData->decoded;
        minProbRatio[i] = req[i].minProbRatio;
    }
    // Lazy rescaling keeps every lane finite: no overflow, no recycle.
    findSimilarities(sims, profile, decoded, minProbRatio, scratch, activeCount,
                               nullptr, true);
    // Always-on accounting (one fetch_add per batch).
    g_fastBatches.fetch_add(1, std::memory_order_relaxed);

    emitFinalSimilarities(similarities, sims, req, profile, profileNum, charVec, activeCount);
}

void findFinalSimilaritiesBatched(const std::vector<Sequence> &sequences,
                                  std::vector<std::vector<SequenceRequest>> &allRequests,
                                  const std::vector<Profile> &profiles, const char *charVec,
                                  ThreadPool &threadPool, std::vector<DPScratch> &threadScratches,
                                  double evalueOpt, size_t totSequenceLength
#ifdef FILTER_PASS
                                  , double filter_evalue, bool skipPrefilter
#endif
                                   ) {
    int batchStride = simdWidth;
    static const char* lanes_env = getenv("DUMMEST_MAX_LANES");
    if (lanes_env) batchStride = std::clamp(std::atoi(lanes_env), 1, simdWidth);

    auto printSims = [&](const std::vector<FinalSimilarity> &jobSimilarities) {
        // Format immediately (unordered): nothing downstream depends on
        // record order, and each hit's alignment is freed right away.
        // Formatting (exp + iostream) runs OUTSIDE any lock; emitChunk
        // writes the whole chunk with one syscall (lockless on files).
        std::ostringstream os;
        for (const auto &sim : jobSimilarities) {
            const Profile &p = profiles[sim.profileNum];
            const Sequence &s = sequences[sim.strandNum / 2];
            double evalue = p.gumbelKmidAnchored * totSequenceLength *
                            std::exp(-p.lambda * sim.logProbRatio);
            if (evalueOpt > 0 && evalue > evalueOpt)
                continue;
            printSimilarity(os, charVec, p, s, sim, evalue);
        }
        emitChunk(os.str());
    };

#ifdef FILTER_PASS
    // Filter pass: same E-value semantics as full scoring.
    // No enable threshold gate: it always runs unless bypassed via --max.
    // Length sort keeps DP width (batch max length) tight.
    for (size_t i = 0; i < profiles.size(); ++i)
        std::sort(allRequests[i].begin(), allRequests[i].end(), std::greater<>());
    const bool useFilter = !skipPrefilter;
#else
    for (size_t i = 0; i < profiles.size(); ++i)
        std::sort(allRequests[i].begin(), allRequests[i].end(), std::greater<>());
    const bool useFilter = false;
#endif
    if (useFilter) {
        // Lean score-only filter: float backward-only per batch (2-row
        // rolling W1, no matrix store, no alignment). Survivors collect per
        // profile, length-sort, and run through the regular slow path below
        // (same as --max). Inf scores always survive (slow-wave firewall).
        struct FilterJob {
            size_t profileIdx;
            size_t startRequestIdx;
            int activeCount;
        };
        std::vector<FilterJob> filterJobs;
        for (size_t i = 0; i < profiles.size(); ++i) {
            const auto &requests = allRequests[i];
            size_t n = requests.size();
            for (size_t start = 0; start < n; start += simdWidth) {
                int count = (int)std::min((size_t)simdWidth, n - start);
                filterJobs.push_back({i, start, count});
            }
        }

        std::vector<std::vector<SequenceRequest>> filteredRequests(profiles.size());
        std::vector<std::vector<SequenceRequest>> filterJobResults(filterJobs.size());
        std::atomic<size_t> completedFilter{0};
        std::mutex filterMtx;
        std::condition_variable filterCv;

        for (size_t jobIdx = 0; jobIdx < filterJobs.size(); ++jobIdx) {
            threadPool.enqueue([&, jobIdx](int threadId) {
                DPScratch &ts = threadScratches[threadId];
                const auto &job = filterJobs[jobIdx];
                const auto &p = profiles[job.profileIdx];
                const auto &requests = allRequests[job.profileIdx];

                std::array<std::vector<uint8_t>*, simdWidth> decodedPtrs = {};
                for (int k = 0; k < job.activeCount; k++)
                    decodedPtrs[k] = &requests[job.startRequestIdx + k].seqData->decoded;

                std::array<Float, simdWidth> bestScores;
                findSimilaritiesBackwardOnly(bestScores, p, decodedPtrs, ts, job.activeCount);
                g_filterBatches.fetch_add(1, std::memory_order_relaxed);

                auto &out = filterJobResults[jobIdx];
                for (int k = 0; k < job.activeCount; k++) {
                    if (bestScores[k] > 0) {
                        double log_probRatio = log((double)bestScores[k]);
                        double probRatio = exp(log_probRatio * p.lambda_filter);
                        double evalue = p.gumbel_k_filter * totSequenceLength / probRatio;
                        bool infScore = !std::isfinite(log_probRatio);
                        if (evalue <= filter_evalue || infScore)
                            out.push_back(requests[job.startRequestIdx + k]);
                    }
                }
                if (out.empty())
                    g_filterDiscarded.fetch_add(1, std::memory_order_relaxed);

                if (++completedFilter == filterJobs.size()) {
                    std::lock_guard<std::mutex> lock(filterMtx);
                    filterCv.notify_one();
                }
            });
        }

        if (!filterJobs.empty()) {
            std::unique_lock<std::mutex> lock(filterMtx);
            filterCv.wait(lock, [&]{ return completedFilter == filterJobs.size(); });
        }
        // Concat per-job survivors per profile and length-sort (the slow
        // wave sizes scratch per batch, so length sort minimizes padding).
        for (size_t j = 0; j < filterJobs.size(); ++j)
            for (auto &r : filterJobResults[j])
                filteredRequests[filterJobs[j].profileIdx].push_back(std::move(r));
        for (size_t i = 0; i < profiles.size(); ++i)
            std::sort(filteredRequests[i].begin(), filteredRequests[i].end(), std::greater<>());
        allRequests = std::move(filteredRequests);
    }
    struct BatchJob {
        size_t profileIdx;
        size_t startRequestIdx;
        int activeCount;
    };

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

            std::vector<FinalSimilarity> jobSimilarities;
            findFinalSimilarities(jobSimilarities, curBatch,
                                  profiles[job.profileIdx], job.profileIdx,
                                  charVec, threadScratch, job.activeCount);

            printSims(jobSimilarities);

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
}

