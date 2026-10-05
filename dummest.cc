// Author: Martin C. Frith 2025
// SPDX-License-Identifier: BSD-3-Clause

// See [Frith2025]: "Simple and thorough detection of related
// sequences with position-varying probabilities of substitutions,
// insertions, and deletions", MC Frith 2025

#include "dummest-core.hh"
#include "dummest-sequence.hh"
#include "dummest-output.hh"
#include "dummest-dp.hh"
#include "dummest-traceback.hh"
#include "dummest-dp-backward.hh"
#include "dummest-dp-forward.hh"
#include "dummest-search.hh"
#include "dummest-calibration.hh"
#include "dummest-profile.hh"

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
#ifdef FILTER_PASS
    double filter_evalue_opt = -1;
    bool filter_evalue_explicit = false;
#endif
    long long totSequenceLengthOverride = -1;
    double filterStdDev = 0;
    bool keepNonvaryingTerm = false;
    int randomSeqNum = OPT_t;
    int randomSeqLen = OPT_l;
    int border = OPT_b;
    int batchSequencesOpt = 0; // 0 => threads * 25000
    int backgroundProbsType = 'G';
    char* scoresFilename = nullptr;
    bool warmupOpt = false;
    int numThreadsOpt = std::thread::hardware_concurrency();
    if (numThreadsOpt == 0) {
        numThreadsOpt = 1;
    }

    const char help[] = "\
usage: dummest profiles.hmm [sequences.fa]\n\
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
  --batch N         stream sequences in chunks of N, printing as each is done\n\
                    (default: threads * 25000, i.e. one chunk for small inputs)\n\
    -S F, --scores-file F  write adjusted bit scores of random sequences to file F\n\
                               and exit (skips sequence search)\n\
        --warmup       calibrate cache-missing profiles and exit (skips sequence\n\
                               search; profiles already in the cache are skipped;\n\
                               has no effect when combined with -S)\n\
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
  --max             skip/bypass filter pass (overrides -W/--filter-evalue)\n"
#ifdef FILTER_PASS
"\n\
Filter pass options:\n\
  -W E, --filter-evalue E  Filter-pass E-value threshold (default: value of -e)\n"
#endif
;

    const char sOpts[] = "hVve:N:s:m:d:D:t:l:b:T:S:"
#ifdef FILTER_PASS
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
                                    {"batch", required_argument, 0, OPT_BATCH_CODE},
#ifdef FILTER_PASS
                                     {"filter-evalue", required_argument, 0, 'W'},
#endif
                                     {"scores-file", required_argument, 0, 'S'},
                                     {"warmup", no_argument, 0, OPT_WARMUP_CODE},
                                     {0, 0, 0, 0}};

    int c;
    while ((c = getopt_long(argc, argv, sOpts, lOpts, &c)) != -1) {
        switch (c) {
        case 'h':
            std::cout << help;
            return 0;
        case 'V':
            std::cout << "DUMMEST "
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
        case OPT_WARMUP_CODE:
            warmupOpt = true;
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
#ifdef FILTER_PASS
        case 'W':
            filter_evalue_opt = strtod(optarg, 0);
            if (filter_evalue_opt < 0)
                return badOpt();
            filter_evalue_explicit = true;
            break;
#endif
        case OPT_MAX_CODE:
            maxModeOpt = true;
            break;
        case OPT_BATCH_CODE:
            batchSequencesOpt = intFromText(optarg);
            if (batchSequencesOpt < 1)
                return badOpt();
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

#ifdef FILTER_PASS
    // -W defaults to the -e value; --max bypasses the prefilter separately
    // via skipPrefilter (no enable-threshold gate). Note: with -e0 the
    // prefilter passes nothing unless --max is given.
    if (!maxModeOpt && !filter_evalue_explicit)
        filter_evalue_opt = evalueOpt;
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

    std::cout << "# DUMMEST "
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
        std::cout << "# Max mode: filter pass disabled\n";
#ifdef FILTER_PASS
    else if (argc - optind > 1)
        std::cout << "# Filter E-value <= " << filter_evalue_opt << "\n";
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

    int warmupHits = 0, warmupCalibrated = 0;
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
        // one pass calibrates both the full and filter models
        bool wasCached = false;
        estimateK(p, bgProbs, &charVec[seqIdx], randomSeqLen, border, randomSeqNum, printVerbosity, threadPool, threadScratches, scoresFilename ? &scoresFile : nullptr, warmupOpt ? &wasCached : nullptr);
        if (warmupOpt) {
            std::cout << "# Warmup: " << &charVec[p.nameIdx]
                      << (wasCached ? " cache hit" : " calibrated") << "\n";
            if (wasCached)
                ++warmupHits;
            else
                ++warmupCalibrated;
        }
#endif
    }

    if (warmupOpt) {
        std::cout << "# Warmup: " << numOfProfiles << " profiles, "
                  << warmupHits << " cache hits, " << warmupCalibrated << " calibrated\n";
    }

    if (scoresFilename || warmupOpt) {
        if (scoresFilename)
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
    const size_t nameBase = seqIdx; // profile/consensus data; names are chunk-scoped
    size_t totSequenceLength = 0;
    if (totSequenceLengthOverride >= 0)
        totSequenceLength = (size_t)totSequenceLengthOverride;

    // ---------------------------------------------------------
    // Stream the sequences in chunks of --batch so only that many
    // are resident at once. Calibration has already run; jobs are
    // size-sorted within each chunk (no global sort across chunks).
    // ---------------------------------------------------------
    std::ifstream file;
    std::istream *in = nullptr;
    std::filesystem::path spoolPath;
    bool needTotalScan = (totSequenceLengthOverride < 0);
    if (isDash(argv[optind + 1]) && needTotalScan) {
        // stdin is not seekable: spool it so the total-length pass can
        // re-read it before streaming.
        std::error_code ec;
        spoolPath = std::filesystem::temp_directory_path(ec) /
                    ("dummest-stdin-" + std::to_string(std::random_device{}()) + ".fa");
        {
            std::ofstream out(spoolPath, std::ios::binary);
            out << std::cin.rdbuf();
        }
        file.open(spoolPath, std::ios::binary);
        if (!file)
            return 1;
        in = &file;
    } else if (isDash(argv[optind + 1])) {
        in = &std::cin;
    } else {
        file.open(argv[optind + 1]);
        if (!file) {
            std::cerr << "can't open file: " << argv[optind + 1] << "\n";
            return 1;
        }
        in = &file;
    }

    if (needTotalScan) {
        // Pass 0: total sequence length, needed by the filter
        // pre-filter before any chunk is scored.
        Sequence scanSeq;
        Contig scanContig = {0, 0};
        std::vector<char> scanVec;
        while (readContig(*in, scanSeq, scanContig, scanVec, charToNumberIUPAC)) {
            if (scanContig.length == 0)
                continue;
            totSequenceLength += scanContig.length;
            if (strandOpt == 2)
                totSequenceLength += scanContig.length;
            scanVec.clear();
        }
        if (in != &file)
            return err("cannot stream non-seekable input without -N");
        file.clear();
        file.seekg(0);
    }

    std::cout << "# Total sequence length: " << totSequenceLength << "\n";
    std::cout.precision(3);

    int batchSequences = (batchSequencesOpt > 0)
                             ? batchSequencesOpt
                             : (int)((long)numThreadsOpt * 25000);

    std::vector<Sequence> sequences;
    std::vector<std::vector<SequenceRequest>> allRequests(numOfProfiles);
    Sequence sequence;
    Contig contig = {0, 0};
    // Running total used for the per-sequence score gate. With -N this is the
    // fixed override (as before); otherwise it accumulates in read order,
    // preserving the historic running-partial-sum behaviour.
    size_t minProbRatioTotal = (totSequenceLengthOverride >= 0) ? totSequenceLength : 0;

    auto flushChunk = [&]() {
        if (sequences.empty())
            return;
        findFinalSimilaritiesBatched(sequences, allRequests, profiles, charVec.data(),
                                     threadPool, threadScratches, evalueOpt, totSequenceLength
#ifdef FILTER_PASS
                                     , filter_evalue_opt, maxModeOpt
#endif
        );
        sequences.clear();
        for (auto &r : allRequests)
            r.clear();
        charVec.resize(nameBase);
    };

    while (readContig(*in, sequence, contig, charVec, charToNumberIUPAC)) {
        if (contig.length == 0)
            continue; // delimiter between records; metadata pushed below
        seqIdx = charVec.size() - contig.length;
        size_t maskedSeqIdx = (maskOpt & 2) ? charVec.size() : seqIdx;
        // The algorithms need one arbitrary letter past the end
        // Then round up to a multiple of the SIMD length
        charVec.resize(maskedSeqIdx + simdRoundUp(contig.length + 1));
        if (totSequenceLengthOverride < 0) {
            minProbRatioTotal += contig.length;
            if (strandOpt == 2)
                minProbRatioTotal += contig.length;
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
                                ? (std::pow(p.gumbelKmidAnchored * minProbRatioTotal / evalueOpt,
                                            1.0 / 1.0 /* p.lambda */))
                                : -1;
                        if (verbosity > 1)
                            std::cerr << "Profile: " << &charVec[p.nameIdx] << "\n";

                        SequenceRequest rq{sd, minProbRatio};
                        allRequests[j].push_back(std::move(rq));
#ifdef PIPELINE_MODE
                    }
#endif
                }
            }
        }
        sequences.push_back(sequence);
        charVec.resize(seqIdx);

        if ((int)sequences.size() >= batchSequences)
            flushChunk();
    }

    flushChunk();

    {
        size_t fb = g_fastBatches.load(),
               cb = g_calibBatches.load(),
               fib = g_filterBatches.load(), fid = g_filterDiscarded.load();
        std::cout << "# Batches: fast_batches=" << fb
                  << " calib_batches=" << cb << "\n";
        std::cout << "# Filter stats: filter_batches=" << fib
                  << " discarded_empty=" << fid
                  << " max_dp_width=" << g_maxDpWidth.load() << "\n";
    }

    if (!spoolPath.empty()) {
        std::error_code ec;
        std::filesystem::remove(spoolPath, ec);
    }

    return 0;
}
