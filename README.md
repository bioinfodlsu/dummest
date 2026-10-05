# DUMMEST

DUMMEST (DUMMER without Explicit Sequence Translation) 
it finds similar regions between DNA sequences and
profile HMMs allowing for frameshifts.  



## Setup

Clone the repository and initialize its submodules:

    git clone https://github.com/bioinfodlsu/dummest.git
    cd dummest
    git submodule update --init --recursive

Build the C++ programs with CMake:

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel

The build copies `dummest`, `dummestl`, and `dummest-build` into `bin/`.
For the command examples below, add that directory to your `PATH`:

    export PATH="$PWD/bin:$PATH"

Builds target the native host by default (`-march=native`).  For a container
or another machine, pass `-DDUMMEST_PORTABLE=ON` to build a portable AVX2
(`x86-64-v3`) baseline instead (requires Haswell/2013+).
The direct DUMMEST programs require a C++20 compiler and CMake.  The genomic
filtering pipeline additionally requires:

* Python 3 with the `pybedtools` package.
* `seqkit` for six-frame translation.
* `bedtools` for extracting genomic windows.
* MMseqs2 with GPU support: the custom fork with per-profile ungapped-prefilter
  p-values (`--ungapped-pvalue`/`--ungapped-calib`/`--ungapped-recalibrate`).
  This is bundled in the container; on the host, build it and point the
  pipeline at it with `--mmseqs-bin`.

## Usage

DUMMEST can compare sequences in FASTA format to profiles in HMMER3/f
format.  You can make profiles with `dummest-build` (new and
lightly-tested).  Or you can (ab)use [HMMER][] profiles (whose
definition doesn't quite fit): you can get DNA profiles from [Dfam][],
or protein profiles from [Pfam][].  Run it like this:

    dummest profiles.hmm sequences.fasta

The output shows similar regions in [MAF][] format:

    a score=44.6 E=8.15e-05 anchor=227,9133
    s myProfile   195 61 +   262 ATACATGTAAAGCGCTTAGAACAGTGCCTGGCAcacacacacaGCTCAATAAATGTTAGCT
    s mySequence 9102 60 + 10000 CTACACCTTCAGGACTTAGG-CTGTGCCTGGCACATACTAGTAGCTCAGTAGACACTGGTT

* The `score` reflects the likelihood that these regions are related
  rather than random.

* The *E*-value (`E=`) is the expected number of distinct sequence
  regions with equal or higher score, if we compared that profile to
  random sequences with the same length as all of `sequences.fasta`.

* The `anchor` shows profile,sequence coordinates.  It means there are
  similar regions around these coordinates.

In more detail, the score comes from adding the probabilities of all
possible alignments passing through the anchor:

score = log<sub>2</sub>[ (sum of alignment probabilities) / (probability of length-0 alignment) ]

DUMMEST tries all possible anchors, and outputs all whose *E*-values
are &le; a threshold and are local optima.

The `s` lines show a representative alignment.  This aligns letters
whose probability of being aligned is > 0.5, among all possible
alignments with that anchor.

Lowercase indicates positions judged (by [tantan][]) to be simple
repeats, like atatatatatatatat.  Such sequences evolve frequently and
independently, resulting in similarities between unrelated sequences.
So, DUMMEST ignores similarity at these positions.

## Genomic filtering pipeline

`bin/pipeline.py` is intended for searching nucleotide genomes with protein
or translated-family profiles.  It accepts an HMM profile, an MSA used to
construct the MMseqs2 query profile, a nucleotide FASTA file, and a CPU count:

    python3 bin/pipeline.py profile.hmm profile.msa genome.fa 8 --dummest-bin bin/dummest

MMseqs2 is resolved from `--mmseqs-bin`, else `mmseqs` on `PATH`.  Use the
custom GPU fork (see above) for the `--ungapped-pvalue` prefilter.

The pipeline performs these steps:

1. Translate the nucleotide FASTA in all six reading frames with `seqkit`.
2. Build MMseqs2 target and query databases.
3. Search the translated target database with MMseqs2.
4. Map protein hits back to strand-aware genomic coordinates.
5. Merge and extract candidate windows with `bedtools`.
6. Run DUMMEST on the extracted windows.

The CPU argument controls the MMseqs2 and translation stages.  The pipeline's
current DUMMEST invocation uses its own configured thread setting, so use
`dummest --help` and the source defaults when tuning execution on a particular
machine.

Use `--max` to disable the MMseqs2 filtering path and run DUMMEST against the
complete set of raw contigs and both strands.  In this mode the pipeline does
not run `seqkit` or MMseqs2; it creates forward and reverse-complement FASTA
entries and passes them directly to DUMMEST.  This is useful as a
maximum-sensitivity comparison, but can require substantially more time and
memory:

    python3 bin/pipeline.py profile.hmm profile.msa genome.fa 8 --max --dummest-bin bin/dummest

Other supported pipeline options include:

* `--skip-dummest`: generate the extracted debug FASTA without running DUMMEST.
* `--output-fa FILE`: retain a copy of the extracted debug FASTA.
* `--target-db-pad PATH`: reuse an existing padded MMseqs2 target database.
* `--query-db PATH`: reuse an existing MMseqs2 query profile database.
* `--dummest-bin PATH`: select the DUMMEST executable explicitly.
* `--mmseqs-bin PATH`: select the MMseqs2 binary explicitly (default:
  `mmseqs` on `PATH`).
* `--prefilter-mode 1`: MMseqs2 ungapped prefilter (the only mode the custom
  fork enables; mode 3 is disabled).
* `--no-gpu`: run MMseqs2 on CPU only (passes `--gpu 0` to `createdb` and `search`).
* `--ungapped-pvalue F` (default: `0.02`): keep prefilter pairs whose per-profile
  ungapped p-value is `<= F` (`1.0` disables the gate). Passed to `mmseqs search`.
* `--ungapped-calib PATH` (default: `.dummest-cache/ungappedcalib.tsv`):
  sidecar cache for the per-profile `(lambda, K)` calibration.
* `--ungapped-recalibrate`: ignore cache entries and recalibrate every profile.
* `--prefilter-pvalue F` (default: `0.005`): MMseqs2 prefilter p-value.
  The pipeline passes `-e <nseq*6*F>` to `mmseqs search`, where `nseq`
  is the number of genome entries. Ignored in `--max` mode, which
  skips MMseqs2.
* `--prefilter-max-seqs N` (default: `1000`): MMseqs2 `--max-seqs` — max
  prefilter results per query profile / protein family allowed to pass
  the prefilter. Passed as `--max-seqs` to `mmseqs search`. Ignored in
  `--max` mode, which skips MMseqs2.
* `--evalue F` (default: `10`): DUMMEST *E*-value threshold. The pipeline
  forwards it as both `-e` and `-W` to DUMMEST, so the filter pass
  pre-filter uses the same *E*-value threshold as final scoring.
* `--insert1/--insert2/--delete1/--delete2`, `--stop-codon-prob`,
  `--bg-stop-codon-prob`, `--tantan-threshold`: forwarded to DUMMEST
  (see below); unset means use the DUMMEST default.

### Example

The repository's `test.sh` provides a small intended-use example:

    time python3 bin/pipeline.py MET-test.hmm MET.msa MET-target.fa 16 --max --batch 1

It searches the included test target with the `MET-test.hmm` profile in
maximum-sensitivity mode.  It does not exercise the MMseqs2 filtering path;
remove `--max` to use the normal candidate-filtering workflow.  The pipeline
defaults to `bin/dummestl`; override with `--dummest-bin bin/dummest`.

## Fast, low-memory version

`dummestl` is the pipeline default: it is faster and, with the packed
overflow-proof score representation, uses less memory than the double build,
but it is more prone to numeric overflow.  (It uses single-precision instead
of double-precision floating-point numbers.)

## Options

Show all options and default values:

    dummest --help

Compare each profile to just the forward strand of each DNA sequence:

    dummest -s1 profiles.hmm sequences.fasta

`-s0` means reverse strands only, `-s1` means forward strands only,
and `-s2` means both strands (the default).  This is ignored for
proteins.

Get similarities with *E*-value at most (say) 0.01:

    dummest -e0.01 profiles.hmm sequences.fasta

The filter pass uses the same *E*-value threshold by default.
Override it separately with `-W` (e.g. `-e0.01 -W1`), or bypass the
pre-filter entirely with `--max`. With `-e0`, the pre-filter passes nothing
unless `--max` is given.

Turn off simple-sequence detection:

    dummest -m0 profiles.hmm sequences.fasta

`-m0` means find simple regions in neither profile nor sequence, `-m1`
means find them in profiles only, `-m2` means sequences only, and
`-m3` means both (the default).

### Frameshifts, stop codons, and masking thresholds

These DUMMEST options override the compiled-in defaults (current
defaults in parentheses). They are also accepted by
### Cache location

`dummest` and `pipeline.py` share a `.dummest-cache` directory holding the
E-value calibration (`cache.bin`) and the MMseqs2 ungapped calibration
(`ungappedcalib.tsv`).  It is resolved from `$DUMMEST_CACHE_DIR`, then
`$XDG_CACHE_HOME/.dummest-cache`, then `$HOME/.dummest-cache`, then
`./.dummest-cache`, then `/tmp/.dummest-cache`, and finally the current
directory.  On shared filesystems (e.g. an HPC home directory) concurrent runs
can overwrite each other's entries, so point `DUMMEST_CACHE_DIR` at node-local
scratch for parallel jobs.

`bin/pipeline.py`, which forwards them to its DUMMEST invocation:

* `--insert1 F` (default: `0.0171`): 1-base insertion rate per base.
* `--insert2 F` (default: `0.0018`): 2-base insertion rate per base.
* `--delete1 F` (default: `0.0328`): 1-base deletion rate per base.
* `--delete2 F` (default: `0.0083`): 2-base deletion rate per base.
* `--stop-codon-prob F` (default: `0.0005`): stop codon probability.
* `--bg-stop-codon-prob F` (default: `0.046875`): background stop codon
  probability.
* `--tantan-threshold F` (default: `0.5`): tantan simple-region masking
  threshold.

The 1-bp background frameshift branch is `insert1 + delete2` and the
2-bp branch is `insert2 + delete1`. For example:

    dummest --insert1 0.02 --delete1 0.03 profiles.hmm sequences.fasta

    python3 bin/pipeline.py profile.hmm profile.msa genome.fa 8 --insert1 0.02 --prefilter-pvalue 0.05 --dummest-bin bin/dummest

DUMMEST prints the active values at startup and includes them in its
*E*-value calibration cache hash, so changing them invalidates stale
cache entries. See
[`docs/frameshift-rate-approximation.md`](docs/frameshift-rate-approximation.md)
for how the default indel rates were approximated.

## Unusual symbols in sequences

For nucleotide sequences, `U` (uracil) is converted to `T` (thymine).
For proteins, `U` (selenocysteine) is treated the same as `C`
(cysteine), and `O` (pyrrolysine) the same as `K` (lysine).

`dummest` treats other unusual symbols as barriers that break the
sequence into contigs (contiguous sequence).

## Rarely useful features

### Strongest similarities

`-e0` has a special meaning: for each profile versus each contig strand, it
shows the maximum end-anchored, start-anchored, and mid-anchored
scores (in that order).  These scores sum over all alignments ending
at, starting at, or passing through the anchor.  DUMMEST gets each kind
of score for all possible anchors, and shows the maximum.

### Random sequences

To calculate *E*-values, DUMMEST needs to estimate a *K* parameter for
each profile.  To do that, it compares the profile to random
sequences.  To see details of this, give it a profile file only:

    dummest profiles.hmm

For each profile versus each sequence, it shows the maximum
end-anchored, start-anchored, and mid-anchored scores.

For each kind of score, it then shows various estimates of *K* (and
another &lambda; parameter that isn't used currently):

* `lamMM` is &lambda; estimated by the method of moments.
* `kMM` is *K* estimated by the method of moments.
* `kMM1` is *K* estimated by the method of moments assuming &lambda; = 1.
* `lamML` is &lambda; estimated by maximum-likelihood.
* `kML` is *K* estimated by maximum-likelihood.
* `kML1` is *K* estimated by maximum-likelihood assuming &lambda; = 1.
* `lamLM` is &lambda; estimated by the method of L-moments.
* `kLM` is *K* estimated by the the method of L-moments.

(`kLM1` isn't shown, because it's identical to `kMM1`.)

Only `kMM1` is used to calculate *E*-values.

These options affect the random sequences:

- `-t T`, `--trials T`: generate this many random sequences.

- `-l L`, `--length L`: length of each random sequence.

- `-b B`, `--border B`: add this size border to each random sequence.
  It first generates a random sequence of length L, then appends B/L
  copies of this sequence to itself.  This aims to avoid [edge
  effects][] on the distribution of scores.

## Details

* The random sequences have letter frequencies equal to the profile's
  "background" letter frequencies.

* A profile's background letter frequencies are set to the geometric
  mean of its position-specific letter probabilities ([Barrett et
  al. 1997](https://doi.org/10.1093/bioinformatics/13.2.191)),
  ignoring lowercase postions.

* The letter probabilities of lowercase profile positions are set
  equal to the background probabilities.

* A lowercase sequence letter matched to any profile position is
  treated as the worst-matching letter type at that position.

* An *E*-value is: *KN* / 2^score, where *N* is the sum of sequence
  lengths.

* &lambda; comes from a not-yet-successful attempt to get better
  *E*-values for short profiles.  Here, the *E*-value is:
  *KN* / (2^score)^&lambda;.  That would surely work and give us
  better *E*-values, if only we could determine a good value for &lambda;.

## dummest-build

New and experimental.  It makes a profile from an aligned family of
related sequences (in [Stockholm][] format):

    dummest-build alignments.stk > profiles.hmm

It may be slow, or fail due to overflow.  This is because of a
"Baum-Welch" step that refines the profile (which HMMER 3.4 lacks).
You can omit this step:

    dummest-build --countonly alignments.stk > profiles.hmm

* dummest-build down-weights sequences that are similar to each other
  (e.g. human and chimp versions of a sequence).

* Option `--enone` scales the absolute sequence weights so that the
  maximum weight is 1.  That seems good for probabilities fitted to
  the input sequences.

* Often, however, we want probabilities fitted to more-distantly
  related sequences than those in the input.  It does that by
  something similar to HMMER's "entropy weighting", but slightly
  different.  At each position, if the total letter count at that
  position exceeds an "effective sequence number (EFFN)" threshold,
  those counts are downscaled so their total equals the threshold.
  Insertion and deletion counts are treated similarly.

  
## Current status

* Direct DUMMEST searches use thorough dynamic programming, including
  provisional one- and two-nucleotide frameshift transitions.  They can
  therefore be slow and memory-consuming.

* The current build applies a filter pass before final scoring,
  using the same *E*-value threshold as final scoring (`-W` defaults to the
  `-e` value; `--max` bypasses it). The final alignment still
  integrates evidence from alternative alignment paths.

* Standalone direct DUMMEST invocation is not well tested at present.  For
  current genomic searches, using `bin/pipeline.py` is preferred, including
  when running in `--max` mode.

* For large genomic searches, `bin/pipeline.py` provides an optional
  filtering pipeline.  It translates nucleotide sequences in all six reading
  frames, uses MMseqs2 to find candidate regions, extracts merged genomic
  windows, and then runs DUMMEST on those windows.

* It can fail due to overflow (numbers getting too big).  This only
  happens when there are very strong similarities.

* The *E*-values are over-estimated when the profile or sequence is
  short.

The filtering pipeline is intended to reduce the amount of sequence passed
to DUMMEST, not to replace the final DUMMEST alignment.  Its sensitivity and
runtime depend on the MMseqs2 settings and on the quality of the translated
candidate hits.

The provisional background frameshift rates are documented in
[`docs/frameshift-rate-approximation.md`](docs/frameshift-rate-approximation.md).
They are rough per-base estimates and are not yet calibrated HMM transition
probabilities.


[Dfam]: https://dfam.org/home
[Pfam]: https://www.ebi.ac.uk/interpro/entry/pfam/#table
[frith2025]: https://doi.org/10.1101/2025.03.14.643233
[dmmbuild]: https://github.com/Padraig20/dmmbuild
[edge effects]: https://doi.org/10.1093/nar/29.2.351
[HMMER]: http://hmmer.org
[HMMER's theory]: https://doi.org/10.1371/journal.pcbi.1000069
[MAF]: https://genome.ucsc.edu/FAQ/FAQformat.html#format5
[Stockholm]: https://en.wikipedia.org/wiki/Stockholm_format
[tantan]: https://gitlab.com/mcfrith/tantan
