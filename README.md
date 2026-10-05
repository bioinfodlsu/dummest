# DUMMEST

DUMMEST (DUMMER without Explicit Sequence Translation) it finds similar regions between DNA sequences and profile HMMs allowing for frameshifts.

## Running DUMMEST

DUMMEST ships as a single Apptainer image, `dummest.sif`, containing everything you need.  You do not need to install or build any of it yourself.

Download the image from the [GitHub releases][releases] page.

DUMMEST needs three input files:

* a protein profile in HMMER3/f format (`profile.hmm`),
* the protein alignment that profile was built from, in Stockholm format (`profile.msa`), and
* a genome in FASTA format (`genome.fa`).

Then run:

```bash
apptainer exec --nv dummest.sif pipeline.py profile.hmm profile.msa genome.fa 8
```

The last argument (8) is the number of CPUs to use.

Without an NVIDIA GPU, drop `--nv`; the pipeline falls back to the CPU automatically:

```bash
apptainer exec dummest.sif pipeline.py profile.hmm profile.msa genome.fa 8
```

Inside the container, temporary files always go to `/tmp`, which is RAM-backed.  For genomes larger than your RAM, bind a disk directory over it:

```bash
apptainer exec --bind /scratch/tmp:/tmp --nv dummest.sif \
    pipeline.py profile.hmm profile.msa genome.fa 8
```

Running the pipeline directly on the host instead uses `/tmp` only for genomes under 2 GiB; anything larger puts temporary files in the current directory, which is usually what you want since `/tmp` may itself be RAM-backed.  Set `DUMMEST_TMP_BYPASS=0` to always use `/tmp` when running directly on the host.

Your working directory, `$HOME`, and `/tmp` are visible inside the container automatically, so relative paths from your data directory are fine.  Data anywhere else needs an explicit mount:

```bash
apptainer exec --bind /scratch/data:/mnt/data --nv dummest.sif \
    pipeline.py profile.hmm profile.msa genome.fa 8
```

Results go to standard output as MAF-format records: an `a` line giving the score, *E*-value, and anchor coordinates, followed by two `s` lines showing a representative alignment of the profile and the sequence.

### Making a profile

For now, profiles are built with hmmbuild.

```bash
hmmbuild --amino data/profile.hmm data/profile.msa
```

### Try it on the bundled sample data

The image ships with a small test case.  GPU use is selected automatically when one is present; add `--nv` to enable it. To run with 8 threads:

```bash
apptainer exec dummest.sif pipeline.py /opt/dummest/samples/MET-test.hmm \
    /opt/dummest/samples/MET.msa /opt/dummest/samples/MET-target.fa 8
```

```bash
apptainer exec --nv dummest.sif pipeline.py /opt/dummest/samples/MET-test.hmm \
    /opt/dummest/samples/MET.msa /opt/dummest/samples/MET-target.fa 8
```

## Genomic filtering pipeline


The pipeline performs these steps:

1. Translate the nucleotide FASTA in all six reading frames with `seqkit`.
2. Build MMseqs2 target and query databases.
3. Search the translated target database with MMseqs2.
4. Map protein hits back to strand-aware genomic coordinates.
5. Merge and extract candidate windows with `bedtools`.
6. Run DUMMEST on the extracted windows.

The CPU argument controls the MMseqs2 and translation stages, and is passed through to DUMMEST as its thread count (`-T`). 

Use `--max` to disable the MMseqs2 filtering path and run DUMMEST against the complete set of raw contigs and both strands.  In this mode the pipeline does not run `seqkit` or MMseqs2; it creates forward and reverse-complement FASTA entries and passes them directly to DUMMEST.  This is useful as a maximum-sensitivity comparison, but can require substantially more time and memory:

```bash
python3 bin/pipeline.py profile.hmm profile.msa genome.fa 8 --max
```

The filtering pipeline is intended to reduce the amount of sequence passed to DUMMEST, not to replace the final DUMMEST alignment.  Its sensitivity and runtime depend on the MMseqs2 settings and on the quality of the translated candidate hits.

## Pipeline options

Thread count is set by the fourth positional argument, not by a flag: `pipeline.py profile.hmm profile.msa genome.fa 8` uses 8 CPUs. It sets the thread count for `seqkit translate`, the MMseqs2 database-building and search steps, and DUMMEST (`-T`).

The MMseqs2 search runs in two stages: an ungapped prefilter, then a gapped (Smith-Waterman) alignment of whatever survives. Each stage has its own significance threshold:

* `--gpu auto|on|off` (default: `auto`): MMseqs2 GPU use. `auto` uses the GPU when `nvidia-smi` reports one and falls back to CPU otherwise. `on` passes `--gpu 1` without checking for a GPU first, so it fails inside MMseqs2 when no GPU is present. `off` forces CPU-only (`--gpu 0`). Note that CPU mode also sets `--spaced-kmer-mode 0` and `-s 7.5`, so it is not equivalent to `--gpu 0` alone and may not find the same hits as the GPU path.
* `--ungapped-pvalue F` (default: `0.02`): stage 1, the ungapped prefilter. Maximum per-pair p-value for a hit to reach the gapped stage. `1.0` disables the gate, leaving the score floor only. Ignored in `--max` mode, which skips MMseqs2.
* `--gapped-evalue F` (default: `0.005`): stage 2, the gapped alignment. Passed as `-e <nseq*6*F>` to `mmseqs search`, where `nseq` is the number of genome entries (six reading frames each). Ignored in `--max` mode, which skips MMseqs2.
* `--evalue F` (default: `10`): DUMMEST *E*-value threshold. In the filtering path the pipeline forwards it as both `-e` and `-W`, so the filter pass uses the same *E*-value threshold as final scoring. In `--max` mode only `-e` is passed, since `--max` bypasses the filter pass.
* `--batch N` (default: `800000`): DUMMEST stream chunk size, in sequences. Lower it to force multi-chunk streaming, which uses less memory at the cost of more per-chunk overhead.

The pipeline runs `dummest` for you. Run `pipeline.py --help` for its remaining options.

## Cache location

`dummest` and `pipeline.py` share a `.dummest-cache` directory holding estimated parameters.  It is resolved from `$DUMMEST_CACHE_DIR`, then `$XDG_CACHE_HOME/.dummest-cache`, then `$HOME/.dummest-cache`, then `./.dummest-cache`, then `$TMPDIR/.dummest-cache`, then `/tmp/.dummest-cache`.  If none of those can be created, `dummest` falls back to `./cache.bin` and `pipeline.py` to the current directory itself.  On shared filesystems (e.g. an HPC home directory) concurrent runs can overwrite each other's entries, so point `DUMMEST_CACHE_DIR` at node-local scratch for parallel jobs.

## Frameshifts, stop codons, and masking thresholds

These DUMMEST options override the compiled-in defaults (current defaults in parentheses). They are also accepted by `bin/pipeline.py`, which forwards them to its DUMMEST invocation:

* `--insert1 F` (default: `0.0171`): 1-base insertion rate.
* `--insert2 F` (default: `0.0018`): 2-base insertion rate.
* `--delete1 F` (default: `0.0328`): 1-base deletion rate.
* `--delete2 F` (default: `0.0083`): 2-base deletion rate.
* `--stop-codon-prob F` (default: `0.0005`): stop codon probability.
* `--bg-stop-codon-prob F` (default: `0.046875`): background stop codon probability.
* `--tantan-threshold F` (default: `0.5`): tantan simple-region masking threshold.

 The indel defaults are approximated from Badread's `nanopore2020` error model (Oxford Nanopore R9.4.1) at 90% target identity.

## Current quirks

* Standalone direct DUMMEST invocation is not well tested at present.  For current genomic searches, using `bin/pipeline.py` is preferred, including when running in `--max` mode.
* The tool might not be suitable for short sequences or profiles.
* There is currently no way to estimate frameshift parameters from sequences like in `last-train`

## Building the container image

Clone the repository and initialize its submodules:

```bash
git clone https://github.com/bioinfodlsu/dummest.git
cd dummest
git submodule update --init --recursive

# The MMseqs2 fork is built from a separate clone alongside the repo, and has
# submodules of its own.
git clone https://github.com/3liteking148/MMseqs2.git ../MMseqs2
git -C ../MMseqs2 submodule update --init --recursive
```

Both DUMMEST (portable AVX2) and the custom GPU MMseqs2 fork are built on the host and copied in, so the image carries no compiler and no CUDA toolkit:

```bash
MMSEQS_SRC=/path/to/MMseqs2 ./build_apptainer.sh
```

Host build tools required: `apptainer`, `ninja`, `ccache` (`sudo apt install ninja-build ccache`), plus `cmake`, the CUDA toolkit, and `cargo` (Rust).

`build_apptainer.sh` builds the fork with the host CUDA toolkit. **The host must match the image base (Ubuntu noble, amd64)**, since the binaries are compiled on the host and run in the image.  The Ubuntu version is checked at build time (`SKIP_DISTRO_CHECK=1` to skip), but the architecture is not, so confirm your host is amd64 yourself before skipping the check.

To reuse the host apt caches (same distro/arch only), opt in with `APT_CACHE=1`:

```bash
MMSEQS_SRC=/path/to/MMseqs2 APT_CACHE=1 ./build_apptainer.sh
```

Validate a build with the in-container suite (smoke + overflow regression; slow on first run):

```bash
apptainer test dummest.sif         # CPU-only
apptainer test --nv dummest.sif    # adds CPU-vs-GPU prefilter parity (skips without a GPU)
```

NVIDIA GPU runs need the host driver injected with `apptainer --nv` (the bundled `mmseqs` statically links `libcudart`, so only `libcuda.so.1` is required at runtime).  `./build_apptainer.sh` passes `--nv` automatically when a GPU is present (`NV=0` to disable).

## Building directly

Build the C++ programs with CMake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The build copies `dummest`, `dummestl`, and `dummest-build` into `bin/`. For the command examples below, add that directory to your `PATH`:

```bash
export PATH="$PWD/bin:$PATH"
```

Builds target the native host by default (`-march=native`).  For a container or another machine, pass `-DDUMMEST_PORTABLE=ON` to build a portable AVX2 (`x86-64-v3`) baseline instead (requires Haswell/2013+).

## Host requirements

The direct DUMMEST programs require a C++20 compiler and CMake.  The genomic filtering pipeline additionally requires:

* Python 3 with the `pybedtools` package.
* `seqkit` for six-frame translation.
* `bedtools` for extracting genomic windows.
* MMseqs2 with GPU support: the custom fork with per-profile ungapped-prefilter p-values (`--ungapped-pvalue`/`--ungapped-calib`/`--ungapped-recalibrate`). This is bundled in the container; on the host, build it and point the pipeline at it with `--mmseqs-bin`.

[releases]: https://github.com/bioinfodlsu/dummest/releases
