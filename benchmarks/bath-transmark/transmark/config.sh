#!/bin/bash
# Shared config for the transmark benchmark steps (step1..step5).
# Source from each step script:  source "$(dirname "$0")/config.sh"
# Override on the command line, e.g.  FULL_BASE=mybench ./step1-create.sh full
set -u

# Benchmark basenames (no "test" in names). The full DB reuses the
# already-generated benchmark files (renamed from test-001.*); the lite
# DB is generated fresh by step1. The *-over DBs embed only the middle
# 50% of each test seq (transmark-create --over) for overextension
# benchmarking; lite-over is the fast safety variant, full-over is the
# slow production variant (1GB fa, ~5GB fs reads).
FULL_BASE="${FULL_BASE:-transmark-full}"
LITE_BASE="${LITE_BASE:-transmark-lite}"
FULL_OVER_BASE="${FULL_OVER_BASE:-transmark-full-over}"
LITE_OVER_BASE="${LITE_OVER_BASE:-transmark-lite-over}"

# --- transmark-create params (step1) ---
# full: frozen copy of the old test.sh invocation (N=10, L=100M, 1500 fams, ~1GB .fa)
FULL_CREATE_ARGS="${FULL_CREATE_ARGS:---seed 67 --min-alen 50 --pre -N 10 -L 100000000 -R 10 -E 10 --maxtrain 30 --maxtest 20 --maxfams 1500}"
# lite: small DB for fast iteration (N=10, L=100k, 10 fams, ~1MB .fa).
# -X 0.15: with short background seqs the positives exceed the default
# 5% coverage cap (observed 0.082), so relax it for lite.
LITE_CREATE_ARGS="${LITE_CREATE_ARGS:---seed 67 --min-alen 50 --pre -N 10 -L 100000 -R 10 -E 10 --maxtrain 30 --maxtest 20 --maxfams 10 -X 0.15}"
# over variants: same as base plus --over (middle-50% embed). Same seed,
# but the alen<180 filter shifts family sampling slightly vs non-over.
FULL_OVER_CREATE_ARGS="${FULL_OVER_CREATE_ARGS:-$FULL_CREATE_ARGS --over}"
LITE_OVER_CREATE_ARGS="${LITE_OVER_CREATE_ARGS:-$LITE_CREATE_ARGS --over}"
CREATE_PFAM_SEED="${CREATE_PFAM_SEED:-../../Pfam-A.seed}"
CREATE_DNA_TFILE="${CREATE_DNA_TFILE:-seed-files/train-test.DNA.tfile}"
CREATE_AA_TFILE="${CREATE_AA_TFILE:-seed-files/train-test.AA.tfile}"
CREATE_BLAST_DIR="${CREATE_BLAST_DIR:-../../ncbi-blast-2.17.0+/bin/}"
CREATE_EASEL_DIR="${CREATE_EASEL_DIR:-../../hmmer/easel}"
CREATE_BG_HMM="${CREATE_BG_HMM:-transmark3-bg.hmm}"
CREATE_FS_RATE="${CREATE_FS_RATE:-0.00}"

# --- prepare params (step2) ---
PREP_HMMER_DIR="${PREP_HMMER_DIR:-../../hmmer/src}"
PREP_BATH1_DIR="${PREP_BATH1_DIR:-../../BATH-v1.0.4.ubuntu-latest.x64}"
PREP_BATH2_DIR="${PREP_BATH2_DIR:-/home/harvsftw/Documents/bioinfo/BATH-v2.0.0-rc1.ubuntu-latest.x64/}"
PREP_DUMMER_DIR="${PREP_DUMMER_DIR:-../../seq-position-probs/bin}"
PREP_CPUS="${PREP_CPUS:-8}"

# --- benchmark params (steps 3-4) ---
BATHSEARCH_DIR="${BATHSEARCH_DIR:-/home/harvsftw/Documents/bioinfo/BATH-v2.0.0-rc1.ubuntu-latest.x64/}"
DUMMER_DIR="${DUMMER_DIR:-../../seq-position-probs/bin}"
BENCH_CPUS="${BENCH_CPUS:-8}"
BATH_EMAX="${BATH_EMAX:-10}"
DUMMER_EMAX="${DUMMER_EMAX:-1}"

# --- LAST params (step5-last.sh via x-lastal, restored pre-dummer x-last) ---
LAST_DIR="${LAST_DIR:-/home/harvsftw/Documents/bioinfo/last/bin}"
LAST_EMAX="${LAST_EMAX:-10}"
LAST_EASEL_DIR="${LAST_EASEL_DIR:-$CREATE_EASEL_DIR}"

# --- frameshift params (step2-frameshift.sh, opt-in Badread simulation) ---
# BADREAD_DIR must point at an UPSTREAM Badread clone (github.com/rrwick/Badread);
# only its badread-runner.py + the pip-installed `badread` package are used.
# The Transmark-specific helper scripts live in BADREAD_HELPERS (in-repo).
BADREAD_DIR="${BADREAD_DIR:-/home/harvsftw/Documents/bioinfo/Badread}"
BADREAD_HELPERS="${BADREAD_HELPERS:-badread-helpers}"
FS_QUANTITY="${FS_QUANTITY:-5x}"
FS_SEED="${FS_SEED:-1}"
FS_ERROR_MODEL="${FS_ERROR_MODEL:-nanopore2020}"
FS_QSCORE_MODEL="${FS_QSCORE_MODEL:-nanopore2020}"
FS_IDENTITY="${FS_IDENTITY:-90,98,5}"
FS_CHIMERAS="${FS_CHIMERAS:-0}"

# --- post-processing params (transmark-pp.sh) ---
# Fixed seed for transmark-rocplot's Bayesian bootstraps so .xy is
# reproducible (suite convention: seed 1).
ROCPLOT_SEED="${ROCPLOT_SEED:-1}"

# PROMOTION: transmark-full.* are the former test-001.* files and global
# MMseqs DBs, renamed in place so the full DB is never regenerated.
# transmark-full.fa.orig / transmark-full.pos.orig are kept as provenance.
