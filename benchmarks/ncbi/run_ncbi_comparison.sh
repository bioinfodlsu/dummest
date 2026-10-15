#!/usr/bin/env bash
# run_ncbi_comparison.sh — 3-way comparison:
# hmmsearch vs BATH (+ BATH --no-fs) vs DUMMER no-seeds, then compare_methods.py.
#
# Paths/params come from config.sh if present (see config.example.sh):
#   READS, DUMMER_REPO, HMM_DB, BHMM, MSA, N_READS, CPU,
#   SEARCH_EVALUE, EVALUE, BASE.
# SEARCH_EVALUE is the E-value passed to every tool (and sweep upper bound);
# EVALUE is the comparison-only threshold.
#
# Usage:
#   ./run_ncbi_comparison.sh                # default sensitivity
#   ./run_ncbi_comparison.sh --max          # disable heuristic filters
#   ./run_ncbi_comparison.sh --only dummer  # DUMMER leg only
# Requires: seqtk, seqkit, mmseqs (on PATH), python3 + pysam, biopython.
set -euo pipefail

MAX=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX=1; shift ;;
        --only) ONLY="${2:-}"; shift 2 ;;
        --only=*) ONLY="${1#*=}"; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "$ONLY" && "$ONLY" != "dummer" ]]; then
    echo "unknown --only value: $ONLY (expected: dummer)" >&2; exit 2
fi
# Unquoted on purpose: empty expands to nothing, word-splitting is desired.
MAXFLAG=""
if [[ "$MAX" == "1" ]]; then
    MAXFLAG="--max"
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$REPO_DIR/lib.sh"
load_config "$REPO_DIR"

READS=${READS:-reads.fastq}
export READS
DUMMER_REPO=${DUMMER_REPO:-$REPO_DIR/../seq-position-probs}
export DUMMER_REPO
DUMMER_ENGINE=${DUMMER_ENGINE:-$DUMMER_REPO/bin/dummestl_core}
export DUMMER_ENGINE

HMMDB=${HMM_DB:-$REPO_DIR/ncbi_hmms/ncbi_amr_families.hmm}
BHMM=${BHMM:-$REPO_DIR/ncbi_hmms/ncbi_amr_families.bhmm}
MSA=${MSA:-$REPO_DIR/ncbi_hmms/ncbi_amr_families.msa}
COMPARE=$REPO_DIR/compare_methods.py

N_READS=${N_READS:-10000}
CPU=${CPU:-8}
SEARCH_EVALUE=${SEARCH_EVALUE:-1}
EVALUE=${EVALUE:-0.00001}
BASE=${BASE:-/tmp/comparison_ncbi_${N_READS}}
if [[ "$MAX" == "1" ]]; then
    BASE="$BASE/max"
fi
HMM_DIR=$BASE/hmmer
BATH_DIR=$BASE/bath
BATH_NOFS_DIR=$BASE/bath_nofs
DUM_NOSEED=${DUM_NOSEED:-$BASE/dummer_noseed}

# Build the HMM DB if missing.
if [[ ! -s "$HMMDB" || ! -s "$BHMM" || ! -s "$MSA" ]]; then
    echo "NCBI HMM DB incomplete, building via prepare_ncbi_hmms.sh ..." >&2
    bash "$REPO_DIR/prepare_ncbi_hmms.sh" "$REPO_DIR/ncbi_hmms"
fi
for f in "$HMMDB" "$BHMM" "$MSA"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
require_cmds

WARMUP_LOG=${WARMUP_LOG:-$BASE/warmup.log}
dummer_warmup "$DUMMER_ENGINE" "$HMMDB" "$WARMUP_LOG"

MODE="default"
if [[ "$MAX" == "1" ]]; then
    MODE="--max (filters off)"
fi
TITLE="NCBI 3-way comparison (hmmsearch / BATH / DUMMER no-seeds)"
[[ "$ONLY" == "dummer" ]] && TITLE="DUMMER no-seeds"
echo "=== $TITLE ==="
echo "Mode:    $MODE"
echo "Input:   $READS"
echo "Reads:   $N_READS"
echo "CPU:     $CPU"
echo "Search E-value: $SEARCH_EVALUE"
echo "Compare E-value: $EVALUE"
echo "DB:      $HMMDB"
echo "Dummer repo: $DUMMER_REPO"
echo "Base dir:    $BASE"
echo ""

if [[ "$ONLY" != "dummer" ]]; then
    echo "--- hmmsearch ---"
    rm -rf "$HMM_DIR"
    python3 "$REPO_DIR/run_hmmsearch.py" "$READS" \
        --hmmdb "$HMMDB" -o "$HMM_DIR" --cpu "$CPU" --max-reads "$N_READS" \
        --evalue "$SEARCH_EVALUE" $MAXFLAG

    echo "--- BATH ---"
    rm -rf "$BATH_DIR"
    python3 "$REPO_DIR/run_bath.py" "$READS" \
        --hmmdb "$BHMM" -o "$BATH_DIR" --cpu "$CPU" --max-reads "$N_READS" \
        --evalue "$SEARCH_EVALUE" $MAXFLAG

    echo "--- BATH --no-fs ---"
    rm -rf "$BATH_NOFS_DIR"
    python3 "$REPO_DIR/run_bath.py" "$READS" \
        --bath-no-fs --hmmdb "$BHMM" -o "$BATH_NOFS_DIR" \
        --cpu "$CPU" --max-reads "$N_READS" \
        --evalue "$SEARCH_EVALUE" $MAXFLAG
fi

echo "--- DUMMER no-seeds ---"
rm -rf "$DUM_NOSEED"
python3 "$REPO_DIR/run_dummer.py" "$READS" \
    --hmmdb "$HMMDB" --msa "$MSA" -o "$DUM_NOSEED" \
    --cpu "$CPU" --max-reads "$N_READS" \
    --evalue "$SEARCH_EVALUE" $MAXFLAG

if [[ "$ONLY" != "dummer" ]]; then
    echo ""
    echo "=== Comparison (E<$EVALUE) ==="
    SWEEP_TSV=${SWEEP_TSV:-$BASE/sweep.tsv}
    SWEEP_PLOT=${SWEEP_PLOT:-$BASE/sweep.png}
    BP_TSV=${BP_TSV:-$BASE/bp.tsv}
    python3 "$COMPARE" "$HMM_DIR" "$BATH_DIR" "$DUM_NOSEED" \
        --evalue "$EVALUE" --sweep-max "$SEARCH_EVALUE" \
        --sweep-tsv "$SWEEP_TSV" --sweep-plot "$SWEEP_PLOT" \
        --bp-tsv "$BP_TSV" --bath-no-fs-dir "$BATH_NOFS_DIR"
    echo "Sweep TSV:  $SWEEP_TSV"
    echo "Sweep plot: $SWEEP_PLOT (pairs + bp, dual axis)"
    echo "BP TSV:     $BP_TSV"
else
    echo "DUMMER output: $DUM_NOSEED/dummer_output.txt"
    echo "Warmup log:    $WARMUP_LOG"
fi
