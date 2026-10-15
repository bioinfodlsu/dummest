#!/usr/bin/env bash
#
# prepare_ncbi_hmms.sh — build the NCBI AMRFinderPlus HMM database
# (download -> filter audit -> build).
#
# Stages:
#   1. Download pinned inputs into ncbi_source/ (skipped if present).
#   2. Extract the SEED tarball.
#   3. Filter audit: keep type==AMR & scope==core models with >= MIN_SEQS
#      seed sequences; write <outdir>.kept_models.tsv.
#   4. Build hmmbuild -> hmmpress -> Stockholm MSA -> bathbuild -> MMseqs2
#      DB via build_ncbi_hmms.py.
#
# Config (config.sh, see config.example.sh): BATH_BIN, MMSEQS_BIN, MIN_SEQS,
#   OUTDIR, SRC_DIR.
# Idempotent: completed stages are skipped unless REBUILD=1.
# Usage: ./prepare_ncbi_hmms.sh [outdir]   (default outdir: ncbi_hmms)
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG=${CONFIG:-$REPO_DIR/config.sh}
[[ -f "$CONFIG" ]] && source "$CONFIG"

OUTDIR="${1:-${OUTDIR:-ncbi_hmms}}"
MIN_SEQS="${MIN_SEQS:-10}"
SRC_DIR="${SRC_DIR:-./ncbi_source}"
SEED_DIR="$SRC_DIR/SEED"
LOG="prepare_ncbi.log"

NCBFAM_URL="https://ftp.ncbi.nlm.nih.gov/hmm/NCBIfam-AMRFinder/latest/NCBIfam-AMRFinder.tsv"
PGAP_URL="https://ftp.ncbi.nlm.nih.gov/hmm/current/hmm_PGAP.tsv"
SEED_URL="https://ftp.ncbi.nlm.nih.gov/hmm/NCBIfam-AMRFinder/latest/NCBIfam-AMRFinder.SEED.tar.gz"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

: > "$LOG"
log "=== NCBI HMM preparation: outdir=$OUTDIR min_seqs=$MIN_SEQS ==="

log "--- Stage 1: downloads -> $SRC_DIR ---"
mkdir -p "$SRC_DIR"
dl() {
    local url="$1" out="$2"
    if [[ -s "$out" ]]; then
        log "  present, skipping: $out"
    else
        log "  downloading: $url"
        curl -sSL --retry 3 --max-time 600 -o "$out" "$url"
        log "  saved $(du -h "$out" | cut -f1): $out"
    fi
}
dl "$NCBFAM_URL" "$SRC_DIR/NCBIfam-AMRFinder.tsv"
dl "$PGAP_URL"   "$SRC_DIR/hmm_PGAP.tsv"
dl "$SEED_URL"   "$SRC_DIR/NCBIfam-AMRFinder.SEED.tar.gz"

log "--- Stage 2: extract SEED tarball ---"
if [[ -d "$SEED_DIR" ]] && [[ "$(ls "$SEED_DIR"/*.SEED 2>/dev/null | wc -l)" -gt 700 ]]; then
    log "  SEED dir populated ($(ls "$SEED_DIR"/*.SEED | wc -l) files), skipping"
else
    log "  extracting ..."
    mkdir -p "$SEED_DIR"
    tar -xzf "$SRC_DIR/NCBIfam-AMRFinder.SEED.tar.gz" -C "$SRC_DIR"
    # Flatten if the tarball nests SEED/.
    if [[ ! -f "$SEED_DIR/"*".SEED" ]] && [[ -d "$SRC_DIR/SEED" ]]; then
        SEED_DIR="$SRC_DIR/SEED"
    fi
    log "  extracted $(ls "$SEED_DIR"/*.SEED | wc -l) SEED files"
fi

log "--- Stage 3: filter audit (min $MIN_SEQS sequences, core AMR only) ---"
python3 build_ncbi_hmms.py "$SEED_DIR" \
    "$SRC_DIR/NCBIfam-AMRFinder.tsv" "$SRC_DIR/hmm_PGAP.tsv" \
    -o "$OUTDIR" -n "$MIN_SEQS" --audit 2>&1 | tee -a "$LOG"

log "--- Stage 4: build HMMs -> $OUTDIR ---"
if [[ "${REBUILD:-0}" != "1" ]] \
    && [[ -s "$OUTDIR/ncbi_amr_families.hmm" ]] \
    && [[ -s "$OUTDIR/ncbi_amr_families.msa" ]] \
    && [[ -s "$OUTDIR/ncbi_amr_families.bhmm" ]]; then
    log "  complete outputs present, skipping build (REBUILD=1 to force)"
else
    if [[ "${REBUILD:-0}" == "1" ]] && [[ -d "$OUTDIR" ]]; then
        log "  REBUILD=1: wiping $OUTDIR"
        rm -rf "$OUTDIR"
    fi
    log "  running build_ncbi_hmms.py (see $OUTDIR/build.log for per-model progress)"
    mkdir -p "$OUTDIR"
    python3 -u build_ncbi_hmms.py "$SEED_DIR" \
        "$SRC_DIR/NCBIfam-AMRFinder.tsv" "$SRC_DIR/hmm_PGAP.tsv" \
        -o "$OUTDIR" -n "$MIN_SEQS" --mmseqs-bin "${MMSEQS_BIN:-}" \
        2>&1 | tee "$OUTDIR/build.log"
fi

{
    echo "db_version=$(grep -m1 -o '20[0-9-]*\.[0-9]' "$SRC_DIR/NCBIfam-AMRFinder.tsv" || true)"
    echo "min_seqs=$MIN_SEQS"
    echo "built=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "models=$(grep -c '^NAME' "$OUTDIR/ncbi_amr_families.hmm")"
} > "$OUTDIR/versions.txt" 2>/dev/null || true

log "=== done ==="
log "outputs: $OUTDIR/ncbi_amr_families.{hmm,msa,bhmm} $OUTDIR/ncbi_amr_refDB* $OUTDIR/manifest.tsv"
